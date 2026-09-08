import os
import glob
import tqdm
import math
import imageio
import copy
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import warnings
import tensorboardX
import raymarching
import numpy as np
import pandas as pd

import time
from contextlib import contextmanager
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips
from torchmetrics.functional import structural_similarity_index_measure

from nerf.adv.pgd import attack_pgd
from nerf.adv.rand import attack_random

from nerf.info.loss import EntropyLoss, SmoothingLoss

from table import write_table

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')
    
#新增1函数
def uniform_sphere_sampling(shape, device, dtype=torch.float32):
    """
    在单位球面上均匀采样方向
    shape: 例如 (B, N)
    return: [B, N, 3]
    """
    u = torch.rand(*shape, device=device, dtype=dtype)
    v = torch.rand(*shape, device=device, dtype=dtype)

    theta = torch.acos(2 * u - 1)       # [0, pi]
    phi = 2 * math.pi * v               # [0, 2pi]

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack([x, y, z], dim=-1)
#新增2函数
def jsd_divergence(p, q, eps=1e-8):
    """
    p, q: [..., T]，已经是概率分布
    return: [...]
    JSD = 0.5 KL(p||m) + 0.5 KL(q||m), m = 0.5(p+q)
    """
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)
    m = 0.5 * (p + q)

    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)
    
@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1, ray_inds=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    # choose rays corresponding to the given ray_inds
    if ray_inds is not None and N > 0:
        i = torch.gather(i, -1, ray_inds)
        j = torch.gather(j, -1, ray_inds)
        results['inds'] = ray_inds
    elif N > 0:
        N = min(N, H*W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:
            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

class Trainer(object):
    _RESUME_CONFIG_IGNORED_KEYS = {
        'bg_mode',
        'eval_bg_color',
        'ckpt',
        'checkpoint_interval_steps',
        'dataset_name',
        'eval_expected_step',
        'eval_output_dir',
        'eval_overwrite',
        'eval_split',
        'eval_variants',
        'gui',
        'implementation_name',
        'milestone_steps',
        'profile_report_steps',
        'profile_training',
        'stop_at_step',
        'test',
        'workspace',
        'write_table',
    }

    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=1, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 checkpoint_load_mode="resume", # resume, evaluation, or model
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 awp_adversary=None, # whether to use adversarial weight perturbation
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        if checkpoint_load_mode not in {'resume', 'evaluation', 'model'}:
            raise ValueError(
                f'checkpoint_load_mode must be resume, evaluation, or model; '
                f'got {checkpoint_load_mode!r}'
            )
        self.checkpoint_load_mode = checkpoint_load_mode
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.awp_adversary = awp_adversary

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.last_checkpoint_path = None
        self.checkpoint_config = None
        self.ema_checkpoint_loaded = False
        self.milestone_steps = frozenset(
            int(step) for step in getattr(self.opt, 'milestone_steps', [])
        )
        self._profile_enabled = bool(
            getattr(self.opt, 'profile_training', False)
            and self.local_rank == 0
            and self.device.type == 'cuda'
        )
        self._profile_cpu = {}
        self._profile_cuda_events = {}
        self._profile_window_step = self.global_step
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            self.milestone_ckpt_path = os.path.join(
                self.ckpt_path,
                'milestones',
            )
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                if self.checkpoint_load_mode == 'evaluation':
                    raise ValueError('evaluation mode requires a checkpoint; scratch is invalid')
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint(load_mode=self.checkpoint_load_mode)
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(load_mode='model')
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(
                        self.best_path,
                        load_mode=self.checkpoint_load_mode,
                    )
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint(load_mode=self.checkpoint_load_mode)
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(
                    self.use_checkpoint,
                    load_mode=self.checkpoint_load_mode,
                )
        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...

        # entropy and gain loss (KL-Divergence)
        if opt.entropy:
            self.fun_entropy_loss = EntropyLoss(opt)
        if opt.smoothing:
            self.fun_KL_divergence_loss = SmoothingLoss(opt)


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def _profile_begin(self, name):
        if not self._profile_enabled:
            return None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return name, time.perf_counter(), start_event, end_event

    def _profile_end(self, token):
        if token is None:
            return
        name, cpu_start, start_event, end_event = token
        end_event.record()
        cpu_seconds, calls = self._profile_cpu.get(name, (0.0, 0))
        self._profile_cpu[name] = (
            cpu_seconds + time.perf_counter() - cpu_start,
            calls + 1,
        )
        self._profile_cuda_events.setdefault(name, []).append(
            (start_event, end_event)
        )

    @contextmanager
    def _profile_phase(self, name):
        token = self._profile_begin(name)
        try:
            yield
        finally:
            self._profile_end(token)

    def _profile_report(self):
        if not self._profile_enabled:
            return
        torch.cuda.synchronize()
        cuda_seconds = {
            name: sum(start.elapsed_time(end) for start, end in events) / 1000.0
            for name, events in self._profile_cuda_events.items()
        }
        window_steps = self.global_step - self._profile_window_step
        step_cuda = cuda_seconds.get('step_total', 0.0)
        step_cpu = self._profile_cpu.get('step_total', (0.0, 0))[0]
        self.log(
            f'[PROFILE] steps={self._profile_window_step}->{self.global_step} '
            f'count={window_steps} step_cuda={step_cuda:.6f}s '
            f'step_cpu={step_cpu:.6f}s'
        )
        for name in sorted(cuda_seconds, key=cuda_seconds.get, reverse=True):
            if name == 'step_total':
                continue
            cpu_total, calls = self._profile_cpu.get(name, (0.0, 0))
            cuda_total = cuda_seconds[name]
            share = 100.0 * cuda_total / step_cuda if step_cuda > 0 else 0.0
            cuda_ms_per_step = (
                1000.0 * cuda_total / window_steps if window_steps > 0 else 0.0
            )
            self.log(
                f'[PROFILE] phase={name} calls={calls} '
                f'cuda={cuda_total:.6f}s cpu={cpu_total:.6f}s '
                f'cuda_ms/step={cuda_ms_per_step:.3f} share={share:.2f}%'
            )
        self._profile_cpu.clear()
        self._profile_cuda_events.clear()
        self._profile_window_step = self.global_step

    @staticmethod
    def _atomic_torch_save(state, file_path):
        """Write a checkpoint atomically so an interrupted save keeps the old file valid."""
        tmp_path = f'{file_path}.tmp.{os.getpid()}'
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, file_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _atomic_copy_file(source_path, destination_path):
        """Atomically copy a checkpoint without exposing a partial destination."""
        os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
        tmp_path = f'{destination_path}.tmp.{os.getpid()}'
        try:
            with open(source_path, 'rb') as source, open(tmp_path, 'xb') as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(tmp_path, destination_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _atomic_write_text(text, file_path):
        """Atomically write a UTF-8 text file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        tmp_path = f'{file_path}.tmp.{os.getpid()}'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, file_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @classmethod
    def _atomic_json_dump(cls, value, file_path):
        cls._atomic_write_text(
            json.dumps(
                cls._json_safe(value),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ) + '\n',
            file_path,
        )

    @staticmethod
    def _sha256_file(file_path):
        digest = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _json_safe(cls, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): cls._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        return repr(value)

    def _model_state_sha256(self):
        """Fingerprint all model parameters and buffers in a stable key order."""
        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode('utf-8'))
            if not torch.is_tensor(tensor):
                digest.update(repr(tensor).encode('utf-8'))
                continue
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode('ascii'))
            digest.update(str(tuple(value.shape)).encode('ascii'))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    @contextmanager
    def _use_model_variant(self, variant):
        """Temporarily expose either raw or EMA parameters, restoring raw on exit."""
        if variant == 'raw':
            yield
            return
        if variant != 'ema':
            raise ValueError(f'Unsupported model variant: {variant!r}')
        if self.ema is None:
            raise RuntimeError('EMA evaluation requested, but no EMA state is loaded')

        self.ema.store()
        self.ema.copy_to()
        try:
            yield
        finally:
            self.ema.restore()

    @staticmethod
    def _git_provenance(repo_root):
        def run_git(*args):
            result = subprocess.run(
                ['git', '-C', repo_root, *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Leading spaces in `git status --porcelain` encode index/worktree
            # state and must not be stripped.
            return result.stdout.rstrip('\r\n')

        try:
            commit = run_git('rev-parse', 'HEAD').strip()
            status = run_git('status', '--porcelain')
            return {
                'commit': commit,
                'dirty': bool(status),
                'status_porcelain': status.splitlines(),
            }
        except (OSError, subprocess.CalledProcessError) as exc:
            return {
                'commit': None,
                'dirty': None,
                'error': str(exc),
            }

    @staticmethod
    def _evaluation_code_sha256(repo_root):
        code_files = [
            os.path.join(repo_root, 'main_nerf.py'),
            os.path.join(repo_root, 'nerf', 'provider.py'),
            os.path.join(repo_root, 'nerf', 'utils.py'),
        ]
        return {
            os.path.relpath(path, repo_root): Trainer._sha256_file(path)
            for path in code_files
        }

    def _environment_provenance(self):
        cuda_device = None
        if torch.cuda.is_available():
            cuda_device = {
                'name': torch.cuda.get_device_name(self.device),
                'capability': list(torch.cuda.get_device_capability(self.device)),
                'cuda_runtime': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(),
            }
        return {
            'python': sys.version,
            'platform': platform.platform(),
            'torch': torch.__version__,
            'numpy': np.__version__,
            'opencv': cv2.__version__,
            'device': str(self.device),
            'cuda': cuda_device,
        }

    def _is_managed_checkpoint_path(self, file_path):
        """Only checkpoint files directly inside this Trainer's workspace are removable."""
        return (
            os.path.dirname(os.path.realpath(file_path))
            == os.path.realpath(self.ckpt_path)
        )

    @staticmethod
    def _capture_rng_state():
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_state(rng_state):
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'].cpu())
        if torch.cuda.is_available() and rng_state.get('cuda') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng_state['cuda']])

    def _validate_resume_config(self, saved_config):
        current_config = vars(self.opt)
        differences = []
        for key in sorted(set(saved_config).intersection(current_config)):
            if key in self._RESUME_CONFIG_IGNORED_KEYS:
                continue
            if saved_config[key] != current_config[key]:
                differences.append(
                    f'{key}: checkpoint={saved_config[key]!r}, current={current_config[key]!r}'
                )

        if differences:
            details = '\n  '.join(differences)
            raise ValueError(
                'Refusing to resume with a different experiment configuration:\n'
                f'  {details}'
            )

    def _optimize_with_amp_retry(self, closure):
        """Run one optimizer update, retrying the same stochastic batch after AMP overflow."""
        max_retries = getattr(self.opt, 'amp_max_retries', 4)
        with self._profile_phase('rng_snapshot'):
            retry_rng_state = self._capture_rng_state()

        for attempt in range(max_retries + 1):
            if attempt > 0:
                self._restore_rng_state(retry_rng_state)

            self.optimizer.zero_grad()

            with self._profile_phase('forward'):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = closure()

            with self._profile_phase('finite_check'):
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        f'Non-finite forward loss at global step {self.global_step}; '
                        'AMP scale reduction cannot repair a non-finite forward pass.'
                    )

            with self._profile_phase('backward_optimizer'):
                scale_before = self.scaler.get_scale()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()

            # GradScaler skips optimizer.step() and reduces its scale when a
            # gradient overflow is found. Replay the same stochastic batch so
            # global_step continues to count successful optimizer updates.
            overflow = self.fp16 and scale_after < scale_before
            if not overflow:
                return preds, truths, loss

            if attempt == max_retries:
                raise FloatingPointError(
                    f'AMP gradients still overflow at global step {self.global_step} '
                    f'after {max_retries} retries; final scale={scale_after:g}.'
                )

            self.log(
                f'[WARN] AMP gradient overflow at global step {self.global_step}: '
                f'scale {scale_before:g} -> {scale_after:g}; replaying batch '
                f'({attempt + 1}/{max_retries}).'
            )

        raise RuntimeError('unreachable AMP retry state')


 

    def build_virtual_rays(self, rays_o, rays_d, z_vals, weights, K=10):
        """
        根据原始射线构造 K 条虚拟射线

        rays_o: [B, N, 3]
        rays_d: [B, N, 3]
        z_vals: [B*N, T] 或 [B, N, T]
        weights: [B*N, T] 或 [B, N, T]

        return:
            O_prime: [B*N*K, 3]
            d_prime: [B*N*K, 3]
            P_s:     [B*N, 3]
            t_s:     [B*N]
        """
        with torch.no_grad():
            B, N, _ = rays_o.shape
            M = B * N

            rays_o_f = rays_o.reshape(M, 3)
            rays_d_f = rays_d.reshape(M, 3)

            z_vals = z_vals.reshape(M, -1).detach()
            weights = weights.reshape(M, -1).detach()

            # 1) 找到权重最大的采样点索引 s
            s_idx = torch.argmax(weights, dim=-1, keepdim=True)   # [M, 1]

            # 2) 取出对应的 t_s
            t_s = torch.gather(z_vals, dim=-1, index=s_idx).squeeze(-1)  # [M]

            # 3) P_s = O + t_s d
            P_s = rays_o_f + t_s.unsqueeze(-1) * rays_d_f  # [M, 3]

            # 4) 半径 r = ||O - P_s||
            radius = torch.norm(rays_o_f - P_s, dim=-1).clamp_min(1e-6)  # [M]

            # 5) 在单位球面上均匀采样 K 个方向
            sphere_dir = uniform_sphere_sampling((M, K), device=rays_o.device, dtype=rays_o.dtype)  # [M, K, 3]

            # 6) O' = P_s + r * dir
            P_s_k = P_s[:, None, :].expand(M, K, 3)
            O_prime = P_s_k + radius[:, None, None] * sphere_dir  # [M, K, 3]

            # 7) d' 指向 P_s
            d_prime = F.normalize(P_s_k - O_prime, dim=-1, eps=1e-6)  # [M, K, 3]

            return (
                O_prime.reshape(-1, 3),
                d_prime.reshape(-1, 3),
                P_s,
                t_s
            )
    #######结束######
    @staticmethod
    def compute_neurtv_loss(model, bound, num_samples, device, fd_epsilon):
        """
        Use central finite differences to approximate density-field spatial
        gradients, then compute their L1 norm as NeurTV loss.

        The hash-grid encoder has a custom CUDA backward without reliable
        higher-order derivatives. Finite differences keep the NeurTV objective
        differentiable with respect to model parameters using first-order
        backpropagation only.
        """
        fd_step = float(bound) * fd_epsilon
        sample_extent = float(bound) - fd_step
        if sample_extent <= 0:
            raise ValueError('NeurTV finite-difference step must be smaller than the scene bound')

        with torch.cuda.amp.autocast(enabled=False):
            points = (
                torch.rand(num_samples, 3, device=device, dtype=torch.float32) * 2 - 1
            ) * sample_extent
            offsets = torch.eye(3, device=device, dtype=points.dtype) * fd_step

            points_plus = points[:, None, :] + offsets[None, :, :]
            points_minus = points[:, None, :] - offsets[None, :, :]
            query_points = torch.cat([points_plus, points_minus], dim=1).reshape(-1, 3)

            sigma = model.density(query_points)['sigma'].reshape(num_samples, 6, -1)
            sigma_plus = sigma[:, :3]
            sigma_minus = sigma[:, 3:]
            density_grad = (sigma_plus - sigma_minus) / (2 * fd_step)

            return density_grad.abs().sum(dim=1).mean()

    ### ------------------------------

    def train_step(self, data, adv_perturb={}):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        # for fre_nll after input encoding and anneal near-far
        total_iter = int(self.opt.iters * self.opt.total_iter_end_rate)
        current_iter = self.global_step

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, perturb=True, force_all_rays=True, current_iter=current_iter, total_iter=total_iter, **vars(self.opt))
            pred_rgb = outputs['image'][:, :self.opt.num_rays].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in train_step
            #torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)
            
            return pred_rgb, None, loss

        images = data['images'] # [B, N, 3/4]

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            eval_bg = getattr(self.opt, 'eval_bg_color', None) or getattr(self.opt, 'bg_mode', 'white')
            bg_color = {'black': 0, 'gray': 0.5}.get(eval_bg, 1)
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        if len(adv_perturb) != 0:
            outputs_adv = self.model.render(rays_o, rays_d, data['cam_id'], staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, adv_perturb=adv_perturb, current_iter=current_iter, total_iter=total_iter, **vars(self.opt)) #with adv
            pred_rgb_adv = outputs_adv['image'][:, :self.opt.num_rays]
            loss_adv = self.criterion(pred_rgb_adv, gt_rgb).mean(-1)  # [B, N, 3] --> [B, N]

            
            


        with self._profile_phase('main_render'):
            outputs_clean = self.model.render(
                rays_o, rays_d,
                staged=False,
                bg_color=bg_color,
                perturb=True,
                force_all_rays=False if self.opt.patch_size == 1 else True,
                current_iter=current_iter,
                total_iter=total_iter,
                **vars(self.opt)
            )  # without adv

        pred_rgb_clean = outputs_clean['image'][:, :self.opt.num_rays]
        # MSE loss
        loss_clean = self.criterion(pred_rgb_clean, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
        
        if len(adv_perturb) != 0:
            loss = (1.0 - self.opt.adv_lambda) * loss_clean + self.opt.adv_lambda * loss_adv
        else:
            loss = loss_clean

        # =========================
        # virtual ray augmentation
        # =========================
        if self.opt.virtual_ray and self.global_step >= self.opt.virtual_ray_start_iter:
            # 单独调用一次 run()，拿到 weights / z_vals
            render_kwargs = vars(self.opt).copy()
            for k in [
                'num_rays', 'num_steps', 'upsample_steps',
                'start_ptr', 'current_iter', 'total_iter',
                'return_intermediates', 'cam_id', 'bg_color',
                'perturb', 'adv_perturb'
            ]:
                render_kwargs.pop(k, None)
            with self._profile_phase('virtual_original_render'):
                coarse_outputs = self.model.run(
                    rays_o, rays_d,
                    cam_id=data.get('cam_id', None),
                    num_steps=self.opt.num_steps,
                    upsample_steps=self.opt.upsample_steps,
                    bg_color=None,
                    perturb=True,
                    adv_perturb={},
                    current_iter=current_iter,
                    total_iter=total_iter,
                    start_ptr=1,
                    return_intermediates=True,
                    geometry_only=True,
                    **render_kwargs
                )

            orig_weights = coarse_outputs['weights'].detach()   # [M, T]
            orig_z_vals = coarse_outputs['z_vals'].detach()     # [M, T]

            # normalize original distribution
            orig_p = orig_weights / (orig_weights.sum(dim=-1, keepdim=True) + 1e-8)  # [M, T]

            # generate K virtual rays per original ray
            virtual_k = self.opt.virtual_ray_k
            with self._profile_phase('virtual_build'):
                O_prime, d_prime, P_s, _ = self.build_virtual_rays(
                    rays_o, rays_d, orig_z_vals, orig_weights, K=virtual_k
                )  # O_prime/d_prime: [M*K, 3], P_s: [M, 3]

            # render virtual rays
            with self._profile_phase('virtual_screen_render'):
                virtual_outputs = self.model.run(
                    O_prime.unsqueeze(0),   # [1, M*K, 3]
                    d_prime.unsqueeze(0),   # [1, M*K, 3]
                    cam_id=None,
                    num_steps=self.opt.num_steps,
                    upsample_steps=self.opt.upsample_steps,
                    bg_color=None,
                    perturb=True,
                    adv_perturb={},
                    current_iter=current_iter,
                    total_iter=total_iter,
                    start_ptr=1,
                    return_intermediates=True,
                    geometry_only=True,
                    **render_kwargs
                )

            with self._profile_phase('virtual_jsd_mask'):
                virtual_weights = virtual_outputs['weights'].detach()   # [M*K, T]
                virtual_weights = virtual_weights.reshape(-1, virtual_k, virtual_weights.shape[-1])  # [M, K, T]
                virtual_q = virtual_weights / (virtual_weights.sum(dim=-1, keepdim=True) + 1e-8)    # [M, K, T]

                # JSD(p || q)
                orig_p_rep = orig_p[:, None, :].expand_as(virtual_q)  # [M, K, T]
                eps = 1e-8
                m = 0.5 * (orig_p_rep + virtual_q)
                jsd = 0.5 * (
                    (orig_p_rep * (torch.log(orig_p_rep + eps) - torch.log(m + eps))).sum(dim=-1) +
                    (virtual_q * (torch.log(virtual_q + eps) - torch.log(m + eps))).sum(dim=-1)
                )  # [M, K]

                mask = jsd < self.opt.virtual_ray_jsd_th   # [M, K]

                # 保留下来的虚拟射线
                mask_flat = mask.reshape(-1)  # [M*K]
            if mask_flat.any():
                selected_O = O_prime[mask_flat]       # [Ns, 3]
                selected_d = d_prime[mask_flat]       # [Ns, 3]

                # 对应的 P_s 也展开到每个虚拟射线
                P_s_rep = P_s[:, None, :].expand(-1, virtual_k, -1).reshape(-1, 3)
                selected_Ps = P_s_rep[mask_flat]      # [Ns, 3]

                # 再渲染保留的虚拟射线
                with self._profile_phase('virtual_selected_render'):
                    selected_outputs = self.model.run(
                        selected_O.unsqueeze(0),   # [1, Ns, 3]
                        selected_d.unsqueeze(0),   # [1, Ns, 3]
                        cam_id=None,
                        num_steps=self.opt.num_steps,
                        upsample_steps=self.opt.upsample_steps,
                        bg_color=None,
                        perturb=True,
                        adv_perturb={},
                        current_iter=current_iter,
                        total_iter=total_iter,
                        start_ptr=1,
                        return_intermediates=False,
                        geometry_only=True,
                        **render_kwargs
                    )

                pred_depth_virtual = selected_outputs['depth'].reshape(-1)  # [Ns]
                target_depth_virtual = torch.norm(selected_O - selected_Ps, dim=-1)  # [Ns]

                loss_virtual_depth = F.l1_loss(pred_depth_virtual, target_depth_virtual)
                loss = loss + self.opt.virtual_ray_depth_lambda * loss_virtual_depth
             

        # patch-based rendering
        if self.opt.patch_size > 1:
            ps = self.opt.patch_size
            reshape_to_patch = lambda x, dim: x.reshape(-1, ps, ps, dim)
            # only the first num_rays rays carry patch structure; entropy /
            # smoothing rays appended by the collate function have none.
            depth = reshape_to_patch(
                outputs_clean['depth'].reshape(-1)[:self.opt.num_rays], 1)

            weighting = reshape_to_patch(
                outputs_clean['weights_sum'].reshape(-1)[:self.opt.num_rays], 1)[:, :-1, :-1]
            if self.opt.rgb_weighting:
                num_rays_patches = self.opt.num_rays // (ps ** 2)
                weighting_rgb = reshape_to_patch(torch.exp(-torch.abs(pred_rgb_clean-gt_rgb)/self.opt.patch_gamma).mean(-1), 1)[:, :-1, :-1]
                weighting = torch.cat([weighting_rgb, weighting[num_rays_patches:]], dim=0)


            loss_georeg = self.compute_ds_loss(depth, 'l2', weighting).mean()
            loss = loss + self.opt.depth_reg_lambda * loss_georeg

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()

        if getattr(self.opt, 'pose_app_reg', 0) > 0 and 'pose_app_reg' in outputs_clean:
            loss = loss + self.opt.pose_app_reg * outputs_clean['pose_app_reg']


        # --- NeurTV regularization on density field ---
        if (
            getattr(self.opt, 'neurtv', False)
            and self.global_step >= getattr(self.opt, 'neurtv_start_iter', 0)
            and (
                getattr(self.opt, 'neurtv_end_iter', None) is None
                or self.global_step <= self.opt.neurtv_end_iter
            )
        ):
            with self._profile_phase('neurtv'):
                tv_loss = self.compute_neurtv_loss(
                    self.model,
                    bound=self.opt.bound,
                    num_samples=self.opt.neurtv_num_samples,
                    device=self.device,
                    fd_epsilon=self.opt.neurtv_fd_epsilon
                )
            loss = loss + self.opt.neurtv_lambda * tv_loss

            # 记录到日志
            if self.local_rank == 0 and self.global_step % 100 == 0:
                self.log(f"[NeurTV] step {self.global_step}: tv_loss = {tv_loss.item():.6f}")

        ### Geometric diffusion loss
        if self.opt.diff_reg and self.global_step >= self.opt.diff_reg_start_iter and self.global_step/self.opt.iters < self.opt.diff_reg_end_rate:
            # loss_dist
            if self.opt.loss_dist:
                weights = outputs_clean['weights']
                z_vals = outputs_clean['z_vals']
                depth = None
                #loss_dist
                if self.opt.use_depth:
                    depth = outputs_clean['depth']
                with self._profile_phase('distortion_loss'):
                    loss_dist = self.distortion_loss(z_vals, weights, depth)
                loss = loss + loss_dist * self.opt.dist_lambda

            #loss_fg
            if self.opt.loss_fg:
                weights_sum = outputs_clean['weights_sum'][:self.opt.num_rays]
                if images.shape[-1] == 4:
                    # alpha-channel data: push alpha->1 only inside the object
                    # mask, and suppress alpha on background rays (DNGaussian-style)
                    alpha_gt = images[0, :self.opt.num_rays, 3].detach()
                    fg_mask = (alpha_gt > 0.5).float()
                    n_fg = fg_mask.sum().clamp(min=1)
                    n_bg = (1 - fg_mask).sum().clamp(min=1)
                    loss_fg = (
                        ((1 - weights_sum)**2 * fg_mask).sum() / n_fg
                        + (weights_sum**2 * (1 - fg_mask)).sum() / n_bg
                    )
                else:
                    loss_fg = (1 - weights_sum)**2
                    loss_fg = loss_fg.mean()
                loss = loss + loss_fg * self.opt.fg_lambda


        if self.opt.entropy or self.opt.smoothing:
            acc_raw = outputs_clean['weights_sum']
            alpha_raw = outputs_clean['alpha']

        entropy_ray_zvals_loss = 0
        smoothing_loss = 0


        # Ray Entropy Minimiation Loss

        if self.opt.entropy:
            entropy_ray_zvals_loss = self.fun_entropy_loss.ray_zvals(alpha_raw, acc_raw)
        if self.opt.entropy_end_iter is not None:
            if self.global_step > self.opt.entropy_end_iter:
                entropy_ray_zvals_loss = 0

        # Infomation Gain Reduction Loss (KL-Divergence)

        smoothing_lambda = self.opt.smoothing_lambda * self.opt.smoothing_rate ** (int(self.global_step / self.opt.smoothing_step_size))

        if self.opt.smoothing:
            with self._profile_phase('smoothing_loss'):
                smoothing_loss = self.fun_KL_divergence_loss(outputs_clean['weights'])
            if self.opt.smoothing_end_iter is not None:
                if self.global_step > self.opt.smoothing_end_iter:
                    smoothing_loss = 0

        loss = loss \
              + self.opt.entropy_ray_zvals_lambda * entropy_ray_zvals_loss \
              + smoothing_lambda * smoothing_loss

        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss = loss + loss_ws.mean()

        return pred_rgb_clean, gt_rgb, loss

    def compute_ds_loss(self, values, losstype='l2', weighting=None):

        v00 = values[:, :-1, :-1]
        v01 = values[:, :-1, 1:]
        v10 = values[:, 1:, :-1]

        if losstype == 'l2':
            loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
        elif losstype == 'l1':
            loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
        else:
            raise ValueError('Not supported losstype.')

        if weighting is not None:
            loss = loss * weighting
        return loss

    def distortion_loss(self, z_vals, weights, depth):

        assert z_vals is not None and weights is not None, "Ray samples must have z_vals and weights"

        # suppose we want to compute loss_dist on some training rays in range num_rays
        size = self.opt.diff_num_rays
        if self.opt.num_rays < size:
            size = self.opt.num_rays
        inds = np.random.choice(self.opt.num_rays, size=size, replace=False)
        z_vals = z_vals[inds]
        weights = weights[inds]
        if depth is not None:
            depth = torch.permute(depth, (1, 0))   # from (1, ...) to (..., 1)
            depth = depth[inds]

        starts = z_vals[..., :, :-1, None]  # add a temporary dimension for computation
        ends = z_vals[..., :, 1:, None]
        midpoints = (starts + ends) / 2.0  # (..., num_samples, 1)
        weights_new = weights[..., :, :-1, None]    # to match midpoints dimension

        """"
        # avoid CUDA OOM
        loss = torch.empty(0, device=self.device)
        for i_midpoints, i_weights in zip(midpoints, weights_new):
            i_loss = (i_weights * i_weights[..., None, :, 0] * torch.abs(i_midpoints - i_midpoints[..., None, :, 0]))
            i_loss = torch.sum(i_loss, dim=(-1, -2))[..., None]
            loss = torch.cat([loss, i_loss], dim=0)
        loss = loss[..., None]
        """

        loss = (
                weights_new * weights_new[..., None, :, 0] * torch.abs(midpoints - midpoints[..., None, :, 0])
        )  # (..., num_samples, num_samples)
        loss = torch.sum(loss, dim=(-1, -2))[..., None]  # (..., num_samples)
        loss = loss + 1 / 3.0 * torch.sum(weights_new ** 2 * (ends - starts), dim=-2)

        # manage too low depth values
        if depth is not None:
            depth = depth + 1e-4


        if depth is not None:
            loss = loss / depth

        loss = loss.mean()

        return loss

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        eval_bg = getattr(self.opt, 'eval_bg_color', None) or getattr(self.opt, 'bg_mode', 'white')
        bg_color = {'black': 0, 'gray': 0.5}.get(eval_bg, 1)
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        render_kwargs = dict(vars(self.opt))
        if getattr(self.opt, 'eval_num_steps', None):
            render_kwargs['num_steps'] = self.opt.eval_num_steps
        if getattr(self.opt, 'eval_upsample_steps', None):
            render_kwargs['upsample_steps'] = self.opt.eval_upsample_steps
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **render_kwargs)

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map

        self.log(f"\nARGS: {self.opt}\n")

        missed_milestones = []
        if self.workspace is not None:
            missed_milestones = [
                step
                for step in sorted(self.milestone_steps)
                if step <= self.global_step
                and not os.path.isfile(self._milestone_checkpoint_path(step))
            ]
        if missed_milestones:
            self.log(
                '[WARN] Training resumed after milestone steps '
                f'{missed_milestones}, but their full checkpoints do not exist. '
                'Past model states cannot be reconstructed and will not be backfilled.'
            )

        steps_per_epoch = len(train_loader)
        target_global_step = max_epochs * steps_per_epoch
        if self.global_step > target_global_step:
            raise ValueError(
                f'global_step={self.global_step} exceeds the training target '
                f'{target_global_step}'
            )
        if self._profile_enabled:
            self._profile_cpu.clear()
            self._profile_cuda_events.clear()
            self._profile_window_step = self.global_step

        self._train_max_epochs = max_epochs
        self._train_progress = None
        if self.local_rank == 0:
            self._train_progress = tqdm.tqdm(
                total=target_global_step,
                initial=self.global_step,
                unit='step',
                mininterval=1.0,
                dynamic_ncols=True,
                desc=f'Epoch {self.epoch}/{max_epochs}',
                bar_format=(
                    '{desc} |{bar}| step {n_fmt}/{total_fmt} '
                    '[{elapsed} 已用, {remaining} 剩余] {postfix}'
                ),
            )

        try:
            perf_window_start = time.perf_counter()
            perf_window_step = self.global_step
            perf_train_seconds = 0.0
            perf_checkpoint_seconds = 0.0
            perf_validation_seconds = 0.0
            for epoch in range(self.epoch + 1, max_epochs + 1):
                self.epoch = epoch

                phase_start = time.perf_counter()
                self.train_one_epoch(train_loader)
                perf_train_seconds += time.perf_counter() - phase_start

                profile_report_due = (
                    self._profile_enabled
                    and (
                        self.global_step == target_global_step
                        or self.global_step - self._profile_window_step
                        >= self.opt.profile_report_steps
                    )
                )
                if profile_report_due:
                    self._profile_report()

                checkpoint_due = (
                    self.global_step == target_global_step
                    or self.global_step in self.milestone_steps
                    or self.global_step % self.opt.checkpoint_interval_steps == 0
                )
                if (
                    checkpoint_due
                    and self.workspace is not None
                    and self.local_rank == 0
                ):
                    phase_start = time.perf_counter()
                    self.save_checkpoint(full=True, best=False)
                    self.save_milestone_checkpoint()
                    perf_checkpoint_seconds += time.perf_counter() - phase_start

                if self.epoch % self.eval_interval == 0:
                    phase_start = time.perf_counter()
                    self.evaluate_one_epoch(valid_loader)
                    self.save_checkpoint(full=False, best=True)
                    perf_validation_seconds += time.perf_counter() - phase_start

                if checkpoint_due and self.local_rank == 0:
                    window_seconds = time.perf_counter() - perf_window_start
                    window_steps = self.global_step - perf_window_step
                    throughput = (
                        window_steps / window_seconds if window_seconds > 0 else 0.0
                    )
                    self.log(
                        f'[PERF] steps={perf_window_step}->{self.global_step} '
                        f'wall={window_seconds:.3f}s train={perf_train_seconds:.3f}s '
                        f'checkpoint={perf_checkpoint_seconds:.3f}s '
                        f'validation={perf_validation_seconds:.3f}s '
                        f'throughput={throughput:.3f} step/s'
                    )
                    perf_window_start = time.perf_counter()
                    perf_window_step = self.global_step
                    perf_train_seconds = 0.0
                    perf_checkpoint_seconds = 0.0
                    perf_validation_seconds = 0.0
        finally:
            if self._train_progress is not None:
                self._train_progress.close()
                self._train_progress = None
            if self.use_tensorboardX and self.local_rank == 0:
                self.writer.close()

    def evaluate(self, loader, name=None, type=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name, type=type)
        self.use_tensorboardX = use_tensorboardX

    @staticmethod
    def _metric_to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _write_png(file_path, rgb_or_gray):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        success = cv2.imwrite(file_path, rgb_or_gray)
        if not success:
            raise IOError(f'OpenCV failed to write image: {file_path}')

    def _evaluate_variant_to_directory(self, loader, variant, output_dir):
        """Evaluate one parameter variant and archive per-frame float metrics."""
        if self.world_size != 1 or self.local_rank != 0:
            raise NotImplementedError(
                'Formal evaluation archives currently require a single process'
            )
        if not loader.has_gt:
            raise ValueError('Formal evaluation requires ground-truth images')

        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        meters = {
            'psnr': PSNRMeter(),
            'lpips_alex': LPIPSMeter(net='alex', device=self.device),
            'ssim': SSIMMeter(device=self.device),
        }
        metric_sums = {name: 0.0 for name in meters}
        per_frame = []
        total_loss = 0.0
        expected_frame_ids = list(loader._data.frame_ids)

        self.log(
            f'==> Formal {variant.upper()} evaluation: '
            f'{len(expected_frame_ids)} frames -> {output_dir}'
        )
        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format=(
                '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} '
                '[{elapsed}<{remaining}, {rate_fmt}]'
            ),
        )

        with torch.no_grad():
            for data in loader:
                if 'frame_id' not in data:
                    raise KeyError(
                        'Evaluation batch has no frame_id; dataset provenance '
                        'cannot be established'
                    )
                frame_ids = data['frame_id'].reshape(-1).tolist()
                if len(frame_ids) != 1:
                    raise ValueError(
                        'Formal evaluation currently requires batch_size=1; '
                        f'got frame IDs {frame_ids}'
                    )
                frame_id = int(frame_ids[0])

                # eval_step converts images in-place for linear color space.
                # Clone the tensor so the second model variant sees identical GT.
                eval_data = dict(data)
                eval_data['images'] = data['images'].clone()

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(eval_data)

                if not torch.isfinite(preds).all():
                    raise FloatingPointError(
                        f'Non-finite RGB prediction for {variant} frame {frame_id}'
                    )
                if not torch.isfinite(preds_depth).all():
                    raise FloatingPointError(
                        f'Non-finite depth prediction for {variant} frame {frame_id}'
                    )

                frame_metrics = {}
                for metric_name, meter in meters.items():
                    meter.clear()
                    meter.update(preds, truths)
                    metric_value = self._metric_to_float(meter.measure())
                    if not math.isfinite(metric_value):
                        raise FloatingPointError(
                            f'Non-finite {metric_name} for '
                            f'{variant} frame {frame_id}'
                        )
                    frame_metrics[metric_name] = metric_value
                    metric_sums[metric_name] += metric_value

                loss_value = float(loss.detach().cpu().item())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(
                        f'Non-finite MSE loss for {variant} frame {frame_id}'
                    )
                total_loss += loss_value

                display_preds = preds
                display_truths = truths
                if self.opt.color_space == 'linear':
                    display_preds = linear_to_srgb(display_preds)
                    display_truths = linear_to_srgb(display_truths)

                pred_rgb = np.clip(
                    display_preds[0].detach().cpu().numpy(),
                    0.0,
                    1.0,
                )
                truth_rgb = np.clip(
                    display_truths[0].detach().cpu().numpy(),
                    0.0,
                    1.0,
                )
                pred_rgb_u8 = np.rint(pred_rgb * 255.0).astype(np.uint8)
                truth_rgb_u8 = np.rint(truth_rgb * 255.0).astype(np.uint8)
                depth = preds_depth[0].detach().float().cpu().numpy()

                prefix = f'frame_{frame_id:04d}'
                rgb_name = f'{prefix}_rgb.png'
                gt_name = f'{prefix}_gt.png'
                depth_name = f'{prefix}_depth.npy'
                depth_preview_name = f'{prefix}_depth_preview.png'

                self._write_png(
                    os.path.join(frames_dir, rgb_name),
                    cv2.cvtColor(pred_rgb_u8, cv2.COLOR_RGB2BGR),
                )
                self._write_png(
                    os.path.join(frames_dir, gt_name),
                    cv2.cvtColor(truth_rgb_u8, cv2.COLOR_RGB2BGR),
                )
                np.save(os.path.join(frames_dir, depth_name), depth)

                finite_depth = depth[np.isfinite(depth)]
                depth_min = float(finite_depth.min())
                depth_max = float(finite_depth.max())
                if depth_max > depth_min:
                    depth_preview = (
                        (depth - depth_min) / (depth_max - depth_min) * 255.0
                    )
                else:
                    depth_preview = np.zeros_like(depth)
                depth_preview = np.clip(depth_preview, 0, 255).astype(np.uint8)
                self._write_png(
                    os.path.join(frames_dir, depth_preview_name),
                    depth_preview,
                )

                per_frame.append({
                    'frame_id': frame_id,
                    'metrics': frame_metrics,
                    'mse_loss': loss_value,
                    'files': {
                        'rgb': os.path.join('frames', rgb_name),
                        'ground_truth': os.path.join('frames', gt_name),
                        'depth_float32': os.path.join('frames', depth_name),
                        'depth_preview': os.path.join(
                            'frames',
                            depth_preview_name,
                        ),
                    },
                    'depth_range': {
                        'min': depth_min,
                        'max': depth_max,
                    },
                })

                pbar.set_description(
                    f'{variant} frame={frame_id} '
                    f'PSNR={frame_metrics["psnr"]:.4f}'
                )
                pbar.update(loader.batch_size)

        pbar.close()
        seen_frame_ids = [item['frame_id'] for item in per_frame]
        if seen_frame_ids != expected_frame_ids:
            raise RuntimeError(
                f'{variant} evaluation frame order mismatch: '
                f'expected {expected_frame_ids}, got {seen_frame_ids}'
            )
        if not per_frame:
            raise RuntimeError(f'{variant} evaluation produced no frames')

        count = len(per_frame)
        result = {
            'schema_version': 1,
            'variant': variant,
            'frame_count': count,
            'frame_ids': seen_frame_ids,
            'mean': {
                metric_name: metric_sum / count
                for metric_name, metric_sum in metric_sums.items()
            },
            'mean_mse_loss': total_loss / count,
            'per_frame': per_frame,
            'metric_protocol': {
                'psnr': 'per-image PSNR averaged across frames; higher is better',
                'lpips_alex': (
                    'LPIPS AlexNet with normalize=True, averaged across frames; '
                    'lower is better'
                ),
                'ssim': (
                    'torchmetrics structural_similarity_index_measure, '
                    'averaged across frames; higher is better'
                ),
            },
        }
        self._atomic_json_dump(result, os.path.join(output_dir, 'metrics.json'))
        self.log(
            f'<== {variant.upper()}: '
            f'PSNR={result["mean"]["psnr"]:.6f}, '
            f'LPIPS={result["mean"]["lpips_alex"]:.6f}, '
            f'SSIM={result["mean"]["ssim"]:.6f}'
        )
        return result

    @staticmethod
    def _compare_evaluation_variants(variant_results):
        directions = {
            'psnr': 'max',
            'lpips_alex': 'min',
            'ssim': 'max',
        }
        best_by_metric = {}
        for metric_name, direction in directions.items():
            values = {
                variant: result['mean'][metric_name]
                for variant, result in variant_results.items()
            }
            choose = max if direction == 'max' else min
            winner = choose(values, key=values.get)
            best_by_metric[metric_name] = {
                'direction': direction,
                'variant': winner,
                'value': values[winner],
                'all_values': values,
            }

        comparison = {
            'schema_version': 1,
            'variants': {
                variant: result['mean']
                for variant, result in variant_results.items()
            },
            'best_by_metric': best_by_metric,
            'selection_policy': {
                'scene_level': (
                    'Diagnostic only: retain both variants and record the '
                    'per-metric winner without hiding its source.'
                ),
                'paper_aggregate': (
                    'After all eight LLFF scenes finish, compare the eight-scene '
                    'macro mean and choose at most one global variant per metric; '
                    'never choose independently per scene before averaging.'
                ),
            },
        }
        if 'raw' in variant_results and 'ema' in variant_results:
            comparison['raw_minus_ema'] = {
                metric_name: (
                    variant_results['raw']['mean'][metric_name]
                    - variant_results['ema']['mean'][metric_name]
                )
                for metric_name in directions
            }
        return comparison

    def evaluate_variants_archive(
        self,
        loader,
        variants=('raw', 'ema'),
        checkpoint_path=None,
        expected_step=None,
        output_dir=None,
        split_type='test',
        overwrite=False,
    ):
        """Evaluate raw/EMA weights and atomically publish a provenance archive."""
        variants = list(variants)
        if not variants:
            raise ValueError('At least one evaluation variant is required')
        if len(variants) != len(set(variants)):
            raise ValueError(f'Duplicate evaluation variants: {variants}')
        unsupported = sorted(set(variants) - {'raw', 'ema'})
        if unsupported:
            raise ValueError(f'Unsupported evaluation variants: {unsupported}')
        if split_type not in {'test', 'val'}:
            raise ValueError(f'Unsupported formal evaluation split: {split_type}')

        checkpoint_path = checkpoint_path or self.last_checkpoint_path
        if checkpoint_path is None:
            raise ValueError('Formal evaluation requires an explicit loaded checkpoint')
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Formal evaluation checkpoint does not exist: {checkpoint_path}'
            )
        if self.last_checkpoint_path is not None:
            if os.path.realpath(checkpoint_path) != os.path.realpath(
                self.last_checkpoint_path
            ):
                raise ValueError(
                    'The requested archive checkpoint is not the checkpoint '
                    'currently loaded by the Trainer'
                )
        if expected_step is not None and self.global_step != expected_step:
            raise ValueError(
                f'Expected checkpoint global_step={expected_step}, '
                f'but loaded global_step={self.global_step}'
            )
        if self.epoch < 0 or self.global_step < 0:
            raise ValueError(
                f'Invalid checkpoint metadata: epoch={self.epoch}, '
                f'global_step={self.global_step}'
            )
        if (
            'ema' in variants
            and (self.ema is None or not self.ema_checkpoint_loaded)
        ):
            raise RuntimeError(
                'EMA was requested but the loaded checkpoint has no usable EMA state'
            )

        checkpoint_sha256 = self._sha256_file(checkpoint_path)
        split_file = getattr(self.opt, 'split_file', None)
        split_sha256 = None
        split_spec = None
        if split_file is not None:
            split_file = os.path.abspath(os.path.expanduser(split_file))
            split_sha256 = self._sha256_file(split_file)
            with open(split_file, 'r', encoding='utf-8') as f:
                split_spec = json.load(f)

        expected_frame_ids = list(loader._data.frame_ids)
        if split_spec is not None:
            declared_ids = split_spec.get(f'{split_type}_ids')
            if declared_ids is not None and list(declared_ids) != expected_frame_ids:
                raise ValueError(
                    f'Loader frame IDs {expected_frame_ids} do not match '
                    f'{split_type}_ids in {split_file}: {declared_ids}'
                )

        if output_dir is None:
            output_dir = os.path.join(
                self.workspace,
                'evaluation',
                f'step_{self.global_step:06d}',
            )
        output_dir = os.path.abspath(output_dir)
        workspace_path = os.path.abspath(self.workspace)
        if (
            os.path.commonpath([workspace_path, output_dir]) != workspace_path
            or output_dir == workspace_path
        ):
            raise ValueError(
                f'Evaluation output must be a child of workspace '
                f'{workspace_path}, got {output_dir}'
            )
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        evaluation_code_sha256 = self._evaluation_code_sha256(repo_root)

        complete_path = os.path.join(output_dir, 'COMPLETE')
        if os.path.isfile(complete_path) and not overwrite:
            manifest_path = os.path.join(output_dir, 'manifest.json')
            comparison_path = os.path.join(output_dir, 'comparison.json')
            with open(complete_path, 'r', encoding='utf-8') as f:
                complete = json.load(f)
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            if manifest['checkpoint']['sha256'] != checkpoint_sha256:
                raise RuntimeError(
                    f'Existing archive at {output_dir} belongs to a different '
                    'checkpoint; use --eval_overwrite to preserve and replace it'
                )
            if manifest['evaluation']['variants'] != variants:
                raise RuntimeError(
                    f'Existing archive variants '
                    f'{manifest["evaluation"]["variants"]} do not match {variants}; '
                    'use --eval_overwrite'
                )
            if manifest['dataset']['frame_ids'] != expected_frame_ids:
                raise RuntimeError(
                    'Existing archive frame IDs do not match the current split; '
                    'use --eval_overwrite'
                )
            if manifest['dataset'].get('split_sha256') != split_sha256:
                raise RuntimeError(
                    'Existing archive split hash does not match the current split; '
                    'use --eval_overwrite'
                )
            if (
                manifest.get('provenance', {}).get('code_sha256')
                != evaluation_code_sha256
            ):
                raise RuntimeError(
                    'Existing archive was produced by different evaluation code; '
                    'use --eval_overwrite'
                )
            for variant in variants:
                metrics_path = os.path.join(output_dir, variant, 'metrics.json')
                if not os.path.isfile(metrics_path):
                    raise RuntimeError(
                        f'Archive has COMPLETE but is missing {metrics_path}'
                    )
            artifact_hashes = complete.get('artifact_sha256')
            if not isinstance(artifact_hashes, dict) or not artifact_hashes:
                raise RuntimeError(
                    'Existing archive COMPLETE has no artifact hash inventory; '
                    'use --eval_overwrite'
                )
            for relative_path, expected_sha256 in artifact_hashes.items():
                artifact_path = os.path.abspath(
                    os.path.join(output_dir, relative_path)
                )
                if os.path.commonpath([output_dir, artifact_path]) != output_dir:
                    raise RuntimeError(
                        f'Archive hash inventory escapes its root: {relative_path}'
                    )
                if not os.path.isfile(artifact_path):
                    raise RuntimeError(
                        f'Archive hash inventory references a missing file: '
                        f'{relative_path}'
                    )
                actual_sha256 = self._sha256_file(artifact_path)
                if actual_sha256 != expected_sha256:
                    raise RuntimeError(
                        f'Archive artifact hash mismatch for {relative_path}: '
                        f'{expected_sha256} != {actual_sha256}'
                    )
            with open(comparison_path, 'r', encoding='utf-8') as f:
                comparison = json.load(f)
            self.log(
                f'[INFO] Verified complete matching evaluation archive: {output_dir}'
            )
            return comparison

        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        staging_dir = f'{output_dir}.staging.{os.getpid()}'
        if os.path.exists(staging_dir):
            raise FileExistsError(
                f'Refusing to reuse an existing staging directory: {staging_dir}'
            )
        os.makedirs(staging_dir)

        raw_fingerprint_before = self._model_state_sha256()
        prior_training_mode = self.model.training
        variant_results = {}
        variant_fingerprints = {}
        try:
            for variant in variants:
                variant_dir = os.path.join(staging_dir, variant)
                os.makedirs(variant_dir)
                with self._use_model_variant(variant):
                    self.model.eval()
                    variant_fingerprints[variant] = self._model_state_sha256()
                    variant_results[variant] = self._evaluate_variant_to_directory(
                        loader,
                        variant,
                        variant_dir,
                    )

                restored_fingerprint = self._model_state_sha256()
                if restored_fingerprint != raw_fingerprint_before:
                    raise RuntimeError(
                        f'Raw model state changed after {variant} evaluation: '
                        f'{raw_fingerprint_before} -> {restored_fingerprint}'
                    )

            comparison = self._compare_evaluation_variants(variant_results)
            self._atomic_json_dump(
                comparison,
                os.path.join(staging_dir, 'comparison.json'),
            )
            if split_spec is not None:
                self._atomic_json_dump(
                    split_spec,
                    os.path.join(staging_dir, 'split_snapshot.json'),
                )

            dataset_root = os.path.abspath(loader._data.root_path)
            transforms_path = os.path.join(dataset_root, 'transforms.json')
            ema_state = self.ema.state_dict() if self.ema is not None else {}
            ema_num_updates = ema_state.get('num_updates')
            if torch.is_tensor(ema_num_updates):
                ema_num_updates = int(ema_num_updates.detach().cpu().item())
            manifest = {
                'schema_version': 1,
                'created_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                'checkpoint': {
                    'path': checkpoint_path,
                    'sha256': checkpoint_sha256,
                    'size_bytes': os.path.getsize(checkpoint_path),
                    'epoch': self.epoch,
                    'global_step': self.global_step,
                    'model_raw_sha256': raw_fingerprint_before,
                    'variant_model_sha256': variant_fingerprints,
                },
                'dataset': {
                    'root': dataset_root,
                    'scene': os.path.basename(os.path.normpath(dataset_root)),
                    'split_type': split_type,
                    'frame_ids': expected_frame_ids,
                    'frame_count': len(expected_frame_ids),
                    'split_file': split_file,
                    'split_sha256': split_sha256,
                    'transforms_sha256': (
                        self._sha256_file(transforms_path)
                        if os.path.isfile(transforms_path)
                        else None
                    ),
                },
                'evaluation': {
                    'variants': variants,
                    'fp16_autocast': self.fp16,
                    'color_space': self.opt.color_space,
                    'downscale': self.opt.downscale,
                    'background_color': 1,
                    'perturb': False,
                    'ema': {
                        'loaded_from_checkpoint': self.ema_checkpoint_loaded,
                        'decay': (
                            float(ema_state['decay'])
                            if 'decay' in ema_state
                            else None
                        ),
                        'num_updates': ema_num_updates,
                    },
                    'metric_means': {
                        variant: result['mean']
                        for variant, result in variant_results.items()
                    },
                },
                'experiment_config': (
                    self.checkpoint_config
                    if self.checkpoint_config is not None
                    else vars(self.opt)
                ),
                'evaluation_command': sys.argv,
                'provenance': {
                    'git': self._git_provenance(repo_root),
                    'environment': self._environment_provenance(),
                    'code_sha256': evaluation_code_sha256,
                },
            }
            self._atomic_json_dump(
                manifest,
                os.path.join(staging_dir, 'manifest.json'),
            )

            log_lines = [
                f'created_utc={manifest["created_utc"]}',
                f'checkpoint={checkpoint_path}',
                f'checkpoint_sha256={checkpoint_sha256}',
                f'epoch={self.epoch}',
                f'global_step={self.global_step}',
                f'split={split_type}',
                f'frame_ids={expected_frame_ids}',
            ]
            for variant in variants:
                mean = variant_results[variant]['mean']
                log_lines.append(
                    f'{variant}: PSNR={mean["psnr"]:.9f}, '
                    f'LPIPS_ALEX={mean["lpips_alex"]:.9f}, '
                    f'SSIM={mean["ssim"]:.9f}'
                )
            self._atomic_write_text(
                '\n'.join(log_lines) + '\n',
                os.path.join(staging_dir, 'eval.log'),
            )
            artifact_sha256 = {}
            for artifact_root, _, artifact_names in os.walk(staging_dir):
                for artifact_name in sorted(artifact_names):
                    artifact_path = os.path.join(artifact_root, artifact_name)
                    relative_path = os.path.relpath(artifact_path, staging_dir)
                    artifact_sha256[relative_path] = self._sha256_file(
                        artifact_path
                    )
            self._atomic_json_dump(
                {
                    'status': 'complete',
                    'checkpoint_sha256': checkpoint_sha256,
                    'global_step': self.global_step,
                    'variants': variants,
                    'artifact_sha256': artifact_sha256,
                },
                os.path.join(staging_dir, 'COMPLETE'),
            )

            backup_dir = None
            if os.path.exists(output_dir):
                backup_dir = f'{output_dir}.replaced.{timestamp}'
                if os.path.exists(backup_dir):
                    raise FileExistsError(
                        f'Evaluation backup path already exists: {backup_dir}'
                    )
                os.replace(output_dir, backup_dir)
            try:
                os.replace(staging_dir, output_dir)
            except Exception:
                if (
                    backup_dir is not None
                    and os.path.exists(backup_dir)
                    and not os.path.exists(output_dir)
                ):
                    os.replace(backup_dir, output_dir)
                raise

            self.log(f'[INFO] Published formal evaluation archive: {output_dir}')
            return comparison
        except Exception:
            if os.path.exists(staging_dir):
                failed_dir = f'{output_dir}.failed.{timestamp}.{os.getpid()}'
                os.replace(staging_dir, failed_dir)
                self.log(
                    f'[ERROR] Preserved failed evaluation staging directory: '
                    f'{failed_dir}'
                )
            raise
        finally:
            self.model.train(prior_training_mode)

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
    
    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            preds, truths, loss = self._optimize_with_amp_retry(
                lambda: self.train_step(data)
            )
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def broadcast_lists(self, *lists):
        lengths = [len(l) for l in lists]
        max_len = max(lengths)
        ret_lists = []
        for l in lists:
            if len(l) == max_len:
                ret_lists.append(l)
            elif len(l) == 1:
                ret_lists.append([l[0]] * max_len)
            else:
                raise ValueError(f'Unable to broadcast a list of length {len(l)} to {max_len}')
        return ret_lists

    def get_adv_templates(self, adv, H, W, near, far, num_images, num_rays, args):

        perturb_sizes, epsilons, alphas = {}, {}, {}
        iters, norms = {}, {}
        broadcast = self.broadcast_lists(args.adv, args.pgd_alpha, args.pgd_eps, args.pgd_iters, args.pgd_norm)
        for adv, pgd_alpha, pgd_eps, pgd_iters, pgd_norm in zip(*broadcast):
            iters[adv] = pgd_iters
            norms[adv] = pgd_norm

            if adv == 'zval_c':
                perturb_sizes['zval_c'] = [num_rays, args.num_steps]
                epsilons['zval_c'] = (far - near) / (args.num_steps - 1) / 2. * pgd_eps
                alphas['zval_c'] = pgd_alpha
            elif adv == 'pts_c':
                perturb_sizes['pts_c'] = [num_rays, args.num_steps, 3]
                epsilons['pts_c'] = min(2. / max(H, W), (far - near) / (args.num_steps - 1) / 2.) * pgd_eps
                alphas['pts_c'] = pgd_alpha

            elif adv == 'zval_f':
                perturb_sizes['zval_f'] = [num_rays, args.upsample_steps]
                epsilons['zval_f'] = (far - near) / (args.upsample_steps - 1) / 2. * pgd_eps
                alphas['zval_f'] = pgd_alpha
            elif adv == 'pts_f':
                perturb_sizes['pts_f'] = [num_rays, args.upsample_steps, 3]
                epsilons['pts_f'] = min(2. / max(H, W),
                                        (far - near) / (args.upsample_steps - 1) / 2.) * pgd_eps
                alphas['pts_f'] = pgd_alpha

            elif adv == 'raw_c':
                perturb_sizes['raw_c'] = [num_rays, args.num_steps, 1] #apply on sigma
                epsilons['raw_c'] = pgd_eps
                alphas['raw_c'] = pgd_alpha
            elif adv == 'raw_f':
                perturb_sizes['raw_f'] = [num_rays, args.upsample_steps, 1]
                epsilons['raw_f'] = pgd_eps
                alphas['raw_f'] = pgd_alpha

            elif adv == 'feat_c':
                perturb_sizes['feat_c'] = [num_rays, args.num_steps, self.model.geo_feat_dim+1] #+1 cause of ['sigma'] in MLP representation
                epsilons['feat_c'] = pgd_eps
                alphas['feat_c'] = pgd_alpha
            elif adv == 'feat_f':
                perturb_sizes['feat_f'] = [num_rays, args.upsample_steps, self.model.geo_feat_dim+1]
                epsilons['feat_f'] = pgd_eps
                alphas['feat_f'] = pgd_alpha

            elif adv == 'cam_r':
                perturb_sizes['cam_r'] = [num_images, 3]
                epsilons['cam_r'] = np.deg2rad(pgd_eps)
                alphas['cam_r'] = pgd_alpha
            elif adv == 'cam_t':
                perturb_sizes['cam_t'] = [num_images, 3]
                epsilons['cam_t'] = pgd_eps
                alphas['cam_t'] = pgd_alpha

            elif adv == 'rgb':
                perturb_sizes['rgb'] = [num_rays, 3]
                epsilons['rgb'] = pgd_eps
                alphas['rgb'] = pgd_alpha

        return perturb_sizes, epsilons, alphas, iters, norms

    def train_one_epoch(self, loader):
        # Match the old Python-float accumulation order while avoiding a
        # device synchronization at every step. One scalar is transferred at
        # the epoch boundary instead (six steps for LLFF 6-view).
        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        #self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        self.local_step = 0

        num_cameras = loader._data.images.shape[0]
        loader_iterator = iter(loader)
        for _ in range(len(loader)):
            profile_token = self._profile_begin('data_prepare')
            data = next(loader_iterator)
            self._profile_end(profile_token)
            step_profile_token = self._profile_begin('step_total')
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            adv_perturb={}
            if(len(self.opt.adv) != 0 and 'images' in data):
                # 1. Obtain adversarial sample
                self.model.eval()

                H = data['H']
                W = data['W']

                #near, far = raymarching._near_far_from_aabb(data['rays_o'], data['rays_d'], self.model.aabb_train, self.opt.min_near)
                #near.unsqueeze_(-1)
                #far.unsqueeze_(-1)
                near, far = -self.opt.bound, self.opt.bound

                perturb_sizes, epsilons, alphas, pgd_iters, pgd_norms = self.get_adv_templates(self.opt.adv, H, W, near, far,
                                                                                          num_cameras, self.opt.num_rays,
                                                                                          self.opt)
                if self.opt.adv_type == 'pgd':
                    adv_perturb = attack_pgd(self.opt, self.model, (data['rays_o'], data['rays_d'], (near, far), data['cam_id']), data['images'], perturb_sizes,
                                             epsilons, alphas, loss_fn=self.criterion, attack_iters=pgd_iters,
                                             norm=pgd_norms, unadv=self.opt.unadv)
                elif self.opt.adv_type == 'random':
                    adv_perturb = attack_random(self.model, (data['rays_o'], data['rays_d'], (near, far)), data['images'], perturb_sizes, epsilons,
                                                norm=pgd_norms)

                # 2. Apply weight perturbation if enabled
                if self.global_step >= self.opt.awp_warmup and self.awp_adversary is not None:
                    awp = self.awp_adversary.calc_awp(self.opt, inputs=(data['rays_o'], data['rays_d'], (near, far), data['cam_id']), targets=data['images'],
                                                 delta=adv_perturb, loss_fn=self.criterion, unadv=self.opt.unadv)
                    self.awp_adversary.perturb(awp)

            self.model.train()

            preds, truths, loss = self._optimize_with_amp_retry(
                lambda: self.train_step(data, adv_perturb=adv_perturb)
            )

            if self.scheduler_update_every_step:
                with self._profile_phase('lr_scheduler'):
                    self.lr_scheduler.step()

            total_loss.add_(loss.detach().to(dtype=torch.float64))

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self._train_progress is not None:
                    self._train_progress.set_description_str(
                        f'Epoch {self.epoch}/{self._train_max_epochs}',
                        refresh=False,
                    )
                    self._train_progress.update(1)
            self._profile_end(step_profile_token)

        if self.ema is not None:
            with self._profile_phase('ema_update'):
                self.ema.update()

        with self._profile_phase('epoch_loss_sync'):
            average_loss = total_loss.item() / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss", average_loss, self.global_step)
            if self._train_progress is not None:
                progress_values = {'avg_loss': f'{average_loss:.4f}'}
                if self.scheduler_update_every_step:
                    progress_values['lr'] = (
                        f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                    )
                self._train_progress.set_postfix(
                    progress_values,
                    refresh=False,
                )
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

    def evaluate_one_epoch(self, loader, name=None, type=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            table_metrics = []
            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if type == 'test' and self.opt.write_table:
                    table_metrics.append(metric.measure())
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()
            if type == 'test' and self.opt.write_table:
                write_table(table_metrics, [self.opt.implementation_name], self.opt.dataset_name)

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
            'config': vars(self.opt).copy(),
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            state['rng_state'] = self._capture_rng_state()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            old_ckpt = None
            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)

            self._atomic_torch_save(state, file_path)
            self.last_checkpoint_path = os.path.abspath(file_path)
            self.checkpoint_config = copy.deepcopy(state['config'])
            self.ema_checkpoint_loaded = self.ema is not None and 'ema' in state
            if old_ckpt is not None and old_ckpt != file_path:
                if not self._is_managed_checkpoint_path(old_ckpt):
                    self.log(
                        f'[WARN] Refusing to remove checkpoint outside the current '
                        f'workspace: {old_ckpt}'
                    )
                elif os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    self._atomic_torch_save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def _milestone_checkpoint_path(self, step):
        if self.workspace is None:
            raise RuntimeError('Milestone checkpoints require a workspace')
        return os.path.join(
            self.milestone_ckpt_path,
            f'{self.name}_step_{int(step):06d}.pth',
        )

    def save_milestone_checkpoint(self):
        """Preserve the current rolling full checkpoint at configured steps."""
        if self.global_step not in self.milestone_steps:
            return None
        if self.last_checkpoint_path is None:
            raise RuntimeError(
                f'Cannot preserve milestone step {self.global_step}: '
                'no full checkpoint was saved first'
            )

        source_path = os.path.abspath(self.last_checkpoint_path)
        destination_path = os.path.abspath(
            self._milestone_checkpoint_path(self.global_step)
        )
        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f'Milestone source checkpoint does not exist: {source_path}'
            )

        if os.path.isfile(destination_path):
            source_sha256 = self._sha256_file(source_path)
            destination_sha256 = self._sha256_file(destination_path)
            if source_sha256 == destination_sha256:
                self.log(
                    f'[INFO] Milestone checkpoint already exists and matches: '
                    f'{destination_path}'
                )
                return destination_path
            raise FileExistsError(
                f'Refusing to overwrite a different checkpoint at milestone '
                f'step {self.global_step}: {destination_path}'
            )

        self._atomic_copy_file(source_path, destination_path)
        self.log(
            f'[INFO] Preserved full milestone checkpoint at global step '
            f'{self.global_step}: {destination_path}'
        )
        return destination_path
            
    def load_checkpoint(
        self,
        checkpoint=None,
        model_only=False,
        load_mode=None,
    ):
        if load_mode is None:
            load_mode = 'model' if model_only else 'resume'
        elif model_only:
            raise ValueError('Specify either model_only=True or load_mode, not both')
        if load_mode not in {'resume', 'evaluation', 'model'}:
            raise ValueError(
                f'load_mode must be resume, evaluation, or model; '
                f'got {load_mode!r}'
            )

        if checkpoint is None:
            checkpoint_list = glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth')
            if checkpoint_list:
                checkpoint = max(checkpoint_list, key=os.path.getmtime)
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint = os.path.abspath(checkpoint)
        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.last_checkpoint_path = checkpoint
            self.log("[INFO] loaded model.")
            return

        if 'config' in checkpoint_dict:
            self._validate_resume_config(checkpoint_dict['config'])
            self.checkpoint_config = copy.deepcopy(checkpoint_dict['config'])
            self.log("[INFO] checkpoint configuration matches the current run.")
        elif load_mode != 'model':
            self.log("[WARN] Checkpoint has no saved configuration; resume compatibility cannot be verified.")

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        self.ema_checkpoint_loaded = False
        if 'ema' in checkpoint_dict:
            if self.ema is None:
                self.log(
                    '[WARN] Checkpoint contains EMA state, but this Trainer was '
                    'constructed without ema_decay; EMA was not loaded.'
                )
            else:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.ema_checkpoint_loaded = True
                self.log("[INFO] loaded EMA state.")
        elif self.ema is not None:
            self.log("[WARN] Checkpoint has no EMA state.")

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        self.last_checkpoint_path = checkpoint
        if load_mode == 'model':
            return

        required_metadata = ['stats', 'epoch', 'global_step']
        missing_metadata = [
            key for key in required_metadata if key not in checkpoint_dict
        ]
        if missing_metadata:
            raise KeyError(
                f'Checkpoint {checkpoint} is missing metadata required for '
                f'{load_mode} loading: {missing_metadata}'
            )

        self.stats = copy.deepcopy(checkpoint_dict['stats'])
        self.epoch = int(checkpoint_dict['epoch'])
        self.global_step = int(checkpoint_dict['global_step'])
        self.log(
            f"[INFO] load at epoch {self.epoch}, "
            f"global step {self.global_step} ({load_mode})"
        )

        if load_mode == 'evaluation':
            self.log(
                '[INFO] Evaluation-only load skipped optimizer, scheduler, '
                'GradScaler, RNG, and checkpoint-rotation restoration.'
            )
            return

        checkpoint_history = self.stats.get('checkpoints', [])
        managed_history = [
            file_path for file_path in checkpoint_history
            if self._is_managed_checkpoint_path(file_path)
        ]
        if self._is_managed_checkpoint_path(checkpoint):
            loaded_checkpoint = os.path.abspath(checkpoint)
            managed_realpaths = {
                os.path.realpath(file_path) for file_path in managed_history
            }
            if os.path.realpath(loaded_checkpoint) not in managed_realpaths:
                managed_history.append(loaded_checkpoint)
            managed_history = managed_history[-self.max_keep_ckpt:]
        if len(managed_history) != len(checkpoint_history):
            self.log(
                '[INFO] Reset checkpoint rotation history for the current workspace; '
                'external checkpoint paths will never be removed.'
            )
        self.stats['checkpoints'] = managed_history
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except Exception as exc:
                raise RuntimeError("Failed to restore optimizer state.") from exc
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except Exception as exc:
                raise RuntimeError("Failed to restore scheduler state.") from exc
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except Exception as exc:
                raise RuntimeError("Failed to restore AMP scaler state.") from exc

        if 'rng_state' in checkpoint_dict:
            self._restore_rng_state(checkpoint_dict['rng_state'])
            self.log("[INFO] restored Python, NumPy, PyTorch, and CUDA RNG states.")
        else:
            self.log("[WARN] Checkpoint has no RNG state; resumed training will not be bitwise reproducible.")
