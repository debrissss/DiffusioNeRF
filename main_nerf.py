import os
import sys
import copy
from functools import partial

import torch
import configargparse
import numpy as np

from nerf.provider import NeRFDataset
# 尝试导入 GUI；如果环境变量 NO_GUI=1 或导入失败则回退到 headless
try:
    if os.environ.get("NO_GUI", "0") == "1":
        raise ImportError("NO_GUI set, skip GUI import")
    from nerf.gui import NeRFGUI
    GUI_AVAILABLE = True
except Exception as _e:
    NeRFGUI = None
    GUI_AVAILABLE = False
    print("Warning: dearpygui / GUI disabled — running headless. Reason:", _e)

from nerf.utils import *
from nerf.adv.awp import AdvWeightPerturb

from loss import huber_loss


if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--num_testval_images', type=int, default=None, help="how many validation/test images to use")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### table options
    parser.add_argument('--write_table', action='store_true')
    parser.add_argument('--implementation_name', type=str, default=None, help="name of the implementation")
    parser.add_argument('--dataset_name', type=str, default=None, help="name of the file where to output the table")

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--stop_at_step', type=int, default=None,
                        help='optional global step at which to stop while keeping --iters as the LR schedule horizon')
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--amp_max_retries', type=int, default=4,
                        help='maximum retries of the same batch after an FP16 gradient overflow')
    parser.add_argument('--detect_anomaly', action='store_true',
                        help='enable expensive autograd anomaly detection for debugging')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=128, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument('--few_shot', type=int, default=0, help="how many images to train (for few-shot setting)")

    ### Depth Smoothness regularization with patch_size=True
    parser.add_argument('--rgb_weighting', action='store_true', help="use color difference b/w gt and prediction as geometric loss weighting factor")
    parser.add_argument('--patch_gamma', type=int, default=1, help="gamma for geometric loss weighting factor")
    parser.add_argument('--depth_reg_lambda', type=float, default=1.0, help="lambda parameter for depth geometric regularization")
    ### Sample Space Annealing
    parser.add_argument('--anneal_nearfar', action='store_true', help="anneal near and far bound during initial training, useful to catch center of the scene in few-shot")
    parser.add_argument('--anneal_nearfar_steps', type=int, default=512, help="steps for near/far annealing")
    parser.add_argument('--anneal_nearfar_perc', type=float, default=0.2, help="percentage for near/far annealing")
    parser.add_argument('--anneal_mid_perc', type=float, default=0.5, help="percentage for near/far mid point")

    ### Frequency Regularization
    parser.add_argument('--fre_nll_sigma', action='store_true',
                        help="use frequency regularization mask after input encoding in sigma network")
    parser.add_argument('--fre_nll_color', action='store_true',
                        help="use frequency regularization mask after input encoding in color network")
    parser.add_argument('--total_iter_end_rate', type=float, default=0.9, help="set total iteration rate for encoding mask regularization")
    parser.add_argument('--start_ptr', type=int, default=1, help="how many feature embeddings we consider at the beginning")
    parser.add_argument('--num_levels', type=int, default=16, help="number of hash encoding levels for the density network")

    ### nll regularization
    parser.add_argument('--use_nll_sigma', action='store_true', help="use nll in sigma network")
    parser.add_argument('--use_nll_color', action='store_true', help="use nll in color network")
    
    # --- NeurTV on density field ---
    parser.add_argument('--neurtv', action='store_true',
                        help='use NeurTV regularization on density field sigma')
    parser.add_argument('--neurtv_lambda', type=float, default=1e-7,
                        help='lambda coefficient of NeurTV')
    parser.add_argument('--neurtv_start_iter', type=int, default=3000,
                        help='start applying NeurTV after this iteration')
    parser.add_argument('--neurtv_end_iter', type=int, default=None,
                        help='stop applying NeurTV after this iteration')
    parser.add_argument('--neurtv_num_samples', type=int, default=4096,
                        help='number of random 3D samples used by NeurTV per training step')
    parser.add_argument('--neurtv_fd_epsilon', type=float, default=0.01,
                        help='central finite-difference step as a fraction of the scene bound')

    ### Virtual ray augmentation
    virtual_ray_group = parser.add_mutually_exclusive_group()
    virtual_ray_group.add_argument('--virtual_ray', dest='virtual_ray', action='store_true',
                                   help='enable virtual ray augmentation')
    virtual_ray_group.add_argument('--no_virtual_ray', dest='virtual_ray', action='store_false',
                                   help='disable virtual ray augmentation')
    parser.set_defaults(virtual_ray=True)
    parser.add_argument('--virtual_ray_start_iter', type=int, default=1000,
                        help='start virtual ray augmentation at this training iteration')
    parser.add_argument('--virtual_ray_k', type=int, default=10,
                        help='number of virtual rays generated for each original ray')
    parser.add_argument('--virtual_ray_jsd_th', type=float, default=0.02,
                        help='Jensen-Shannon divergence threshold for accepting virtual rays')
    parser.add_argument('--virtual_ray_depth_lambda', type=float, default=0.1,
                        help='lambda coefficient of virtual-ray depth consistency')

    ### Diffusion Geometric regularization
    parser.add_argument('--diff_reg', action='store_true', help="use diffusion geometric regulazition additional losses")
    parser.add_argument('--loss_dist', action='store_true', help="use loss_dist")
    parser.add_argument('--dist_lambda', type=float, default=0.001, help="lambda parameter for dist_loss term")
    parser.add_argument('--diff_reg_start_iter', type=int, default=2000, help="start diffusion geometric regularization after certain iter")
    parser.add_argument('--diff_reg_end_rate', type=float , default=1.0, help="stop diffusion geometric regularization after a ceratain iters percentage")
    parser.add_argument('--use_depth', action='store_true', help="use depth for computing the loss_dist")
    parser.add_argument('--diff_num_rays', type=int, default=250, help="how many rays to consider for the loss_dist computation")
    parser.add_argument('--loss_fg', action='store_true', help="use loss_fg")
    parser.add_argument('--fg_lambda', type=float, default=0.01, help="lambda parameter for fg_loss (sum of weight unity)")

    ### Adversarial training
    parser.add_argument("--adv", nargs='*', type=str, default=[],
                        help='turn on adv training. support combination of adv type')
    parser.add_argument("--unadv", action='store_true', default=False,
                        help='turn on unadv training')
    parser.add_argument("--adv_type", type=str, default='pgd', choices=['random', 'pgd'],
                        help='type of adv noises: random or pgd')
    parser.add_argument("--adv_lambda", type=float, default=0.5,
                        help='lambda coefficient of adv loss')
    parser.add_argument("--pgd_alpha", nargs='*', type=float, default=[1e-5],
                        help='alpha for pgd noise searching')
    parser.add_argument("--pgd_iters", nargs='*', type=int, default=[1],
                        help='iteration number for pgd noise searching')
    parser.add_argument("--pgd_eps", nargs='*', type=float, default=[1e-5],
                        help='maximal perturbation stength in ratio or magnitude')
    parser.add_argument("--pgd_norm", nargs='*', type=str, default=['l_inf'],
                        help='boundary in norm of pgd noise searching')
    parser.add_argument("--awp_warmup", type=int, default=0,
                        help='warm up iterations for awp')
    parser.add_argument("--awp_gamma", type=float, default=0.01,
                        help='gamma for awp training')
    parser.add_argument("--awp_lrate", type=float, default=5e-4,
                        help='lrate for proxy optimizer in awp training')

    ### Ray Entropy Minimization Loss
    # entropy
    parser.add_argument("--N_entropy", type=int, default=100,
                        help='number of entropy ray')
    # entropy type
    parser.add_argument("--entropy", action='store_true',
                        help='using entropy ray loss')
    parser.add_argument("--entropy_log_scaling", action='store_true',
                        help='using log scaling for entropy loss')
    parser.add_argument("--entropy_ignore_smoothing", action='store_true',
                        help='ignoring entropy for ray for smoothing')
    parser.add_argument("--entropy_end_iter", type=int, default=None,
                        help='end iteratio of entropy')
    parser.add_argument("--entropy_type", type=str, default='log2', choices=['log2', '1-p'],
                        help='choosing type of entropy')
    parser.add_argument("--entropy_acc_threshold", type=float, default=0.1,
                        help='threshold for acc masking')
    parser.add_argument("--computing_entropy_all", action='store_true',
                        help='computing entropy for both seen and unseen ')
    # lambda
    parser.add_argument("--entropy_ray_lambda", type=float, default=1,
                        help='entropy lambda for ray entropy loss')
    parser.add_argument("--entropy_ray_zvals_lambda", type=float, default=1,
                        help='entropy lambda for ray zvals entropy loss')


    ### Infomation Gain Reduction Loss (KL-Divergence loss)
    parser.add_argument("--smoothing", action='store_true',
                        help='using information gain reduction loss')
    # choosing between rotating camera pose & near pixel
    parser.add_argument("--smooth_sampling_method", type=str, default='near_pose',
                        help='how to sample the near rays, near_pose: modifying camera pose, near_pixel: sample near pixel',
                        choices=['near_pose', 'near_pixel'])
    # 1) sampling by rotating camera pose
    parser.add_argument("--near_c2w_type", type=str, default='rot_from_origin',
                        help='random augmentation method')
    parser.add_argument("--near_c2w_rot", type=float, default=5,
                        help='random augmentation rotate: degree')
    parser.add_argument("--near_c2w_trans", type=float, default=0.1,
                        help='random augmentation translation')
    # 2) sampling with near pixel
    parser.add_argument("--smooth_pixel_range", type=int, default=1,
                        help='the maximum distance between the near ray & the original ray (pixel dimension)')
    # optimizing
    parser.add_argument("--smoothing_lambda", type=float, default=0.001,
                        help='lambda for smoothing loss')
    parser.add_argument("--smoothing_activation", type=str, default='norm',
                        help='how to make alpha to the distribution')
    parser.add_argument("--smoothing_step_size", type=int, default=5000,
                        help='reducing smoothing every')
    parser.add_argument("--smoothing_rate", type=float, default=0.5,
                        help='reducing smoothing rate')
    parser.add_argument("--smoothing_end_iter", type=int, default=None,
                        help='when smoothing will be end')

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--aabb_box', type=float, nargs=6, default=None, help="aabb only used for generating points")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--downscale', type=int, default=8, help="Set downscale for resolution of images")#缩小 1-8
    parser.add_argument('--split_file', type=str, default=None,
                        help='JSON file containing explicit train_ids/val_ids/test_ids in transforms.json frame order')

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")


    opt = parser.parse_args()

    if opt.neurtv_num_samples <= 0:
        parser.error('--neurtv_num_samples must be positive')
    if not 0 < opt.neurtv_fd_epsilon < 1:
        parser.error('--neurtv_fd_epsilon must be in the open interval (0, 1)')
    if opt.iters <= 0:
        parser.error('--iters must be positive')
    if opt.amp_max_retries < 0:
        parser.error('--amp_max_retries must be non-negative')
    if opt.stop_at_step is not None and not 0 < opt.stop_at_step <= opt.iters:
        parser.error('--stop_at_step must be in the range [1, --iters]')
    if opt.neurtv_start_iter < 0:
        parser.error('--neurtv_start_iter must be non-negative')
    if opt.neurtv_end_iter is not None and opt.neurtv_end_iter < opt.neurtv_start_iter:
        parser.error('--neurtv_end_iter must be greater than or equal to --neurtv_start_iter')
    if opt.virtual_ray_start_iter < 0:
        parser.error('--virtual_ray_start_iter must be non-negative')
    if opt.virtual_ray_k <= 0:
        parser.error('--virtual_ray_k must be positive')
    if opt.virtual_ray_jsd_th < 0:
        parser.error('--virtual_ray_jsd_th must be non-negative')
    if opt.virtual_ray_depth_lambda < 0:
        parser.error('--virtual_ray_depth_lambda must be non-negative')

    torch.autograd.set_detect_anomaly(opt.detect_anomaly)

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    print(opt)
    
    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        device=device,
        num_levels=opt.num_levels,
        nll_color=opt.use_nll_color,
        nll_sigma=opt.use_nll_sigma,
        fre_nll_color=opt.fre_nll_color,
        fre_nll_sigma=opt.fre_nll_sigma,
        aabb_box=opt.aabb_box,
    )
    
    #print(model)
    awp_adversary = None
    if 'awp' in opt.adv:
        proxy = copy.deepcopy(model)
        proxy_optim = torch.optim.Adam(params=proxy.parameters(), lr=opt.awp_lrate, betas=(0.9, 0.999))
        awp_adversary = AdvWeightPerturb(model, proxy, proxy_optim, opt.awp_gamma)

    criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?


    if opt.test:
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        # 如果请求 GUI 且 GUI 可用，就打开 GUI 渲染
        if opt.gui and GUI_AVAILABLE:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        else:
            # headless: 直接执行测试逻辑
            print("Headless mode: running test/eval.")

            test_loader = NeRFDataset(opt, device=device, type='test', downscale=opt.downscale).dataloader()

            try:
                trainer.evaluate(test_loader, type='test')
            except Exception as _e:
                print("Warning: trainer.evaluate failed or not supported for this dataset. Reason:", _e)

            # 运行 test（并保存视频）
            try:
                trainer.test(test_loader, write_video=False)
            except Exception as _e:
                print("Warning: trainer.test failed. Reason:", _e)

            # 保存网格
            try:
                trainer.save_mesh(resolution=256, threshold=10)
            except Exception as _e:
                print("Warning: trainer.save_mesh failed or is not supported. Reason:", _e)
    
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type='train', downscale=opt.downscale).dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50, awp_adversary=awp_adversary)

        # GUI: 若请求 GUI 且可用则打开；否则直接训练
        if opt.gui and GUI_AVAILABLE:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        else:
            print("Headless mode: running training.")

            valid_loader = None
            try:
                valid_loader = NeRFDataset(opt, device=device, type='val', downscale=opt.downscale).dataloader()
            except Exception as _e:
                print("Warning: failed to create validation loader. Reason:", _e)

            try:
                steps_per_epoch = len(train_loader)
                target_step = opt.stop_at_step if opt.stop_at_step is not None else opt.iters
                if target_step % steps_per_epoch != 0:
                    raise ValueError(
                        f'target global step {target_step} is not divisible by '
                        f'{steps_per_epoch} steps/epoch; choose an epoch-aligned '
                        'target so the checkpoint is exactly resumable'
                    )
                if trainer.global_step % steps_per_epoch != 0:
                    raise ValueError(
                        f'checkpoint global_step={trainer.global_step} is not aligned '
                        f'to {steps_per_epoch} steps/epoch'
                    )
                if trainer.epoch != trainer.global_step // steps_per_epoch:
                    raise ValueError(
                        f'checkpoint epoch/global_step mismatch: epoch={trainer.epoch}, '
                        f'global_step={trainer.global_step}, steps_per_epoch={steps_per_epoch}'
                    )
                if trainer.global_step > target_step:
                    raise ValueError(
                        f'checkpoint global_step={trainer.global_step} is already beyond '
                        f'the requested target global_step={target_step}'
                    )
                max_epoch = target_step // steps_per_epoch
                print(
                    f'Training target: global_step={target_step}; '
                    f'LR schedule horizon: {opt.iters}; '
                    f'resume from global_step={trainer.global_step}.'
                )
            except Exception as _e:
                print("Warning: failed to compute max_epoch from train_loader length. Reason:", _e)
                raise

            try:
                if max_epoch is None:
                    trainer.train(train_loader, valid_loader)
                else:
                    trainer.train(train_loader, valid_loader, max_epoch)
            except Exception as _e:
                print("Error: trainer.train failed. Reason:", _e)
                raise

            # 训练结束后再测试并保存（尝试性运行并捕获异常）
            try:
                test_loader = NeRFDataset(opt, device=device, type='test', downscale=opt.downscale).dataloader()
                try:
                    trainer.evaluate(test_loader, type='test')
                except Exception as _e:
                    print("Warning: trainer.evaluate failed or not supported. Reason:", _e)
                try:
                    trainer.test(test_loader, write_video=False)
                except Exception as _e:
                    print("Warning: trainer.test failed. Reason:", _e)
                try:
                    trainer.save_mesh(resolution=256, threshold=10)
                except Exception as _e:
                    print("Warning: trainer.save_mesh failed. Reason:", _e)
            except Exception as _e:
                print("Warning: post-train evaluation/test/save failed. Reason:", _e)
