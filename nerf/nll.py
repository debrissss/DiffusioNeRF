# utils/mixnerf.py
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

class MixHeads(nn.Module):
    """
    给定 backbone 的 per-sample 特征 x_view -> 预测:
      - mu_c: 颜色均值, shape [..., M, 3] 或 [..., M, 3] 对齐每个 sample
      - beta: 颜色尺度 beta (per-sample per-channel), shape [..., M, 3]
      - mu_depth: 每个 sample 的深度估计（非负标量）, shape [..., M]
    用法：把 model 的颜色/feature 输出改为返回这些值（把这些 head 拼上）。
    注意：该模块不负责 density / sigma（仍由原 model 输出）。
    """
    def __init__(self, feat_dim, num_mudepth_channels=3, num_rgb_channels=3, num_gamma_channels=3):
        super().__init__()
        # heads
        self.rgb_head = nn.Linear(feat_dim, num_rgb_channels)
        self.beta_head = nn.Linear(feat_dim, num_gamma_channels)  # raw -> softplus
        self.mudepth_head = nn.Linear(feat_dim, num_mudepth_channels)  # vector -> norm
        nn.init.xavier_uniform_(self.rgb_head.weight)
        nn.init.xavier_uniform_(self.beta_head.weight)
        nn.init.xavier_uniform_(self.mudepth_head.weight)

    def forward(self, x):
        """
        x: [..., M, feat_dim] 或 [..., feat_dim] （注意对齐）
        返回:
          mu_c: [..., M, 3] (in [0,1])
          beta: [..., M, 3] (>0)
          mu_depth: [..., M]  (>=0)
        """
        # 如果 x 是 [..., M, feat_dim]，直接线性应用（按最后一维）
        mu_c = torch.sigmoid(self.rgb_head(x))  # [.., M, 3]
        beta_raw = self.beta_head(x)            # [.., M, 3]
        beta = F.softplus(beta_raw) + 1e-6      # 保证 >0
        mud_vec = self.mudepth_head(x)          # [.., M, D]
        mu_depth = torch.norm(mud_vec, dim=-1)  # L2 -> [..., M]
        return mu_c, beta, mu_depth


def normalize_pi(weights, dim=-1):
    """把 alpha weights 归一化为 pi。weights: [..., M]"""
    denom = torch.sum(weights, dim=dim, keepdim=True)
    return weights / (denom + EPS)


def log_laplace_pdf_per_comp(gt_c, mu_c, beta):
    """
    gt_c: [B, 3] 或 [B, 1, 3] （per-ray GT color）
    mu_c: [B, M, 3]
    beta:  [B, M, 3]
    返回: log_pdf_per_component: [B, M]
    """
    # 广播使形状对齐
    # diff: [B, M, 3]
    diff = torch.abs(gt_c.unsqueeze(1) - mu_c)
    log_pdf_ch = -torch.log(2.0 * beta + EPS) - diff / (beta + EPS)
    log_pdf = torch.sum(log_pdf_ch, dim=-1)  # sum over RGB -> [B, M]
    return log_pdf


def mixture_nll_color(gt_c, mu_c, beta, pi, reduce='mean'):
    """
    计算 mixture Laplace NLL for colors.
    gt_c: [B, 3]
    mu_c: [B, M, 3]
    beta: [B, M, 3]
    pi:   [B, M]  (normalized)
    返回标量 loss（平均）
    """
    log_pdf = log_laplace_pdf_per_comp(gt_c, mu_c, beta)  # [B, M]
    log_pi = torch.log(pi + EPS)
    joint = log_pi + log_pdf  # [B, M]
    # log-sum-exp across components
    log_prob = torch.logsumexp(joint, dim=1)  # [B]
    nll = -log_prob
    return nll.mean() if reduce == 'mean' else nll  # 返回标量或 per-ray


def mixture_nll_depth(gt_d, mu_d, beta_scalar_or_3ch, pi, reduce='mean'):
    """
    类似 color 的 depth NLL； beta 是三通道共享也可以使用。
    gt_d: [B] or [B,1]
    mu_d: [B, M]
    beta_scalar_or_3ch: 是 [B, M, 3]，会取 mean 做标量，传入 [B, M] 标量
    pi: [B, M]
    """
    if beta_scalar_or_3ch.dim() == 3:
        # 把 per-channel beta 合并成标量 beta_d
        beta = torch.mean(beta_scalar_or_3ch, dim=-1)  # [B, M]
    else:
        beta = beta_scalar_or_3ch  # [B,M]
    # build per-component log pdf for laplace on scalar depth:
    diff = torch.abs(gt_d.unsqueeze(1) - mu_d)  # [B, M]
    log_pdf = -torch.log(2.0 * beta + EPS) - diff / (beta + EPS)  # [B, M]
    log_pi = torch.log(pi + EPS)
    joint = log_pi + log_pdf
    log_prob = torch.logsumexp(joint, dim=1)  # [B]
    nll = -log_prob
    return nll.mean() if reduce == 'mean' else nll
