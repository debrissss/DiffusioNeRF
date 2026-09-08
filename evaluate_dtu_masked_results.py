#!/usr/bin/env python3
"""Evaluate archived DTU renders with DNGaussian's DTU protocol.

Protocol:
  * PSNR: foreground pixels inside the provided object mask only.
  * SSIM/LPIPS: composite prediction and GT onto the same white background.
  * SSIM: report both the 3DGS implementation and skimage implementation.
  * LPIPS: VGG on inputs in [0, 1], matching DNGaussian's metrics_dtu.py.

This script does not render or load checkpoints. It only consumes archived
RGB/GT PNGs produced by the formal evaluator.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
import lpips


TEST_FRAME_IDS = [1, 2, 9, 10, 11, 12, 14, 15, 23, 24, 26, 27, 29, 30,
                  31, 32, 33, 34, 35, 41, 42, 43, 45, 46, 47]


def read_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


def read_mask(path, shape):
    mask = np.asarray(Image.open(path).convert('L')) > 0
    if mask.shape != shape[:2]:
        raise ValueError(f'Mask/image size mismatch: {path} {mask.shape} vs {shape[:2]}')
    if not mask.any():
        raise ValueError(f'Empty object mask: {path}')
    return mask


def tensor_image(image, device):
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim_3dgs(pred, gt, window_size=11):
    channels = pred.shape[1]
    window = gaussian_window(
        window_size, 1.5, channels, pred.device, pred.dtype,
    )
    padding = window_size // 2
    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_gt = F.conv2d(gt, window, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.pow(2)
    mu_gt_sq = mu_gt.pow(2)
    mu_cross = mu_pred * mu_gt
    sigma_pred = (
        F.conv2d(pred * pred, window, padding=padding, groups=channels)
        - mu_pred_sq
    )
    sigma_gt = (
        F.conv2d(gt * gt, window, padding=padding, groups=channels)
        - mu_gt_sq
    )
    sigma_cross = (
        F.conv2d(pred * gt, window, padding=padding, groups=channels)
        - mu_cross
    )
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = (
        (2 * mu_cross + c1) * (2 * sigma_cross + c2)
        / ((mu_pred_sq + mu_gt_sq + c1) * (sigma_pred + sigma_gt + c2))
    )
    return score.mean()


def evaluate_variant(variant_dir, mask_dir, device, lpips_model, frame_ids):
    frame_results = []
    psnr_values, ssim_values, ssim_sk_values, lpips_values = [], [], [], []

    for frame_id in frame_ids:
        stem = f'frame_{frame_id:04d}'
        pred_path = variant_dir / 'frames' / f'{stem}_rgb.png'
        gt_path = variant_dir / 'frames' / f'{stem}_gt.png'
        mask_path = mask_dir / f'{frame_id:06d}.png'
        pred = read_rgb(pred_path)
        gt = read_rgb(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f'RGB/GT size mismatch: {pred_path} vs {gt_path}')
        mask = read_mask(mask_path, pred.shape)

        diff = pred[mask] - gt[mask]
        mse = float(np.mean(diff * diff))
        psnr = float(-10.0 * math.log10(max(mse, 1e-12)))

        mask_rgb = mask[..., None].astype(np.float32)
        pred_white = pred * mask_rgb + (1.0 - mask_rgb)
        gt_white = gt * mask_rgb + (1.0 - mask_rgb)
        pred_tensor = tensor_image(pred_white, device)
        gt_tensor = tensor_image(gt_white, device)
        with torch.inference_mode():
            ssim = float(ssim_3dgs(pred_tensor, gt_tensor).item())
            lpips_value = float(lpips_model(
                gt_tensor, pred_tensor, normalize=True,
            ).item())
        ssim_sk = float(structural_similarity(
            pred_white, gt_white, channel_axis=2, data_range=1.0,
        ))

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        ssim_sk_values.append(ssim_sk)
        lpips_values.append(lpips_value)
        frame_results.append({
            'frame_id': frame_id,
            'mask_pixels': int(mask.sum()),
            'mask_ratio': float(mask.mean()),
            'bbox_xyxy': [
                int(np.where(mask)[1].min()), int(np.where(mask)[0].min()),
                int(np.where(mask)[1].max()), int(np.where(mask)[0].max()),
            ],
            'metrics': {
                'psnr': psnr,
                'ssim': ssim,
                'ssim_sk': ssim_sk,
                'lpips_vgg': lpips_value,
            },
        })

    result = {
        'schema_version': 1,
        'variant': variant_dir.name,
        'frame_count': len(frame_results),
        'frame_ids': frame_ids,
        'mean': {
            'psnr': float(np.mean(psnr_values)),
            'ssim': float(np.mean(ssim_values)),
            'ssim_sk': float(np.mean(ssim_sk_values)),
            'lpips_vgg': float(np.mean(lpips_values)),
        },
        'per_frame': frame_results,
        'metric_protocol': {
            'psnr': 'foreground mask pixels only; per-frame mean then scene mean',
            'ssim': '3DGS SSIM on full images composited onto white background',
            'ssim_sk': (
                'skimage structural_similarity on full images composited onto '
                'white background'
            ),
            'lpips_vgg': (
                'full images composited onto white background; LPIPS VGG with '
                'inputs supplied in [0, 1] (richzhang normalize=True equivalent)'
            ),
            'mask_source': 'DTU provided object mask',
        },
    }
    output_path = variant_dir / 'masked_metrics_dngaussian.json'
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('test_DTU'))
    parser.add_argument('--dataset-root', type=Path, default=Path('data/DTU_standard'))
    parser.add_argument('--summary', type=Path,
                        default=Path(
                            'dtu_3view_dngaussian_protocol_all_checkpoints_summary.json'
                        ))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--archive', type=Path, action='append', default=[],
        help='evaluate this checkpoint archive directly; may be repeated',
    )
    parser.add_argument(
        '--frame-ids', type=int, nargs='+', default=TEST_FRAME_IDS,
        help='zero-based DTU frame IDs to evaluate',
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    lpips_model = lpips.LPIPS(net='vgg').eval().to(device)

    archives = args.archive or sorted(args.root.glob(
        'test_scan*/few_shot3/*/evaluation/all_checkpoints/*'
    ))
    rows = []
    for archive in archives:
        if not (archive / 'COMPLETE').is_file():
            continue
        with (archive / 'manifest.json').open(encoding='utf-8') as f:
            archive_manifest = json.load(f)
        scene = next(p.name[len('test_'):]
                     for p in archive.parents if p.name.startswith('test_scan'))
        mask_dir = args.dataset_root / scene / 'masks'
        for variant in ('raw', 'ema'):
            variant_dir = archive / variant
            if not (variant_dir / 'metrics.json').is_file():
                continue
            result = evaluate_variant(
                variant_dir, mask_dir, device, lpips_model, args.frame_ids,
            )
            rows.append({
                'scene': scene,
                'checkpoint': archive.name,
                'archive': str(archive),
                'split_file': archive_manifest['dataset'].get('split_file'),
                'variant': variant,
                'mean': result['mean'],
                'frame_count': result['frame_count'],
                'masked_metrics': str(
                    variant_dir / 'masked_metrics_dngaussian.json'
                ),
            })
            print(
                f'{scene} {archive.name} {variant}: '
                f'PSNR={result["mean"]["psnr"]:.4f} '
                f'SSIM={result["mean"]["ssim"]:.4f} '
                f'SSIM_SK={result["mean"]["ssim_sk"]:.4f} '
                f'LPIPS_VGG={result["mean"]["lpips_vgg"]:.4f}',
                flush=True,
            )

    summary = {
        'schema_version': 1,
        'protocol': 'DNGaussian official white-background DTU metrics',
        'metric_protocol': {
            'psnr': 'foreground mask pixels only',
            'ssim': '3DGS SSIM on white-background composited full images',
            'ssim_sk': 'skimage SSIM on white-background composited full images',
            'lpips_vgg': 'LPIPS VGG on white-background composited full images',
        },
        'result_count': len(rows),
        'results': rows,
    }
    with args.summary.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    print(f'Wrote {len(rows)} masked results to {args.summary}')


if __name__ == '__main__':
    main()
