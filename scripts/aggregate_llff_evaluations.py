#!/usr/bin/env python3
"""Aggregate complete Raw/EMA LLFF evaluation archives without cherry-picking scenes."""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime


SCENES = [
    'fern',
    'flower',
    'fortress',
    'horns',
    'leaves',
    'orchids',
    'room',
    'trex',
]
VARIANTS = ['raw', 'ema']
METRICS = {
    'psnr': 'max',
    'lpips_alex': 'min',
    'ssim': 'max',
}
WORKSPACE_SUFFIX = os.path.join(
    'few_shot3',
    'test_DiffusioNeRF_NeurTV_Ray_30k_seed0',
)


def atomic_write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f'{path}.tmp.{os.getpid()}'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_write_json(path, value):
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n',
    )


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene(repo_root, scene, step):
    workspace = os.path.join(
        repo_root,
        'test_LLFF',
        f'test_{scene}',
        WORKSPACE_SUFFIX,
    )
    archive = os.path.join(workspace, 'evaluation', f'step_{step:06d}')
    required = [
        os.path.join(archive, 'COMPLETE'),
        os.path.join(archive, 'manifest.json'),
        os.path.join(archive, 'comparison.json'),
    ]
    for variant in VARIANTS:
        required.append(os.path.join(archive, variant, 'metrics.json'))
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        return None, missing

    complete = load_json(os.path.join(archive, 'COMPLETE'))
    manifest = load_json(os.path.join(archive, 'manifest.json'))
    if complete.get('status') != 'complete':
        raise ValueError(f'{scene}: COMPLETE status is not complete')
    if manifest['checkpoint']['global_step'] != step:
        raise ValueError(
            f'{scene}: expected step {step}, got '
            f'{manifest["checkpoint"]["global_step"]}'
        )
    if complete.get('checkpoint_sha256') != manifest['checkpoint']['sha256']:
        raise ValueError(f'{scene}: COMPLETE/checkpoint hash mismatch')
    artifact_hashes = complete.get('artifact_sha256')
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError(f'{scene}: COMPLETE has no artifact hash inventory')
    for relative_path, expected_hash in artifact_hashes.items():
        artifact_path = os.path.abspath(os.path.join(archive, relative_path))
        if os.path.commonpath([archive, artifact_path]) != archive:
            raise ValueError(
                f'{scene}: hash inventory escapes archive: {relative_path}'
            )
        if not os.path.isfile(artifact_path):
            raise ValueError(
                f'{scene}: hash inventory file is missing: {relative_path}'
            )
        if sha256_file(artifact_path) != expected_hash:
            raise ValueError(
                f'{scene}: artifact hash mismatch: {relative_path}'
            )
    if manifest['evaluation']['variants'] != VARIANTS:
        raise ValueError(
            f'{scene}: expected variants {VARIANTS}, got '
            f'{manifest["evaluation"]["variants"]}'
        )

    metrics = {}
    for variant in VARIANTS:
        result = load_json(os.path.join(archive, variant, 'metrics.json'))
        if result['variant'] != variant:
            raise ValueError(f'{scene}: mislabeled {variant} metrics')
        if result['frame_ids'] != manifest['dataset']['frame_ids']:
            raise ValueError(f'{scene}: {variant} frame IDs disagree with manifest')
        metrics[variant] = result

    return {
        'scene': scene,
        'archive': archive,
        'manifest': manifest,
        'metrics': metrics,
    }, []


def build_summary(scene_results, step):
    rows = []
    for result in scene_results:
        for variant in VARIANTS:
            mean = result['metrics'][variant]['mean']
            rows.append({
                'scene': result['scene'],
                'variant': variant,
                'global_step': step,
                'frame_count': result['metrics'][variant]['frame_count'],
                'psnr': mean['psnr'],
                'lpips_alex': mean['lpips_alex'],
                'ssim': mean['ssim'],
                'checkpoint_sha256': result['manifest']['checkpoint']['sha256'],
                'archive': result['archive'],
            })

    macro = {}
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row['variant'] == variant]
        macro[variant] = {
            metric: sum(row[metric] for row in variant_rows) / len(variant_rows)
            for metric in METRICS
        }

    best_by_metric = {}
    for metric, direction in METRICS.items():
        choose = max if direction == 'max' else min
        winner = choose(VARIANTS, key=lambda variant: macro[variant][metric])
        best_by_metric[metric] = {
            'direction': direction,
            'variant': winner,
            'value': macro[winner][metric],
            'all_values': {
                variant: macro[variant][metric] for variant in VARIANTS
            },
        }

    return rows, {
        'schema_version': 1,
        'created_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'global_step': step,
        'scene_count': len(scene_results),
        'scenes': [result['scene'] for result in scene_results],
        'aggregation': (
            'Unweighted macro mean across scene-level metrics. '
            'No per-scene model-variant cherry-picking is applied.'
        ),
        'macro_mean': macro,
        'best_by_metric': best_by_metric,
        'paper_reporting': {
            'preferred': 'Report Raw and EMA as separate rows.',
            'hybrid_if_required': (
                'For each metric, use the globally winning variant shown in '
                'best_by_metric and disclose the source variant.'
            ),
        },
    }


def write_csv(path, rows):
    fieldnames = [
        'scene',
        'variant',
        'global_step',
        'frame_count',
        'psnr',
        'lpips_alex',
        'ssim',
        'checkpoint_sha256',
        'archive',
    ]
    tmp_path = f'{path}.tmp.{os.getpid()}'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        with open(tmp_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_markdown(rows, summary):
    by_scene = {
        (row['scene'], row['variant']): row
        for row in rows
    }
    lines = [
        '# LLFF 3-view NeurTV + Ray evaluation',
        '',
        f'- Fixed checkpoint: {summary["global_step"]} steps',
        f'- Complete scenes: {summary["scene_count"]}',
        '- Average: unweighted macro mean across scenes',
        '',
        '| Scene | Variant | PSNR ↑ | LPIPS-Alex ↓ | SSIM ↑ |',
        '|---|---|---:|---:|---:|',
    ]
    for scene in summary['scenes']:
        for variant in VARIANTS:
            row = by_scene[(scene, variant)]
            lines.append(
                f'| {scene} | {variant} | {row["psnr"]:.6f} | '
                f'{row["lpips_alex"]:.6f} | {row["ssim"]:.6f} |'
            )
    for variant in VARIANTS:
        mean = summary['macro_mean'][variant]
        lines.append(
            f'| **Macro mean** | **{variant}** | {mean["psnr"]:.6f} | '
            f'{mean["lpips_alex"]:.6f} | {mean["ssim"]:.6f} |'
        )
    lines.extend([
        '',
        '## Global best by metric',
        '',
        '| Metric | Source variant | Value |',
        '|---|---|---:|',
    ])
    for metric in METRICS:
        selection = summary['best_by_metric'][metric]
        lines.append(
            f'| {metric} | {selection["variant"]} | '
            f'{selection["value"]:.6f} |'
        )
    lines.append('')
    return '\n'.join(lines)


def parse_args():
    default_repo = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', default=default_repo)
    parser.add_argument('--step', type=int, default=30000)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--output-dir', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    output_dir = args.output_dir or os.path.join(
        repo_root,
        'test_LLFF',
        'aggregates',
        f'llff_3v_neurtv_ray_seed0_step_{args.step:06d}',
    )
    output_dir = os.path.abspath(output_dir)

    scene_results = []
    incomplete = {}
    for scene in SCENES:
        result, missing = load_scene(repo_root, scene, args.step)
        if result is None:
            incomplete[scene] = missing
        else:
            scene_results.append(result)

    if incomplete and not args.allow_partial:
        print('Refusing to aggregate an incomplete eight-scene experiment:', file=sys.stderr)
        for scene, missing in incomplete.items():
            print(f'  {scene}: missing {len(missing)} required files', file=sys.stderr)
        return 2
    if not scene_results:
        print('No complete scene archives found.', file=sys.stderr)
        return 2

    rows, summary = build_summary(scene_results, args.step)
    summary['incomplete_scenes'] = sorted(incomplete)
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, 'summary_long.csv'), rows)
    atomic_write_json(os.path.join(output_dir, 'summary.json'), summary)
    atomic_write_text(
        os.path.join(output_dir, 'summary.md'),
        build_markdown(rows, summary),
    )
    atomic_write_json(
        os.path.join(output_dir, 'COMPLETE'),
        {
            'status': 'complete' if not incomplete else 'partial',
            'global_step': args.step,
            'scenes': summary['scenes'],
            'incomplete_scenes': summary['incomplete_scenes'],
        },
    )
    print(output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
