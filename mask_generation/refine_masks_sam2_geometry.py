#!/usr/bin/env python3
"""Refine DTU geometry masks with SAM 2.1 and shadow-aware prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def pngs(path: Path) -> list[Path]:
    return sorted(p for p in path.glob("*.png") if not p.name.startswith("._"))


def bbox(mask: np.ndarray, padding: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Empty geometry mask")
    h, w = mask.shape
    return np.array(
        [
            max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(w - 1, int(xs.max()) + padding),
            min(h - 1, int(ys.max()) + padding),
        ],
        dtype=np.float32,
    )


def angular_sample(
    candidates: np.ndarray,
    score: np.ndarray,
    center_xy: tuple[float, float],
    count: int,
) -> np.ndarray:
    """Pick high-score pixels while spreading them around the object."""
    ys, xs = np.nonzero(candidates)
    if not len(xs) or count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    cx, cy = center_xy
    angles = (np.arctan2(ys - cy, xs - cx) + 2 * np.pi) % (2 * np.pi)
    bins = np.floor(angles * count / (2 * np.pi)).astype(int).clip(0, count - 1)
    points: list[list[float]] = []
    used: set[int] = set()
    for bin_id in range(count):
        ids = np.flatnonzero(bins == bin_id)
        if not len(ids):
            continue
        local = ids[np.argmax(score[ys[ids], xs[ids]])]
        used.add(int(local))
        points.append([float(xs[local]), float(ys[local])])
    if len(points) < count:
        order = np.argsort(score[ys, xs])[::-1]
        min_dist2 = 20.0**2
        for idx in order:
            if int(idx) in used:
                continue
            x, y = float(xs[idx]), float(ys[idx])
            if all((x - px) ** 2 + (y - py) ** 2 >= min_dist2 for px, py in points):
                points.append([x, y])
                if len(points) == count:
                    break
    return np.asarray(points, dtype=np.float32)


def make_prompts(
    rgb: np.ndarray,
    geometry: np.ndarray,
    box_padding: int,
    positive_points: int,
    negative_points: int,
    shadow_points: int,
    negative_min_distance: float,
    negative_max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    ys, xs = np.nonzero(geometry)
    center = (float(np.median(xs)), float(np.median(ys)))
    inside_distance = cv2.distanceTransform(geometry.astype(np.uint8), cv2.DIST_L2, 5)
    positive_candidates = inside_distance >= max(8.0, np.percentile(inside_distance[geometry], 45))
    positives = angular_sample(positive_candidates, inside_distance, center, positive_points)

    outside = ~geometry
    outside_distance = cv2.distanceTransform(outside.astype(np.uint8), cv2.DIST_L2, 5)
    ring = outside & (outside_distance >= negative_min_distance) & (
        outside_distance <= negative_max_distance
    )
    # Standard negative prompts surround the silhouette. A closeness score prevents
    # them from drifting to unrelated image corners.
    target_distance = (negative_min_distance + negative_max_distance) / 2
    surround_score = -np.abs(outside_distance - target_distance)
    negatives = angular_sample(ring, surround_score, center, negative_points)

    # Shadows in DTU are typically low-luminance regions on the table immediately
    # outside the geometric silhouette. Sample the darkest candidates separately.
    lab_l = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    ring_values = lab_l[ring]
    dark_threshold = float(np.percentile(ring_values, 35)) if len(ring_values) else 0.0
    dark_ring = ring & (lab_l <= dark_threshold)
    darkness_score = 255.0 - lab_l - 0.15 * outside_distance
    dark_negatives = angular_sample(dark_ring, darkness_score, center, shadow_points)

    coords = np.concatenate([positives, negatives, dark_negatives], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(positives), dtype=np.int32),
            np.zeros(len(negatives) + len(dark_negatives), dtype=np.int32),
        ]
    )
    info = {
        "positive_points": int(len(positives)),
        "negative_points": int(len(negatives)),
        "shadow_negative_points": int(len(dark_negatives)),
        "shadow_l_threshold": dark_threshold,
    }
    return coords, labels, bbox(geometry, box_padding), info


def component_overlapping_geometry(mask: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    best_label = max(range(1, count), key=lambda label: np.logical_and(labels == label, geometry).sum())
    selected = (labels == best_label).astype(np.uint8)
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(selected)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, -1)
    return filled.astype(bool)


def select_candidate(
    masks: np.ndarray,
    predicted_ious: np.ndarray,
    geometry: np.ndarray,
    rgb: np.ndarray,
    shadow_l_threshold: float,
) -> tuple[int, list[dict]]:
    geom_area = max(1, int(geometry.sum()))
    eroded = cv2.erode(geometry.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    lab_l = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    shadow_region = (~geometry) & (lab_l <= shadow_l_threshold)
    rows = []
    for index, raw_mask in enumerate(masks):
        mask = raw_mask > 0
        intersection = int(np.logical_and(mask, geometry).sum())
        union = int(np.logical_or(mask, geometry).sum())
        geom_recall = intersection / geom_area
        geom_iou = intersection / max(1, union)
        sure_recall = np.logical_and(mask, eroded).sum() / max(1, eroded.sum())
        expansion = mask.sum() / geom_area
        shadow_leak = np.logical_and(mask, shadow_region).sum() / geom_area
        area_penalty = abs(float(np.log(max(1e-6, expansion))))
        score = (
            0.50 * float(predicted_ious[index])
            + 0.85 * geom_recall
            + 0.35 * geom_iou
            + 0.35 * sure_recall
            - 0.45 * shadow_leak
            - 0.20 * area_penalty
        )
        rows.append(
            {
                "candidate": index,
                "sam_iou": float(predicted_ious[index]),
                "geometry_recall": float(geom_recall),
                "geometry_iou": float(geom_iou),
                "sure_foreground_recall": float(sure_recall),
                "area_ratio_to_geometry": float(expansion),
                "shadow_leak_ratio": float(shadow_leak),
                "selection_score": float(score),
            }
        )
    return int(np.argmax([row["selection_score"] for row in rows])), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--geometry-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--box-padding", type=int, default=180)
    parser.add_argument("--positive-points", type=int, default=10)
    parser.add_argument("--negative-points", type=int, default=10)
    parser.add_argument("--shadow-points", type=int, default=8)
    parser.add_argument("--refinement-points", type=int, default=8)
    parser.add_argument("--negative-min-distance", type=float, default=40)
    parser.add_argument("--negative-max-distance", type=float, default=220)
    parser.add_argument("--dilation", type=int, default=7, help="Final odd-sized ellipse kernel; 1 disables")
    parser.add_argument(
        "--preserve-geometry",
        action="store_true",
        help="Union the final SAM mask with the input geometry mask",
    )
    parser.add_argument(
        "--preserve-geometry-below-area-ratio",
        type=float,
        default=0.0,
        help="Use geometry fallback only when SAM/geometry area ratio is below this value",
    )
    parser.add_argument("--frames", default="", help="Comma-separated frame indices; empty means all")
    args = parser.parse_args()
    if args.dilation < 1 or args.dilation % 2 == 0:
        raise ValueError("--dilation must be a positive odd integer")

    images, masks = pngs(args.image_dir), pngs(args.geometry_mask_dir)
    if not images or len(images) != len(masks):
        raise ValueError(f"Count mismatch: images={len(images)}, masks={len(masks)}")
    selected_frames = (
        {int(value) for value in args.frames.split(",") if value.strip()}
        if args.frames
        else set(range(len(images)))
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.output_dir / "prompts"
    diagnostics_dir.mkdir(exist_ok=True)

    model = build_sam2(args.config, str(args.checkpoint), device="cuda")
    predictor = SAM2ImagePredictor(model, max_hole_area=64, max_sprinkle_area=64)
    stats = []
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for frame, (image_path, geometry_path) in enumerate(zip(images, masks)):
        if frame not in selected_frames:
            continue
        rgb = np.asarray(Image.open(image_path).convert("RGB")).copy()
        geometry = np.asarray(Image.open(geometry_path).convert("L")) > 127
        coords, labels, box_prompt, prompt_info = make_prompts(
            rgb,
            geometry,
            args.box_padding,
            args.positive_points,
            args.negative_points,
            args.shadow_points,
            args.negative_min_distance,
            args.negative_max_distance,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
            predictor.set_image(rgb)
            candidates, predicted_ious, low_res_logits = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                box=box_prompt,
                multimask_output=True,
            )
        chosen, candidate_stats = select_candidate(
            candidates, predicted_ious, geometry, rgb, prompt_info["shadow_l_threshold"]
        )
        first_result = candidates[chosen] > 0
        missed_geometry = geometry & ~first_result
        inside_distance = cv2.distanceTransform(geometry.astype(np.uint8), cv2.DIST_L2, 5)
        ys, xs = np.nonzero(geometry)
        center = (float(np.median(xs)), float(np.median(ys)))
        correction_points = angular_sample(
            missed_geometry, inside_distance, center, args.refinement_points
        )
        if len(correction_points):
            refined_coords = np.concatenate([coords, correction_points], axis=0)
            refined_labels = np.concatenate(
                [labels, np.ones(len(correction_points), dtype=np.int32)]
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
                refined_masks, refined_ious, _ = predictor.predict(
                    point_coords=refined_coords,
                    point_labels=refined_labels,
                    box=box_prompt,
                    mask_input=low_res_logits[chosen][None, ...],
                    multimask_output=False,
                )
            result = refined_masks[0] > 0
            refinement_iou = float(refined_ious[0])
        else:
            refined_coords, refined_labels = coords, labels
            result = first_result
            refinement_iou = float(predicted_ious[chosen])
        result = component_overlapping_geometry(result, geometry)
        if args.dilation > 1:
            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (args.dilation, args.dilation)
            )
            result = cv2.dilate(result.astype(np.uint8), dilation_kernel) > 0
        area_ratio_before_fallback = result.sum() / max(1, geometry.sum())
        geometry_fallback = args.preserve_geometry or (
            args.preserve_geometry_below_area_ratio > 0
            and area_ratio_before_fallback < args.preserve_geometry_below_area_ratio
        )
        if geometry_fallback:
            result |= geometry
        cv2.imwrite(str(args.output_dir / f"{frame:03d}.png"), result.astype(np.uint8) * 255)

        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        x0, y0, x1, y1 = box_prompt.astype(int)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 3)
        for (x, y), label in zip(refined_coords.astype(int), refined_labels):
            cv2.circle(overlay, (x, y), 7, (0, 255, 0) if label else (0, 0, 255), -1)
        cv2.imwrite(str(diagnostics_dir / f"{frame:03d}.jpg"), overlay)
        row = {
            "frame": frame,
            "chosen_candidate": chosen,
            "geometry_pixels": int(geometry.sum()),
            "result_pixels": int(result.sum()),
            "refinement_points": int(len(correction_points)),
            "refinement_sam_iou": refinement_iou,
            "area_ratio_before_geometry_fallback": float(area_ratio_before_fallback),
            "geometry_fallback": bool(geometry_fallback),
            **prompt_info,
            "candidates": candidate_stats,
        }
        stats.append(row)
        print(
            f"[{len(stats):02d}/{len(selected_frames):02d}] frame {frame:03d}: "
            f"candidate={chosen}, area={result.sum() / max(1, geometry.sum()):.4f}",
            flush=True,
        )

    metadata = {
        "method": "SAM 2.1 geometry prompts with shadow-aware negative points",
        "checkpoint": str(args.checkpoint.resolve()),
        "config": args.config,
        "parameters": {
            "box_padding": args.box_padding,
            "positive_points": args.positive_points,
            "negative_points": args.negative_points,
            "shadow_points": args.shadow_points,
            "refinement_points": args.refinement_points,
            "negative_min_distance": args.negative_min_distance,
            "negative_max_distance": args.negative_max_distance,
            "dilation": args.dilation,
            "preserve_geometry": args.preserve_geometry,
            "preserve_geometry_below_area_ratio": args.preserve_geometry_below_area_ratio,
        },
        "frames": stats,
    }
    (args.output_dir / "sam2_stats.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
