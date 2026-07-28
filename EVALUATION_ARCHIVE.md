# LLFF Raw/EMA formal evaluation

The LLFF 3-view NeurTV + virtual-ray runner treats the fixed-step checkpoint as
the source of truth. It evaluates both instantaneous (`raw`) and exponential
moving average (`ema`) parameters and never uses `checkpoints/ngp.pth` for
formal reporting.

## Commands

Train missing scenes, resume incomplete scenes, and archive each completed
scene:

```bash
CKPT_MODE=latest ./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh
```

Evaluate an existing completed scene without restoring optimizer, scheduler,
GradScaler, or RNG state:

```bash
EVAL_ONLY=1 CKPT_MODE=latest \
  ./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh fern
```

Preserve the current archive and regenerate it:

```bash
EVAL_ONLY=1 EVAL_OVERWRITE=1 CKPT_MODE=latest \
  ./run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh fern
```

Create a diagnostic summary from the scenes that are already complete:

```bash
./scripts/aggregate_llff_evaluations.py --allow-partial
```

Create the formal eight-scene summary:

```bash
./scripts/aggregate_llff_evaluations.py
```

The formal aggregator exits with status 2 while any standard LLFF scene is
missing or lacks a complete Raw/EMA archive.

## Archive contract

Each scene is published under:

```text
evaluation/step_030000/
├── COMPLETE
├── manifest.json
├── comparison.json
├── eval.log
├── split_snapshot.json
├── raw/
│   ├── metrics.json
│   └── frames/
└── ema/
    ├── metrics.json
    └── frames/
```

`COMPLETE` is written last and contains a SHA256 inventory for every other
artifact. A matching archive is reused only after verifying:

- checkpoint SHA256 and global step;
- evaluation-code SHA256;
- split SHA256 and ordered frame IDs;
- requested Raw/EMA variants;
- every archived artifact hash.

Each `metrics.json` contains full-precision per-frame and scene-mean PSNR,
LPIPS-Alex, and SSIM. RGB, ground truth, float32 depth, and a depth preview use
the original `transforms.json` frame ID in their filename.

## Reporting policy

Raw and EMA are always retained as separate, auditable results. The scene-level
`comparison.json` records which variant wins each metric for diagnosis.

For an eight-scene paper summary, selection is made only after computing the
unweighted scene macro mean:

- PSNR and SSIM choose the higher global variant mean;
- LPIPS chooses the lower global variant mean;
- a metric uses one variant across all scenes;
- values are never selected independently per scene before averaging.

The most transparent paper table reports `Ours (Raw)` and `Ours (EMA)` as
separate rows. If a one-row per-metric selection is used, the source variant
must be disclosed.
