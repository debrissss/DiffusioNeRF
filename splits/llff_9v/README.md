# LLFF standard 9-view splits

These manifests use zero-based frame IDs in the original `transforms.json`
order.

- Test IDs follow the LLFF holdout-8 rule: `frame_id % 8 == 0`.
- The training pool contains every frame not in the test split.
- Nine training IDs are selected at approximately equal intervals along the
  ordered training pool, using round-half-up for fractional positions.
- Validation uses frame 0 to match the existing formal 3-view and 6-view
  runners. It is for monitoring only; formal results use a fixed checkpoint
  and every ID in `test_ids`.

Pass a manifest with `--split_file`. The loader validates frame counts, IDs,
missing images, train/test overlap, and agreement with `--few_shot 9`.
