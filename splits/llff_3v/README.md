# LLFF 3-view standard splits

These manifests use zero-based frame IDs in the original `transforms.json`
order.

- Test IDs follow the LLFF holdout-8 rule: `frame_id % 8 == 0`.
- Training IDs are three approximately equally spaced samples from the
  remaining trajectory, using round-half-up when the middle pool position is
  fractional.
- Validation uses frame 0 to preserve the project's one-view validation
  behavior. It is used for monitoring only; formal results must use the fixed
  final checkpoint and every ID in `test_ids`.

Pass a manifest with `--split_file`. When an explicit split is active, the
loader rejects missing images, out-of-range or duplicate IDs, train/test
overlap, and a `--few_shot` value that disagrees with the manifest.
