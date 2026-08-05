# Standard sparse-view DTU 6-view splits

These manifests follow the DTU protocol documented by FreeNeRF and inherited
from RegNeRF/PixelNeRF. IDs are zero-based indices in each scene's 49-frame
`transforms.json`.

- Training IDs: `25, 22, 28, 40, 44, 48`
- Test IDs: `1, 2, 9, 10, 11, 12, 14, 15, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 41, 42, 43, 45, 46, 47`
- Validation uses only test frame 1 because this benchmark does not define a
  separate validation set. It is monitoring-only and keeps periodic validation
  inexpensive; experiments use a fixed training-step budget and never select
  checkpoints by validation performance.
- Remaining frames are excluded from benchmark evaluation because the standard
  protocol excludes views with unsuitable exposure.
