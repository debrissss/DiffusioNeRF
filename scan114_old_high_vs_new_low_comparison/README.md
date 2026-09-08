# scan114 old-high vs new-low paired comparison

- Left: historical high-score render from `test/our/DTU/DTU/scan114-30k/color_XXX.png`.
- Right: newly trained legacy-camera/legacy-split render at step 6000 (raw) from `test_DTU/test_scan114/few_shot3_legacy_old43/test_DiffusioNeRF_NeurTV_Ray_30k_seed0/evaluation/step_006000/raw/frames/frame_XXXX_rgb.png`.
- Each pair is matched by the reconstructed legacy 43-view frame ordering. The actual DTU frame ID is encoded in the output filename and listed in `mapping.csv`.
- Layout: `OLD HIGH (30k)` on the left and `NEW LOW (step 6000 raw)` on the right.

Image count: 43 paired PNG files.
