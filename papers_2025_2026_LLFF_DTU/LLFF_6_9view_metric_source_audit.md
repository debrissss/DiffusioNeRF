# LLFF 6/9-view metric source audit

This audit covers the PDFs stored in this directory. A method is included in
the 6-view and 9-view comparison CSVs only when the paper has a quantitative
table explicitly labeled as LLFF with 6 and 9 training views, and the reported
row is the paper's own method rather than a reproduced baseline.

## Confirmed paper-reported results

| Method | Paper table | LLFF setting | 6-view PSNR / SSIM / LPIPS | 9-view PSNR / SSIM / LPIPS |
|---|---|---|---:|---:|
| MixNeRF | CVPR 2023, Table 2 | LLFF 3/6/9-view | 23.76 / 0.791 / 0.115 | 25.20 / 0.833 / 0.087 |
| DiffusioNeRF | CVPR 2023, Table 1 | LLFF 3/6/9-view | 23.79 / 0.747 / 0.237 | 25.02 / 0.785 / 0.212 |
| CoR-GS | ECCV 2024, Table 1 | LLFF 3/6/9-view | 24.49 / 0.837 / 0.115 | 26.06 / 0.874 / 0.089 |
| ReVoRF | CVPR 2024, Table 4 | LLFF 3/6/9-view | 22.21 / 0.720 / 0.269 | 23.04 / 0.753 / 0.225 |
| SE-GS | ICCV 2025, Table 1 | LLFF 3/6/9-view | 24.78 / 0.839 / 0.110 | 26.36 / 0.878 / 0.084 |
| ICO-GS | CVPR 2026, Table 1 | LLFF 3/6/9-view | 25.37 / 0.856 / 0.109 | 26.45 / 0.881 / 0.096 |

## Checked but not eligible as LLFF 6/9-view results

- FrugalNeRF reports LLFF 2/3/4-view results.
- SYN3R reports LLFF only at 3 views; its 6/9-view columns are for DL3DV.
- FlipNeRF reports LLFF at 3 views; its 3/6/9-view table is for DTU.
- Binocular3DGS + Gaussian Dropout, DNGaussian, IPSM-Gaussian, EdgeNeRF,
  DivCon-NeRF, and the downloaded LM-Gaussian paper do not report their own
  LLFF results at both 6 and 9 views in the checked versions.
- A later paper may reproduce a baseline at 6/9 views. Such reproduced baseline
  rows are not attributed as the original method's own reported results here.

## Comparability caveats

- Paper rows are copied from each paper's cross-scene LLFF average table.
- The project rows use unweighted macro means over its eight scene-level means.
- DiffusioNeRF explicitly reports LPIPS-VGG, while this project's formal
  evaluation uses LPIPS-Alex. LPIPS values across those protocols are not
  directly equivalent.
- Training iterations, initialization, implementation, and exact sparse-view
  split details can differ between papers even when they refer to the common
  LLFF 3/6/9-view benchmark.
