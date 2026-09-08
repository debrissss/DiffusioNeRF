# Unified LLFF 3-view comparison

This table normalizes the reported LLFF three-view results to the same columns: PSNR (↑), SSIM (↑), and LPIPS (↓). Published values are copied from the cited papers/project pages; they were not rerun in this repository. The project result uses the common standard split in this repository: every eighth image for test, three uniformly selected non-test training views, and the EMA checkpoint at 12,000 global steps.

| Method | Year/venue | Representation | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Result type | Source/protocol note |
|---|---|---|---:|---:|---:|---|---|
| SparseNeRF | 2023 CVPR | NeRF | 19.86 | 0.620 | 0.330 | Published | 3-view LLFF table |
| SPARF | 2023 CVPR | NeRF | 20.20 | 0.630 | 0.330 | Published | 3-view LLFF table |
| MVPGS | 2024 ECCV | 3DGS | 20.65 | 0.880 | 0.100 | Published | 3-view LLFF table |
| SCGaussian | 2024 NeurIPS | 3DGS | 20.77 | 0.710 | 0.220 | Published | 3-view LLFF table |
| NexusGS | 2025 CVPR | 3DGS | 20.80 | 0.770 | 0.240 | Published | 3-view LLFF table |
| Path Matters (ICLR 2026 submission) | 2026 | 3DGS | 20.93 | 0.780 | 0.230 | Published | 3-view LLFF table |
| GoLF-NRT | 2025 CVPR | Generalizable NVS | 24.20 | 0.821 | 0.148 | Published | 3-view LLFF table |
| DiffusioNeRF + NeurTV + Ray (EMA@12k) | This project | NeRF | 19.796847 | 0.677369 | 0.196361 | Local evaluation | Common 8-scene split in this repository |

## Interpretation

The local result is below the recent 3DGS and generalizable-NVS entries in PSNR and SSIM. Its LPIPS is better than SparseNeRF, SPARF, SCGaussian, and NexusGS in the reported table, but worse than MVPGS and GoLF-NRT. The comparison is not a claim that every method used bit-identical preprocessing: only the metric schema and nominal three-view setting are normalized here.

The local EMA checkpoint trajectory is retained separately in `test_LLFF/aggregates/llff_3v_neurtv_ray_common_milestones/summary.md`: EMA@6000 has the best PSNR (20.029096), EMA@12000 has the best SSIM (0.677369), and EMA@18000 has the best LPIPS (0.193104). A paper table should select one fixed checkpoint or report this selection rule explicitly.

## Sources

- [GoLF-NRT CVPR 2025 results](https://klmav.cuc.edu.cn/2025/0529/c2286a256259/page.htm)
- [Path Matters / ICLR 2026 submission, Table 3](https://openreview.net/pdf/1ad8bd881c26849b79ebf5025a130a786ccd32ec.pdf)
- [Local eight-scene common-milestone summary](/root/autodl-tmp/DiffusioNeRF-main/test_LLFF/aggregates/llff_3v_neurtv_ray_common_milestones/summary.md)
