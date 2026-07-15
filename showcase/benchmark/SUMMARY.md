# Benchmark summary

Comparisons are ordered **blurred + noisy -> result**, followed by a reference when available.

| Dataset | Objective type | Full-resolution objective | PSNR | SSIM | Runtime (s) |
| --- | --- | ---: | ---: | ---: |
| gopro-flower | reference | 0.694603 | 25.370 | 0.8816 | 0.599 |
| 01_df2_object_motion | no-reference-proxy | 0.632094 | N/A | N/A | 2.637 |
| 02_building_low_light | no-reference-proxy | 0.640904 | N/A | N/A | 0.905 |
| 05_new1_parking | no-reference-proxy | 0.750229 | N/A | N/A | 3.449 |

Reference-backed entries report PSNR and SSIM against their published reference.
No-reference-proxy entries use a conservative full-resolution proxy objective, not PSNR, SSIM, or noisy-input similarity.
