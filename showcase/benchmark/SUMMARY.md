# Benchmark summary

Comparisons are ordered **blurred + noisy -> result**, followed by a reference when available.

| Dataset | Objective type | Full-resolution objective | Blurred PSNR | Result PSNR | Result SSIM | Runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| gopro-flower | reference | 0.694603 | 28.331 | 25.370 | 0.8816 | 0.475 |
| 01_df2_object_motion | no-reference-proxy | 0.632649 | N/A | N/A | N/A | 2.805 |
| 02_building_low_light | no-reference-proxy | 0.640904 | N/A | N/A | N/A | 1.762 |
| 05_new1_parking | no-reference-proxy | 0.707996 | N/A | N/A | N/A | 3.716 |

Reference-backed entries report PSNR and SSIM against their published reference.
No-reference-proxy entries use a conservative full-resolution proxy objective, not PSNR, SSIM, or noisy-input similarity.
