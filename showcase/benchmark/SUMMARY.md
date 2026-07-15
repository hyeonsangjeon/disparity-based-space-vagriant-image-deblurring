# Benchmark summary

Comparisons are ordered **blurred + noisy -> result**, followed by a reference when available.

| Dataset | Evaluation | Score | PSNR | SSIM | Runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| gopro-flower | reference | 0.713320 | 26.325 | 0.8998 | 0.248 |
| 01_df2_object_motion | no-reference proxy | 0.616397 | N/A | N/A | 4.137 |
| 02_building_low_light | no-reference proxy | 0.639017 | N/A | N/A | 2.021 |
| 05_new1_parking | no-reference proxy | 0.713232 | N/A | N/A | 15.001 |

No-reference results use a conservative proxy objective, not a noisy-input similarity objective.
