# Series summary: runs/series_v2

Final numbers on the official validation set (already inspected in preliminary runs).
Energy = compute-only analytical estimate, not a measurement.

## Individual runs

| arm | seed | top1_restricted | top1_1000 | top5_1000 | firing_rate | est_energy_mJ | ft_updates | p1_updates | ft_examples | p1_examples | ft_seconds | p1_seconds | total_seconds | resumed | split |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p1_ft05 | seed0 | 61.61 | 61.61 | 92.89 | 0.0704 | 0.53 | 1330 | 6916 | 42560 | 221312 | 4931.30 | 5459.40 | 10390.70 | 0 | test |
| p1_ft05 | seed1 | 61.17 | 61.17 | 93.15 | 0.0698 | 0.53 | 1330 | 6916 | 42560 | 221312 | 4390.80 | 5475.70 | 9866.50 | 0 | test |
| p1_ft05 | seed2 | 60.84 | 60.84 | 92.92 | 0.0699 | 0.53 | 1330 | 6916 | 42560 | 221312 | 5561.50 | 5509.60 | 11071.10 | 0 | test |
| nop1_ft05 | seed0 | 48.33 | 48.33 | 89.68 | 0.0910 | 0.75 | 1330 | 0 | 42560 | 0 | 4957.60 | 0.00 | 4957.60 | 0 | test |
| nop1_ft05 | seed1 | 47.90 | 47.90 | 89.96 | 0.0898 | 0.75 | 1330 | 0 | 42560 | 0 | 5177.10 | 0.00 | 5177.10 | 0 | test |
| nop1_ft05 | seed2 | 48.46 | 48.46 | 90.14 | 0.0911 | 0.76 | 1330 | 0 | 42560 | 0 | 4935.70 | 0.00 | 4935.70 | 0 | test |
| p1_ft12 | seed0 | 70.57 | 70.57 | 93.86 | 0.0750 | 0.57 | 3192 | 6916 | 102144 | 221312 | 11710.80 | 5459.40 | 17170.20 | 0 | test |
| p1_ft12 | seed1 | 69.58 | 69.58 | 93.40 | 0.0740 | 0.56 | 3192 | 6916 | 102144 | 221312 | 11326.80 | 5475.70 | 16802.50 | 0 | test |
| p1_ft12 | seed2 | 70.42 | 70.42 | 93.66 | 0.0746 | 0.56 | 3192 | 6916 | 102144 | 221312 | 11500.20 | 5509.60 | 17009.80 | 0 | test |
| nop1_ft12 | seed0 | 60.18 | 60.18 | 91.90 | 0.0931 | 0.78 | 3192 | 0 | 102144 | 0 | 11718.00 | 0.00 | 11718.00 | 0 | test |
| nop1_ft12 | seed1 | 59.72 | 59.72 | 92.05 | 0.0928 | 0.78 | 3192 | 0 | 102144 | 0 | 11379.90 | 0.00 | 11379.90 | 0 | test |
| nop1_ft12 | seed2 | 59.69 | 59.69 | 91.77 | 0.0929 | 0.77 | 3192 | 0 | 102144 | 0 | 11483.20 | 0.00 | 11483.20 | 0 | test |

## Mean and sample standard deviation over seeds

| arm | n | top1_restricted mean | top1_restricted sd | top1_1000 mean | top1_1000 sd | top5_1000 mean | top5_1000 sd | firing_rate mean | firing_rate sd | total_seconds mean | total_seconds sd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p1_ft05 | 3 | 61.21 | 0.38 | 61.21 | 0.38 | 92.99 | 0.14 | 0.07 | 0.00 | 10442.77 | 603.99 |
| nop1_ft05 | 3 | 48.23 | 0.29 | 48.23 | 0.29 | 89.93 | 0.23 | 0.09 | 0.00 | 5023.47 | 133.50 |
| p1_ft12 | 3 | 70.19 | 0.54 | 70.19 | 0.54 | 93.64 | 0.23 | 0.07 | 0.00 | 16994.17 | 184.35 |
| nop1_ft12 | 3 | 59.86 | 0.27 | 59.86 | 0.27 | 91.91 | 0.14 | 0.09 | 0.00 | 11527.03 | 173.26 |

## Paired differences p1 - nop1 (same seed, same fine-tuning schedule)

- ft05, top1_restricted: seed0: +13.27, seed1: +13.27, seed2: +12.38 | mean +12.98, sd 0.51 | positive in 3/3 seeds
- ft05, top1_1000: seed0: +13.27, seed1: +13.27, seed2: +12.38 | mean +12.98, sd 0.51 | positive in 3/3 seeds
- ft12, top1_restricted: seed0: +10.39, seed1: +9.86, seed2: +10.73 | mean +10.33, sd 0.44 | positive in 3/3 seeds
- ft12, top1_1000: seed0: +10.39, seed1: +9.86, seed2: +10.73 | mean +10.33, sd 0.44 | positive in 3/3 seeds

The p1 arms also include the phase-1 cost (p1_updates, p1_seconds): the factorial design compares equal fine-tuning schedules, not equal total budgets.
