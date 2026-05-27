# Final-checkpoint frozen-policy analysis report

- Final-eval rows parsed: **16**
- τ values: [10.0]
- methods: ['context50', 'memoryless']
- bootstrap: 10,000 resamples, RNG seed 12345, resampling over **seeds** (not eval episodes)

## Coverage: seeds per (τ, method)

| τ | method | n_seeds | seeds |
|---|---|---|---|
| 10 | context50 | 8 | [42, 43, 44, 45, 46, 47, 48, 49] |
| 10 | memoryless | 8 | [42, 43, 44, 45, 46, 47, 48, 49] |

## Summary by method — τ = 10

| method | metric | n | mean | median | IQM | std | min | max | mean 95% CI | IQM 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| context50 | mean_F | 8 | 0.7215 | 0.7915 | 0.7627 | 0.2035 | 0.4050 | 0.9391 | [0.5849, 0.8493] | [0.5394, 0.8892] |
| context50 | median_F | 8 | 0.7555 | 0.8591 | 0.8156 | 0.2369 | 0.3558 | 0.9661 | [0.5918, 0.9016] | [0.5378, 0.9465] |
| context50 | p25_F | 8 | 0.6171 | 0.6936 | 0.6460 | 0.2739 | 0.2462 | 0.9189 | [0.4373, 0.7945] | [0.3455, 0.8530] |
| context50 | p10_F | 8 | 0.4580 | 0.4746 | 0.4355 | 0.2794 | 0.1636 | 0.8459 | [0.2794, 0.6435] | [0.1711, 0.7101] |
| context50 | min_F | 8 | 0.0719 | 0.0205 | 0.0238 | 0.1077 | 0.0000 | 0.2831 | [0.0108, 0.1497] | [0.0002, 0.1606] |
| context50 | P_gt_90 | 8 | 0.3941 | 0.4429 | 0.4167 | 0.3125 | 0.0000 | 0.8054 | [0.1888, 0.5960] | [0.0916, 0.6588] |
| context50 | P_gt_95 | 8 | 0.2771 | 0.2592 | 0.2624 | 0.2406 | 0.0000 | 0.6206 | [0.1243, 0.4372] | [0.0554, 0.4942] |
| context50 | P_gt_99 | 8 | 0.0975 | 0.0632 | 0.0751 | 0.1038 | 0.0000 | 0.2854 | [0.0352, 0.1686] | [0.0116, 0.1831] |
| context50 | worst5_mean | 8 | 0.2746 | 0.2185 | 0.2036 | 0.2487 | 0.0439 | 0.7032 | [0.1243, 0.4486] | [0.0735, 0.5099] |
| memoryless | mean_F | 8 | 0.7247 | 0.7520 | 0.7438 | 0.1705 | 0.4868 | 0.9443 | [0.6145, 0.8322] | [0.5648, 0.8664] |
| memoryless | median_F | 8 | 0.7594 | 0.7870 | 0.7784 | 0.2023 | 0.4910 | 0.9920 | [0.6277, 0.8886] | [0.5637, 0.9456] |
| memoryless | p25_F | 8 | 0.5981 | 0.6095 | 0.6071 | 0.2466 | 0.2432 | 0.9351 | [0.4392, 0.7567] | [0.3842, 0.8094] |
| memoryless | p10_F | 8 | 0.4536 | 0.4626 | 0.4368 | 0.2410 | 0.1174 | 0.8374 | [0.2954, 0.6135] | [0.2563, 0.6453] |
| memoryless | min_F | 8 | 0.0780 | 0.0328 | 0.0561 | 0.0804 | 0.0083 | 0.2252 | [0.0314, 0.1324] | [0.0197, 0.1409] |
| memoryless | P_gt_90 | 8 | 0.3704 | 0.3134 | 0.3297 | 0.2782 | 0.0000 | 0.8190 | [0.2012, 0.5603] | [0.1602, 0.6166] |
| memoryless | P_gt_95 | 8 | 0.2747 | 0.2076 | 0.2030 | 0.2506 | 0.0000 | 0.7008 | [0.1245, 0.4504] | [0.0888, 0.4984] |
| memoryless | P_gt_99 | 8 | 0.1412 | 0.0408 | 0.0412 | 0.2104 | 0.0000 | 0.5082 | [0.0222, 0.2984] | [0.0107, 0.3424] |
| memoryless | worst5_mean | 8 | 0.2847 | 0.2357 | 0.2568 | 0.1899 | 0.0533 | 0.6148 | [0.1668, 0.4126] | [0.1365, 0.4390] |

## Paired differences (CA50 − memoryless) — τ = 10

| metric | n_pairs | mean Δ | median Δ | IQM Δ | std Δ | min Δ | max Δ | CA50 wins | mean Δ 95% CI | IQM Δ 95% CI | sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean_F | 8 | -0.0032 | 0.0028 | 0.0206 | 0.2458 | -0.4537 | 0.3264 | 4/8 (50%) | [-0.1717, 0.1484] | [-0.1846, 0.1904] | 1.0000 |
| median_F | 8 | -0.0040 | -0.0029 | 0.0006 | 0.2733 | -0.4874 | 0.3332 | 4/8 (50%) | [-0.1901, 0.1683] | [-0.2176, 0.2249] | 1.0000 |
| p25_F | 8 | 0.0190 | 0.0279 | 0.0532 | 0.3583 | -0.6038 | 0.5086 | 4/8 (50%) | [-0.2256, 0.2436] | [-0.2519, 0.3042] | 1.0000 |
| p10_F | 8 | 0.0044 | -0.0032 | 0.0300 | 0.3971 | -0.6688 | 0.5918 | 4/8 (50%) | [-0.2624, 0.2579] | [-0.2925, 0.3140] | 1.0000 |
| min_F | 8 | -0.0060 | -0.0134 | -0.0324 | 0.1622 | -0.2250 | 0.2556 | 4/8 (50%) | [-0.1067, 0.1021] | [-0.1303, 0.1387] | 1.0000 |
| P_gt_90 | 8 | 0.0237 | 0.0789 | 0.0615 | 0.4155 | -0.8190 | 0.6362 | 5/8 (62%) | [-0.2653, 0.2762] | [-0.2542, 0.3014] | 0.7266 |
| P_gt_95 | 8 | 0.0024 | 0.0462 | 0.0313 | 0.3576 | -0.7008 | 0.5318 | 4/8 (50%) | [-0.2451, 0.2230] | [-0.2425, 0.2473] | 1.0000 |
| P_gt_99 | 8 | -0.0438 | 0.0098 | -0.0001 | 0.2469 | -0.4496 | 0.2782 | 4/8 (50%) | [-0.2102, 0.1099] | [-0.2630, 0.1390] | 1.0000 |
| worst5_mean | 8 | -0.0101 | -0.0265 | -0.0301 | 0.3722 | -0.5709 | 0.5787 | 4/8 (50%) | [-0.2491, 0.2350] | [-0.3036, 0.2918] | 1.0000 |

## Interpretation (per τ, descriptive only)

> Each τ is analyzed independently — no cross-τ comparisons are made here. n is small in every condition, so bootstrap CIs are wide. "CI excludes 0" is described as a direction in the available seeds, **not** as a statistical significance claim.

- **τ = 10** — n_pairs = 8; mean Δ mean_F = -0.0032 (95% CI [-0.1717, +0.1484]); CA50 wins 4/8 (50%). The bootstrap CI for the per-seed Δ straddles 0 — the sign of the effect is **seed-dependent** at this n. The data here do not support a directional claim either way.

## Files in this directory

- `final_checkpoint_raw.csv` — one row per final eval (all parsed rows)
- `final_checkpoint_summary_by_tau_method.csv` — per (τ, method, metric) stats + bootstrap CIs
- `final_checkpoint_paired_seed_differences.csv` — per-seed CA50 − memoryless deltas (intersection only)
- `final_checkpoint_paired_difference_summary.csv` — per-(τ, metric) paired-Δ stats + bootstrap CIs
- `performance_profiles_final_long.csv` — long-form P(metric > θ) for θ ∈ [0, 1] step 0.01
- `paired_diff_final_{mean_F,p10_F}_tau*.png` — per-seed Δ bar plots
- `performance_profile_final_{mean_F,p10_F}_tau*.png` — survival-style profiles across seeds
- `scatter_context_vs_memoryless_final_meanF_tau*.png` — per-seed scatter with y=x
