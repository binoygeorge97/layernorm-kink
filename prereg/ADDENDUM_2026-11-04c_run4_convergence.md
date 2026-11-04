# Addendum to PREREG_opt.md (c41967c): convergence condition for the lr 1e-4 setting (P4)

Written 2026-11-04, before Colab Run 4 (optimiser baselines from scratch) has started. No from-scratch result of any
OPT_MODELS setting exists. P4 itself is unchanged ("lr 1e-4 from scratch does NOT rescue: at most 1 of the 4 target seeds
rescued").

Why: a learning rate 10 times smaller may simply not finish training within the same 9000-epoch curriculum. A "not
rescued" seed would then say nothing about the trap. P4 is only informative if the setting trains a good controller
where the plain optimiser does.

Healthy LN MSE seeds (GRU, plain AdamW lr 1e-3, data/dpc_out/dpc_results.csv; typical-start gap against the oracle of
the same seed and repeat):
| seed | gap rep 0 | gap rep 1 | gap rep 2 |
|---|---|---|---|
| 0 | 1.2 % | 2.1 % | 0.9 % |
| 3 | 2.1 % | 0.8 % | 0.6 % |
| 4 | 0.2 % | 0.2 % | 0.2 % |
| 6 | 4.5 % | 2.6 % | 2.6 % |
| 7 | 0.2 % | 0.2 % | 0.2 % |
Seed 5 is excluded: it failed once with the plain optimiser (rep 0, gap 29.8 %, model mismatch, not trapped). Seeds 1, 2,
8, 9 are the target (trapped) seeds.

Rule (fixed now): model "LN, MSE (3000), lr 1e-4" counts as CONVERGED only if each of the five healthy seeds 0, 3, 4, 6, 7
reaches a typical-start gap <= 10 % in at least 2 of its 3 repeats (the same 2-of-3 rule as the rescue rule of
PREREG_opt). If it is converged, P4 is scored with the rescue rule as registered. If not, P4 is reported as
"not testable: not converged", with the gaps of all 15 healthy runs and of the 12 target runs listed. Also reported in
either case (not part of the rule): how many of the 15 healthy runs individually reach <= 10 %.
