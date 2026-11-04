# Augmented-state landing amplification Lambda*_aug for the GRU (B4, 2026-11-04) - RETROSPECTIVE

Definition: docs/theory/exact_lambda_aug/exact_lambda_aug.tex (PDF out/theory/exact_lambda_aug/), committed (156e71b) before
computing. Operator norm of the product of the 65 x 65 closed-loop Jacobians of z = (x, h), stretch anchored at the landing
step, maximised over its end; exact SVD at every step. Code theory/t2_aug.py (about 60 s per run on CPU); all 180 GRU runs at
B = 0.01, out/theory/t2_aug_B001.csv. The pre-registered direct-gain Lambda* stays the primary statistic.

Retrospective result (trap labels known):
- Zero-false-alarm interval on the 160 non-trapped runs: thresholds above 11.21 (LN MSE 3000 seed 0 rep 1). Hit rate across it:

| threshold | traps flagged | false alarms |
|---|---|---|
| 11.22 | 15/20 | 0/160 |
| 12.08 | 15/20 | 0/160 |
| 13.0 | 15/20 | 0/160 |
| 13.02 | 14/20 | 0/160 |
| 16.0 | 13/20 | 0/160 |
| 20 | 12/20 | 0/160 |
| 30 | 9/20 | 0/160 |
| 50 | 5/20 | 0/160 |

  i.e. 15 of 20 for every threshold in (11.21, 13.01], then falling.
- LN MSE 3000 separates (11.21 < 13.01); geometric midpoint 12.08 (the rule of 12.1), fixed for Run 6 in PREREG_B01_aug.md
  Part 1 (8b26564) before any B = 0.1 value was computed.
- Per family at 12.08:

| family | traps | flagged | false alarms | largest non-trapped |
|---|---|---|---|---|
| LN MSE 3000 | 12.0 | 12.0 | 0.0 | 11.21 |
| LN MSE 12000 | 0.0 | 0.0 | 0.0 | 3.91 |
| LN slope from 0 | 8.0 | 3.0 | 0.0 | 8.92 |
| LN two-stage | 0.0 | 0.0 | 0.0 | 3.33 |
| no norm | 0.0 | 0.0 | 0.0 | 1.47 |
| static norm | 0.0 | 0.0 | 0.0 | 1.48 |

- Comparison on the same 180 runs: direct-gain Lambda* 17/20 on its interval (8.61, 17.03]; the same landing-anchored stretch
  with the scalar multiplier 17/20 on (8.23, 16.10]. The augmented statistic misses two more slope-from-0 traps (seed 9 reps 0
  and 2: Lambda*_aug 9.4 and 7.6 against direct-gain 84 and 45), and the three traps no version catches (slope-from-0 s4 r1, s9
  r1, s3 r2) stay low (2.3, 3.8, 4.6).
- Consistent with the exact-price version (RESULTS_t2_exact.md): including the GRU memory lowers the amplification of several
  trapped runs; the memory of the GOOD controller compensates part of what the direct gain sees. The anchored scalar statistic
  equals Lambda* on 111 of 180 runs (on the others the maximising stretch starts later than the landing step).
