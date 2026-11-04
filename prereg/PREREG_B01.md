# Pre-registration: the same scalar plant with ordinary authority, B = 0.1 (Task 2a)

Two registrations in one file: Part 1 (qualitative, about the surrogates and the fixed LQR) is written and committed BEFORE
any B = 0.1 surrogate exists; Part 2 (run-level DPC trap predictions from Lambda*) is appended after Stage A (surrogates and
GRU oracle at B = 0.1) and committed BEFORE Stage B (DPC through the surrogates). Nothing above a registration line is edited
after it is committed.

## Part 1 (written 2026-11-03, before Stage A)
Results that exist at the time of writing: everything at B = 0.01 (dpc_out). No data, label, surrogate, LQR or DPC result at
B = 0.1 exists (the code that produces them was finished today: src/ at the commit of this file).

### Question
Do the kink's effects persist when the input matters 10 times more (B = 0.1, A = 1.03)? At B = 0.1: K* = 5.2538, P = 10.41,
closed-loop pole 0.505; the safe band of the multiplier for the LQR gain is 0.057 B < J_u_hat < 3.86 B (0.29 B to 19.7 B at
B = 0.01), so the closed loop is MORE sensitive to a given relative slope error; whether traps persist depends on whether the
surrogate's relative slope errors shrink.

### Design (fixed before results)
- Stage A (Colab A100): DPC notebook, `PLANT = L.Plant(A=1.03, B=0.1, eps_nl=0.0)`, `ROOT = "dpc_out_B01"` (new, empty;
  labels inside it), `SURROGATES_ONLY = True`, `ONLY = ["true plant (oracle)"]`, `REPS = [0, 1, 2]`, `ARCH = "gru"`,
  `DTYPE = "float64"`, `QUICK = False`, `RUN_ROBUST = False`, `RUN_OPT = False`. Same data recipe (x ~ U[-4, 4],
  u ~ U[-10, 10], sigma 0.01, keys 1000 + s etc.), same six families, same training.
- Local: `LK_PLANT=1.03,0.1 LK_DATA=data/dpc_out_B01 LK_OUT=out/B01 python companion/run_lqr_study.py` (same 11 setpoints
  x 9 starts x 10 seeds, same metrics; u_ss(r) = -0.3 r); Lambda* of the B = 0.1 GRU oracle through the B = 0.1
  surrogates (`T2_PLANT=1.03,0.1 T2_DATA=../data/dpc_out_B01 T2_OUT=../out/B01/theory python t2_test.py ...`).
- Definitions: peak relative input-slope error = surrogates.csv `Ju_sup_rel` (sup over the box of |J_u_hat / B - 1|, the
  Level 1 metric); fixed-LQR design "outside the band" = true closed-loop pole |A - B DARE(A_hat, B_hat)| >= 1 with
  (A_hat, B_hat) the surrogate's slopes at its own equilibrium for the setpoint; catastrophic = diverged or gap > 100 %;
  runs with |x0 - r| >= 0.5; the four LN families.

### Predictions (Part 1)
| id | prediction | basis | how it is scored |
|---|---|---|---|
| Q1 | The relative input-slope errors are smaller at B = 0.1: median over the 10 seeds of `Ju_sup_rel` of LN MSE 3000 below 18.7 (its B = 0.01 value), and the same direction for at least 3 of the 4 LN families (B = 0.01 medians: MSE 12000 2.80, slope from 0 7.92, two-stage 1.19) | the kink's absolute slope error is set by the value error over the half-width; dividing by a 10x larger B shrinks the relative error | medians from surrogates.csv of both roots |
| Q2 | Fixed-LQR catastrophic failures occur only where the setpoint design leaves the band: P(catastrophic given pole inside the band) <= 2 % over the four LN families (1.3 % at B = 0.01, n = 3440) | the fixed LQR reads only the setpoint slopes | rate over all LN runs with |x0 - r| >= 0.5 |
| Q3 | Every design outside the band (|pole| >= 1) is catastrophic (100 % at B = 0.01, n = 80), if there are any | as Q2 | rate; "no such design" is reported as not testable |
| U1 | No directional prediction for the fixed- or rescheduled-LQR failure rates, nor for whether DPC traps persist (Part 2 gives run-level predictions from Lambda*) | the two effects (smaller relative errors, narrower band) act in opposite directions | reported |

Decision rule (CLAUDE.md, stated in advance): if DPC traps vanish at B = 0.1, the paper's claim becomes "low-authority
plants", and the two-state plant (Task 2b) should also be low authority.

## Outcomes Part 1 (fill AFTER Stage A; do not edit anything above this line)
| id | outcome (held / failed / partly) | numbers |
|---|---|---|

Part 1 registered at commit 8ecc750aa43798a65efc08265db7ab8f3e570acb (2026-11-03), before any B = 0.1 data existed.

## Outcomes Part 1 (computed 2026-11-03 from Stage A, `data/dpc_out_B01`)
| id | outcome | numbers |
|---|---|---|
| Q1 | held | median peak abs(J_u_hat/B - 1), B = 0.01 -> 0.1: LN, J_x + J_u (3000) 7.92 -> 0.72; LN, MSE (12000) 2.80 -> 0.27; LN, MSE (3000) 18.67 -> 2.63; LN, two-stage 1.19 -> 0.14; no norm, MSE 0.77 -> 0.07; static norm, MSE 0.46 -> 0.06; LN families smaller: 4 of 4 |
| Q2 | held | P(catastrophic given pole inside the band) = 0.26 % (n = 3472) |
| Q3 | held | designs outside the band: 48, catastrophic 100.0 % |

LQR failure rates at B = 0.1 (% failed / catastrophic of runs with abs(x0 - r) >= 0.5):
| model | failed, LQR fixed | failed, LQR sched | catastrophic, LQR fixed | catastrophic, LQR sched |
|---|---|---|---|---|
| LN MSE 12000 | 1.5 | 0.8 | 0.9 | 0.0 |
| LN MSE 3000 | 25.1 | 15.5 | 5.6 | 3.9 |
| LN slope from step 0 | 1.4 | 0.6 | 0.0 | 0.1 |
| LN two-stage | 0.6 | 0.3 | 0.0 | 0.0 |
| no norm | 0.0 | 0.0 | 0.0 | 0.0 |
| static norm | 0.0 | 0.0 | 0.0 | 0.0 |

LN MSE 3000 kink geometry at B = 0.1 (per seed):
| seed | Ju_sup_rel | zmin_x | zmin_u | rstar_x | rstar_u |
|---|---|---|---|---|---|
| 0 | 0.784 | 0.011 | -0.06 | 0.443 | 0.652 |
| 1 | 5.21 | -0.017 | 0.005 | 0.125 | 0.166 |
| 2 | 12.2 | 0.08 | -0.021 | 0.117 | 0.106 |
| 3 | 2.34 | 0.038 | -0.036 | 0.231 | 0.352 |
| 4 | 1.74 | 0.043 | 0.042 | 0.274 | 0.297 |
| 5 | 0.985 | -0.011 | -0.075 | 0.514 | 0.898 |
| 6 | 0.609 | 0.041 | 0.055 | 0.563 | 0.775 |
| 7 | 2.91 | -0.028 | 0.01 | 0.246 | 0.275 |
| 8 | 5.52 | -0.082 | -0.022 | 0.212 | 0.27 |
| 9 | 4.24 | -0.067 | 0.039 | 0.215 | 0.252 |

## Part 2 (written 2026-11-03, after Stage A, BEFORE Stage B)
Results that exist at the time of writing: Stage A at B = 0.1 (surrogates, labels, GRU oracle repeats [0, 1, 2]), the LQR study
and Lambda* above. No DPC controller trained through a B = 0.1 surrogate exists.
Stage B (Colab A100): `PLANT = L.Plant(A=1.03, B=0.1, eps_nl=0.0)`, `ROOT = "dpc_out_B01"`, `SURROGATES_ONLY = False`,
`ONLY = None`, `REPS = [0, 1, 2]`, `ARCH = "gru"`, `RUN_ROBUST = False`, `RUN_OPT = False`, `DTYPE = "float64"`.
Primary outcome: trapped (CLAUDE.md 1.4, oracle of the same plant, seed and repeat). Predictor: Lambda* of the B = 0.1 GRU
oracle through the surrogate (theory/t2_test.py, T2_PLANT = 1.03,0.1), threshold 12.1 NOT re-tuned.

| id | prediction | how it is scored |
|---|---|---|
| R1 | trapped iff Lambda* > 12.1; predicted trapped: 1 of 180 runs (list below, `prereg/PREREG_B01_predicted.csv`) | confusion matrix; held if hit rate >= 0.75 and false-alarm rate <= 0.05 (no predicted traps: held if at most 1 run is trapped) |
| R2 | two-stage, no norm and static norm: no trapped run | count |

Per family:
| family | runs | predicted_trapped | max_Lambda |
|---|---|---|---|
| LN MSE 3000 | 30 | 1 | 14.7 |
| LN MSE 12000 | 30 | 0 | 1.52 |
| LN slope from 0 | 30 | 0 | 0.729 |
| LN two-stage | 30 | 0 | 0 |
| no norm | 30 | 0 | 0 |
| static norm | 30 | 0 | 0 |

Runs predicted trapped:
- LN MSE 3000, seed 2, repeat 2: Lambda* = 14.7

## Outcomes Part 2 (fill AFTER Stage B; do not edit anything above this line)
| id | outcome (held / failed / partly) | numbers |
|---|---|---|

Part 2 registered at commit 77d8455ebd10da88547b436f3ea277a6bad376d8 (2026-11-03), before any DPC controller was trained through a B = 0.1 surrogate.

## Outcomes Part 2 (filled 2026-11-04 from Run 6 with the frozen analysis/b01_report.py 180ec7b; details docs/RESULTS_B01.md)
| id | outcome | numbers |
|---|---|---|
| R1 | not held (hit condition failed; false-alarm condition held) | predicted 1, trapped 7 (LN MSE 3000 s1 r0-r2, s2 r0-r2, s9 r2): 1 hit (s2 r2), 6 misses, 0 false alarms; hit rate 0.14, false-alarm rate 0 |
| R2 | held | 0 of 90 two-stage / no norm / static norm runs trapped |
Decision rule (Part 1): traps do not vanish at B = 0.1, so the claim is not restricted to low-authority plants.
