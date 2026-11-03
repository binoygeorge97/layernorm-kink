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
