# Development protocol for a replacement trap predictor (R5), fixed BEFORE any candidate is computed on the development set

Written and committed 2026-11-04, before any warm-start expulsion, or any other candidate below, has been computed on
the MLP, linear or B = 0.1 runs. Already computed before this file: on the GRU B = 0.01 runs, the 10 harness pairs
(warm-start harness, all settings) and the oracle-gradient statistics (t2_test). On the MLP and linear runs: the
oracle-gradient statistics (R2, docs/RESULTS_lambda_posthoc.md). Those statistics are therefore NOT eligible as new
candidates; they enter only as the baseline.

## Development set (outcomes known)
All runs with a trap label (registered definition: gap > 10 % and own controller > 1.05 x the oracle controller through
the same surrogate). Four sets of 6 families x 10 seeds x 3 repeats = 180 runs each:
- GRU at B = 0.01 (data/dpc_out);
- MLP at B = 0.01 (data/run5_mlp, oracle data/dpc_out_mlp);
- linear at B = 0.01 (data/run5_linear, oracle data/dpc_out_linear);
- GRU at B = 0.1 (data/run6_B01, oracle data/dpc_out_B01).
That is 720 runs and 74 traps.

## Candidates (each computed from the architecture's oracle controller of the same seed and repeat, its 1000 training starts and the surrogate; no DPC-from-scratch run)
- C1 warm-start expulsion, true cost. Plain AdamW (lr 1e-3, weight decay 1e-4), started at the oracle, 100 epochs at N = 200,
  full batch of the 1000 training starts. Predicted trap iff the true-plant cost of the current controller on the training
  starts exceeds 1.10 x its epoch-0 value at any logged epoch 10, 15, ..., 100. This is the PREREG_opt expulsion rule,
  with no free parameter.
- C2 warm-start expulsion, surrogate cost. As C1, but on the cost through the surrogate (needs no true plant at training time).
- C3 warm-start expulsion with AdamW beta1 = 0. As C1, but AdamW without momentum. Included because R3 is testing
  momentum. It is computed only if C1 and C2 are computed anyway: same cost, one more run per unit.
- B0 (baseline, not eligible) direct-gain Lambda* > 12.1 as pre-registered.

## Selection rule (fixed now)
1. Eligible: candidates whose false-alarm rate is <= 5 % in EACH of the four sets.
2. Among the eligible, choose the largest worst-case hit rate over the three sets that contain traps (GRU B = 0.01,
   MLP, GRU B = 0.1).
3. Tie-break, in order:
   - larger pooled hit rate;
   - then lower pooled false-alarm rate;
   - then the cheaper candidate (C2 < C1 < C3: C2 needs no true plant).
4. If no candidate is eligible: report "no candidate met the criterion", list each candidate's numbers, and freeze nothing
   without your decision.
5. The chosen candidate has no threshold to tune: 1.10 and the epoch window are fixed above.

## Freezing
The chosen candidate is frozen in a commit (script, rule and commit hash recorded in this file's appendix) BEFORE the Run 8
pre-registration. Run 8 (GRU, seeds 10-29, 4 families) is then its first prospective test, together with the registered
direct-gain Lambda* (threshold 12.1).

## Compute plan
analysis/trap_predictor.py, vmapped over the 10 seeds of a (family, repeat). It logs the true and surrogate cost every 5
epochs and stops a unit early once both rules have fired for every seed. CPU, run in the background.

## Amendment (2026-11-04, before any candidate was computed)
1. The trap count is 67, not 74: 20 (GRU B = 0.01) + 40 (MLP) + 0 (linear) + 7 (GRU B = 0.1).
2. C3 (beta1 = 0) is dropped from the development set for cost. C1/C2 need about 12 CPU-hours for the 720 runs, and C3
   would double that. Momentum is examined separately on the 10 harness pairs (R3). Candidates are now C1 and C2. The rule
   is otherwise unchanged; the tie-break order is C2 < C1.
