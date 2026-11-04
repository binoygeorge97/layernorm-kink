# DRAFT (R4, 2026-11-04): changes to the Lambda* claims - NOT APPLIED. Approve or edit; nothing changes before that.

Wording used throughout (as you gave it), with the numbers:
- Lambda* bounds the gradient size: ||grad L|| <= 4.7 e^{Lambda*} at the good controller holds on 180/180 GRU runs.
- It separates GRU traps at B = 0.01 retrospectively: 17/20 traps, 0/160 others, threshold 12.1 fitted on those data
  (on LN MSE 3000).
- It failed three prospective tests, each pre-registered with threshold 12.1:
  - MLP (f945a5c): 0 of 40 traps predicted.
  - Linear (322e353): 12 false alarms, no trap.
  - GRU at B = 0.1 (77d8455): 1 of 7 traps predicted, 0 false alarms.
- A trap also needs a tame region for the policy to settle in. This is theory.tex's own remark after Corollary 6. It
  explains the linear false alarms, but it was not part of the pre-registration.

Two facts behind the drafts, from docs/RESULTS_lambda_posthoc.md (post hoc):
- The bound also holds on 90/90 linear runs. On the MLP runs it holds on 89/90 (one at 6.1 e^{Lambda*}).
- At the MLP oracle the gradient is tame on the trapped runs (no exploding start). At the linear oracle on the 12 false
  alarms it explodes (1e27-1e34, one start carries 94-100 %).

Every OLD string below is copied verbatim from the file. Replace it by NEW.

---------------------------------------------------------------------------------------------------------------------------
## 1. docs/CLAIMS.md (section "Theorem 2 and Lambda*")

T1, OLD:
`| T1 | Lambda* > 12.1 (fixed on LN MSE 3000) flags the traps from the good controller alone | 17/20 hits = 85 [55, 100] %, 0/160 false alarms [0, 0] | theory/t2_test.py -> out/theory/t2_runs_*.csv | established (reproduced 180/180) |`
T1, NEW:
`| T1 | Lambda* > 12.1 separates the GRU traps at B = 0.01 retrospectively (threshold fitted on LN MSE 3000 of the same data) | 17/20 hits = 85 [55, 100] %, 0/160 false alarms [0, 0] | theory/t2_test.py -> out/theory/t2_runs_*.csv | established as a retrospective fit (reproduced 180/180); NOT a validated predictor, see T6-T8 |`

T2, OLD:
`| T2 | the result does not hinge on 12.1 | any threshold in [8.61, 17.03) gives 17/20 and 0/160; leave one seed out 11/12, 0/18 | out/stats/lambda_threshold_sweep.csv, lambda_loso.csv | established |`
T2, NEW:
`| T2 | within the GRU B = 0.01 data the separation does not hinge on 12.1 | any threshold in [8.61, 17.03) gives 17/20 and 0/160; leave one seed out 11/12, 0/18 | out/stats/lambda_threshold_sweep.csv, lambda_loso.csv | established (within that data set only; the zero-false-alarm interval is different on every prospective set: MLP (5.97, 7.36], GRU B = 0.1 (10.73, 14.75], linear > 93.4) |`

T4, OLD:
`| T4 | gradient bound | grad <= 4.63 e^Lambda* on all 180 runs | t2_runs_*.csv | established (an observation, not a theorem for the GRU) |`
T4, NEW:
`| T4 | Lambda* bounds the gradient size at the good controller | grad <= 4.7 e^Lambda* on 180/180 GRU runs (max 4.63); 90/90 linear (max 0.41); 89/90 MLP (one at 6.1) | t2_runs_*.csv, out/theory/arch_gradients_*.csv | established (an observation, not a theorem for the GRU) |`

T5, OLD:
`| T5 | memoryless LQR-gain policy: Lambda* <= 8.75 on all 180 -> linear DPC predicted never trapped | max 8.746 | theory/t2_memoryless.py, t2_arch.py --unit | established; test pending Runs 2 and 5 |`
T5, NEW:
`| T5 | memoryless LQR-gain policy: Lambda* <= 8.75 on all 180 -> linear DPC predicted never trapped (P2) | max 8.746; Run 5b: 0 of 180 linear runs trapped | theory/t2_memoryless.py, t2_arch.py --unit, docs/RESULTS_arch.md | held (P2). But the trained linear oracle has gain about 17, not 10.3, and its own Lambda* (up to 93) predicted 12 traps that did not occur (P2' failed) |`

T6, OLD:
`| T6 | Lambda* of the MLP oracle predicts MLP traps (prospective) | - | analysis/make_prereg_arch.py, arch_report.py | pending Runs 1 and 5 |`
T6, NEW (three rows):
`| T6 | Lambda* of the MLP oracle predicts MLP traps (prospective, f945a5c) | predicted 0, trapped 40: 0/40 hits, 0/140 false alarms; MLP oracle Lambda* <= 7.4 on all 180 | analysis/make_prereg_arch.py, arch_report.py | FAILED |`
`| T7a | Lambda* of the trained linear oracle predicts linear traps (prospective, 322e353) | predicted 12, trapped 0: 12 false alarms (6.7 %) | same | FAILED |`
`| T8 | Lambda* of the B = 0.1 GRU oracle predicts B = 0.1 traps (prospective, 77d8455) | predicted 1, trapped 7: 1/7 hits, 0/173 false alarms; the augmented Lambda*_aug (91238b4): 0/7 | analysis/b01_report.py | FAILED (hit condition) |`
(The existing T7 row, Lambda*_mem, keeps its number.)

Add after the table:
`A trap also needs a tame region for the policy to settle in (theory.tex, remark after Corollary 6). This explains the linear false alarms: the gradient at the linear oracle explodes (1e27-1e34) on all 12, yet two parameters leave no region to fall into. It was not part of the pre-registration. The MLP traps occur without an exploding gradient at the MLP oracle (docs/RESULTS_lambda_posthoc.md). So at B = 0.01 the explosion at the good controller was neither necessary (MLP) nor sufficient (linear) for a trap.`

---------------------------------------------------------------------------------------------------------------------------
## 2. docs/NOTES_theory.md

(a) OLD:
`  fixed there; other families: 0 false positives in 150 runs, slope-from-0 traps caught 5/8 (missed s4 rep1, s9 rep1, s3 rep2,
  Lambda* 2.5-6.5). Overall 17/20 traps, 0/160 others.`
NEW:
`  fixed there; other families: 0 false positives in 150 runs, slope-from-0 traps caught 5/8 (missed s4 rep1, s9 rep1, s3 rep2,
  Lambda* 2.5-6.5). Overall 17/20 traps, 0/160 others: a RETROSPECTIVE separation (threshold fitted on these data).
- Prospective tests (pre-registered, threshold 12.1), all failed:
  - MLP (f945a5c): 0/40 traps flagged.
  - Linear (322e353): 12 false alarms, 0 traps.
  - GRU at B = 0.1 (77d8455): 1/7 traps flagged, 0 false alarms.
  A trap also needs a tame region for the policy to settle in (theory.tex). That explains the linear false alarms, but it
  was not pre-registered.`

(b) OLD:
`- Linear DPC (2 params) through LN MSE 3000: not trapped on any of the 30 runs (memoryless Lambda* <= 8.7 everywhere).
- MLP DPC: compute Lambda* of the MLP reference controller (trained through the true plant) BEFORE looking at the MLP DPC results;
  prediction = trapped iff Lambda* > 12.1.`
NEW:
`- Linear DPC (2 params) through LN MSE 3000: not trapped on any of the 30 runs (memoryless Lambda* <= 8.7 everywhere).
  OUTCOME (Run 5b): held, 0/180 trapped. The trained-oracle version (P2', Lambda* up to 93) failed: 12 false alarms.
- MLP DPC: compute Lambda* of the MLP reference controller (trained through the true plant) BEFORE looking at the MLP DPC results;
  prediction = trapped iff Lambda* > 12.1. OUTCOME (Run 5a): failed. 40 MLP traps (27 LN MSE 3000, 12 slope from 0,
  1 MSE 12000), none flagged; the MLP oracle's gradient is tame on the trapped runs.`

---------------------------------------------------------------------------------------------------------------------------
## 3. docs/theory/theory.tex

(a) Summary paragraph (line 50). OLD (the sentence starting):
`With the threshold fixed on LN MSE 3000 and applied to the other 150 runs it flags no untrapped run and 5 of the 8 slope-from-step-0 traps: 17 of 20 traps and 0 of 160 others overall.`
NEW:
`With the threshold fixed on LN MSE 3000 and applied to the other 150 runs it flags no untrapped run and 5 of the 8 slope-from-step-0 traps: 17 of 20 traps and 0 of 160 others overall. This separation is retrospective (the threshold was fitted on these data). Three pre-registered prospective tests with the same threshold failed: an MLP policy (0 of 40 traps flagged), the trained two-parameter linear policy (12 false alarms, no trap), and the GRU on the same plant with $B=0.1$ (1 of 7 traps flagged, no false alarm). What holds throughout is the bound: the gradient at the good controller is at most $4.7\,e^{\Lambda^\star}$ on all 180 GRU runs.`
And OLD (same paragraph):
`The condition depends on the policy: a memoryless policy at the LQR gain never expands on any surrogate, which predicts that the two-parameter linear DPC policy is not trapped.`
NEW:
`The condition depends on the policy: a memoryless policy at the LQR gain never expands on any surrogate, and the two-parameter linear DPC policy was indeed never trapped; but its trained good controller, with gain about 17, does expand (up to $e^{93}$) without being trapped, because a trap also needs a tame region for the policy to settle in (remark after Corollary~\ref{cor:trap}).`

(b) "What it does not show", item 9. OLD:
`\item The threshold was chosen after seeing the LN MSE runs. The held-out families support it (no false positives in 150 runs) but contain only 8 traps.`
NEW:
`\item The threshold was chosen after seeing the LN MSE runs. The held-out families support it (no false positives in 150 runs) but contain only 8 traps. Prospectively it failed. MLP policy: 0 of 40 traps flagged; the MLP's good controller never expands ($\Lambda^\star\le7.4$) and has a tame gradient on the trapped runs. Trained linear policy: 12 false alarms. GRU at $B=0.1$: 1 of 7 traps flagged. Within each set $\Lambda^\star$ still ranks the traps above most untrapped runs (post hoc, ROC AUC 0.81 within LN MSE 3000 at $B=0.1$), but no single threshold transfers.`

(c) Remark after Corollary 6 (line 348). OLD:
`That gradient descent then fails, i.e.\ the trap itself, also needs somewhere tame to go: a flexible policy can settle where the gradients are honest (just outside the arms), while the two-parameter linear policy of the architecture study has no such place and was not trapped in the smoke test.`
NEW:
`That gradient descent then fails, i.e.\ the trap itself, also needs somewhere tame to go: a flexible policy can settle where the gradients are honest (just outside the arms), while the two-parameter linear policy has no such place. It was not trapped on any of 180 runs, including the 12 where its good controller's gradient explodes ($10^{27}$ to $10^{34}$). This condition was not part of the pre-registered predictions; it explains the linear false alarms after the fact.`

---------------------------------------------------------------------------------------------------------------------------
## 4. Companions (companion/full/main.tex, companion/part1/main.tex)
No hit: neither companion states a Lambda* claim (grep for Lambda, Theorem 2, 17/20, 12.1). No change.

---------------------------------------------------------------------------------------------------------------------------
## 5. companion/background/main.tex (background note for slides 17-19)

(a) OLD (line 678, last sentence):
`On every one of the 180 runs the gradient at the good controller is at most $4.7\,e^{\Lambda^\star}$, so $\Lambda^\star$ also bounds the gradient from above, which $e^{N\lambda}$ did not reliably do.`
NEW:
`On every one of the 180 runs the gradient at the good controller is at most $4.7\,e^{\Lambda^\star}$, so $\Lambda^\star$ also bounds the gradient from above, which $e^{N\lambda}$ did not reliably do. The separation of traps above is retrospective: the threshold was fitted on these runs.`

(b) OLD (line 690, last sentence):
`Prediction: a linear DPC controller would not be trapped on any of these surrogates, and for an MLP controller the test should be run on its own good controller before looking at the results.`
NEW:
`Prediction, made before the runs: a linear DPC controller would not be trapped, and the MLP's own good controller decides for the MLP. Outcome:
- The linear controller was never trapped. But its trained good controller (gain about 17) expands up to $e^{93}$ on 12 runs, and those were false alarms.
- The MLP was trapped on 40 runs, none of them flagged: its good controller never expands ($\Lambda^\star\le7.4$).`

(c) OLD (line 694, last sentence):
`Its empirical record above (17 of 20 traps, no false alarm) is what supports it.`
NEW:
`Its record above (17 of 20 traps, no false alarm) is a retrospective fit. In three pre-registered prospective tests it failed: MLP 0 of 40 traps flagged, linear 12 false alarms, GRU at $B=0.1$ 1 of 7 traps flagged. What it reliably measures is the size of the gradient at the good controller. A trap also needs a tame region for the policy to settle in, which $\Lambda^\star$ does not see.`

(d) OLD (line 701):
`\item Keep $\lambda$ for the intuition, but quote $\Lambda^\star$ for the claim: 17 of 20 traps flagged, no false alarm in 160 runs.`
NEW:
`\item Keep $\lambda$ for the intuition. Quote $\Lambda^\star$ as a bound on the gradient size (holds on 180 of 180 runs) and as a retrospective separation of the GRU traps (17 of 20, no false alarm in 160, threshold fitted on these runs). Not as a trap predictor: it failed three prospective tests.`

---------------------------------------------------------------------------------------------------------------------------
## 6. out/slide_fixes.md: proposed new entries for the deck

Slides 17 (dpcwhy), 18 (dpctrap) and 19 (dpcwarm) do not state any Lambda* claim. I read the deck, version
1793714113-a1c3: their text and notes use lambda, the gradient sizes and the warm start only. Nothing to correct there for
Lambda*. Two other slides state things that Run 5 now contradicts (proposals, your decision):

15. Slide 23 (limits), speaker notes, third paragraph
   OLD: `the trap statistics are already policy-independent because they are measured at the reference controller.`
   NEW: `the trap statistics are measured at each architecture's own reference controller. Run 5 showed they do not transfer: the MLP is trapped (40 of 180 runs) with a tame gradient at its reference, and the linear policy is never trapped although its reference's gradient explodes on 12 runs.`
16. Slide 22 (synthesis), speaker notes, last paragraph
   OLD: `for gradient-based control, check the gradient norm at a known-good controller;`
   NEW: `for gradient-based control, a huge gradient at a known-good controller is a warning sign (it accompanied every GRU trap at B = 0.01), not a test (the linear policy survives it, the MLP is trapped without it);`
17. Slide 23 (limits), left column, bullet 2
   OLD: `One architecture (pre-norm, H = 16); RMSNorm and post-norm are predictions, not results`
   NEW: `One surrogate block (pre-norm, H = 16); RMSNorm and post-norm are predictions, not results`
   (The controller architecture is no longer one: MLP and linear DPC were run. "Architecture" there means the surrogate.)

---------------------------------------------------------------------------------------------------------------------------
## 7. S3 (added 2026-11-04): optimiser wording after Run 4 - NOT APPLIED
Source: docs/RESULTS_opt.md, Run 4 section and S2 follow-ups (post hoc parts labelled there).

Proposed wording (the whole optimiser story in four sentences):
- From scratch, global clipping (clip 3) and lr 1e-4 also rescue most target seeds: 3 of 4 and 4 of 4 (rescued = not failed
  in >= 2 of 3 repeats; seeds 1, 2, 8, 9). Global clipping leaves 2 traps in 30 runs and lr 1e-4 leaves 3.
- Unit per-start aggregation gives 0 of 30 failed (and 0 trapped).
- Two-stage training with a plain optimiser remains the best: per-seed median gap 0.04 %, against 0.9-2.4 % for unit, per-start
  clip, global clip and lr 1e-4. It beats per-start clip, global clip and lr 1e-4 on 10 of 10 seeds (Holm p = 0.016 each)
  and unit on 9 of 10 (p = 0.020, Holm 0.078: not significant after Holm). Run-level medians: 0.04 % vs 0.94-2.16 %.
- Per-start clipping removes the traps (0 of 30) but leaves 5 model-mismatch failures (seeds 5, 8, 9; gaps 20-55 %): the
  surrogate rates those controllers 1-3 % better than the oracle controller.
- Caveat to state with it: the N <= 50, N <= 100 and batch 100 trap counts are confounded with horizon/noise mismatch until
  the matched-oracle control is run (RESULTS_opt.md S2(b)); do not quote "truncation does not avoid the trap".
Note: the range "0.9-2.2 %" in the request is the run-level range without lr 1e-4's per-seed value; with per-seed medians
it is 0.94-2.42 %, with run-level medians 0.94-2.16 %. I propose quoting the per-seed numbers, since the test uses them.

Every place that says otherwise (OLD copied verbatim; NEW proposed):
18. docs/CLAIMS.md, row D11
   OLD: `| D11 | generic optimiser tricks do not keep it: global clip 3 / median of means / minibatch 100 |`
   NEW: `| D11 | in the WARM START, generic optimiser tricks do not keep it: global clip 3 / median of means / minibatch 100 |`
   (true for the warm start; add a note "from scratch see D13").
19. docs/CLAIMS.md, row D13
   OLD: `| D13 | optimiser baselines from scratch (P1, P2a, P3a, P4, P5) | - | analysis/opt_report.py | pending Run 4 |`
   NEW: `| D13 | from scratch: unit 0/30 failed; per-start clip 0/30 trapped, 5 mismatch failures; global clip rescues 3/4 targets (2 traps), lr 1e-4 4/4 (3 traps); median of means 1/4; two-stage plain best (per-seed median gap 0.04 % vs 0.94-2.42 %) | Run 4 | analysis/opt_report.py (218cbbe), run4_followups.py | P1 held; P2a, P3a, P4 failed; P5a confounded; P5b held |`
20. docs/CLAIMS.md, row G4
   OLD: `| G4 | "global clipping does not help", "MLP/linear behave like ..." | may NOT be claimed before Runs 4 and 5 |`
   NEW: `| G4 | "global clipping does not help" | may NOT be claimed: false from scratch (Run 4: 3 of 4 target seeds rescued); true only in the warm start |`
21. Slide 19 (dpcwarm), speaker notes, paragraph 2
   OLD: `This is an argument, not a measurement: global clipping has not been run (the same warm-start test with the summed gradient clipped is the next cheap check).`
   NEW: `Measured since: in the warm start, global clipping is expelled on 4 of 6 broken pairs (per-start clipping 0 of 6). From scratch, however, global clipping rescues 3 of the 4 trapped seeds, and lr 1e-4 all 4: step size matters there, not only direction.`
22. Slide 19 (dpcwarm), speaker notes, paragraph 3
   OLD: `Caveat: clipping from a random initialisation (not a warm start) has not been run yet (RUN_ROBUST, A100). Present per-start clipping as a diagnosis (it identifies the cause), not as a fix. The fix is two-stage training (next slides).`
   NEW: `From a random initialisation (Run 4): per-start clipping leaves 0 of 30 trapped, with 5 model-mismatch failures; unit per-start aggregation 0 of 30 failed. Present them as working optimiser-side fixes that cost 4x the training time; two-stage training is still better (median gap 0.04 % vs 0.9-2.4 %) and costs nothing extra (next slides).`
23. Slide 23 (limits), left column, bullet 3
   OLD: `Per-start clipping not yet run from scratch: a causal test, not yet a fix`
   NEW: `Optimiser fixes work from scratch too (unit 0/30 failed), but cost 4x and leave a larger gap than two-stage`
24. Slide 23 (limits), right column, bullet 2
   OLD: `Per-start gradient clipping trained from a random start`
   NEW: `Matched-oracle control for the truncated-horizon and minibatch baselines`
25. Slide 23 (limits), speaker notes, paragraph 4
   OLD: `Clipping: rescues a warm start; whether it lets training from a random start find the good basin is pending (RUN_ROBUST, A100).`
   NEW: `Clipping: rescues a warm start and, from a random start, removes every trap (Run 4); global clipping and lr 1e-4 also rescue most trapped seeds from scratch, though not in the warm start.`
26. docs/NOTES_dpc_level1.md, lines 336-337 (T2a)
   OLD: `    Still open: whether clipping also lets DPC FIND the good controller from scratch (dpc_core.ROBUST_MODELS, 'clip' = 3 and`
   `    'unit' vote; notebook block "Optimiser-side control", A100 run pending).`
   NEW: append a dated line instead (NOTES are history): `    Update 2026-11-04 (Run 4): from scratch, clip 3 leaves 0/30 trapped (5 mismatch failures), unit 0/30 failed; global clip 3 and lr 1e-4 rescue 3/4 and 4/4 target seeds. See RESULTS_opt.md.`
27. docs/NOTES_dpc_level1.md line 310 ("robust per-start gradient clipping / median aggregation as the generic optimiser-side fix"): same dated update; median of means does NOT work from scratch (1 of 4).
28. CLAUDE.md 1.6, "May not (yet): "global clipping does not help" (never run)": now run; the statement is false from scratch.
   CLAUDE.md is your handout, so I leave it; it is consistent if read as "may not be claimed".
29. docs/RESULTS_opt.md, Run 4 "Reading", bullet 4: `Truncating the horizon to N <= 50 or <= 100 does not avoid the trap: 15 and 12 trapped. P5a failed.`
   Already qualified by the dated S2 note (P5a confounded). No edit (results file history).
Not affected (checked): companions full/part1 (no optimiser claim), theory.tex, background note (its per-start clipping
section is about the warm start and stays true), out/slide_fixes.md, paper/ (tables regenerated from Run 4 already).

---------------------------------------------------------------------------------------------------------------------------
## 8. S3 (added 2026-11-04): kink-centre wording - NOT APPLIED
Measured centres, max(|x*|, |u*|), all LN families available (data/dpc_out/surrogates.csv, data/dpc_out_s10_29/surrogates.csv):
| set | families x seeds | x* range | u* range | largest offset |
|---|---|---|---|---|
| seeds 0-9 | 4 LN families x 10 | -0.091 to 0.072 | -0.068 to 0.105 | 0.105 (LN MSE 12000 seed 4) |
| seeds 0-9, LN MSE 3000 only | 10 | -0.091 to 0.072 | -0.068 to 0.072 | 0.091 |
| seeds 10-29 | LN MSE 3000, two-stage x 20 | -0.071 to 0.070 | -0.136 to 0.144 | 0.144 (two-stage seed 28) |
| all | 80 centres | -0.091 to 0.072 | -0.136 to 0.144 | 0.144 |
Record (new finding while checking): the "within 0.1" statement already missed on the original 10 seeds, by one centre:
LN MSE 12000 seed 4 at (0.067, 0.105). On held-out seeds it missed on four (pre-registered H3, 23b5835, failed): LN MSE 3000
seed 28 (-0.046, +0.141), seed 29 (-0.006, +0.131); two-stage seed 10 (-0.071, -0.136), seed 28 (-0.053, +0.144).
Statements restricted to the ten LN MSE 3000 surrogates (|x*| <= 0.09, |u*| <= 0.07) remain true.

Proposed wording: "every LN kink centre lies within 0.15 of (0, 0): |x*| <= 0.09 and |u*| <= 0.15 over 80 centres (4 LN
families on seeds 0-9, LN MSE 3000 and two-stage on seeds 10-29)". Run 7 adds the other two LN families on seeds 10-29; the
range should be re-measured then.

Places (OLD verbatim; NEW proposed):
30. docs/CLAIMS.md, row K2
   OLD: `| K2 | every LN kink centre lies within 0.1 of (0, 0) | 10/10 seeds | surrogates.csv (zmin_x, zmin_u) | established |`
   NEW: `| K2 | every LN kink centre lies within 0.15 of (0, 0): abs(x*) <= 0.09, abs(u*) <= 0.15 | 80 centres (seeds 0-9 x 4 LN families, seeds 10-29 x 2) | surrogates.csv (zmin_x, zmin_u) | corrected 2026-11-04 (was "within 0.1, 10/10 seeds"; missed on MSE 12000 seed 4 and on 4 held-out centres) |`
31. Slide 5 (kinkmap), big number and caption
   OLD: `0.1` / `every kink centre lies within 0.1 of the origin (10 of 10 seeds)`
   NEW: `0.15` / `every kink centre lies within 0.15 of the origin (80 centres, 30 seeds)`
   (alternatively keep 0.1 and say "LN MSE 3000, 10 of 10 seeds", which is true but no longer the general statement.)
32. Slide 7 (kinkwhy), speaker notes
   OLD: `training moves b, which widens the kink while the centre stays within 0.1 of the origin.`
   NEW: `training moves b, which widens the kink while the centre stays within 0.15 of the origin.`
33. docs/theory/theory.tex line 229 (LN MSE 3000 context): `Their centres stay within $0.1$ of the origin` - true for the ten
   LN MSE 3000 seeds it describes; propose adding "on seeds 0-9 (within 0.15 on 20 held-out seeds)".
34. companion/full/main.tex lines 132, 304, 496 and companion/part1/main.tex lines 157, 326, 514: all three describe the ten
   LN MSE 3000 surrogates ("all centred within 0.1 of the origin"; "$|x^\star|\le0.09$, $|u^\star|\le0.07$"; "centres within
   0.1 of the origin"). True as written; no change needed unless the companions start covering other families or seeds.
35. docs/NOTES_dpc_level1.md line 148: `every LN surrogate's kink centre is within 0.1 of (0, 0)` -> dated update "(within
   0.15 over 80 centres, 2026-11-04)".
36. src/build_phase_notebook.py line 37 (markdown text in the phase notebook): `The kink centre of every LayerNorm surrogate sits within 0.1 of (0, 0)`
   -> `within 0.15 of (0, 0)`. A text-only change, but it changes the notebook file, so it needs a rebuild; I would do it
   with the next notebook rebuild.
37. CLAUDE.md 1.2 `every LN kink centre within 0.1 of (0, 0)`: your handout; same correction if you want it.
(prereg/PREREG_heldout.md H3 and PREREG_rms.md M1 use 0.1 as a pre-registered criterion: not to be edited.)
