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
