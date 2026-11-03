# Handoff: presentation package, corrections and theory (as of 2026-10-28)

Read this first in a new chat. It summarises a long chat (Oct 23-28) that built the presentation package for the
LayerNorm-kink control results, corrected several claims, and started the theory. Science details are in
NOTES_dpc_level1.md, NOTES_controller_arch.md, HANDOFF_level1.md and the companion LaTeX (claude/companion/main.tex).

## Deliverables and where they live
| Item | Where | Notes |
|---|---|---|
| Slide deck "When does the LayerNorm kink reach a controller?" | Slides artifact https://claude.ai/artifact/K51GsPfQQ1jdU1aw6MaeHy (version 27) | Read it with the Artifact tool (action read, paths project/deck.json, project/slides/*.html) before any publish; publish with root = the folder holding project/ |
| Presenter's briefing | Doc artifact https://claude.ai/artifact/GkfPr6HhxRKsCKWTfkeZpC | Per slide: plots, math, hypothesis/observation/conclusion, likely questions; companion page refs give full companion first, Part 1 version in brackets |
| Handover doc "LayerNorm kink and control: what changed (Oct 23-27)" | Doc artifact https://claude.ai/artifact/Hiy6mvaiXoT7mBMorSb5yT | For people not in the chat; has corrections table and open-items checklist |
| Companion, full (35 pp, printed slides 1-14 with DPC pointers) | claude/companion/main.tex | Figures f01-f16 from claude/companion/make_figs.py (f01-f15) and make_fig3d.py (f16); lqr_study.csv from run_lqr_study.py |
| Companion, Part 1 only (35 pp, no DPC) | claude/companion/main_part1.tex | Same figures; extra glossary section, so section numbers are one higher than the full version |
| Slide style skill | research-talk-slides (user's skills) | Still describes the old footer/cover; needs updating to: no footer text, slide number only; cover = title + subtitle |

Data needed to recompute anything: the saved surrogates and controllers live on the user's Google Drive (MyDrive/dpc_out:
surrogates/*.npz, repeat-0..2 controllers). Code: claude/dpc/{level1_core, dpc_core, phase_core}.py, claude/dpc/story/*.py.

## Deck structure (printed numbers; viewer position = printed + 2 after the cover because 2 glossary slides are unnumbered)
cover | glossary (1 of 2: models and scoring, 13 terms) | glossary2 (2 of 2: controllers and training, 9 terms) |
2 question | 3 setup | 4 why the early analysis was hard (3 lessons: only J_x, one seed, u fixed) | 5 kink maps 2-D |
6 kink as crests and troughs (3-D, added 27 Oct) | 7 closed form and petals | 8 fixed LQR | 9 rescheduled LQR |
10 path-colour test | 11 failure vs setpoint | 12 setpoint law | 13 swap example | 14 which slope (swap rates) [end of Part 1] |
15 DPC bridge | 16 DPC failure | 17 why: exploding gradient | 18 no barrier / chaotic gradient | 19 warm start |
20 DPC swap | 21 scorecard | 22 synthesis | 23 limits | 24 takeaway | appendix A1-A8 (A1 LQR anim, A2 DPC anim, A3 warm anim,
A4 data/kinks, A5 full box, A6 LQR timeline, A7 swap seed 8, A8 warm phase plane). Part 1 talk = slides 1-14 + A1, A4-A7.

House style: white background, black text and titles, dark blue #1B3A6B for a few key phrases, dark red #8B1A1A for
highlights/limits, thick dark-red boxes only on slides 2, 12, 20, 24, no other boxes or rules, Calibri / Calibri Light /
Aptos Display (Carlito, Inter Tight in browser), no footer, no white boxes behind figures, "-" not middle dots.

## Notation (decided 27-28 Oct; matches the reference derivation prenorm_layernorm_jacobian_derivation.pdf)
h = E z + b, E in R^{16x2} with columns e_x, e_u (code stores E transposed). r = P h = Q_E z + c,
Q_E = P E = [q_x q_u] (q_x = the reference's q for a sweep along x), c = P b. Centre z* = (x*, u*) = -(Q_E^T Q_E)^{-1} Q_E^T c,
c_perp = Q_E z* + c, offset s = z - z*, sigma(s) = sqrt(s^T Q_E^T Q_E s / H + delta^2), delta^2 = ||c_perp||^2 / H + eps,
D = sqrt(H) delta (kept as the quoted kink size 0.13-0.90). Half-widths r*_x = D / ||q_x||, r*_u = D / ||q_u||.
Q_E is named so to avoid the LQR weight Q = 5. Far-field branch function Phi(theta). Old names (A_E, z_min, r_0, delta as
offset, G(theta)) are gone everywhere.

## Corrections made in this chat (all propagated to deck, briefing, companions, handover)
- Seeds 3, 4, 7 are middle-width, not "wide"; the widest are 0, 5, 6 (1-9 % catastrophic). Predictor of catastrophic
  fixed-LQR failure = a setpoint on a wrong-sign region (seeds 0, 1, 2, 8, 9), not width.
- lambda > 0 on 5 of the 6 broken pairs of repeat 0 (slope-from-0 seed 9: endpoint lambda -0.025, spike on the way in);
  across repeats seed 0 has small positive lambda and never fails. Separating statistic = largest gradient along the
  straight line (>= 1e3 failed, <= 1e2 healthy, 30/30 LN MSE, 15/18 slope-from-0). Healthy gradient at the good
  controller 0.03-3.5.
- "Global clipping does not help" was never run: now stated as an expectation. Per-start clipping = diagnosis from a warm
  start, not a fix. MLP / linear DPC = 2-seed smoke test only.
- Stale "Slide 2" / "Slides 4-5" titles in f03 / f08 regenerated and swapped into the deck.

## Open items (most valuable first)
1. Pending user decision: slide 5 says wrong-sign arms reach "2-4 half-widths"; measured farthest reach is 1.7-6.8 on 8
   seeds, 9.5 (seed 1), 32.6 (seed 2); 56-99 % of wrong-sign area lies outside the ellipse. Offer: rephrase to "2-7 on
   most seeds, more on the narrowest" in slide 5, notes, companions, briefing.
2. Theory (see below): write Theorem 1 + corollaries; then Theorem 2 (sufficient condition for the DPC trap).
3. Cheap tests: warm-start test with global clipping; optimiser baselines (truncated BPTT / shorter horizon, lr sweep);
   per-start clipping from scratch (RUN_ROBUST, A100); MLP and linear DPC (E-A1, E-A2); checkpoints for the 3 unexplained
   slope-from-0 failures; seed-level bootstrap CIs on all rates; failure rate vs threshold (5-100 %) and direct offset /
   settling-time metrics; G'(theta) amplitude from weights to settle the decay exponents.
4. Generality: a second plant (2 states or mildly nonlinear, or A = 1.07); RMSNorm prediction.
5. Offered, not started: separate Part 1 version of the deck; MPC experiment (fixed linear MPC, LTV MPC, gradient NMPC at
   several horizons, MPPI; prediction: damage grows with reliance on derivatives, MPPI nearly immune); update the
   research-talk-slides skill.

## Review of Part 1 (27 Oct)
Math verified (scalar DARE, kink derivation, far field, closed-loop arithmetic). Soft spots: "K scales like 1/B_hat" is
overstated (B_hat = 18.3 B gives K 3.98, factor 2.6); decay exponents (-0.98 setpoint line vs -1.57/-1.22 max on circle)
need the exact form; no confidence intervals. Rough odds for Part 1 alone: CDC/ACC ~50 %, L4DC ~40 % (55-60 % with fixes and
a 2nd plant), NeurIPS/ICML/ICLR main 10-15 %, TMLR ~40 %. Part 2 (DPC trap) is what would carry an ML venue.
DPC verdict (28 Oct): presentable with scoped claims; for a paper needs Theorem 2, optimiser baselines, memoryless
policies, a second plant.

## Theory status (28 Oct)
User's Theorem 1 (exact low-dimensional reduction of prenorm LayerNorm with affine pre-norm map, z in R^m, rank Q_E = k):
centre Q_E z* = -Pi c (unique iff k = m); LN(h(z)) = V phi(w(z)), phi(w) = sqrt(H) [w; ||c_perp||] / sqrt(||w||^2 + D^2),
||w|| = ||Q_E (z - z*)||; c_perp = 0 for all b iff k = H - 1, else a (k+1)-dim subspace of biases.
- Verified numerically to 1e-15 (random E, b; rank-deficient Q_E; eps = 0, 1e-6, 1e-2).
- Fix two edge cases: (a) k = H - 1 (V with k+1 orthonormal columns orthogonal to 1 cannot exist; assume k < H - 1 or
  drop the last column when c_perp = 0); (b) eps = 0 and c_perp = 0 (D = 0, LN undefined at z*; require ||w||^2 + D^2 > 0).
  When c_perp = 0 and k < H - 1 the last column of V is arbitrary.
- Novelty: not found stated in the literature (closest: Brody et al. 2023 project-and-scale view; Riechers 2024
  hyperellipsoid; Chun 2026 LLC), but elementary: present as a structural lemma whose value is in the corollaries.
- Corollaries to write: (1) eps = 0: phi is the inverse gnomonic projection of the plane at height ||c_perp|| onto the
  radius-sqrt(H) sphere (image = open hemisphere); (2) singular values of D phi: tangential sqrt(H)/sqrt(rho^2 + D^2)
  (multiplicity k - 1), radial sqrt(H) D/(rho^2 + D^2) at eps = 0 (general eps: sqrt(H) sqrt(D^4 + ||c_perp||^2 rho^2) /
  (rho^2 + D^2)^{3/2}); both sqrt(H)/D at the centre; 1/rho across vs 1/rho^2 along (rigorous petal statement), verified
  numerically; (3) hemisphere area independent of D: peak gain sqrt(H)/D unbounded as D -> 0 while outputs stay bounded
  (formal reason MSE cannot see the kink; generalises the reference's pi sqrt(H)); (4) block Jacobian = skip +
  w_d^T J_G diag(gamma) V D phi(w) U^T Q_E: all spatial structure from D phi and J_G on the hemisphere; (5) eps irrelevant
  for trained nets (c_perp != 0 a.s.; zero-bias init gives c_perp = 0 exactly).
- Next: Theorem 2 = sufficient condition for the DPC trap: closed-loop multiplier |J_x_hat - J_u_hat K| > 1 on a large
  enough share of the good policy's landing path gives lambda > 0 and ||grad|| ~ e^{N lambda}; must account for arm
  orientation (MSE 12000 seed 2 and LN MSE seed 7 are tall but survive). Test against the 30 LN MSE runs.
  Cite exploding-gradient / chaotic-differentiation work (Pascanu et al. 2013; Metz et al. 2021; Suh et al. 2022).
  [DONE 28 Oct: see NOTES_theory.md and docs/theory/theory.tex.]
