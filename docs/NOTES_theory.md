# Theory: Theorem 1 (exact reduction) and Theorem 2 (closed-loop multiplier condition) - 2026-10-28

Document: claude/theory/theory.tex (+ tab_lnmse.tex, tab_families.tex, figures/t1_theorem2_test.png), 14 pp, compiles with
pdflatex. Style: plain theorem heads (no bold/italic), no boxes, same preamble as the companion. Scripts in claude/theory/,
results in claude/theory/results/. Data: MyDrive/dpc_out (surrogates/*.npz, controllers/true_plant__oracle[__repN].npz,
controllers/LN__MSE__3000[__repN].npz, dpc_results.csv), pulled with the Google Drive connector (download_file_content
saves base64 JSON to a tool-results file; decode with python).

## Theorem 1 (written, proved, verified)
LN(Ez + b) = V phi(w), w = U^T Q_E (z - z*), phi(w) = sqrt(H) [w; ||c_perp||] / sqrt(||w||^2 + D^2), n = rank Q_E.
Notation clash fixes used in the doc: input dimension n_z, rank n, costate p_k, sources chi_k, transition Psi_{k,j},
multiplier m_k, landing amplification Lambda (Phi stays the far-field branch function).
Edge cases: (a) n = H-1 -> c_perp = 0 for every b, use LN = U phi_0(w), phi_0 = sqrt(H) w / sqrt(||w||^2 + H eps);
(b) eps = 0 and c_perp = 0 -> undefined on z* + ker Q_E. c_perp = 0 iff b in range(E) + span(1) (dim n+1).
Corollaries: 1 inverse gnomonic projection (eps = 0, image = open hemisphere); 2 singular values sigma_tan = sqrt(H)/sqrt(rho^2+D^2)
(mult n-1), sigma_rad = sqrt(H) sqrt(D^4 + a^2 rho^2)/(rho^2+D^2)^{3/2}; peak sqrt(H)/D; far field 1/rho across, <= 2 sqrt(H) D/rho^2
along (rigorous petal statement); kink ellipse rho <= D = gain within sqrt 2 of peak; 3 image volume (1/2) H^{n/2}|S^n| independent
of D (pi sqrt H, 2 pi H), value bounded / slope ~ 1/D / data mass in half-gain region ~ D^n; 4 block Jacobian
w_d^T E + w_d^T J_G diag(gamma) V Dphi U^T Q_E, far-field branch Phi(direction); 5 eps irrelevant once b has moved
(relative change <= H eps / (2 ||c_perp||^2) = 0.05 % at seed 2). Remark: RMSNorm = same theorem with P -> I (prediction).
verify_thm1.py: identity to 8e-15 (full rank, rank 3, n = H-1, eps 0/1e-6/1e-2), singular values 1.7e-6 rel, area exact for
D = 1e-3..7, block Jacobian vs finite differences 1e-9, trained surrogates through the reduction = F_point to 2e-15.
Literature checked: Riechers 2024 (arXiv 2405.04134) decomposes LN geometrically, no affine pre-map, no derivatives;
Brody et al. 2023 project-and-scale; Xu et al. 2019 1/||h|| gradient. Reduction itself not found stated.

## Theorem 2 (drafted, proved for memoryless policies, tested)
m_k = Jx_hat - Ju_hat K_k, K_k = -d pi/dx. (a) dx_k/dx_j = prod m; costate p_j = sum_k Psi_{k,j} chi_k;
grad = sum_j (2R u_j/N + Ju_hat(j) p_{j+1}) grad_theta pi(x_j). (b) band: m > 1 iff Ju_hat < (Jx_hat-1)/K, m < -1 iff
Ju_hat > (Jx_hat+1)/K; at A, K* = 10.2875: [0.29 B, 19.7 B]. (c) ||grad|| <= poly(N) e^{Lambda_all}. (d) uniformly expanding
segment (rate tau, slack kappa) + |p_b| >= 2 chi_bar e^kappa/(1-e^-tau) -> |p_a| >= 0.5 |p_b| e^{tau(b-a)-kappa}.
Recurrent policies: augmented Jacobian, m_k = state-state entry with the DIRECT gain; not exact for the GRU.
Landing window: k >= first k with |x_k| <= 0.2. Lambda(x0) = max segment sum of log|m_k| on it; Lambda* = max over the 1000 starts.

## Test (180 GRU runs, reference controller of each repeat through each surrogate; no DPC training)
- LN MSE 3000: trapped (12) Lambda* 17.0-158, others (18) <= 8.6 -> complete separation. Threshold 12.1 (geometric midpoint)
  fixed there; other families: 0 false positives in 150 runs, slope-from-0 traps caught 5/8 (missed s4 rep1, s9 rep1, s3 rep2,
  Lambda* 2.5-6.5). Overall 17/20 traps, 0/160 others: a RETROSPECTIVE separation (threshold fitted on these data).
- Prospective tests (pre-registered, threshold 12.1), all failed (added 2026-11-04):
  - MLP (f945a5c): 0/40 traps flagged.
  - Linear (322e353): 12 false alarms, 0 traps.
  - GRU at B = 0.1 (77d8455): 1/7 traps flagged, 0 false alarms.
  A trap also needs a tame region for the policy to settle in (theory.tex). That explains the linear false alarms, but it
  was not pre-registered.
- Earlier single-policy diagnostics: gradient > 1e3 at the reference catches 14/20 (misses LN MSE rep2 s8: 3.7; healthy rep1 s0 = 106);
  exact lyap > 0 on 16 trapped + 6 untrapped. Line spike (notes) 30/30 but needs the trapped controller.
- ||grad L|| <= 4.7 e^{Lambda*} on all 180 runs (<= 0.16 e^{Lambda*} on LN MSE). An upper bound, not tight (2026-11-04): of the 28 runs
  with Lambda* > 5 the gradient is within 10^3 of it on 8; also 90/90 linear and 89/90 MLP runs (one at 6.1). One start carries median 84 % of ||grad||^2 on traps.
- Swap along the same paths: Ju_hat -> B brings Lambda* below threshold on 9/12 LN MSE traps (to <= 5.7); Jx_hat -> A on 0/12.
  Exceptions s9 rep1-2, s1 rep2: state slope alone expands too (Jx_hat > 1 + B K ~ 1.2).
- Orientation: seed 1 (peak 24.4 B) trapped 3/3, seed 7 (23.1 B) 0/3; MSE 12000 seed 2 (35.5 B) Lambda* <= 2.5.
- Policy dependence: memoryless u = 10 tanh(-K* x/10) has Lambda* <= 8.7 on all 180 (factorisation exact to 3e-8).
  GRU reference direct gain at landing 13-22 (2x LQR; memory supplies the rest) -> band upper edge 9-16 B; its landing orbit
  moves (seed 2 rep 0: u from -1.6 to -0.07 = 17 r*_u) and sweeps peak and arms. Seed 2: expanding steps mostly ABOVE the band
  (Ju/B ~ +29, m < -1), not the wrong-sign arm.
- Neighbourhood (8 draws, weights x (1 + 1e-3 xi)): trapped seeds up to 5e43; healthy <= 448 (rep1 s0); rep2 s8 only 20 ->
  its reference GRU is genuinely tame; Lambda* flags it anyway. Interpretation (untested): DPC visits GRU policies without the
  reference's memory compensation. Test: training checkpoints.
- Reproduction: geometry 3 digits, surrogate costs 4 digits; gradient norms at the reference agree in order of magnitude only
  (seed 2: 3.9e35 vs 1.1e35 in notes; seed 9: 1.9e11 vs 1.2e14 / 9e11 in two earlier computations) - float accumulation order.

- Whole-path Lambda_all exceeds Lambda* by <= 3.3 (median 1.9); ||grad L|| <= 0.61 e^{Lambda_all} on all 180 runs (part (c),
  observation for the GRU). Scalar costate max within a factor 18-3000 of e^{Lambda*} on the 17 flagged traps (no cancellation).

## Independent proof review (separate agent, two rounds, 28 Oct) - all points fixed in theory.tex
Fixed: z* near 0 is NOT explained by Cor 5 (z* = -Q_E^+ P b moves with Pi c; trained ||Pi c|| = 0.03-0.12 vs ||c_perp|| = 0.13-0.90);
kink ellipse = where the LARGEST gain (sigma_tan, n >= 2) is within sqrt 2 of peak, radial gain 0.35-0.5 of peak on rho = D;
Cor 3(c) stated as containment in {rho <= sqrt3 D} (equality for n >= 2); band: wrong-sign below band only if Jx_hat >= 1;
Thm 2(c) factor is a polynomial in N and rollout maxima; new Thm 2(e): for the segment achieving Lambda,
|p_k1| >= e^Lambda (|p_k2| - sum |chi_k|) (every suffix of a maximal segment has gain >= 1); Cor 6 (i) uses (e) only for the
maximal segment, (ii) needs the skip slope w_d^T E in the band, (iii) assumes a positive exponent of the landed orbit;
Remark 5: (a), (c) carry over to the augmented GRU loop, (b), (d), (e) are scalar -> Lambda* with the direct gain is a statistic
suggested by the theorem, not a consequence. Renamed: Delta = rho^2 + D^2, J_phi (Jacobian of phi), mathcal G (branch on the
image), L_g, segment [k_1, k_2), segment length T.

## Predictions to pre-register before the architecture runs (E-A1, E-A2)
- Linear DPC (2 params) through LN MSE 3000: not trapped on any of the 30 runs (memoryless Lambda* <= 8.7 everywhere).
  OUTCOME (Run 5b): held, 0/180 trapped. The trained-oracle version (P2', Lambda* up to 93) failed: 12 false alarms.
- MLP DPC: compute Lambda* of the MLP reference controller (trained through the true plant) BEFORE looking at the MLP DPC results;
  prediction = trapped iff Lambda* > 12.1. OUTCOME (Run 5a): failed. 40 MLP traps (27 LN MSE 3000, 12 slope from 0,
  1 MSE 12000), none flagged; the MLP oracle's gradient is tame on the trapped runs. Two-stage, no norm and static norm: 0 of 30
  MLP runs trapped or failed each (median gaps 0.06, 0.61, 0.12 %).

## Slide 5 ("2-4 half-widths") - recomputed 2026-10-28 (arm_reach.py, 641 x 801 grid of the data box, wrong-sign region
connected to the centre, no edge artefacts): farthest reach 1.7, 2.8, 2.9, 3.2, 4.4, 4.7, 5.5, 6.8 (seeds 6, 0, 4, 5, 3, 7, 8, 9),
9.5 (seed 1), 32.6 (seed 2); median 4.6; 56-99 % of the wrong-sign area outside the ellipse. Along the setpoint line u = -3x
wrong-sign setpoint gains stop at d ~ 4 (slide 12) - consistent, since the arms run mostly along u ~ 0.
Recommended wording (offered 28 Oct, decision pending): big number "2–7", caption "half-widths: how far the wrong-sign arms
reach on 8 of 10 seeds (10 and 33 on the two narrowest)". Notes: "The wrong-sign arms reach 2 to 7 half-widths from the
centre on 8 seeds, and 10 and 33 on the two narrowest kinks (seeds 1 and 2); 56 to 99 % of the wrong-sign area lies outside the
ellipse. Along the setpoint line u = -3x the wrong-sign setpoint gains stop at about d = 4 (slide 12), because the arms run
mostly along u = 0." Alternative big number: "56–99 %" of the wrong-sign area outside the kink ellipse.
