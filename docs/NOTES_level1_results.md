# Level 1 results (Kaggle GPU run, 2026-10-19): Jacobian loss with estimated labels, uniform slope-check points

Source: `level1_out_backup.zip` (14 configs x 10 seeds, sigma = 0.01, lambda = 1, M = 2048, k = 1280 on every seed,
Epanechnikov). Notebook: `claude/level1/level1_jacobian_loss.ipynb` (now 19 configs: + 5 controls, see 17).
Second Kaggle run 2026-10-21 (`level1_out_backup (4).zip`): restored the 14 configs unchanged and ran the longer-training
block on 10 seeds (finding 14). It did NOT train the 5 controls: the Kaggle copy of the notebook predates them (no control
rows, no Fig. 11). Finding 17 is still a 3-seed preview.
Medians [IQR], worst = max over 10 seeds.

| config | sup J_u/B | worst | RMS J_u/B | sup J_x | flip % | LQR-unst % | rel-MSE |
|---|---|---|---|---|---|---|---|
| MSE 3000 | 18.7 [10.2, 32.2] | 201 | 0.51 | 0.32 | 0.77 | 3.7 | 6.4e-7 |
| MSE 3000+1000 | 8.6 [4.6, 31.3] | 152 | 0.23 | 0.12 | 0.29 | 1.8 | 2.9e-7 |
| MSE 6000 / 12000 (one phase) | 9.4 / 2.3 | 157 / 30 | 0.25 / 0.056 | 0.12 / 0.025 | 0.23 / 0.023 (int.) | 1.8 / 0.25 | 8.3e-8 / 2.9e-8 |
| J_x only 3000 / 4000 / two-stage | 21 / 14.2 / 2.3 | 171 / 180 / 22 | 0.23 / 0.18 / 0.09 | 0.22 / 0.14 / 0.014 | 0.16 / 0.10 / 0.04 | 1.3 / 0.8 / 0.8 | |
| J_u only 3000 / 4000 / two-stage | 15.6 / 17.1 / 1.8 | 73 / 71 / 16 | 0.15 / 0.20 / 0.09 | 0.28 / 0.28 / 0.023 | 0.05 / 0.04 / 0 | 1.1 / 0.9 / 0 | |
| J_x+J_u 3000 / 4000 / two-stage | 7.9 / 6.7 / 1.15 | 28 / 27 / 3.9 | 0.089 / 0.059 / 0.048 | 0.105 / 0.083 / 0.012 | 0.03 / 0.01 / 0 | 0.9 / 0.6 / 0 | 3.1e-7 / 3.2e-7 / 3.5e-7 |
| J_x+J_u TRUE labels 3000 | 8.0 | 29 | 0.088 | 0.105 | 0.03 | 0.9 | 2.7e-7 |
| static norm, MSE (3000 / 6000 / 12000) | 0.47 / 0.23 / 0.094 | 1.01 / 0.51 / 0.15 | 0.091 / 0.046 / 0.018 | 0.012 / 0.006 / 0.003 | 0 | 0 | 8.0e-7 / 9.6e-8 / 3.9e-8 |
| no norm, MSE (3000 / 6000 / 12000) | 0.83 / 0.42 / 0.15 | 2.2 / 0.97 / 0.28 | 0.19 / 0.079 / 0.029 | 0.020 / 0.009 / 0.004 | 0 | 0 | 8.7e-7 / 1.8e-7 / 4.2e-8 |

## Findings
1. Jacobian loss helps on average, every seed: RMS J_u 10/10 seeds better than MSE for all three variants (p = 0.002).
2. Both derivatives are needed: J_u only and J_x only are each ~3x worse than both on sup J_u (0-1/10 wins, p <= 0.01).
   Weight-matched control (17a): holds cleanly for two-stage; from scratch J_u only at lambda = 2 ~ both.
3. Schedule: two-stage training (3000 MSE + 1000 Jacobian) beats 4000 steps of Jacobian loss from scratch on sup J_u 10/10
   (median ratio 0.24, p = 0.002); 4000 vs 3000 from scratch: no difference (4/10). Train-longer MSE (with an Adam
   restart at 3000) halves the median peak; 17b shows the restart itself lowers the average gain error, the peak less reliably.
   Two-stage (4000 steps) also beats 12 000 continuous MSE steps on the peak 10/10 (ratio 0.35, p = 0.002; see 14c).
4. Two-stage training reaches the kink-free models on average (RMS J_u 0.048 vs static 0.091, no-norm 0.19; sign flips on
   3/10 seeds (max 0.08 %), unstable setpoints on 2/10 seeds (0.37 %, 1.5 %); medians 0) but not on the peak: sup J_u 1.15 vs static 0.47 (2/10 wins, p = 0.02), vs no-norm 0.83 (p = 0.23).
5. Estimated labels = true labels (median ratio 0.998, 5/10). Label RMS error 0.043 B, predicted 0.038 B.
6. Mechanism: from scratch the Jacobian loss leaves a NARROWER kink (r*_x 0.10, r*_u 0.11 vs 0.26, 0.34 under MSE):
   ~4 training points inside it, ~0.5 slope-check draws per step, vs ~40 points / ~4 draws for MSE and two-stage training.
   The two-stage training keeps the wide MSE kink, which the uniform draws can see, and flattens it. Within every config,
   narrower kink <-> larger peak (Spearman -0.78 to -0.92). This motivates Level 2 (put the checks on the kink).
7. Reproducibility: CPU rerun reproduces the Kaggle numbers to every printed digit; grouping steps by 250 or 500 changes
   nothing. But the per-seed peak from scratch depends strongly on WHICH points are drawn: with two other draw streams
   (same data, same init) the median was 7.2 and 9.7 (vs 8.0), single seeds moved 2.3 <-> 36. Two-stage training is stable
   across streams (1.27, 1.03 vs 1.15). E1 Table 1's 4.4 (scratch) is therefore within draw-to-draw variation; E1 likely
   used an older sampling call. Report scratch-Level-1 peaks with this caveat, or average over draw streams.
   (12 000-step runs do drift between GPU and CPU; see 14f.)
8. Fit is never hurt: rel-MSE for J_x+J_u 3000 is 0.54x MSE (7/10, n.s.); early in training it is 10-20x lower.
9. no-norm seed 3 fit poorly (rel-MSE 2e-3) and differs between GPU and CPU: its grey band in Fig. 1 (RMS, rel-MSE)
   is stretched by that one seed.
10. Sign flips away from the box edges (outer 5 % excluded, |x| <= 3.8, |u| <= 9.5): kink-free models 0 on every seed
   (their few flips were all at the box edges); MSE 0.86 % median; J_x+J_u 3000 0.032 %; two-stage 0 median (max 0.085 %).
   All Jacobian variants beat MSE on 10/10 seeds (p = 0.002).
11. Fig. 6 (kink width): over the 100 Jacobian-trained models, training points inside the kink vs peak error has Spearman
   rho = -0.87. Below ~10 points inside (~1 uniform slope check per step) the peak stays above the kink-free range.
12. Kink vs approximation error (from the saved 201x201 J_u maps; median [min-max] over seeds):
    crest factor (peak/RMS): LN MSE 33 [19-100], static 4.5 [3.5-6.7], no norm 4.4 [1.2-5.6], two-stage 23 [9-50];
    share of squared gain error within |x|<=0.5, |u|<=1.5 (1.9 % of box): LN MSE 93 %, static 0.3 %, no norm 2 %, two-stage 43 %;
    steepest gain change (B per input std): LN MSE 800, static 1.4, no norm 3.7, two-stage 44;
    worst point: LN at the centre every seed; static at the box edge 10/10, no norm 9/10.
13. Noise-free check (3 seeds, sigma = 0 vs 0.01): static/no-norm peaks and RMS unchanged (e.g. static 0.29/0.50/0.73 vs
    0.29/0.53/0.72 x B); LN MSE peaks unchanged (11/24/193 vs 11/24/201). So the kink-free waves are approximation error,
    not fitted noise, and the kink exists with perfect data. Longer training: see 14. Not yet run: wider net for static & no norm.
14. Longer MSE training, 10 SEEDS (Kaggle run 2026-10-21, `level1_out_backup (4).zip`, block RUN_WAVES; one continuous
    Adam phase each, 3000 / 6000 / 12000 steps). REPLACES the 3-seed local preview (seeds 0-2), which happened to contain
    2 of the 4 most stubborn seeds and wrongly suggested "the LayerNorm median peak does not fall" (24/31/25). It does fall.
    Medians [IQR] (worst seed):
      peak J_u/B   static 0.47 / 0.23 / 0.094 (max 0.15 at 12000); no norm 0.83 / 0.42 / 0.15 (max 0.28);
                   LayerNorm 18.7 / 9.4 / 2.3 [1.1, 11] (worst 201 / 157 / 30)
      RMS J_u/B    static 0.091 / 0.046 / 0.018; no norm 0.19 / 0.079 / 0.029; LayerNorm 0.51 / 0.25 / 0.056
      crest        static 4.5 / 5.0 / 4.1; no norm 4.4 / 5.0 / 4.5; LayerNorm 33 / 47 / 41 (range at 12000: 11-141)
      centre share LayerNorm 93 / 95 / 88 %; kink-free <= 1 % at every length
      LQR-unst.    LayerNorm 3.7 / 1.8 / 0.25 % (still > 0 on 5/10 seeds at 12000); kink-free 0 everywhere
      int. flips   LayerNorm 0.86 / 0.23 / 0.023 %; kink-free 0
    Per seed, LayerNorm peak 3000 -> 12000: 11.4->1.07, 24.4->20.7, 201->30.2, 9.2->1.7, 9.8->1.2, 14.3->0.95, 6.7->0.57,
    23.1->2.8, 47.8->11.0, 34.8->10.3.
    a) Training longer helps every model on every seed: LN 12000 vs 3000 peak 10/10 (median ratio 0.137, p = 0.002),
       static 10/10 (0.18). So "ordinary MSE training never fixes the kink" is too strong; say "within the usual 3000-4000
       steps MSE barely touches it; 4x longer shrinks it ~7x but never to the kink-free level".
    b) The kink keeps its EXCESS: per-seed LN/static peak ratio 37 / 42 / 37 at 3000 / 6000 / 12000 steps. LN 12000 loses
       to static 12000 and no norm 12000 on 10/10 seeds (ratios 37 and 25, p = 0.002); smallest LN peak (0.57) > largest
       kink-free peak (0.28). Shape unchanged: crest ~40 vs ~4-5, worst point at the centre 10/10 at every length
       (kink-free: box edge 10/10). The floor and the kink shrink at about the same rate; the kink stays ~40x above it.
    c) Two-stage training (4000 steps in total) beats 12000 MSE steps: peak 10/10 (ratio 0.35, p = 0.002), RMS 9/10
       (0.67, p = 0.037). Medians 1.15 vs 2.3, worst 3.9 vs 30. Three times fewer steps, better on every seed.
    d) WHICH seeds keep their kink: the narrow ones. Training points inside the kink at 3000 steps vs peak at 12000:
       Spearman -0.99 (seeds 1, 2, 8, 9: 3.5-21 points -> 10-30 x B at 12000; seeds 0, 3, 4, 5, 6: 51-258 points ->
       0.57-1.7 x B). The narrow kinks do not widen (seed 1: 6.8 -> 2.5 points; seed 2: 3.5 -> 4.1). Caveat as in 6/11:
       narrow <-> tall is partly built into the theory, and the 3000-step peak alone predicts the 12000 peak too
       (rho 0.84). Still, it is the same width story as findings 6 and 11, now with no slope loss at all: whatever the
       loss, the error signal comes from training points, and a narrow kink holds few of them.
    e) rel-MSE improves ~20x for all three norms (LN 6.4e-7 / 8.3e-8 / 2.9e-8). Isolated spikes at the last checkpoint in
       every architecture (LN seed 9 1.2e-5, no norm seed 8 9.6e-5, static seed 6 4.8e-6): the full-batch fixed-lr Adam
       spikes of 17b, not an LN effect.
    f) Hardware drift: seeds 0-2 at 12000 steps were 1.07 / 20.7 / 30.2 on the Kaggle GPU vs 1.0 / 25 / 29 in the local
       CPU preview. The 3000-step runs agree to every printed digit (finding 7); 12000-step runs drift. Use GPU numbers.
    Fig. 10 (fig10_longer_training) regenerated from these 10 seeds.
15. Kink tests on all 140 models: all 120 LayerNorm models outside the kink-free range on crest, centre share and steepness;
    worst point at the centre 10/10 in every LN configuration, 0/10 for static and no norm. Also true for the 20 longer
    LN MSE models (6000, 12000 steps): crest 11-141, worst point at the centre 10/10 at both lengths.
16. Level 2 plan additions: error-guided (RAD-style) sampling baseline; wrong-centre lens control; lens share sweep
    {0 (= Level 1), 0.25, 0.5, 0.75, 1.0} including the 100 % lens arm the user asked for; 3 draw streams per config;
    keep the Adam restart at the phase switch and report the continuous MSE run alongside (see 17b). Add a long-MSE
    baseline (12 000 steps) to the Level 2 comparisons: it is the natural "just train longer" objection (14c).
17. Controls (3-seed preview, seeds 0-2, local CPU; notebook now trains them as 5 extra configs, 19 total; NOT yet run on
    Kaggle - the 2026-10-21 run used an older notebook copy):
    a) Weight control, single-slope at lambda = 2 (same total slope weight as both at lambda = 1). Peak J_u/B per seed:
       from scratch J_x only 6.5/39.4/171 -> 3.6/22.6/92.2; J_u only 2.1/12.7/26.6 -> 2.1/9.6/25.5; both 1.6/10.9/28.2.
       two-stage J_x only 1.7/13.4/21.8 -> 1.9/7.8/10.9; J_u only 1.4/5.3/16.0 -> 1.0/3.2/30.6; both 0.67/2.7/3.8.
       RMS medians two-stage: both 0.064, J_u lam2 0.128, J_x lam2 0.143; from scratch: both 0.135, J_u lam2 0.098.
       -> two-stage: both beats each lam2 variant 3/3 seeds ("both needed" holds cleanly);
          from scratch: J_x lam2 still loses 3/3, J_u lam2 ~ both (both better 1/3) -> claim "both needed" for two-stage only.
    b) Reset control, MSE 4000 continuous (no Adam restart) vs MSE 3000+1000 (restart). Identical to step 3000.
       RMS J_u/B at 3000 -> 3250: restart 0.50/0.52/2.37 -> 0.25/0.30/2.02; continuous -> 0.43/0.47/2.34.
       rel-MSE drops ~3x in 250 steps after the restart vs ~15 % without. Peak at 4000: restart 4.9/33.5/152,
       continuous 6.3/27.3/205 (restart lower 2/3). Final RMS restart 0.18/0.35/1.24 vs 0.53/0.36/1.76 (lower 3/3).
       Isolated rel-MSE spikes at step 4000 in both arms (full-batch Adam, fixed lr) -> compare at 3750.
       -> the restart alone lowers the average gain error (warm-restart effect: v reset -> step ~ lr); the peak can go
          either way. Two-stage vs MSE 3000+1000 shares the restart, so its advantage (peak ratio 0.111, 10/10) is the slope
          loss. Reword finding 3: "train longer WITH a restart lowers average gain error; peak not reliably".
    Fig. 11 (fig11_controls). Needs the 10-seed Kaggle run to confirm.
