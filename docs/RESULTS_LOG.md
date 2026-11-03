# Results log (append only)

Format per entry: date | task | commit | what ran (script or notebook + settings) | data version | key numbers | deviations.

2026-11-03 | handoff | - | repository assembled from the claude.ai project; theory/verify_thm1.py run on the bundled
surrogates: identity errors <= 1e-14, trained surrogates max |F_reduced - F_point| 2.2e-15 | partial dpc_out | - | none

2026-11-03 | Task 0 | 1a2d759 (paths), e1d3e7f | data: missing files of MyDrive/dpc_out read with the Drive connector (dpc_results.csv,
dpc_diag.csv, surrogates.csv, 9 controller files); tools/check_data.py all pass | verify_thm1.py; t2_test.py (6 families);
t2_memoryless.py A B; t2_swap.py; lambda_all.py; make_t2_tables.py; make_t2_fig.py; companion/run_lqr_study.py | dpc_out (Drive,
3 Nov) | Lambda* 180/180 equal to lambda_star_180runs.csv (3 dp); threshold 12.11, 17/20, 0/160; memoryless max 8.746; swap
9/12; grad <= 4.63 e^Lambda*; lqr_study.csv identical (max diff 0); 26.5 / 10.8 % | gradient norms differ from earlier runs in
magnitude (seed 9 rep 0 1.9e11 vs 1.2e14), as expected (pitfall 9); make_t2_tables/fig first failed on a relative T2_OUT in my
shell command, rerun with defaults
2026-11-03 | Task 0.4 + regression | d6525aa | tools/run_quick.py on the DPC notebook, QUICK = True, CPU: handoff version
(38de1d6) 678 s, current 611 s, no errors | QUICK data | every output identical (controllers, trajectories, dpc_results,
dpc_diag, curves, surrogates) except the new 'plant'/'data_fp' keys and wall-clock fields | none
2026-11-03 | Task 1.1 | e184b32 | theory/t2_arch.py --unit (linear policy at K*, all 180 runs) | dpc_out | max |Lambda* - t2_memoryless|
2.0e-9 (round-off: k stored as K/10; tolerance set to 1e-8 after the first check, reason in the code); vs reference CSV 4.75e-4
(3-dp rounding); factorisation error <= 4.0e-8 | tolerance change documented
2026-11-03 | Task 3.1-3.3 | e184b32, c41967c (prereg) | dpc_core OPT_MODELS (global, mom, batch, lr, max_horizon), RUN_OPT block;
tools/regress_dpc.py (old vs new bit-identical for MODELS[:2] + ROBUST_MODELS, GRU and MLP); tools/test_opt_units.py (all pass);
prereg/PREREG_opt.md committed c41967c before any optimiser-baseline result | - | - | none
2026-11-03 | Task 3.2 warm start | c41967c | analysis/warmstart_harness.py --epochs 100 (local CPU, settings run one after
another) | dpc_out repeat-0 oracle | plain: broken pairs expelled 5 of 6 (LN MSE s1 kicked to 3.85 and back, s8 2.61 -> 5.25, s9,
slope s2, s9); LN MSE s2 drifts 2.64 -> 2.89 (+9.6 %, below the pre-registered 1.10 line; reference run reached 3.80); healthy
0 of 4 | broken-pair trajectories not reproducible in detail (chaotic gradients); other settings: see RESULTS_opt.md when done
2026-11-03 | Task 4 | e1d3e7f | analysis/stats.py, analysis/swap_lqr_runs.py | lqr_study (regenerated), dpc_out | 26.5 [19.7, 33.1]
/ 10.8 [4.2, 17.6]; path colours and swap rates reproduced exactly; Holm: two-stage vs no norm 0.049 -> 0.098 (not
significant); MSE 12000 vs 3000 8/10 seeds p = 0.014 with all repeats (9/10, 0.006 was repeat 0 only); Lambda* LOSO 11/12, 0
false alarms; any threshold in [8.61, 17.03) gives 17/20 and 0/160 | lqr_study extra=True adds settle_time only (other
columns identical to the default)
2026-11-03 | Task 5 | e1d3e7f, d6525aa | companions edited (arms; K vs B_hat x3; decay wording), latexmk both: 0 overfull; slide 16
figure regenerated (story_data.py, dpc_story.py) -> out/figs/S2_dpc_how_it_fails.png; out/slide_fixes.md | dpc_out | seed 1
parks at -0.581 (7 starts) / +1.71 (3); seed 2 at 1.08-1.10 (9 starts), x0 = -2.5 still cycling | slide 16 description had
no hit in the companions; B_hat = 18.3 B gives K = 3.985 (handout says 3.98)
2026-11-03 | Task 2a.1 | e1d3e7f, 8ecc750 (prereg Part 1) | plant-derived constants, surrogate and label cache guards,
PLANT/ROOT/SURROGATES_ONLY; tools/regress_plant.py all pass; PREREG_B01.md Part 1 committed before any B = 0.1 data | - | - | -
2026-11-03 | notebook smoke tests (Colab runs 1-4 emulated, QUICK, CPU) | see commit after b1 | tools/run_quick.py with --set:
MLP oracle-only, B = 0.1 surrogates-only, Run 4 (ONLY = [], robust [0, 1], RUN_OPT) on top of a fresh default folder | QUICK |
all pass after fixes | FOUND AND FIXED: (1) oracle-only runs crashed in the diagnostic cell -> stage-A runs now zip and stop
after training; (2) PRE-EXISTING BUG: run_dpc rewrote surrogates.csv with only the models of the current call, so the robust
block (default RUN_ROBUST = True) replaced the main rows; Run 4 would have done this to MyDrive/dpc_out/surrogates.csv. Now
merged (main rows kept, verified identical). Deliberate change of a default output (bug fix, ground rule 4 exception);
the Drive file was never affected (the robust block never ran on Colab). (3) D6 restricted to the main models.
Also seen (after PREREG_B01 Part 1 was committed): QUICK-mode B = 0.1 surrogates (2 seeds, 300 steps) give LN MSE median peak
|J_u/B - 1| 7.5; not a result (short training), noted for transparency.
