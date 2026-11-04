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
2026-11-03 | Task 2a.3 tooling | (next commit) | analysis/make_prereg_B01.py dry-run on the QUICK B = 0.1 smoke folder (2 seeds,
300-step surrogates) | QUICK only | pipeline works (LQR study 13 s, Lambda*, Part 2 text); the QUICK numbers (e.g. static norm
fixed-LQR 35 % failed) reflect undertrained surrogates and are not results; seen after Part 1 was committed (8ecc750), which is
unchanged | t2_test.py now loads the DPC controllers only for --line (no output change)
2026-11-03 | Task 3.2 warm start (complete) | see commit | warmstart_harness.py, 7 settings x 10 pairs, 100 epochs (harness 1 hit
the 2 h limit, global3 rerun alone) | dpc_out | broken pairs pushed off: plain 5, clip3 0, unit 0, global3 4, mom 5, batch100
4, lr1e-4 0 (of 6); healthy 0 of 28 | P2b held, P3b FAILED, P6 held (written into PREREG_opt outcomes; nothing above edited)
2026-11-03 | Task 1.3 prereg Part A (MLP) | f945a5cd587d2a3cd21423ae0fbc1ba1e21dfece | data: MyDrive/dpc_out_mlp_stageA.zip (read with the Drive connector, created
18:00) -> data/dpc_out_mlp; MLP oracle true cost 2.4969-2.5007 (30 runs); analysis/make_prereg_arch.py --mlp | - | MLP oracle
Lambda* max 7.36 (LN MSE 3000), 0 of 180 predicted trapped; median K_land 10.1-10.8 | linear part deferred (zip not yet
available); 1 run with sensitivity underflow in the exactness check (contraction), noted in the file
2026-11-03 | items 2-4 (exploratory) | see commits | mom_diagnose.py, mom_moment_test.py; lr 1e-4 1000 epochs; t2_exact.py | dpc_out |
mom: 11-498 of 1000 starts > 1e3 at the good controller, median of means exploded at epoch 1 (2e31 on s2); Adam-moment reset
does not stop the drift; lr 1e-4 x 1000 epochs keeps all 6 broken pairs (max +1.5 %); Lambda*_mem: no separation on LN MSE,
post hoc 14/20, 0/160, the 3 missed traps not caught | all labelled exploratory in the results docs
2026-11-03 | Task 1.3 Part B (linear) + Task 2a.3 | 322e353, 77d8455ebd10da88547b436f3ea277a6bad376d8 | linear oracle (Drive zip) K 16.7-17.0, cost 2.5235; Lambda* ->
12/180 predicted trapped (LN MSE s1, 2, 8, 9 x 3 reps), P2/A2 unchanged; B01 Stage A read file by file from MyDrive/dpc_out_B01
(zip too large for the connector); make_prereg_B01.py: Q1 held (LN MSE median peak 18.7 -> 2.6 x B), Q2 held (0.26 %), Q3 held
(48/48); fixed LQR LN MSE 25.1 % failed / 5.6 % catastrophic; Part 2: 1 of 180 predicted trapped (LN MSE s2 rep 2, 14.7) | - |
exactness check of t2_arch invalid under chaos (documented in PREREG_arch)

2026-11-04 | A1 oracle hashes (before Runs 5 and 6) | 1459b51 | tools/hash_params.py: SHA-256 over sorted (key, dtype, shape, bytes) of the npz arrays; oracle files fetched read-only from MyDrive today; all 9 equal to the local copies used for PREREG_arch (f945a5c, 322e353) and PREREG_B01 (77d8455) | Drive |
  dpc_out_mlp/controllers/true_plant__oracle.npz 437aeab32e68c11eaf5b55da18c6bd688091de81a757211f6d0bc273781bee5a
  dpc_out_mlp/controllers/true_plant__oracle__rep1.npz 23daf5e81f58c627ed1a77985d730afb316101688a8f36a863a62619c518d380
  dpc_out_mlp/controllers/true_plant__oracle__rep2.npz 4efd7626f0b4afdc29ed100be64b56183c6a987a1d316b5972702c193a3a0c3f
  dpc_out_linear/controllers/true_plant__oracle.npz d4c421ab4c04d36432f8bbc3d432cb51c849a400b72126966c0ae3385ad91a32
  dpc_out_linear/controllers/true_plant__oracle__rep1.npz 7e802850b28aada79721f058131b14213ae302672a6d515bd921209d6c86f44c
  dpc_out_linear/controllers/true_plant__oracle__rep2.npz 1c5cf704bbbebb1de7b3612557b2bcc829e4a101c3143e501e03b24e3c712ff1
  dpc_out_B01/controllers/true_plant__oracle.npz 64200edf62001f97b3ec35b08126a9d22e53b1a016586bb7d5b15c0781186545
  dpc_out_B01/controllers/true_plant__oracle__rep1.npz c9b0a7ac64806070ba7a18a1825e9125350ad8a18dafd1789a33b25d2cced55b
  dpc_out_B01/controllers/true_plant__oracle__rep2.npz 02717862ae1705c9cca3fede38dc155c4d559e0f13b36c0cd213dbadb22e3920
2026-11-04 | B2 exactness caveat | see commit | analysis/exactness_same_traj.py on 4 linear-oracle LN MSE runs | dpc_out | along one
stored path: one-step jvp vs m_k <= 3.7e-12 rel, sum of logs <= 1.8e-11; second rollout separates (1e-12 by steps 29-69, 1e-6 by
51-104) | addendum prereg/ADDENDUM_2026-11-04_exactness.md (committed pre-registrations untouched)
