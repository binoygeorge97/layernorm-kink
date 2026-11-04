# B01 report (analysis/b01_report.py), root ../data/dpc_out

Row check: {'rows_per_family': {'LN MSE 3000': 30, 'LN MSE 12000': 30, 'LN slope from 0': 30, 'LN two-stage': 30, 'no norm': 30, 'static norm': 30}, 'duplicates': 0, 'diag_missing': 0}

Excluded rows (not a registered family, exploratory): 0 (none)

## LQR failure rates
| plant | family | controller | n | failed_pct | catastrophic_pct |
|---|---|---|---|---|---|
| B = 0.01 | LN MSE 3000 | LQR fixed | 880 | 26.5 | 10.8 |
| B = 0.01 | LN MSE 3000 | LQR sched | 880 | 20.5 | 3.41 |
| B = 0.01 | LN MSE 12000 | LQR fixed | 880 | 6.48 | 1.14 |
| B = 0.01 | LN MSE 12000 | LQR sched | 880 | 3.52 | 0.227 |
| B = 0.01 | LN slope from 0 | LQR fixed | 880 | 5.8 | 1.36 |
| B = 0.01 | LN slope from 0 | LQR sched | 880 | 2.84 | 0 |
| B = 0.01 | LN two-stage | LQR fixed | 880 | 5.68 | 0.909 |
| B = 0.01 | LN two-stage | LQR sched | 880 | 3.52 | 0.114 |
| B = 0.01 | no norm | LQR fixed | 880 | 2.61 | 0 |
| B = 0.01 | no norm | LQR sched | 880 | 5.68 | 3.64 |
| B = 0.01 | static norm | LQR fixed | 880 | 0.568 | 0 |
| B = 0.01 | static norm | LQR sched | 880 | 0.795 | 0 |
| B = 0.01 (under test) | LN MSE 3000 | LQR fixed | 880 | 26.5 | 10.8 |
| B = 0.01 (under test) | LN MSE 3000 | LQR sched | 880 | 20.5 | 3.41 |
| B = 0.01 (under test) | LN MSE 12000 | LQR fixed | 880 | 6.48 | 1.14 |
| B = 0.01 (under test) | LN MSE 12000 | LQR sched | 880 | 3.52 | 0.227 |
| B = 0.01 (under test) | LN slope from 0 | LQR fixed | 880 | 5.8 | 1.36 |
| B = 0.01 (under test) | LN slope from 0 | LQR sched | 880 | 2.84 | 0 |
| B = 0.01 (under test) | LN two-stage | LQR fixed | 880 | 5.68 | 0.909 |
| B = 0.01 (under test) | LN two-stage | LQR sched | 880 | 3.52 | 0.114 |
| B = 0.01 (under test) | no norm | LQR fixed | 880 | 2.61 | 0 |
| B = 0.01 (under test) | no norm | LQR sched | 880 | 5.68 | 3.64 |
| B = 0.01 (under test) | static norm | LQR fixed | 880 | 0.568 | 0 |
| B = 0.01 (under test) | static norm | LQR sched | 880 | 0.795 | 0 |

## Setpoint law
| plant | n | coef | exponent |
|---|---|---|---|
| B = 0.01 | 440 | 0.394 | -0.879 |
| B = 0.01 (under test) | 440 | 0.394 | -0.879 |

## Path colour
| plant | controller | wrong_sign_share | n | failed_pct |
|---|---|---|---|---|
| B = 0.01 | LQR fixed | 0 | 2014 | 0.993 |
| B = 0.01 | LQR fixed | 0-2 % | 204 | 0.98 |
| B = 0.01 | LQR fixed | 2-10 % | 46 | 0 |
| B = 0.01 | LQR fixed | > 10 % | 24 | 0 |
| B = 0.01 | LQR sched | 0 | 2083 | 1.1 |
| B = 0.01 | LQR sched | 0-2 % | 111 | 9.01 |
| B = 0.01 | LQR sched | 2-10 % | 44 | 27.3 |
| B = 0.01 | LQR sched | > 10 % | 50 | 40 |
| B = 0.01 (under test) | LQR fixed | 0 | 2014 | 0.993 |
| B = 0.01 (under test) | LQR fixed | 0-2 % | 204 | 0.98 |
| B = 0.01 (under test) | LQR fixed | 2-10 % | 46 | 0 |
| B = 0.01 (under test) | LQR fixed | > 10 % | 24 | 0 |
| B = 0.01 (under test) | LQR sched | 0 | 2083 | 1.1 |
| B = 0.01 (under test) | LQR sched | 0-2 % | 111 | 9.01 |
| B = 0.01 (under test) | LQR sched | 2-10 % | 44 | 27.3 |
| B = 0.01 (under test) | LQR sched | > 10 % | 50 | 40 |

## DPC per family
| plant | family | runs | failed | trapped | median_gap_pct | best_seed_gap_pct | trapped_seeds |
|---|---|---|---|---|---|---|---|
| B = 0.01 | LN MSE 3000 | 30 | 13 | 12 | 4.05 | 0.167 | s1r0 s2r0 s8r0 s9r0 s1r1 s2r1 s8r1 s9r1 s1r2 s2r2 s8r2 s9r2 |
| B = 0.01 | LN MSE 12000 | 30 | 0 | 0 | 0.134 | 0.0432 |  |
| B = 0.01 | LN slope from 0 | 30 | 8 | 8 | 0.131 | 0.0236 | s2r0 s9r0 s2r1 s4r1 s9r1 s2r2 s3r2 s9r2 |
| B = 0.01 | LN two-stage | 30 | 0 | 0 | 0.0401 | 0.00555 |  |
| B = 0.01 | no norm | 30 | 0 | 0 | 0.573 | 0.0859 |  |
| B = 0.01 | static norm | 30 | 0 | 0 | 0.111 | 0.0159 |  |
| B = 0.01 (under test) | LN MSE 3000 | 30 | 13 | 12 | 4.05 | 0.167 | s1r0 s2r0 s8r0 s9r0 s1r1 s2r1 s8r1 s9r1 s1r2 s2r2 s8r2 s9r2 |
| B = 0.01 (under test) | LN MSE 12000 | 30 | 0 | 0 | 0.134 | 0.0432 |  |
| B = 0.01 (under test) | LN slope from 0 | 30 | 8 | 8 | 0.131 | 0.0236 | s2r0 s9r0 s2r1 s4r1 s9r1 s2r2 s3r2 s9r2 |
| B = 0.01 (under test) | LN two-stage | 30 | 0 | 0 | 0.0401 | 0.00555 |  |
| B = 0.01 (under test) | no norm | 30 | 0 | 0 | 0.573 | 0.0859 |  |
| B = 0.01 (under test) | static norm | 30 | 0 | 0 | 0.111 | 0.0159 |  |

## Confusion matrices
| statistic | runs | predicted | trapped | hits | misses | false_alarms | correct_rejections | hit_rate | false_alarm_rate | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| direct-gain Lambda* > 12.1 (R1, primary) | 180 | 17 | 20 | 17 | 3 | 0 | 160 | 0.85 | 0 | held |
| augmented Lambda*_aug > 12.08 (secondary) | 180 | 15 | 20 | 15 | 5 | 0 | 160 | 0.75 | 0 | held |

Misses (direct): LN slope from 0 s4 r1 (Lambda_direct 2.49); LN slope from 0 s9 r1 (Lambda_direct 3.37); LN slope from 0 s3 r2 (Lambda_direct 6.45)
False alarms (direct): none
Misses (aug): LN slope from 0 s9 r0 (Lambda_aug 9.38); LN slope from 0 s4 r1 (Lambda_aug 2.27); LN slope from 0 s9 r1 (Lambda_aug 3.84); LN slope from 0 s3 r2 (Lambda_aug 4.59); LN slope from 0 s9 r2 (Lambda_aug 7.55)
False alarms (aug): none

## Outcomes
| id | outcome | numbers |
|---|---|---|
| R1 | held | 17 hits, 3 misses, 0 false alarms of 180 runs |
| R2 | held | 0 of 90 two-stage / no norm / static norm runs trapped (none) |
| aug rule | held | 15 hits, 5 misses, 0 false alarms |

Decision rule: traps persist (20 trapped LN runs): the claim is not restricted to low-authority plants