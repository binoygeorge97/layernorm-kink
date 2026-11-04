# B01 report (analysis/b01_report.py), root ../data/run6_B01

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
| B = 0.1 | LN MSE 3000 | LQR fixed | 880 | 25.1 | 5.57 |
| B = 0.1 | LN MSE 3000 | LQR sched | 880 | 15.5 | 3.86 |
| B = 0.1 | LN MSE 12000 | LQR fixed | 880 | 1.48 | 0.909 |
| B = 0.1 | LN MSE 12000 | LQR sched | 880 | 0.795 | 0 |
| B = 0.1 | LN slope from 0 | LQR fixed | 880 | 1.36 | 0 |
| B = 0.1 | LN slope from 0 | LQR sched | 880 | 0.568 | 0.114 |
| B = 0.1 | LN two-stage | LQR fixed | 880 | 0.568 | 0 |
| B = 0.1 | LN two-stage | LQR sched | 880 | 0.341 | 0 |
| B = 0.1 | no norm | LQR fixed | 880 | 0 | 0 |
| B = 0.1 | no norm | LQR sched | 880 | 0 | 0 |
| B = 0.1 | static norm | LQR fixed | 880 | 0 | 0 |
| B = 0.1 | static norm | LQR sched | 880 | 0 | 0 |

## Setpoint law
| plant | n | coef | exponent |
|---|---|---|---|
| B = 0.01 | 440 | 0.394 | -0.879 |
| B = 0.1 | 440 | 0.0531 | -0.716 |

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
| B = 0.1 | LQR fixed | 0 | 2848 | 0.421 |
| B = 0.1 | LQR fixed | 0-2 % | 8 | 0 |
| B = 0.1 | LQR fixed | 2-10 % | 0 | nan |
| B = 0.1 | LQR fixed | > 10 % | 0 | nan |
| B = 0.1 | LQR sched | 0 | 2846 | 0.527 |
| B = 0.1 | LQR sched | 0-2 % | 2 | 0 |
| B = 0.1 | LQR sched | 2-10 % | 0 | nan |
| B = 0.1 | LQR sched | > 10 % | 8 | 12.5 |

## DPC per family
| plant | family | runs | failed | trapped | median_gap_pct | best_seed_gap_pct | trapped_seeds |
|---|---|---|---|---|---|---|---|
| B = 0.01 | LN MSE 3000 | 30 | 13 | 12 | 4.05 | 0.167 | s1r0 s2r0 s8r0 s9r0 s1r1 s2r1 s8r1 s9r1 s1r2 s2r2 s8r2 s9r2 |
| B = 0.01 | LN MSE 12000 | 30 | 0 | 0 | 0.134 | 0.0432 |  |
| B = 0.01 | LN slope from 0 | 30 | 8 | 8 | 0.131 | 0.0236 | s2r0 s9r0 s2r1 s4r1 s9r1 s2r2 s3r2 s9r2 |
| B = 0.01 | LN two-stage | 30 | 0 | 0 | 0.0401 | 0.00555 |  |
| B = 0.01 | no norm | 30 | 0 | 0 | 0.573 | 0.0859 |  |
| B = 0.01 | static norm | 30 | 0 | 0 | 0.111 | 0.0159 |  |
| B = 0.1 | LN MSE 3000 | 30 | 7 | 7 | 1.62 | 0.0796 | s1r0 s2r0 s1r1 s2r1 s1r2 s2r2 s9r2 |
| B = 0.1 | LN MSE 12000 | 30 | 0 | 0 | 0.0351 | -0.0144 |  |
| B = 0.1 | LN slope from 0 | 30 | 0 | 0 | 0.0181 | -0.000784 |  |
| B = 0.1 | LN two-stage | 30 | 0 | 0 | 0.0176 | -0.000662 |  |
| B = 0.1 | no norm | 30 | 0 | 0 | 0.0571 | 0.00155 |  |
| B = 0.1 | static norm | 30 | 0 | 0 | 0.0253 | 0.000323 |  |

## Confusion matrices
| statistic | runs | predicted | trapped | hits | misses | false_alarms | correct_rejections | hit_rate | false_alarm_rate | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| direct-gain Lambda* > 12.1 (R1, primary) | 180 | 1 | 7 | 1 | 6 | 0 | 173 | 0.143 | 0 | partly |
| augmented Lambda*_aug > 12.08 (secondary) | 180 | 0 | 7 | 0 | 7 | 0 | 173 | 0 | 0 | failed |

Misses (direct): LN MSE 3000 s1 r0 (Lambda_direct 2.55); LN MSE 3000 s2 r0 (Lambda_direct 3.14); LN MSE 3000 s1 r1 (Lambda_direct 2.07); LN MSE 3000 s2 r1 (Lambda_direct 11.45); LN MSE 3000 s1 r2 (Lambda_direct 10.12); LN MSE 3000 s9 r2 (Lambda_direct 6.45)
False alarms (direct): none
Misses (aug): LN MSE 3000 s1 r0 (Lambda_aug 2.90); LN MSE 3000 s2 r0 (Lambda_aug 4.60); LN MSE 3000 s1 r1 (Lambda_aug 2.30); LN MSE 3000 s2 r1 (Lambda_aug 6.60); LN MSE 3000 s1 r2 (Lambda_aug 9.12); LN MSE 3000 s2 r2 (Lambda_aug 7.21); LN MSE 3000 s9 r2 (Lambda_aug 6.57)
False alarms (aug): none

## Outcomes
| id | outcome | numbers |
|---|---|---|
| R1 | partly | 1 hits, 6 misses, 0 false alarms of 180 runs |
| R2 | held | 0 of 90 two-stage / no norm / static norm runs trapped (none) |
| aug rule | failed | 0 hits, 7 misses, 0 false alarms |

Decision rule: traps persist (7 trapped LN runs): the claim is not restricted to low-authority plants