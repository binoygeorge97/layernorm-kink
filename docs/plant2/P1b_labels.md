# Task 2b P1b (2026-11-04): slope-label accuracy in three inputs

Script analysis/plant2_labels.py (out/plant2/labels_accuracy_k1280_n20000.csv). Labels from 20 000 noisy transitions
(sigma = 0.01, data key 1000 + s), local-quadratic Epanechnikov estimator, k = 1280, evaluated at the training points, seeds
0-2; error against the true Jacobian. u column relative to |B| = 0.01; state entries absolute. Median over the 3 seeds.

| plant | entry | median, all | p95, all | max, all | median, interior | p95, interior | max, interior |
|---|---|---|---|---|---|---|---|
| P1 (3 inputs) | dx2+/du (true B) | 1.39 % | 7.2 % | 26 % | 1.02 % | 3.2 % | 9.0 % |
| P1 (3 inputs) | dx1+/du (true 0; the h channel) | 1.45 % | 7.0 % | 19 % | 1.11 % | 3.4 % | 7.5 % |
| scalar (2 inputs) | dx+/du | 1.88 % | 8.8 % | 31 % | 1.77 % | 5.0 % | 9.9 % |
| P1 | state entries (4) | 1.5-3e-4 | 0.9-1.1e-3 | 2.7-3.7e-3 | | | |
| scalar | dx+/dx | 5e-4 | 1.7e-3 | 5.6e-3 | | | |
Interior = every input inside 80 % of its box half-width. Time per seed (CPU, all 20 000 points): P1 14 s, scalar 7 s.

Reading: the three-input labels are not worse than the scalar ones; they are slightly better (u column median 1.4 % vs 1.9 %
of |B|, max 26 % vs 31 %). The likely reason is the larger state box: the u-direction bandwidth in data-std units is similar,
and the noise is the same. Even the worst x1-column label (19 % of |B|, i.e. h = 0.19) is far inside the safe range of P1a
(-0.77 < h < 3.03). Conclusion: no change of data size is needed; keep 20 000 transitions and k = 1280 (as in the scalar
study). Note: plant2_core.train's default label_k was 1500 (used in the CPU QUICK test); the P1 runs will pass k = 1280.
