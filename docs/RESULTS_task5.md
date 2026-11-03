# Task 5 - wording and figure fixes (2026-11-03)

## 5.1 Companions (companion/full/main.tex, companion/part1/main.tex)
grep for every phrase first; hits and changes (identical in both files):
| fix | where (full / part1 line before editing) | old | new |
|---|---|---|---|
| (a) arms | 499 / 517 | "Wrong-sign regions reach $d\approx4$ along the arms." | "The wrong-sign arms reach 2 to 7 half-widths from the centre on 8 of 10 seeds (10 and 33 on the two narrowest, seeds 1 and 2); 56 to 99 percent of the wrong-sign area lies outside the kink ellipse. Along the setpoint line $u=-3x$ wrong-sign setpoint gains stop at about $d=4$, because the arms run mostly along $u\approx0$." |
| (b) K vs B_hat | 361 / 383 (scalar LQR gain) | "For small $R$ ... so $K$ scales like $1/\hat B$." | "An overestimated input slope lowers $K$, but far less than in proportion: $\hat B=18.3B$ gives $K=3.99$ (with $a=A$), a factor 2.6 below $K^*$, not 18. It becomes dangerous when $K<3$, where the true pole $1.03-0.01K$ reaches 1, i.e. roughly $\hat B>30B$ ($K=3$ at $\hat B=28.6B$)." |
| (b) | 553 / 571 (slide 8 intuition) | "$K$ scales like $1/\hat B$." | "$K$ has the sign of $\hat B$ and falls as $\hat B$ grows, but much more slowly than $1/\hat B$ ($\hat B=18.3B$ gives $K=3.99$ with the true $A$, a factor 2.6, not 18)." |
| (b) | 756 / 774 (H3) | "(low authority makes $K\propto1/\hat B$ hypersensitive)" | "(with low authority a wrong-signed $\hat B$ flips the sign of $K$, and an overestimated $\hat B$ lowers $K$ towards the stability limit $K=3$)" |
| (c) slide 16 panels | no hit | - | neither companion describes the slide 16 panels (they cover slides 2-14 and the DPC settings) |
| 5.4 decay | 324 / 346, 683 / 701 | "the decay is $1/d$ (log-log slope $-0.98$)"; "follows $0.49\,d^{-0.98}$" | "the decay is about $1/d$ (fitted log-log slope $-0.98$ over 440 setpoint designs, with about a decade of scatter)"; "follows about $0.49/d$ (fit $0.49\,d^{-0.98}$)" |

Not changed (consistent with the corrected facts, listed for completeness): "setpoints two to four half-widths away still see
input-slope errors of order one" (full 502, part1 520) is about setpoints, not arms, and agrees with "wrong-sign setpoint gains
to d = 4"; the equation $0.49\,d^{-0.98}$ in "The law" is introduced as a fit with its scatter.
Numbers checked with level1_core.dare_K: DARE(1.03, 0.183) = 3.985 (CLAUDE.md says 3.98; 3.99 when rounded), DARE(1.022, 0.183)
= 3.948, K = 3 at B_hat = 28.6 B.
Compiled with latexmk: `out/companion/full/main.pdf` (and part1), 0 overfull boxes, no undefined references.

## 5.2 Slide 16 figure
story/story_data.py then story/dpc_story.py (CPU, ~3 min). `out/figs/S2_dpc_how_it_fails.png` (also out/story/). Titles:
seed 1 "parks at x = -0.58 or +1.71"; seed 2 "oscillates, then parks at x = 1.1 (one start keeps cycling)"; long titles on
two lines. Verified on the saved controllers: seed 1 rests at -0.581 from starts -2.5 ... 0.4 and +1.71 from 0.8, 1.5, 2.5;
seed 2 rests at 1.08-1.10 from 9 starts after up to 26 saturated steps, and from x0 = -2.5 it still oscillates at step 200.

## 5.3 Slide fixes
`out/slide_fixes.md`: 10 exact old -> new replacements for slides 5, 8, 16, 17 (read from the deck, version 1793714113-a1c3;
the deck was not edited). Slide 17 now writes m = J_x_hat - J_u_hat K with u = -K x, names both band edges, and treats
e^(N lambda) and gradient norms as orders of magnitude (the rule fails on seed 1: e^2.4 ~ 11 vs a gradient of ~1e3).

## 5.4
Phi(theta) not computed (optional); the "-0.98" wording softened as above.
