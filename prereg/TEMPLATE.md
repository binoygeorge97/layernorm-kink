# Pre-registration: <experiment name>

Date written: <YYYY-MM-DD>
Commit hash of this file (fill after committing, then commit again): <hash>
Results that exist at the time of writing: <list; must NOT include the outcomes predicted below>

## Question
<one sentence>

## Design (fixed before results)
- Code version / notebook and exact settings cell: <ARCH, ONLY, REPS, DTYPE, PLANT, roots>
- Units: <families x seeds x repeats>
- Primary outcome and its definition: <e.g. trapped = failed AND own_on_model > 1.05 x ref_on_model>
- Secondary outcomes: <...>
- Thresholds (not to be changed after seeing results): <e.g. Lambda* > 12.1>

## Predictions
| id | prediction | basis | how it is scored |
|---|---|---|---|
| P1 | | | |

Run-level predictions (if any) in `prereg/<name>_predicted.csv` (columns family, seed, rep, predicted).

## Analysis plan
<tables, tests (seed-level, Wilcoxon / bootstrap), figures>

## Outcomes (fill AFTER results; do not edit anything above this line)
| id | outcome (held / failed / partly) | numbers |
|---|---|---|
