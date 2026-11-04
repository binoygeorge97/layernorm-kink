# Addendum 2026-11-04 (b): timing wording in two earlier addenda

Times: git commit times (UTC) and the Drive modifiedTime reported by the user (CT = UTC - 6 h on 3-4 Nov 2026).
- Run 5a (MLP) dpc_results.csv reached MyDrive/dpc_out_mlp at 17:54 CT on 3 Nov (23:54 UTC); Run 5b (linear) at about 20:59 CT
  (about 02:59 UTC on 4 Nov).
- ADDENDUM_2026-11-04_lambda_robustness.md was committed at 16:09:44 UTC on 4 Nov (606b0d3), ADDENDUM_2026-11-04_linear_gain.md at
  16:11:02 UTC (3fc3459): both AFTER the Run 5 results existed on the Drive.
Corrections (the committed files are not edited):
- lambda_robustness: "Written and committed before any result file of Run 5 ... or Run 6 ... existed locally or on the Drive"
  -> "Written and committed before any Run 5 result was read, and before Run 6 began writing" (Run 6's first file in
  MyDrive/dpc_out_B01: 16:50:14 UTC on 4 Nov, after 606b0d3).
- linear_gain: "Written before any Run 5 result existed" -> "Written before any Run 5 result was read".
No Run 5 result file had been opened by me before this addendum; none of the analyses in those addenda used one. No prediction
depends on the wording. The pre-registrations themselves were committed before their runs wrote results: PREREG_arch Part A
f945a5c at 18:27 UTC on 3 Nov (before 23:54 UTC), Part B 322e353 at 21:06 UTC on 3 Nov (before about 02:59 UTC on 4 Nov),
PREREG_B01 Part 2 77d8455 on 3 Nov (before Run 6 began writing at 16:50 UTC on 4 Nov).
