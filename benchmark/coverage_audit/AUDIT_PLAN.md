# PigTrack scene-coverage audit: FROZEN analysis plan

**Status: EXPLORATORY and POST HOC. Nothing here is a registered result, a claim for the paper, or a manuscript edit.**
It informs the user's choice between (a) dropping PigTrack from the paper, (b) keeping it with a seen-versus-unfamiliar-scene
analysis, (c) keeping it as a pre-specified boundary-condition result. It decides none of them.

Written by the Define step (2026-10-27). The SHA-256 of this file, of `coverage_table.csv` and of the supporting files are in
`plan_frozen.json` (a file cannot contain its own hash). After that file is written, none of the frozen files changes. The
Outcomes step applies this plan exactly; every deviation or addition is labelled **"EXPLORATORY, added after seeing outcomes"**.

---

## 0. Order of operations, blinding, and what the Define step knew

1. Signatures (Tol_78, no performance file read) -> **this Define step** (reads signatures only) -> freeze (hashes) -> Outcomes step.
2. **Blinding is incomplete and I say so.** The task text that started this step stated, before I began: the shortfall
   (S-PT minus SAM 3, HOTA) over the 25 test sequences has mean -12.9, SD 9.3, range -36.9 (pigtrack0067) to -1.1
   (pigtrack0049); SAM 2.1-small minus SAM 3 has mean -4.6, SD 2.5; the earlier pen-label-only pass gave Spearman +0.22
   (same-pen training targets) and +0.45 (night); and the pen-7 montage was already seen. So the Define step knew the identity of
   the two extreme sequences and the first-pass correlations.
   - Every threshold below comes from `controls.json` or is a declared convention fixed **before** the first tier was computed;
     none was chosen to move a named sequence.
   - After the first run I looked at the table. Under the frozen rules the worst sequence (0067) is *uncovered* (score 0.500) and the
     best (0049) is *partly covered* (score 0.704). Nothing was changed in response.
   - **The only change after first seeing the table:** two extra sensitivity variants were added (`floorL12`, because 0004 sits at
     n = 7, one above the chance ceiling; `penaware`, because four night sequences are uncovered only through an IR-versus-colour
     blind spot). Both are variants, not replacements of the primary definition, and both were added before any outcome file was opened.
3. **Files opened by this Define step (complete list).**
   - Signatures folder: `signatures/SIGNATURES.md`, `signatures/run_record.json`, `signatures/controls.json`,
     `signatures/pairs.csv`, `signatures/signatures.json`, `signatures/montage_pen1.jpg` ... `montage_pen9.jpg` (all nine viewed).
   - Notebook code (not data): `Tol_78_pigtrack_scene_signatures.py` (to read the definitions of box size, ratio, brightness), `signatures_submit.md`.
   - Reference-set definition: `pt_mining/_ref/vol/manifest.csv` (the script reads only columns `seq`, `role`; a `head` of the file
     also displayed teacher-feature mean/std columns, which are not performance), `pt_mining/_ref/vol/split.txt`, `_pt_train_seqinfo.txt`.
   - Directory listings only (file names, not opened): `coverage_audit/`, `coverage_audit/signatures/`, `pt_mining/_ref/vol/`.
   - Written by this step: `define_coverage.py`, `coverage_table.csv`, `coverage_table_variants.csv`, `coverage_constants.json`,
     `audit_analysis_lib.py`, `power_simulation.json`, `power_err.txt` (empty), this file.
   - **NOT opened:** any `per_sequence.csv`, `summary.csv`, `score.json`, contrast file or telemetry file; nothing under
     `pt_mining/scores/`, `_results/`, `j5c/`, `j7/`, `j7v/`; not `pigtrack_test_objframes.json`, not `pen_map_test25.json`, not the
     training GT files, not `spt_v2dp01_history.json`. **No outcome file was opened.** (A mistaken command to delete a scratch file at
     the filesystem root was refused by the safety check and did not run; nothing was deleted.)
4. Reference set **R** = the 31 whitelisted training sequences (manifest `role = train`). The 4 distillation-validation sequences
   (0041, 0050, 0064, 0070; checkpoint selection, no gradient) are **not** in R, and neither are the 20 official-val and 25 test sequences.

---

## 1. Validation controls and the view-match definition (from the controls alone)

Controls (`controls.json`; no performance data), RANSAC homography inliers from symmetric SIFT matches on 640-px median backgrounds:

| control | n | statistic | value |
|---|---|---|---|
| (i) split-half self-match, same view, same conditions | 80 sequences | min / p05 / p25 / median / p95 / max | **124** / 278.15 / 521 / 783 / 1224.6 / 1587 |
| (i) split-half inlier hull / image area | 80 | min / p05 | 0.561 / 0.747 |
| (ii) negative control: different pen AND different resolution | 1,116 pairs | p95 / max inliers | 4 / **5** |
| (ii') stricter: different pen, same resolution | 1,632 pairs | p95 / max inliers | 4 / **6** (3 pairs at 6) |
| (ii) negative-control inlier hull (pairs with >= 4 matches) | 227 | median / p95 / max | 0.142 / **0.349** / 0.547 |

Separation: smallest self-match 124 vs largest chance match 6 (20.7x). I checked the ten highest negative-control pairs (e.g. 0003/0053,
0007/0039, 0053/0075) on the montages: all are visibly different pens (floor, fixtures, resolution); none shares a view, so 5-6
inliers is chance, not a hidden same-view pair. All 115 pairs in the whole set with more than 6 inliers are same-pen. Of those 115,
11 fail the homography / scale / rotation test (all with <= 11 inliers and hull <= 0.241) and 16 more fail only the hull guard
(<= 24 inliers each, except 0006/0033 with 116 inliers and hull 0.333, a val-val pair that touches no reference-set match).
All 37 pairs with >= 124 inliers pass both tests (minimum hull 0.537), so geometry and hull never decide a *same* call at U.

**Numbers chosen**

| symbol | value | justification |
|---|---|---|
| L (chance ceiling) | **6** | max inliers over all 2,748 different-pen pairs (controls (ii) and (ii')) |
| U (self-match floor) | **124** | minimum split-half inliers over the 80 sequences: a pair this strong is no weaker than the weakest within-sequence, same-view self-match |
| hull guard | **>= 0.349** | negative-control p95 hull; guards against matches that sit on one overlay patch (time stamp); never binds for a pair at or above U |
| scale band | **[0.8, 1.25]** | the task's example; scale of the q->r homography |
| rotation | **<= 10 deg** | the task's example |
| homography | `h_ok` (det > 0) | pipeline flag |

**View category for a pair (q, r), r in R, r != q**
- *geometry valid* = `h_ok` and scale in [0.8, 1.25] and |rotation| <= 10 deg and hull >= 0.349
- **none**: inliers <= 6 or geometry invalid; **partial**: 6 < inliers < 124 and geometry valid; **same**: inliers >= 124 and geometry valid.
- View score `V` = 0 (none) | log(inliers/6) / log(124/6) (partial) | 1 (same).

Montage check of the resulting view calls (by eye, not a measurement): pen 3 looks like one stable camera and its test sequences match R; pen 7's
sequences do not share one view (the feeder sits at the bottom, right, top-right or top-centre depending on the sequence) and the rule gives
0067 and 0077 no match; pen 4's 0011 (white tile floor, smaller pigs) and pen 5's 0010 (closer zoom, zoom ratio 1.90) look different from their
pen-mates and get none. The calls are consistent with what the montages show; I did not check every pair by eye.

## 2. Zoom, lighting, crowding

- **Zoom ratio** = median GT box size of q (sqrt(w*h)/imHeight) divided by that of the best-matching reference; **R overall** (median over R\{q} of per-sequence medians)
  when no view matches. Tolerance tau_z = log2(1.25) = **0.322** (the same-view scale band). Check: among the 37 same-view, same-night pairs the 95th percentile of
  |log2 box ratio| is 0.319, so the tolerance is not tighter than within-view variation.
- **Lighting match** = night flags equal AND d = max(|dBrightness| / tau_b, |dSaturation| / tau_s) <= 1, with tau_b = **15.645** (0-255 scale) and tau_s = **0.0693**,
  the pooled within-night-flag SD over the 80 sequences. Check: within-view p95 differences are 12.0 (brightness) and 0.026 (saturation), inside both tolerances.
  Against R overall: the median of R's sequences with the same night flag.
- Credit shape (convention): `credit(d) = clip((3 - d)/2, 0, 1)` (full credit within 1 tolerance, zero at 3). `Z = credit(|log2 ratio| / tau_z)`; `Lt = night_equal * credit(d)`.
- **Crowding / objects: descriptive only.** Columns `objects` (distinct GT tracks), `objs_per_frame_mean`, `crowding_mean` are in the table and are not in the score or tier.

## 3. Coverage score and tier (signatures only)

- Pair coverage `c(q, r) = 0.5 V + 0.25 Z + 0.25 Lt` for r with V > 0 (Z, Lt relative to r); weights are a declared convention (view dominates).
- Typicality `c_R(q) = 0.25 Z_R + 0.25 Lt_R` (zoom and lighting against R overall).
- **Coverage score `S(q) = max( max_r c(q, r), c_R(q) )`**, in [0, 1]. Best-matching reference = argmax of V (ties: c, raw inliers, DINO cosine, name).
  `S` = 1.0 exactly for a fully covered sequence; <= 0.5 when no reference matches the view.
- **Tier**
  - **covered**: some r in R has view = same AND |log2 zoom| <= tau_z AND night flags equal AND d <= 1;
  - **uncovered**: no r in R has view category other than none;
  - **partly covered**: everything else.
- DINO nearest-reference cosine (background, and mean of 8 raw frames) is reported in the table and is **not** in the score (its separating power is not validated by a split-half control).
- Leave-one-out for the 31 R members uses R minus itself; same rules.

### Tier counts (primary definition; test = the 25 benchmark sequences)

| split | n | covered | partly covered | uncovered |
|---|---|---|---|---|
| **test** | 25 | **7** | **5** | **13** |
| official val | 20 | 4 | 5 | 11 |
| distillation-val | 4 | 0 | 1 | 3 |
| R, leave-one-out | 31 | 7 | 7 | 17 |

What "covered" looks like: among the 31 training sequences themselves only 7 have a camera-consistent, same-conditions twin in the rest of R (all in pens 3 and 9) and
17 have no match at all. **Most PigTrack views are singletons**; "covered" is a high bar, and the covered-versus-uncovered contrast is essentially "a camera view seen
elsewhere in training" versus "a view seen nowhere else".

Test sequences by tier (score; best reference; inliers):
- **covered (7)**: 0005 (pen 3; 0057; 135), 0018 (pen 8; 0024; 862), 0038 (pen 3; 0053; 422), 0042 (pen 3; 0023; 675), 0060 (pen 9; 0075; 499), 0068 (pen 1; 0063; 195), 0078 (pen 3; 0059; 173); all S = 1.000.
- **partly covered (5)**: 0004 (S 0.500; n = 7), 0021 (0.786; 34), 0030 (0.685; 25), 0049 (0.704; 24), 0074 (0.980; 110).
- **uncovered (13)**: 0002 (0.489), 0010 (0.172), 0011 (0.440), 0017 (0.500), 0040 (0.351), 0054 (0.211), 0055 (0.409), 0058 (0.375), 0065 (0.453), 0067 (0.500), 0071 (0.500), 0072 (0.410), 0077 (0.500).

Full per-sequence detail: `coverage_table.csv` (80 rows: sequence, split, pen, night, size, fps, objects, best_ref, inliers, scale, rotation_deg, zoom_ratio,
dino_nearest_ref_cos, coverage_score, tier, plus extra descriptive columns). Test score distribution: 15 distinct values, 7 at 1.0, 5 at 0.5 (ties handled by mid-ranks).

### Structure the Outcomes step must keep in mind (signature-only facts)
- Coverage is confounded with the 1080p camera: all four pen-3 (1920x1080) test sequences are covered, i.e. 4 of the 7 covered sequences; Spearman(S, is1080) = **+0.55**.
  Collinearity with night +0.17, with objects -0.12, with crowding -0.03, with fps +0.04.
- Night is a *component* of the score (a night-flag mismatch removes lighting credit), so adjusting for night in A4 removes some legitimate coverage variance.
- 25 test sequences come from 9 pens (sizes 1, 4, 4, 3, 1, 4, 4, 2, 2); sequences within a pen are not independent.

## 4. Known limits of the definition (stated before any outcome)

1. **SIFT is blind across IR versus colour.** A night query whose only same-pen references are day (or the reverse) cannot show a match; its view status is unmeasurable, not known to be new.
   Four test sequences are affected (0017, 0058, 0071, 0072; column `tier_note`). Handled by variant D2.
2. **The absolute floor U = 124 penalises low-texture backgrounds** (pen 8: 0074 reaches 110 and is "partial"); handled by the loose variant.
3. The view test uses the *background median* of a sequence; it says nothing about animal appearance, posture or crowding. The student saw about 85 frames per R sequence (2,626 / 31).
4. Weights (0.5/0.25/0.25), the credit shape and the zoom/lighting tolerances are conventions; checked only for consistency with within-view variation.
5. Box size uses GT boxes of every sequence including test (descriptive statistic, not an outcome).
6. "Uncovered" does not mean "the student never saw anything similar", only "no reference sequence in R matches the camera view".
7. n = 25, no multiplicity correction anywhere in this plan; nothing here is a confirmatory test.

## 5. Outcome definitions (the Outcomes step reads the outcome files; this step did not)

- Arms: **T** = SAM 3, **S21** = SAM 2.1-small, **S-PT** = the PigTrack student. Metric: per-sequence **HOTA** on the 25 test sequences.
  If more than one file holds all three arms, the primary is `_results/j7v/score_main/per_sequence.csv` and the `pt_mining/scores/` directory with `ref_arm = T` is a cross-check; report any
  per-sequence mismatch above 0.05 HOTA points, do not pick between files after the fact. The 25 sequences must equal the test rows of `coverage_table.csv`
  (assert); no imputation; if a sequence is missing for an arm, drop it for that analysis and report n.
- **D_PT = HOTA(S-PT) - HOTA(T)** (negative = student worse). **D_21 = HOTA(S21) - HOTA(T)** (control).
- **Hypothesis H**: the student's shortfall is smaller where the training set covers the scene, i.e. coverage and D_PT are **positively** correlated
  (this is the controller's sign convention; equivalently rho <= -0.4 with the shortfall defined as T minus S-PT).
- All functions are in `audit_analysis_lib.py` (frozen; self-tested on synthetic data only). Seed 20261027, B = 10,000, percentile 95% CI, mid-ranks for ties,
  invalid (zero-variance) resamples excluded and counted.

## 6. Analyses (each states what would and would not support H)

| id | analysis | function | reading |
|---|---|---|---|
| **A1** | Spearman rho(S, D_PT), n = 25, bootstrap CI over sequences | `spearman_boot` | supports H if rho >= +0.40 and CI lower bound > 0 |
| **A2** | mean D_PT(covered, n = 7) minus mean D_PT(uncovered, n = 13), stratified bootstrap CI; partly covered (n = 5) reported descriptively | `strat_diff_boot` | supports H if the difference is > 0 with CI lower bound > 0. (Both groups have >= 3, so A2 is estimable; if a variant leaves a group with < 3, report "not estimable", do not substitute.) |
| **A3** | CONTROL: the same A1 and A2 with D_21; plus paired Delta-rho = rho(S, D_PT) - rho(S, D_21) | `spearman_boot`, `strat_diff_boot`, `paired_delta_rho_boot` | control "clearly weaker" if rho_ctrl <= 0.5 * rho_A1 AND Delta-rho >= +0.25 (point estimates); report the Delta-rho CI, and if its lower bound <= 0 say the control contrast is suggestive only. If rho_ctrl is about as large as rho_A1, coverage tracks general scene difficulty, not student-specific coverage |
| **A4** | A1 adjusted for **night** and **objects per sequence (`objects`)**: partial Spearman and standardised rank-OLS coefficient, bootstrap CI. A4b adds `is1080` (size 1920x1080) as a third covariate | `partial_rank_corr_boot` | supports H if partial rho >= +0.30. **Can** show whether the A1 sign survives two named covariates (21 residual df). **Cannot** separate coverage from anything else correlated with it (pen identity, camera, season, litter), cannot resolve effects below about 0.3, and night shares variance with the score by construction; with 4 covariates fitted on 25 points A4b is only a robustness read |
| **A5** | Sensitivity: A1, A2, A3 (rho_ctrl, Delta-rho) and A4 recomputed with **strict U = 278** (split-half p05) and **loose U = 27** (geometric mean of L and U); both from `coverage_table_variants.csv` | same | supports H if rho >= +0.30 in both. Extra, same file, reported with them: `floorL12` (chance floor 12 = 2L), `viewonly` (score = V, tier as primary), `penaware` (D2: the four sequences of limit 1 recoded partly covered; A2 only). Report all variants, none replaces the primary |
| **A6** | Post hoc, **DESCRIPTIVE** only: mean (S-PT minus S21) on the covered tier (n = 7): mean, SD, bootstrap CI; shown beside the same quantity on all 25, partly covered and uncovered | `describe_boot` | **Restricting a registered contrast to a subset afterwards is not a registered contrast.** No inference, no claim; it only shows whether the student-versus-generalist gap looks different where coverage is high |
| **D1** | Dependence: A1 with a **pen-cluster bootstrap** (9 clusters) and a **leave-one-pen-out** range of rho (this includes dropping pen 3, the 1080p camera) | `cluster_boot_rho`, `leave_one_cluster_out_rho` | descriptive; if rho changes sign or falls below +0.20 when pen 3 is dropped, report that the effect rests on one camera |

Descriptive table to report with A2: per tier, n, mean and SD of D_PT and D_21 (no test).

## 7. What "support" means in numbers

- **SUPPORTED (exploratory)**: all of: A1 rho >= +0.40 with CI lower bound > 0; A2 difference > 0 with CI lower bound > 0; A3 control clearly weaker (above);
  A4 partial rho >= +0.30; A5 rho >= +0.30 in both the strict and the loose variant.
- **NOT SUPPORTED**: A1 rho < +0.20, or the A1 95% CI upper bound < +0.40 (the data then do not fit an effect of the size set as the support bar).
- **INCONCLUSIVE**: anything else. State which of the conditions held. In particular rho between +0.20 and +0.40, or A1 met but the control as large as the main result.
- None of the three labels is a verdict on whether PigTrack is a fair benchmark, and none is a claim for the paper.

### What n = 25 can and cannot resolve (ESTIMATES, synthetic data, `power_simulation.json`)

Latent bivariate-normal simulation, n = 25, real coverage-score tie structure, 400 simulations x 1,000 bootstrap resamples per row:

| true Spearman rho | mean estimate | P(A1 support rule met) | P(CI lower > 0) | P(point >= 0.40) | P(point < 0.20) |
|---|---|---|---|---|---|
| 0.0 | -0.01 | 0.01 | 0.01 | 0.02 | 0.85 |
| 0.2 | 0.18 | 0.13 | 0.14 | 0.15 | 0.52 |
| 0.3 | 0.28 | 0.25 | 0.27 | 0.28 | 0.34 |
| 0.4 | 0.37 | 0.45 | 0.47 | 0.48 | 0.16 |
| 0.5 | 0.47 | 0.67 | 0.68 | 0.69 | 0.06 |
| 0.6 | 0.57 | 0.84 | 0.85 | 0.88 | 0.02 |
| 0.7 | 0.66 | 0.96 | 0.97 | 0.97 | 0.01 |

Reading: a real effect of rho = 0.4 meets the A1 rule less than half the time; a null effect meets it about 1 % of the time; an effect of 0.3 is called "not supported" by the
point rule about a third of the time. A failure to support is therefore weak evidence of absence; only a large coverage effect (rho of about 0.6 or more) would be detected reliably.
By a Fisher-z approximation an observed rho of 0.40 at n = 25 has a 95% CI of roughly -0.02 to +0.70.

## 8. Reporting rules for the Outcomes step

1. Report numbers exactly as computed, labelled ESTIMATES, with n, CI and the count of invalid bootstrap draws. Show every variant, including those that disagree.
2. Compare with the earlier pen-label pass (+0.22, p 0.29; night +0.45) only as context; say that the controller had already seen those numbers.
3. Any analysis, covariate, subset or threshold not listed here is labelled **"EXPLORATORY, added after seeing outcomes"**.
4. Official-val sequences have only S-Edi, S-PT and S-PT-dp0 (no T or S21): **no analysis on val is pre-specified**; any would be added after seeing outcomes.
5. No manuscript edit, no BENCH_PROTOCOL.md edit, no number to RESULTS_paper1.md. A pending section-15 text may go to `_s15_pending/` as a new file. Frames and montages are for analysis only, not for a paper figure.
6. Reading guide, not a verdict: if A1-A5 are supported, the shortfall concentrates in scenes outside the training set and a seen-versus-unfamiliar split (option b) would be informative, still post hoc on 25 sequences;
   if not supported or inconclusive, this definition of "unseen scene" does not explain the shortfall, which neither licenses nor refutes the fairness concern.

## 9. Frozen artefacts

`AUDIT_PLAN.md`, `coverage_table.csv` (primary pair), and, also frozen: `coverage_table_variants.csv`, `coverage_constants.json`, `define_coverage.py`, `audit_analysis_lib.py`, `power_simulation.json`.
Re-running `python -I define_coverage.py` reproduces the table and variants byte for byte (checked twice).
