# Disclosures

These blocks travel with the numbers they qualify. Where a table in [RESULTS.md](RESULTS.md) names a model below, the block applies.

The blocks marked "as registered" are copied by script from the project's records. Only record keys were turned into model names in the sentences of RESULTS.md; the two long blocks below keep the keys as filed. `S-PT` is the PigTrack student, `S-Edi` the Edinburgh student, `S21` SAM 2.1-small, `T` SAM 3, `M` the student under test. Codes such as `b1.6`, `J7`, `DESIGN` and `REVIEW` point to the benchmark's dated protocol log, which is not in this repository.

## EfficientSAM3 encoder

Required wherever this model appears.

- Description: EfficientSAM3 Stage-1 encoder inside SAM 3's unchanged tracker; Stage-2 memory weights unreleased as of 2026-10-08.
- Size: 30.65 M [20.62 M TinyViT trunk].
- Agreement, as registered: run on an A10 under our common bf16 protocol; agreement with the authors' code (mean mask IoU): 0.962 Edinburgh / 0.846 crowded PigTrack; we load the checkpoint before eval() so TinyViT attention biases are non-zero.
- Agreement with the authors' code as shipped: with the authors' code as shipped (eval() before the checkpoint load, TinyViT attention biases left at zero) the agreement is 0.947 Edinburgh / 0.815 crowded PigTrack.
- The pre-specified parity threshold is 0.95, so the parity gate fails on `pigtrack0055`; against the as-shipped code both values are below it.
- The extracted trunk is never redistributed.

## Calf reference

Calf identity is scored against expert-verified identity chains on masks that the night-mined calf student produced in an earlier tracking run of both days; mask agreement is measured against SAM 3. The masks are not manual segmentations, and calf-frames whose reference identity was not trusted are excluded. Neither reference, the identity chains on the student's masks or the SAM 3 masks, is independent of the SAM 3 family or of that student; which models they favour was not measured. All trackers received the same seed masks. Calf results are descriptive, carry no verdict, and measure seeded-track continuity against this reference, not identity accuracy against independent ground truth.

## SAM 2.1-small calf row

The SAM 2.1-small calf row was added after the other calf results were known, and only after it passed its own seeding gate. Scoring rules, reference, windows, seeds, code and bootstrap are unchanged. The EfficientSAM3 encoder has no calf row: it failed the pre-specified seeding gate on the night window of 22 April, where it stopped emitting 2 of the 4 mask-seeded calves within 4 frames (valid run, no harness defect). Its Edinburgh and PigTrack rows are unaffected. The Edinburgh student was not run on calves.

## 23 April dusk window

The 23 April dusk window was re-picked on 2026-10-20, because the seed of animal 0 in the first version sat on a floor patch. The new window is 20:52:13.000 to 21:22:12.8. 6,663 of its 9,000 frames (74 %) lie after 21:00, in the night phase; the pre-specified label "dusk" is kept.

## Edinburgh results

n = 4 clips, one at ceiling. Pooled values are descriptive, with no interval and no verdict. Tracks start from reference boxes (oracle-init). The Edinburgh student was distilled on clips of the same pen. Edinburgh numbers never override PigTrack verdicts.

## PigTrack mining: disclosure block D

Attached to every table and sentence that names a student with extra PigTrack frames. As registered.

<details>
<summary>Show block D</summary>

* **(a) Post hoc.** Requested 2026-10-24, after the J7 / J12 results (S-PT 73.25, S21 80.90) were known, and motivated by S-PT's shortfall. Registered before any mined target, student or test output of the new arms existed, except that the mining job J14 had already run when R2 was restored (b1.8 item 1(b)). This is **attempt 1** of a mined PigTrack student scored on these 25 test sequences; any later variant (larger M, official-val frames, a different criterion) scored on them is reported with its attempt number (DESIGN s.8 item 1; b1.6 item 13).
* **(b) Recipe mirrored from the calf student, mechanism not (b1.6 item 11(b); DESIGN s.8 item 2).** The mining procedure (hardest-first by per-position feature cosine, a night quota, a per-sequence cap, union with the base targets, training from the ImageNet initialisation) is mirrored from calf v2 in recipe, not in mechanism: calf candidates every 5 s over 30-s-spaced base targets vs candidates 0.07 - 0.2 s from a base frame; a night floor with a fill pass vs exact stratum counts; a per-sequence cap that is a ratio (floor(0.79 x eligible frames of the sequence)) rather than the calf's constant 400. The cosine criterion did not predict the calf mask collapse, and S-PT's held-out cosine is lower at night (0.669) than by day (0.694) while its test shortfall against S21 is on day sequences (70.8 vs 84.4; night 75.6 vs 77.6), so its relation to HOTA is unvalidated. Calf v2 was designed after the calf v1 night failure; its gain is confounded by data amount and had no random-frame control; the training recipe is PigTrack's own.
* **(c) No test or official-val frame before J17.** No test frame was read before J17 and no official-val frame was read at all; the night quota (pool share 35.15 %, 747 night + 1,378 day, not the calf 0.45) is a deliberate departure from mining "as well as we did for the calve v2" (the user's wording), accepted by the user, and was chosen with the J7 day / night pattern known (rationale independent of J7, independence not verifiable from the record).
* **(d) Near-duplicates and steps.** M = 2,125 is below the brief's "2 x base" because of the pool; every extra frame is within 0.2 s of a base frame and comes with 1.81 x the optimiser steps per epoch (4,751 vs 2,626 training targets).
* **(e) One run per arm; two draws; two seeds.** One training run per arm; two random draws (R, R2) with two training seeds (42 for the mined arm and for R, 43 for R2). The mined arm and R share training seed 42 and start from bit-identical weights, and run-to-run training noise is not controlled by a shared seed; R2 differs from both in the draw and in the training seed, and with the seed in the random initialisation of the layers that are not loaded from ImageNet. The bootstrap CI reflects test-sequence sampling only. The mined arm's seed variance is not estimated; each of P3a and P3b compares one mined run with one random run. A composition-matched control (R_comp) was registered in b1.6 and withdrawn by the user's decision before any job existed (b1.7); it was not run. The second random draw (R2) was withdrawn in b1.7 and restored by the user's decision in b1.8 before any training job existed. **R is not independent of M** (DESIGN s.8 item 6, last sentence): about a third of the frames of each random draw are shared with the mined set (741 of 2,125, 34.9 %, expected if the sets were independent; realised overlaps M and R 725 = 34.1 %, M and R2 732 = 34.4 %, R and R2 744 = 35.0 %). R - R2 is a single pair and does not bound the random control's draw-to-draw variance (b1.8 item 2(d), drafting recommendation).
* **(f) Test reuse and cross-job contrasts.** The same 25 test sequences were scored for every earlier arm; the contrasts with the seven original arms are cross-job, guarded by the S-PT replay gate (3 of 3 sequences byte-identical) and by the identity gate (0 differences).
* **(g) Scope.** Claims are about Protocol A (oracle-initialised) HOTA of students distilled on the same farm's nine pens, against a zero-shot SAM 2.1-small that is smaller in total parameters and faster (COST); same-farm, not unseen-farm.
* **(h) Expectation.** U3 (the ex-ante expectation that "materially worse" or "inconclusive" was the likely outcome) was not separately answered by the user; the bar was kept as frozen (b1.7 answer 3), T2 as registered, no tier added.
* **(i) Composition.** The mined set is 86 % adjacent-frame-clumped and concentrated in 6 sequences (T1 fired), and the cosine ranking is almost entirely a ranking of sequences; M versus R therefore differs in sequence composition and temporal clumping as well as in the selection rule. R_comp, which would have controlled the composition, was withdrawn by the user.
* **(j) Context item added after the first check (`result_check_fixes.md` F4 i): how hard the mined set is.** By the mining criterion itself the mined set is only moderately harder than the eligible pool: mean per-patch cosine 0.7735 against 0.7921 (about 0.9 pool SDs). "Hard-mined" must not be read as a large contrast with random frames.
* **(k) EXPLORATORY context item (post hoc, not registered; first added after the first check, `result_check_fixes.md` F4 ii; relabelled and corrected after the second review): run-to-run noise is large against the arm differences.** Per-sequence HOTA of the same sequence varies by more than the pooled differences between arms. The two sequences quoted here are the extremes, not typical ones: of the 25 test sequences, pigtrack0010 has the largest spread across the four PigTrack students (24.9 points: S-PT 66.5, rand 67.4, mined 78.2, rand2 53.3) and pigtrack0067 the third largest (12.7 points: S-PT 53.5, mined 50.6, rand 40.8, rand2 48.7; the second is pigtrack0018, 12.9). The median spread over all 25 sequences is 5.45 points (DERIVED, own count from the stored `per_sequence.csv`), against a pooled HOTA range of 2.06 over the four students; the pooled differences between arms are 1 to 2 points, and R minus R2 (+1.37, a pair that differs in draw and training seed) exceeds both mined-versus-random point estimates (-0.35, +1.02). The direction of P1 does not hinge on which run is taken: all four PigTrack students, which are four separate training runs, have CIs wholly below -3.7 against S21 (mined, P1, -8.69 [-14.04, -4.05]; rand and rand2, registered-descriptive pairs, -8.34 [-13.00, -3.85] and -9.71 [-15.27, -4.35]; the S-PT row, -7.65 [-11.44, -3.75], is the unregistered D4 reference row, so this sentence is EXPLORATORY as a whole).
* **(l) Cost footnote (first check, `result_check_fixes.md` F7).** The speed and memory figures of COST were measured on the base student S-PT and S21 (J7 / J9), not re-measured for the three new students.

</details>

## Edinburgh mining disclosure block

Attached to every table, caption and sentence that names the Edinburgh student with extra frames. As registered. No Edinburgh frame is used or shown anywhere in this repository.

<details>
<summary>Show the block</summary>

- **(a) Post hoc, attempt 1.** Requested 2026-10-29, after S-Edi 74.05 / T 75.54 / S21 75.18 on Edinburgh test and the PigTrack
  mining result (b1.10) were known; registered before any new student, selection manifest on the Volume, bench output, training
  code or scoring core exists. This is **attempt 1** for Edinburgh; any later variant (another pool, criterion or q, a retrained
  base, a random arm, a second seed) scored on the same four test clips is attempt 2, registered separately and reported with
  its attempt number.
- **(b) How the hard cases were found.** By an exploratory analysis of non-test clips (HARDCASES.md; few clips, dependent frames,
  no multiplicity correction), including looking at the hardest frames (stockperson episodes) and the val frames. The criterion
  was judged on val (HARDCASES 2.2, 3), and val is also the checkpoint-selection clip, the gate G-M clip and the fidelity
  readout clip: it is used four times and the fidelity readout is not independent. Within clips, feature agreement and tracking
  agreement correlate only moderately (frame-level rho 0.23 - 0.58 in the training clips, 0.27 in val), so the criterion's
  relation to HOTA is unvalidated, as for PigTrack (b1.10 item 10(b)).
- **(c) Test numbers.** The per-clip test numbers of S-Edi (S-Edi - T: -2.17 / -0.75 / -2.06 / -0.78) were known when the design
  was written; they entered no choice (pool = the user's decision, q and criterion = defaults fixed before this version, no
  test-based rule).
- **(d) The pool.** Three November clips of one pen. 000016 is included although it is nearly static (99.7 % static frames) and
  a near-copy of a training scene in teacher space (the user's decision, with the near-duplicate rule not applied); its
  annotation mismatch is the only one confirmed; 000009 and 000033 are unverified and their labels are not used.
- **(e) Targets.** The base and val targets are the stored 2026-02 extraction, the pool targets the 2026-10 Tol_79 extraction
  (cos_global 0.99912 between them on identical frames; effect on the student fidelity metric <= 4.1e-5 per clip).
- **(f) Test reuse and the cross-job contrast.** The same four test clips were scored for every earlier Edinburgh arm (T, S-Edi,
  T21, S21, ET, ES-L; the J4 and J4e scores), so they are not fresh. M - S-Edi is a cross-job contrast: M is tracked in a
  different job from S-Edi, guarded by the replay gate (S-Edi re-tracked on one test clip under the new core; its MOT bytes
  equal J4's) and by the identity gate (the six existing arms' values in the new score equal the J4e score `eb9f09ff`, 0
  differences); both gates are in item 5.
- **(g) Scope.** Claims are about Protocol A (oracle-initialised: the GT frame-0 boxes are the prompts) HOTA and its components
  of a student distilled on one pen and camera and scored on four test clips of the same pen and camera: same-pen, not
  unseen-pen and not unseen-farm.
- **(h) Expectation: none is registered.** The five-arm draft carried a judgement-based ex-ante note ("likely outcome: no
  detectable change", DESIGN_EDI.md 3.3); version 2 dropped it together with the retired category language, no replacement is
  registered here, and b1.6 item 10 requires the user to accept one before it is registered. RECORDED context, not a
  prediction: the three PigTrack students trained with 2,125 extra frames scored 72.22 / 72.56 / 71.20 pooled HOTA against 73.25
  for the base student (b1.10 item 4).
- **(i) Composition.** 209 of the 627 frames (a third) come from the near-static 000016. 122 of 000033's 209 frames (58 %) lie in
  frames 140 - 262, the window that holds the proxy-flagged episode (a person was seen in frame 153; the proxy is derived from
  the same cosine maps as the criterion, so the overlap with the flagged frames is partly by construction; 147 of 209, 70 %, are
  in the three longest runs, which also include 25 frames outside the window). 81 % of the selected frames have a selected
  neighbour at +-1 frame. M therefore differs from any random draw in composition and temporal clumping as well as in the
  selection rule, and no random-frame control exists to separate them.
- **(j) A modest contrast.** By the criterion itself M sits about 1.06 / 1.08 / 1.10 within-clip SDs below its clip means
  (000009 / 000016 / 000033), and in 000016 the 209 frames are a scattered subset of a 99.7 % static pile whose criterion SD is
  0.0035. "Hard-mined" must not be read as a large contrast with the other frames of the clip or with random frames.
- **(k) Training differences.** See item 7: frames, seed (S-Edi set none), GPU class / runtime / libraries, data loading and +26 %
  optimiser steps per epoch, plus a new code path and targets from two extraction environments. Edinburgh is CC BY-NC: no
  Edinburgh frame appears in any figure. Edinburgh never overrides a PigTrack verdict (BENCH section 11).
- **(l) Corrections to the analysis (REVIEW M3 and others; review F9).** The headline "which day it comes from" (eta-squared
  0.966 over 8 clips) is mostly the in-sample fit of the 4 training clips (train vs held-out 0.782). Among val / 000009 / 000033
  the criterion's eta-squared is 0.028 weighted by frame count and 0.085 with clips equal, the GT pig-cell value 0.215, so the
  pig-cell gap *among those three* is mostly within-clip. **With 000016 included it is 0.472 over the four held-out clips and
  0.477 across the three pool clips (0.015 across the two-clip pair 000009 / 000033)**: under P3 the criterion is largely a
  ranking of clips, which is why the per-clip quota is necessary (DESIGN_EDI_v2.md 1.3(d), 8.2). Feeder: strong specular glare,
  not "no visible feed" (m2). Intervals condition on the clips (m5).

</details>
