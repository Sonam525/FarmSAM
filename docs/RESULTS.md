# Results

These are the benchmark scores for the FarmSAM students and the models they are compared with. Every table links to its CSV, and the per-sequence scores from the runs are in [`results/scores/`](../results/scores/).

**Setup.** All runs used one A10 vGPU (`NVIDIA A10-24Q`), DBR 17.3 LTS ML GPU, transformers 5.2.0, torch 2.7.0, bf16 weights and 16-frame pruning with frame-cache release.

**Terms.** HOTA, DetA, AssA, LocA, MOTA, IDF1 and IDSW come from TrackEval. Mean TP IoU is TrackEval's CLEAR MOTP: the mean IoU of matches with IoU of at least 0.5, times 100. A difference is the first model minus the second, so a difference from SAM 3 is negative when SAM 3 scores higher.

## Contents

1. [Dairy calves](#dairy-calves)
2. [Edinburgh](#edinburgh)
3. [PigTrack](#pigtrack)
4. [With an RF-DETR detector](#with-an-rf-detr-detector)
5. [Size, speed and memory](#size-speed-and-memory)
6. [Training details](#training-details)
7. [Notes](#notes)

## Dairy calves

Six 30-minute windows at 5 fps, 9,000 frames each, with 4 calves in one overhead pen. That gives 206,690 evaluable calf-frames. Every tracker receives the same seed masks. The calf student was distilled on two earlier days of the same pen; the other models use their published weights. The [Notes](#notes) explain how identity is scored.

| model | identity-correct % | difference from SAM 3 (pp) [95 % CI] |
|:---|---:|---:|
| SAM 3 | 99.80 |  |
| Calf student (night-mined) | 99.23 | −0.57 [−1.52, +0.16] |
| SAM 2.1-tiny | 98.96 | −0.84 [−2.21, −0.002] |
| SAM 2.1-small | 98.77 | −1.03 [−2.33, −0.11] |
| EdgeTAM | 91.29 | −8.51 [−14.25, −3.54] |

<sub>Data: [`results/calf_pooled.csv`](../results/calf_pooled.csv)</sub>

<sub>Paired 10-minute block bootstrap against SAM 3, B = 10,000, seed 20261014.</sub>

**By phase,** identity-correct %, then mask IoU against SAM 3.

| phase | SAM 3 | Calf student (night-mined) | SAM 2.1-tiny | SAM 2.1-small | EdgeTAM |
|:---|---:|---:|---:|---:|---:|
| day | 99.95 | 99.96 / 0.948 | 99.96 / 0.954 | 99.30 / 0.948 | 73.64 / 0.747 |
| dusk | 99.37 | 97.50 / 0.709 | 96.60 / 0.807 | 96.34 / 0.845 | 86.52 / 0.701 |
| morning | 100.00 | 100.00 / 0.965 | 100.00 / 0.960 | 100.00 / 0.964 | 100.00 / 0.953 |
| night | 100.00 | 100.00 / 0.905 | 100.00 / 0.898 | 100.00 / 0.911 | 99.88 / 0.849 |

<sub>Data: [`results/calf_by_phase.csv`](../results/calf_by_phase.csv)</sub>

**By window,** identity-correct %. Merge, fragment and coverage percentages are in the CSV.

| window | SAM 3 | Calf student (night-mined) | SAM 2.1-tiny | SAM 2.1-small | EdgeTAM |
|:---|---:|---:|---:|---:|---:|
| 0422_day | 99.95 | 99.96 | 99.96 | 99.30 | 73.64 |
| 0422_dusk | 98.86 | 99.98 | 92.39 | 91.77 | 73.14 |
| 0422_night | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 0423_morning | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 0423_dusk | 99.75 | 95.65 | 99.74 | 99.74 | 96.50 |
| 0423_night | 100.00 | 100.00 | 100.00 | 100.00 | 99.76 |

<sub>Data: [`results/calf_by_window.csv`](../results/calf_by_window.csv)</sub>

## Edinburgh

4 test clips of 600 decoded frames, with 8 pigs per clip. Tracks are seeded from reference boxes at the first frame. The Edinburgh students were distilled on clips of the same pen; the other models use their published weights.

| model | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | mean TP IoU | FP / FN |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 3 | 75.5437 | 72.95 | 79.27 | 80.00 | 96.7500 | 98.3750 | 0 | 76.44 | 312 / 312 |
| Edinburgh student | 74.0540 | 70.75 | 78.83 | 79.15 | 95.1667 | 97.5833 | 0 | 75.16 | 464 / 464 |
| Edinburgh student (mined) | 74.1332 | 70.76 | 79.03 | 79.22 | 95.2917 | 97.6458 | 0 | 75.20 | 452 / 452 |
| SAM 2.1-tiny | 74.5168 | 71.58 | 78.81 | 79.19 | 95.2917 | 97.6458 | 2 | 75.45 | 451 / 451 |
| SAM 2.1-small | 75.1813 | 72.48 | 79.04 | 79.72 | 96.3333 | 98.1667 | 0 | 76.11 | 352 / 352 |
| EdgeTAM | 71.2139 | 66.07 | 78.36 | 79.38 | 86.2604 | 93.1198 | 2 | 75.76 | 1,318 / 1,318 |
| EfficientSAM3 encoder | 58.0042 | 51.09 | 67.32 | 75.48 | 61.4479 | 79.1017 | 45 | 71.41 | 3,248 / 4,109 |

<sub>Data: [`results/edinburgh_pooled.csv`](../results/edinburgh_pooled.csv)</sub>

**Per clip, HOTA.** MOTA and IDF1 per clip are in the CSV.

| model | 000005 | 000208 | 000060 | 000078 |
|:---|---:|---:|---:|---:|
| SAM 3 | 74.37 | 81.04 | 78.25 | 66.69 |
| Edinburgh student | 72.20 | 80.28 | 76.19 | 65.91 |
| Edinburgh student (mined) | 72.66 | 80.53 | 75.57 | 65.88 |
| SAM 2.1-tiny | 72.70 | 79.57 | 77.85 | 66.39 |
| SAM 2.1-small | 74.16 | 80.12 | 78.49 | 66.01 |
| EdgeTAM | 73.61 | 64.71 | 79.57 | 65.40 |
| EfficientSAM3 encoder | 68.34 | 46.60 | 71.91 | 42.60 |

<sub>Data: [`results/edinburgh_per_clip.csv`](../results/edinburgh_per_clip.csv)</sub>

**Nine clips.** Four of these nine clips are training clips of the Edinburgh student, so this table has no student row.

| model | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | mean TP IoU |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 3 | 75.1165 | 71.72 | 80.01 | 80.52 | 93.6065 | 96.8032 | 0 | 76.83 |
| SAM 2.1-tiny | 71.3465 | 66.42 | 78.00 | 79.78 | 85.8449 | 92.2292 | 9 | 75.99 |
| SAM 2.1-small | 70.7738 | 65.69 | 77.62 | 80.13 | 84.4745 | 91.4699 | 37 | 76.45 |
| EdgeTAM | 67.8313 | 62.22 | 75.28 | 79.61 | 78.5394 | 87.8704 | 39 | 75.89 |
| EfficientSAM3 encoder | 49.0614 | 41.26 | 59.87 | 75.44 | 39.7106 | 63.7285 | 191 | 71.96 |

<sub>Data: [`results/edinburgh_nine_clips.csv`](../results/edinburgh_nine_clips.csv)</sub>

## PigTrack

25 test sequences with 8,939 frames and 180,927 boxes, 15 by day and 10 by night. Tracks are seeded from reference boxes at the first frame. The PigTrack students were distilled on training sequences from the nine pens that also appear in the test set; the other models use their published weights.

| model | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | HOTA difference from SAM 3 [95 % CI] |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 3 | 85.8720 | 83.52 | 88.63 | 90.14 | 92.85 | 95.96 | 41 |  |
| PigTrack student | 73.2508 | 67.49 | 80.28 | 84.80 | 73.59 | 85.58 | 141 | −12.62 [−16.39, −8.79] |
| PigTrack student (mined) | 72.2162 | 65.67 | 80.32 | 84.32 | 71.04 | 84.75 | 114 | −13.66 [−19.04, −8.90] |
| SAM 2.1-tiny | 79.3755 | 75.41 | 84.23 | 86.78 | 86.07 | 92.53 | 79 | −6.50 [−7.51, −5.70] |
| SAM 2.1-small | 80.9033 | 76.94 | 85.62 | 87.83 | 86.75 | 92.73 | 71 | −4.97 [−5.70, −4.32] |
| EdgeTAM | 71.1625 | 66.60 | 76.69 | 84.77 | 69.93 | 83.31 | 282 | −14.71 [−16.53, −12.69] |
| EfficientSAM3 encoder | 42.1272 | 34.21 | 53.82 | 76.20 | 9.52 | 48.75 | 926 | −43.74 [−48.28, −38.45] |

<sub>Data: [`results/pigtrack.csv`](../results/pigtrack.csv)</sub>

<sub>Paired bootstrap of the 25 sequences, B = 10,000, seed 20261014, stratified by day and night.</sub>

Other paired differences in HOTA, with 95 % CI: PigTrack student minus SAM 2.1-small, −7.65 [−11.44, −3.75]; PigTrack student minus EfficientSAM3 encoder, +31.12 [+26.07, +36.26]; PigTrack student (mined) minus PigTrack student, −1.03 [−3.21, +1.48]; PigTrack student (mined) minus SAM 2.1-small, −8.69 [−14.04, −4.05]. They are in [`results/pigtrack_differences.csv`](../results/pigtrack_differences.csv).

**Day and night.**

| model | day HOTA | day AssA | day IDF1 | night HOTA | night AssA | night IDF1 | difference from SAM 3, day | difference from SAM 3, night |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 3 | 88.39 | 90.58 | 97.10 | 83.50 | 86.63 | 94.90 |  |  |
| PigTrack student | 70.82 | 79.11 | 81.29 | 75.56 | 81.29 | 89.62 | −17.57 [−22.77, −10.52] | −7.94 [−10.02, −5.73] |
| PigTrack student (mined) | 68.36 | 78.14 | 79.41 | 75.87 | 82.16 | 89.74 | −20.03 [−27.01, −11.37] | −7.62 [−9.91, −5.57] |
| SAM 2.1-tiny | 83.63 | 86.98 | 95.22 | 75.34 | 81.12 | 90.01 | −4.76 [−5.73, −3.95] | −8.16 [−9.97, −6.87] |
| SAM 2.1-small | 84.40 | 87.54 | 95.50 | 77.61 | 83.60 | 90.13 | −3.98 [−4.59, −3.12] | −5.88 [−7.29, −4.75] |
| EdgeTAM | 74.21 | 79.04 | 85.65 | 68.27 | 74.26 | 81.11 | −14.18 [−16.86, −10.57] | −15.22 [−17.54, −12.76] |
| EfficientSAM3 encoder | 44.90 | 55.22 | 51.49 | 39.44 | 52.38 | 46.14 | −43.49 [−48.35, −36.88] | −44.06 [−51.57, −35.97] |

<sub>Data: [`results/pigtrack_day_night.csv`](../results/pigtrack_day_night.csv)</sub>

<p align="center"><img src="../figures/10_pigtrack_scatter.png" alt="Per-sequence HOTA against SAM 3" width="760"></p>

<sub>Per-sequence HOTA of the PigTrack student and SAM 2.1-small against SAM 3.</sub>

## With an RF-DETR detector

Here an RF-DETR detector finds the animals, instead of reference boxes at the first frame. New tracks start from detections, and a track that dies stays in the session, so its compute cost is included.

**PigTrack.** RF-DETR Small at 512, trained on PigTrack. The detection threshold of 0.70 was chosen on PigTrack validation with SAM 3.

| model | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | FP | FN | HOTA day | HOTA night | HOTA difference from SAM 3 [95 % CI] |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 3 | 77.9936 | 71.40 | 85.51 | 90.12 | 78.64 | 87.87 | 35 | 5,856 | 32,752 | 81.17 | 74.95 |  |
| PigTrack student | 67.4853 | 59.00 | 77.96 | 84.78 | 63.18 | 78.89 | 117 | 20,906 | 45,596 | 67.13 | 67.82 | −10.51 [−14.00, −6.82] |
| SAM 2.1-small | 73.6828 | 66.59 | 82.03 | 87.84 | 74.07 | 84.98 | 53 | 10,652 | 36,214 | 77.58 | 69.96 | −4.31 [−5.39, −3.14] |

<sub>Data: [`results/rfdetr_pigtrack.csv`](../results/rfdetr_pigtrack.csv)</sub>

**Edinburgh.** The same detector fine-tuned on Edinburgh clips. 4 test clips.

| model | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | mean TP IoU |
|:---|---:|---:|---:|---:|---:|---:|---:|
| SAM 3 | 70.8906 | 65.34 | 77.89 | 86.2552 | 92.9719 | 0 | 75.92 |
| Edinburgh student | 69.8713 | 63.84 | 77.62 | 84.9479 | 92.3040 | 0 | 74.99 |
| SAM 2.1-small | 70.3029 | 64.71 | 77.40 | 86.3958 | 93.0443 | 0 | 75.21 |

<sub>Data: [`results/rfdetr_edinburgh.csv`](../results/rfdetr_edinburgh.csv)</sub>

## Size, speed and memory

| model | backbone parameters | total parameters | GFLOPs, total | GFLOPs, backbone |
|:---|---:|---:|---:|---:|
| SAM 3 | 446,236,672 | 465,782,146 | 9,497.787 | 5,398.574 |
| Edinburgh student | 40,655,408 | 60,200,882 | 4,520.839 | 421.625 |
| PigTrack student | 40,655,408 | 60,200,882 | 4,520.839 | 421.625 |
| Calf student (night-mined) | 40,655,408 | 60,200,882 | 4,520.839 | 421.625 |
| SAM 2.1-tiny | 26,849,472 | 38,962,498 | 2,776.107 | 285.028 |
| SAM 2.1-small | 33,947,328 | 46,060,354 | 2,843.844 | 352.764 |
| EdgeTAM | 4,713,712 | 13,895,794 | 460.097 | 34.119 |
| EfficientSAM3 encoder | 30,650,496 [TinyViT trunk 20,620,416] | 50,195,970 | 4,301.782 | 202.569 |

<sub>Data: [`results/models.csv`](../results/models.csv)</sub>

GFLOPs are per frame at 4 objects. Total parameters include the tracker, so the student system (60.20 M) is larger in total than SAM 2.1-tiny (38.96 M), SAM 2.1-small (46.06 M) and EdgeTAM (13.90 M). Every student has the same architecture. The two mined students were not timed separately.

The next tables are at 4 objects. Input: the calf window of the night of 22 April, 2304×1296 frames, 32 warm-up plus 500 timed frames, decode in the loop. For students the "model span" excludes the student, which runs outside the Hugging Face graph.

**Latency, ms per frame, median [IQR].**

| model | backbone / student | neck | memory attention | memory encoder | mask decoder | model span | wall / frame | e2e fps [IQR] |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 3 | 152.46 [151.45, 153.14] | 4.31 | 108.63 | 10.42 | 5.86 | 285.72 | 307.85 | 3.2483 [3.2272, 3.2725] |
| Edinburgh student | 41.56 [41.41, 41.99] | 4.94 | 109.39 | 10.26 | 5.65 | 134.18 | 199.65 | 5.0087 [4.9538, 5.0523] |
| PigTrack student | 42.23 [41.99, 42.81] | 4.93 | 109.34 | 10.26 | 5.64 | 134.11 | 206.43 | 4.8444 [4.7991, 4.9349] |
| Calf student (night-mined) | 41.64 [41.38, 42.01] | 4.94 | 109.31 | 10.26 | 5.63 | 134.07 | 203.49 | 4.9143 [4.8539, 4.9743] |
| SAM 2.1-tiny | 12.78 [12.66, 12.91] | 1.92 | 73.32 | 8.04 | 4.60 | 104.09 | 124.98 | 8.0014 [7.9022, 8.0542] |
| SAM 2.1-small | 15.04 [14.90, 15.14] | 1.92 | 73.58 | 8.11 | 4.65 | 106.68 | 127.78 | 7.8262 [7.7553, 7.8645] |
| EdgeTAM | 10.03 [9.71, 10.68] | 2.66 | 12.91 | 6.74 | 11.15 | 55.19 | 76.65 | 13.0471 [12.7322, 13.3440] |
| EfficientSAM3 encoder | 16.95 [16.92, 16.98] | 4.91 | 108.58 | 10.26 | 5.61 | 150.36 | 211.03 | 4.7386 [4.6838, 4.7931] |

<sub>Data: [`results/efficiency_latency.csv`](../results/efficiency_latency.csv)</sub>

**Memory, MiB.** 2,500 frames, peak reset after frame 31.

| model | weights | CUDA context | steady peak allocated / reserved | session tensors | growth OLS (MiB / frame) | reserved peak + context | fits an A10 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| SAM 3 | 916.6 | 2,405.6 | 1,928.95 / 2,056 | 259.17 | −0.0055 | 4,461.6 | yes |
| Edinburgh student | 183.1 | 2,455.6 | 1,443.18 / 1,602 | 259.17 | +0.0135 | 4,057.6 | yes |
| PigTrack student | 183.1 | 2,455.6 | 1,289.31 / 1,442 | 259.17 | +0.0012 | 3,897.6 | yes |
| Calf student (night-mined) | 183.1 | 2,455.6 | 1,289.31 / 1,462 | 259.17 | −0.0032 | 3,917.6 | yes |
| SAM 2.1-tiny | 82.4 | 2,405.6 | 963.71 / 1,044 | 238.53 | −0.0022 | 3,449.6 | yes |
| SAM 2.1-small | 95.9 | 2,405.6 | 984.14 / 1,112 | 238.53 | +0.0133 | 3,517.6 | yes |
| EdgeTAM | 35.3 | 2,405.6 | 404.21 / 600 | 75.03 | +0.0102 | 3,005.6 | yes |
| EfficientSAM3 encoder | 113.3 | 2,443.6 | 1,061.21 / 1,224 | 259.17 | +0.0064 | 3,667.6 | yes |

<sub>Data: [`results/efficiency_memory.csv`](../results/efficiency_memory.csv)</sub>

<p align="center"><img src="../figures/03_session_memory.png" alt="Session pruning" width="760"></p>

<sub>What the session keeps, and the two steps that bound it.</sub>

**Concurrent streams per A10.** This is a calculation, not a measurement. Every model on SAM 3's tracker adds the same 249.95 MiB per stream, because the per-stream state belongs to the tracker.

| model | level after first pass, MiB | per stream, MiB | streams |
|:---|---:|---:|---:|
| SAM 3 | 981.01 | 249.95 | 84 |
| Edinburgh student, PigTrack student, Calf student (night-mined) | 389.45 | 249.95 | 86 |
| SAM 2.1-tiny | 124.48 | 251.24 | 87 |
| SAM 2.1-small | 138.02 | 251.24 | 87 |
| EdgeTAM | 75.93 | 89.74 | 246 |
| EfficientSAM3 encoder | 177.77 | 249.95 | 87 |

<sub>Data: [`results/efficiency_streams.csv`](../results/efficiency_streams.csv)</sub>

**End to end with the detector live.**

| run | objects | detector per call, ms | backbone / student, ms | memory attention, ms | wall / frame, ms | e2e fps median [IQR] (mean) | reserved peak + context, MiB | growth OLS, MiB / frame | fits an A10 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| RF-DETR Small + SAM 3, pigtrack0002 (985 frames) | 19 seeds → 25 | 15.25 | 160.37 | 686.56 | 977.86 | 1.0226 [1.0208, 1.0626] (1.0514) | 8,171.6 | 0.239 | not assessable |
| RF-DETR Small + PigTrack student, pigtrack0002 | 19 seeds → 28 (2 deaths) | 14.75 | 38.17 | 702.10 | 888.26 | 1.1258 [1.0893, 1.1554] (1.1492) | 8,243.6 | 0.670 | not assessable |
| calf RF-DETR detector + Calf student (night-mined), 0422_night, 2,500 frames | 4 | 33.18 | 42.82 | 110.37 | 204.47 | 4.8908 [4.6874, 4.9781] (4.7342) | 4,051.6 | −0.0037 | yes |

<sub>Data: [`results/efficiency_end_to_end.csv`](../results/efficiency_end_to_end.csv)</sub>

"Not assessable" means the sequence is too short: pigtrack0002 has 985 frames, and the growth check needs 2,000.

## Training details

Each student is a TinyViT-21M-512 trunk with a feature-pyramid head, 40,655,408 parameters. Students learn from unlabelled frames with AdamW, a cosine schedule with warm restarts, an EMA of the weights (decay 0.9995), early stopping after 15 epochs and batch size 4 at 1,024². Frames below are training plus checkpoint-selection frames. Training time is the sum of per-epoch wall time on one A100 unless stated.

| student | frames (train + selection) | epochs run / best | training time | best EMA val loss | per-patch cosine |
|:---|---:|---:|---:|---:|---:|
| Calf student (night-mined) | 10,514 (10,173 + 341) | 76 / 61 | 10.58 h (+ 3 h 46 m selection job) | 0.1973 | 0.8121 (training log) |
| Edinburgh student | 3,000 (2,400 + 600) | 90 / 75 (one A10) | not logged | 0.25042 | 0.7570 on the 4 test clips (480 frames); 0.7508 on the validation clip |
| Edinburgh student (mined) | 3,627 (3,027 + 600) | 85 / 70 (one A100; early stopping) | 4.26 h (15342.1 s; job wall 4.457 h) | 0.24652 | 0.7555 on the validation clip (600 frames); not measured on the 4 test clips |
| PigTrack student | 3,000 (2,626 + 374) | 47 / 32 (A100) | 1.95 h (7,003.4 s) | 0.3266 | 0.6807 (374 selection frames of 4 whole held-out sequences) |
| PigTrack student (mined) | 5,125 (4,751 + 374) | 42 / 27 | 2.89 h | 0.3268 | 0.6787 |

<sub>Data: [`results/training.csv`](../results/training.csv)</sub>

**Feature fidelity.** Per-patch cosine is the mean over the 72×72 positions of the 1,024-d cosine. Global cosine is the cosine of the flattened tensors. Scale ratio is std(student) / std(SAM 3).

Edinburgh student:

| frames | n | per-patch cosine (± sd) | global cosine | scale ratio | MSE |
|:---|---:|---:|---:|---:|---:|
| 4 held-out test clips | 480 | 0.7570 ± 0.0061 (bf16 0.7569) | 0.7178 | 0.9499 | 1.1071 |
| val clip 2019_11_28_000113 | 120 | 0.7508 | 0.6922 | 0.9558 | 1.1942 |
| first 20 frames of training clip 2019_11_05_000002 | 20 | 0.8173 | 0.8078 | 0.9489 | 0.7631 |

<sub>Data: [`results/fidelity_edinburgh.csv`](../results/fidelity_edinburgh.csv)</sub>

PigTrack students, on 374 selection frames of 4 whole held-out sequences:

| student | per-patch cosine (± sd) | global cosine | scale ratio | MSE | day / night per-patch |
|:---|---:|---:|---:|---:|---:|
| PigTrack student | 0.6807 ± 0.048 | 0.6077 | 0.9304 | 1.5548 | 0.6939 / 0.6691 |
| PigTrack student (mined) | 0.6787 | 0.6076 | 0.9253 | 1.5474 | 0.6875 / 0.6708 |

<sub>Data: [`results/fidelity_pigtrack.csv`](../results/fidelity_pigtrack.csv)</sub>

## Notes

- **Calf identity.** Identity is scored against expert-verified identity chains on masks from an earlier tracking run of the night-mined calf student, so the reference is not independent of that student. Mask agreement is measured against SAM 3. The scores measure seeded-track continuity. The masks are not manual segmentations, and calf-frames whose reference identity was not trusted are excluded.
- **Calf student and calf windows.** The night-mined frames were added after an earlier calf student had night tracking problems on 22 April, one of the two evaluation days. No frame of 22 or 23 April was used for training. The 23 April dusk window runs from 20:52:13.0 to 21:22:12.8, so 74 % of its frames fall after 21:00. It was picked again, by a rule that used the reference pipeline's detector, after the seed of one calf in the first pick landed on a floor patch.
- **Edinburgh.** With 4 clips there are too few for intervals, so the Edinburgh tables have none.
- **Mined students.** Each is one training run. The Edinburgh mined and base students also differ in seed, GPU and library versions, so their small gap should not be read as an effect of the extra frames.
- **EfficientSAM3 encoder.** This is the EfficientSAM3 Stage-1 encoder inside SAM 3's unchanged tracker, not the full EfficientSAM3 system: Stage-2 memory weights were unreleased as of 2026-10-08. The module has 30.65 M parameters [20.62 M TinyViT trunk]. We ran it on an A10 under our common bf16 setup. Agreement with the authors' code (mean mask IoU) is 0.962 on Edinburgh and 0.846 on crowded PigTrack. We load the checkpoint before eval(), so TinyViT attention biases are non-zero; with the authors' code as shipped, where eval() comes first and the biases stay at zero, agreement is 0.947 and 0.815. It has no calf row because it did not pass seeding: on the night window of 22 April it stopped emitting 2 of the 4 mask-seeded calves within 4 frames.
- **Memory.** The memory bound comes from pruning plus frame-cache release, not from distillation.
- **Speed.** The student's speed ratio over SAM 3 is 1.49–1.54× at 4 objects and falls with more objects: 1.10× end to end on one PigTrack sequence with 25 and 28 objects in the session. Speed and memory were measured on one A10 vGPU only. Nothing was measured on an embedded device, and energy was not available on the vGPU.
- **Attention.** Every model uses SDPA attention except EdgeTAM, whose `timm` backbone needs eager attention.
