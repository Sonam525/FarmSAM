# Checkpoints

Three student checkpoints and three RF-DETR detectors are on Hugging Face at [`Sonam5/FarmSAM`](https://huggingface.co/Sonam5/FarmSAM). GitHub holds no weights.

Each student has the same architecture: TinyViT-21M-512 trunk plus a feature-pyramid head, 40,655,408 parameters. It replaces SAM 3's image backbone. You still need SAM 3's own tracker weights from Meta.

| Student | Folder on Hugging Face | File | Bytes | SHA-256 |
|:---|:---|:---|---:|:---|
| Edinburgh student | `edinburgh_noncommercial/edinburgh_student/` | `best_backbone_student.pt` | 650,867,753 | `9686b4c0f60e02251d8c2febee2029c1e7ebe74e85fbefda7e0c5daffc2075f2` |
| PigTrack student | `pigtrack/pigtrack_student/` | `best_backbone_student_pigtrack.pt` | 650,834,744 | `c07140f2de80a5e9faf175bdb4fc071bb1064e4757b91c3eefe55e271e5da69c` |
| Calf student (night-mined) | `calf/calf_student_night_mined/` | `best_backbone_student_v2.pt` | 325,420,983 | `ebf7a771bd5ac853ec1a44f209e5efe97bbdd4436a12c758d901634e2660376d` |

## Detectors

RF-DETR detectors for the detector-started runs ([RESULTS.md](RESULTS.md#with-an-rf-detr-detector)). Each folder also holds `detector.json` with the resolution, threshold, training data and test AP.

| Detector | Folder on Hugging Face | File | Bytes | SHA-256 |
|:---|:---|:---|---:|:---|
| Calf, RF-DETR-Seg Small, 624 px | `detectors/calf_rfdetr_seg/` | `checkpoint_best_total.pth` | 136,459,491 | `163b73b1c3b0c2cb61aa70931990c1f79ddd1aa27f72da2d13dbc2606e0f7057` |
| PigTrack, RF-DETR Small, 512 px | `detectors/pigtrack_rfdetr/` | `checkpoint_best_total.pth` | 127,449,329 | `c26492df2b527e9a75a530be43319455407e5ad11839ceb6be9d9f81c8cda72e` |
| Edinburgh, RF-DETR Small, 512 px | `edinburgh_noncommercial/edinburgh_rfdetr/` | `checkpoint_best_total.pth` | 127,449,201 | `19a5d98d357f8e43583eef429a3e9b72b3a031b7525c94b3d4b687dc6d441d90` |

Load one with `rfdetr==1.10.1`: `from_checkpoint(path, resolution=512)` (624 for the calf detector). Pass the resolution: the file does not store it. `load_rfdetr_detector` in [`benchmark/core/bench_core.py`](../benchmark/core/bench_core.py) does this. The pig results used a threshold of 0.70.

## Which one to use

Pick the student distilled on the setting closest to yours: the calf student for overhead calf pens, the PigTrack student for pig pens, and the Edinburgh student for non-commercial work on that pen. For a new barn, distilling your own student gives the closest match; [REPRODUCE.md](REPRODUCE.md) shows how, and [RESULTS.md](RESULTS.md) has the scores for each student.

## What the files hold

The files are PyTorch `.pt` checkpoints and hold the student only: trunk, head, `output_scale`, `output_bias`. Two of the three are about 651 MB, because they also hold optimiser state. The calf file is about 325 MB (model and EMA).

Use the EMA weights. `load_student` in [`benchmark/core/bench_core.py`](../benchmark/core/bench_core.py) prefers `ema_shadow`, then checks the parameter count, the absence of BatchNorm and the GroupNorm train and eval match.

`.pt` is pickle, so load only files that match the hashes above. `load_student` tries `weights_only=True` first and falls back to `False` for these files.

Check a download:

```bash
sha256sum best_backbone_student_pigtrack.pt
```

## Use with SAM 3

```python
import sys
sys.path.insert(0, "benchmark/core")
import bench_core as core

student, meta = core.load_student(path, device="cuda")
```

`core.load_arm` swaps the student into `Sam3TrackerVideoModel` through `DummyBackbone`. SAM 3's neck, mask decoder, memory attention and memory encoder stay unchanged. Input frames go through `core.make_student_transform()`: resize to 1,024 × 1,024, TinyViT mean and std. The output map is `[1, 1024, 72, 72]`.

## Terms

Not MIT. All three students are released under the [SAM License](../LICENSES/SAM_LICENSE.txt), and the Hugging Face repository carries a copy. If you pass a student on, you must do so under those terms and include a copy of the licence. If you publish results from a student, you must acknowledge SAM 3.

The Edinburgh student, in `edinburgh_noncommercial/`, may also be used for non-commercial purposes only, because it was distilled on CC BY-NC frames. See [the README](../README.md#datasets-and-licences) and [NOTICE.md](../NOTICE.md).
