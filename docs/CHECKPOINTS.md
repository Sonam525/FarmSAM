# Checkpoints

Nine student checkpoints are on Hugging Face at [`Sonam5/FarmSAM`](https://huggingface.co/Sonam5/FarmSAM). GitHub holds no weights.

Each student has the same architecture: TinyViT-21M-512 trunk plus a feature-pyramid head, 40,655,408 parameters. It replaces SAM 3's image backbone. You still need SAM 3's own tracker weights from Meta.

| Student | Record key | Folder on Hugging Face | File | Bytes | SHA-256 |
|:---|:---|:---|:---|---:|:---|
| Edinburgh student | `S-Edi` | `edinburgh_noncommercial/edinburgh_student/` | `best_backbone_student.pt` | 650,867,753 | `9686b4c0f60e02251d8c2febee2029c1e7ebe74e85fbefda7e0c5daffc2075f2` |
| Edinburgh student (mined) | `S-Edi-mined` | `edinburgh_noncommercial/edinburgh_student_mined/` | `best_backbone_student_edinburgh.pt` | 650,834,744 | `8df6a0f86354f4113a5713e69b972f5cef5c73f3d584f0ba74462d5721550f2f` |
| PigTrack student | `S-PT` | `pigtrack/pigtrack_student/` | `best_backbone_student_pigtrack.pt` | 650,834,744 | `c07140f2de80a5e9faf175bdb4fc071bb1064e4757b91c3eefe55e271e5da69c` |
| PigTrack student without drop-path | `S-PT-dp0` | `pigtrack/pigtrack_student_no_droppath/` | `best_backbone_student_pigtrack.pt` | 650,834,680 | `c4edf8805520b83fe4c1fe094f70287d97864e45bde341fb7a641c1fdcb777cd` |
| PigTrack student (mined) | `S-PT-mined` | `pigtrack/pigtrack_student_mined/` | `best_backbone_student_pigtrack.pt` | 650,834,744 | `b47ab0fa5801219946b88de4b2ad14f2b6ff71caf9d4a1ef6b14222177f9e920` |
| PigTrack student (random, draw 1) | `S-PT-rand` | `pigtrack/pigtrack_student_random1/` | `best_backbone_student_pigtrack.pt` | 650,834,744 | `f6da71ebe4f90278df380e5d79a4ea83ba61a234adeb5ec461ee71f433cec99f` |
| PigTrack student (random, draw 2) | `S-PT-rand2` | `pigtrack/pigtrack_student_random2/` | `best_backbone_student_pigtrack.pt` | 650,834,744 | `fbbc69d50157ae8dbba764beb4cf926c5b8ce6355c5873df5debe10e9d28631a` |
| Calf student (evenly spaced) | `S-v1` | `calf/calf_student_evenly_spaced/` | `best_backbone_student.pt` | 325,419,515 | `ae7a0ba8fd423450a5101393f2bc933facfc62581cec713cb3ea400dec8e2388` |
| Calf student (night-mined) | `S-v2` | `calf/calf_student_night_mined/` | `best_backbone_student_v2.pt` | 325,420,983 | `ebf7a771bd5ac853ec1a44f209e5efe97bbdd4436a12c758d901634e2660376d` |

The record keys match the benchmark's code and logs. The names in the first column are the ones the results use.

## Which one to use

- The **headline** PigTrack student is `S-PT`. `S-PT-dp0` is an ablation row, not the headline.
- The rows with extra frames are one training run each and carry the disclosures in [DISCLOSURES.md](DISCLOSURES.md).
- All numbers for each student are in [RESULTS.md](RESULTS.md). No result here says that one of these students is a better tracker than SAM 2.1.

## What the files hold

The files are PyTorch `.pt` checkpoints and hold the student only: trunk, head, `output_scale`, `output_bias`. Seven of the nine are about 651 MB, because they also hold optimiser state. The two calf files are about 325 MB (model and EMA).

Use the EMA weights. `load_student` in [`benchmark/core_bc46ecfb/tol_bench_core.py`](../benchmark/core_bc46ecfb/tol_bench_core.py) prefers `ema_shadow`, then checks the parameter count, the absence of BatchNorm and the GroupNorm train and eval match.

`.pt` is pickle. Load only files that match the hashes above. `load_student` tries `weights_only=True` first and falls back to `False` for these pinned files.

Check a download:

```bash
sha256sum best_backbone_student_pigtrack.pt
```

## Use with SAM 3

```python
student, meta = core.load_student(path, device="cuda")
```

`core.load_arm` swaps the student into `Sam3TrackerVideoModel` through `DummyBackbone`. SAM 3's neck, mask decoder, memory attention and memory encoder stay unchanged. Input frames go through `make_student_transform()`: resize to 1,024 × 1,024, TinyViT mean and std. The output map is `[1, 1024, 72, 72]`.

## Terms

Not MIT. All nine students are released under the [SAM License](../LICENSES/SAM_LICENSE.txt), and the Hugging Face repository carries a copy. If you pass a student on, you must do so under those terms and include a copy of the licence. If you publish results from a student, you must acknowledge SAM 3.

The two Edinburgh students, in `edinburgh_noncommercial/`, may also be used for non-commercial purposes only, because they were distilled on CC BY-NC frames. See [the README](../README.md#datasets-and-licences) and [NOTICE.md](../NOTICE.md).
