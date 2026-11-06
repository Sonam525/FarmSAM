# How to reproduce

This page says what ran, in which order, and which parts you can rerun today.

## What you can and cannot rerun

| Result | Can you rerun it? |
|:---|:---|
| Student training on PigTrack | Partly. It needs PigTrack and SAM 3 access, and you stage the dataset yourself: the staging notebook, seqmaps and split file are not in this repository |
| Student training on Edinburgh | Yes, with the Edinburgh dataset and SAM 3 access. Non-commercial data |
| Student training on calves | Not until the calf dataset hold ends |
| Tracking on PigTrack, Protocol A and D | Partly. `core_bc46ecfb` runs it after you stage PigTrack and copy [`benchmark/protocol/`](../benchmark/protocol/README.md) under your registry root. Protocol D also needs a detector from `Tol_71_rfdetr_pig_train.py` |
| Tracking on Edinburgh, Protocol A | Partly. The core that ran it is here (`core_5e705fff`), but the `Tol_63` copy that ran with it is not. Use `core_bc46ecfb`: its MOT files for SAM 3 and the Edinburgh student are byte-identical to the original run on all 4 clips. SAM 2.1 and EdgeTAM have no such byte check |
| Accuracy scores | Not from this repository: the scorer is held back (see below) |
| Calf scoring | No. It needs the calf videos and the reference masks, which are not released |
| Latency and memory | Not from this repository for the 4-object rows: they read `calf_windows_v1.json` and a calf clip, and neither is released. The numbers are for one A10 vGPU |

`requirements.txt` lists the packages as they ran. It has not been installed as a whole outside the run environment. The runs used Python 3.12.3.

## Environment

The runs used Databricks, DBR 17.3 LTS ML GPU.

- **Benchmark:** one A10 (`Standard_NV36ads_A10_v5`, a vGPU reporting `NVIDIA A10-24Q`). bf16 is used throughout, so the harness needs an Ampere GPU or newer.
- **Training:** one A100. The Edinburgh student was trained on an A10.
- **Libraries:** transformers 5.2.0, torch 2.7.0, timm 1.0.25 for training and 1.0.30 for the benchmark, SDPA attention, TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`, py-motmetrics 1.4.0, `supervision` 0.31.1 for RF-DETR.

```bash
pip install -r requirements.txt
```

Model revisions are pinned in [`benchmark/protocol/hf_revisions_v1.json`](../benchmark/protocol/hf_revisions_v1.json).

## Databricks parts

The scripts are Databricks source-format notebooks. Lines that start with `# MAGIC` or `# COMMAND` are notebook markers. Most scripts read `dbutils` widgets and secrets, for example `dbutils.secrets.get("hf", "token")`. To run elsewhere, replace those calls: a widget value becomes an argument, and the token comes from your Hugging Face login. `tol_bench_core.py` itself imports with only numpy and the standard library; torch and the rest load when they are needed.

Some hard-coded paths in `distillation/` name our storage. They are not secrets. Change them to yours.

## Paths

Benchmark code resolves every path through a registry.

1. Copy [`benchmark/paths_example.yaml`](../benchmark/paths_example.yaml) and set the three `roots`.
2. Set `CALF_PATHS` to your copy.

```bash
export CALF_PATHS=/path/to/paths.yaml
cd benchmark/core_bc46ecfb
python -c "import tol_paths; print(tol_paths.load().path('datasets.pigtrack.root'))"
```

## Keep code sets flat

Each folder under `benchmark/` is one code set, as it ran. The scripts import each other by file name, and some are recorded by hash. Do not sort them or mix files between sets. [`benchmark/CODE_MANIFEST.csv`](../benchmark/CODE_MANIFEST.csv) gives the SHA-256 of every file.

| Code set | Used for |
|:---|:---|
| `core_5e705fff` | Edinburgh Protocol A: SAM 3, Edinburgh student, SAM 2.1, EdgeTAM. The core only, for reference. See the table above |
| `core_bc46ecfb` | EfficientSAM3 encoder on Edinburgh, PigTrack Protocol A, Protocol D |
| `calf_windows` | The six calf windows. Pairs with the core and helpers of `core_bc46ecfb`. It cannot run until the window file and calf video are released |
| `efficiency` | Latency, memory, pruning factorial, end-to-end rows |
| `core_8689f125` | PigTrack mining students |
| `core_27b870bb` | Edinburgh mining student |
| `gates` | Checks before and after the runs |
| `coverage_audit` | PigTrack camera-view audit, post hoc |

## Steps

### 1. Teacher targets

| Dataset | Script | Output |
|:---|:---|:---|
| PigTrack | `distillation/Tol_66_pigtrack_teacher_targets.py` | 3,000 targets (2,626 train, 374 val) |
| Calves | `distillation/Tol_01_extract_teacher_targets.py` | 3,846 targets from 46 clips |
| Edinburgh | `notebooks/arxiv/SAM3_Baseline___feature_extraction.ipynb` | Targets of the preprint |

### 2. Student training

| Student | Script |
|:---|:---|
| Edinburgh student | `notebooks/arxiv/Student_TeacherDecoder_TinyViT_cleaned.ipynb` |
| PigTrack student, and without drop-path | `distillation/Tol_02b_distill_student_pigtrack.py` |
| Calf students | `distillation/Tol_02_distill_student.py` |
| PigTrack mined and random students | `distillation/Tol_75_pigtrack_mine_targets.py`, then `distillation/Tol_02c_distill_pigtrack_mined.py` |
| Calf night mining | `distillation/Tol_26_mine_targets.py`, then `Tol_02_distill_student.py` |
| Edinburgh mined student | `distillation/Tol_79_…`, `Tol_80_…`, `Tol_81_…`, then `Tol_02d_distill_edinburgh_mined.py` |

Recipe: AdamW, cosine schedule with warm restarts, EMA of weights (decay 0.9995), early stop after 15 epochs, batch 4 at 1,024². Recorded training times are in [RESULTS.md](RESULTS.md#1-models). Set `drop_path_rate = 0` if you need GroupNorm train and eval outputs to be identical. The PigTrack student and the Edinburgh student trained with stochastic depth on.

### 3. Tracking

`Tol_63_track_bench.py` from the matching code set runs one model on one dataset and writes MOT files and RLE masks. `core_8689f125` and `core_27b870bb` carry the same `tol_paths.py` and `tol_es3.py` as `core_bc46ecfb`. The detector for Protocol D is trained with `benchmark/gates/Tol_71_rfdetr_pig_train.py`.

Pruning runs at `k16` with the `processed_frames` pop for every model. `k8` is not lossless.

### 4. Scoring

Scoring ran through `Tol_64_score_bench.py`, with TrackEval at the pinned commit and py-motmetrics as a cross-check. The MOT xy offset is frozen at 0 on both pig datasets.

**`Tol_64_score_bench.py`, `Tol_67_calf_windows.py`, `tol_v5_core.py` and `test_tol_bench_core.py` are not in this repository.** Each needs one path line reviewed before it can be published. The stored score tables are in [`results/scores/`](../results/scores/).

### 5. Efficiency

`benchmark/efficiency/Tol_65_efficiency_bench.py`. Numbers are for one A10 vGPU at 4 objects unless a row says otherwise. Energy was not measurable.

### 6. Checks

| Check | Script |
|:---|:---|
| Harness gate | `benchmark/gates/Tol_60_bench_g1.py` |
| EfficientSAM3 parity | `benchmark/gates/Tol_61_es3_port.py` |
| Edinburgh audit, fidelity | `benchmark/gates/Tol_68_edinburgh_audit.py` |
| Independent fidelity re-check | `benchmark/gates/Tol_74_fidelity_recheck.py` |
| Behaviour leak audit (the preprint's classifier) | `benchmark/gates/Tol_69_behaviour_resplit.py` |
| Birth-frame probe | `benchmark/gates/Tol_73_protD_birth_probe.py` |
| Mining gates | `benchmark/gates/Tol_76_ptmine_gates.py`, `Tol_77_ptmine_gates.py`, `distillation/Tol_82_edimine_gates.py` |

The distillation core's unit tests run on a laptop, with numpy only:

```bash
cd distillation
PYTHONPATH=../benchmark/core_bc46ecfb python test_tol_distill_core.py
```

## Not included

- Weights: see [CHECKPOINTS.md](CHECKPOINTS.md).
- Frames, ground truth or tracks of the Edinburgh dataset. It is CC BY-NC.
- Calf reference masks and `calf_windows_v1.json`. Their release is an open question.
- `Tol_62_pigtrack_stage`, the seqmaps and the split file for PigTrack.
- The dated protocol log and the internal ledger. The tables come from that ledger through `tools/build_results.py`, which is here so you can see that nothing was retyped.
- Reverse-transfer code and results. That run is pending.
