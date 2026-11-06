# How to reproduce

The runs used Databricks, so the scripts are Databricks source-format notebooks. They run elsewhere with small changes, described below. This page covers the environment, the data, training a student, tracking and scoring.

## Environment

The runs used Python 3.12.3 on DBR 17.3 LTS ML GPU, with one A10 vGPU for the benchmark and one A100 for training. The Edinburgh student was trained on an A10. bf16 is used throughout, so you need an Ampere GPU or newer.

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the packages as they ran. It has not been installed as a whole outside Databricks.

**Outside Databricks.** Lines that start with `# MAGIC` or `# COMMAND` are notebook markers. Most scripts read `dbutils` widgets and secrets, for example `dbutils.secrets.get("hf", "token")`. Turn a widget into an argument, and take the token from your Hugging Face login. Storage paths at the top of the calf scripts are placeholders, so set them to yours.

**Paths.** Benchmark code finds everything through a small registry:

```bash
cp benchmark/paths_example.yaml my_paths.yaml      # then set the three roots
export CALF_PATHS=$PWD/my_paths.yaml
cd benchmark/core
python -c "import farm_paths; print(farm_paths.load().path('datasets.pigtrack.root'))"
```

Copy the files in [`benchmark/config/`](../benchmark/config/) to `paper1_distillation/protocol/` under your storage root. They hold the Hugging Face commit pins, the detector threshold and the EfficientSAM3 file details.

## Data

| You want to work with | You need |
|:---|:---|
| Any student | Access to the gated [`facebook/sam3`](https://huggingface.co/facebook/sam3) repository |
| PigTrack | [PigTrack](https://doi.org/10.25625/P7VQTP) (CC BY 4.0), staged as sequences with seqmaps and a split file. The staging script is not in this repository |
| Edinburgh | The [Edinburgh pig dataset](https://homepages.inf.ed.ac.uk/rbf/PIGDATA/) (CC BY-NC), for non-commercial work |
| Calves | The calf dataset, which is on hold while people in the frames are masked. The calf reference masks and window files are not released either, so calf tracking and scoring cannot be rerun yet |

## Train a student

Training has three steps: store SAM 3 targets, mine extra frames if you want them, then train. The target and mining scripts run on an A100 unless noted.

| Student | Targets | Extra frames | Training |
|:---|:---|:---|:---|
| Edinburgh | `notebooks/arxiv/SAM3_Baseline___feature_extraction.ipynb` | | `notebooks/arxiv/Student_TeacherDecoder_TinyViT_cleaned.ipynb` |
| Edinburgh (mined) | as above | `measure_edinburgh_hard_frames.py` (A10), `edinburgh_frame_similarity.py`, `select_targets_edinburgh.py` (both CPU) | `distill_edinburgh_mined.py` |
| PigTrack | `extract_teacher_targets_pigtrack.py` | | `distill_student_pigtrack.py` |
| PigTrack (mined) | as above | `mine_targets_pigtrack.py` | `distill_pigtrack_mined.py` |
| Calf (night-mined) | `extract_teacher_targets.py` | `mine_targets.py` | `distill_student.py` |

Scripts without a folder are in [`distillation/`](../distillation/README.md). Recipe and recorded training times are in [RESULTS.md](RESULTS.md#training-details). To apply this to your own farm, the calf scripts are the simplest starting point: they read a folder of videos, so set the storage paths, the day list and the file name pattern at the top.

The distillation helpers have tests that run on a laptop with numpy:

```bash
cd distillation
PYTHONPATH=../benchmark/core python test_distill_core.py
```

## Track

[`benchmark/core/track_bench.py`](../benchmark/core/track_bench.py) runs one model on one dataset split and writes MOT files and RLE masks. Set the `arm` widget to a model key from the [benchmark README](../benchmark/README.md). `protocol=A` seeds tracks from reference boxes at the first frame. `protocol=D` runs an RF-DETR detector, which you train with `train_rfdetr_pig.py`.

Every run keeps 16 frames of memory and releases the frame cache. Keeping 8 frames changes the masks, so stay at 16 if you want results comparable to ours.

`efficiency_bench.py` measures speed and memory on an A10. Its 4-object rows read a calf window, so they need the calf data.

## Score

Scores come from [TrackEval](https://github.com/JonathonLuiten/TrackEval) at commit `12c8791b303e0a0b50f753af204249e622d0281a`, with py-motmetrics 1.4.0 as a cross-check. Point TrackEval at the `mot/` folder of a run and your ground truth in MOT format. The MOT box offset is 0 on both pig datasets.

Our scoring script is not in this repository, so you set up the TrackEval call yourself. The scores it produced are in [`results/scores/`](../results/scores/).
