# Benchmark code

Everything lives in `core/`, in one flat folder, because the scripts import each other by file name.

| File | What it does |
|:---|:---|
| `core/bench_core.py` | Shared code: model loading, the streaming tracking loop, writers, telemetry, bootstrap |
| `core/track_bench.py` | Runs one model over one dataset split |
| `core/efficiency_bench.py` | Measures size, speed and memory |
| `core/train_rfdetr_pig.py` | Trains the RF-DETR pig detectors |
| `core/es3_encoder.py` | EfficientSAM3 encoder inside SAM 3's tracker. Not under MIT: see [NOTICE.md](../NOTICE.md) |
| `core/farm_paths.py` | Reads the path registry |
| `paths_example.yaml` | Example path registry |
| `config/` | Hugging Face commit pins, the detector threshold, EfficientSAM3 file details |

The `arm` setting takes these model keys:

| Key | Model |
|:---|:---|
| `T` | SAM 3 |
| `S-Edi`, `S-Edi-mined` | Edinburgh student, Edinburgh student (mined) |
| `S-PT`, `S-PT-mined` | PigTrack student, PigTrack student (mined) |
| `S-v2` | Calf student (night-mined) |
| `T21`, `S21` | SAM 2.1-tiny, SAM 2.1-small |
| `ET` | EdgeTAM |
| `ES-L` | EfficientSAM3 encoder |

`ES-L` needs the authors' checkpoint and their `tiny_vit.py`. `config/efficientsam3_encoder.json` names both, and `extract_trunk` in `es3_encoder.py` pulls the encoder out of the checkpoint.

Setup and steps are in [docs/REPRODUCE.md](../docs/REPRODUCE.md).
