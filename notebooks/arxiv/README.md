# Preprint notebooks

These are the three notebooks of the preprint [arXiv 2604.27128](https://arxiv.org/abs/2604.27128), unchanged. They ran on Databricks with `dbutils` mounts and `/dbfs/` paths. `requirements_arxiv.txt` is their original list. The trainer in the Student notebook is the code behind the published Edinburgh checkpoint.

| Notebook | What it does |
|:---|:---|
| `SAM3_Baseline___feature_extraction.ipynb` | SAM 3 baseline and teacher target extraction |
| `Student_TeacherDecoder_TinyViT_cleaned.ipynb` | TinyViT student training and first benchmark |
| `DinoV3_ViTS_Behavior_Classification.ipynb` | DINOv3 embeddings and BiLSTM behaviour classifier |

## What later work corrected

The benchmark in this repository replaced several preprint results. If you use a preprint number, read this first.

| Preprint statement | Current state |
|:---|:---|
| Behaviour accuracy 97.34 % | This came from a window-level split: 98.9 % of test windows share a crop with a training window. Under a clip-level split the accuracy is 45.56 ± 7.42 %, below the 56.45 % majority baseline. The classifier was removed from the revised paper. |
| 92.29 / 96.15 MOTA / IDF1 | Old protocol, seeded from reference boxes, reproduced only on DBR 16.4. See `results/edinburgh_old_protocol_bridge.csv`. These numbers are not comparable with the new tables without saying so. |
| OWLv2 detection before SAM 3 | The evaluated Edinburgh runs were seeded with reference frame-0 boxes (80 / 80). |
| Fidelity 0.808 on held-out frames | Measured on training frames. Held-out test clips: 0.7178 global, 0.757 per-patch. |
| Peak VRAM 19.52 to 6.49 GB | The teacher was unpruned. The memory bound comes from pruning plus frame-cache release. With both applied, the Edinburgh peaks are 2,323.3 MiB (SAM 3) and 1,752.0 MiB (student). |
| Fits a Jetson Orin NX | Nothing was measured on an edge device. |
| Pruning to 8 outputs bounds memory | Without the frame-cache release, memory still grows. k8 also changes masks; k16 is output-identical. |
| GroupNorm makes train and eval identical | Only with `drop_path_rate = 0`. |
| "MOTP in pixels" | That value is mean TP IoU. |
| Teacher "native 1024×1024 input" | SAM 3's processor runs at 1,008². The student input is 1,024². |
