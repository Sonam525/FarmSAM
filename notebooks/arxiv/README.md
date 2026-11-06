# Preprint notebooks

These are the three notebooks of the preprint [arXiv 2604.27128](https://arxiv.org/abs/2604.27128), unchanged. They ran on Databricks with `dbutils` mounts and `/dbfs/` paths, and `requirements_arxiv.txt` is their original list. The trainer in the Student notebook is the code behind the published Edinburgh checkpoint.

| Notebook | What it does |
|:---|:---|
| `SAM3_Baseline___feature_extraction.ipynb` | SAM 3 baseline and teacher target extraction |
| `Student_TeacherDecoder_TinyViT_cleaned.ipynb` | TinyViT student training and first benchmark |
| `DinoV3_ViTS_Behavior_Classification.ipynb` | DINOv3 embeddings and BiLSTM behaviour classifier |

## Preprint and current numbers

The current numbers are in [docs/RESULTS.md](../../docs/RESULTS.md). The benchmark there came after the preprint and uses a fuller setup, so where the two differ, use these:

| In the preprint | Current |
|:---|:---|
| Behaviour accuracy 97.34 % | That figure used a window-level split, and 98.9 % of test windows share a crop with a training window. Under a clip-level split the accuracy is 45.56 ± 7.42 %, against a majority baseline of 56.45 %. The revised paper leaves the classifier out. |
| 92.29 / 96.15 MOTA / IDF1 | Older setup, seeded from reference boxes, reproduced on DBR 16.4. Compare these only with each other, not with the tables in RESULTS.md. |
| OWLv2 detection before SAM 3 | The evaluated Edinburgh runs were seeded with reference frame-0 boxes (80 / 80). |
| Fidelity 0.808 on held-out frames | That frame set came from training clips. Held-out test clips: 0.7178 global, 0.757 per-patch. |
| Peak VRAM 19.52 to 6.49 GB | SAM 3 was unpruned there. The memory bound comes from pruning plus frame-cache release. With both applied, the Edinburgh peaks are 2,323.3 MiB (SAM 3) and 1,752.0 MiB (student). |
| Fits a Jetson Orin NX | Measurements so far are on an A10 vGPU only. |
| Pruning to 8 outputs bounds memory | Memory is bounded by pruning together with the frame-cache release. Benchmark runs keep 16 outputs, which gives the same output as no pruning. |
| GroupNorm makes train and eval identical | This holds with `drop_path_rate = 0`. |
| "MOTP in pixels" | That value is mean TP IoU. |
| Teacher "native 1024×1024 input" | SAM 3's processor runs at 1,008². The student input is 1,024². |
