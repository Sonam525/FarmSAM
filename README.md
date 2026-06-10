# Longitudinal Behaviour Analysis with SAM3 Distillation & DINOv3-Tiny

Research code for a paper on **efficient, deployable vision models for longitudinal animal-behaviour
monitoring**. The goal is to take large foundation models — Meta's **SAM 3** (video tracker) and
**DINOv3** (self-supervised ViT) — and compress them into compact students that run on edge hardware
(e.g. NVIDIA Jetson Orin) while preserving tracking and classification quality.

The case study is **longitudinal pig-behaviour monitoring** from fixed-camera farm video, but the
pipeline is general: track animals, extract per-frame features, and classify behaviour over time.

> **Status:** research / paper-artifact code. The notebooks were developed on **Databricks** with an
> **NVIDIA A10** GPU, so they use `dbutils` mounts and `/dbfs/...` paths. They are shared for
> transparency and reproducibility of the methods — see [Running the code](#running-the-code).

---

## What's in here

| Notebook | Role | Summary |
|---|---|---|
| [`SAM3_Baseline___feature_extraction.ipynb`](SAM3_Baseline___feature_extraction.ipynb) | **Teacher baseline** | Runs SAM 3 video tracking on benchmark clips with box prompts; measures tracking quality (MOTA / IDF1), latency, throughput and GPU memory; extracts teacher **backbone + 4-level FPN features**, **binary masks** and **bbox annotations** to disk for distillation. |
| [`Student_TeacherDecoder_TinyViT_cleaned.ipynb`](Student_TeacherDecoder_TinyViT_cleaned.ipynb) | **Backbone distillation** | Distills a **TinyViT-21M-512 + FPN** student to match the SAM 3 backbone feature map `[1024, 72, 72]`. The student then drives SAM 3's *real* neck + mask decoder + memory modules, and is benchmarked head-to-head against the teacher, producing publication figures. |
| [`DinoV3_ViTS_Behavior_Classification.ipynb`](DinoV3_ViTS_Behavior_Classification.ipynb) | **Behaviour classification** | Evaluates the pre-distilled **DINOv3-ViT-S (21 M, dim 384)** as a drop-in replacement for DINOv3-ViT-7B (6.7 B, dim 4096). Extracts embeddings for cropped animal frames and trains a **BiLSTM** sequence classifier over 9 behaviours, with full metrics, confusion matrices and speed benchmarks. |

---

## Method overview

```
                     ┌─────────────────────────────────────────────┐
   Farm video  ──►   │  SAM 3 (teacher)  — track + segment animals  │
                     │  ViT-H backbone → neck (FPN) → mask decoder   │
                     └───────────────┬─────────────────────────────┘
                                     │  extract features + masks (teacher targets)
                                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │  Backbone distillation                                    │
        │  TinyViT-21M-512 + FPN  ──►  match  [1024, 72, 72]         │
        │  (GroupNorm everywhere → zero train/eval mismatch)        │
        └───────────────┬──────────────────────────────────────────┘
                        │  student backbone + SAM 3 neck/decoder
                        ▼
              Compact tracker  →  per-animal crops over time
                        │
                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │  Behaviour classification                                 │
        │  DINOv3-ViT-S embeddings  ──►  BiLSTM (seq_len=3)          │
        │  → 9 behaviour classes                                    │
        └──────────────────────────────────────────────────────────┘
```

**Key ideas**

- **Feature-level knowledge distillation.** The student learns to reproduce the teacher's intermediate
  feature maps (directional + cosine + scale losses), not just final outputs, so it can plug directly
  into the teacher's downstream neck/decoder.
- **GroupNorm student.** All BatchNorm in the TinyViT backbone is swapped for GroupNorm, eliminating
  train/eval statistics mismatch — verified to give bit-for-bit identical train/eval outputs.
- **Pre-distilled DINOv3-S for behaviour.** A 21 M ViT-S replaces a 6.7 B ViT-7B (~300× fewer params,
  embedding dim 4096 → 384) as the frame encoder feeding a lightweight temporal classifier.
- **Edge-deployment framing.** Throughput, latency and peak GPU memory are tracked throughout and
  compared against Jetson Orin NX (8 GB / 16 GB) budgets.

---

## Repository layout

```
.
├── SAM3_Baseline___feature_extraction.ipynb      # teacher baseline + feature/mask extraction
├── Student_TeacherDecoder_TinyViT_cleaned.ipynb  # TinyViT backbone distillation + benchmark + figures
├── DinoV3_ViTS_Behavior_Classification.ipynb     # DINOv3-S embeddings + BiLSTM behaviour classifier
├── requirements.txt
├── LICENSE
└── README.md
```

Each notebook is organised into clearly-titled sections (mount → install → load model →
extract/train → evaluate → figures) and runs top-to-bottom.

---

## Running the code

These notebooks were authored on **Databricks (A10 GPU)** and assume:

1. **A data store mounted at `/mnt/playbehavior`.** Cell 1 of the SAM 3 / TinyViT notebooks mounts an
   Azure Blob container via `dbutils.fs.mount`. The account name, container and key are **redacted**
   (`<STORAGE_ACCOUNT>`, `<CONTAINER>`, `<REDACTED_AZURE_STORAGE_ACCOUNT_KEY>`) — supply your own, or
   replace these cells with local paths.
2. **Hugging Face access.** SAM 3 (`facebook/sam3`) and DINOv3 weights are pulled from the Hub and may
   require accepting model terms. Insert your own token where the notebooks call
   `login(token=...)` — the committed values are intentionally blank / placeholders.
3. **Dependencies** from [`requirements.txt`](requirements.txt) (`pip install -r requirements.txt`).
   A recent `transformers` (≥ 4.56) is required for the SAM 3 and DINOv3 classes; `numpy` is pinned
   `< 2.0` for Databricks compatibility.

To run **outside Databricks**, replace the `dbutils.fs.mount(...)` cell and `/dbfs/...` path prefixes
with ordinary local directories; the rest of the logic is standard PyTorch / Hugging Face.

### Data

The pig-behaviour dataset (fixed-camera farm video, per-frame behaviour labels and tracking ground
truth) is **not included** — it is private research data. Folder names such as
`Pig Behavior Edinburgh/...` in the paths indicate the expected directory structure, not redistributable
content. The code is dataset-agnostic: point it at any tracked-video dataset with box prompts and
behaviour labels.

---

## Behaviour classes

`standing`, `lying`, `eat`, `drink`, `sitting`, `sleep`, `run`, `playwithtoy`, `nose-to-nose`

---

## Security / privacy note

All notebooks have been scrubbed of credentials before release:

- Hugging Face tokens are blank or `<REDACTED_HF_TOKEN>`.
- Azure storage account, container and key are redacted to placeholders.
- No emails, usernames or private endpoints appear in code or cell outputs.

If you fork this, **do not paste real tokens or storage keys** into the notebooks before committing.

---

## Citation

If you use this code, please cite the accompanying paper (details to be added on publication):

```bibtex
@misc{sam3_dinov3_distillation,
  title  = {Lightweight Distillation of SAM 3 and DINOv3 for Edge-Deployable Individual-Level Livestock Monitoring and Longitudinal Visual Analytics},
  author = {Haiyu Yang, Miel Hostens},
  year   = {2026},
  note   = {[https://github.com/Sonam525](https://arxiv.org/abs/2604.27128)}
}
```

## License

Released under the [MIT License](LICENSE).
