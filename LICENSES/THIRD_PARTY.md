# Third-party components

None of these is copied into this repository, except where the "Use" column says so. Check each upstream licence yourself.

| Component | Upstream terms | Use |
|:---|:---|:---|
| SAM 3 (`facebook/sam3`) | SAM License, last updated 19 Nov 2025; gated | Teacher and tracker. Downloaded by the reader. A copy of the licence is in this folder. |
| Transformers | Apache-2.0 | Installed by pip. |
| TinyViT | MIT (Microsoft, `Cream/TinyViT`); `timm` weights tagged Apache-2.0 | Student trunk. Installed through `timm`; weights downloaded. |
| SAM 2.1 | Apache-2.0 | Baseline. Downloaded by the reader. |
| EdgeTAM (`yonigozlan/EdgeTAM-hf`) | Apache-2.0 tag | Baseline. Downloaded by the reader. |
| EfficientSAM3 | `apache-2.0` tag on Hugging Face; no licence file found at the repository root | Comparison encoder. Downloaded by the reader. The extracted trunk is never redistributed. `benchmark/core_bc46ecfb/tol_es3.py` re-implements glue with the authors' module names. |
| RF-DETR | Apache-2.0 | Detector for Protocol D. Installed by pip. |
| TrackEval | See upstream | Scorer. Installed by pip at a pinned commit. Not vendored. |
| py-motmetrics | See upstream | Cross-check scorer. Installed by pip. Not vendored. |
| DINOv3 | DINOv3 License, 19 Aug 2025 | Only the preprint's behaviour notebook loads it. No DINOv3 weights ship here. |
