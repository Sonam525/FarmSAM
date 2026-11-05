# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N61 — Paper 1 bench, gate G2: the EfficientSAM3 ES-TV-L port (job J1)
# MAGIC
# MAGIC Ports the EfficientSAM3 Stage-1 image encoder **ES-TV-L** (TinyViT-21M + Conv-BN-GELU-Conv head; Zeng, Jiang, Zhang,
# MAGIC github.com/SimonZeng7108/efficientsam3) into SAM 3's HF tracker as the bench arm **ES-L** (`tol_es3.py`, deployed
# MAGIC beside `tol_bench_core.py`) and runs the **G2 parity checks** (baselines.md §4a checks 1–4) against the authors' own
# MAGIC code on the **A10**. Rules: BENCH_PROTOCOL §13 G2; thresholds and the arm definition: §15 "G2" entry (written before
# MAGIC this notebook ran).
# MAGIC
# MAGIC | stage | what |
# MAGIC |---|---|
# MAGIC | `fetch` | clone the authors' repository read-only at `tol_es3.ES3_COMMIT` (git-blob sha256 pins); download `stage1_all_converted/efficient_sam3_tinyvit_l.pt` from HF `Simon7108528/EfficientSAM3` (revision + LFS sha256 recorded); keep it with a NOTICE and the upstream source files under `models/baselines/third_party/efficientsam3/` (**never redistributed**); extract the encoder state_dict (strict load into ES-TV-L) to registry key `models.es3_trunk` |
# MAGIC | `parity` | the checks below |
# MAGIC
# MAGIC | check | compared | pass rule |
# MAGIC |---|---|---|
# MAGIC | **G2-1** teacher trunk | HF `vision_encoder.backbone` (`facebook/sam3` at the J0 pin, fp32) vs the authors' code's ViT trunk built from `sam3.pt` of the same HF repo (fp32), on the HF processor's `pixel_values` of the check frames, TF32 off | global cosine ≥ 0.9999 and relative L2 ≤ 1e-3 on every frame |
# MAGIC | **G2-1b** ES3 encoder | the production encoder (`tol_es3`, `models.es3_trunk`, fp32) vs the authors' model's `detector.backbone.vision_backbone.trunk` (their builder + the checkpoint, after `model.eval()`), on the ES-L frames, TF32 off | cosine ≥ 0.99999 and relative L2 ≤ 1e-5 on every frame |
# MAGIC | **G2-2** preprocessing | the ES-L frame (`tol_es3.make_es3_transform`) vs the authors' `init_state(video_path)` images (their loader, on the GPU) at the check frames; the HF processor's size / resample / mean / std are recorded | max abs difference ≤ 1e-6 on every check frame, and both at 1008² |
# MAGIC | **G2-3** route A vs route B | authors' SAM2-task tracker (`build_efficientsam3_video_model(..., apply_temporal_disambiguation=False)` + `model.eval()` after loading; `predictor = model.tracker; predictor.backbone = model.detector.backbone`; `init_state(video_path)`; relative boxes at frame 0; `propagate_in_video(..., propagate_preflight=True)`; bf16 autocast + TF32 as their notebook) vs the production ES-L arm (`tol_es3.load_es3_arm` = `load_arm('T')` + the swap; bf16, unpruned, pop) on `pigtrack0055` (98 frames, 12 pigs) and Edinburgh `2019_12_02_000005` (first 300 frames, 8 pigs), every pig box-prompted at frame 0 (Protocol A boxes) | mean mask IoU over object-frames where either side emitted > 0.95 on **each** clip, and the authors' build left no tracker / vision-backbone tensor unloaded |
# MAGIC | **G2-4** weights | every `detector.backbone.vision_backbone.sam2_convs.*` (SAM 3's tracker neck) and `tracker.*` tensor of the ES-TV-L checkpoint vs `sam3.pt` | bit-identical, no key missing on either side |
# MAGIC
# MAGIC Diagnostics (reported, no verdict): `asis` = the authors' video builder exactly as shipped (it calls `eval()` before
# MAGIC loading the weights, so TinyViT's cached attention biases stay at their zero initialisation); `td` = memory selection
# MAGIC on (`apply_temporal_disambiguation=True`, their demo default); `hfpix` = ES-L on the HF processor's `pixel_values`;
# MAGIC `teacher` = SAM 3 through the authors' code vs the HF T arm (on the authors' frame and on the HF frame).
# MAGIC
# MAGIC Outputs (never overwritten): `paper1_distillation/evidence/g2/<tag>/`; `models/baselines/es3_trunk_tinyvit_l.pt`
# MAGIC (`models.es3_trunk`); `models/baselines/third_party/efficientsam3/`; the canonical run (`tag=g2_v1`, the pre-registered
# MAGIC clips) writes `paper1_distillation/protocol/es3_g2_v1.json`, which `tol_bench_core.load_arm('ES-L')` requires (PASS).
# MAGIC
# MAGIC **Local check** (no GPU / Volume): `TOL61_SELFTEST=1 python Tol_61_es3_port.py`.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm" "scipy" "rfdetr==1.10.1" "pycocotools" "peft"
# MAGIC %pip uninstall -y opencv-python opencv-contrib-python
# MAGIC %pip install --no-deps --force-reinstall "opencv-python-headless==4.11.0.86"

# COMMAND ----------

try:
    dbutils.library.restartPython()   # noqa: F821
except NameError:
    pass

# COMMAND ----------

import collections
import gc
import glob
import gzip
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import zipfile

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

NB_NAME = "Tol_61_es3_port"
NB_VERSION = "tol61-2026-10-15.1"
SELFTEST = os.environ.get("TOL61_SELFTEST") == "1"
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None


def _this_dir():
    env = os.environ.get("TOL61_CODE_DIR")
    if env:
        return env
    if not IS_NOTEBOOK:
        return os.path.dirname(os.path.abspath(__file__))
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    return "/Workspace" + os.path.dirname(nbp)


def _import_exact(name, dirs):
    for d in dirs:
        if not d or not os.path.isfile(os.path.join(d, name + ".py")):
            continue
        while d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
        sys.modules.pop(name, None)
        mod = __import__(name)
        got = os.path.realpath(os.path.dirname(os.path.abspath(mod.__file__)))
        if os.path.normcase(got) != os.path.normcase(os.path.realpath(d)):
            raise RuntimeError(f"{name} resolved to {mod.__file__}, not {d}")
        return mod
    raise RuntimeError(f"{name}.py not found in {[d for d in dirs if d]}")


CODE_DIR = _this_dir()
os.environ.setdefault("TOL_CORE_DIR", CODE_DIR)
bc = _import_exact("tol_bench_core", [CODE_DIR])
es3 = _import_exact("tol_es3", [CODE_DIR])
dc = _import_exact("tol_distill_core", [CODE_DIR])          # publish_file_guarded (models/ targets)
_PATHS_DIRS = [CODE_DIR, os.environ.get("TOL_PATHS_DIR")]
if SELFTEST:
    _PATHS_DIRS.append(os.path.join(os.path.dirname(CODE_DIR), "catalog", "build"))
tol_paths = _import_exact("tol_paths", _PATHS_DIRS)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print(f"{NB_NAME} {NB_VERSION}: tol_bench_core {bc.BENCH_VERSION} (sha256 {_sha256(bc.__file__)[:16]}), "
      f"tol_es3 {es3.ES3_VERSION} (sha256 {_sha256(es3.__file__)[:16]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters, pre-registered thresholds, pure helpers (exercised by the selftest)

# COMMAND ----------

WIDGETS = collections.OrderedDict([
    ("mode", "all"), ("tag", "g2_v1"), ("es3_revision", ""), ("clips", "pigtrack0055,2019_12_02_000005"),
    ("edi_frames", "300"), ("diagnostics", "asis,td,hfpix,teacher"), ("revisions", "protocol:hf_revisions_v1.json"),
    ("require_gpu", "A10"), ("write_protocol", "auto"), ("progress_every", "0"),
])
MODES = ("all", "fetch", "parity")
DIAGNOSTICS = ("asis", "td", "hfpix", "teacher")
GATE_CLIPS = ("pigtrack0055", "2019_12_02_000005")    # §15 G2: BOTH are gates for G2-3
GATE_EDI_FRAMES = 300
# ---- pre-registered thresholds (BENCH_PROTOCOL §15 G2 entry) ----
TEACHER_COS_MIN, TEACHER_RELL2_MAX = 0.9999, 1e-3     # G2-1 (fp32, TF32 off)
ES3_COS_MIN, ES3_RELL2_MAX = 0.99999, 1e-5            # G2-1b (fp32, TF32 off)
PIXEL_MAXABS_MAX = 1e-6                               # G2-2
PARITY_IOU_MIN = 0.95                                 # G2-3 (= G1-8's rule and aggregation)
TAG_RX = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
CANONICAL_TAG_RX = re.compile(r"^g2_v\d+$")
SAM3_HF_ID = "facebook/sam3"
SAM3_PT = "sam3.pt"
META_DEPS = ("ftfy==6.1.1", "wcwidth", "iopath==0.1.10", "portalocker", "einops", "pycocotools")
NOTICE = """# Third-party material: EfficientSAM3 ES-TV-L (Stage 1)

Downloaded by {nb} on {date} for INTERNAL benchmarking only (Paper 1, bench arm ES-L, gate G2). **Do not redistribute.**

* Weights: https://huggingface.co/{repo} , file `{file}`, revision `{rev}`, sha256 `{sha}` ({size} bytes).
  Model-card licence tag: `{card_license}`. The file is a full SAM 3 checkpoint whose ViT trunk is replaced by the
  Stage-1 student encoder; its other tensors are SAM 3 parameters (Meta), governed by the SAM License.
* Code: {code_url} at commit `{commit}`. The README states Apache-2.0; the LICENSE file of the inner `sam3` package
  (the authors' fork of facebookresearch/sam3) is the SAM License (copied here as `{src_dir}/LICENSE_sam3_package`);
  `tiny_vit.py` carries the Microsoft TinyViT header (MIT). `{src_dir}/` holds the upstream files the ES-L arm executes
  or replicates (git-blob sha256 in SOURCE.json).
* Derived: `models.es3_trunk` (es3_trunk_tinyvit_l.pt) = the encoder state_dict extracted from the file above
  (prefix `{prefix}`); same restrictions.
"""


def parse_params(raw):
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}; known: {list(WIDGETS)}")
    g = {k: str(raw.get(k) if raw.get(k) not in (None, "") else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace()
    P.mode = g["mode"]
    if P.mode not in MODES:
        raise ValueError(f"mode={P.mode!r}: expected one of {MODES}")
    P.tag = g["tag"]
    if not TAG_RX.match(P.tag):
        raise ValueError(f"tag={P.tag!r}")
    P.es3_revision = g["es3_revision"]
    if P.es3_revision and not re.match(r"^[0-9a-f]{40}$", P.es3_revision):
        raise ValueError("es3_revision: a 40-hex commit (or empty = resolve main and record it)")
    P.clips = [c for c in re.split(r"[,\s]+", g["clips"]) if c]
    for c in P.clips:
        if not (c in bc.PIGTRACK_TEST or c in bc.EDI_TEST):
            raise ValueError(f"clip {c}: a PigTrack or Edinburgh TEST sequence (val is T-only, train distillation-only)")
    if not P.clips:
        raise ValueError("clips: at least one")
    P.edi_frames = int(g["edi_frames"])
    if not 1 <= P.edi_frames <= 300:
        raise ValueError("edi_frames: 1..300 (baselines §4a: <= 300 frames)")
    P.diagnostics = [d for d in re.split(r"[,\s]+", g["diagnostics"]) if d]
    bad = [d for d in P.diagnostics if d not in DIAGNOSTICS]
    if bad:
        raise ValueError(f"diagnostics {bad}: a subset of {DIAGNOSTICS}")
    P.revisions_spec = g["revisions"]
    P.require_gpu = g["require_gpu"]
    if P.require_gpu not in ("A10", "A100", "any"):
        raise ValueError(f"require_gpu={P.require_gpu!r}")
    canonical = (bool(CANONICAL_TAG_RX.match(P.tag)) and tuple(P.clips) == GATE_CLIPS
                 and P.edi_frames == GATE_EDI_FRAMES)
    wp = g["write_protocol"].lower()
    if wp not in ("auto", "true", "false"):
        raise ValueError(f"write_protocol={wp!r}")
    P.write_protocol = canonical if wp == "auto" else wp == "true"
    if P.write_protocol and not canonical:
        raise ValueError("write_protocol=true needs the canonical configuration (tag g2_v<N>, the pre-registered clips "
                         f"{GATE_CLIPS}, edi_frames={GATE_EDI_FRAMES})")
    P.canonical = canonical
    P.progress_every = int(g["progress_every"])
    return P


def feature_compare(test, ref):
    """[1, C, H, W] arrays -> global cosine, relative L2 (||test - ref|| / ||ref||), max |diff|, per-position cosine."""
    a = np.asarray(test, dtype=np.float64).reshape(np.asarray(test).shape[1], -1)
    b = np.asarray(ref, dtype=np.float64).reshape(np.asarray(ref).shape[1], -1)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(np.asarray(test).shape), list(np.asarray(ref).shape)]}
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    pp = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-30)
    return {"cos_global": float((a * b).sum() / (na * nb + 1e-30)), "rel_l2": float(np.linalg.norm(a - b) / (nb + 1e-30)),
            "max_abs": float(np.abs(a - b).max()), "cos_patch_mean": float(pp.mean()), "cos_patch_min": float(pp.min()),
            "ref_std": float(b.std()), "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "identical": bool(np.array_equal(a, b))}


def feature_verdict(rows, cos_min, rel_max):
    """Every frame: finite, same shape, cos >= cos_min and rel L2 <= rel_max."""
    if not rows:
        return "FAIL"
    ok = all(r.get("finite") and "shape_mismatch" not in r and r["cos_global"] >= cos_min and r["rel_l2"] <= rel_max
             for r in rows.values())
    return "PASS" if ok else "FAIL"


def rle_iou(a, b):
    if a is None and b is None:
        return None
    if a is None or b is None:
        return 0.0
    ma, mb = bc.rle_decode(a), bc.rle_decode(b)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return inter / union if union else 1.0


def _stats(v):
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": round(float(a.mean()), 6), "median": round(float(np.median(a)), 6),
            "p05": round(float(np.percentile(a, 5)), 6), "min": round(float(a.min()), 6)}


def norm_masks(rec):
    """{fi: {oid: rle}} with int keys (route A writes JSON string keys)."""
    return {int(f): {int(o): r for o, r in d.items()} for f, d in rec.items()}


def parity_compare(test, ref, *, iou_min=PARITY_IOU_MIN):
    """Two {fi: {oid: rle}} records: mask IoU over object-frames where EITHER side emitted (a one-sided emission
    counts 0) -- the G1-8 aggregation -- plus the frame-0 IoU per object and per-object means."""
    test, ref = norm_masks(test), norm_masks(ref)
    ious, per_obj, f0 = [], collections.defaultdict(list), {}
    frames = sorted(set(test) | set(ref))
    one_sided = 0
    for f in frames:
        a, b = test.get(f, {}), ref.get(f, {})
        for oid in sorted(set(a) | set(b)):
            v = rle_iou(a.get(oid), b.get(oid))
            if (oid in a) != (oid in b):
                one_sided += 1
            ious.append(v)
            per_obj[oid].append(v)
            if f == 0:
                f0[str(oid)] = round(v, 4)
    st = _stats(ious)
    st["frac_ge_0.95"] = round(float(np.mean(np.asarray(ious) >= 0.95)), 6) if ious else None
    return {"frames": len(frames), "frames_test": len(test), "frames_ref": len(ref), "object_frames": len(ious),
            "one_sided_object_frames": one_sided, "mask_iou": st, "frame0_iou": f0,
            "per_object_mean_iou": {str(k): round(float(np.mean(v)), 5) for k, v in sorted(per_obj.items())},
            "pass": bool(ious) and st["mean"] > iou_min}


def weights_diff(a_sd, b_sd, prefixes):
    """Per prefix: keys only in a / only in b, tensors that differ (shape, dtype or value), max |diff| over the
    shared keys. Values: numpy arrays or torch tensors."""
    def _np(v):
        return v.detach().cpu().float().numpy() if hasattr(v, "detach") else np.asarray(v, dtype=np.float32)

    def _dt(v):
        return str(getattr(v, "dtype", "")).replace("torch.", "")
    out = {}
    for p in prefixes:
        ka = sorted(k for k in a_sd if k.startswith(p))
        kb = sorted(k for k in b_sd if k.startswith(p))
        sa, sb = set(ka), set(kb)
        diff, worst, dtype_diff = [], 0.0, []
        for k in sorted(sa & sb):
            va, vb = a_sd[k], b_sd[k]
            if tuple(va.shape) != tuple(vb.shape):
                diff.append(k)
                continue
            if _dt(va) != _dt(vb):
                dtype_diff.append(k)
            x, y = _np(va), _np(vb)
            d = float(np.abs(x - y).max()) if x.size else 0.0
            if d > 0 or _dt(va) != _dt(vb):
                diff.append(k)
            worst = max(worst, d)
        out[p] = {"n_a": len(ka), "n_b": len(kb), "only_a": sorted(sa - sb)[:10], "n_only_a": len(sa - sb),
                  "only_b": sorted(sb - sa)[:10], "n_only_b": len(sb - sa), "n_differ": len(diff),
                  "differ_sample": diff[:10], "n_dtype_differ": len(dtype_diff), "max_abs_diff": worst,
                  "bit_identical": len(sa) > 0 and sa == sb and not diff}
    return out


def aggregate(checks):
    """{name: verdict} -> overall PASS iff every gate check is PASS (and there is at least one)."""
    return "PASS" if checks and all(v == "PASS" for v in checks.values()) else "FAIL"


def g2_record(P, verdicts, fetch, evidence_rel, extra):
    """The protocol file (tol_es3.validate_g2 checks its shape; load_arm('ES-L') requires verdict PASS)."""
    src_rel = "/".join(list(es3.THIRD_PARTY_SUBDIR) + [f"src_{es3.ES3_COMMIT[:12]}", "tiny_vit.py"])
    return {"version": es3.G2_FILE.rsplit(".", 1)[0], "verdict": aggregate(verdicts), "checks": verdicts,
            "trunk": {"registry_key": es3.REGISTRY_KEY, "sha256": fetch["trunk"]["sha256"],
                      "format": es3.TRUNK_FORMAT, "encoder": fetch["trunk"]["encoder"]},
            "tinyvit_source": {"relpath": src_rel, "sha256": es3.ES3_UPSTREAM_SHA256[es3.TINYVIT_UPSTREAM]},
            "checkpoint": fetch["checkpoint"], "efficientsam3_commit": es3.ES3_COMMIT,
            "arm": {"pixel_values": "authors' SAM2-task loader (tol_es3.make_es3_transform)", "dtype": "bfloat16",
                    "backbone": "ES3Backbone (in the HF graph)", "g2_route_b": "tol_es3.load_es3_arm"},
            "thresholds": {"G2-1": [TEACHER_COS_MIN, TEACHER_RELL2_MAX], "G2-1b": [ES3_COS_MIN, ES3_RELL2_MAX],
                           "G2-2": PIXEL_MAXABS_MAX, "G2-3": PARITY_IOU_MIN},
            "clips": P.clips, "edi_frames": P.edi_frames, "evidence": evidence_rel, "tag": P.tag,
            "notebook": f"{NB_NAME} {NB_VERSION}", **extra}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Route A: the authors' code in its own process

# COMMAND ----------

META_SCRIPT = r'''
"""Tol_61 G2 route A: the AUTHORS' code (the efficientsam3 fork of facebookresearch/sam3) in its own process, with
the repository on PYTHONPATH. Job = JSON (argv[1]); results in job["out_dir"] (result.json + .npy + masks_*.json)."""
import contextlib, gc, io, json, os, sys, time
import numpy as np
import torch
job = json.load(open(sys.argv[1]))
OUT = job["out_dir"]
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
res = {"model": job["model"], "torch": torch.__version__, "variants": {}, "trunk": {}, "images": {}, "events": []}
# the authors' notebook preamble (efficientsam3_for_sam2_video_task_example.ipynb): bf16 autocast for the whole
# process and TF32 on Ampere
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
import sam3
from sam3.model_builder import build_efficientsam3_video_model, build_sam3_video_model
res["sam3_file"] = sam3.__file__
__RLE_SRC__
__STALE_SRC__


def ev(msg):
    res["events"].append([round(time.time() - t0, 1), msg])
    print(msg, flush=True)


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    if job["model"] == "es3":
        model = build_efficientsam3_video_model(checkpoint_path=job["ckpt"], backbone_type="tinyvit", model_name="21m",
                                                load_from_HF=False, device="cuda", strict_state_dict_loading=False,
                                                apply_temporal_disambiguation=False, compile=False)
    else:
        model = build_sam3_video_model(checkpoint_path=job["ckpt"], load_from_HF=False, strict_state_dict_loading=False,
                                       apply_temporal_disambiguation=False, device="cuda", compile=False)
res["builder_stdout_tail"] = buf.getvalue()[-3000:]
ev("built")
ck = torch.load(job["ckpt"], map_location="cpu", weights_only=True)
if isinstance(ck, dict) and isinstance(ck.get("model"), dict):
    ck = ck["model"]
if job["model"] == "es3":                       # the builder's own key cleaning
    ck = {k.replace("student_trunk.", ""): v for k, v in ck.items()}
    if not any(k.startswith(("detector.", "tracker.")) for k in ck) and any(k.startswith("backbone.") for k in ck):
        ck = {"detector." + k: v for k, v in ck.items()}
msd = model.state_dict()
missing = sorted(k for k in msd if k not in ck)
unexpected = sorted(k for k in ck if k not in msd)
shape_bad = sorted(k for k in msd if k in ck and tuple(msd[k].shape) != tuple(ck[k].shape))


def _grp(keys):
    g = {}
    for k in keys:
        p = ".".join(k.split(".")[:3])
        g[p] = g.get(p, 0) + 1
    return g


res["coverage"] = {"n_model": len(msd), "n_ckpt": len(ck), "missing": _grp(missing), "missing_sample": missing[:12],
                   "unexpected": _grp(unexpected), "unexpected_sample": unexpected[:12],
                   "shape_mismatch": shape_bad[:12],
                   "missing_tracker_or_vision": [k for k in missing
                                                 if k.startswith(("tracker.", "detector.backbone.vision_backbone."))][:50]}
del ck, msd
gc.collect()
predictor = model.tracker
predictor.backbone = model.detector.backbone
trunk = model.detector.backbone.vision_backbone.trunk
res["modules_in_training_mode"] = sum(1 for m in model.modules() if m.training)


def stale():
    n, s, w = stale_attention_bias_caches(trunk)
    return {"n_attention": n, "n_stale": s, "max_abs_diff": (None if w == float("inf") else w)}


@torch.inference_mode()
def dump_trunk(state):
    for item in job.get("trunk_inputs", []):
        x = torch.load(item["path"], map_location="cpu", weights_only=True).to("cuda").float()
        for prec in ("fp32", "authors"):
            if prec == "fp32":           # fp32 reference: autocast off, TF32 off
                keep = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                try:
                    with torch.autocast("cuda", enabled=False):
                        y = trunk(x)[-1]
                finally:
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = keep
            else:                        # the authors' own setting (process-wide bf16 autocast + TF32)
                y = trunk(x)[-1]
            f = os.path.join(OUT, f"trunk_{state}_{prec}_{item['label']}.npy")
            np.save(f, y.float().cpu().numpy())
            res["trunk"].setdefault(state, {}).setdefault(prec, {})[item["label"]] = os.path.basename(f)
    ev(f"trunk dumped ({state})")


@torch.inference_mode()
def track(variant, clip):
    st = predictor.init_state(video_path=clip["jpeg_dir"])
    n, W, H = st["num_frames"], st["video_width"], st["video_height"]
    if clip.get("dump_images") and clip["name"] not in res["images"]:
        res["images"][clip["name"]] = {}
        for fi in clip["dump_images"]:
            f = os.path.join(OUT, f"images_{clip['name']}_{fi}.npy")
            np.save(f, st["images"][fi].float().cpu().numpy())
            res["images"][clip["name"]][str(fi)] = os.path.basename(f)
    for oid, x1, y1, x2, y2 in clip["prompts"]:
        predictor.add_new_points_or_box(inference_state=st, frame_idx=0, obj_id=int(oid),
                                        box=np.array([x1 / W, y1 / H, x2 / W, y2 / H], dtype=np.float32))
    masks, scores, bad = {}, {}, 0
    t1 = time.time()
    for f_idx, obj_ids, low, vid, sc in predictor.propagate_in_video(st, start_frame_idx=0, max_frame_num_to_track=n,
                                                                    reverse=False, propagate_preflight=True):
        m = (vid[:, 0] > 0.0).cpu().numpy()
        s = sc.float().reshape(len(obj_ids), -1)[:, 0].cpu().numpy()
        d = {}
        for k, oid in enumerate(obj_ids):
            if m[k].any() and s[k] > 0:          # the bench output rule: object score > 0 and a non-empty mask
                d[str(int(oid))] = rle_encode(m[k])
            elif m[k].any():
                bad += 1
        masks[str(int(f_idx))] = d
        scores[str(int(f_idx))] = [round(float(v), 4) for v in s]
    rec = {"frames": len(masks), "n_frames_state": int(n), "size": [int(H), int(W)], "seconds": round(time.time() - t1, 1),
           "mask_without_score": bad, "use_memory_selection": bool(getattr(predictor, "use_memory_selection", False)),
           "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    f = os.path.join(OUT, f"masks_{variant}_{clip['name']}.json")
    with open(f, "w") as fh:
        json.dump({"masks": masks, "scores": scores}, fh)
    res["variants"].setdefault(variant, {})[clip["name"]] = dict(rec, file=os.path.basename(f))
    del st
    gc.collect()
    torch.cuda.empty_cache()
    ev(f"{variant} {clip['name']}: {rec['frames']} frames, {rec['seconds']} s")


variants = job["variants"]
if job["model"] == "es3":
    res["stale_as_built"] = stale()
    dump_trunk("asis")
    if "asis" in variants:
        for c in job["clips"]:
            track("asis", c)
    # the authors' IMAGE builder calls model.eval() AFTER loading (_setup_device_and_mode); their VIDEO builder only
    # before it (Sam3VideoBase.__init__), so TinyViT's attention-bias cache still holds the zero initialisation
    model.eval()
    res["stale_after_eval"] = stale()
    dump_trunk("fixed")
    for c in job["clips"]:
        track("fixed", c)
    if "td" in variants:                     # = build_tracker(apply_temporal_disambiguation=True) for this path
        predictor.use_memory_selection = True
        for c in job["clips"]:
            track("fixed_td", c)
else:
    dump_trunk("teacher")
    if "teacher" in variants:
        for c in job["clips"]:
            track("teacher", c)
res["seconds"] = round(time.time() - t0, 1)
with open(os.path.join(OUT, "result.json"), "w") as fh:
    json.dump(res, fh)
print("META_DONE", res["seconds"], flush=True)
'''


def render_meta_script():
    return (META_SCRIPT.replace("__RLE_SRC__", inspect.getsource(bc.rle_encode))
            .replace("__STALE_SRC__", inspect.getsource(es3.stale_attention_bias_caches)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Staging, fetch, checks

# COMMAND ----------


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json_local(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp_{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, default=_jd) + "\n")
    os.replace(tmp, path)
    return path


def out_path(ctx, name):
    if not TAG_RX.match(name.replace(".json", "").replace(".gz", "")):
        raise ValueError(f"output name {name!r}")
    return ctx.R.out("paper1.root", "evidence", "g2", ctx.P.tag, name)


def evidence_rel(ctx):
    return "paper1_distillation/evidence/g2/" + ctx.P.tag


def load_done(ctx, name):
    p = out_path(ctx, name)
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def publish(ctx, name, obj):
    loc = write_json_local(os.path.join(ctx.local, "out", name), obj)
    bc.publish_no_overwrite(loc, out_path(ctx, name), ctx.dbutils)
    return obj


def publish_gz(ctx, name, obj):
    loc = os.path.join(ctx.local, "out", name)
    os.makedirs(os.path.dirname(loc), exist_ok=True)
    with gzip.GzipFile(loc, "wb", mtime=0) as f:
        f.write(json.dumps(obj, sort_keys=True, default=_jd).encode("utf-8"))
    bc.publish_no_overwrite(loc, out_path(ctx, name), ctx.dbutils)


def model_dirs(R):
    """models.es3_trunk and the third-party directories derived from it (paths_v1 has no key for them; never a
    literal path)."""
    trunk = R.out(es3.REGISTRY_KEY)
    base = os.path.dirname(trunk).replace("\\", "/")
    tp = "/".join([base] + list(es3.THIRD_PARTY_SUBDIR))
    return {"trunk": trunk, "base": base, "third_party": tp, "src": f"{tp}/src_{es3.ES3_COMMIT[:12]}"}


def _git(args, cwd=None, timeout=600):
    return subprocess.run(["git"] + args, capture_output=True, text=True, timeout=timeout, cwd=cwd)


def fetch_repo_at_commit(ctx):
    """The authors' repository at ES3_COMMIT, read-only: git fetch of that commit (fallback: the GitHub zip of the
    commit). -> (dir, how). Every pinned file is checked against its git-blob sha256."""
    root = os.path.join(ctx.local, "es3_repo")
    marker = os.path.join(root, ".tol61_ok")
    if not os.path.isfile(marker):
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root)
        how = "git"
        ok = (_git(["init", "-q"], root).returncode == 0
              and _git(["remote", "add", "origin", es3.ES3_REPO_URL], root).returncode == 0
              and _git(["fetch", "-q", "--depth", "1", "origin", es3.ES3_COMMIT], root, 900).returncode == 0
              and _git(["checkout", "-q", "FETCH_HEAD"], root).returncode == 0)
        if not ok:
            ctx.log("git fetch of the pinned commit failed; using the GitHub zip of that commit")
            import urllib.request
            shutil.rmtree(root, ignore_errors=True)
            zp = root + ".zip"
            urllib.request.urlretrieve(es3.ES3_REPO_URL.replace("https://github.com/", "https://codeload.github.com/")
                                       + "/zip/" + es3.ES3_COMMIT, zp)
            with zipfile.ZipFile(zp) as z:
                top = z.namelist()[0].split("/")[0]
                z.extractall(os.path.dirname(root))
            os.replace(os.path.join(os.path.dirname(root), top), root)
            how = "zip"
        with open(marker, "w") as f:
            f.write(how)
    with open(marker) as f:
        how = f.read().strip()
    shas = {}
    for rel, want in es3.ES3_UPSTREAM_SHA256.items():
        with open(os.path.join(root, rel), "rb") as f:
            b = f.read().replace(b"\r\n", b"\n")
        shas[rel] = hashlib.sha256(b).hexdigest()
        if shas[rel] != want:
            raise RuntimeError(f"{rel}: sha256 {shas[rel][:16]} != pinned {want[:16]} (commit {es3.ES3_COMMIT[:12]})")
    head = _git(["rev-parse", "HEAD"], root).stdout.strip() if how == "git" else None
    if how == "git" and head != es3.ES3_COMMIT:
        raise RuntimeError(f"checked-out HEAD {head} != {es3.ES3_COMMIT}")
    return root, {"how": how, "head": head, "upstream_sha256": shas}


def hf_file_info(repo, filename, revision=None, token=None):
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=token)
    sib = [s for s in (info.siblings or []) if s.rfilename == filename]
    if not sib:
        raise FileNotFoundError(f"{repo}@{info.sha}: no {filename}")
    s = sib[0]
    lfs = getattr(s, "lfs", None)
    lfs_sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    lic = None
    if card is not None:
        lic = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    return {"repo": repo, "file": filename, "revision": info.sha, "size": getattr(s, "size", None),
            "lfs_sha256": lfs_sha, "card_license": lic, "gated": getattr(info, "gated", None)}


def hf_download(ctx, info, token=None):
    from huggingface_hub import hf_hub_download
    p = os.path.realpath(hf_hub_download(info["repo"], info["file"], revision=info["revision"], token=token,
                                         cache_dir=os.path.join(ctx.local, "hf")))   # the blob, not the snapshot link
    sha = _sha256(p)
    if info.get("lfs_sha256") and sha != info["lfs_sha256"]:
        raise RuntimeError(f"{info['repo']}/{info['file']}: sha256 {sha[:16]} != LFS {info['lfs_sha256'][:16]}")
    if info.get("size") and os.path.getsize(p) != info["size"]:
        raise RuntimeError(f"{info['file']}: {os.path.getsize(p)} B != {info['size']} B")
    return p, sha


def stage_fetch_outputs(ctx, rec):
    """Local copies of the published fetch outputs (a parity-only task, or a resumed run), sha256-checked."""
    md = model_dirs(ctx.R)
    loc = os.path.join(ctx.local, "es3_files")
    ck_name = os.path.basename(es3.ES3_HF_FILE)
    ck, _ = bc.stage_file(f"{md['third_party']}/{ck_name}", os.path.join(loc, ck_name), ctx.dbutils,
                          rec["checkpoint"]["sha256"])
    tr, _ = bc.stage_file(md["trunk"], os.path.join(loc, os.path.basename(md["trunk"])), ctx.dbutils,
                          rec["trunk"]["sha256"])
    tv, _ = bc.stage_file(f"{md['src']}/tiny_vit.py", os.path.join(loc, "tiny_vit.py"), ctx.dbutils,
                          es3.ES3_UPSTREAM_SHA256[es3.TINYVIT_UPSTREAM])
    return {"ckpt": ck, "trunk": tr, "tinyvit": tv}


def do_fetch(ctx):
    """Stage 'fetch' -> the fetch record (published as fetch.json) + ctx.files (local paths)."""
    prev = load_done(ctx, "fetch.json")
    if prev is not None:
        ctx.log("fetch.json already published: staging its outputs")
        ctx.files = stage_fetch_outputs(ctx, prev)
        return prev
    torch = bc.torch
    t0 = time.time()
    repo, repo_meta = fetch_repo_at_commit(ctx)
    try:
        info = hf_file_info(es3.ES3_HF_REPO, es3.ES3_HF_FILE, ctx.P.es3_revision or None)
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001 -- public repo; authenticated metadata only if anonymous access failed
        ctx.log(f"anonymous model_info failed ({type(e).__name__}); retrying with the HF token")
        info = hf_file_info(es3.ES3_HF_REPO, es3.ES3_HF_FILE, ctx.P.es3_revision or None, token=ctx.token)
    try:
        ck_local, ck_sha = hf_download(ctx, info)
        dl_auth = "anonymous"
    except Exception as e:  # noqa: BLE001 -- a public repo; retry authenticated only if anonymous access failed
        ctx.log(f"anonymous download failed ({type(e).__name__}); retrying with the HF token")
        ck_local, ck_sha = hf_download(ctx, info, token=ctx.token)
        dl_auth = "token"
    md = model_dirs(ctx.R)
    loc = os.path.join(ctx.local, "es3_files")
    os.makedirs(loc, exist_ok=True)
    # 1) the raw checkpoint + NOTICE + SOURCE + the upstream files, beside models.es3_trunk (never redistributed)
    src_dir_name = os.path.basename(md["src"])
    up_files = {"tiny_vit.py": es3.TINYVIT_UPSTREAM, "sam2_utils.py": "sam3/sam3/model/utils/sam2_utils.py",
                "model_builder.py": "sam3/sam3/model_builder.py", "LICENSE_sam3_package": "sam3/LICENSE"}
    up_local = {}
    for name, rel in up_files.items():
        with open(os.path.join(repo, rel), "rb") as f:
            b = f.read().replace(b"\r\n", b"\n")
        p = os.path.join(loc, "src", name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b)
        up_local[name] = {"upstream_path": rel, "sha256": hashlib.sha256(b).hexdigest(), "local": p}
    ck_name = os.path.basename(es3.ES3_HF_FILE)
    checkpoint = {"hf_repo": info["repo"], "file": info["file"], "revision": info["revision"], "sha256": ck_sha,
                  "size": os.path.getsize(ck_local), "card_license": info["card_license"],
                  "relpath": "/".join(list(es3.THIRD_PARTY_SUBDIR) + [ck_name]), "download": dl_auth}
    notice = NOTICE.format(nb=f"{NB_NAME} {NB_VERSION}", date=_now()[:10], repo=info["repo"], file=info["file"],
                           rev=info["revision"], sha=ck_sha, size=checkpoint["size"], card_license=info["card_license"],
                           code_url=es3.ES3_REPO_URL, commit=es3.ES3_COMMIT, src_dir=src_dir_name,
                           prefix=es3.TRUNK_PREFIX)
    notice_p = os.path.join(loc, "NOTICE.md")
    with open(notice_p, "w", encoding="utf-8", newline="\n") as f:
        f.write(notice)
    source = {"checkpoint": checkpoint, "efficientsam3": {"url": es3.ES3_REPO_URL, "commit": es3.ES3_COMMIT,
                                                          "fetched_by": repo_meta["how"]},
              "upstream_files": {k: {kk: vv for kk, vv in v.items() if kk != "local"} for k, v in up_local.items()},
              "notebook": f"{NB_NAME} {NB_VERSION}"}
    source_p = write_json_local(os.path.join(loc, "SOURCE.json"), source)
    published = {}
    published[ck_name] = dc.publish_file_guarded(ck_local, f"{md['third_party']}/{ck_name}", md["third_party"],
                                                 ctx.dbutils)
    published["NOTICE.md"] = dc.publish_file_guarded(notice_p, f"{md['third_party']}/NOTICE.md", md["third_party"],
                                                     ctx.dbutils)
    published["SOURCE.json"] = dc.publish_file_guarded(source_p, f"{md['third_party']}/SOURCE.json",
                                                       md["third_party"], ctx.dbutils)
    for name, v in up_local.items():
        published[f"{src_dir_name}/{name}"] = dc.publish_file_guarded(v["local"], f"{md['src']}/{name}", md["src"],
                                                                      ctx.dbutils)
    ctx.log(f"third-party files published: {published}")
    # 2) the encoder state_dict -> models.es3_trunk (strict load into ES-TV-L validates every key and shape)
    ck = torch.load(ck_local, map_location="cpu", weights_only=True)
    enc_sd, rep = es3.extract_trunk(ck)
    tv = es3.load_upstream_tinyvit(up_local["tiny_vit.py"]["local"])
    _enc, emeta = es3.load_es3_encoder(enc_sd, tv, device="cpu", dtype=torch.float32)
    del _enc
    emeta = {k: v for k, v in emeta.items() if k != "dtype"}
    payload = es3.trunk_payload(enc_sd, {"source": {"hf_repo": info["repo"], "file": info["file"],
                                                    "revision": info["revision"], "sha256": ck_sha,
                                                    "prefix": es3.TRUNK_PREFIX, "efficientsam3_commit": es3.ES3_COMMIT},
                                         "encoder": emeta})
    trunk_local = os.path.join(loc, os.path.basename(md["trunk"]))
    torch.save(payload, trunk_local)
    trunk_sha = _sha256(trunk_local)
    published[os.path.basename(md["trunk"])] = dc.publish_file_guarded(trunk_local, md["trunk"], md["base"],
                                                                       ctx.dbutils)
    del ck, payload, enc_sd
    gc.collect()
    rec = {"stage": "fetch", "checkpoint": checkpoint, "efficientsam3": source["efficientsam3"],
           "upstream_sha256": repo_meta["upstream_sha256"], "extraction": rep,
           "trunk": {"registry_key": es3.REGISTRY_KEY, "sha256": trunk_sha, "size": os.path.getsize(trunk_local),
                     "format": es3.TRUNK_FORMAT, "encoder": emeta},
           "published": published, "licence_note": "internal benchmarking only; never redistributed (NOTICE.md)",
           "wall_s": round(time.time() - t0, 1), "notebook": f"{NB_NAME} {NB_VERSION}", "created_utc": _now(),
           "core_sha256": _sha256(bc.__file__), "es3_sha256": _sha256(es3.__file__)}
    publish(ctx, "fetch.json", rec)
    ctx.files = {"ckpt": ck_local, "trunk": trunk_local, "tinyvit": up_local["tiny_vit.py"]["local"],
                 "repo": repo}
    return rec


# ---------------------------------------------------------------------------------------------------- clips
def stage_clip(ctx, clip):
    """Stage one parity clip; prompts = its Protocol A boxes, all at frame 0 (asserted); jpeg_dir = a directory the
    authors' loader reads (frames sorted by int(stem)), byte-identical to the frames the HF side reads."""
    if clip in ctx.staged:
        return ctx.staged[clip]
    R = ctx.R
    if clip in bc.PIGTRACK_TEST:
        d = os.path.join(ctx.local, "stage", "pigtrack", "test", clip)
        bc.stage_tree(R.path("datasets.pigtrack.sequences", None, "test", clip), d, ctx.dbutils)
        src = bc.PigTrackFrames(d)
        ps, _census = bc.protocol_a_from_mot(bc.read_mot_txt(src.gt_path))
        n = len(src)
        jpeg_dir = os.path.join(d, "img1")
        names = sorted(os.listdir(jpeg_dir))
        if [int(os.path.splitext(x)[0]) for x in names] != list(range(1, n + 1)):
            raise ValueError(f"{jpeg_dir}: frames are not 1..{n}")
        rec = {"name": clip, "kind": "pigtrack", "dir": d, "n": n, "prompts": ps, "jpeg_dir": jpeg_dir,
               "size": [src.height, src.width]}
    else:
        date, num = re.match(r"^(\d{4}_\d{2}_\d{2})_(\d{6})$", clip).groups()
        d = os.path.join(ctx.local, "stage", "edinburgh", clip, "video1")
        bc.stage_tree(R.path("datasets.edinburgh_pigs.decoded_frames", None, clip, "video1"), d, ctx.dbutils)
        gt, _ = bc.stage_file(R.path("datasets.edinburgh_pigs.annotated", None, date, num, "output.json"),
                              os.path.join(ctx.local, "stage", "edinburgh", clip, "output.json"), ctx.dbutils)
        txt, _ = bc.stage_file(R.path("datasets.edinburgh_pigs.initial_prompts", None, clip, "video1",
                                      f"BoundingBoxes_{clip}.txt"),
                               os.path.join(ctx.local, "stage", "edinburgh", clip, "BoundingBoxes.txt"), ctx.dbutils)
        ps = bc.protocol_a_edinburgh(bc.read_edinburgh_keyframes(gt))
        if {p.obj_id: list(p.box) for p in ps} != bc.read_bbox_prompts_txt(txt):
            raise bc.BenchAssertError(f"{clip}: GT-derived prompts differ from the published seeds (§4)")
        n = ctx.P.edi_frames
        jpeg_dir = os.path.join(ctx.local, "meta_jpg", clip)
        os.makedirs(jpeg_dir, exist_ok=True)
        for k in range(n):
            name = f"{k + 1:07d}.jpg"
            dst = os.path.join(jpeg_dir, name)
            if not os.path.isfile(dst):
                shutil.copyfile(os.path.join(d, name), dst)
            if _sha256(dst) != _sha256(os.path.join(d, name)):
                raise RuntimeError(f"{dst}: copy differs from the staged frame")
        from PIL import Image
        with Image.open(os.path.join(d, "0000001.jpg")) as im:
            w, h = im.size
        rec = {"name": clip, "kind": "edinburgh", "dir": d, "n": n, "prompts": ps, "jpeg_dir": jpeg_dir, "size": [h, w]}
    if any(p.frame != 0 for p in rec["prompts"]) or any(p.kind != "box" for p in rec["prompts"]):
        raise ValueError(f"{clip}: G2-3 needs every object box-prompted at frame 0")
    rec["check_frames"] = [0, rec["n"] // 2]
    ctx.staged[clip] = rec
    return rec


def clip_frames(ctx, clip):
    st = ctx.staged[clip]
    return bc.PigTrackFrames(st["dir"]) if st["kind"] == "pigtrack" else bc.EdinburghFrames(st["dir"])


def check_frame_pils(ctx, clip):
    st = ctx.staged[clip]
    out = []
    it = iter(clip_frames(ctx, clip))
    try:
        for fi, pil, _m in it:
            if fi in st["check_frames"]:
                out.append((f"{clip}_{fi}", fi, pil.copy()))
            if fi >= max(st["check_frames"]):
                break
    finally:
        c = getattr(it, "close", None)          # joins the prefetch thread
        if callable(c):
            c()
    return out


class MaskRecorder:
    """track_stream on_frame: the emitted masks per frame as RLE (the bench output rule: score > 0, non-empty)."""

    def __init__(self):
        self.masks = {}

    def __call__(self, fi, sess, out, mask_logits, binm, rows, live, pruned):
        self.masks[int(fi)] = {int(r.obj_id): bc.rle_encode(np.asarray(bc.to_numpy(binm[r.k]), dtype=bool))
                               for r in rows}


def hf_track(ctx, arm, clip):
    st = ctx.staged[clip]
    cap = MaskRecorder()
    t0 = time.time()
    s = bc.track_stream(arm, clip_frames(ctx, clip), protocol="A", prompts=st["prompts"], keep_last_n=None, pop=True,
                        max_frames=st["n"], mot=bc.MotWriter(), on_frame=cap, progress_every=ctx.P.progress_every,
                        log=ctx.log)
    ctx.hooks["mem_settle"]()
    if s["frames"] != st["n"]:
        raise bc.BenchAssertError(f"{clip}: {s['frames']} frames tracked, expected {st['n']}")
    return cap.masks, {"frames": s["frames"], "forwards": s["forwards"], "objects": s["objects"],
                       "wall_s": round(time.time() - t0, 1), "frame_hashes": s["frame_hashes"],
                       "session_alive_after_teardown": (s.get("teardown") or {}).get("session_alive_after_teardown")}


def run_meta(ctx, job, name):
    """One route-A process (the authors' code). -> result dict (+ 'dir')."""
    run_dir = os.path.join(ctx.local, "meta", name)
    if os.path.isfile(os.path.join(run_dir, "result.json")):
        with open(os.path.join(run_dir, "result.json")) as f:
            res = json.load(f)
        res["dir"] = run_dir
        return res
    os.makedirs(run_dir, exist_ok=True)
    script = os.path.join(run_dir, "meta_route_a.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(render_meta_script())
    job = dict(job, out_dir=run_dir)
    jp = write_json_local(os.path.join(run_dir, "job.json"), job)
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ctx.files["repo"], "sam3"), ctx.meta_deps])
    t0 = time.time()
    r = subprocess.run([sys.executable, script, jp], capture_output=True, text=True, timeout=5400, cwd=run_dir, env=env)
    tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-4000:]
    with open(os.path.join(run_dir, "log.txt"), "w", encoding="utf-8") as f:
        f.write((r.stdout or "") + "\n----- stderr -----\n" + (r.stderr or ""))
    if r.returncode != 0 or not os.path.isfile(os.path.join(run_dir, "result.json")):
        raise RuntimeError(f"route A {name} failed (rc {r.returncode}): {tail}")
    with open(os.path.join(run_dir, "result.json")) as f:
        res = json.load(f)
    res["dir"] = run_dir
    res["wall_s"] = round(time.time() - t0, 1)
    res["stderr_tail"] = (r.stderr or "")[-1500:]
    ctx.log(f"route A {name}: done in {res['wall_s']} s")
    return res


def install_meta_deps(ctx):
    d = os.path.join(ctx.local, "meta_deps")
    if os.path.isfile(os.path.join(d, ".ok")):
        return d
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", d, *META_DEPS],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"pip install of the authors' code deps failed: {(r.stderr or r.stdout)[-800:]}")
    open(os.path.join(d, ".ok"), "w").close()
    return d


def load_meta_masks(res, variant, clip):
    v = res["variants"][variant][clip]
    with open(os.path.join(res["dir"], v["file"])) as f:
        return json.load(f)["masks"], {k: vv for k, vv in v.items() if k != "file"}


def npy(res, fname):
    return np.load(os.path.join(res["dir"], fname))


def hf_processor_record(proc):
    ip = getattr(proc, "image_processor", proc)
    out = {"type": type(ip).__name__}
    for k in ("size", "resample", "image_mean", "image_std", "rescale_factor", "do_resize", "do_rescale",
              "do_normalize", "do_pad", "do_convert_rgb"):
        v = getattr(ip, k, None)
        out[k] = v if isinstance(v, (int, float, str, bool, list, tuple, dict, type(None))) else str(v)
    return json.loads(json.dumps(out, default=str))


# ---------------------------------------------------------------------------------------------------- parity
def do_parity(ctx, fetch):
    """Checks G2-1, 1b, 2, 3, 4 (+ diagnostics). -> (verdicts, results)."""
    torch = bc.torch
    P = ctx.P
    results, verdicts = {}, {}
    work = os.path.join(ctx.local, "work")
    os.makedirs(work, exist_ok=True)
    for c in P.clips:
        stage_clip(ctx, c)
    ctx.meta_deps = install_meta_deps(ctx)
    # ---- the teacher's sam3.pt (gated repo; the J0 pin, else the repo's main, recorded)
    rev = ctx.revisions.get(SAM3_HF_ID)
    try:
        sinfo = hf_file_info(SAM3_HF_ID, SAM3_PT, rev, token=ctx.token)
    except FileNotFoundError:
        sinfo = hf_file_info(SAM3_HF_ID, SAM3_PT, None, token=ctx.token)
    sam3_pt, sam3_sha = hf_download(ctx, sinfo, token=ctx.token)
    teacher_src = {"repo": SAM3_HF_ID, "file": SAM3_PT, "revision": sinfo["revision"], "sha256": sam3_sha,
                   "hf_weights_pin": rev, "same_revision_as_hf_weights": sinfo["revision"] == rev}
    # ---- check frames: the HF processor's pixel_values (as Arm.preprocess makes them) and the ES-L frame
    from transformers import Sam3TrackerVideoProcessor, Sam3TrackerVideoModel
    kw = {k: v for k, v in (("token", ctx.token), ("revision", rev)) if v}
    proc = Sam3TrackerVideoProcessor.from_pretrained(SAM3_HF_ID, **kw)
    tf = es3.make_es3_transform()
    items_hf, items_es3, px = [], [], {}
    for c in P.clips:
        for label, fi, pil in check_frame_pils(ctx, c):
            a = proc(images=pil, device="cuda", return_tensors="pt")["pixel_values"].float().cpu()
            b = tf(pil).unsqueeze(0)
            pa, pb = os.path.join(work, f"px_hf_{label}.pt"), os.path.join(work, f"px_es3_{label}.pt")
            torch.save(a, pa)
            torch.save(b, pb)
            items_hf.append({"label": label, "path": pa})
            items_es3.append({"label": label, "path": pb})
            px[label] = {"clip": c, "fi": fi, "hf": a, "es3": b}
    # ---- G2-4: weights (CPU)
    es3_sd = es3.checkpoint_state_dict(torch.load(ctx.files["ckpt"], map_location="cpu", weights_only=True))
    s3 = torch.load(sam3_pt, map_location="cpu", weights_only=True)
    s3 = s3["model"] if isinstance(s3, dict) and isinstance(s3.get("model"), dict) else s3
    gate_prefixes = ("detector.backbone.vision_backbone.sam2_convs.", "tracker.")
    info_prefixes = ("detector.backbone.vision_backbone.convs.", "detector.backbone.language_backbone.",
                     "detector.transformer.", "detector.segmentation_head.", "detector.geometry_encoder.",
                     "detector.dot_prod_scoring.", "detector.backbone.vision_backbone.trunk.")
    wd = weights_diff(es3_sd, s3, gate_prefixes + info_prefixes)
    other = sorted({".".join(k.split(".")[:2]) for k in set(es3_sd) | set(s3)
                    if not any(k.startswith(p) for p in gate_prefixes + info_prefixes)})
    v4 = "PASS" if all(wd[p]["bit_identical"] for p in gate_prefixes) else "FAIL"
    results["g2_4"] = {"check": "G2-4", "a": "ES-TV-L checkpoint", "b": teacher_src, "gate_prefixes": gate_prefixes,
                       "diff": wd, "other_top_level_prefixes": other, "verdict": v4,
                       "n_a": len(es3_sd), "n_b": len(s3)}
    verdicts["G2-4 weights (sam2_convs, tracker)"] = v4
    del es3_sd, s3
    gc.collect()
    # ---- G2-1 HF side: fp32 HF backbone on the HF pixels (TF32 off)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m32 = Sam3TrackerVideoModel.from_pretrained(SAM3_HF_ID, torch_dtype=torch.float32, attn_implementation="sdpa",
                                                **kw).to("cuda").eval()
    hf32 = {}
    with torch.no_grad():
        for label, d in px.items():
            y = m32.vision_encoder.backbone(d["hf"].to("cuda")).last_hidden_state
            hf32[label] = y.reshape(1, 72, 72, -1).permute(0, 3, 1, 2).float().cpu().numpy()
    del m32
    ctx.hooks["mem_settle"]()
    # ---- G2-1b production side: the encoder from models.es3_trunk, fp32 and bf16, on the ES-L frames
    tv = es3.load_upstream_tinyvit(ctx.files["tinyvit"])
    payload = es3.read_trunk_payload(ctx.files["trunk"])
    prod = {}
    for prec, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        enc, _m = es3.load_es3_encoder(payload["state_dict"], tv, device="cuda", dtype=dt)
        with torch.no_grad():
            for label, d in px.items():
                prod[(prec, label)] = enc(d["es3"].to("cuda", dt)).float().cpu().numpy()
        del enc
    del payload
    ctx.hooks["mem_settle"]()
    # ---- route A processes
    clips_job = [{"name": c, "jpeg_dir": ctx.staged[c]["jpeg_dir"],
                  "prompts": [[int(p.obj_id)] + [float(v) for v in p.box] for p in ctx.staged[c]["prompts"]],
                  "dump_images": ctx.staged[c]["check_frames"]} for c in P.clips]
    meta_t = run_meta(ctx, {"model": "sam3", "ckpt": sam3_pt, "trunk_inputs": items_hf,
                            "variants": ["teacher"] if "teacher" in P.diagnostics else [], "clips": clips_job},
                      "teacher")
    meta_e = run_meta(ctx, {"model": "es3", "ckpt": ctx.files["ckpt"], "trunk_inputs": items_es3,
                            "variants": [v for v in ("asis", "td") if v in P.diagnostics], "clips": clips_job}, "es3")
    # ---- G2-1 verdict
    rows1 = {lb: feature_compare(hf32[lb], npy(meta_t, meta_t["trunk"]["teacher"]["fp32"][lb])) for lb in px}
    v1 = feature_verdict(rows1, TEACHER_COS_MIN, TEACHER_RELL2_MAX)
    results["g2_1"] = {"check": "G2-1", "test": "HF vision_encoder.backbone fp32", "ref": "authors' ViT trunk fp32",
                       "teacher_source": teacher_src, "frames": rows1, "coverage": meta_t["coverage"],
                       "authors_autocast_vs_fp32": {lb: feature_compare(
                           npy(meta_t, meta_t["trunk"]["teacher"]["authors"][lb]),
                           npy(meta_t, meta_t["trunk"]["teacher"]["fp32"][lb])) for lb in px},
                       "thresholds": [TEACHER_COS_MIN, TEACHER_RELL2_MAX], "verdict": v1}
    verdicts["G2-1 teacher trunk"] = v1
    # ---- G2-1b verdict (+ the stale-cache diagnosis of the authors' video builder)
    rows1b = {lb: feature_compare(prod[("fp32", lb)], npy(meta_e, meta_e["trunk"]["fixed"]["fp32"][lb])) for lb in px}
    v1b = feature_verdict(rows1b, ES3_COS_MIN, ES3_RELL2_MAX)
    results["g2_1b"] = {
        "check": "G2-1b", "test": "production ES-L encoder fp32 (tol_es3, models.es3_trunk)",
        "ref": "authors' model trunk fp32 after model.eval()", "frames": rows1b,
        "thresholds": [ES3_COS_MIN, ES3_RELL2_MAX], "verdict": v1b,
        "diag_bf16_production_vs_authors_autocast": {lb: feature_compare(
            prod[("bf16", lb)], npy(meta_e, meta_e["trunk"]["fixed"]["authors"][lb])) for lb in px},
        "diag_bf16_vs_fp32_production": {lb: feature_compare(prod[("bf16", lb)], prod[("fp32", lb)]) for lb in px},
        "diag_asis_vs_fixed_authors_fp32": {lb: feature_compare(
            npy(meta_e, meta_e["trunk"]["asis"]["fp32"][lb]), npy(meta_e, meta_e["trunk"]["fixed"]["fp32"][lb]))
            for lb in px},
        "stale_attention_bias_cache": {"as_built": meta_e.get("stale_as_built"),
                                       "after_model_eval": meta_e.get("stale_after_eval")},
        "coverage": meta_e["coverage"], "modules_in_training_mode": meta_e.get("modules_in_training_mode")}
    verdicts["G2-1b ES3 encoder"] = v1b
    # ---- G2-2 preprocessing
    rows2 = {}
    for lb, d in px.items():
        ref = npy(meta_e, meta_e["images"][d["clip"]][str(d["fi"])])
        mine = d["es3"][0].numpy()
        rows2[lb] = {"max_abs": float(np.abs(mine.astype(np.float64) - ref.astype(np.float64)).max()),
                     "identical": bool(np.array_equal(mine, ref)), "shape": list(ref.shape),
                     "hf_minus_es3_mean_abs": float(np.abs(d["hf"][0].numpy() - mine).mean())}
    v2 = "PASS" if rows2 and all(r["max_abs"] <= PIXEL_MAXABS_MAX and r["shape"] == [3, es3.IMAGE_SIZE,
                                                                                         es3.IMAGE_SIZE]
                                 for r in rows2.values()) else "FAIL"
    results["g2_2"] = {"check": "G2-2", "frames": rows2, "hf_processor": hf_processor_record(proc),
                       "es3_frame": {"resize": "PIL Image.resize((1008, 1008)) default filter (BICUBIC)",
                                     "scale": "/255", "mean": es3.IMG_MEAN, "std": es3.IMG_STD,
                                     "source": "efficientsam3 sam2_utils.load_video_frames defaults"},
                       "threshold": PIXEL_MAXABS_MAX, "verdict": v2}
    verdicts["G2-2 preprocessing"] = v2
    del proc
    # ---- HF tracking: T (diagnostic), then ES-L (route B, = tol_bench_core.load_arm('ES-L') after G2)
    hf = {}
    if "teacher" in P.diagnostics:
        arm = ctx.hooks["arm_loader"](ctx, "T")
        try:
            for c in P.clips:
                hf[("T_hfpx", c)] = hf_track(ctx, arm, c)
            raw_proc = arm.proc
            arm.proc = es3.ES3Processor(raw_proc, es3.make_es3_transform())
            for c in P.clips:
                hf[("T_authorspx", c)] = hf_track(ctx, arm, c)
            arm.proc = raw_proc
        finally:
            del arm
            ctx.hooks["mem_settle"]()
    arm = ctx.hooks["es3_loader"](ctx)
    try:
        es3_info = dict(arm.info)
        for c in P.clips:
            hf[("ESL", c)] = hf_track(ctx, arm, c)
        if "hfpix" in P.diagnostics:
            wrapped = arm.proc
            arm.proc = wrapped._proc
            for c in P.clips:
                hf[("ESL_hfpx", c)] = hf_track(ctx, arm, c)
            arm.proc = wrapped
    finally:
        del arm
        ctx.hooks["mem_settle"]()
    # ---- G2-3 per clip
    cov = meta_e["coverage"]["missing_tracker_or_vision"]
    for c in P.clips:
        a_masks, a_meta = load_meta_masks(meta_e, "fixed", c)
        b_masks, b_meta = hf[("ESL", c)]
        cmp = parity_compare(b_masks, a_masks)
        diag = {}

        def _d(name, test, ref, _diag=diag):
            _diag[name] = {k: v for k, v in parity_compare(test, ref).items() if k != "per_object_mean_iou"}
        if "asis" in P.diagnostics:
            _d("route_A_as_shipped_vs_route_B", b_masks, load_meta_masks(meta_e, "asis", c)[0])
            _d("route_A_as_shipped_vs_route_A_fixed", load_meta_masks(meta_e, "asis", c)[0], a_masks)
        if "td" in P.diagnostics:
            _d("route_A_memory_selection_vs_route_B", b_masks, load_meta_masks(meta_e, "fixed_td", c)[0])
        if "hfpix" in P.diagnostics:
            _d("route_B_hf_pixels_vs_route_A", hf[("ESL_hfpx", c)][0], a_masks)
        if "teacher" in P.diagnostics:
            tm = load_meta_masks(meta_t, "teacher", c)[0]
            _d("teacher_HF_authors_pixels_vs_authors_code", hf[("T_authorspx", c)][0], tm)
            _d("teacher_HF_bench_T_vs_authors_code", hf[("T_hfpx", c)][0], tm)
            _d("ESL_vs_bench_T", b_masks, hf[("T_hfpx", c)][0])
        ok = cmp["pass"] and not cov
        v3 = "PASS" if ok else "FAIL"
        results[f"g2_3_{c}"] = {"check": "G2-3", "clip": c, "frames": ctx.staged[c]["n"],
                                "objects": len(ctx.staged[c]["prompts"]), "route_a": a_meta,
                                "route_b": {k: v for k, v in b_meta.items() if k != "frame_hashes"},
                                "route_b_frame_hashes": b_meta.get("frame_hashes"), "compare": cmp,
                                "authors_build_missing_tracker_or_vision_tensors": cov, "diagnostics": diag,
                                "threshold": PARITY_IOU_MIN, "verdict": v3}
        verdicts[f"G2-3 parity {c}"] = v3
        publish_gz(ctx, f"masks_route_a_{c}.json.gz", a_masks)
        publish_gz(ctx, f"masks_route_b_{c}.json.gz", {str(k): {str(o): r for o, r in d.items()}
                                                       for k, d in b_masks.items()})
    results["_arm_info"] = {k: es3_info.get(k) for k in ("arm", "family", "hf_id", "commit_hash", "attn_request",
                                                         "params", "es3", "pixel_values", "weights_mem_mib")}
    results["_route_a_runs"] = {n: {k: v for k, v in r.items() if k in ("seconds", "wall_s", "events", "torch",
                                                                        "sam3_file", "builder_stdout_tail",
                                                                        "stderr_tail")}
                                for n, r in (("teacher", meta_t), ("es3", meta_e))}
    return verdicts, results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production hooks and the driver

# COMMAND ----------


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cuda_probe(retries=4, sleep_s=60):
    """= Tol_60.cuda_probe: a subprocess CUDA start-up probe before this process touches CUDA."""
    code = ("import torch; torch.zeros(1, device='cuda'); torch.cuda.synchronize(); "
            "print(torch.cuda.get_device_name(0))")
    last = None
    for k in range(retries):
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                return {"ok": True, "attempt": k + 1, "gpu": (r.stdout.strip().splitlines() or ["?"])[-1]}
            last = (r.stderr or r.stdout or "")[-600:]
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        _log(f"CUDA probe attempt {k + 1}/{retries} failed: {last}")
        if k + 1 < retries:
            time.sleep(sleep_s)
    raise RuntimeError(f"CUDA start-up failed {retries}x: {last}")


def gpu_preflight(P):
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError("PYTORCH_CUDA_ALLOC_CONF lacks expandable_segments:True (job cluster spec)")
    probe = cuda_probe()
    import torch
    name = torch.cuda.get_device_name(0)
    up = name.upper()
    cls = "A100" if "A100" in up else ("A10" if re.search(r"A10(?!\d)", up) else name)
    if P.require_gpu != "any" and cls != P.require_gpu:
        raise RuntimeError(f"GPU {name} (class {cls}) but require_gpu={P.require_gpu}")
    return {"gpu": name, "class": cls, "probe": probe}


def _prod_arm_loader(ctx, name):
    return bc.load_arm(name, device="cuda", hf_token=ctx.token, revisions=ctx.revisions, registry=ctx.R,
                       dbutils=ctx.dbutils, log=ctx.log)


def _prod_es3_loader(ctx):
    """Route B = the production ES-L arm (tol_es3.load_es3_arm: load_arm('T') + the swap), from the local fetch files."""
    f = ctx.fetch
    return es3.load_es3_arm(bc, trunk_local=ctx.files["trunk"], tinyvit_local=ctx.files["tinyvit"],
                            trunk_sha256=f["trunk"]["sha256"], tinyvit_sha256=es3.ES3_UPSTREAM_SHA256[es3.TINYVIT_UPSTREAM],
                            g2_status=f"pending (Tol_61 G2 parity run, tag {ctx.P.tag})", device="cuda",
                            hf_token=ctx.token, revisions=ctx.revisions, registry=ctx.R, dbutils=ctx.dbutils,
                            log=ctx.log)


def _prod_token(dbu):
    tok = bc.hf_token_from_dbutils(dbu)
    os.environ["HF_TOKEN"] = tok             # Tol_68 / J3e: the runtime's from_pretrained wrapper may ignore token=
    return tok


def resolve_revisions(R, spec):
    if not spec:
        return {}
    if spec.startswith("protocol:"):
        with open(bc.protocol_file_path(R, spec[len("protocol:"):]), encoding="utf-8") as f:
            obj = json.load(f)
        obj = obj.get("revisions", obj)
    else:
        obj = json.loads(spec)
    return dict(sorted(obj.items()))


PROD_HOOKS = {"local_root": "/local_disk0/tol61", "arm_loader": _prod_arm_loader, "es3_loader": _prod_es3_loader,
              "gpu_preflight": gpu_preflight, "hf_token": _prod_token, "mem_settle": lambda: bc.settle_memory(),
              "fetch": do_fetch, "parity": do_parity, "log": _log}


def main(raw, *, dbutils_=None, R=None, hooks=None):
    h = dict(PROD_HOOKS)
    h.update(hooks or {})
    P = parse_params(raw)
    R = R if R is not None else tol_paths.load(version=1)
    t0 = time.time()
    ctx = types.SimpleNamespace(P=P, R=R, dbutils=dbutils_, hooks=h, log=h["log"], local=h["local_root"], staged={},
                                token=None, files={}, fetch=None, meta_deps=None,
                                revisions=resolve_revisions(R, P.revisions_spec))
    done = load_done(ctx, "G2_SUMMARY.json")
    if done is not None and P.mode in ("all", "parity"):
        ctx.log("G2_SUMMARY.json already published: nothing to do")
        return {"status": "done", "verdict": done.get("verdict"), "verdicts": done.get("verdicts")}
    if P.mode in ("all", "parity"):
        ev_dir = os.path.dirname(out_path(ctx, "G2_SUMMARY.json"))
        partial = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ev_dir, "g2_*.json*"))
                         + glob.glob(os.path.join(ev_dir, "masks_route_*.json.gz")))
        if partial:
            raise RuntimeError(f"tag {P.tag}: parity evidence {partial[:4]} exists without G2_SUMMARY.json (an "
                               "interrupted publish); parity outputs are never overwritten -- use a new tag")
    probe = h["gpu_preflight"](P) if h.get("gpu_preflight") else {"skipped": True}
    ctx.token = h["hf_token"](dbutils_) if h.get("hf_token") else None
    ctx.fetch = h["fetch"](ctx)
    if P.mode == "fetch":
        return {"status": "fetched", "trunk_sha256": ctx.fetch["trunk"]["sha256"],
                "checkpoint_sha256": ctx.fetch["checkpoint"]["sha256"]}
    if not ctx.files:
        ctx.files = stage_fetch_outputs(ctx, ctx.fetch)
    if "repo" not in ctx.files:
        ctx.files["repo"] = fetch_repo_at_commit(ctx)[0]
    verdicts, results = h["parity"](ctx, ctx.fetch)
    for name, r in results.items():
        if name.startswith("_"):
            continue
        r.update({"notebook": f"{NB_NAME} {NB_VERSION}", "core_sha256": _sha256(bc.__file__),
                  "es3_sha256": _sha256(es3.__file__), "created_utc": _now()})
        publish(ctx, f"{name}.json", r)
    verdict = aggregate(verdicts)
    summary = {"notebook": f"{NB_NAME} {NB_VERSION}", "tag": P.tag, "verdict": verdict, "verdicts": verdicts,
               "canonical": P.canonical, "params": vars(P), "probe": probe, "fetch": {
                   k: ctx.fetch.get(k) for k in ("checkpoint", "trunk", "efficientsam3", "upstream_sha256")},
               "arm_info": results.get("_arm_info"), "route_a_runs": results.get("_route_a_runs"),
               "core_sha256": _sha256(bc.__file__), "es3_sha256": _sha256(es3.__file__),
               "versions": bc.versions() if bc.TORCH_OK else None, "gpu": bc.gpu_info() if bc.TORCH_OK else None,
               "created_utc": _now(), "wall_s": round(time.time() - t0, 1)}
    publish(ctx, "G2_SUMMARY.json", summary)
    proto = None
    if P.write_protocol:
        rec = g2_record(P, verdicts, ctx.fetch, evidence_rel(ctx),
                        {"created_utc": _now(), "core_sha256": _sha256(bc.__file__),
                         "es3_sha256": _sha256(es3.__file__)})
        probs = es3.validate_g2(rec)
        if probs:
            raise RuntimeError(f"G2 record malformed: {probs}")
        loc = write_json_local(os.path.join(ctx.local, "out", es3.G2_FILE), rec)
        proto = bc.protocol_file_path(R, es3.G2_FILE)
        bc.publish_no_overwrite(loc, proto, ctx.dbutils)
    ctx.log("G2 VERDICT " + verdict + " " + json.dumps(verdicts))
    return {"status": "done", "verdict": verdict, "verdicts": verdicts, "protocol_file": proto,
            "wall_s": summary["wall_s"]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local selftest (`TOL61_SELFTEST=1`)

# COMMAND ----------


def selftest():
    res = []

    def check(name, cond, detail=""):
        res.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + ("" if cond or not detail else f"  [{detail}]"), flush=True)

    def raises(fn, exc=Exception, match=None):
        try:
            fn()
        except exc as e:  # noqa: BLE001
            return match is None or re.search(match, str(e)) is not None
        return False

    # ---- params
    P = parse_params({})
    check("params: defaults = the canonical, protocol-writing configuration",
          P.canonical and P.write_protocol and P.clips == list(GATE_CLIPS) and P.edi_frames == 300
          and P.diagnostics == list(DIAGNOSTICS))
    Ps = parse_params({"tag": "smoke_es3", "clips": "pigtrack0055"})
    check("params: a smoke tag never writes the protocol file", not Ps.canonical and not Ps.write_protocol)
    check("params: validation", raises(lambda: parse_params({"clips": "pigtrack0001"}), ValueError)
          and raises(lambda: parse_params({"edi_frames": "301"}), ValueError)
          and raises(lambda: parse_params({"diagnostics": "foo"}), ValueError)
          and raises(lambda: parse_params({"tag": "x", "write_protocol": "true"}), ValueError, "canonical")
          and raises(lambda: parse_params({"es3_revision": "main"}), ValueError)
          and raises(lambda: parse_params({"bogus": "1"}), ValueError)
          and raises(lambda: parse_params({"tag": "../x"}), ValueError))
    # ---- feature comparison and verdicts
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1, 8, 4, 4)).astype(np.float32)
    fc = feature_compare(a, a)
    check("feature_compare: identical", fc["identical"] and fc["rel_l2"] == 0.0 and fc["cos_global"] > 0.999999)
    b = a + 1e-3 * rng.normal(size=a.shape).astype(np.float32)
    fb = feature_compare(b, a)
    check("feature_compare: small noise -> rel L2 ~1e-3, cosine ~1", 1e-4 < fb["rel_l2"] < 1e-2
          and fb["cos_global"] > 0.9999)
    check("feature_verdict: thresholds on every frame",
          feature_verdict({"f0": fc, "f1": fb}, 0.9999, 1e-2) == "PASS"
          and feature_verdict({"f0": fc, "f1": fb}, 0.9999, 1e-5) == "FAIL" and feature_verdict({}, 0, 1) == "FAIL"
          and feature_verdict({"x": feature_compare(a, a[:, :4])}, 0, 1) == "FAIL")
    # ---- parity comparison
    m1 = np.zeros((10, 10), bool)
    m1[2:6, 2:6] = True
    m2 = np.zeros((10, 10), bool)
    m2[2:6, 2:8] = True
    r1, r2 = bc.rle_encode(m1), bc.rle_encode(m2)
    pc = parity_compare({0: {0: r1, 1: r1}, 1: {0: r1}}, {"0": {"0": r1, "1": r1}, "1": {"0": r1}})
    check("parity_compare: identical (string keys from route A normalised) -> 1.0, pass",
          pc["pass"] and pc["mask_iou"]["mean"] == 1.0 and pc["frame0_iou"] == {"0": 1.0, "1": 1.0})
    pc2 = parity_compare({0: {0: r2}, 1: {}}, {0: {0: r1}, 1: {0: r1}})
    check("parity_compare: one-sided emission counts 0; partial overlap = IoU",
          not pc2["pass"] and pc2["one_sided_object_frames"] == 1 and abs(pc2["mask_iou"]["mean"] - (16 / 24) / 2) < 1e-6)
    # ---- weights diff
    A = {"tracker.a": np.ones(3, np.float32), "tracker.b": np.zeros(2, np.float32),
         "detector.backbone.vision_backbone.sam2_convs.0.w": np.ones(2, np.float32)}
    B = dict(A)
    wd = weights_diff(A, B, ("tracker.", "detector.backbone.vision_backbone.sam2_convs."))
    check("weights_diff: identical groups are bit_identical", all(v["bit_identical"] for v in wd.values()))
    B2 = dict(A, **{"tracker.b": np.array([0, 1e-7], np.float32)})
    B2.pop("tracker.a")
    wd2 = weights_diff(A, B2, ("tracker.",))["tracker."]
    check("weights_diff: a changed value and a missing key are caught", not wd2["bit_identical"]
          and wd2["n_only_a"] == 1 and wd2["n_differ"] == 1 and wd2["max_abs_diff"] > 0)
    check("weights_diff: an empty group is not 'identical'", not weights_diff({}, {}, ("tracker.",))["tracker."][
        "bit_identical"])
    # ---- aggregation and the protocol record
    check("aggregate: PASS only if every check passes", aggregate({"a": "PASS", "b": "PASS"}) == "PASS"
          and aggregate({"a": "PASS", "b": "FAIL"}) == "FAIL" and aggregate({}) == "FAIL")
    fetch = {"trunk": {"sha256": "c" * 64, "encoder": {"n_params": 30650496}},
             "checkpoint": {"hf_repo": es3.ES3_HF_REPO, "sha256": "d" * 64}}
    rec = g2_record(P, {"G2-1": "PASS", "G2-3 x": "PASS"}, fetch, "paper1_distillation/evidence/g2/g2_v1", {})
    check("g2_record: well-formed for tol_es3.validate_g2 and passes its gate", es3.validate_g2(rec) == []
          and es3.g2_gate(rec)["verdict"] == "PASS", str(es3.validate_g2(rec)))
    recf = g2_record(P, {"G2-1": "PASS", "G2-3 x": "FAIL"}, fetch, "x", {})
    check("g2_record: a failed check -> FAIL record that the gate refuses (R-scope)",
          recf["verdict"] == "FAIL" and raises(lambda: es3.g2_gate(recf), NotImplementedError, "R-scope"))
    check("g2_record: tiny_vit relpath lies under third_party/efficientsam3/src_<commit12>/",
          rec["tinyvit_source"]["relpath"] == f"third_party/efficientsam3/src_{es3.ES3_COMMIT[:12]}/tiny_vit.py")
    # ---- the route-A script and the notice
    try:
        compile(render_meta_script(), "meta_route_a.py", "exec")
        ok = True
    except SyntaxError as e:
        ok = repr(e)
    check("route-A script renders and compiles", ok is True, str(ok))
    s = render_meta_script()
    check("route-A script: the authors' builder, no memory selection for the gate, eval() before 'fixed'",
          "apply_temporal_disambiguation=False" in s and s.index("model.eval()") < s.index('track("fixed"')
          and 'track("asis"' in s and "propagate_preflight=True" in s and "use_memory_selection = True" in s)
    n = NOTICE.format(nb="x", date="2026-10-15", repo="r", file="f", rev="v", sha="s", size=1, card_license="l",
                      code_url="u", commit="c", src_dir="d", prefix="p")
    check("NOTICE renders and says never redistribute", "Do not redistribute" in n and "SAM License" in n)
    # ---- registry-derived model directories (temporary registry root)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = tmp.replace("\\", "/")
        reg_yaml = os.environ.get("TOL61_REGISTRY_YAML") or os.path.join(os.path.dirname(tol_paths.__file__),
                                                                         "paths_v1.yaml")
        R0 = tol_paths.load(path=reg_yaml)
        data = json.loads(json.dumps(R0.data))
        data["roots"].update(volume=tmp + "/vol", mount=tmp + "/mnt", mount_uri="file:" + tmp + "/mnt")
        R = tol_paths.Registry(data, source=reg_yaml, sha256=R0.sha256)
        md = model_dirs(R)
        check("model_dirs: derived from models.es3_trunk (models/baselines/...)",
              md["trunk"].endswith("/models/baselines/es3_trunk_tinyvit_l.pt")
              and md["third_party"].endswith("/models/baselines/third_party/efficientsam3")
              and md["src"].endswith(f"/third_party/efficientsam3/src_{es3.ES3_COMMIT[:12]}"))
        ctx = types.SimpleNamespace(P=P, R=R, local=tmp + "/local", dbutils=None)
        check("evidence paths live under paper1_distillation/evidence/g2/<tag>/",
              out_path(ctx, "fetch.json").endswith("/paper1_distillation/evidence/g2/g2_v1/fetch.json")
              and raises(lambda: out_path(ctx, "../x.json"), ValueError))
        # the guarded models/ publish: inside the allowed dir only, never overwriting
        loc = os.path.join(tmp, "f.bin")
        with open(loc, "wb") as f:
            f.write(b"x")
        check("publish_file_guarded: models/ target ok, identical re-publish ok, other dir refused",
              dc.publish_file_guarded(loc, md["third_party"] + "/f.bin", md["third_party"]) == "copied"
              and dc.publish_file_guarded(loc, md["third_party"] + "/f.bin", md["third_party"]) == "identical"
              and raises(lambda: dc.publish_file_guarded(loc, md["base"] + "/f.bin", md["third_party"]),
                         PermissionError))
        # the driver with stub stages: verdict aggregation, publishing, protocol file, resume
        calls = collections.Counter()

        def fake_fetch(c):
            calls["fetch"] += 1
            c.files = {"repo": "r"}
            return fetch

        def fake_parity(c, f):
            calls["parity"] += 1
            return ({"G2-1 teacher trunk": "PASS", "G2-3 parity pigtrack0055": "PASS"},
                    {"g2_1": {"verdict": "PASS"}, "_arm_info": {"arm": "ES-L"}})
        revp = bc.protocol_file_path(R, "hf_revisions_v1.json")
        os.makedirs(os.path.dirname(revp), exist_ok=True)
        with open(revp, "w", encoding="utf-8") as fh:
            json.dump({"revisions": {SAM3_HF_ID: "3c879f39826c281e95690f02c7821c4de09afae7"}}, fh)
        hooks = {"local_root": tmp + "/local", "gpu_preflight": None, "hf_token": None, "fetch": fake_fetch,
                 "parity": fake_parity, "log": lambda *a: None, "mem_settle": lambda: 0.0}
        out = main({}, R=R, hooks=hooks)
        pf = bc.protocol_file_path(R, es3.G2_FILE)
        with open(pf, encoding="utf-8") as f:
            prec = json.load(f)
        check("driver (stubs): PASS published, protocol file written and accepted by the ES-L gate; revisions pinned",
              out["verdict"] == "PASS" and calls["fetch"] == 1 and os.path.isfile(pf) and es3.g2_gate(prec)["verdict"] == "PASS"
              and os.path.isfile(out_path(ctx, "g2_1.json")) and os.path.isfile(out_path(ctx, "G2_SUMMARY.json")))
        out2 = main({}, R=R, hooks=hooks)
        check("driver (stubs): a published summary is not recomputed", out2["status"] == "done"
              and calls["parity"] == 1)
        out3 = main({"tag": "smoke_x", "clips": "pigtrack0055"}, R=R, hooks=hooks)
        check("driver (stubs): a smoke tag writes no protocol file", out3["protocol_file"] is None)
    # ---- static audit
    with open(os.path.abspath(__file__), "rb") as fh:
        src = fh.read().decode("utf-8")
    check("audit: no literal Volume / dbfs path; HF token only through tol_bench_core",
          ("/Vol" + "umes") not in src and ("/db" + "fs") not in src and "secrets." + "get(" not in src)
    api = bc.api_summary()
    used = sorted(set(re.findall(r"\bbc\.([A-Za-z_][A-Za-z0-9_]*)", src)))
    off = [n for n in used if not (n.isupper() or n.startswith("_") or n in ("api_summary", "torch")
                                   or re.search(r"\b" + re.escape(n) + r"\b", api))]
    check("audit: tol_bench_core used through its public API", not off, str(off))
    n_fail = sum(1 for r in res if not r)
    print(f"\n{NB_NAME} selftest: {len(res) - n_fail}/{len(res)} passed")
    return n_fail


if SELFTEST:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(1 if selftest() else 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------


def read_widgets():
    out = {}
    for name, default in WIDGETS.items():
        if IS_NOTEBOOK:
            try:
                dbutils.widgets.text(name, default)
            except Exception:  # noqa: BLE001
                pass
            out[name] = dbutils.widgets.get(name) or default
        else:
            out[name] = os.environ.get("TOL61_" + name.upper(), default)
    return out


RAW = read_widgets()
print("widgets:", json.dumps(RAW, sort_keys=True))
SUMMARY = main(RAW, dbutils_=dbutils)
print(json.dumps(SUMMARY, indent=1, default=str))

# COMMAND ----------

if IS_NOTEBOOK:
    dbutils.notebook.exit(json.dumps(SUMMARY, default=str))
