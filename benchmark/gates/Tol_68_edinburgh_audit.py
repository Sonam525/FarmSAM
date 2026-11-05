# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N68 — Edinburgh audit (Paper 1 bench, job **J3**, **A10**)
# MAGIC
# MAGIC BENCH_PROTOCOL.md b1.0 §13 ("J3(c) reproduce-old") + PLAN §A5 `Tol_68` + review R0.2. Four stages, each resumable,
# MAGIC each published ONCE under `paper1_distillation/evidence/edinburgh_audit/<tag>/` (registry key `paper1.root`):
# MAGIC
# MAGIC | stage | what | GPU |
# MAGIC |---|---|---|
# MAGIC | **a** seeds | every `initial_prompts/<clip>/video1/BoundingBoxes_<clip>.txt` vs frame 0 of `annotated/<date>/<seq>/output.json`: exact x,y,w,h, line order = GT object order, first keyframe = frame 0, `isGroundTruth`; strays; frame count / max GT frame of the test clips | no |
# MAGIC | **b** reproduce-old | the SAM 3 teacher (**T**) and the published student (**S-Edi** = `models.student_edinburgh`, EMA epoch 75) under the OLD protocol on the 4 test clips; per-frame boxes regenerated in the old format; scored with the verbatim old scorer; teacher boxes compared frame by frame with the surviving `Teacher_Annotations` | yes |
# MAGIC | **c** GroupNorm | train/eval max diff of the published student with `drop_path_rate=0.0` vs the timm default | small |
# MAGIC | **d** fidelity | per-patch + global cosine, scale ratio, MSE, distillation loss of the published student vs the stored teacher targets on frames of the 4 TEST clips; bridge rows: val clip (vs the training history) and the paper's 20 training frames (vs cell 21) | small |
# MAGIC
# MAGIC **The OLD protocol (b)** is copied from the two cells that produced the published numbers:
# MAGIC
# MAGIC | | T | S-Edi |
# MAGIC |---|---|---|
# MAGIC | source | `(Clone) SAM3_Baseline + feature extraction` cell 17 (NEW-EXTRACTION; wrote the surviving `Teacher_Annotations` = 93.97 / 96.98) | `Student+TeacherDecoder_TinyViT_Fixed` cell 26 (the published 92.29 / 96.15) |
# MAGIC | model | `Sam3TrackerVideoModel` bf16, default attention (no `attn_implementation`) | same model; `vision_encoder.backbone` swapped for the cell-26 `DummyBackbone`, teacher backbone parked on the CPU |
# MAGIC | student | — | EMA weights, **fp32**, eval, export transform (`Resize((1024,1024))`, TinyViT-512 mean/std) |
# MAGIC | frame index / pixel cache | implicit (`len(processed_frames)`), never popped | same |
# MAGIC | pruning | **none**; `gc.collect()` + `empty_cache()` every 100 frames | `keep_last_n=8` every 25 frames (+ gc + empty_cache) |
# MAGIC | prompts | the BoundingBoxes txt, names sorted -> obj ids 0..7, ONE `add_inputs` at frame 0 with `obj_ids.copy()` | same |
# MAGIC | output | `sigmoid > 0.5`, tight box `[x0, y0, x1-x0, y1-y0]`, `[0,0,0,0]` for an empty mask, every object every frame | same |
# MAGIC | scorer | `evaluate_tracking_quality`: hold GT, `range(600)`, IoU 0.5, py-motmetrics 1.4.0 `motchallenge_metrics` | same |
# MAGIC
# MAGIC The teacher's published run was **unpruned** (only the student was pruned to 8), so the teacher row reproduces the
# MAGIC unpruned run; set `teacher_k8_row=true` for an extra, non-gated row of the teacher under the student's k8 schedule.
# MAGIC
# MAGIC **Acceptance (pre-registered here, BENCH_PROTOCOL §13):** per arm and clip, |ΔFP| ≤ `tol_boxes` (5) and |ΔFN| ≤ 5
# MAGIC against the published per-clip counts, and ΔIDSW = 0. T must pass for the environment to count as a faithful
# MAGIC replication; if T passes and S-Edi fails, **stop**: S-Edi's Edinburgh numbers are not interpreted until the
# MAGIC checkpoint question is resolved with the user. A failed verdict is a result, not a job error.
# MAGIC
# MAGIC **Execution order** is fixed by the cells: a, c, d (minutes), then b (about 1 h on an A10). A stage that raises is
# MAGIC recorded and the next stage still runs; the notebook raises at the end if anything failed.
# MAGIC
# MAGIC **Outputs** (`paper1_distillation/evidence/edinburgh_audit/<tag>/`, each file published once):
# MAGIC `AUDIT_CONFIG.json` (code sha256s, settings, pinned versions, GPU, registry provenance; a resume must match it) ·
# MAGIC `a_seeds/seed_vs_gt_frame0.{json,txt}` · `b_reproduce_old/<arm>__oldprot/<clip>/annotations_id <k>.json` (old
# MAGIC format) + `mot/<clip>.txt` (bench MOT convention, empty masks dropped) + `telemetry/<clip>.json` (VRAM every 50
# MAGIC frames, peak vs published, empty masks, processed_frames / non-cond counts) + `T__oldprot/target_consistency/` ·
# MAGIC `b_reproduce_old/{scores,verdict,teacher_vs_surviving,sam3_load,student_load}.json` ·
# MAGIC `c_groupnorm/groupnorm_train_eval.json` · `d_fidelity/{fidelity.json,per_frame.csv}` ·
# MAGIC `AUDIT_SUMMARY.{json,md}` (only once all four stages exist) · `runs/<utc>.json` (one per run) · `.done_*` markers.
# MAGIC A reduced run (`max_frames` < 600 or a clip subset) needs a tag starting with `smoke` and gives no verdict.
# MAGIC
# MAGIC **Deploy:** this file as a SOURCE notebook; `tol_bench_core.py` and the registry's `tol_paths.py` as RAW files in
# MAGIC the same Workspace folder (the notebook checks that `tol_paths.py` equals the registry copy).
# MAGIC
# MAGIC **Rules.** Paths only through the registry (`tol_paths`, deployed RAW beside this notebook with `tol_bench_core.py`);
# MAGIC writes only under `paper1_distillation/evidence/edinburgh_audit/<tag>/`, never over existing content
# MAGIC (`publish_no_overwrite`); HF token only from the secret scope; never run the old Paper 1 notebooks.
# MAGIC Local check (no torch / GPU / Volume): `TOL68_SELFTEST=1 python Tol_68_edinburgh_audit.py`.
# MAGIC
# MAGIC **Environment diagnostics** (`tol68-2026-10-14.3`): widget `transformers_pin` (default `5.2.0` = the b1.0 pin;
# MAGIC `5.3.0` = the published student's kernel) selects the transformers version the `%pip` cell installs (through a
# MAGIC requirements file written by the first code cell; every other pin is unchanged), and `expect_torch` (e.g.
# MAGIC `2.6.0`, empty = no check) asserts the runtime's torch. A run with a non-default pin, or any run whose tag starts
# MAGIC with `smoke_envdiag`, is an **environment diagnostic**: the tag must start with `smoke_envdiag`, the per-clip rows
# MAGIC are reported, and the overall verdict is `ENV_DIAG` (never PASS / STOP). A non-diagnostic run must be on DBR 17.3.
# MAGIC
# MAGIC **BENCH_PROTOCOL b1.1 #2** (`tol68-2026-10-15.1`; user-approved amendment of 2026-10-14): widget `b11_diag_tags` (a
# MAGIC comma list of PUBLISHED `smoke_envdiag*` tags; only for a complete protocol run) makes stage (b) read those tags'
# MAGIC `AUDIT_CONFIG.json` + `b_reproduce_old/verdict.json` (read-only) and apply the amended student rule to a b1.0
# MAGIC `STOP_STUDENT`: S-Edi is accepted (`PASS_B11`) when each test clip's MOTA is within ±1.0 of the published value AND a
# MAGIC valid transformers-5.3.0 diagnostic (all 4 clips × 600 frames, same GPU class, scorer valid) reproduces the published
# MAGIC S-Edi counts within ±`tol_boxes` (ΔIDSW = 0); otherwise the b1.0 stop rule stands. The teacher's rule is unchanged.
# MAGIC The b1.0 verdict is kept under `overall.b10`, the evaluation (with the cause of the residual: runtime image vs
# MAGIC transformers, from a 5.2.0 control diagnostic) under `b11`. Also new in this version: the old-protocol session is
# MAGIC released in place after each clip (`core.release_session`, J0b; outputs unaffected).

# COMMAND ----------

# ---- environment selection for the %pip cell below (runs BEFORE the restart; the main code re-checks the file) -------
# b1.0 pins transformers 5.2.0 for every run; 5.3.0 is accepted only as an environment diagnostic (checked after the
# restart: the tag must start with "smoke_envdiag"). Keep _TOL68_PIP_FIXED / the transformers position identical to
# PIP_FIXED / pip_requirements() below (the selftest executes this cell and compares the file it writes).
import os as _os
_TOL68_REQ_PATH = _os.environ.get("TOL68_REQ_PATH") or "/local_disk0/tol68_env/requirements.txt"
_TOL68_TF_ALLOWED = ("5.2.0", "5.3.0")
_TOL68_PIP_FIXED = ("numpy<2.0", "Pillow==11.1.0", "huggingface_hub", "accelerate", "timm==1.0.25", "scipy",
                    "motmetrics==1.4.0")
try:
    _tol68_dbu = dbutils  # noqa: F821
except NameError:
    _tol68_dbu = None     # local plain python / selftest import: nothing is written
if _tol68_dbu is not None:
    try:
        _tol68_dbu.widgets.text("transformers_pin", "5.2.0")
    except Exception:  # noqa: BLE001
        pass
    _tol68_tf = (_tol68_dbu.widgets.get("transformers_pin") or "5.2.0").strip()
    if _tol68_tf not in _TOL68_TF_ALLOWED:
        raise ValueError(f"transformers_pin {_tol68_tf!r} not in {_TOL68_TF_ALLOWED}")
    _os.makedirs(_os.path.dirname(_TOL68_REQ_PATH), exist_ok=True)
    with open(_TOL68_REQ_PATH, "w", encoding="utf-8", newline="\n") as _f:
        _f.write("\n".join(_TOL68_PIP_FIXED[:1] + (f"transformers=={_tol68_tf}",) + _TOL68_PIP_FIXED[1:]) + "\n")
    print(f"%pip requirements -> {_TOL68_REQ_PATH}: transformers=={_tol68_tf}")

# COMMAND ----------

# MAGIC %pip install -r /local_disk0/tol68_env/requirements.txt

# COMMAND ----------

try:
    dbutils.library.restartPython()  # noqa: F821
except NameError:
    pass   # local selftest: plain python

# COMMAND ----------

import contextlib
import copy
import gc
import glob
import json
import math
import os
import os.path as osp
import re
import statistics
import sys
import tempfile
import time
import traceback
from collections import defaultdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before this process touches CUDA

import numpy as np

NB_VERSION = "tol68-2026-10-15.2"
IS_NOTEBOOK = True
try:
    dbutils  # noqa: F821,B018
except NameError:
    IS_NOTEBOOK = False
SELFTEST = os.environ.get("TOL68_SELFTEST") == "1"
if not IS_NOTEBOOK and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _code_dir():
    """Directory holding tol_bench_core.py / tol_paths.py: this file's directory locally, the notebook's Workspace
    directory on Databricks (the working directory if the notebook context is unreadable)."""
    if not IS_NOTEBOOK:
        return osp.dirname(osp.abspath(__file__))
    try:
        return "/Workspace" + osp.dirname(
            dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())  # noqa: F821
    except Exception as e:  # noqa: BLE001
        print(f"notebook path unreadable ({e!r}); using the working directory")
        return os.getcwd()


def _import_from(dirpath, name):
    """Import module `name` from EXACTLY dirpath (nothing else on sys.path may shadow it); None if the file is absent."""
    if not dirpath or not osp.isfile(osp.join(dirpath, name + ".py")):
        return None
    while dirpath in sys.path:
        sys.path.remove(dirpath)
    sys.path.insert(0, dirpath)
    sys.modules.pop(name, None)
    mod = __import__(name)
    got = osp.normcase(osp.realpath(osp.dirname(osp.abspath(mod.__file__))))
    if got != osp.normcase(osp.realpath(dirpath)):
        raise RuntimeError(f"{name} resolved to {mod.__file__}, not {dirpath}")
    return mod


CODE_DIR = _code_dir()
core = _import_from(CODE_DIR, "tol_bench_core")
if core is None:
    raise RuntimeError(f"tol_bench_core.py is not in {CODE_DIR}: deploy it (RAW) beside this notebook")
_TP_DIR = CODE_DIR
if not osp.isfile(osp.join(_TP_DIR, "tol_paths.py")) and not IS_NOTEBOOK:
    _TP_DIR = osp.join(osp.dirname(CODE_DIR), "catalog", "build")      # local selftest: the catalog build copy
tol_paths = _import_from(_TP_DIR, "tol_paths")
if tol_paths is None:
    raise RuntimeError(f"tol_paths.py is not in {CODE_DIR}: deploy the registry's tol_paths.py (RAW) beside this notebook")
print(f"Tol_68 {NB_VERSION}; tol_bench_core {core.BENCH_VERSION} sha256 {core.sha256_file(core.__file__)[:16]}; "
      f"tol_paths sha256 {core.sha256_file(tol_paths.__file__)[:16]}")

# =====================================================================================================================
# Pre-registered constants (fixed before any J3 run; a change is a new NB_VERSION + a BENCH_PROTOCOL §15 amendment)
# =====================================================================================================================
HF_SAM3 = "facebook/sam3"
CLIP_RX = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{6}$")
TAG_RX = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")          # one path segment: never "." / ".."
OLD_N_FRAMES = 600                    # evaluate_tracking_quality: for fi in range(600)
OLD_IOU = 0.5
OLD_STUDENT_PRUNE_K = 8               # cell 26: prune_inference_session(session, keep_last_n=8) every 25 frames
OLD_PRUNE_EVERY = 25
OLD_TEACHER_GC_EVERY = 100            # cell 17: gc.collect(); torch.cuda.empty_cache() every 100 frames, no pruning
REPRO_TOL_BOXES = 5                   # per clip, |dFP| and |dFN| (each of 4,800 GT boxes; 5 boxes = 0.21 MOTA points)
# BENCH_PROTOCOL b1.1 #2 (user-approved 2026-10-14; implemented tol68-2026-10-15.1): S-Edi accepted when each test clip's
# MOTA is within +-1.0 of the published value AND the transformers 5.3.0 diagnostic run of (b) reproduces the published
# counts (+- REPRO_TOL_BOXES); the teacher's rule is unchanged
B11_MOTA_TOL = 1.0
B11_DIAG_TF = "5.3.0"
ARM_DIRS = {"T": "T__oldprot", "T-k8": "T-k8__oldprot", "S-Edi": "S-Edi__oldprot"}
ARMS_B = ("T", "T-k8", "S-Edi")
OLD_PEAK_START_MAX_GB = 2.5          # a run that starts above this (weights ~0.9-1.1 GB) has a contaminated peak
CKPT_EPOCH0 = 74                      # ckpt['epoch'] (0-based) -> "Loaded EMA weights (epoch 75)"
CKPT_EMA_VAL_LOSS = 0.25041554232438407
GN_SEEDS = (0, 1, 2, 3, 4)
EXPECT_MOTMETRICS = "1.4.0"

# published per-clip results (catalog copies of the 2026-03 Metrics JSONs; asserted equal to them at run time)
PUBLISHED = {
    "T": {   # SAM3_baseline_summary.json (= re-score of the surviving Teacher_Annotations)
        "2019_12_02_000005": {"fp": 215, "fn": 215, "idsw": 0, "mota": 91.04166666666667, "idf1": 95.52083333333333},
        "2019_12_02_000208": {"fp": 39, "fn": 39, "idsw": 0, "mota": 98.375, "idf1": 99.1875},
        "2019_12_10_000060": {"fp": 1, "fn": 1, "idsw": 0, "mota": 99.95833333333334, "idf1": 99.97916666666666},
        "2019_12_10_000078": {"fp": 324, "fn": 324, "idsw": 0, "mota": 86.5, "idf1": 93.25},
    },
    "S-Edi": {   # Student_TinyViT_Test_Results.json (the per-clip Student_TinyViT_SAM3_* files hold _Improved values)
        "2019_12_02_000005": {"fp": 318, "fn": 318, "idsw": 0, "mota": 86.75, "idf1": 93.375},
        "2019_12_02_000208": {"fp": 35, "fn": 35, "idsw": 0, "mota": 98.54166666666667, "idf1": 99.27083333333333},
        "2019_12_10_000060": {"fp": 2, "fn": 2, "idsw": 0, "mota": 99.91666666666667, "idf1": 99.95833333333334},
        "2019_12_10_000078": {"fp": 385, "fn": 385, "idsw": 0, "mota": 83.95833333333333, "idf1": 91.97916666666667},
    },
}
PUBLISHED_MEAN = {"T": {"mota": 93.96875, "idf1": 96.984375},
                  "S-Edi": {"mota": 92.29166666666667, "idf1": 96.14583333333333}}
PUBLISHED_FILES = {"T": "SAM3_baseline_summary.json", "S-Edi": "Student_TinyViT_Test_Results.json",
                   "improved": "Student_TinyViT_Test_Results_Improved.json", "final": "Final_TinyViT_Comparison.json"}
SURVIVING_DIRS = {"T": "Teacher_Annotations", "improved": "Student_Backbone_Annotations"}

# stage (c): published observations, for the report only
PUB_GN = {"cell21_train_frame0_ema": {"max": 63.35067749, "mean": 0.03444041},
          "cell13_randn_untrained": {"max": 2.26131129}}
# stage (d) bridges: cell 21 output (first 20 frames of training clip 0, fp32 EMA student) and the training history
# entry of epoch 75 (val clip 2019_11_28_000113, EMA weights, fp16 autocast, batches of 4)
PAPER_CELL21 = {"cos_global": 0.8080, "scale_ratio": 0.9489, "mse": 0.7631}
BRIDGE_TOL_CELL21 = 0.003
HIST_EP75 = {"ema_cosine_loss": 0.24925004174311957, "ema_val_loss": 0.25041554232438407}
BRIDGE_TOL_VAL = 0.01
FID_KEYS = ("cos_patch", "cos_patch_p05", "cos_patch_p50", "cos_global", "scale_ratio", "mse", "mse_dir",
            "ch_scale", "loss_total", "student_std", "teacher_std")
COS_LOSS_EPS = 1e-12                  # F.cosine_embedding_loss EPSILON
COS_SIM_EPS = 1e-8                    # F.cosine_similarity eps
NORM_EPS = 1e-12                      # F.normalize eps
# environment diagnostics (tol68-2026-10-14.3): the %pip pins (the first code cell writes the same list) and the rules
ENV_TF_ALLOWED = ("5.2.0", "5.3.0")   # 5.2.0 = b1.0 pin (core.TRANSFORMERS_PIN); 5.3.0 = published student's kernel
ENV_DIAG_PREFIX = "smoke_envdiag"     # every environment diagnostic tag starts with this (no verdict, ever)
PROTOCOL_DBR_PREFIX = "17.3"          # BENCH_PROTOCOL §3 runtime; a non-diagnostic run elsewhere is refused
PIP_FIXED = ("numpy<2.0", "Pillow==11.1.0", "huggingface_hub", "accelerate", "timm==1.0.25", "scipy",
             "motmetrics==1.4.0")
REQ_PATH = "/local_disk0/tol68_env/requirements.txt"
REGISTRY_KEYS = ("datasets.edinburgh_pigs.annotated", "datasets.edinburgh_pigs.decoded_frames",
                 "datasets.edinburgh_pigs.initial_prompts", "models.student_edinburgh",
                 "paper1.evidence.edinburgh_metrics", "paper1.evidence.edinburgh_mask_coordinates",
                 "paper1.teacher_targets.edinburgh", "paper1.root")


# =====================================================================================================================
# pure helpers (stdlib + numpy; exercised by the local selftest)
# =====================================================================================================================
def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def parse_csv(s, allowed=None, name="value"):
    out = [x.strip() for x in str(s).split(",") if x.strip()]
    if allowed is not None:
        bad = [x for x in out if x not in allowed]
        if bad:
            raise ValueError(f"{name}: {bad} not in {sorted(allowed)}")
    if len(set(out)) != len(out):
        raise ValueError(f"{name}: duplicates in {out}")
    return out


def parse_ints(s, name="value"):
    try:
        return [int(x) for x in parse_csv(s, name=name)]
    except ValueError as e:
        raise ValueError(f"{name}: {s!r} is not a comma list of integers") from e


def parse_bool(s):
    v = str(s).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off", ""):
        return False
    raise ValueError(f"not a boolean: {s!r}")


def pip_requirements(tf_pin):
    """The %pip requirement lines for transformers==tf_pin (the first code cell writes exactly these)."""
    if tf_pin not in ENV_TF_ALLOWED:
        raise ValueError(f"transformers_pin {tf_pin!r} not in {ENV_TF_ALLOWED}")
    return list(PIP_FIXED[:1]) + [f"transformers=={tf_pin}"] + list(PIP_FIXED[1:])


def env_spec(tf_pin, expect_torch, tag, runtime=None, protocol_pin=None):
    """Validate the environment widgets. A run is an environment diagnostic iff its pin differs from the protocol pin
    or its tag starts with ENV_DIAG_PREFIX; a diagnostic needs that tag prefix (so it is always a smoke tag and never
    carries a verdict), and a non-diagnostic run must be on the protocol runtime (DBR 17.3) when the runtime is known."""
    protocol_pin = protocol_pin or core.TRANSFORMERS_PIN
    tf = (tf_pin or protocol_pin).strip()
    if tf not in ENV_TF_ALLOWED:
        raise ValueError(f"transformers_pin {tf!r} not in {ENV_TF_ALLOWED}")
    et = (expect_torch or "").strip()
    if et and not re.match(r"^\d+\.\d+\.\d+$", et):
        raise ValueError(f"expect_torch {et!r} is not X.Y.Z")
    diag = tf != protocol_pin or str(tag).startswith(ENV_DIAG_PREFIX)
    if diag and not str(tag).startswith(ENV_DIAG_PREFIX):
        raise ValueError(f"transformers {tf} != the b1.0 pin {protocol_pin}: an environment diagnostic needs a tag "
                         f"starting with {ENV_DIAG_PREFIX!r}")
    if not diag and runtime and not str(runtime).startswith(PROTOCOL_DBR_PREFIX):
        raise ValueError(f"Databricks runtime {runtime} is not {PROTOCOL_DBR_PREFIX} (BENCH_PROTOCOL §3): use a tag "
                         f"starting with {ENV_DIAG_PREFIX!r} for an environment diagnostic")
    return {"transformers_pin": tf, "expect_torch": et or None, "env_diag": bool(diag), "databricks_runtime": runtime,
            "requirements": pip_requirements(tf)}


def gt_rel_parts(clip):
    """evaluate_tracking_quality: date_part, seq_part = video_id[:10], video_id[-6:] -> annotated/<date>/<seq>/output.json"""
    if not CLIP_RX.match(clip):
        raise ValueError(f"bad clip id {clip!r}")
    return (clip[:10], clip[-6:], "output.json")


def parse_prompt_txt_ordered(text):
    """VERBATIM line parse of load_bounding_box_prompts (Fixed cell 10), keeping the LINE ORDER and the raw x,y,w,h."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        name, coords = line.split(':', 1)
        x, y, w, h = map(float, coords.strip().split(','))
        out.append((name.strip(), [x, y, w, h]))
    return out


def old_prompt_boxes(ordered):
    """= load_bounding_box_prompts' dict: name -> [int(x), int(y), int(x+w), int(y+h)] (a repeated name overwrites)."""
    prompts = {}
    for name, (x, y, w, h) in ordered:
        prompts[name] = [int(x), int(y), int(x + w), int(y + h)]
    return prompts


def _iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    un = a[2] * a[3] + b[2] * b[3] - inter
    return inter / un if un > 0 else 0.0


def seed_check_clip(clip, prompt_text, gt):
    """(a) one clip: are the BoundingBoxes_<clip>.txt seeds exactly GT frame 0, in GT object order?"""
    ordered = parse_prompt_txt_ordered(prompt_text)
    names = [n for n, _ in ordered]
    issues = []
    if len(set(names)) != len(names):
        issues.append("duplicate prompt names")
    gt_first, gt_order = {}, []
    for o in gt["objects"]:
        oid = int(o["id"])
        gt_order.append(oid)
        gt_first[oid] = min(o["frames"], key=lambda f: int(f["frameNumber"]))
    rows, prompt_ids, name_of = [], [], {}
    for name, xywh in ordered:
        m = re.match(r"^id (\d+)$", name, re.IGNORECASE)          # 'id k' (8 clips) or 'ID k' (2 training clips)
        if not m:
            issues.append(f"prompt name {name!r} is not 'id <k>'")
            continue
        k = int(m.group(1))
        prompt_ids.append(k)
        name_of[k] = name
        f0 = gt_first.get(k)
        if f0 is None:
            issues.append(f"{name}: no GT object {k}")
            continue
        b = f0["bbox"]
        g = [float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"])]
        rows.append({"name": name, "prompt_xywh": xywh, "gt_xywh": g, "gt_first_frame": int(f0["frameNumber"]),
                     "isGroundTruth": f0.get("isGroundTruth"), "visible": f0.get("visible"),
                     "exact": [float(v) for v in xywh] == g, "iou": round(_iou_xywh(xywh, g), 6),
                     "bbox_key_order": list(b.keys())})
    missing = sorted(set(gt_first) - set(prompt_ids))
    old_int = old_prompt_boxes(ordered)
    try:
        pa = {p.obj_id: [int(v) for v in p.box] for p in core.protocol_a_edinburgh(core.read_edinburgh_keyframes(gt))}
        prot_a_equal = all(k in name_of and old_int.get(name_of[k]) == pa[k] for k in pa) and len(pa) == len(old_int)
        prot_a_err = None
    except Exception as e:  # noqa: BLE001
        prot_a_equal, prot_a_err = False, f"{type(e).__name__}: {e}"
    srt = sorted(old_int)
    rec = {
        "clip": clip, "n_prompts": len(ordered), "n_gt_objects": len(gt_order),
        "n_exact": int(sum(r["exact"] for r in rows)), "min_iou": min([r["iou"] for r in rows], default=None),
        "order_equals_gt_object_order": prompt_ids == gt_order, "prompt_order": prompt_ids, "gt_order": gt_order,
        "all_first_keyframes_at_0": all(r["gt_first_frame"] == 0 for r in rows) and bool(rows),
        "all_isGroundTruth": all(r["isGroundTruth"] is True for r in rows) and bool(rows),
        "bbox_key_order_xywh": all(r["bbox_key_order"] == ["x", "y", "width", "height"] for r in rows),
        "old_int_boxes_equal_protocol_a": prot_a_equal, "protocol_a_error": prot_a_err,
        "sorted_name_index_equals_id": [int(n.split()[-1]) for n in srt] == list(range(len(srt))),
        "name_prefixes": sorted({n.rsplit(" ", 1)[0] for n in names if " " in n}),
        "gt_stepSize": gt.get("stepSize"), "missing_gt_ids": missing, "issues": issues, "rows": rows,
    }
    rec["ok"] = bool(not issues and not missing and rec["n_exact"] == rec["n_prompts"] == rec["n_gt_objects"]
                     and rec["order_equals_gt_object_order"] and rec["all_first_keyframes_at_0"]
                     and rec["all_isGroundTruth"] and prot_a_equal)
    return rec


def seed_table_text(recs):
    lines = ["clip | n_prompts | n_gt_obj | exact_frame0 | min_IoU | order_eq | first_kf_0 | isGT | protA_eq | "
             "idx=id | strays | ok"]
    for r in recs:
        g = r.get
        lines.append(f"{g('clip')} | {g('n_prompts', 0)} | {g('n_gt_objects', 0)} | {g('n_exact', 0)}/"
                     f"{g('n_prompts', 0)} | {g('min_iou')} | {g('order_equals_gt_object_order')} | "
                     f"{g('all_first_keyframes_at_0')} | {g('all_isGroundTruth')} | "
                     f"{g('old_int_boxes_equal_protocol_a')} | {g('sorted_name_index_equals_id')} | "
                     f"{g('strays', [])} | {g('ok')}" + (f" | issues {g('issues')}" if g("issues") else ""))
    tot = sum(r.get("n_exact", 0) for r in recs)
    n = sum(r.get("n_prompts", 0) for r in recs)
    lines.append(f"TOTAL exact {tot}/{n}; clips ok {sum(r['ok'] for r in recs)}/{len(recs)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- old scorer
def old_gt_boxes(gt_data, n_frames=OLD_N_FRAMES):
    """VERBATIM evaluate_tracking_quality GT parse (Fixed cell 23): each keyframe box HELD until the next keyframe."""
    gt_boxes = {}
    for obj in gt_data["objects"]:
        pid = int(obj["id"])
        for i, fr in enumerate(obj["frames"]):
            end = obj["frames"][i + 1]["frameNumber"] if i + 1 < len(obj["frames"]) else n_frames
            x, y, w, h = fr["bbox"].values()
            for fi in range(fr["frameNumber"], end):
                gt_boxes[(fi, pid)] = (x, y, x + w, y + h)
    return gt_boxes


def interp_gt_boxes(gt_data, n_frames=OLD_N_FRAMES):
    """Bench GT (BENCH_PROTOCOL §2.1): keyframes LINEARLY INTERPOLATED (core.edinburgh_gt_rows), same dict layout."""
    rows = core.edinburgh_gt_rows(core.read_edinburgh_keyframes(gt_data), n_frames=n_frames, interpolate=True)
    return {(int(r[0]), int(r[1])): (float(r[2]), float(r[3]), float(r[2] + r[4]), float(r[3] + r[5])) for r in rows}


def load_old_annotations(track_dir):
    """VERBATIM evaluate_tracking_quality tracker parse ('annotations_id <k>.json', frame 'NNNNNNN.jpg' -> fi = N-1);
    file order sorted (the original iterated os.listdir; a re-score of the surviving boxes in sorted order reproduces
    the published numbers exactly)."""
    tr_boxes = {}
    for fname in sorted(os.listdir(track_dir)):
        if not fname.startswith("annotations_id") or not fname.endswith(".json"):
            continue
        pid = int(fname.split()[-1].split(".")[0])
        with open(osp.join(track_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
        for frame_name, ann in data.items():
            fi = int(frame_name.split(".")[0]) - 1
            x, y, w, h = ann["bounding_box"]
            tr_boxes[(fi, pid)] = (x, y, x + w, y + h)
    return tr_boxes


def ids_by_frame(boxes):
    """{fi: [pid, ...]} in dict insertion order = exactly the per-frame lists of the verbatim
    `[p for (f, p) in boxes if f == fi]` (one pass instead of one scan per frame)."""
    out = defaultdict(list)
    for (f, p) in boxes:
        out[f].append(p)
    return out


def old_score(video_id, gt_boxes, tr_boxes, n_frames=OLD_N_FRAMES, iou_threshold=OLD_IOU):
    """VERBATIM evaluate_tracking_quality matching + py-motmetrics motchallenge_metrics. Percent for mota / idf1."""
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    gt_f, tr_f = ids_by_frame(gt_boxes), ids_by_frame(tr_boxes)
    for fi in range(n_frames):
        gt_ids = list(gt_f.get(fi, ()))
        det_ids = list(tr_f.get(fi, ()))
        cost = np.zeros((len(gt_ids), len(det_ids)))
        for i, gid in enumerate(gt_ids):
            g = gt_boxes[(fi, gid)]
            for j, did in enumerate(det_ids):
                t = tr_boxes[(fi, did)]
                xa, ya = max(g[0], t[0]), max(g[1], t[1])
                xb, yb = min(g[2], t[2]), min(g[3], t[3])
                inter = max(0, xb - xa) * max(0, yb - ya)
                union = (g[2] - g[0]) * (g[3] - g[1]) + (t[2] - t[0]) * (t[3] - t[1]) - inter
                cost[i, j] = 1 - (inter / union if union > 0 else 0)
        cost[cost > 1 - iou_threshold] = np.nan
        acc.update(gt_ids, det_ids, cost)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=video_id)
    raw = {n: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for n, v in summary.loc[video_id].items()}
    motp = raw.get("motp")
    return {"mota": 100.0 * raw["mota"], "idf1": 100.0 * raw["idf1"], "fp": int(raw["num_false_positives"]),
            "fn": int(raw["num_misses"]), "idsw": int(raw["num_switches"]), "motp": motp,
            "mean_tp_iou": (1.0 - motp) if motp is not None and not math.isnan(motp) else None,
            "n_frames": int(n_frames), "n_gt_boxes": int(sum(1 for (f, _p) in gt_boxes if f < n_frames)),
            "raw": raw}


def bbox_from_binary(mb):
    """VERBATIM cell 26 / cell 17 box: argwhere -> [x0, y0, x1-x0, y1-y0] (inclusive extremes, no +1), area = sum;
    an empty mask gives [0, 0, 0, 0] and area 0 (the old protocol writes it as a box)."""
    nz = np.argwhere(mb)
    if len(nz) == 0:
        return [0, 0, 0, 0], 0
    y0, x0 = nz.min(0).tolist()
    y1, x1 = nz.max(0).tolist()
    return [x0, y0, x1 - x0, y1 - y0], int(np.asarray(mb).sum())


def prune_inference_session(session, keep_last_n=5):
    """VERBATIM Fixed cell 26 (the published student's pruning; called with keep_last_n=8)."""
    for obj_idx in range(session.get_obj_num()):
        non_cond = session.output_dict_per_obj[obj_idx].get("non_cond_frame_outputs", {})
        if len(non_cond) <= keep_last_n:
            continue
        sorted_frames = sorted(non_cond.keys())
        frames_to_remove = sorted_frames[:-keep_last_n]
        for frame_key in frames_to_remove:
            del non_cond[frame_key]


def write_old_annotations(out_dir, annots):
    """The old per-object files: annotations_id <oid>.json = {frame file: {bounding_box, object_name, mask_area}}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for oid, ann in annots.items():
        p = osp.join(out_dir, f"annotations_id {oid}.json")
        core.write_text_atomic(p, json.dumps(ann, indent=2))
        paths.append(p)
    return paths


def old_annots_to_mot(annots, xy_offset=0):
    """Old-protocol boxes as bench MOT txt (core.MotWriter: frame fi+1, id oid+1, w = x1-x0+1). Rows with mask_area 0
    (the old [0,0,0,0] boxes) are dropped. Returns the MOT lines."""
    w = core.MotWriter(xy_offset=xy_offset)
    for oid, ann in annots.items():
        for fname, a in ann.items():
            if int(a.get("mask_area", 1)) <= 0:
                continue
            fi = int(fname.split(".")[0]) - 1
            x, y, bw, bh = (int(v) for v in a["bounding_box"])
            w.add(fi, [core.Row(int(oid), int(oid), x, y, x + bw, y + bh, int(a.get("mask_area", 0)), 1.0)])
    return w.lines()


def _box_empty(b):
    return (b[2] - b[0]) <= 0 or (b[3] - b[1]) <= 0


def _iou_xyxy_or_empty(a, b):
    ea, eb = _box_empty(a), _box_empty(b)
    if ea and eb:
        return 1.0
    if ea or eb:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    un = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / un if un > 0 else 0.0


def compare_box_tables(a, b, max_worst=20):
    """Direct per-frame check of two {(fi, pid): (x1,y1,x2,y2)} tables (regenerated vs surviving)."""
    ka, kb = set(a), set(b)
    common = sorted(ka & kb)
    ious, ident, worst, first_low = [], 0, [], {}
    for k in common:
        x, y = tuple(a[k]), tuple(b[k])
        if x == y:
            ident += 1
            iou = 1.0
        else:
            iou = _iou_xyxy_or_empty(x, y)
        ious.append(iou)
        if iou < 0.9 and str(k[1]) not in first_low:
            first_low[str(k[1])] = int(k[0])
        if iou < 1.0:
            worst.append((iou, int(k[0]), int(k[1]), list(x), list(y)))
    worst.sort(key=lambda t: (t[0], t[1], t[2]))
    v = np.asarray(ious, dtype=np.float64)
    return {"n_a": len(ka), "n_b": len(kb), "n_common": len(common), "only_a": len(ka - kb), "only_b": len(kb - ka),
            "n_identical": ident, "frac_identical": (ident / len(common)) if common else None,
            "iou_mean": float(v.mean()) if v.size else None, "iou_min": float(v.min()) if v.size else None,
            "iou_p01": float(np.percentile(v, 1)) if v.size else None,
            "n_iou_lt_0.5": int((v < 0.5).sum()), "n_iou_lt_0.9": int((v < 0.9).sum()),
            "first_frame_iou_lt_0.9_by_id": first_low,
            "worst": [{"iou": round(i, 4), "fi": f, "id": p, "a": xa, "b": xb} for i, f, p, xa, xb in worst[:max_worst]]}


def published_per_clip(obj, clips):
    """Per-clip tracking_quality_metrics of a 2026-03 results JSON ({clip: {status, summary: {...}}})."""
    out = {}
    for c in clips:
        v = obj[c]
        if v.get("status") != "success":
            raise ValueError(f"{c}: status {v.get('status')}")
        t = v["summary"]["tracking_quality_metrics"]
        out[c] = {"fp": int(t["num_false_positives"]), "fn": int(t["num_misses"]), "idsw": int(t["num_switches"]),
                  "mota": float(t["mota"]), "idf1": float(t["idf1"])}
    return out


def check_published(arm, per_clip, final=None):
    """The evidence JSONs must hold exactly the pre-registered PUBLISHED numbers (guards a wrong / changed file)."""
    exp = PUBLISHED[arm]
    if sorted(per_clip) != sorted(exp):
        raise ValueError(f"{arm}: published clips {sorted(per_clip)} != {sorted(exp)}")
    for c, e in exp.items():
        g = per_clip[c]
        for k in ("fp", "fn", "idsw"):
            if g[k] != e[k]:
                raise ValueError(f"{arm} {c} {k}: evidence {g[k]} != pre-registered {e[k]}")
        for k in ("mota", "idf1"):
            if abs(g[k] - e[k]) > 1e-9:
                raise ValueError(f"{arm} {c} {k}: evidence {g[k]} != pre-registered {e[k]}")
    for k in ("mota", "idf1"):
        m = float(np.mean([per_clip[c][k] for c in exp]))
        if abs(m - PUBLISHED_MEAN[arm][k]) > 1e-9:
            raise ValueError(f"{arm} mean {k} {m} != {PUBLISHED_MEAN[arm][k]}")
    if final is not None:
        side = final["teacher" if arm == "T" else "student"]
        if abs(side["avg_mota"] - PUBLISHED_MEAN[arm]["mota"]) > 1e-9 or \
                abs(side["avg_idf1"] - PUBLISHED_MEAN[arm]["idf1"]) > 1e-9:
            raise ValueError(f"{arm}: Final_TinyViT_Comparison averages differ from the per-clip means")
    return True


def score_matches(new, pub, exact=True, tol=0):
    if exact:
        return (new["fp"] == pub["fp"] and new["fn"] == pub["fn"] and new["idsw"] == pub["idsw"]
                and abs(new["mota"] - pub["mota"]) < 1e-9 and abs(new["idf1"] - pub["idf1"]) < 1e-9)
    return abs(new["fp"] - pub["fp"]) <= tol and abs(new["fn"] - pub["fn"]) <= tol and new["idsw"] == pub["idsw"]


def repro_rows(new_by_clip, pub_by_clip, tol):
    rows = []
    for clip in sorted(pub_by_clip):
        p, n = pub_by_clip[clip], new_by_clip.get(clip)
        if n is None:
            rows.append({"clip": clip, "status": "missing", "within_tol": False})
            continue
        r = {"clip": clip, "status": "scored",
             "fp_new": n["fp"], "fp_pub": p["fp"], "d_fp": n["fp"] - p["fp"],
             "fn_new": n["fn"], "fn_pub": p["fn"], "d_fn": n["fn"] - p["fn"],
             "idsw_new": n["idsw"], "idsw_pub": p["idsw"],
             "mota_new": n["mota"], "mota_pub": p["mota"], "d_mota": n["mota"] - p["mota"],
             "idf1_new": n["idf1"], "idf1_pub": p["idf1"], "d_idf1": n["idf1"] - p["idf1"]}
        r["within_tol"] = bool(abs(r["d_fp"]) <= tol and abs(r["d_fn"]) <= tol and n["idsw"] == p["idsw"])
        rows.append(r)
    return rows


def arm_verdict(rows, tol):
    scored = [r for r in rows if r["status"] == "scored"]
    ok = bool(rows) and len(scored) == len(rows) and all(r["within_tol"] for r in rows)
    return {"verdict": "PASS" if ok else "FAIL", "tolerance_boxes": tol, "n_clips": len(rows),
            "n_within_tol": int(sum(r["within_tol"] for r in rows)),
            "max_abs_d_fp": max([abs(r["d_fp"]) for r in scored], default=None),
            "max_abs_d_fn": max([abs(r["d_fn"]) for r in scored], default=None),
            "mean_mota_new": float(np.mean([r["mota_new"] for r in scored])) if scored else None,
            "mean_mota_pub": float(np.mean([r["mota_pub"] for r in scored])) if scored else None,
            "mean_idf1_new": float(np.mean([r["idf1_new"] for r in scored])) if scored else None,
            "mean_idf1_pub": float(np.mean([r["idf1_pub"] for r in scored])) if scored else None,
            "rows": rows}


def overall_verdict(v_t, v_s, scorer_valid=True):
    """BENCH_PROTOCOL §13 J3(c) stop rule."""
    if not scorer_valid:
        return {"status": "SCORER_INVALID", "stop": True,
                "message": "the verbatim old scorer does not reproduce the published teacher from the surviving "
                           "Teacher_Annotations in this environment; no reproduction verdict is possible"}
    if v_t is None or v_s is None:
        return {"status": "INCOMPLETE", "stop": True, "message": "T or S-Edi was not run/scored in this tag"}
    t_ok, s_ok = v_t["verdict"] == "PASS", v_s["verdict"] == "PASS"
    if t_ok and s_ok:
        return {"status": "PASS", "stop": False,
                "message": "the old protocol reproduces both published runs within tolerance: the published student "
                           "checkpoint (EMA epoch 75) is confirmed as the one behind 92.29 / 96.15"}
    if t_ok:
        return {"status": "STOP_STUDENT", "stop": True,
                "message": "the teacher reproduces but the student does not: S-Edi's Edinburgh numbers are not "
                           "interpreted until the checkpoint question is resolved with the user (BENCH_PROTOCOL §13). "
                           "Known confound to rule out first: the published student run's kernel had transformers "
                           "5.3.0 (the teacher's 5.2.0); this audit pins 5.2.0 for both, so a transformers-minor "
                           "difference can also produce a student-only mismatch"}
    return {"status": "STOP_TEACHER", "stop": True,
            "message": "the teacher does not reproduce: the replication environment differs (GPU class, torch / "
                       "transformers, HF revision, attention); the student comparison is uninformative until resolved"}


def score_stage_b(clips, gt_by_clip, tr_by_arm, surviving, published, improved_pub, tol, n_frames=OLD_N_FRAMES,
                  env_diag=None):
    """(b) scoring, pure: tr_by_arm {arm: {clip: tr_boxes}} (regenerated), surviving {"T"/"improved": {clip: boxes}},
    published {"T"/"S-Edi": per-clip}, improved_pub per-clip of the _Improved JSON. Returns (scores, verdict, direct).
    env_diag: None for a protocol run, else a short description of the diagnostic environment -> overall ENV_DIAG
    (the per-arm rows are still computed; no reproduction verdict)."""
    full = n_frames == OLD_N_FRAMES
    complete = full and sorted(clips) == sorted(core.EDI_TEST)
    hold = {c: old_gt_boxes(gt_by_clip[c], OLD_N_FRAMES) for c in clips}
    interp = {c: interp_gt_boxes(gt_by_clip[c], OLD_N_FRAMES) for c in clips}
    scores = {}
    for arm, by_clip in tr_by_arm.items():
        for c, tr in by_clip.items():
            scores.setdefault(arm, {})[c] = {"hold_gt": old_score(c, hold[c], tr, n_frames),
                                             "interp_gt": old_score(c, interp[c], tr, n_frames)}
    sv = {"surviving_teacher": {}, "surviving_student_dir_is_improved": {}}
    for c in clips:
        if c in surviving.get("T", {}):
            s = old_score(c, hold[c], surviving["T"][c], OLD_N_FRAMES)
            sv["surviving_teacher"][c] = {"score": _slim(s), "equals_published_T": score_matches(s, published["T"][c])}
        if c in surviving.get("improved", {}) and improved_pub is not None:
            s = old_score(c, hold[c], surviving["improved"][c], OLD_N_FRAMES)
            sv["surviving_student_dir_is_improved"][c] = {
                "score": _slim(s), "equals_improved_json": score_matches(s, improved_pub[c])}
    scorer_valid = bool(sv["surviving_teacher"]) and all(v["equals_published_T"]
                                                         for v in sv["surviving_teacher"].values())
    verdict = {"tolerance_boxes": tol, "full_clips": full, "complete_test_set": complete, "clips": list(clips),
               "rule": "per clip |dFP| <= tol and |dFN| <= tol vs the published per-clip counts and dIDSW = 0 "
                       "(hold GT, old scorer); an arm passes when all 4 test clips pass",
               "scorer_validation": sv, "scorer_valid": scorer_valid, "arms": {}}
    for arm in ("T", "S-Edi"):
        if arm in scores and full:
            new = {c: scores[arm][c]["hold_gt"] for c in scores[arm]}
            verdict["arms"][arm] = arm_verdict(repro_rows(new, {c: published[arm][c] for c in clips}, tol), tol)
    if "T-k8" in scores:
        verdict["arms"]["T-k8"] = {"verdict": "NOT_GATED", "vs_T_oldprot": {
            c: {"d_fp": scores["T-k8"][c]["hold_gt"]["fp"] - scores["T"][c]["hold_gt"]["fp"],
                "d_mota": scores["T-k8"][c]["hold_gt"]["mota"] - scores["T"][c]["hold_gt"]["mota"]}
            for c in scores["T-k8"] if c in scores.get("T", {})}}
    if env_diag:
        verdict["env_diag"] = env_diag
        verdict["overall"] = {"status": "ENV_DIAG", "stop": True,
                              "message": f"environment diagnostic ({env_diag}; max_frames={n_frames}, "
                                         f"clips={list(clips)}): per-clip rows only, no reproduction verdict"}
    elif complete:
        verdict["overall"] = overall_verdict(verdict["arms"].get("T"), verdict["arms"].get("S-Edi"), scorer_valid)
    else:
        verdict["overall"] = {"status": "SMOKE", "stop": True,
                              "message": f"reduced run (max_frames={n_frames}, clips={list(clips)}): no verdict"}
    direct = {}
    if "T" in tr_by_arm:
        for c, tr in tr_by_arm["T"].items():
            if c in surviving.get("T", {}):
                direct[c] = compare_box_tables(tr, surviving["T"][c])
    if "S-Edi" in tr_by_arm and "T" in tr_by_arm:
        direct["_student_vs_teacher_regenerated"] = {
            c: {k: v for k, v in compare_box_tables(tr_by_arm["S-Edi"][c], tr_by_arm["T"][c]).items() if k != "worst"}
            for c in tr_by_arm["S-Edi"] if c in tr_by_arm["T"]}
    slim_scores = {a: {c: {g: _slim(s) for g, s in d.items()} for c, d in by.items()} for a, by in scores.items()}
    return slim_scores, verdict, direct


def _slim(s):
    return {k: v for k, v in s.items() if k != "raw"}


# ---------------------------------------------------------------------------------------------------- b1.1 #2
def parse_b11_tags(s, tag, env_diag, smoke):
    """Widget b11_diag_tags (tol68-2026-10-15.1): the published environment-diagnostic tags whose stage-(b) verdicts the
    BENCH_PROTOCOL b1.1 #2 rule reads. Only a complete protocol run may name them, and each must be another tag that
    starts with ENV_DIAG_PREFIX."""
    tags = parse_csv(s, name="b11_diag_tags")
    if not tags:
        return []
    if env_diag or smoke:
        raise ValueError("b11_diag_tags: only a complete protocol run (not a diagnostic, not a smoke) applies b1.1 #2")
    bad = [t for t in tags if not TAG_RX.match(t) or not t.startswith(ENV_DIAG_PREFIX) or t == tag]
    if bad:
        raise ValueError(f"b11_diag_tags {bad}: each must be another tag starting with {ENV_DIAG_PREFIX!r}")
    return tags


def b11_diag_record(tag, audit_config, verdict):
    """One published environment-diagnostic tag (its AUDIT_CONFIG.json + b_reproduce_old/verdict.json), summarised for
    the b1.1 #2 rule (pure). An arm 'reproduces' when all 4 test clips are scored and within the stage's tolerance."""
    ac, v = audit_config or {}, verdict or {}
    env, ver = ac.get("env") or {}, ac.get("versions_pinned") or {}
    arms = {}
    for arm in ("T", "S-Edi"):
        a = (v.get("arms") or {}).get(arm)
        if not a or "rows" not in a:
            continue
        rows = a.get("rows") or []
        scored = sorted(r["clip"] for r in rows if r.get("status") == "scored")
        arms[arm] = {"n_clips": len(rows), "n_within_tol": int(sum(bool(r.get("within_tol")) for r in rows)),
                     "max_abs_d_fp": a.get("max_abs_d_fp"), "max_abs_d_fn": a.get("max_abs_d_fn"),
                     "reproduces": bool(rows) and scored == sorted(core.EDI_TEST)
                     and all(bool(r.get("within_tol")) for r in rows),
                     "rows": [{k: r.get(k) for k in ("clip", "fp_new", "fp_pub", "fn_new", "fn_pub", "idsw_new",
                                                     "idsw_pub", "mota_new", "mota_pub", "within_tol")} for r in rows]}
    return {"tag": tag, "status": (v.get("overall") or {}).get("status"), "env_diag": env.get("env_diag") is True,
            "transformers_pin": env.get("transformers_pin"), "databricks_runtime": env.get("databricks_runtime"),
            "transformers": ver.get("transformers"), "torch": ver.get("torch"), "cuda": ver.get("cuda"),
            "gpu_class": (ac.get("gpu") or {}).get("class"), "notebook": ac.get("notebook"),
            "full_test_set": v.get("full_clips") is True and v.get("complete_test_set") is True,
            "scorer_valid": v.get("scorer_valid") is True, "tolerance_boxes": v.get("tolerance_boxes"), "arms": arms}


def _b11_env_text(d):
    return (f"tag {d['tag']}: transformers {d.get('transformers') or d.get('transformers_pin')}, torch {d.get('torch')}, "
            f"DBR {d.get('databricks_runtime')}, {d.get('gpu_class')}")


def apply_b11(verdict, diag_records, gpu_class=None, mota_tol=B11_MOTA_TOL, diag_tf=B11_DIAG_TF, protocol_pin=None):
    """BENCH_PROTOCOL b1.1 #2 (user-approved 2026-10-14) on a COMPLETE protocol-run stage-(b) verdict (pure; returns a
    new dict; the b1.0 overall verdict is kept under overall['b10'] and the evaluation under verdict['b11']).

    Only STOP_STUDENT can change, to PASS_B11, and only when
      (1) S-Edi's MOTA is within +-mota_tol of the published per-clip value on each of the 4 test clips (this run), and
      (2) a valid diagnostic tag at transformers `diag_tf` (the published student's kernel; "the 5.3.0 diagnostic run
          of Tol_68 (b)") reproduces the published S-Edi counts on all 4 test clips within the same +-tol boxes and
          dIDSW = 0 (600 frames, scorer valid, same GPU class as this run).
    Otherwise the b1.0 stop rule stands. The teacher's rule is unchanged (STOP_TEACHER / PASS are never touched).
    The cause of the residual is classified from the diagnostics (informational, not a condition): 'runtime_image' when
    a valid diagnostic at THIS run's transformers pin also reproduces S-Edi (the same code and the same transformers on
    another runtime image reproduce, so the image's library stack, not transformers, explains the residual);
    'transformers' when only the diag_tf diagnostic reproduces and a valid one at this run's pin on the same runtime does
    not; else 'not_isolated'."""
    protocol_pin = protocol_pin or core.TRANSFORMERS_PIN
    v = copy.deepcopy(verdict)
    ov = v["overall"]
    tol = v.get("tolerance_boxes")
    recs = []
    for d in diag_records:
        why = []
        if not d.get("env_diag") or d.get("status") != "ENV_DIAG":
            why.append(f"not an environment diagnostic (status {d.get('status')})")
        if not d.get("full_test_set"):
            why.append("not the full 4-clip x 600-frame test set")
        if not d.get("scorer_valid"):
            why.append("old scorer not validated in that run")
        if d.get("tolerance_boxes") != tol:
            why.append(f"tolerance {d.get('tolerance_boxes')} != {tol}")
        if gpu_class is not None and d.get("gpu_class") != gpu_class:
            why.append(f"GPU class {d.get('gpu_class')} != {gpu_class}")
        if "S-Edi" not in (d.get("arms") or {}):
            why.append("no S-Edi row")
        recs.append(dict(d, valid_for_b11=not why, invalid_reasons=why))
    rec = {"rule": f"BENCH_PROTOCOL b1.1 #2: S-Edi accepted when each test clip's MOTA is within +-{mota_tol} of the "
                   f"published value AND the transformers {diag_tf} diagnostic run of Tol_68 (b) reproduces the "
                   f"published S-Edi counts within +-{tol} boxes (dIDSW = 0) on all 4 clips; otherwise the b1.0 stop "
                   f"rule applies. The teacher's rule is unchanged.",
           "mota_tol": mota_tol, "diag_transformers_pin": diag_tf, "b10_status": ov["status"], "diagnostics": recs}
    if ov["status"] != "STOP_STUDENT":
        rec.update(status="NOT_APPLICABLE",
                   reason=f"b1.0 status {ov['status']}: b1.1 #2 changes only a STOP_STUDENT (T reproduces, S-Edi "
                          f"misses the +-{tol}-box rule)")
        v["b11"] = rec
        return v
    rows = (v["arms"].get("S-Edi") or {}).get("rows") or []
    per_clip = [{"clip": r["clip"], "d_mota": r.get("d_mota"),
                 "within": r.get("status") == "scored" and abs(r["d_mota"]) <= mota_tol + 1e-9} for r in rows]
    c1 = sorted(p["clip"] for p in per_clip) == sorted(core.EDI_TEST) and all(p["within"] for p in per_clip)
    valid = [d for d in recs if d["valid_for_b11"]]
    pub_env = [d for d in valid if d.get("transformers_pin") == diag_tf and d["arms"]["S-Edi"]["reproduces"]]
    c2 = bool(pub_env)
    rt_pub = {d.get("databricks_runtime") for d in pub_env}
    ctrl_ok = [d for d in valid if d.get("transformers_pin") == protocol_pin and d["arms"]["S-Edi"]["reproduces"]]
    ctrl_bad = [d for d in valid if d.get("transformers_pin") == protocol_pin and not d["arms"]["S-Edi"]["reproduces"]
                and d.get("databricks_runtime") in rt_pub]
    cause = "runtime_image" if c2 and ctrl_ok else ("transformers" if c2 and ctrl_bad else "not_isolated")
    cause_text = {"runtime_image": "the runtime image's library stack: the same code at this run's transformers pin "
                                   f"{protocol_pin} also reproduces on the original image ("
                                   + "; ".join(_b11_env_text(d) for d in ctrl_ok) + ")",
                  "transformers": f"transformers {diag_tf} vs {protocol_pin}: at {protocol_pin} the same runtime does "
                                  "not reproduce",
                  "not_isolated": "not isolated by the diagnostics run"}[cause]
    rec.update(per_clip_mota=per_clip, cond1_mota_within_tol=c1, cond2_diag_reproduces=c2,
               diag_reproducing=[d["tag"] for d in pub_env], cause=cause, cause_text=cause_text)
    if c1 and c2:
        rec["status"] = "ACCEPT"
        v["overall"] = {
            "status": "PASS_B11", "stop": False, "b10": ov,
            "message": (f"the teacher reproduces the published per-clip counts within +-{tol} boxes; S-Edi misses that "
                        f"rule on this runtime but is within +-{mota_tol} MOTA of the published value on all 4 test "
                        f"clips, and on the published student's environment ("
                        + "; ".join(_b11_env_text(d) for d in pub_env)
                        + f") the same code reproduces the published S-Edi counts within +-{tol} boxes on all 4 clips: "
                          f"the residual is environmental (cause: {cause_text}). S-Edi is accepted under "
                          "BENCH_PROTOCOL b1.1 #2; the published student checkpoint (EMA epoch 75) is the one behind "
                          "92.29 / 96.15")}
    else:
        missing = sorted(set(core.EDI_TEST) - {p["clip"] for p in per_clip})
        why = ([] if c1 else [f"S-Edi |dMOTA| > {mota_tol} (or unscored) on "
                              + ", ".join([p["clip"] for p in per_clip if not p["within"]] + missing)]) \
            + ([] if c2 else [f"no valid transformers {diag_tf} diagnostic reproduces the published S-Edi counts "
                              f"(diagnostics: " + ("; ".join(f"{d['tag']} valid={d['valid_for_b11']}"
                                                            for d in recs) or "none") + ")"])
        rec.update(status="REJECT", reasons=why)
        v["overall"] = dict(ov, message=ov["message"] + " | BENCH_PROTOCOL b1.1 #2 does not apply: " + "; ".join(why))
    v["b11"] = rec
    return v


# ---------------------------------------------------------------------------------------------------- fidelity
def _ustd(a):
    m = a.mean()
    return float(np.sqrt(((a - m) ** 2).sum() / (a.size - 1)))


def fidelity_metrics(s, t):
    """Student vs teacher backbone features ([1,]C,H,W), float64 on the host. Definitions = the training loss
    (Fixed cell 17) and the paper's verification (cell 21):
      cos_patch   mean over the H*W positions of the C-dim cosine (1 - F.cosine_embedding_loss, eps 1e-12)
      cos_global  cosine of the flattened tensors (F.cosine_similarity, cell 21 'Cosine')
      scale_ratio std(student) / std(teacher), unbiased (torch .std(), cell 21 'Scale Ratio')
      mse         F.mse_loss (cell 21 'MSE');  mse_dir  MSE of the channel-L2-normalised maps (F.normalize dim=1)
      ch_scale    MSE of per-channel std (unbiased) + MSE of per-channel mean over H*W
      loss_total  1.0 mse_dir + 0.5 (1 - cos_patch) + 0.3 ch_scale + 0.1 mse (BackboneDistillationLoss, batch 1)"""
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    while s.ndim > 3 and s.shape[0] == 1:
        s = s[0]
    while t.ndim > 3 and t.shape[0] == 1:
        t = t[0]
    if s.shape != t.shape or s.ndim != 3:
        raise ValueError(f"feature shapes {s.shape} vs {t.shape}")
    C = s.shape[0]
    S, T = s.reshape(C, -1), t.reshape(C, -1)
    dot = (S * T).sum(0)
    ns, nt = (S * S).sum(0), (T * T).sum(0)
    cos_p = dot / np.sqrt((ns + COS_LOSS_EPS) * (nt + COS_LOSS_EPS))
    cos_g = float((S * T).sum() / max(math.sqrt(float(ns.sum())) * math.sqrt(float(nt.sum())), COS_SIM_EPS))
    Sn = S / np.maximum(np.sqrt(ns), NORM_EPS)
    Tn = T / np.maximum(np.sqrt(nt), NORM_EPS)
    ch_scale = float(((S.std(axis=1, ddof=1) - T.std(axis=1, ddof=1)) ** 2).mean()
                     + ((S.mean(axis=1) - T.mean(axis=1)) ** 2).mean())
    mse = float(((S - T) ** 2).mean())
    mse_dir = float(((Sn - Tn) ** 2).mean())
    cp = float(cos_p.mean())
    ss, ts = _ustd(S), _ustd(T)
    return {"cos_patch": cp, "cos_patch_p05": float(np.percentile(cos_p, 5)),
            "cos_patch_p50": float(np.percentile(cos_p, 50)), "cos_global": cos_g,
            "scale_ratio": ss / ts if ts > 0 else float("nan"), "mse": mse, "mse_dir": mse_dir, "ch_scale": ch_scale,
            "loss_total": 1.0 * mse_dir + 0.5 * (1.0 - cp) + 0.3 * ch_scale + 0.1 * mse,
            "student_std": ss, "teacher_std": ts}


def cosine_global(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), COS_SIM_EPS))


def consistency_stats(captured, stored):
    """Teacher backbone captured now (fp16 cast, as cell 11 saved it) vs the stored distillation target."""
    a = np.asarray(captured, dtype=np.float32)
    b = np.asarray(stored, dtype=np.float32)
    if a.shape != b.shape:
        return {"shape_equal": False, "captured": list(a.shape), "stored": list(b.shape)}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"shape_equal": True, "frac_equal": float((a == b).mean()), "max_abs": float(d.max()),
            "mean_abs": float(d.mean()), "cos_global": cosine_global(a, b)}


def fidelity_plan(test_clips, stride, n_frames, bridges):
    if stride < 1:
        raise ValueError("fid_stride must be >= 1")
    plan = [("test", c, fi) for c in test_clips for fi in range(0, n_frames, stride)]
    if bridges:
        plan += [("val_bridge", core.EDI_VAL[0], fi) for fi in range(0, OLD_N_FRAMES, stride)]
        plan += [("train20_bridge", core.EDI_TRAIN[0], fi) for fi in range(20)]
    return plan


def summarize(rows, keys=FID_KEYS):
    out = {"n": len(rows)}
    for k in keys:
        v = np.asarray([r[k] for r in rows], dtype=np.float64)
        if not v.size:
            continue
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max()), "median": float(np.median(v))}
    return out


def summarize_fidelity(frame_rows):
    """frame_rows: dicts {group, clip, fi, fp32: metrics, bf16: metrics, cos_bf16_vs_fp32}."""
    out = {"groups": {}, "per_clip": {}}
    groups = sorted({r["group"] for r in frame_rows})
    for g in groups:
        rs = [r for r in frame_rows if r["group"] == g]
        out["groups"][g] = {"fp32": summarize([r["fp32"] for r in rs]), "bf16": summarize([r["bf16"] for r in rs]),
                            "cos_bf16_vs_fp32": summarize([{"c": r["cos_bf16_vs_fp32"]} for r in rs], ("c",)).get("c"),
                            "clips": sorted({r["clip"] for r in rs})}
        for c in sorted({r["clip"] for r in rs}):
            rc = [r for r in rs if r["clip"] == c]
            out["per_clip"][f"{g}/{c}"] = {"fp32": summarize([r["fp32"] for r in rc]),
                                           "bf16": summarize([r["bf16"] for r in rc])}
    return out


def bridge_checks(summ):
    out = {}
    tr = summ["groups"].get("train20_bridge", {}).get("fp32")
    if tr and tr.get("n"):
        out["train20_vs_paper_cell21"] = {
            k: {"ours": tr[k]["mean"], "paper": PAPER_CELL21[k], "diff": tr[k]["mean"] - PAPER_CELL21[k],
                "within_tol": abs(tr[k]["mean"] - PAPER_CELL21[k]) <= BRIDGE_TOL_CELL21}
            for k in PAPER_CELL21}
        out["train20_vs_paper_cell21"]["tolerance"] = BRIDGE_TOL_CELL21
    va = summ["groups"].get("val_bridge", {}).get("fp32")
    if va and va.get("n"):
        ours = {"ema_cosine_loss": 1.0 - va["cos_patch"]["mean"], "ema_val_loss": va["loss_total"]["mean"]}
        out["val_vs_history_epoch75"] = {
            k: {"ours": ours[k], "history": HIST_EP75[k], "diff": ours[k] - HIST_EP75[k],
                "within_tol": abs(ours[k] - HIST_EP75[k]) <= BRIDGE_TOL_VAL} for k in HIST_EP75}
        out["val_vs_history_epoch75"]["tolerance"] = BRIDGE_TOL_VAL
        out["val_vs_history_epoch75"]["note"] = ("history = fp16 autocast, raw-vs-EMA as logged, batches of 4 over all "
                                                 "600 frames; ours = fp32, every fid_stride-th frame")
    flags = [v["within_tol"] for blk in out.values() if isinstance(blk, dict)
             for v in blk.values() if isinstance(v, dict) and "within_tol" in v]
    out["all_within_tol"] = all(flags) if flags else None          # None = no bridge row was computed
    return out


def summarize_gn(res, gn_max=None):
    """res: {variant: {input: {"train_vs_eval": [{"seed","max","mean"}], ...}}}, cross: {input: max|eval-eval|}."""
    gn_max = core.GN_TRAIN_EVAL_MAX if gn_max is None else gn_max
    v = res["variants"]
    dp0 = v["drop_path_0"]
    dft = v["timm_default"]
    m0 = max(d["max"] for inp in dp0["inputs"].values() for d in inp["train_vs_eval"])
    md = max(d["max"] for inp in dft["inputs"].values() for d in inp["train_vs_eval"])
    cross = max(res["eval_default_vs_eval_dp0"].values())
    return {"max_train_eval_diff_drop_path_0": m0, "max_train_eval_diff_timm_default": md,
            "max_eval_diff_between_variants": cross, "gn_gate": gn_max,
            "groupnorm_claim_holds_with_drop_path_0": bool(m0 < gn_max),
            "timm_default_breaks_train_eval_identity": bool(md >= gn_max),
            "inference_unaffected_by_drop_path": bool(cross == 0.0),
            "timm_default_drop_path_probs": dft.get("drop_path_probs"),
            "published": PUB_GN}


def build_summary(a_res, b_ver, c_res, d_res):
    """Final AUDIT_SUMMARY (json, markdown) from the four published stage results (deterministic)."""
    s = {"notebook": NB_VERSION, "bench_version": core.BENCH_VERSION}
    md = [f"# Edinburgh audit ({NB_VERSION})", ""]
    if a_res is not None:
        s["a_seeds"] = {"clips": a_res["n_clips"], "clips_ok": a_res["n_clips_ok"], "boxes_exact": a_res["n_exact"],
                        "boxes": a_res["n_prompts"], "all_ok": a_res["all_ok"], "strays": a_res["strays"]}
        md += [f"**(a) seeds:** {a_res['n_exact']}/{a_res['n_prompts']} boxes equal GT frame 0 "
               f"({a_res['n_clips_ok']}/{a_res['n_clips']} clips fully OK); strays: {a_res['strays'] or 'none'}", ""]
    if b_ver is not None:
        ov = b_ver["overall"]
        s["b_reproduce_old"] = {"overall": ov, "scorer_valid": b_ver["scorer_valid"],
                                "arms": {a: {k: v for k, v in d.items() if k != "rows"}
                                         for a, d in b_ver["arms"].items()}}
        md += [f"**(b) reproduce-old: {ov['status']}** — {ov['message']}", "",
               "| arm | clip | FP new / pub | FN new / pub | IDSW new / pub | MOTA new / pub | within tol |",
               "|---|---|---|---|---|---|---|"]
        for arm in ("T", "S-Edi"):
            for r in (b_ver["arms"].get(arm) or {}).get("rows", []):
                if r["status"] != "scored":
                    md.append(f"| {arm} | {r['clip']} | missing | | | | False |")
                    continue
                md.append(f"| {arm} | {r['clip']} | {r['fp_new']} / {r['fp_pub']} | {r['fn_new']} / {r['fn_pub']} | "
                          f"{r['idsw_new']} / {r['idsw_pub']} | {r['mota_new']:.2f} / {r['mota_pub']:.2f} | "
                          f"{r['within_tol']} |")
        md.append("")
        b11 = b_ver.get("b11")
        if b11:
            s["b_reproduce_old"]["b11"] = {k: b11.get(k) for k in ("status", "b10_status", "cond1_mota_within_tol",
                                                                    "cond2_diag_reproduces", "diag_reproducing",
                                                                    "cause", "reasons", "reason")}
            md += [f"**BENCH_PROTOCOL b1.1 #2:** {b11['status']} (b1.0 status {b11['b10_status']}"
                   + (f"; cause {b11.get('cause')}" if b11.get("cause") else "") + ")", "",
                   "| diagnostic tag | transformers | torch | DBR | valid | S-Edi clips within tol | T clips within tol |",
                   "|---|---|---|---|---|---|---|"]
            for d in b11.get("diagnostics", []):
                sa, ta = d["arms"].get("S-Edi") or {}, d["arms"].get("T") or {}
                md.append(f"| {d['tag']} | {d.get('transformers')} | {d.get('torch')} | {d.get('databricks_runtime')} | "
                          f"{d['valid_for_b11']} | {sa.get('n_within_tol', '-')}/{sa.get('n_clips', '-')} | "
                          f"{ta.get('n_within_tol', '-')}/{ta.get('n_clips', '-')} |")
            md.append("")
    if c_res is not None:
        g = c_res["summary"]
        s["c_groupnorm"] = g
        md += [f"**(c) GroupNorm:** train/eval max diff {g['max_train_eval_diff_drop_path_0']:.3g} with "
               f"drop_path_rate=0.0 (gate {g['gn_gate']:g}: holds={g['groupnorm_claim_holds_with_drop_path_0']}); "
               f"{g['max_train_eval_diff_timm_default']:.3g} with the timm default; eval outputs of the two "
               f"variants differ by {g['max_eval_diff_between_variants']:.3g} (inference unaffected="
               f"{g['inference_unaffected_by_drop_path']})", ""]
    if d_res is not None:
        te = d_res["summary"]["groups"].get("test", {}).get("fp32", {})
        s["d_fidelity"] = {"test_fp32": {k: te.get(k, {}).get("mean") for k in ("cos_patch", "cos_global",
                                                                                 "scale_ratio", "mse")},
                           "n_test_frames": te.get("n"), "bridges": d_res["bridges"]}
        if te:
            md += [f"**(d) fidelity on the 4 TEST clips** ({te['n']} frames, fp32 student): per-patch cosine "
                   f"{te['cos_patch']['mean']:.4f} ± {te['cos_patch']['sd']:.4f}, global cosine "
                   f"{te['cos_global']['mean']:.4f}, scale ratio {te['scale_ratio']['mean']:.4f}, MSE "
                   f"{te['mse']['mean']:.4f}; bridges within tolerance: {d_res['bridges'].get('all_within_tol')}", ""]
    return s, "\n".join(md) + "\n"


# ---------------------------------------------------------------------------------------------------- old loop
class NumpyBackend:
    """Host stand-in for the torch backend (selftest stub tracker)."""
    def no_grad(self):
        return contextlib.nullcontext()

    def binarize(self, ml):
        return (1.0 / (1.0 + np.exp(-np.asarray(ml, dtype=np.float64)))) > 0.5

    def finite(self, x):
        return bool(np.isfinite(np.asarray(x, dtype=np.float64)).all())

    def reset_peak(self):
        pass

    def mem_gb(self):
        return 0.0

    def peak_gb(self):
        return 0.0

    def gc_empty(self):
        gc.collect()


def bf16_copy(m32, dtype):
    """An eval copy of an fp32 student in `dtype` (bf16). timm 1.0.25's TinyViT caches attention_biases[:, idxs] per
    device on its first eval forward, and Module.to() does not convert that dict (BENCH_PROTOCOL §15, J5 item 6b), so a
    copy of a student that already ran an fp32 eval forward (the GroupNorm gate) fails in SDPA with "invalid dtype for
    bias" (J3 tag v1, stage d). The cache is dropped after the cast and rebuilt from the cast parameters (the Tol_02b fix,
    tol68-2026-10-15.2); the fp32 original is untouched."""
    m = copy.deepcopy(m32).to(dtype).eval()
    for mod in m.modules():
        if isinstance(getattr(mod, "attention_bias_cache", None), dict):
            mod.attention_bias_cache = {}
    return m


def n_processed(sess):
    return len(getattr(sess, "processed_frames", None) or {})


def run_old_protocol(frames, prompts_ordered, proc, model, be, *, device=None, dtype=None, student_fn=None,
                     dummy=None, prune_k=None, prune_every=OLD_PRUNE_EVERY, gc_every=None, max_frames=None,
                     capture=None, capture_fis=(), progress_every=100, label=""):
    """The OLD loop (cell 17 teacher / cell 26 student), one clip, one session. Implicit frame index (no frame_idx),
    processed_frames never popped, ONE add_inputs at frame 0 with obj_ids.copy(). Returns (annots, stats)."""
    prompts = old_prompt_boxes(prompts_ordered)
    obj_names = sorted(prompts.keys())
    obj_ids = list(range(len(obj_names)))
    input_boxes = [[prompts[n] for n in obj_names]]                  # prepare_sam3_bbox_format(bboxes) = [bboxes]
    session = proc.init_video_session(inference_device=device, dtype=dtype)
    annots = {oid: {} for oid in obj_ids}
    st = {"label": label, "frames": 0, "prunes": 0, "gc_calls": 0, "empty_masks": {str(o): 0 for o in obj_ids},
          "vram_gb_every_50": [], "obj_names": obj_names, "prune_k": prune_k, "gc_every": gc_every}
    be.reset_peak()
    st["mem_before_gb"] = be.mem_gb()
    t0 = time.time()
    it = iter(frames)
    try:
        for fi, frame, meta in it:
            if max_frames is not None and fi >= max_frames:
                break
            if fi != st["frames"]:
                raise core.BenchAssertError(f"frame index {fi} != {st['frames']}")
            inputs = proc(images=frame, device=device, return_tensors="pt")
            if fi == 0:
                proc.add_inputs_to_inference_session(
                    inference_session=session, frame_idx=0,
                    obj_ids=obj_ids.copy(), input_boxes=input_boxes,
                    original_size=inputs.original_sizes[0])
            if capture is not None:
                capture.want(fi if fi in capture_fis else None)
            with be.no_grad():
                if student_fn is not None:
                    dummy.student_features = student_fn(frame)
                out = model(inference_session=session, frame=inputs.pixel_values[0])
            if student_fn is not None:
                dummy.student_features = None
            if capture is not None:
                capture.want(None)
            if not be.finite(out.pred_masks):
                raise core.BenchAssertError(f"{label}: non-finite pred_masks at frame {fi}")
            masks = proc.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes, binarize=False)[0]
            fname = meta["file"]
            for i, oid in enumerate(obj_ids):
                ml = masks[i, -1] if len(masks.shape) == 4 else masks[i]
                mb = be.binarize(ml)
                bbox, area = bbox_from_binary(mb)
                if area == 0:
                    st["empty_masks"][str(oid)] += 1
                annots[oid][fname] = {"bounding_box": bbox, "object_name": obj_names[oid], "mask_area": area}
            del inputs, out, masks
            st["frames"] += 1
            if prune_k is not None and (fi + 1) % prune_every == 0:
                prune_inference_session(session, keep_last_n=prune_k)
                be.gc_empty()
                st["prunes"] += 1
            if gc_every and (fi + 1) % gc_every == 0:
                be.gc_empty()
                st["gc_calls"] += 1
            if (fi + 1) % 50 == 0:
                st["vram_gb_every_50"].append([fi + 1, round(be.mem_gb(), 4)])
            if progress_every and (fi + 1) % progress_every == 0:
                el = time.time() - t0
                log(f"  [{label}] {fi + 1} frames, {(fi + 1) / max(el, 1e-9):.2f} fps, {be.mem_gb():.2f} GB allocated")
    finally:
        close = getattr(it, "close", None)
        if callable(close):
            close()
    st["wall_s"] = round(time.time() - t0, 2)
    st["fps"] = round(st["frames"] / max(st["wall_s"], 1e-9), 4)
    st["peak_gb"] = round(be.peak_gb(), 4)
    # J0 smoke (2026-10-14): the T session (~16.9 GB) was still allocated when the S-Edi run started and was freed only
    # at its first in-loop gc, so S-Edi's peak included it. The peak is comparable with the published one only when the
    # run started from (about) weights-only memory; peak_above_start_gb is reported alongside.
    st["peak_above_start_gb"] = round(st["peak_gb"] - st["mem_before_gb"], 4)
    st["peak_comparable"] = bool(st["mem_before_gb"] < OLD_PEAK_START_MAX_GB)
    st["obj_num"] = session.get_obj_num()
    st["obj_ids_intact"] = obj_ids == list(range(len(obj_names)))
    st["processed_frames_end"] = n_processed(session)
    st["noncond_end_per_obj"] = [len(session.output_dict_per_obj[i].get("non_cond_frame_outputs", {}))
                                 for i in range(session.get_obj_num())]
    # J0b (BENCH_PROTOCOL §15): release the session IN PLACE after the run (outputs unaffected) -- dropping the reference
    # left the 16.9 GB teacher session allocated into the S-Edi run of the J0 smoke
    release = getattr(core, "release_session", None)
    st["session_released_mib"] = (release(session) or {}).get("released_mib") if callable(release) else None
    del session
    be.gc_empty()
    return annots, st


# =====================================================================================================================
# local selftest (TOL68_SELFTEST=1 python Tol_68_edinburgh_audit.py) -- no torch, GPU or Volume
# =====================================================================================================================
class _Skip(Exception):
    pass


def _selftest():
    data = os.environ.get("TOL68_SELFTEST_DATA") or osp.join(osp.dirname(CODE_DIR), "catalog", "paper1_ws")
    gt_dir = osp.join(data, "edinburgh_seed_check", "gt")
    pr_dir = osp.join(data, "edinburgh_seed_check", "prompts")
    mc = osp.join(data, "mount_copies")
    reg_yaml = osp.join(_TP_DIR, "paths_v1.yaml")
    results = []

    def need(p):
        if not osp.exists(p):
            raise _Skip(f"missing {p}")
        return p

    def jload(p):
        with open(need(p), encoding="utf-8") as f:
            return json.load(f)

    def gt_of(clip):
        return jload(osp.join(gt_dir, f"{clip}_output.json"))

    def run(name, fn):
        try:
            fn()
            results.append((name, "PASS", ""))
        except _Skip as e:
            results.append((name, "SKIP", str(e)))
        except Exception:  # noqa: BLE001
            results.append((name, "FAIL", traceback.format_exc()))

    # ---- 1 static audit of this file
    def t_static():
        with open(osp.abspath(__file__), encoding="utf-8") as f:
            src = f.read()
        lines = src.splitlines()
        assert lines[0] == "# Databricks notebook source", lines[0]
        assert src.count("# COMMAND ----------") >= 8
        forbidden = ["/Vol" + "umes", "/db" + "fs", "db" + "fs:", "Pig Behavior " + "Edinburgh", "mnt/play" + "behavior",
                     "C:" + "/Users", "C:" + "\\Users", "App" + "Data", "hy6" + "25", "@gm" + "ail", "817b" + "0017",
                     "un" + "mount(", "rm" + "tree(", "fs." + "rm(", "Final " + "Notebooks", "account." + "key"]
        hits = [w for w in forbidden if w in src]
        assert not hits, f"forbidden literals in the notebook: {hits}"
        assert not re.search(r"hf_[A-Za-z0-9]{20,}", src), "token-shaped literal"
        assert "hf_token_from_dbutils(dbutils)" in src and not re.search(r"print\([^)]*HF_TOKEN", src)
        # the token reaches the environment only from the secret-scope value, exactly once, after it is fetched
        env_set = 'os.environ["HF_' + 'TOKEN"]'              # split so these test lines do not match themselves
        fetch = "HF_" + "TOKEN = core.hf_token_from_dbutils(dbutils)"
        assert src.count(env_set) == 1 and src.count(env_set + " = HF_" + "TOKEN\n") == 1 and src.count(fetch) == 1
        assert src.index(fetch) < src.index(env_set)
        assert not re.search(r"(log|write_\w+|json\.dump\w*)\([^)]*HF_TOKEN", src)
        assert "frame_idx=int(fi)" not in src.split("def run_old_protocol", 1)[1].split("def ", 1)[0]
        assert "publish_no_overwrite" in src and "R.out(" in src

    # ---- 2 prompt parsing
    def t_prompt_parse():
        txt = "id 1: 958,219,155,98\n\nnot a line\n id 0 : 716.9,224,213,145 \nid 2:1,2,3,4\n"
        o = parse_prompt_txt_ordered(txt)
        assert [n for n, _ in o] == ["id 1", "id 0", "id 2"]
        assert o[1][1] == [716.9, 224.0, 213.0, 145.0]
        assert old_prompt_boxes(o) == {"id 1": [958, 219, 1113, 317], "id 0": [716, 224, 929, 369],
                                      "id 2": [1, 2, 4, 6]}
        with tempfile.TemporaryDirectory() as d:
            p = osp.join(d, "b.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt)
            assert core.read_bbox_prompts_txt(p) == {int(k.split()[-1]): v for k, v in old_prompt_boxes(o).items()}
        assert gt_rel_parts("2019_12_02_000005") == ("2019_12_02", "000005", "output.json")
        for bad in ("2019_12_02_00005", "../x", "2019-12-02_000005"):
            try:
                gt_rel_parts(bad)
                raise AssertionError(bad)
            except ValueError:
                pass

    # ---- 3 seed check, synthetic
    def _gt_syn():
        def fr(n, x, y, w, h, gtf=True):
            return {"frameNumber": n, "bbox": {"x": x, "y": y, "width": w, "height": h}, "isGroundTruth": gtf,
                    "visible": True}
        return {"stepSize": 0.1, "objects": [{"id": "1", "frames": [fr(0, 10, 20, 30, 40), fr(5, 11, 20, 30, 40)]},
                                             {"id": "0", "frames": [fr(0, 50, 60, 70, 80)]}]}

    def t_seed_syn():
        good = "id 1: 10,20,30,40\nid 0: 50,60,70,80\n"
        r = seed_check_clip("2019_01_01_000001", good, _gt_syn())
        assert r["ok"] and r["n_exact"] == 2 and r["order_equals_gt_object_order"], r
        assert r["sorted_name_index_equals_id"]
        r = seed_check_clip("x", "id 1: 10,20,30,41\nid 0: 50,60,70,80\n", _gt_syn())
        assert not r["ok"] and r["n_exact"] == 1 and not r["old_int_boxes_equal_protocol_a"]
        r = seed_check_clip("x", "id 0: 50,60,70,80\nid 1: 10,20,30,40\n", _gt_syn())
        assert not r["ok"] and r["n_exact"] == 2 and not r["order_equals_gt_object_order"]
        r = seed_check_clip("x", "id 1: 10,20,30,40\n", _gt_syn())
        assert not r["ok"] and r["missing_gt_ids"] == [0]
        g = _gt_syn()
        g["objects"][1]["frames"][0]["frameNumber"] = 3
        r = seed_check_clip("x", good, g)
        assert not r["ok"] and not r["all_first_keyframes_at_0"]
        g = _gt_syn()
        g["objects"][0]["frames"][0]["isGroundTruth"] = False
        assert not seed_check_clip("x", good, g)["ok"]
        r = seed_check_clip("x", good + "pig 7: 1,1,1,1\n", _gt_syn())
        assert not r["ok"] and r["issues"]

    # ---- 4 seed check on the 10 real clips (local copies of the catalog inputs)
    def t_seed_real():
        txts = sorted(glob.glob(osp.join(need(pr_dir), "BoundingBoxes_*.txt")))
        assert len(txts) == 10, len(txts)
        recs = []
        for p in txts:
            clip = osp.basename(p)[len("BoundingBoxes_"):-4]
            with open(p, encoding="utf-8") as f:
                recs.append(seed_check_clip(clip, f.read(), gt_of(clip)))
        assert all(r["ok"] for r in recs), [(r["clip"], r["issues"]) for r in recs if not r["ok"]]
        assert sum(r["n_exact"] for r in recs) == 80 and all(r["sorted_name_index_equals_id"] for r in recs)
        assert all(r["bbox_key_order_xywh"] for r in recs)
        assert "TOTAL exact 80/80" in seed_table_text(recs)
        upper = sorted(r["clip"] for r in recs if r["name_prefixes"] == ["ID"])
        assert upper == ["2019_11_05_000009", "2019_11_11_000036"], upper      # name-format variant (not test clips)
        assert all(r["name_prefixes"] == ["id"] for r in recs if r["clip"] in core.EDI_TEST)

    # ---- 5 verbatim hold GT == core.edinburgh_gt_rows(interpolate=False); interpolation differs only between kfs
    def t_gt_hold():
        n_files = 0
        for p in sorted(glob.glob(osp.join(need(gt_dir), "*_output.json"))):
            g = jload(p)
            hold = old_gt_boxes(g)
            rows = core.edinburgh_gt_rows(core.read_edinburgh_keyframes(g), interpolate=False)
            ref = {(int(r[0]), int(r[1])): (r[2], r[3], r[2] + r[4], r[3] + r[5]) for r in rows}
            assert set(hold) == set(ref) and all(tuple(map(float, hold[k])) == ref[k] for k in ref), p
            assert len(hold) == 8 * OLD_N_FRAMES, (p, len(hold))
            itp = interp_gt_boxes(g)
            assert set(itp) == set(hold)
            for o in g["objects"]:                      # keyframes themselves are identical in both
                for fr in o["frames"]:
                    k = (int(fr["frameNumber"]), int(o["id"]))
                    assert np.allclose(itp[k], hold[k])
            n_files += 1
        assert n_files == 10

    # ---- 6 bbox_from_binary = the old argwhere logic
    def t_bbox():
        rng = np.random.default_rng(3)
        for _ in range(50):
            m = rng.random((23, 31)) > 0.93
            b, a = bbox_from_binary(m)
            if not m.any():
                assert (b, a) == ([0, 0, 0, 0], 0)
                continue
            ys, xs = np.nonzero(m)
            assert b == [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()] and a == int(m.sum())
        assert bbox_from_binary(np.zeros((4, 5), bool)) == ([0, 0, 0, 0], 0)
        one = np.zeros((4, 5), bool)
        one[0, 0] = True
        assert bbox_from_binary(one) == ([0, 0, 0, 0], 1)     # degenerate, but NOT empty (area decides)

    # ---- 7-8 the old loop on the core's stub tracker (teacher config, student config), old files, MOT, scorer
    def t_old_loop():
        sys.path.insert(0, CODE_DIR)
        try:
            import test_tol_bench_core as tb
        except ImportError as e:
            raise _Skip(f"test_tol_bench_core not importable: {e}")
        H, W, N = 100, 160, 60
        world = {"a": tb.static((10, 10, 30, 40)), "b": tb.moving((50, 20, 80, 60), 0.5),
                 "c": tb.static((100, 50, 140, 90), frames=set(range(0, 40)))}
        ordered = [("id 1", [50.0, 20.0, 30.0, 40.0]), ("id 0", [10.0, 10.0, 20.0, 30.0]),
                   ("id 2", [100.0, 50.0, 40.0, 40.0])]
        names = {"id 0": "a", "id 1": "b", "id 2": "c"}

        def frames(n):
            for fi in range(n):
                yield fi, f"img{fi}", {"file": f"{fi + 1:07d}.jpg"}

        for cfg in ({"prune_k": None, "gc_every": OLD_TEACHER_GC_EVERY, "student": False},
                    {"prune_k": OLD_STUDENT_PRUNE_K, "gc_every": None, "student": True}):
            proc, model = tb.StubProcessor(H, W), tb.StubModel(world, H, W)
            dummy = type("D", (), {"student_features": None})()
            seen = []
            if cfg["student"]:
                inner = model.__call__

                class _M:
                    def __call__(self, **kw):
                        assert dummy.student_features is not None
                        assert "frame_idx" not in kw            # implicit index (old protocol)
                        seen.append(dummy.student_features)
                        return inner(**kw)
                mdl = _M()
                sfn = lambda pil: ("feat", pil)                  # noqa: E731
            else:
                mdl, sfn = model, None
            annots, st = run_old_protocol(frames(N), ordered, proc, mdl, NumpyBackend(), student_fn=sfn,
                                          dummy=dummy, prune_k=cfg["prune_k"], gc_every=cfg["gc_every"],
                                          progress_every=0, label="stub")
            sess = proc.sessions[-1]
            assert sess.add_calls == [(0, [0, 1, 2])], sess.add_calls
            assert st["obj_ids_intact"] and st["frames"] == N and st["processed_frames_end"] == N
            assert sess.cond_reuse == 0 and not sess.unprompted
            if cfg["prune_k"]:
                assert st["prunes"] == 2 and st["noncond_end_per_obj"] == [18, 18, 18], st
                assert len(seen) == N and dummy.student_features is None
            else:
                assert st["prunes"] == 0 and st["noncond_end_per_obj"] == [N - 1] * 3, st
            assert st["empty_masks"] == {"0": 0, "1": 0, "2": N - 40}, st["empty_masks"]
            for oid, nm in enumerate(sorted(names)):
                for fi in range(N):
                    wb = world[names[nm]](fi)
                    a = annots[oid][f"{fi + 1:07d}.jpg"]
                    assert a["object_name"] == nm
                    if wb is None:
                        assert a["bounding_box"] == [0, 0, 0, 0] and a["mask_area"] == 0
                    else:
                        x1, y1, x2, y2 = (int(round(v)) for v in wb)
                        assert a["bounding_box"] == [x1, y1, x2 - 1 - x1, y2 - 1 - y1], (a, wb)
            with tempfile.TemporaryDirectory() as d:
                write_old_annotations(d, annots)
                tr = load_old_annotations(d)
            assert len(tr) == 3 * N
            gt = {(fi, oid): tr[(fi, oid)] for (fi, oid) in tr if annots[oid][f"{fi + 1:07d}.jpg"]["mask_area"] > 0}
            try:
                sc = old_score("stub", gt, tr, n_frames=N)
            except ImportError:
                raise _Skip("motmetrics not installed")
            assert sc["fn"] == 0 and sc["idsw"] == 0 and sc["fp"] == N - 40, sc     # [0,0,0,0] rows are FPs
            lines = old_annots_to_mot(annots)
            assert len(lines) == 3 * N - (N - 40)
            assert lines[0] == "1,1,10,10,20,30,1,-1,-1,-1", lines[0]

    # ---- 9 verbatim scorer on the surviving per-frame boxes == the published per-clip numbers
    def t_scorer_published():
        try:
            import motmetrics
        except ImportError:
            raise _Skip("motmetrics not installed")
        assert motmetrics.__version__ == EXPECT_MOTMETRICS, motmetrics.__version__
        pubT = published_per_clip(jload(osp.join(mc, PUBLISHED_FILES["T"])), core.EDI_TEST)
        pubS = published_per_clip(jload(osp.join(mc, PUBLISHED_FILES["S-Edi"])), core.EDI_TEST)
        final = jload(osp.join(mc, PUBLISHED_FILES["final"]))
        assert check_published("T", pubT, final) and check_published("S-Edi", pubS, final)
        bad = copy.deepcopy(pubS)
        bad["2019_12_02_000005"]["fp"] += 1
        try:
            check_published("S-Edi", bad)
            raise AssertionError("check_published accepted a changed count")
        except ValueError:
            pass
        for c in core.EDI_TEST:
            gtb = old_gt_boxes(gt_of(c))
            trb = load_old_annotations(need(osp.join(mc, "Teacher_Annotations", c)))
            s = old_score(c, gtb, trb)
            assert score_matches(s, pubT[c]), (c, _slim(s), pubT[c])
            gf, tf_ = ids_by_frame(gtb), ids_by_frame(trb)          # grouped lists == the verbatim per-frame scans
            for fi in (0, 1, 299, 599):
                assert gf[fi] == [p for (f, p) in gtb if f == fi] and tf_[fi] == [p for (f, p) in trb if f == fi]

    # ---- 10 the whole (b) scoring path on real data: surviving T as "regenerated T" -> PASS (exact);
    #      the _Improved boxes as "regenerated S-Edi" -> FAIL -> STOP_STUDENT
    def t_score_stage_b():
        try:
            import motmetrics  # noqa: F401
        except ImportError:
            raise _Skip("motmetrics not installed")
        clips = list(core.EDI_TEST)
        gt = {c: gt_of(c) for c in clips}
        survT = {c: load_old_annotations(need(osp.join(mc, "Teacher_Annotations", c))) for c in clips}
        survI = {c: load_old_annotations(need(osp.join(mc, "Student_Backbone_Annotations", c))) for c in clips}
        pub = {"T": published_per_clip(jload(osp.join(mc, PUBLISHED_FILES["T"])), clips),
               "S-Edi": published_per_clip(jload(osp.join(mc, PUBLISHED_FILES["S-Edi"])), clips)}
        imp = published_per_clip(jload(osp.join(mc, PUBLISHED_FILES["improved"])), clips)
        scores, verdict, direct = score_stage_b(clips, gt, {"T": survT, "S-Edi": survI},
                                                {"T": survT, "improved": survI}, pub, imp, REPRO_TOL_BOXES)
        assert verdict["scorer_valid"]
        assert all(v["equals_improved_json"] for v in verdict["scorer_validation"]
                   ["surviving_student_dir_is_improved"].values())
        assert verdict["arms"]["T"]["verdict"] == "PASS" and verdict["arms"]["T"]["max_abs_d_fp"] == 0
        assert verdict["arms"]["S-Edi"]["verdict"] == "FAIL"
        assert verdict["overall"]["status"] == "STOP_STUDENT", verdict["overall"]
        assert abs(verdict["arms"]["T"]["mean_mota_new"] - PUBLISHED_MEAN["T"]["mota"]) < 1e-9
        assert all(direct[c]["frac_identical"] == 1.0 for c in clips)
        assert set(direct["_student_vs_teacher_regenerated"]) == set(clips)
        for c in clips:                     # interpolated GT: every box still present, closed set
            assert scores["T"][c]["interp_gt"]["n_gt_boxes"] == 4800
        json.dumps([scores, verdict, direct], default=core._json_default)
        s, md = build_summary(None, verdict, None, None)
        assert "STOP_STUDENT" in md and "| T | 2019_12_02_000005 | 215 / 215 |" in md, md
        # a clip subset: rows for that clip only, no overall verdict; fewer frames: no rows either
        sc2, v2, _ = score_stage_b(clips[:1], {clips[0]: gt[clips[0]]}, {"T": {clips[0]: survI[clips[0]]}},
                                   {"T": {clips[0]: survT[clips[0]]}}, pub, None, REPRO_TOL_BOXES)
        assert v2["arms"]["T"]["verdict"] == "FAIL" and v2["arms"]["T"]["n_clips"] == 1
        assert v2["overall"]["status"] == "SMOKE" and v2["scorer_valid"]
        _, v3, _ = score_stage_b(clips[:1], {clips[0]: gt[clips[0]]}, {"T": {clips[0]: survT[clips[0]]}},
                                 {"T": {clips[0]: survT[clips[0]]}}, pub, None, REPRO_TOL_BOXES, n_frames=50)
        assert v3["overall"]["status"] == "SMOKE" and "T" not in v3["arms"]
        # T-k8 row (not gated) is reported against our T row
        _, v4, _ = score_stage_b(clips, gt, {"T": survT, "T-k8": survI, "S-Edi": survT},
                                 {"T": survT}, pub, None, REPRO_TOL_BOXES)
        assert v4["arms"]["T-k8"]["verdict"] == "NOT_GATED" and v4["overall"]["status"] == "STOP_STUDENT"
        assert v4["arms"]["T-k8"]["vs_T_oldprot"]["2019_12_02_000005"]["d_fp"] == 736 - 215
        # environment diagnostic: rows are computed, the overall verdict is ENV_DIAG even on the complete set
        _, v5, _ = score_stage_b(clips, gt, {"T": survT, "S-Edi": survI}, {"T": survT}, pub, None, REPRO_TOL_BOXES,
                                 env_diag="transformers 5.3.0, torch 2.6.0")
        assert v5["overall"]["status"] == "ENV_DIAG" and v5["overall"]["stop"] and v5["env_diag"]
        assert v5["arms"]["T"]["verdict"] == "PASS" and v5["arms"]["S-Edi"]["verdict"] == "FAIL"
        _, v6, _ = score_stage_b(clips[:1], {clips[0]: gt[clips[0]]}, {"S-Edi": {clips[0]: survT[clips[0]]}},
                                 {"T": {clips[0]: survT[clips[0]]}}, pub, None, REPRO_TOL_BOXES, env_diag="x")
        assert v6["overall"]["status"] == "ENV_DIAG" and v6["arms"]["S-Edi"]["rows"][0]["fp_new"] == 215
        assert "env_diag" not in v4

    # ---- 11 direct box comparison
    def t_compare():
        a = {(0, 0): (0, 0, 10, 10), (0, 1): (0, 0, 0, 0), (1, 0): (0, 0, 10, 10), (1, 1): (5, 5, 6, 6)}
        b = {(0, 0): (0, 0, 10, 10), (0, 1): (0, 0, 0, 0), (1, 0): (0, 0, 10, 20), (2, 0): (1, 1, 2, 2)}
        r = compare_box_tables(a, b)
        assert r["n_common"] == 3 and r["only_a"] == 1 and r["only_b"] == 1 and r["n_identical"] == 2
        assert abs(r["iou_min"] - 0.5) < 1e-12 and r["n_iou_lt_0.9"] == 1 and r["first_frame_iou_lt_0.9_by_id"] == {"0": 1}
        assert _iou_xyxy_or_empty((0, 0, 0, 0), (0, 0, 5, 5)) == 0.0

    # ---- 12 verdict logic
    def t_verdict():
        pub = {"c1": {"fp": 10, "fn": 10, "idsw": 0, "mota": 99.0, "idf1": 99.0},
               "c2": {"fp": 0, "fn": 0, "idsw": 0, "mota": 100.0, "idf1": 100.0}}
        new = {"c1": {"fp": 15, "fn": 15, "idsw": 0, "mota": 98.8, "idf1": 98.9},
               "c2": {"fp": 0, "fn": 0, "idsw": 0, "mota": 100.0, "idf1": 100.0}}
        assert arm_verdict(repro_rows(new, pub, 5), 5)["verdict"] == "PASS"
        new["c1"]["fp"] = 16
        assert arm_verdict(repro_rows(new, pub, 5), 5)["verdict"] == "FAIL"
        new["c1"]["fp"], new["c2"]["idsw"] = 15, 1
        assert arm_verdict(repro_rows(new, pub, 5), 5)["verdict"] == "FAIL"
        del new["c2"]
        v = arm_verdict(repro_rows(new, pub, 5), 5)
        assert v["verdict"] == "FAIL" and v["n_within_tol"] == 1
        P_, F_ = {"verdict": "PASS"}, {"verdict": "FAIL"}
        assert overall_verdict(P_, P_)["status"] == "PASS" and not overall_verdict(P_, P_)["stop"]
        assert overall_verdict(P_, F_)["status"] == "STOP_STUDENT"
        assert overall_verdict(F_, P_)["status"] == "STOP_TEACHER"
        assert overall_verdict(P_, None)["status"] == "INCOMPLETE"
        assert overall_verdict(P_, P_, scorer_valid=False)["status"] == "SCORER_INVALID"

    # ---- 13 fidelity metrics vs naive references
    def t_fidelity():
        rng = np.random.default_rng(7)
        t = rng.normal(0.1, 1.4, (1, 6, 5, 4))
        m = fidelity_metrics(t, t)
        assert abs(m["cos_patch"] - 1) < 1e-9 and abs(m["cos_global"] - 1) < 1e-12 and m["mse"] == 0
        assert abs(m["scale_ratio"] - 1) < 1e-12 and m["loss_total"] < 1e-9
        m = fidelity_metrics(2 * t, t)
        assert abs(m["scale_ratio"] - 2) < 1e-12 and abs(m["cos_patch"] - 1) < 1e-9 and abs(m["mse"] - (t ** 2).mean()) < 1e-12
        assert abs(fidelity_metrics(-t, t)["cos_global"] + 1) < 1e-12
        s = rng.normal(0, 1, (6, 5, 4))
        m = fidelity_metrics(s, t[0])
        S, T = s.reshape(6, -1), t[0].reshape(6, -1)
        cps = []
        for j in range(S.shape[1]):
            x, y = S[:, j], T[:, j]
            cps.append(float(x @ y) / math.sqrt((float(x @ x) + 1e-12) * (float(y @ y) + 1e-12)))
        assert abs(m["cos_patch"] - statistics.mean(cps)) < 1e-12
        assert abs(m["cos_global"] - float((S * T).sum()) / (np.linalg.norm(S) * np.linalg.norm(T))) < 1e-12
        assert abs(m["scale_ratio"] - statistics.stdev(S.ravel()) / statistics.stdev(T.ravel())) < 1e-12
        sn = S / np.linalg.norm(S, axis=0)
        tn = T / np.linalg.norm(T, axis=0)
        assert abs(m["mse_dir"] - ((sn - tn) ** 2).mean()) < 1e-12
        chs = statistics.mean([(statistics.stdev(S[c]) - statistics.stdev(T[c])) ** 2 for c in range(6)]) + \
            statistics.mean([(statistics.mean(S[c]) - statistics.mean(T[c])) ** 2 for c in range(6)])
        assert abs(m["ch_scale"] - chs) < 1e-12
        assert abs(m["loss_total"] - (m["mse_dir"] + 0.5 * (1 - m["cos_patch"]) + 0.3 * chs + 0.1 * m["mse"])) < 1e-12
        try:
            fidelity_metrics(s, t[0, :5])
            raise AssertionError("shape mismatch accepted")
        except ValueError:
            pass
        c = consistency_stats(t.astype(np.float16), t.astype(np.float16))
        assert c["frac_equal"] == 1.0 and c["max_abs"] == 0.0
        assert consistency_stats(t, t[0])["shape_equal"] is False

    # ---- 14 fidelity plan / summaries / bridges (+ the bridge constants vs the copied history and Final JSON)
    def t_fid_summary():
        plan = fidelity_plan(core.EDI_TEST, 5, OLD_N_FRAMES, True)
        assert len(plan) == 4 * 120 + 120 + 20 and plan[0] == ("test", core.EDI_TEST[0], 0)
        assert plan[-1] == ("train20_bridge", "2019_11_05_000002", 19)
        assert len(fidelity_plan(core.EDI_TEST[:1], 1, 30, False)) == 30
        rows = []
        for g, c, fi in plan:
            base = {k: 0.8 for k in FID_KEYS}
            if g == "train20_bridge":
                base.update(PAPER_CELL21)
            if g == "val_bridge":
                base.update({"cos_patch": 1 - HIST_EP75["ema_cosine_loss"], "loss_total": HIST_EP75["ema_val_loss"]})
            rows.append({"group": g, "clip": c, "fi": fi, "fp32": dict(base), "bf16": dict(base),
                         "cos_bf16_vs_fp32": 0.9999})
        s = summarize_fidelity(rows)
        assert s["groups"]["test"]["fp32"]["n"] == 480 and len(s["per_clip"]) == 6
        b = bridge_checks(s)
        assert b["all_within_tol"] is True and abs(b["train20_vs_paper_cell21"]["cos_global"]["diff"]) < 1e-9
        assert bridge_checks(summarize_fidelity([r for r in rows if r["group"] == "test"]))["all_within_tol"] is None
        rows[-1]["fp32"]["cos_global"] = 0.9
        assert not bridge_checks(summarize_fidelity(rows))["all_within_tol"]
        h = jload(osp.join(mc, "backbone_distill_history.json"))
        e75 = [e for e in h if e["epoch"] == 75][0]
        assert e75["ema_cosine"] == HIST_EP75["ema_cosine_loss"] and e75["ema_val_loss"] == HIST_EP75["ema_val_loss"]
        assert e75["ema_val_loss"] == CKPT_EMA_VAL_LOSS
        f = jload(osp.join(mc, PUBLISHED_FILES["final"]))
        assert f["teacher"]["avg_mota"] == PUBLISHED_MEAN["T"]["mota"]
        assert abs(f["student"]["avg_mota"] - PUBLISHED_MEAN["S-Edi"]["mota"]) < 1e-12
        d_res = {"summary": s, "bridges": b}
        _, md = build_summary(None, None, None, d_res)
        assert "fidelity on the 4 TEST clips" in md

    # ---- 15 GroupNorm summary
    def t_gn():
        res = {"variants": {
            "drop_path_0": {"inputs": {"x": {"train_vs_eval": [{"seed": 0, "max": 0.0, "mean": 0.0}]}}},
            "timm_default": {"drop_path_probs": [0.1, 0.2],
                             "inputs": {"x": {"train_vs_eval": [{"seed": 0, "max": 41.0, "mean": 0.03}]}}}},
            "eval_default_vs_eval_dp0": {"x": 0.0}}
        g = summarize_gn(res)
        assert g["groupnorm_claim_holds_with_drop_path_0"] and g["timm_default_breaks_train_eval_identity"]
        assert g["inference_unaffected_by_drop_path"]
        res["variants"]["drop_path_0"]["inputs"]["x"]["train_vs_eval"][0]["max"] = 1e-3
        res["eval_default_vs_eval_dp0"]["x"] = 1e-7
        g = summarize_gn(res)
        assert not g["groupnorm_claim_holds_with_drop_path_0"] and not g["inference_unaffected_by_drop_path"]
        _, md = build_summary(None, None, {"summary": g}, None)
        assert "(c) GroupNorm" in md

    # ---- 16 registry: every input key resolves inside the catalog, outputs only under paper1_distillation/
    def t_registry():
        R = tol_paths.load(path=need(reg_yaml))
        out = R.out("paper1.root", "evidence", "edinburgh_audit", "v1")
        assert out == R.volume + "/paper1_distillation/evidence/edinburgh_audit/v1", out
        for k in REGISTRY_KEYS:
            p = R.path(k)
            assert p.startswith(R.volume + "/") and "/Tol/" not in p + "/", (k, p)
        assert R.path("datasets.edinburgh_pigs.decoded_frames", None, "2019_12_02_000005", "video1").endswith(
            "/datasets/edinburgh_pigs/decoded_frames/2019_12_02_000005/video1")
        assert R.path("datasets.edinburgh_pigs.annotated", None, *gt_rel_parts("2019_12_02_000005")).endswith(
            "/datasets/edinburgh_pigs/annotated/2019_12_02/000005/output.json")
        assert R.path("paper1.teacher_targets.edinburgh", None, "2019_12_02_000005", "backbone_000000.pt").endswith(
            "/paper1_distillation/teacher_targets/edinburgh/2019_12_02_000005/backbone_000000.pt")
        for bad in ("paper1.teacher_targets.tol_v1", "paper2.tol_root"):
            try:
                R.out(bad)
                raise AssertionError(bad)
            except tol_paths.RegistryError:
                pass
        try:
            R.path("paper1.root", None, "..", "Tol")
            raise AssertionError("'..' accepted")
        except tol_paths.RegistryError:
            pass
        prov = R.provenance(list(REGISTRY_KEYS))
        assert set(prov["resolved"]) == set(REGISTRY_KEYS)

    # ---- 17 widget parsing
    def t_widgets():
        assert parse_csv("a, c,d ,b", {"a", "b", "c", "d"}) == ["a", "c", "d", "b"]
        assert parse_ints("0,150, 599") == [0, 150, 599]
        assert parse_bool("True") and not parse_bool("false") and not parse_bool("")
        for fn in (lambda: parse_csv("a,e", {"a"}), lambda: parse_csv("a,a"), lambda: parse_ints("1,x"),
                   lambda: parse_bool("maybe")):
            try:
                fn()
                raise AssertionError("accepted")
            except ValueError:
                pass
        assert set(ARM_DIRS) == set(ARMS_B) and all(v.endswith("__oldprot") for v in ARM_DIRS.values())
        assert TAG_RX.match("v1") and not TAG_RX.match("../v1")

    # ---- 18 environment diagnostics: the pre-restart cell, the %pip cell and env_spec agree
    def t_env():
        pin = core.TRANSFORMERS_PIN
        assert pin == "5.2.0" and pin in ENV_TF_ALLOWED
        assert pip_requirements("5.2.0") == ["numpy<2.0", "transformers==5.2.0", "Pillow==11.1.0", "huggingface_hub",
                                             "accelerate", "timm==1.0.25", "scipy", "motmetrics==1.4.0"]
        assert pip_requirements("5.3.0")[1] == "transformers==5.3.0"
        with open(osp.abspath(__file__), encoding="utf-8") as f:
            cells = f.read().split("# COMMAND ----------")
        env_i = [i for i, c in enumerate(cells) if c.strip().startswith("# ---- environment selection")]
        pip_i = [i for i, c in enumerate(cells) if c.strip().startswith("# MAGIC %pip")]
        assert sum(c.count("\n# MAGIC %pip") + c.startswith("# MAGIC %pip") for c in cells) == 1   # one %pip line
        assert len(env_i) == 1 and len(pip_i) == 1 and pip_i[0] == env_i[0] + 1, (env_i, pip_i)
        assert cells[pip_i[0]].strip() == f"# MAGIC %pip install -r {REQ_PATH}", cells[pip_i[0]]
        assert env_i[0] == 1, "the env cell must be the first code cell (before %pip and the restart)"

        class _W:
            def __init__(self, vals):
                self.vals = dict(vals)

            def text(self, name, default):
                self.vals.setdefault(name, default)

            def get(self, name):
                return self.vals[name]

        class _D:
            def __init__(self, vals):
                self.widgets = _W(vals)

        old = os.environ.get("TOL68_REQ_PATH")
        try:
            with tempfile.TemporaryDirectory() as d:
                os.environ["TOL68_REQ_PATH"] = osp.join(d, "sub", "requirements.txt")
                for vals, want in (({}, "5.2.0"), ({"transformers_pin": "5.3.0"}, "5.3.0"),
                                   ({"transformers_pin": ""}, "5.2.0"), ({"transformers_pin": " 5.3.0 "}, "5.3.0")):
                    ns = {"dbutils": _D(vals), "__name__": "tol68_env_cell"}
                    exec(compile(cells[env_i[0]], "env_cell", "exec"), ns)          # noqa: S102
                    with open(os.environ["TOL68_REQ_PATH"], encoding="utf-8") as f:
                        assert f.read().splitlines() == pip_requirements(want), vals
                    assert tuple(ns["_TOL68_PIP_FIXED"]) == PIP_FIXED and tuple(ns["_TOL68_TF_ALLOWED"]) == ENV_TF_ALLOWED
                try:
                    exec(compile(cells[env_i[0]], "env_cell", "exec"),                # noqa: S102
                         {"dbutils": _D({"transformers_pin": "4.57.1"}), "__name__": "x"})
                    raise AssertionError("4.57.1 accepted")
                except ValueError:
                    pass
                del os.environ["TOL68_REQ_PATH"]
                ns = {"__name__": "x"}                                   # no dbutils: nothing is written
                exec(compile(cells[env_i[0]], "env_cell", "exec"), ns)               # noqa: S102
                assert ns["_tol68_dbu"] is None and ns["_TOL68_REQ_PATH"] == REQ_PATH
        finally:
            if old is None:
                os.environ.pop("TOL68_REQ_PATH", None)
            else:
                os.environ["TOL68_REQ_PATH"] = old
        e = env_spec("", "", "v1", "17.3.x-gpu-ml-scala2.13")
        assert e["transformers_pin"] == pin and not e["env_diag"] and e["expect_torch"] is None
        assert env_spec("5.2.0", "2.7.0", "v1", None)["expect_torch"] == "2.7.0"
        e = env_spec("5.3.0", "2.6.0", "smoke_envdiag_tf530", "16.4.x-gpu-ml-scala2.12")
        assert e["env_diag"] and e["requirements"][1] == "transformers==5.3.0"
        assert env_spec("5.2.0", "", "smoke_envdiag_ctrl", "16.4.x-gpu-ml-scala2.12")["env_diag"]
        for args in (("5.3.0", "", "smoke_j1", None), ("5.3.0", "", "v1", None), ("4.57.1", "", "smoke_envdiag_x", None),
                     ("5.2.0", "2.6", "v1", None), ("5.2.0", "", "v1", "16.4.x-gpu-ml-scala2.12"),
                     ("5.2.0", "", "smoke_j1", "16.4.x-gpu-ml-scala2.12")):
            try:
                env_spec(*args)
                raise AssertionError(f"accepted {args}")
            except ValueError:
                pass

    # ---- 20 bf16_copy drops the per-device attention-bias cache of every module, never the original's
    def t_bf16_copy():
        class _M:
            def __init__(self, cache=None, kids=()):
                if cache is not None:
                    self.attention_bias_cache = cache
                self.kids, self.dtype, self.training = list(kids), "fp32", True

            def to(self, dtype):
                for m in self.modules():
                    m.dtype = dtype
                return self

            def eval(self):
                for m in self.modules():
                    m.training = False
                return self

            def modules(self):
                yield self
                for k in self.kids:
                    yield from k.modules()
        a1, a2 = _M({"cuda:0": "fp32-bias"}), _M({"cuda:0": "fp32-bias", "cpu": "x"})
        root = _M(None, [_M(None, [a1]), a2, _M({})])
        c = bf16_copy(root, "bf16")
        assert c is not root and all(m.dtype == "bf16" and not m.training for m in c.modules())
        caches = [m.attention_bias_cache for m in c.modules() if hasattr(m, "attention_bias_cache")]
        assert len(caches) == 3 and all(x == {} for x in caches), caches
        assert a1.attention_bias_cache == {"cuda:0": "fp32-bias"} and a2.dtype == "fp32" and root.training
        with open(osp.abspath(__file__), encoding="utf-8") as f:
            src = f.read()
        assert ("copy.deepcopy(" + "s32).to(") not in src and src.count(chr(10) + "    s16 = bf16_copy(s32, torch.bfloat16)" + chr(10)) == 1

    # ---- 19 BENCH_PROTOCOL b1.1 #2 (tol68-2026-10-15.1): pure rule, widget, summary, a real one-clip diagnostic
    def t_b11():
        clips = list(core.EDI_TEST)

        def counts(d):            # FP = FN = 100 + d per clip (4,800 GT boxes): dMOTA = -d / 24
            return {c: {"fp": 100 + d[i], "fn": 100 + d[i], "idsw": 0, "mota": 100 * (1 - (200 + 2 * d[i]) / 4800),
                        "idf1": 97.0} for i, c in enumerate(clips)}
        pubc = counts([0, 0, 0, 0])

        def run_verdict(t_d, s_d, env_diag=False):
            vt = arm_verdict(repro_rows(counts(t_d), pubc, 5), 5)
            vs = arm_verdict(repro_rows(counts(s_d), pubc, 5), 5)
            v = {"tolerance_boxes": 5, "full_clips": True, "complete_test_set": True, "clips": clips,
                 "scorer_valid": True, "arms": {"T": vt, "S-Edi": vs}}
            v["overall"] = {"status": "ENV_DIAG", "stop": True, "message": "diag"} if env_diag else overall_verdict(vt, vs)
            return v

        def diag(tag, tf, rt, s_d, t_d=None, gpu="A10"):
            v = run_verdict(t_d or [0] * 4, s_d, env_diag=True)
            if t_d is None:
                del v["arms"]["T"]
            ac = {"env": {"env_diag": True, "transformers_pin": tf, "databricks_runtime": rt},
                  "versions_pinned": {"transformers": tf, "torch": "2.6.0+cu124", "cuda": "12.4"},
                  "gpu": {"class": gpu}, "notebook": NB_VERSION}
            return b11_diag_record(tag, ac, v)

        v17 = run_verdict([1, 0, 2, -1], [-6, 3, 1, 4])           # T within 5; S-Edi -6 boxes on clip 0
        assert v17["overall"]["status"] == "STOP_STUDENT"
        d530 = diag("smoke_envdiag_a", "5.3.0", "16.4", [0, 0, 0, 0])
        d520 = diag("smoke_envdiag_b", "5.2.0", "16.4", [0, 1, 0, 0], t_d=[0, 0, 0, 0])
        assert d530["arms"]["S-Edi"]["reproduces"] and d530["full_test_set"] and "T" not in d530["arms"]
        assert d520["arms"]["T"]["reproduces"] and d520["arms"]["T"]["n_within_tol"] == 4
        r = apply_b11(v17, [d530, d520], gpu_class="A10")
        assert r["overall"]["status"] == "PASS_B11" and not r["overall"]["stop"] and r["b11"]["status"] == "ACCEPT"
        assert r["overall"]["b10"]["status"] == "STOP_STUDENT" and r["b11"]["cause"] == "runtime_image"
        assert r["b11"]["diag_reproducing"] == ["smoke_envdiag_a"] and r["b11"]["cond1_mota_within_tol"]
        assert v17["overall"]["status"] == "STOP_STUDENT" and "b11" not in v17          # pure: the input is untouched
        json.dumps(r)
        r = apply_b11(v17, [d530], gpu_class="A10")                 # no 5.2.0 control: accepted, cause not isolated
        assert r["overall"]["status"] == "PASS_B11" and r["b11"]["cause"] == "not_isolated"
        d520bad = diag("smoke_envdiag_c", "5.2.0", "16.4", [-6, 0, 0, 0])
        assert apply_b11(v17, [d530, d520bad], gpu_class="A10")["b11"]["cause"] == "transformers"
        # condition 1 fails: d = +30 boxes on clip 0 -> dMOTA -1.25 -> the b1.0 stop rule stands
        r = apply_b11(run_verdict([0] * 4, [30, 0, 0, 0]), [d530], gpu_class="A10")
        assert r["overall"]["status"] == "STOP_STUDENT" and r["overall"]["stop"] and r["b11"]["status"] == "REJECT"
        assert clips[0] in r["b11"]["reasons"][0] and "b1.1 #2 does not apply" in r["overall"]["message"]
        assert abs(r["b11"]["per_clip_mota"][0]["d_mota"] + 1.25) < 1e-9
        # boundary: exactly 1.0 MOTA (24 boxes) is within
        assert apply_b11(run_verdict([0] * 4, [24, 0, 0, 0]), [d530], gpu_class="A10")["overall"]["status"] == "PASS_B11"
        # condition 2 fails: only a 5.2.0 diagnostic / the 5.3.0 one misses / other GPU class / partial / none
        partial = copy.deepcopy(d530)
        partial["full_test_set"] = False
        for ds in ([d520], [diag("smoke_envdiag_d", "5.3.0", "16.4", [0, 0, 7, 0])],
                   [diag("smoke_envdiag_e", "5.3.0", "16.4", [0] * 4, gpu="A100")], [partial], []):
            r = apply_b11(v17, ds, gpu_class="A10")
            assert r["overall"]["status"] == "STOP_STUDENT" and r["b11"]["status"] == "REJECT", [d["tag"] for d in ds]
        assert not apply_b11(v17, [partial], gpu_class="A10")["b11"]["diagnostics"][0]["valid_for_b11"]
        # PASS and STOP_TEACHER are never changed
        for t_d, s_d, st in (([0] * 4, [1] * 4, "PASS"), ([9, 0, 0, 0], [0] * 4, "STOP_TEACHER")):
            r = apply_b11(run_verdict(t_d, s_d), [d530], gpu_class="A10")
            assert r["overall"]["status"] == st and r["b11"]["status"] == "NOT_APPLICABLE", st
        s, md = build_summary(None, apply_b11(v17, [d530, d520], gpu_class="A10"), None, None)
        assert "PASS_B11" in md and "BENCH_PROTOCOL b1.1 #2:** ACCEPT" in md and "| smoke_envdiag_a | 5.3.0 |" in md, md
        assert s["b_reproduce_old"]["b11"]["status"] == "ACCEPT" and s["b_reproduce_old"]["overall"]["status"] == "PASS_B11"
        # the widget
        assert parse_b11_tags("", "v1", False, False) == []
        assert parse_b11_tags("smoke_envdiag_a, smoke_envdiag_b", "v1", False, False) == ["smoke_envdiag_a",
                                                                                          "smoke_envdiag_b"]
        for args in (("smoke_envdiag_a", "v1", True, False), ("smoke_envdiag_a", "v1", False, True),
                     ("v0", "v1", False, False), ("smoke_envdiag_a", "smoke_envdiag_a", False, False),
                     ("smoke_envdiag_a,smoke_envdiag_a", "v1", False, False), ("../smoke_envdiag_a", "v1", False, False)):
            try:
                parse_b11_tags(*args)
                raise AssertionError(f"accepted {args}")
            except ValueError:
                pass
        # a real published one-clip diagnostic (J3e, DBR 16.4, transformers 5.3.0): parses, NOT valid (one clip)
        real = osp.join(osp.dirname(osp.abspath(__file__)), "_envdiag", "smoke_envdiag_tf530_dbr164")
        if not osp.isdir(real):
            raise _Skip(f"missing {real}")
        d = b11_diag_record("smoke_envdiag_tf530_dbr164", jload(osp.join(real, "AUDIT_CONFIG.json")),
                            jload(osp.join(real, "b_reproduce_old", "verdict.json")))
        assert d["transformers"] == "5.3.0" and d["torch"].startswith("2.6.0") and d["gpu_class"] == "A10"
        assert d["status"] == "ENV_DIAG" and not d["full_test_set"] and not d["arms"]["S-Edi"]["reproduces"]
        assert d["arms"]["S-Edi"]["n_within_tol"] == 1 and d["arms"]["S-Edi"]["rows"][0]["fp_new"] == 318
        r = apply_b11(v17, [d], gpu_class="A10")
        assert r["overall"]["status"] == "STOP_STUDENT" and not r["b11"]["diagnostics"][0]["valid_for_b11"]

    for name, fn in [("static audit", t_static), ("prompt parse", t_prompt_parse), ("seed check synthetic", t_seed_syn),
                     ("seed check 10 real clips", t_seed_real), ("hold GT == core", t_gt_hold),
                     ("bbox_from_binary", t_bbox), ("old loop on stub tracker", t_old_loop),
                     ("old scorer == published", t_scorer_published), ("stage-b scoring end to end", t_score_stage_b),
                     ("box comparison", t_compare), ("verdict logic", t_verdict), ("fidelity metrics", t_fidelity),
                     ("fidelity summary + bridges", t_fid_summary), ("GroupNorm summary", t_gn),
                     ("registry paths", t_registry), ("widgets", t_widgets), ("environment diagnostics", t_env),
                     ("BENCH_PROTOCOL b1.1 #2 rule", t_b11), ("bf16 copy drops the bias cache", t_bf16_copy)]:
        run(name, fn)
    for name, status, detail in results:
        print(f"{status:4s}  {name}" + (f"  ({detail})" if status == "SKIP" else ""))
        if status == "FAIL":
            print(detail)
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"Tol_68 SELFTEST: {sum(1 for _, s, _ in results if s == 'PASS')} pass, "
          f"{sum(1 for _, s, _ in results if s == 'SKIP')} skip, {n_fail} fail")
    return n_fail == 0


# COMMAND ----------

if SELFTEST and IS_NOTEBOOK:
    raise RuntimeError("the selftest is local-only: TOL68_SELFTEST=1 python Tol_68_edinburgh_audit.py")
if SELFTEST or not IS_NOTEBOOK:
    ok = _selftest() if SELFTEST else True
    if not SELFTEST:
        print("not a notebook and TOL68_SELFTEST != 1: nothing to run")
    raise SystemExit(0 if ok else 1)

# COMMAND ----------

# ---- widgets, registry, output root, run config ----------------------------------------------------------------------
def P(name, default):
    try:
        dbutils.widgets.text(name, default)  # noqa: F821
    except Exception:  # noqa: BLE001
        pass
    v = dbutils.widgets.get(name)  # noqa: F821
    return default if v is None or v == "" else v


W = {"tag": P("tag", "v1"),
     "stages": P("stages", "a,c,d,b"),
     "arms": P("arms", "T,S-Edi"),
     "clips": P("clips", ",".join(core.EDI_TEST)),
     "max_frames": P("max_frames", str(OLD_N_FRAMES)),
     "tol_boxes": P("tol_boxes", str(REPRO_TOL_BOXES)),
     "teacher_k8_row": P("teacher_k8_row", "false"),
     "fid_stride": P("fid_stride", "5"),
     "fid_bridges": P("fid_bridges", "true"),
     "consistency_frames": P("consistency_frames", "0,150,300,450,599"),
     "sam3_revision": P("sam3_revision", ""),
     "require_gpu": P("require_gpu", "A10"),
     "transformers_pin": P("transformers_pin", core.TRANSFORMERS_PIN),
     "expect_torch": P("expect_torch", ""),
     "b11_diag_tags": P("b11_diag_tags", "")}
TAG = W["tag"].strip()
if not TAG_RX.match(TAG):
    raise ValueError(f"tag {TAG!r}")
ENV = env_spec(W["transformers_pin"], W["expect_torch"], TAG, os.environ.get("DATABRICKS_RUNTIME_VERSION"))
with open(REQ_PATH, encoding="utf-8") as _f:          # what the %pip cell actually installed from
    _req_written = _f.read().splitlines()
if _req_written != ENV["requirements"]:
    raise RuntimeError(f"{REQ_PATH} {_req_written} != the expected requirements {ENV['requirements']}")
STAGES = parse_csv(W["stages"], {"a", "b", "c", "d"}, "stages")
ARMS = parse_csv(W["arms"], {"T", "S-Edi"}, "arms") + (["T-k8"] if parse_bool(W["teacher_k8_row"]) else [])
ARMS = [a for a in ARMS_B if a in ARMS]                               # fixed order: T, T-k8, S-Edi
CLIPS = parse_csv(W["clips"], set(core.EDI_TEST), "clips")
CLIPS = [c for c in core.EDI_TEST if c in CLIPS]
NF = int(W["max_frames"])
if not 1 <= NF <= OLD_N_FRAMES:
    raise ValueError("max_frames must be in 1..600")
SMOKE = NF < OLD_N_FRAMES or CLIPS != list(core.EDI_TEST)
if SMOKE and not TAG.startswith("smoke"):
    raise ValueError("a reduced run (max_frames < 600 or a clip subset) must use a tag starting with 'smoke'")
TOL = int(W["tol_boxes"])
FID_STRIDE = int(W["fid_stride"])
FID_BRIDGES = parse_bool(W["fid_bridges"])
CONS_FIS = [f for f in parse_ints(W["consistency_frames"], "consistency_frames") if 0 <= f < NF]
SAM3_REV = W["sam3_revision"].strip() or None
B11_TAGS = parse_b11_tags(W["b11_diag_tags"], TAG, ENV["env_diag"], SMOKE)   # BENCH_PROTOCOL b1.1 #2 inputs

R = tol_paths.load()
OUT = R.out("paper1.root", "evidence", "edinburgh_audit", TAG)
LOCAL = "/local_disk0/tol68/" + TAG
os.makedirs(LOCAL, exist_ok=True)
_vol_tp = R.path("admin.registry", None, "tol_paths.py")
if core.sha256_file(_vol_tp) != core.sha256_file(tol_paths.__file__):
    raise RuntimeError("the deployed tol_paths.py differs from the registry's copy: re-deploy it beside this notebook")

import torch                      # noqa: E402
import transformers               # noqa: E402
import motmetrics                 # noqa: E402
if transformers.__version__ != ENV["transformers_pin"]:
    raise RuntimeError(f"transformers {transformers.__version__} != {ENV['transformers_pin']} (transformers_pin)")
if ENV["expect_torch"] and torch.__version__.split("+")[0] != ENV["expect_torch"]:
    raise RuntimeError(f"torch {torch.__version__} != expect_torch {ENV['expect_torch']}")
ENV_DIAG_DESC = (f"transformers {transformers.__version__}, torch {torch.__version__}, "
                 f"DBR {ENV['databricks_runtime']}") if ENV["env_diag"] else None
if motmetrics.__version__ != EXPECT_MOTMETRICS:
    raise RuntimeError(f"motmetrics {motmetrics.__version__} != {EXPECT_MOTMETRICS} (the published scorer)")
GPU = core.gpu_info()
if W["require_gpu"] != "any" and GPU["class"] != W["require_gpu"]:
    raise RuntimeError(f"GPU class {GPU['class']} != {W['require_gpu']} (R0.2: the published runs were on an A10)")
DEVICE = torch.device("cuda")
HF_TOKEN = core.hf_token_from_dbutils(dbutils)  # noqa: F821  (never printed)
# tol68-2026-10-14.4: on DBR 16.4 the runtime wraps from_pretrained (dbruntime huggingface_patches) and the MODEL download
# of the gated facebook/sam3 went out unauthenticated (401) although token= was passed (the processor download worked).
# The hub falls back to this process-local variable (same secret, never printed or written to disk).
os.environ["HF_TOKEN"] = HF_TOKEN


def vol(rel):
    return OUT + "/" + rel


def loc(rel):
    p = LOCAL + "/" + rel
    os.makedirs(osp.dirname(p), exist_ok=True)
    return p


def publish_file(rel):
    return core.publish_no_overwrite(LOCAL + "/" + rel, vol(rel), dbutils)  # noqa: F821


def publish_json(rel, obj):
    core.write_json_atomic(loc(rel), obj)
    return publish_file(rel)


def publish_text(rel, text):
    core.write_text_atomic(loc(rel), text)
    return publish_file(rel)


def publish_dir(rel_dir):
    st = defaultdict(int)
    base = LOCAL + "/" + rel_dir
    for d, _ds, fs in os.walk(base):
        for f in sorted(fs):
            if f.startswith(".staged_ok") or ".tmp_" in f or f.endswith((".tmp", ".part")):
                continue                    # staging marker / leftovers of a crashed atomic write: never published
            rel = osp.relpath(osp.join(d, f), LOCAL).replace(os.sep, "/")
            st[publish_file(rel)] += 1
    return dict(st)


def is_done(name):
    return osp.exists(vol(".done_" + name))


def mark_done(name, info):
    return publish_json(".done_" + name, info)


def read_vol_json(rel):
    with open(vol(rel), encoding="utf-8") as f:
        return json.load(f)


_STAGED = {}


def stage_once(src, dst):
    if dst not in _STAGED:
        _STAGED[dst] = core.stage_file(src, dst, dbutils)  # noqa: F821
    return _STAGED[dst]


def stage_frames(clip):
    return core.stage_tree(R.path("datasets.edinburgh_pigs.decoded_frames", None, clip, "video1"),
                           f"{LOCAL}/inputs/frames/{clip}", dbutils)  # noqa: F821


def frame_path(clip, fi):
    return f"{LOCAL}/inputs/frames/{clip}/{fi + 1:07d}.jpg"


def load_gt(clip):
    p, sha = stage_once(R.path("datasets.edinburgh_pigs.annotated", None, *gt_rel_parts(clip)),
                        f"{LOCAL}/inputs/gt/{clip}_output.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f), sha


def load_prompt_text(clip):
    p, sha = stage_once(R.path("datasets.edinburgh_pigs.initial_prompts", None, clip, "video1",
                               f"BoundingBoxes_{clip}.txt"), f"{LOCAL}/inputs/prompts/BoundingBoxes_{clip}.txt")
    with open(p, encoding="utf-8") as f:
        return f.read(), sha


def student_ckpt():
    return stage_once(R.path("models.student_edinburgh"), f"{LOCAL}/inputs/models/best_backbone_student.pt")


def target_path(clip, fi):
    p, _ = stage_once(R.path("paper1.teacher_targets.edinburgh", None, clip, f"backbone_{fi:06d}.pt"),
                      f"{LOCAL}/inputs/targets/{clip}/backbone_{fi:06d}.pt")
    return p


def load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 -- our own catalog files only
        return torch.load(path, map_location="cpu", weights_only=False)


AUDIT_CONFIG = {
    "notebook": NB_VERSION, "bench_version": core.BENCH_VERSION, "protocol_doc": "BENCH_PROTOCOL.md b1.0 §13 J3(c)",
    "code": {"tol_bench_core": core.code_sha256(), "tol_paths": core.source_shas(tol_paths.__file__)},
    "settings": {k: W[k] for k in ("tag", "arms", "clips", "max_frames", "tol_boxes", "teacher_k8_row", "fid_stride",
                                   "fid_bridges", "consistency_frames", "sam3_revision", "require_gpu",
                                   "b11_diag_tags")},
    "constants": {"OLD_N_FRAMES": OLD_N_FRAMES, "OLD_IOU": OLD_IOU, "OLD_STUDENT_PRUNE_K": OLD_STUDENT_PRUNE_K,
                  "OLD_PRUNE_EVERY": OLD_PRUNE_EVERY, "OLD_TEACHER_GC_EVERY": OLD_TEACHER_GC_EVERY,
                  "REPRO_TOL_BOXES": REPRO_TOL_BOXES, "PUBLISHED": PUBLISHED, "CKPT_EPOCH0": CKPT_EPOCH0,
                  "PAPER_CELL21": PAPER_CELL21, "HIST_EP75": HIST_EP75},
    "registry": R.provenance(list(REGISTRY_KEYS)), "gpu": GPU,
    "versions_pinned": {k: v for k, v in core.versions().items()          # the full list goes to runs/<utc>.json
                        if k in ("python", "torch", "torchvision", "transformers", "timm", "numpy", "Pillow", "cuda")},
    "motmetrics": motmetrics.__version__,
    "env": {k: ENV[k] for k in ("transformers_pin", "expect_torch", "env_diag", "databricks_runtime", "requirements")},
    "torch_numerics": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                       "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                       "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                       "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)},
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "extra": {"stages_requested": STAGES},
}
if osp.exists(vol("AUDIT_CONFIG.json")):
    _d = core.config_diff(read_vol_json("AUDIT_CONFIG.json"), json.loads(json.dumps(AUDIT_CONFIG, default=str)))
    if _d:
        raise RuntimeError("AUDIT_CONFIG mismatch with the published one -- use a new tag: "
                           + "; ".join(f"{p}: {a!r} -> {b!r}" for p, a, b in _d[:12]))
    log(f"resuming tag {TAG} (identical AUDIT_CONFIG)")
else:
    publish_json("AUDIT_CONFIG.json", json.loads(json.dumps(AUDIT_CONFIG, default=str)))
RUN = {"started_utc": AUDIT_CONFIG["created_utc"], "stages": {}, "errors": [], "versions": core.versions(),
       "stages_requested": STAGES}
log(f"Tol_68 tag {TAG}: stages {STAGES}, arms {ARMS}, clips {CLIPS}, max_frames {NF}, GPU {GPU}, "
    f"env {'DIAGNOSTIC: ' + ENV_DIAG_DESC if ENV_DIAG_DESC else 'protocol (b1.0)'}")

# COMMAND ----------

# ---- (a) seeds == GT frame 0 -----------------------------------------------------------------------------------------
def stage_a():
    pr_root = R.path("datasets.edinburgh_pigs.initial_prompts")
    an_root = R.path("datasets.edinburgh_pigs.annotated")
    df_root = R.path("datasets.edinburgh_pigs.decoded_frames")
    prompt_clips = sorted(d for d in os.listdir(pr_root) if CLIP_RX.match(d))
    frame_clips = sorted(d for d in os.listdir(df_root) if CLIP_RX.match(d))
    gt_clips = sorted(f"{d}_{s}" for d in os.listdir(an_root) if re.match(r"^\d{4}_\d{2}_\d{2}$", d)
                      for s in os.listdir(f"{an_root}/{d}") if osp.isfile(f"{an_root}/{d}/{s}/output.json"))
    recs = []
    for clip in prompt_clips:
        vdir = R.path("datasets.edinburgh_pigs.initial_prompts", None, clip, "video1")
        files = sorted(os.listdir(vdir)) if osp.isdir(vdir) else []
        want = f"BoundingBoxes_{clip}.txt"
        if want not in files:
            recs.append({"clip": clip, "ok": False, "issues": [f"{want} absent"], "files": files, "n_prompts": 0,
                         "n_gt_objects": 0, "n_exact": 0, "min_iou": None, "strays": files})
            continue
        txt, psha = load_prompt_text(clip)
        gt, gsha = load_gt(clip)
        r = seed_check_clip(clip, txt, gt)
        r["strays"] = [f for f in files if f != want]
        r["sha256"] = {"prompt": psha, "gt": gsha}
        recs.append(r)
    frames = {}
    for clip in core.EDI_TEST:
        fd = R.path("datasets.edinburgh_pigs.decoded_frames", None, clip, "video1")
        names = sorted(f for f in os.listdir(fd) if f.lower().endswith(".jpg"))
        gt, _ = load_gt(clip)
        frames[clip] = {"n_jpg": len(names),
                        "contiguous_0000001": names == [f"{k + 1:07d}.jpg" for k in range(len(names))],
                        "gt_max_frameNumber": max(int(f["frameNumber"]) for o in gt["objects"] for f in o["frames"]),
                        "gt_n_objects": len(gt["objects"]),
                        "gt_keyframes_per_object": sorted(len(o["frames"]) for o in gt["objects"]),
                        "gt_stepSize": gt.get("stepSize")}
    res = {"n_clips": len(recs), "n_clips_ok": int(sum(bool(r["ok"]) for r in recs)),
           "n_prompts": int(sum(r["n_prompts"] for r in recs)), "n_exact": int(sum(r["n_exact"] for r in recs)),
           "all_ok": all(r["ok"] for r in recs),
           "strays": {r["clip"]: r["strays"] for r in recs if r.get("strays")},
           "clips_with_frames_but_no_prompt_file": sorted(set(frame_clips) - set(prompt_clips)),
           "clips_with_gt_but_no_prompt_file": sorted(set(gt_clips) - set(prompt_clips)),
           "test_clip_frames": frames, "clips": recs,
           "note": "R0.1 (2026-10-08): decoded frame k = raw color.mp4 frame 3k (pixel MSE), 600 labelled frames per clip"}
    publish_json("a_seeds/seed_vs_gt_frame0.json", res)
    publish_text("a_seeds/seed_vs_gt_frame0.txt", seed_table_text(recs))
    return res


def guarded(stage, fn):
    """Run one stage; an exception is recorded (the notebook raises at the very end) so later stages still run."""
    try:
        fn()
        RUN["stages"][stage] = "done"
    except Exception as e:  # noqa: BLE001
        RUN["stages"][stage] = "error"
        RUN["errors"].append(f"({stage}) {type(e).__name__}: {e}")
        log(traceback.format_exc())


def _run_a():
    if is_done("a"):
        log("(a) already published: skipped")
        return
    t0 = time.time()
    A = stage_a()
    mark_done("a", {"n_exact": A["n_exact"], "n_prompts": A["n_prompts"], "all_ok": A["all_ok"]})
    log(f"(a) seeds: {A['n_exact']}/{A['n_prompts']} exact, {A['n_clips_ok']}/{A['n_clips']} clips OK, "
        f"strays {A['strays']} ({time.time() - t0:.0f} s)")


if "a" in STAGES:
    guarded("a", _run_a)

# COMMAND ----------

# ---- shared GPU pieces -------------------------------------------------------------------------------------------------
import torch.nn as nn            # noqa: E402

core._require_torch(need_transformers=True)


class TorchBackend:
    def no_grad(self):
        return torch.no_grad()

    def binarize(self, ml):
        return (torch.sigmoid(ml) > 0.5).cpu().numpy()                 # cell 26 / cell 17 verbatim threshold

    def finite(self, x):
        return bool(torch.isfinite(x).all())

    def reset_peak(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def mem_gb(self):
        return torch.cuda.memory_allocated() / 1024 ** 3

    def peak_gb(self):
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 ** 3

    def gc_empty(self):
        gc.collect()
        torch.cuda.empty_cache()


BE = TorchBackend()


class OldDummyBackbone(nn.Module):
    """VERBATIM Fixed cell 26 DummyBackbone (functionally = core.DummyBackbone)."""
    def __init__(self):
        super().__init__()
        self.student_features = None  # Set externally before each forward

    def forward(self, *args, **kwargs):
        feat = self.student_features  # [B, 1024, 72, 72]
        B, C, H, W = feat.shape
        fake_lhs = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 5184, 1024]
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=fake_lhs)


class BackboneCapture:
    """Cell 11 BackboneFeatureHook semantics ([1,5184,1024] -> [1,1024,72,72] fp16 on the CPU), on chosen frames only."""
    def __init__(self, module):
        self.fi, self.out = None, {}
        self.handle = module.register_forward_hook(self._hook)

    def want(self, fi):
        self.fi = fi

    def _hook(self, module, inp, output):
        if self.fi is None:
            return
        if hasattr(output, "last_hidden_state"):
            lhs = output.last_hidden_state
            B, N, C = lhs.shape
            H = W_ = int(N ** 0.5)
            self.out[self.fi] = lhs.permute(0, 2, 1).reshape(B, C, H, W_).detach().cpu().half()
        elif isinstance(output, torch.Tensor):
            self.out[self.fi] = output.detach().cpu().half()

    def remove(self):
        self.handle.remove()


def verify_published_ckpt(path):
    ck = torch_load(path)
    hdr = {"epoch0": int(ck.get("epoch", -1)), "keys": sorted(ck.keys()),
           "n_ema": len(ck.get("ema_shadow") or {}), "n_model_state_dict": len(ck.get("model_state_dict") or {}),
           "metrics": {k: float(v) for k, v in (ck.get("metrics") or {}).items() if isinstance(v, (int, float))}}
    del ck
    if hdr["epoch0"] != CKPT_EPOCH0 or abs(hdr["metrics"].get("ema_val_loss", -1) - CKPT_EMA_VAL_LOSS) > 1e-6:
        raise RuntimeError(f"models.student_edinburgh is not the published EMA epoch-75 checkpoint: {hdr}")
    return hdr


def load_student_fp32():
    p, sha = student_ckpt()
    hdr = verify_published_ckpt(p)
    s, meta = core.load_student(p, device=DEVICE, dtype=torch.float32, check_gn=True)
    meta.update({"sha256": sha, "header": hdr})
    return s, meta


def load_sam3_old():
    """The old load (cells 5/8): from_pretrained(torch_dtype=bf16).to(device), eval; NO attn_implementation."""
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    kw = {"token": HF_TOKEN}
    if SAM3_REV:
        kw["revision"] = SAM3_REV
    proc = Sam3TrackerVideoProcessor.from_pretrained(HF_SAM3, **kw)
    model = Sam3TrackerVideoModel.from_pretrained(HF_SAM3, torch_dtype=torch.bfloat16, **kw).to(DEVICE)
    model.eval()
    commit, commit_src = core.hf_commit(model, HF_SAM3, SAM3_REV, HF_TOKEN)   # 5.2.0: config._commit_hash is None
    info = {"hf_id": HF_SAM3, "revision_requested": SAM3_REV,
            "commit_hash": commit, "commit_source": commit_src,
            "attn_resolved": core._attn_impls(model.config),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "processor": {k: v for k, v in core._config_scalars(getattr(proc, "image_processor", proc)).items()
                          if not k.startswith("_")}}
    return proc, model, info


# COMMAND ----------

# ---- (c) GroupNorm train/eval: drop_path_rate=0.0 vs the timm default ---------------------------------------------------
class _TimmNoDropPathArg:
    """Builds the EXPORT class (the Fixed cell 13 call, without [bench-C1] drop_path_rate=0.0): strips that kwarg."""
    def __init__(self, real):
        self._real = real

    def create_model(self, *a, **k):
        k.pop("drop_path_rate", None)
        return self._real.create_model(*a, **k)

    def __getattr__(self, n):
        return getattr(self._real, n)


def build_student(default_drop_path):
    real = core.timm
    if default_drop_path:
        core.timm = _TimmNoDropPathArg(real)
    try:
        return core.StudentEncoderTinyViT(out_channels=1024, pretrained=False)
    finally:
        core.timm = real


def load_weights_like_core(student, ck):
    sd = student.state_dict()
    msd, ema = dict(ck.get("model_state_dict") or {}), dict(ck.get("ema_shadow") or {})
    have = set(msd) | set(ema)
    missing, extra = [k for k in sd if k not in have], [k for k in have if k not in sd]
    if missing or extra:
        raise RuntimeError(f"checkpoint/architecture mismatch: missing {missing[:6]} unused {extra[:6]}")
    sd.update({k: v.float() for k, v in msd.items()})
    sd.update({k: v.float() for k, v in ema.items()})
    student.load_state_dict(sd, strict=True)
    return student


def stage_c():
    import timm
    p, sha = student_ckpt()
    hdr = verify_published_ckpt(p)
    ck = torch_load(p)
    tf = core.make_student_transform()
    for clip in (core.EDI_TRAIN[0], core.EDI_TEST[0]):
        stage_once(R.path("datasets.edinburgh_pigs.decoded_frames", None, clip, "video1", "0000001.jpg"),
                   f"{LOCAL}/inputs/frames_single/{clip}/0000001.jpg")
    g = torch.Generator(device="cpu").manual_seed(0)
    inputs = {"randn_seed0": torch.randn(1, 3, 1024, 1024, generator=g),
              "train_clip_frame0_cell21_input": tf(load_rgb(
                  f"{LOCAL}/inputs/frames_single/{core.EDI_TRAIN[0]}/0000001.jpg")).unsqueeze(0),
              "test_clip_frame0": tf(load_rgb(f"{LOCAL}/inputs/frames_single/{core.EDI_TEST[0]}/0000001.jpg")).unsqueeze(0)}
    res = {"variants": {}, "eval_default_vs_eval_dp0": {}, "timm": timm.__version__, "torch": torch.__version__,
           "checkpoint": {"sha256": sha, "header": hdr}, "seeds": list(GN_SEEDS), "dtype": "float32"}
    evals = {}
    for vname, dflt in (("drop_path_0", False), ("timm_default", True)):
        s = load_weights_like_core(build_student(dflt), ck).to(DEVICE).float()
        dps = [float(m.drop_prob) for m in s.modules() if type(m).__name__ == "DropPath"]
        vr = {"drop_path_probs": dps, "n_droppath_modules": len(dps),
              "n_batchnorm": sum(1 for m in s.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))),
              "n_params": int(sum(q.numel() for q in s.parameters())), "inputs": {}}
        for iname, x in inputs.items():
            x = x.to(DEVICE)
            s.eval()
            with torch.no_grad():
                ev = s(x)
            diffs = []
            for seed in GN_SEEDS:
                torch.manual_seed(seed)
                s.train()
                with torch.no_grad():
                    tr = s(x)
                d = (tr - ev).abs()
                diffs.append({"seed": seed, "max": float(d.max()), "mean": float(d.mean())})
                del tr, d
            s.eval()
            vr["inputs"][iname] = {"train_vs_eval": diffs, "max_over_seeds": max(d["max"] for d in diffs)}
            evals[(vname, iname)] = ev.detach().cpu()
            del ev
        res["variants"][vname] = vr
        del s
        BE.gc_empty()
    for iname in inputs:
        res["eval_default_vs_eval_dp0"][iname] = float(
            (evals[("timm_default", iname)] - evals[("drop_path_0", iname)]).abs().max())
    res["summary"] = summarize_gn(res)
    del ck
    return res


def _run_c():
    if is_done("c"):
        log("(c) already published: skipped")
        return
    t0 = time.time()
    c_res = stage_c()
    publish_json("c_groupnorm/groupnorm_train_eval.json", c_res)
    mark_done("c", c_res["summary"])
    log(f"(c) GroupNorm: {c_res['summary']} ({time.time() - t0:.0f} s)")


if "c" in STAGES:
    guarded("c", _run_c)
    BE.gc_empty()

# COMMAND ----------

# ---- (d) feature fidelity of the published student on frames of the 4 TEST clips -------------------------------------
def stage_d():
    plan = fidelity_plan(CLIPS, FID_STRIDE, NF, FID_BRIDGES)
    t0 = time.time()
    for k, (_g, clip, fi) in enumerate(plan):
        target_path(clip, fi)
        if (k + 1) % 50 == 0:
            log(f"  (d) staged {k + 1}/{len(plan)} teacher targets ({time.time() - t0:.0f} s)")
    for clip in sorted({c for _g, c, _f in plan}):
        stage_frames(clip)
    s32, smeta = load_student_fp32()
    s16 = bf16_copy(s32, torch.bfloat16)
    tf = core.make_student_transform()
    rows = []
    for k, (grp, clip, fi) in enumerate(plan):
        t = torch_load(target_path(clip, fi)).float().numpy()
        x = tf(load_rgb(frame_path(clip, fi))).unsqueeze(0)
        with torch.no_grad():
            f32 = s32(x.to(DEVICE)).float().cpu().numpy()
            f16 = s16(x.to(DEVICE, torch.bfloat16)).float().cpu().numpy()
        if not (np.isfinite(f32).all() and np.isfinite(f16).all()):
            raise core.BenchAssertError(f"(d) non-finite student features {clip} {fi}")
        rows.append({"group": grp, "clip": clip, "fi": fi, "fp32": fidelity_metrics(f32, t),
                     "bf16": fidelity_metrics(f16, t), "cos_bf16_vs_fp32": cosine_global(f16, f32)})
        if (k + 1) % 100 == 0:
            log(f"  (d) {k + 1}/{len(plan)} frames")
    summ = summarize_fidelity(rows)
    res = {"definitions": fidelity_metrics.__doc__, "plan": {"stride": FID_STRIDE, "n_frames": len(rows),
                                                              "bridges": FID_BRIDGES, "test_clips": CLIPS},
           "student": smeta, "summary": summ, "bridges": bridge_checks(summ),
           "note": "teacher targets = the stored distillation targets (paper1.teacher_targets.edinburgh, copied from "
                   "the 2026-02 extraction); the paper's 0.808 used training frames, the honest held-out numbers are "
                   "the 'test' group"}
    cols = ["group", "clip", "fi"] + [f"fp32_{k}" for k in FID_KEYS] + [f"bf16_{k}" for k in FID_KEYS] + \
        ["cos_bf16_vs_fp32"]
    lines = [",".join(cols)]
    for r in rows:
        vals = [r["group"], r["clip"], str(r["fi"])] + [repr(r["fp32"][k]) for k in FID_KEYS] + \
            [repr(r["bf16"][k]) for k in FID_KEYS] + [repr(r["cos_bf16_vs_fp32"])]
        lines.append(",".join(vals))
    del s32, s16
    BE.gc_empty()
    return res, "\n".join(lines) + "\n"


def _run_d():
    if is_done("d"):
        log("(d) already published: skipped")
        return
    t0 = time.time()
    d_res, d_csv = stage_d()
    publish_text("d_fidelity/per_frame.csv", d_csv)
    publish_json("d_fidelity/fidelity.json", d_res)
    te = d_res["summary"]["groups"]["test"]["fp32"]
    mark_done("d", {"test_cos_patch": te["cos_patch"]["mean"], "test_cos_global": te["cos_global"]["mean"],
                    "test_scale_ratio": te["scale_ratio"]["mean"], "bridges_ok": d_res["bridges"]["all_within_tol"]})
    log(f"(d) test fp32: cos_patch {te['cos_patch']['mean']:.4f} cos_global {te['cos_global']['mean']:.4f} "
        f"scale {te['scale_ratio']['mean']:.4f}; bridges {d_res['bridges']} ({time.time() - t0:.0f} s)")


if "d" in STAGES:
    guarded("d", _run_d)
    BE.gc_empty()

# COMMAND ----------

# ---- (b) reproduce-old: run ------------------------------------------------------------------------------------------
def b_rel(arm, *parts):
    return "/".join(("b_reproduce_old", ARM_DIRS[arm]) + parts)


def run_arm_clip(arm, clip, proc, model, student=None, tf=None, dummy=None):
    stage_frames(clip)
    txt, _ = load_prompt_text(clip)
    ordered = parse_prompt_txt_ordered(txt)
    capture = None
    if arm == "T" and CONS_FIS:
        capture = BackboneCapture(model.vision_encoder.backbone)
    kw = dict(device=DEVICE, dtype=torch.bfloat16, max_frames=NF, label=f"{arm} {clip}")
    if arm == "T":
        kw.update(prune_k=None, gc_every=OLD_TEACHER_GC_EVERY, capture=capture, capture_fis=set(CONS_FIS))
    elif arm == "T-k8":
        kw.update(prune_k=OLD_STUDENT_PRUNE_K)
    else:
        kw.update(prune_k=OLD_STUDENT_PRUNE_K,
                  student_fn=lambda pil: student(tf(pil).unsqueeze(0).to(DEVICE)), dummy=dummy)
    try:
        annots, st = run_old_protocol(core.EdinburghFrames(f"{LOCAL}/inputs/frames/{clip}"), ordered, proc, model,
                                      BE, **kw)
    finally:
        if capture is not None:
            capture.remove()
    if st["obj_num"] != len(ordered) or not st["obj_ids_intact"]:
        raise core.BenchAssertError(f"{arm} {clip}: session objects {st['obj_num']} != {len(ordered)} prompts")
    write_old_annotations(f"{LOCAL}/{b_rel(arm, clip)}", annots)
    core.write_text_atomic(loc(b_rel(arm, "mot", f"{clip}.txt")), "\n".join(old_annots_to_mot(annots)) + "\n")
    pub_peak = None
    try:
        src = read_json_local(f"{LOCAL}/inputs/evidence/{PUBLISHED_FILES['T' if arm != 'S-Edi' else 'S-Edi']}")
        pub_peak = src[clip]["summary"]["performance_metrics"]["gpu_memory_peak_gb"]
    except Exception:  # noqa: BLE001
        pass
    st["published_gpu_memory_peak_gb"] = pub_peak
    core.write_json_atomic(loc(b_rel(arm, "telemetry", f"{clip}.json")), st)
    rels = [b_rel(arm, "mot", f"{clip}.txt"), b_rel(arm, "telemetry", f"{clip}.json")]
    if capture is not None:
        cons = {}
        for fi, feat in sorted(capture.out.items()):
            stored = torch_load(target_path(clip, fi))
            cons[str(fi)] = consistency_stats(feat.float().numpy(), stored.float().numpy())
        core.write_json_atomic(loc(b_rel(arm, "target_consistency", f"{clip}.json")), cons)
        rels.append(b_rel(arm, "target_consistency", f"{clip}.json"))
    publish_dir(b_rel(arm, clip))
    for r in rels:
        publish_file(r)
    mark_done(f"b_{ARM_DIRS[arm]}_{clip}", {"frames": st["frames"], "empty_masks": st["empty_masks"]})
    log(f"(b) {arm} {clip}: {st['frames']} frames, {st['fps']} fps, peak {st['peak_gb']} GB "
        f"(published {pub_peak}), empty masks {st['empty_masks']}")


def read_json_local(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stage_b_inputs():
    ev = {}
    for key, fname in PUBLISHED_FILES.items():
        p, sha = stage_once(R.path("paper1.evidence.edinburgh_metrics", None, fname),
                            f"{LOCAL}/inputs/evidence/{fname}")
        ev[key] = (read_json_local(p), sha)
    pub = {"T": published_per_clip(ev["T"][0], core.EDI_TEST), "S-Edi": published_per_clip(ev["S-Edi"][0], core.EDI_TEST)}
    check_published("T", pub["T"], ev["final"][0])
    check_published("S-Edi", pub["S-Edi"], ev["final"][0])
    imp = published_per_clip(ev["improved"][0], core.EDI_TEST)
    surv = {"T": {}, "improved": {}}
    for key, dname in SURVIVING_DIRS.items():
        for clip in CLIPS:
            dst = f"{LOCAL}/inputs/surviving/{dname}/{clip}"
            core.stage_tree(R.path("paper1.evidence.edinburgh_mask_coordinates", None, dname, clip), dst,
                            dbutils)  # noqa: F821
            surv[key][clip] = load_old_annotations(dst)
    return pub, imp, surv, {k: v[1] for k, v in ev.items()}


B_CTX = {}


def _run_b():
    t0b = time.time()
    pub, imp, surv, ev_sha = stage_b_inputs()
    B_CTX.update(pub=pub, imp=imp, surv=surv, ev_sha=ev_sha)
    todo = [(a, c) for a in ARMS for c in CLIPS if not is_done(f"b_{ARM_DIRS[a]}_{c}")]
    log(f"(b) to run: {todo}")
    if todo:
        proc, model, sam3_info = load_sam3_old()
        log(f"(b) SAM 3 loaded: commit {sam3_info['commit_hash']}, attention {sam3_info['attn_resolved']}")
        if not osp.exists(vol("b_reproduce_old/sam3_load.json")):
            publish_json("b_reproduce_old/sam3_load.json", sam3_info)
        for arm in ("T", "T-k8"):
            for clip in CLIPS:
                if (arm, clip) in todo:
                    try:
                        run_arm_clip(arm, clip, proc, model)
                    except Exception as e:  # noqa: BLE001
                        RUN["errors"].append(f"(b) {arm} {clip}: {type(e).__name__}: {e}")
                        log(traceback.format_exc())
                    BE.gc_empty()
        if any(a == "S-Edi" for a, _ in todo):
            student, smeta = load_student_fp32()
            if not osp.exists(vol("b_reproduce_old/student_load.json")):
                publish_json("b_reproduce_old/student_load.json", smeta)
            tf = core.make_student_transform()
            dummy = OldDummyBackbone().to(DEVICE)              # cell 26: swap, park the teacher backbone on the CPU
            orig_backbone = model.vision_encoder.backbone
            model.vision_encoder.backbone = dummy
            orig_backbone.cpu()
            BE.gc_empty()
            try:
                for clip in CLIPS:
                    if ("S-Edi", clip) in todo:
                        try:
                            run_arm_clip("S-Edi", clip, proc, model, student=student, tf=tf, dummy=dummy)
                        except Exception as e:  # noqa: BLE001
                            RUN["errors"].append(f"(b) S-Edi {clip}: {type(e).__name__}: {e}")
                            log(traceback.format_exc())
                        BE.gc_empty()
            finally:
                model.vision_encoder.backbone = orig_backbone.to(DEVICE)
            del student
        del model, proc
        BE.gc_empty()
    RUN["stages"]["b_run_detail"] = {"todo": [list(x) for x in todo], "seconds": round(time.time() - t0b, 1)}


if "b" in STAGES:
    guarded("b_run", _run_b)

# COMMAND ----------

# ---- (b) reproduce-old: score + verdict ---------------------------------------------------------------------------------
def _score_b():
    if not B_CTX:
        raise RuntimeError("(b) inputs unavailable: b_run failed before staging them")
    missing = [(a, c) for a in ARMS for c in CLIPS if not is_done(f"b_{ARM_DIRS[a]}_{c}")]
    if missing:
        log(f"(b) not scored: missing runs {missing}")
        RUN["stages"]["b_score_detail"] = f"missing {missing}"
        return
    if is_done("b"):
        log("(b) verdict already published: skipped")
        return
    tr_by_arm = {}
    for arm in ARMS:
        for clip in CLIPS:
            d = f"{LOCAL}/b_fetched/{ARM_DIRS[arm]}/{clip}"
            core.stage_tree(vol(b_rel(arm, clip)), d, dbutils)  # noqa: F821  (the PUBLISHED files are scored)
            tr_by_arm.setdefault(arm, {})[clip] = load_old_annotations(d)
    gt = {c: load_gt(c)[0] for c in CLIPS}
    scores, verdict, direct = score_stage_b(CLIPS, gt, tr_by_arm, B_CTX["surv"], B_CTX["pub"], B_CTX["imp"], TOL,
                                            n_frames=NF, env_diag=ENV_DIAG_DESC)
    if B11_TAGS:                     # BENCH_PROTOCOL b1.1 #2: read the named diagnostics' PUBLISHED stage-(b) verdicts
        recs = []
        for dt in B11_TAGS:
            base = R.path("paper1.root", None, "evidence", "edinburgh_audit", dt)
            pa, pv = base + "/AUDIT_CONFIG.json", base + "/b_reproduce_old/verdict.json"
            if not (osp.exists(pa) and osp.exists(pv)):
                raise RuntimeError(f"b1.1 #2: diagnostic tag {dt} has no published AUDIT_CONFIG / stage-(b) verdict "
                                   "(run it first; nothing of (b) scoring is published until it exists)")
            recs.append(b11_diag_record(dt, read_json_local(pa), read_json_local(pv)))
            recs[-1]["sha256"] = {"audit_config": core.sha256_file(pa), "verdict": core.sha256_file(pv)}
        if verdict.get("complete_test_set") and not ENV_DIAG_DESC:
            verdict = apply_b11(verdict, recs, gpu_class=GPU["class"])
            log(f"(b) BENCH_PROTOCOL b1.1 #2: {verdict['b11']['status']} (b1.0 {verdict['b11']['b10_status']}; cause "
                f"{verdict['b11'].get('cause')})")
    verdict["evidence_sha256"] = B_CTX["ev_sha"]
    cons = {}
    if "T" in ARMS and CONS_FIS:
        for clip in CLIPS:
            cons[clip] = read_vol_json(b_rel("T", "target_consistency", f"{clip}.json"))
    _cos = [v.get("cos_global", 0) for c in cons.values() for v in c.values()]
    verdict["teacher_target_consistency"] = {
        "frames": cons, "n_frames": len(_cos),
        "all_cos_ge_0.999": (min(_cos) >= 0.999) if _cos else None,
        "note": "teacher backbone captured during the T run vs the stored distillation targets of the same frames"}
    publish_json("b_reproduce_old/scores.json", scores)
    publish_json("b_reproduce_old/teacher_vs_surviving.json", direct)
    publish_json("b_reproduce_old/verdict.json", verdict)
    mark_done("b", verdict["overall"])
    log(f"(b) VERDICT {verdict['overall']['status']}: {verdict['overall']['message']}")
    for arm, v in verdict["arms"].items():
        for r in v.get("rows", []):
            log(f"    {arm} {r['clip']}: FP {r.get('fp_new')} vs {r.get('fp_pub')}, FN {r.get('fn_new')} vs "
                f"{r.get('fn_pub')}, IDSW {r.get('idsw_new')}, MOTA {r.get('mota_new')} vs {r.get('mota_pub')} "
                f"-> {r['within_tol']}")


if "b" in STAGES:
    guarded("b_score", _score_b)

# COMMAND ----------

# ---- summary (only when every stage is published) + run log ------------------------------------------------------------
def _maybe(rel):
    return read_vol_json(rel) if osp.exists(vol(rel)) else None


if all(is_done(s) for s in ("a", "b", "c", "d")) and not SMOKE and not ENV["env_diag"]:
    S_JSON, S_MD = build_summary(_maybe("a_seeds/seed_vs_gt_frame0.json"), _maybe("b_reproduce_old/verdict.json"),
                                 _maybe("c_groupnorm/groupnorm_train_eval.json"), _maybe("d_fidelity/fidelity.json"))
    publish_json("AUDIT_SUMMARY.json", S_JSON)
    publish_text("AUDIT_SUMMARY.md", S_MD)
    print(S_MD)
RUN["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
publish_json(f"runs/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json", RUN)
log(f"outputs: {OUT}")
if RUN["errors"]:
    raise RuntimeError("Tol_68 finished with errors (outputs of the completed parts are published): "
                       + " | ".join(RUN["errors"]))
