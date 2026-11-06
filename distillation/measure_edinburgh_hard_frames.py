# Databricks notebook source
# MAGIC %md
# MAGIC # Find where the Edinburgh student departs from SAM 3
# MAGIC
# MAGIC Computes SAM 3 targets for the three clips the benchmark does not use, then measures student against SAM 3 per frame on
# MAGIC 8 non-test clips (per-position cosine, including inside pig cells). `select_targets_edinburgh` reads the per-frame table.
# MAGIC Runs on one A10. The four test clips are refused by every path helper.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm==1.0.30" "scipy" "rfdetr==1.10.1" "pycocotools" "peft"
# MAGIC %pip uninstall -y opencv-python opencv-contrib-python
# MAGIC %pip install --no-deps --force-reinstall "opencv-python-headless==4.11.0.86"

# COMMAND ----------

try:
    dbutils.library.restartPython()  # noqa: F821
except NameError:
    pass

# COMMAND ----------

import collections
import csv
import gc
import gzip
import hashlib
import io
import json
import math
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
import uuid
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before this process touches CUDA

import numpy as np

NB_NAME = "measure_edinburgh_hard_frames"
NB_VERSION = "measure_edinburgh_hard_frames-2026-10-29.1"
TARGETS_CONFIG_VERSION = "measure_edinburgh_hard_frames-targets-v1"
# the %pip cell above (= track_bench's line with timm pinned to the version every bench run resolved, 1.0.30)
PIP_SPEC = ('"numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm==1.0.30" "scipy" '
            '"rfdetr==1.10.1" "pycocotools" "peft"; opencv-python-headless==4.11.0.86 (--no-deps)')
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None
if not IS_NOTEBOOK and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _code_dir():
    env = os.environ.get("FARMSAM_CODE_DIR")
    if env:
        return env
    if not IS_NOTEBOOK:
        return osp.dirname(osp.abspath(__file__))
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    return "/Workspace" + osp.dirname(nbp)


def _import_exact(name, dirs):
    """Import `name` from the FIRST of dirs holding <name>.py and assert that this file is what runs."""
    for d in dirs:
        if not d or not osp.isfile(osp.join(d, name + ".py")):
            continue
        while d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
        sys.modules.pop(name, None)
        mod = __import__(name)
        if osp.normcase(osp.realpath(osp.dirname(osp.abspath(mod.__file__)))) != osp.normcase(osp.realpath(d)):
            raise RuntimeError(f"{name} resolved to {mod.__file__}, not {d}")
        return mod
    raise RuntimeError(f"{name}.py not found in {[d for d in dirs if d]}")


CODE_DIR = _code_dir()
core = _import_exact("bench_core", [CODE_DIR])
dc = _import_exact("distill_core", [CODE_DIR])
farm_paths = _import_exact("farm_paths", [CODE_DIR, os.environ.get("FARMSAM_PATHS_DIR")])

HF_REV_FILE = "hf_revisions.json"
HF_SAM3 = "facebook/sam3"
SAM3_REV = "3c879f39826c281e95690f02c7821c4de09afae7"
CKPT_SHA256 = "9686b4c0f60e02251d8c2febee2029c1e7ebe74e85fbefda7e0c5daffc2075f2"   # S-Edi EMA epoch 75

# ---- clips ---------------------------------------------------------------------------------------------------------------
EDI_TRAIN = ("2019_11_05_000002", "2019_11_11_000028", "2019_11_11_000036", "2019_11_22_000010")
EDI_VAL = ("2019_11_28_000113",)
UNUSED = ("2019_11_05_000009", "2019_11_11_000016", "2019_11_15_000033")
TRACK_CLIPS = EDI_TRAIN + EDI_VAL
CLIPS = TRACK_CLIPS + UNUSED
ROLE = dict([(c, "train") for c in EDI_TRAIN] + [(c, "val") for c in EDI_VAL] + [(c, "unused") for c in UNUSED])
# GT trust (INVENTORY.md 1.2): 000016's output.json does not describe its video; 000009 / 000033 look plausible, unverified
GT_STATUS = dict([(c, "trusted") for c in TRACK_CLIPS] + [("2019_11_05_000009", "unverified"),
                                                          ("2019_11_11_000016", "wrong"),
                                                          ("2019_11_15_000033", "unverified")])
GT_SHA256_PREFIX = {"2019_11_05_000002": "778e32b4", "2019_11_11_000028": "e26e1f7d", "2019_11_11_000036": "a0aab06b",
                    "2019_11_22_000010": "625bf6b6", "2019_11_28_000113": "c7957d2d", "2019_11_05_000009": "fec70ecb",
                    "2019_11_15_000033": "008e664c"}
# J4 (split 'all', A10, core 5e705fff) T seeded run MOT files of the 5 non-test clips (INVENTORY.md 5; local copies)
J4_T_MOT_SHA256 = {"2019_11_05_000002": "8c13771635dc448022916a8d9a9011860ce73011c1580cf9e098845c0331363f",
                   "2019_11_11_000028": "a0d76e598c2604e35408f382b4d3a850db63692df40343c830f32d65c0cefaa6",
                   "2019_11_11_000036": "80864f2c035bb30ab739118509c207372c9b9a79fb37d4ed6912554fe39d5a52",
                   "2019_11_22_000010": "2884747045be356d191a182c789e8add1ff4c824ef18f861ac82af82d188524e",
                   "2019_11_28_000113": "f849e31083c6e299e57d19f33b452b212c01c681c1a3c5d26cb586db0a2f1d3a"}
# test clips: refused by guard_path; their names are never written to an output (only a count and a digest)
_TEST_IDS = ("2019_12_02_000005", "2019_12_02_000208", "2019_12_10_000060", "2019_12_10_000078")
TEST_IDS_DIGEST = hashlib.sha256(",".join(sorted(_TEST_IDS)).encode()).hexdigest()
CLIP_RX = re.compile(r"^2019_(\d\d)_(\d\d)_(\d{6})$")
N_FRAMES = 600
FRAME_W, FRAME_H = 1280, 720
GRID = 72
CONTROL_FI = (0, 150, 300, 450, 599)          # 5 frames x 4 training clips = the 20-target recompute control
# [FID-E] fp32 means (the fidelity check STORED, fidelity.json sha256 756e2d92...): the val row (stride 5) and the training bridge
FIDE = {"val/2019_11_28_000113": {"cos_patch": 0.7508357814739799, "cos_global": 0.6921778052278609,
                                  "scale_ratio": 0.9557935679024919, "mse": 1.1941566283638172},
        "train20/2019_11_05_000002": {"cos_patch": 0.8172928978956694, "cos_global": 0.8077837131584718,
                                      "scale_ratio": 0.9488702276074521, "mse": 0.7630867285585043}}
FIDE_TOL_COS, FIDE_TOL_REL = 0.005, 0.01

# ---- analysis constants (fixed before any output exists) ----------------------------------------------------------------
COS_LOW = 0.5                 # "position below 0.5" (per-position cosine)
IOU_LOW = 0.5                 # S-Edi vs T mask IoU below this = departure
SUSTAIN_MIN = 3               # a departure event = >= 3 consecutive decoded frames (1.5 s at 2 fps)
SUSTAIN_LONG = 10             # long event = >= 10 frames (5 s)
SWAP_IOU = 0.5                # S-Edi pig i overlaps T pig j != i with IoU >= 0.5 while IoU(S_i, T_i) < 0.5
FG_COV = 0.5                  # a 72x72 cell is "pig" if >= 50 % of its pixels are pig, "background" if 0 %
EPS_P, EPS_G = 1e-12, 1e-8    # per-position / global cosine floors (distill_core.fidelity_metrics)
TXT_PROMPT, TXT_SCORE, TXT_MASK_T = "pig", 0.5, 0.5
REGIONS = ("gt", "t3", "s3", "txt")
REG_STATS = ("cos", "p10", "lt05", "scale")
GLOBAL_STATS = ("cos_mean", "cos_p10", "cos_p50", "frac_lt05", "cos_global", "scale_ratio", "mse")
XCHECK_N = 3                  # GPU float32 metrics vs float64 host code on the first frames of every clip (hard gate)
XCHECK_TOL = 2e-4
CONTROL_MIN_COS = 0.995       # fresh teacher vs stored target: below this the pipeline is wrong -> nothing is published
PROMPT_INDEP_MIN_COS = 0.9999
PRUNE_EVERY_T01, KEEP_LAST_N_T01 = 25, 8


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LeakError(RuntimeError):
    """A test-clip id reached a path or an output: stop (nothing of it is read or published)."""


# =====================================================================================================================
# pure helpers (numpy + stdlib)
# =====================================================================================================================
def guard_path(p):
    s = str(p).replace("\\", "/")
    if any(t in s for t in _TEST_IDS) or "2019_12_" in s:
        raise LeakError(f"refused: a test-clip path ({osp.basename(s)[:40]})")
    return p


def check_clip(clip):
    if clip in _TEST_IDS or not CLIP_RX.match(str(clip)) or str(clip).startswith("2019_12_") or clip not in CLIPS:
        raise LeakError(f"clip {clip!r} is not one of the 8 non-test clips of this notebook")
    return clip


def clip_date_num(clip):
    check_clip(clip)
    return clip[:10], clip[11:]


def frame_name(fi):
    return "%07d.jpg" % (int(fi) + 1)


def target_name(fi):
    return "backbone_%06d.pt" % int(fi)


def tt_basename(clip, fi):
    check_clip(clip)
    if int(fi) < 0:
        raise ValueError(fi)
    return f"{clip}_f{int(fi):06d}"


def scan_text_for_test_ids(data):
    """bytes -> list of forbidden substrings found (test ids, '2019_12_')."""
    hits = [t for t in _TEST_IDS if t.encode() in data]
    if b"2019_12_" in data:
        hits.append("2019_12_")
    return hits


def scan_file(path):
    """Text outputs (and gzip text) are scanned for test-clip ids before they are published."""
    guard_path(path)
    low = str(path).lower()
    if low.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = f.read()
    elif low.endswith((".csv", ".json", ".txt", ".md", ".jsonl")):
        with open(path, "rb") as f:
            data = f.read()
    else:
        return []
    hits = scan_text_for_test_ids(data)
    if hits:
        raise LeakError(f"{osp.basename(path)} contains forbidden ids {hits}")
    return hits


def adaptive_windows(n_in, n_out):
    """PyTorch adaptive_avg_pool windows: [floor(i n_in / n_out), ceil((i+1) n_in / n_out))."""
    return [(int(math.floor(i * n_in / n_out)), int(math.ceil((i + 1) * n_in / n_out))) for i in range(n_out)]


_WIN_CACHE = {}


def coverage_grid(mask, out=GRID):
    """Bool [H, W] -> float32 [out, out]: share of pig pixels in each backbone cell (SAM 3 resizes 1280x720 -> 1008x1008
    -> 72x72 tokens, so token (i, j) covers rows [10 i, 10 i + 10) and columns [17.78 j, 17.78 (j + 1)))."""
    m = np.asarray(mask, dtype=bool)
    H, W = m.shape
    key = (H, W, out)
    if key not in _WIN_CACHE:
        _WIN_CACHE[key] = (adaptive_windows(H, out), adaptive_windows(W, out))
    wy, wx = _WIN_CACHE[key]
    S = np.zeros((H + 1, W + 1), dtype=np.int64)
    S[1:, 1:] = np.cumsum(np.cumsum(m, axis=0, dtype=np.int64), axis=1)
    y0 = np.array([a for a, _ in wy])
    y1 = np.array([b for _, b in wy])
    x0 = np.array([a for a, _ in wx])
    x1 = np.array([b for _, b in wx])
    tot = S[np.ix_(y1, x1)] - S[np.ix_(y0, x1)] - S[np.ix_(y1, x0)] + S[np.ix_(y0, x0)]
    area = np.outer(y1 - y0, x1 - x0)
    return (tot / area).astype(np.float32)


def region_masks(cov, fg=FG_COV):
    """coverage [72, 72] -> (pig cells: cov >= fg, background cells: cov == 0). Mixed cells belong to neither."""
    c = np.asarray(cov, dtype=np.float32)
    return c >= fg, c <= 0.0


def rasterize_boxes(boxes_xywh, H=FRAME_H, W=FRAME_W):
    """Union of [x, y, w, h] boxes (floats) as a bool [H, W] mask: columns floor(x)..ceil(x + w) - 1, clipped."""
    m = np.zeros((H, W), dtype=bool)
    for x, y, w, h in boxes_xywh:
        xa, ya = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
        xb, yb = min(W, int(math.ceil(x + w))), min(H, int(math.ceil(y + h)))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = True
    return m


def union_from_crops(crops, H=FRAME_H, W=FRAME_W):
    """crops: [(x0, y0, x1, y1, bool[y1-y0+1, x1-x0+1])] (inclusive boxes) -> bool [H, W]."""
    m = np.zeros((H, W), dtype=bool)
    for x0, y0, x1, y1, a in crops:
        m[y0:y1 + 1, x0:x1 + 1] |= a
    return m


def crop_overlap(a, b):
    """a, b = (x0, y0, x1, y1, bool crop) with inclusive boxes -> (iou, ios, intersection) of the two full masks."""
    ax0, ay0, ax1, ay1, am = a
    bx0, by0, bx1, by1, bm = b
    na, nb = int(am.sum()), int(bm.sum())
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = 0
    if ix0 <= ix1 and iy0 <= iy1:
        pa = am[iy0 - ay0:iy1 - ay0 + 1, ix0 - ax0:ix1 - ax0 + 1]
        pb = bm[iy0 - by0:iy1 - by0 + 1, ix0 - bx0:ix1 - bx0 + 1]
        inter = int(np.count_nonzero(pa & pb))
    un = na + nb - inter
    iou = inter / un if un > 0 else 0.0
    mn = min(na, nb)
    ios = inter / mn if mn > 0 else 0.0
    return iou, ios, inter


def runs_below(vals, thr, min_len):
    """Maximal runs of consecutive indices with value < thr (NaN breaks a run) of length >= min_len -> [(start, end)]
    with end inclusive."""
    out, start = [], None
    v = list(vals) + [float("nan")]
    for i, x in enumerate(v):
        below = x is not None and not (isinstance(x, float) and math.isnan(x)) and x < thr
        if below and start is None:
            start = i
        elif not below and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def classify_event(frac_absent, frac_swap):
    if frac_absent >= 0.5:
        return "drop"
    if frac_swap >= 0.5:
        return "swap"
    return "drift"


def pairwise_box_stats(boxes):
    """boxes [[x1, y1, x2, y2]] (continuous) -> max pairwise IoU, number of boxes intersecting another, pairs IoU >= 0.1."""
    n = len(boxes)
    if n < 2:
        return {"max_iou": 0.0, "n_touch": 0, "n_pairs_iou01": 0}
    touch = [False] * n
    mx, n01 = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            iou = core.box_iou(boxes[i], boxes[j])
            inter = core._box_inter(boxes[i], boxes[j])
            if inter > 0:
                touch[i] = touch[j] = True
            mx = max(mx, iou)
            n01 += iou >= 0.1
    return {"max_iou": float(mx), "n_touch": int(sum(touch)), "n_pairs_iou01": int(n01)}


def pairwise_crop_stats(crops):
    """crops {id: (x0, y0, x1, y1, m)} -> max pairwise mask IoU / IoS and the number of masks whose box touches another."""
    ids = sorted(crops)
    mx_iou = mx_ios = 0.0
    touch = set()
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            A, B = crops[ids[a]], crops[ids[b]]
            if A[0] <= B[2] + 1 and B[0] <= A[2] + 1 and A[1] <= B[3] + 1 and B[1] <= A[3] + 1:
                touch.update((ids[a], ids[b]))
                iou, ios, _ = crop_overlap(A, B)
                mx_iou, mx_ios = max(mx_iou, iou), max(mx_ios, ios)
    return {"max_iou": float(mx_iou), "max_ios": float(mx_ios), "n_touch": len(touch)}


def gray_of(rgb):
    a = np.asarray(rgb, dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def frame_descriptors(rgb, prev_gray=None):
    """Label-free per-frame descriptors on the full-resolution RGB frame (uint8 [H, W, 3]). Returns (dict, gray)."""
    a = np.asarray(rgb)
    g = gray_of(a)
    af = a.astype(np.float32)
    mx = af.max(axis=2)
    mn = af.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    gs = g[::2, ::2]
    lap = 4.0 * gs[1:-1, 1:-1] - gs[:-2, 1:-1] - gs[2:, 1:-1] - gs[1:-1, :-2] - gs[1:-1, 2:]
    d = {"brightness": float(g.mean()), "contrast": float(g.std()), "saturation": float(sat.mean()),
         "motion": float(np.abs(g - prev_gray).mean()) if prev_gray is not None else float("nan"),
         "sharpness": float(lap.var())}
    return d, g


def fid_np(s, t, regions=None):
    """Per-frame student-vs-teacher statistics in float64 (the reference the GPU code is checked against).
    s, t: [C, H, W] (leading 1-dims squeezed). regions: {name: (fg bool[H,W], bg bool[H,W])}.
    cos map = per-position cosine along C with the eps of distill_core.fidelity_metrics."""
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    while s.ndim > 3 and s.shape[0] == 1:
        s = s[0]
    while t.ndim > 3 and t.shape[0] == 1:
        t = t[0]
    if s.shape != t.shape:
        raise ValueError(f"{s.shape} vs {t.shape}")
    dot = (s * t).sum(0)
    ns = (s * s).sum(0)
    nt = (t * t).sum(0)
    cm = dot / np.sqrt((ns + EPS_P) * (nt + EPS_P))
    out = {"cos_mean": float(cm.mean()), "cos_p10": float(np.percentile(cm, 10)), "cos_p50": float(np.percentile(cm, 50)),
           "frac_lt05": float((cm < COS_LOW).mean()),
           "cos_global": float(dot.sum() / max(math.sqrt(float(ns.sum())) * math.sqrt(float(nt.sum())), EPS_G)),
           "scale_ratio": float(s.std(ddof=1) / t.std(ddof=1)), "mse": float(((s - t) ** 2).mean())}
    for name, pair in (regions or {}).items():
        for part, m in zip(("fg", "bg"), pair):
            m = np.asarray(m, dtype=bool)
            n = int(m.sum())
            out[f"{name}_{part}_n"] = n
            if n == 0:
                for k in REG_STATS:
                    out[f"{name}_{part}_{k}"] = float("nan")
                continue
            v = cm[m]
            out[f"{name}_{part}_cos"] = float(v.mean())
            out[f"{name}_{part}_p10"] = float(np.percentile(v, 10))
            out[f"{name}_{part}_lt05"] = float((v < COS_LOW).mean())
            ts = t[:, m].std(ddof=1)
            out[f"{name}_{part}_scale"] = float(s[:, m].std(ddof=1) / ts) if ts > 0 else float("nan")
    return out, cm


def mot_lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def mot_compare(mine, ref, max_frame=None):
    """Two MOT texts -> row-level comparison over frames <= max_frame (None = all)."""
    def rows(lines):
        out = {}
        for ln in lines:
            v = ln.split(",")
            f, i = int(float(v[0])), int(float(v[1]))
            if max_frame is None or f <= max_frame:
                out[(f, i)] = tuple(int(float(x)) for x in v[2:6])
        return out
    a, b = rows(mot_lines(mine)), rows(mot_lines(ref))
    keys = set(a) | set(b)
    diff = sorted(k for k in keys if a.get(k) != b.get(k))
    return {"rows_mine": len(a), "rows_ref": len(b), "rows_differ": len(diff), "identical": not diff,
            "first_diff": list(diff[0]) if diff else None}


def write_csv(path, header, rows):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else (repr(float(v)) if isinstance(v, (float, np.floating)) else v)
                        for v in (r.get(h) for h in header)] if isinstance(r, dict) else r)
    os.replace(tmp, path)
    return path


def describe(vals):
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=np.float64)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
            "max": float(v.max())}


def fide_compare(ours, ref):
    out = {}
    for m, r in ref.items():
        o = ours.get(m)
        if o is None or not math.isfinite(o):
            out[m] = {"stored": r, "ours": o, "within": False}
            continue
        if m.startswith("cos"):
            out[m] = {"stored": r, "ours": o, "diff": o - r, "within": abs(o - r) <= FIDE_TOL_COS}
        else:
            out[m] = {"stored": r, "ours": o, "rel_diff": abs(o - r) / abs(r), "within": abs(o - r) / abs(r) <= FIDE_TOL_REL}
    return out


def xcheck_bad(gpu, host):
    """Keys where the GPU float32 metrics differ from the float64 host reference. Shares below 0.5 may differ by up to two
    cells (a cosine within float32 rounding of 0.5 can fall on either side); counts must be equal."""
    bad = {}
    for k, hv in host.items():
        gv = gpu.get(k)
        if k.endswith("_n"):
            if gv != hv:
                bad[k] = (gv, hv)
            continue
        if gv is None:
            bad[k] = (gv, hv)
            continue
        if isinstance(hv, float) and math.isnan(hv):
            if not math.isnan(gv):
                bad[k] = (gv, hv)
            continue
        tol = XCHECK_TOL * max(1.0, abs(hv))
        if k.endswith("lt05"):
            n = GRID * GRID if k == "frac_lt05" else host.get(k[:-len("lt05")] + "n", 1)
            tol = max(tol, 2.0 / max(int(n), 1))
        if not abs(gv - hv) <= tol:
            bad[k] = (gv, hv)
    return bad


WIDGETS = collections.OrderedDict([
    ("mode", "smoke"), ("out_tag", ""), ("targets_tag", ""), ("require_gpu", "A10"), ("txt_stride_bench", "10"),
    ("copy_threads", "16"), ("progress_every", "100"), ("smoke_frames", "30"),
])
TAG_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_params(raw):
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}")
    g = {k: str(raw.get(k) if raw.get(k) not in (None, "") else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace()
    P.mode = g["mode"]
    if P.mode not in ("smoke", "full"):
        raise ValueError(f"mode={P.mode!r}")
    P.smoke = P.mode == "smoke"
    P.out_tag = g["out_tag"] or ("smoke_measure_v1" if P.smoke else "measure_v1")
    P.targets_tag = g["targets_tag"] or ("smoke_teacher_targets_unused_v1" if P.smoke else "teacher_targets_unused_v1")
    for t in (P.out_tag, P.targets_tag):
        if not TAG_RX.match(t):
            raise ValueError(f"tag {t!r}")
        if P.smoke != t.startswith("smoke_"):
            raise ValueError(f"tag {t!r}: smoke tags (and only they) start with 'smoke_'")
    if P.out_tag == P.targets_tag:
        raise ValueError("out_tag == targets_tag")
    P.require_gpu = g["require_gpu"]
    if P.require_gpu not in ("A10", "A100"):
        raise ValueError(f"require_gpu={P.require_gpu!r}")
    P.txt_stride = int(g["txt_stride_bench"])
    if P.txt_stride < 1:
        raise ValueError("txt_stride_bench >= 1")
    P.copy_threads = max(1, int(g["copy_threads"]))
    P.progress_every = max(1, int(g["progress_every"]))
    P.n = int(g["smoke_frames"]) if P.smoke else N_FRAMES
    if not 1 <= P.n <= N_FRAMES:
        raise ValueError(f"frames {P.n}")
    return P


# =====================================================================================================================
# =====================================================================================================================


if not IS_NOTEBOOK:
    print("This notebook needs a GPU cluster and the path registry: see docs/REPRODUCE.md")
    raise SystemExit(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: versions, GPU, registry, pins, output folders, staging

# COMMAND ----------

import torch                      # noqa: E402
from PIL import Image             # noqa: E402

T0 = time.time()
TIMINGS = collections.OrderedDict()
ERRORS = []
NOTES = []
for _k, _v in WIDGETS.items():
    dbutils.widgets.text(_k, _v)
RAW = {k: dbutils.widgets.get(k) for k in WIDGETS}
P = parse_params(RAW)
NF = P.n
_log(f"{NB_NAME} {NB_VERSION}: mode {P.mode}, frames per clip {NF}, out {P.out_tag}, targets {P.targets_tag}")

CODE_SHAS = {"bench_core": core.sha256_file(core.__file__), "distill_core": core.sha256_file(dc.__file__),
             "farm_paths": core.sha256_file(farm_paths.__file__)}
if tuple(core.EDI_TRAIN) != EDI_TRAIN or tuple(core.EDI_VAL) != EDI_VAL or tuple(core.EDI_TEST) != _TEST_IDS:
    raise RuntimeError("bench_core split constants differ from this notebook's")
if set(CLIPS) & set(core.EDI_TEST):
    raise LeakError("a test clip is among the clips of this notebook")


def _notebook_source():
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    for cand in ("/Workspace" + nbp, "/Workspace" + nbp + ".py"):
        if osp.isfile(cand):
            with open(cand, "rb") as f:
                b = f.read()
            if b.lstrip(b"\xef\xbb\xbf").startswith(b"# Databricks notebook source"):
                return b
    import base64
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ExportFormat
    return base64.b64decode(WorkspaceClient().workspace.export(nbp, format=ExportFormat.SOURCE).content)


try:
    CODE_SHAS["notebook"] = hashlib.sha256(_notebook_source()).hexdigest()
except Exception as e:  # noqa: BLE001
    CODE_SHAS["notebook"] = f"unavailable: {e!r}"[:200]
VERSIONS = core.versions()
VERSIONS["databricks_runtime"] = os.environ.get("DATABRICKS_RUNTIME_VERSION")
_log("versions", VERSIONS)
if not str(VERSIONS["databricks_runtime"] or "").startswith("17.3"):
    raise RuntimeError(f"Databricks runtime {VERSIONS['databricks_runtime']} is not 17.3")
if VERSIONS.get("transformers") != core.TRANSFORMERS_PIN or VERSIONS.get("timm") != "1.0.30":
    raise RuntimeError(f"transformers {VERSIONS.get('transformers')} / timm {VERSIONS.get('timm')} != 5.2.0 / 1.0.30")


def cuda_probe(retries=3, sleep_s=60):
    code = ("import torch; torch.zeros(1, device='cuda'); torch.cuda.synchronize(); "
            "print(torch.cuda.get_device_name(0))")
    last = None
    for k in range(retries):
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                return {"ok": True, "attempt": k + 1}
            last = (r.stderr or r.stdout or "")[-600:]
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        _log(f"CUDA probe attempt {k + 1}/{retries} failed: {last}")
        if k + 1 < retries:
            time.sleep(sleep_s)
    raise RuntimeError(f"CUDA start-up failed {retries}x: {last}")


GPU = core.gpu_info()
GPU["probe"] = cuda_probe()
if GPU["class"] != P.require_gpu:
    raise RuntimeError(f"GPU {GPU['name']} (class {GPU['class']}) but require_gpu={P.require_gpu}")
DEV = torch.device("cuda")
_log("GPU", GPU)

R = farm_paths.load(version=1)
OUT = R.out("paper1.root", "edi_mining", P.out_tag)
TT_OUT = R.out("paper1.root", "edi_mining", P.targets_tag)
for _o in (OUT, TT_OUT):
    core.assert_publish_target(_o + "/x")
    if osp.exists(_o):
        raise RuntimeError(f"{_o} exists: this notebook writes only NEW folders (use a new tag)")
LOCAL = f"/local_disk0/tol79/{P.out_tag}"
shutil.rmtree(LOCAL, ignore_errors=True)
for _d in ("stage/frames", "stage/gt", "stage/targets", "out", "tt/seqs", "tt/records", "tmp"):
    os.makedirs(osp.join(LOCAL, _d), exist_ok=True)
LOUT = osp.join(LOCAL, "out")


def read_bytes(p):
    guard_path(p)
    with open(p, "rb") as f:
        return f.read()


# ---- protocol files (sha-pinned) ------------------------------------------------------------------------------------------
XY_O = 0                       # MOT xy offset: 0 on both pig datasets
_raw = read_bytes(core.protocol_file_path(R, HF_REV_FILE))
REVISIONS = dict(sorted(json.loads(_raw.decode("utf-8"))["revisions"].items()))
if REVISIONS.get(HF_SAM3) != SAM3_REV:
    raise RuntimeError(f"hf_revisions pins {REVISIONS.get(HF_SAM3)} for SAM 3, not {SAM3_REV}")
del _raw
_log(f"frozen MOT xy offset o = {XY_O}; SAM 3 commit {SAM3_REV[:12]}")

# ---- background publisher (one thread, no-overwrite, test-id scan first) ---------------------------------------------------
PUB = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tol79_pub")
PUB_FUTS = []
PUBLISHED = {}


def _publish_now(local, rel):
    scan_file(local)
    dst = f"{OUT}/{rel}"
    guard_path(dst)
    state = core.publish_no_overwrite(local, dst, dbutils)
    PUBLISHED[rel] = {"sha256": core.sha256_file(local), "bytes": osp.getsize(local), "state": state}
    return state


def publish(local, rel):
    for f in PUB_FUTS:
        if f.done() and f.exception() is not None:
            raise RuntimeError("a background publish failed") from f.exception()
    PUB_FUTS.append(PUB.submit(_publish_now, local, rel))


def write_json_local(path, obj):
    return core.write_text_atomic(path, json.dumps(obj, indent=1, sort_keys=True, default=core._json_default) + "\n")


# ---- staging: frames (all 600 of every clip: EdinburghFrames needs the full clip), GT, seeds -------------------------------
t_ph = time.time()
FRAMES_DIR = {}
STAGE = {}
jobs = []
for clip in CLIPS:
    src = guard_path(R.path("datasets.edinburgh_pigs.decoded_frames", None, check_clip(clip), "video1"))
    dst = osp.join(LOCAL, "stage", "frames", clip, "video1")
    FRAMES_DIR[clip] = dst
    jobs += [(guard_path(f"{src}/{frame_name(k)}"), f"{dst}/{frame_name(k)}", "src", None) for k in range(N_FRAMES)]
STAGE["frames"] = dc.copy_files_parallel(jobs, threads=P.copy_threads, log=_log)
_log("staged frames", STAGE["frames"])
GT_LOCAL, GT_SHA = {}, {}
for clip in CLIPS:
    if GT_STATUS[clip] == "wrong":
        continue
    date, num = clip_date_num(clip)
    src = guard_path(R.path("datasets.edinburgh_pigs.annotated", None, date, num, "output.json"))
    GT_LOCAL[clip], GT_SHA[clip] = core.stage_file(src, osp.join(LOCAL, "stage", "gt", clip, "output.json"), dbutils)
    if not GT_SHA[clip].startswith(GT_SHA256_PREFIX[clip]):
        raise RuntimeError(f"{clip}: GT sha256 {GT_SHA[clip][:12]} != INVENTORY {GT_SHA256_PREFIX[clip]}")
SEED_LOCAL = {}
for clip in TRACK_CLIPS:
    src = guard_path(R.path("datasets.edinburgh_pigs.initial_prompts", None, clip, "video1", f"BoundingBoxes_{clip}.txt"))
    SEED_LOCAL[clip], _ = core.stage_file(src, osp.join(LOCAL, "stage", "gt", clip, "BoundingBoxes.txt"), dbutils)
KF = {c: core.read_edinburgh_keyframes(GT_LOCAL[c]) for c in GT_LOCAL}
GT_ROWS = {c: core.edinburgh_gt_rows(KF[c], n_frames=N_FRAMES, interpolate=True) for c in KF}
GT_BY_F = {}
for c, a in GT_ROWS.items():
    d = collections.defaultdict(dict)
    for r in a:
        d[int(r[0])][int(r[1])] = (float(r[2]), float(r[3]), float(r[4]), float(r[5]))
    GT_BY_F[c] = d
TIMINGS["stage_frames_gt_s"] = round(time.time() - t_ph, 1)

# ---- stored targets of the 5 benchmark clips: staged in the background, one clip after another --------------------------
STAGER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tol79_stage")
STORED_FUT = {}
STORED_DIR = {}
for clip in TRACK_CLIPS:
    src = guard_path(R.path("paper1.teacher_targets.edinburgh", None, clip))
    dst = osp.join(LOCAL, "stage", "targets", clip)
    STORED_DIR[clip] = dst
    tj = [(guard_path(f"{src}/{target_name(k)}"), f"{dst}/{target_name(k)}", "src", None) for k in range(NF)]
    STORED_FUT[clip] = STAGER.submit(dc.copy_files_parallel, tj, P.copy_threads, _log)

# ---- background medians (CPU thread): per clip from every 10th frame + pooled over the 8 clips ----------------------------
BG = {}


def _bg_medians():
    stacks = []
    for clip in CLIPS:
        fr = []
        for k in range(0, N_FRAMES, 10):
            with Image.open(osp.join(FRAMES_DIR[clip], frame_name(k))) as im:
                fr.append(np.asarray(im.convert("RGB")))
        st = np.stack(fr)
        med = np.median(st, axis=0).astype(np.uint8)
        p = osp.join(LOUT, "bg_median", f"{clip}.png")
        os.makedirs(osp.dirname(p), exist_ok=True)
        Image.fromarray(med).save(p)
        stacks.append(st[::2])
        BG[clip] = p
    pooled = np.median(np.concatenate(stacks), axis=0).astype(np.uint8)
    p = osp.join(LOUT, "bg_median", "pooled_8clips.png")
    Image.fromarray(pooled).save(p)
    BG["pooled"] = p
    return {"per_clip_frames": N_FRAMES // 10, "pooled_frames": sum(len(s) for s in stacks)}


BG_FUT = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tol79_bg").submit(_bg_medians)
_log(f"setup done in {time.time() - T0:.0f} s; out {OUT}; targets {TT_OUT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## (a) Teacher targets for the 3 unused clips (Sam3Teacher VERBATIM extract_teacher_targets_pigtrack) + the 20-target recompute control

# COMMAND ----------

class BackboneFeatureHook:
    """VERBATIM extract_teacher_targets (= extract_teacher_targets_pigtrack): capture backbone output, reshape [1,5184,1024] -> [1,1024,72,72], fp16 on CPU."""
    def __init__(self, m):
        self.captured = None
        self.h = m.vision_encoder.backbone.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if hasattr(out, "last_hidden_state"):
            lhs = out.last_hidden_state
            B, N, C = lhs.shape
            H = W = int(N ** 0.5)
            self.captured = lhs.permute(0, 2, 1).reshape(B, C, H, W).detach().cpu().half()
        elif isinstance(out, torch.Tensor):
            self.captured = out.detach().cpu().half()

    def pop(self):
        f, self.captured = self.captured, None
        return f

    def remove(self):
        self.h.remove()


def prune_inference_session(session, keep_last_n=KEEP_LAST_N_T01):
    """VERBATIM extract_teacher_targets."""
    for oi in range(session.get_obj_num()):
        nc = session.output_dict_per_obj[oi].get("non_cond_frame_outputs", {})
        if len(nc) > keep_last_n:
            for k in sorted(nc.keys())[:-keep_last_n]:
                del nc[k]


class Sam3Teacher:
    """VERBATIM extract_teacher_targets_pigtrack (the extraction code of the PigTrack targets; same model class, bf16, processor 1008, hook)."""
    def __init__(self, revision, token, device="cuda"):
        import transformers
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        if transformers.__version__ != core.TRANSFORMERS_PIN:
            raise RuntimeError(f"transformers {transformers.__version__} != {core.TRANSFORMERS_PIN}")
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        kw = {"token": token}
        if revision:
            kw["revision"] = revision
        t0 = time.time()
        self.proc = Sam3TrackerVideoProcessor.from_pretrained(HF_SAM3, **kw)
        self.model = Sam3TrackerVideoModel.from_pretrained(HF_SAM3, torch_dtype=self.dtype, **kw).to(self.device).eval()
        commit, src = core.hf_commit(self.model, HF_SAM3, revision, token)
        ip = getattr(self.proc, "image_processor", self.proc)
        pscal = {k: v for k, v in core._config_scalars(ip).items() if not k.startswith("_")}
        if pscal.get("size.height") != 1008 or pscal.get("size.width") != 1008:
            raise RuntimeError(f"SAM 3 processor size {pscal.get('size.height')}x{pscal.get('size.width')} != 1008")
        self.info = {"hf_id": HF_SAM3, "revision_requested": revision, "commit_hash": commit, "commit_source": src,
                     "attn_resolved": core._attn_impls(self.model.config), "dtype": "bfloat16",
                     "n_params": int(sum(p.numel() for p in self.model.parameters())), "processor": pscal,
                     "load_seconds": round(time.time() - t0, 1)}
        self.hook = BackboneFeatureHook(self.model)

    def _forward_once(self, pil, boxes):
        session = self.proc.init_video_session(inference_device=self.device, dtype=self.dtype)
        inputs = self.proc(images=pil, device=self.device, return_tensors="pt")
        self.proc.add_inputs_to_inference_session(inference_session=session, frame_idx=0,
                                                  obj_ids=list(range(len(boxes))), input_boxes=[boxes],
                                                  original_size=inputs.original_sizes[0])
        with torch.no_grad():
            self.model(inference_session=session, frame=inputs.pixel_values[0])
        f = self.hook.pop()
        del session, inputs
        return f

    def prompt_independence(self, path, w, h):
        with Image.open(path) as im:
            pil = im.convert("RGB")
        a = self._forward_once(pil, dc.dummy_boxes(w, h))
        b = self._forward_once(pil, [[int(0.05 * w), int(0.05 * h), int(0.30 * w), int(0.40 * h)]])
        a32, b32 = a.float(), b.float()
        cos = float(torch.nn.functional.cosine_similarity(a32.reshape(1, -1), b32.reshape(1, -1)).item())
        gc.collect()
        torch.cuda.empty_cache()
        return {"max_abs": float((a32 - b32).abs().max()), "cos_global": cos,
                "bit_identical": bool(torch.equal(a, b)), "shape": list(a.shape)}

    def extract_seq(self, items, w, h):
        """extract_teacher_targets's loop on the selected frames of one sequence: ONE session, dummy boxes on its first frame."""
        session = self.proc.init_video_session(inference_device=self.device, dtype=self.dtype)
        try:
            for si, (fi, path) in enumerate(items):
                with Image.open(path) as im:
                    pil = im.convert("RGB")
                if pil.size != (w, h):
                    raise core.BenchAssertError(f"{path}: {pil.size} != seqinfo {w}x{h}")
                inputs = self.proc(images=pil, device=self.device, return_tensors="pt")
                if si == 0:
                    # fresh lists per session -- the processor MUTATES obj_ids
                    self.proc.add_inputs_to_inference_session(
                        inference_session=session, frame_idx=0, obj_ids=list(range(len(dc.DUMMY_BOX_FRACS))),
                        input_boxes=[dc.dummy_boxes(w, h)], original_size=inputs.original_sizes[0])
                with torch.no_grad():
                    self.model(inference_session=session, frame=inputs.pixel_values[0])
                feat = self.hook.pop()
                if feat is None:
                    raise core.BenchAssertError(f"{path}: backbone hook captured nothing")
                f32 = feat.float()
                stats = {"shape": tuple(feat.shape), "dtype": str(feat.dtype).replace("torch.", ""),
                         "finite": bool(torch.isfinite(f32).all()), "mean": float(f32.mean()), "std": float(f32.std())}
                yield fi, feat, stats
                del inputs
                if (si + 1) % PRUNE_EVERY_T01 == 0:
                    prune_inference_session(session, KEEP_LAST_N_T01)
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            del session
            gc.collect()
            torch.cuda.empty_cache()

    def close(self):
        self.hook.remove()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def compare_features(a16, b16):
    """Two fp16 [1,1024,72,72] CPU tensors -> agreement (float64 on the host)."""
    a = a16.float().reshape(-1).double()
    b = b16.float().reshape(-1).double()
    d = (a - b).abs()
    return {"cos_global": float((a @ b) / max(float(a.norm() * b.norm()), EPS_G)), "max_abs": float(d.max()),
            "mean_abs": float(d.mean()), "frac_equal": float((a16.reshape(-1) == b16.reshape(-1)).float().mean()),
            "bit_identical": bool(torch.equal(a16.reshape(-1), b16.reshape(-1)))}


HF_TOKEN = core.hf_token_from_dbutils(dbutils)
from huggingface_hub import login as _hf_login   # noqa: E402
_hf_login(token=HF_TOKEN, add_to_git_credential=False)

t_ph = time.time()
TEACHER = Sam3Teacher(SAM3_REV, HF_TOKEN)
if TEACHER.info.get("commit_hash") != SAM3_REV:
    raise RuntimeError(f"teacher commit {TEACHER.info.get('commit_hash')} != {SAM3_REV}")
if any(v == "flash_attention_2" for v in (TEACHER.info.get("attn_resolved") or {}).values()):
    raise RuntimeError(f"flash_attention_2 resolved: {TEACHER.info.get('attn_resolved')}")
_log("teacher", {k: TEACHER.info[k] for k in ("commit_hash", "attn_resolved", "load_seconds")})
PI = TEACHER.prompt_independence(osp.join(FRAMES_DIR[UNUSED[0]], frame_name(0)), FRAME_W, FRAME_H)
_log("prompt independence", PI)
if not PI["cos_global"] >= PROMPT_INDEP_MIN_COS:
    raise core.BenchAssertError(f"backbone output depends on the prompt? {PI}")

# ---- control: 20 stored training-clip targets recomputed with exactly this code (nothing published before it passes) -----
CONTROL_ROWS = []
CONTROL_FEATS = {}
for clip in EDI_TRAIN:
    items = [(fi, osp.join(FRAMES_DIR[clip], frame_name(fi))) for fi in CONTROL_FI]
    for fi, feat, stats in TEACHER.extract_seq(items, FRAME_W, FRAME_H):
        stored = torch.load(io.BytesIO(read_bytes(R.path("paper1.teacher_targets.edinburgh", None, clip,
                                                          target_name(fi)))), map_location="cpu", weights_only=True)
        if tuple(stored.shape) != tuple(feat.shape) or stored.dtype != torch.float16:
            raise RuntimeError(f"{clip} f{fi}: stored {tuple(stored.shape)} {stored.dtype} vs fresh {tuple(feat.shape)}")
        cmp_ = compare_features(feat, stored)
        CONTROL_FEATS[(clip, fi)] = feat
        CONTROL_ROWS.append(dict(clip=clip, fi=fi, stored_strides="x".join(map(str, stored.stride())),
                                 fresh_strides="x".join(map(str, feat.stride())), **cmp_))
CONTROL_SUMMARY = {k: describe([r[k] for r in CONTROL_ROWS]) for k in ("cos_global", "max_abs", "mean_abs", "frac_equal")}
CONTROL_SUMMARY["n"] = len(CONTROL_ROWS)
CONTROL_SUMMARY["n_bit_identical"] = sum(r["bit_identical"] for r in CONTROL_ROWS)
_log("control (fresh vs stored, 20 training frames)", CONTROL_SUMMARY)
if min(r["cos_global"] for r in CONTROL_ROWS) < CONTROL_MIN_COS:
    raise core.BenchAssertError(f"control: fresh teacher vs stored target cos_global < {CONTROL_MIN_COS} "
                                f"({CONTROL_SUMMARY['cos_global']}); nothing is published")
write_csv(osp.join(LOUT, "controls", "teacher_recompute_20.csv"),
          ["clip", "fi", "cos_global", "max_abs", "mean_abs", "frac_equal", "bit_identical", "stored_strides",
           "fresh_strides"], CONTROL_ROWS)

# ---- the 3 unused clips: every frame (smoke: the first NF), extract_teacher_targets_pigtrack layout, published per clip in the background -----------
TT_INFO = {"n": N_FRAMES, "night": 0, "pen": "edinburgh_pen", "fps": float(core.EDI_FPS), "width": FRAME_W,
           "height": FRAME_H}
TT_CFG = {"version": TARGETS_CONFIG_VERSION, "notebook": NB_NAME, "nb_version": NB_VERSION, "created_utc": _now(),
          "dataset": {"name": "Edinburgh pig dataset (CC BY-NC); one pen, one overhead camera, 8 pigs",
                      "clips": list(UNUSED), "frames_per_clip": NF, "frame_rule":
                      "decoded frame fi = decoded_frames/<clip>/video1/{fi+1:07d}.jpg = raw color.mp4 frame 3 fi (2 fps)",
                      "labels_read": False, "gt_status": {c: GT_STATUS[c] for c in UNUSED}},
          "role_note": "manifest role=train marks a CANDIDATE training target; no selection has been made",
          "teacher": {"hf_id": HF_SAM3, "revision": SAM3_REV, "dtype": "bfloat16", "transformers": core.TRANSFORMERS_PIN,
                      "code": "Sam3Teacher VERBATIM extract_teacher_targets_pigtrack (extract_teacher_targets BackboneFeatureHook, one session per clip, implicit "
                              "frame index, dummy boxes distill_core.DUMMY_BOX_FRACS, prune keep_last_n=8 every 25)",
                      "info": TEACHER.info, "prompt_independence": PI},
          "control_vs_stored_edinburgh_targets": {"frames": {c: list(CONTROL_FI) for c in EDI_TRAIN},
                                                  "summary": CONTROL_SUMMARY, "min_cos_required": CONTROL_MIN_COS,
                                                  "note": "the stored 2026-02 targets are not bit-reproducible "
                                                          "(INVENTORY.md 4); same pipeline, different environment"},
          "target": {"shape": list(dc.TARGET_SHAPE), "dtype": dc.TARGET_DTYPE, "format": "torch.save(tensor)",
                     "frames": "decoded JPEG copied byte for byte"},
          "versions": VERSIONS, "gpu": GPU, "code": CODE_SHAS}
_cfg_p = write_json_local(osp.join(LOCAL, "tt", "TARGETS_CONFIG.json"), TT_CFG)
scan_file(_cfg_p)
core.publish_no_overwrite(_cfg_p, f"{TT_OUT}/TARGETS_CONFIG.json", dbutils)

TT_PUB = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tol79_ttpub")
TT_FUTS = {}
TT_RECORDS = {}
TT_EXTRACT_S = {}


def publish_tt_clip(clip, seq_local, files, local_sha, t_extract):
    res = dc.publish_tree_attempt(seq_local, f"{TT_OUT}/seqs", clip, local_sha=local_sha, dbutils=dbutils, log=_log)
    rec = {"seq": clip, "dir": f"seqs/{res['name']}",
           "publish": {k: res[k] for k in ("mode", "files", "bytes", "seconds")}, "n": len(files), "files": files,
           "extract_seconds": round(t_extract, 2), "info": dict(TT_INFO), "role": "train", "nb": NB_VERSION,
           "gt_status": GT_STATUS[clip], "feat_mean_of_means": float(np.mean([f["feat_mean"] for f in files])),
           "feat_mean_of_stds": float(np.mean([f["feat_std"] for f in files]))}
    lp = write_json_local(osp.join(LOCAL, "tt", "records", f"{clip}.json"), rec)
    scan_file(lp)
    core.publish_no_overwrite(lp, f"{TT_OUT}/records/{clip}.json", dbutils)
    TT_RECORDS[clip] = rec
    return res


TT_LOCAL_FEAT = {}
for clip in UNUSED:
    seq_local = osp.join(LOCAL, "tt", "seqs", clip)
    shutil.rmtree(seq_local, ignore_errors=True)
    os.makedirs(osp.join(seq_local, "frames"))
    os.makedirs(osp.join(seq_local, "features"))
    t0 = time.time()
    files, local_sha = [], {}
    items = [(fi, osp.join(FRAMES_DIR[clip], frame_name(fi))) for fi in range(NF)]
    for fi, feat, stats in TEACHER.extract_seq(items, FRAME_W, FRAME_H):
        if tuple(stats["shape"]) != tuple(dc.TARGET_SHAPE) or stats["dtype"] != dc.TARGET_DTYPE:
            raise core.BenchAssertError(f"{clip} f{fi}: target {stats['shape']} {stats['dtype']}")
        if not (stats["finite"] and stats["std"] > core.MIN_STD):
            raise core.BenchAssertError(f"{clip} f{fi}: non-finite or collapsed teacher features {stats}")
        base = tt_basename(clip, fi)
        fp = osp.join(seq_local, "features", base + ".pt")
        torch.save(feat, fp)
        ip = osp.join(seq_local, "frames", base + ".jpg")
        shutil.copyfile(items[fi][1], ip)
        fsha, isha = core.sha256_file(fp), core.sha256_file(ip)
        local_sha[f"features/{base}.pt"] = fsha
        local_sha[f"frames/{base}.jpg"] = isha
        TT_LOCAL_FEAT[(clip, fi)] = fp
        files.append({"fi": int(fi), "src_image": frame_name(fi), "image": base + ".jpg", "feature": base + ".pt",
                      "image_bytes": osp.getsize(ip), "image_sha256": isha, "feature_bytes": osp.getsize(fp),
                      "feature_sha256": fsha, "feat_mean": float(stats["mean"]), "feat_std": float(stats["std"])})
    if [f["fi"] for f in files] != list(range(NF)):
        raise core.BenchAssertError(f"{clip}: extracted {len(files)} targets, planned {NF}")
    TT_EXTRACT_S[clip] = round(time.time() - t0, 2)
    _log(f"{clip}: {len(files)} targets in {TT_EXTRACT_S[clip]} s ({TT_EXTRACT_S[clip] / len(files):.3f} s/frame)")
    TT_FUTS[clip] = TT_PUB.submit(publish_tt_clip, clip, seq_local, files, local_sha, TT_EXTRACT_S[clip])
TEACHER_INFO = dict(TEACHER.info)
TEACHER.close()
del TEACHER
gc.collect()
torch.cuda.empty_cache()
TIMINGS["phase_a_teacher_s"] = round(time.time() - t_ph, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Label-free pig region: SAM 3 text prompt "pig" (Sam3Model) - every frame of the unused clips, every `txt_stride_bench`-th frame of the benchmark clips (validation against GT / T masks)

# COMMAND ----------

TXT = {c: {} for c in CLIPS}            # clip -> fi -> {"grid", "n", "boxes", "scores", "areas", "crowd"}
TXT_INFO = {"prompt": TXT_PROMPT, "score_threshold": TXT_SCORE, "mask_threshold": TXT_MASK_T,
            "frames": "unused: every frame; benchmark: fi % txt_stride_bench == 0", "stride_bench": P.txt_stride}
t_ph = time.time()
DET = DPROC = None
try:
    from transformers import Sam3Model, Sam3Processor
    _t = time.time()
    DPROC = Sam3Processor.from_pretrained(HF_SAM3, revision=SAM3_REV, token=HF_TOKEN)
    DET = Sam3Model.from_pretrained(HF_SAM3, revision=SAM3_REV, token=HF_TOKEN, torch_dtype=torch.bfloat16).to(DEV).eval()
    TXT_INFO.update({"load_s": round(time.time() - _t, 1), "attn_resolved": core._attn_impls(DET.config),
                     "commit": core.hf_commit(DET, HF_SAM3, SAM3_REV, HF_TOKEN)[0],
                     "n_params": int(sum(p.numel() for p in DET.parameters()))})
    TXT_INFO["input_cast"] = "none (BatchFeature.to(device))"

    def detect_txt(pil):
        with torch.no_grad():
            di = DPROC(images=pil, text=TXT_PROMPT, return_tensors="pt").to(DEV)
            if TXT_INFO["input_cast"].startswith("bf16"):
                di["pixel_values"] = di["pixel_values"].to(torch.bfloat16)
            do = DET(**di)
            res = DPROC.post_process_instance_segmentation(do, threshold=TXT_SCORE, mask_threshold=TXT_MASK_T,
                                                           target_sizes=di["original_sizes"].tolist())[0]
        masks = [m.cpu().numpy().astype(bool) if torch.is_tensor(m) else np.asarray(m, dtype=bool) for m in res["masks"]]
        scores = [float(s) for s in res["scores"]]
        if len(masks) != len(scores):
            raise RuntimeError(f"text prompt: {len(masks)} masks vs {len(scores)} scores")
        union = np.zeros((FRAME_H, FRAME_W), bool)
        boxes, areas, keep_s = [], [], []
        for mm, sc in zip(masks, scores):
            if mm.shape != (FRAME_H, FRAME_W):
                raise RuntimeError(f"text-prompt mask shape {mm.shape}")
            if not mm.any():
                continue
            ys = np.flatnonzero(mm.any(1))
            xs = np.flatnonzero(mm.any(0))
            boxes.append([int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1])
            areas.append(int(mm.sum()))
            keep_s.append(sc)
            union |= mm
        crowd = pairwise_box_stats(boxes)
        return {"grid": coverage_grid(union), "n": len(boxes), "boxes": boxes, "scores": keep_s, "areas": areas,
                "crowd": crowd}

    def txt_frames(clip):
        return list(range(NF)) if clip in UNUSED else [fi for fi in range(NF) if fi % P.txt_stride == 0]

    _first = True
    for clip in UNUSED + TRACK_CLIPS:
        tc = time.time()
        for fi in txt_frames(clip):
            with Image.open(osp.join(FRAMES_DIR[clip], frame_name(fi))) as im:
                pil = im.convert("RGB")
            if _first:
                try:
                    TXT[clip][fi] = detect_txt(pil)
                except RuntimeError as e:          # dtype mismatch on an fp32 input -> cast once, record it
                    if "dtype" not in str(e).lower() and "type" not in str(e).lower():
                        raise
                    TXT_INFO["input_cast"] = f"bf16 pixel_values (after: {str(e)[:160]})"
                    TXT[clip][fi] = detect_txt(pil)
                _first = False
            else:
                TXT[clip][fi] = detect_txt(pil)
        n = len(TXT[clip])
        _log(f"txt {clip}: {n} frames in {time.time() - tc:.0f} s, detections per frame "
             f"{describe([v['n'] for v in TXT[clip].values()])}")
        TXT_INFO.setdefault("per_clip_s", {})[clip] = round(time.time() - tc, 1)
        rec = [{"fi": fi, "n": v["n"], "boxes": v["boxes"], "scores": [round(s, 4) for s in v["scores"]],
                "areas": v["areas"]} for fi, v in sorted(TXT[clip].items())]
        p = osp.join(LOUT, "clips", clip, "txt_detections.json.gz")
        os.makedirs(osp.dirname(p), exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump({"clip": clip, "prompt": TXT_PROMPT, "score_threshold": TXT_SCORE, "frames": rec}, f)
        publish(p, f"clips/{clip}/txt_detections.json.gz")
    DET = DPROC = None
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:  # noqa: BLE001 -- the text-prompt region is supplementary; the run continues without it
    ERRORS.append(f"(text-prompt region) {type(e).__name__}: {e}")
    _log("text-prompt region FAILED:\n" + traceback.format_exc())
    DET = DPROC = None
    gc.collect()
    torch.cuda.empty_cache()
TIMINGS["phase_txt_s"] = round(time.time() - t_ph, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## (c) + (b) Tracking (T and S-Edi, seeded run, the frozen bench settings) and feature fidelity

# COMMAND ----------

t_ph = time.time()
ARM_T = core.load_arm("T", device="cuda", hf_token=HF_TOKEN, revisions=REVISIONS, registry=R, dbutils=dbutils, log=_log)
ARM_S = core.load_arm("S-Edi", device="cuda", hf_token=HF_TOKEN, revisions=REVISIONS, registry=R, dbutils=dbutils,
                      student_sha256=CKPT_SHA256, log=_log)
for _a in (ARM_T, ARM_S):
    if _a.info.get("commit_hash") != SAM3_REV:
        raise RuntimeError(f"{_a.name}: commit {_a.info.get('commit_hash')} != {SAM3_REV}")
if ARM_S.info["student"]["sha256"] != CKPT_SHA256:
    raise RuntimeError("S-Edi checkpoint sha256 differs from the pin")
CKPT_LOCAL = osp.join("/local_disk0/bench_models", "S-Edi", ARM_S.info["student"]["file"])
if core.sha256_file(CKPT_LOCAL) != CKPT_SHA256:
    raise RuntimeError(f"{CKPT_LOCAL}: sha256 differs from the pin")
# fp32 copy for the [FID-E] replication control only (a fresh fp32 load: no bf16 cast of a cached attention-bias table)
S32, S32_META = core.load_student(CKPT_LOCAL, device="cuda", dtype=torch.float32, check_gn=True)
S32_TF = core.make_student_transform()
ARMS_INFO = {"T": ARM_T.info, "S-Edi": ARM_S.info, "S32_control": S32_META}
TIMINGS["load_arms_s"] = round(time.time() - t_ph, 1)
_log("arms loaded", {"T": ARM_T.info["params"], "S-Edi": ARM_S.info["params"], "student": ARM_S.info["student"]})


def crop_of(binm, r):
    return (r.x0, r.y0, r.x1, r.y1, binm[r.k, r.y0:r.y1 + 1, r.x0:r.x1 + 1].cpu().numpy().astype(bool))


def track_arm(arm, clip, prompts, tag, capture_hook=None, capture_every=None):
    """track_stream with the track_bench settings (k16, prune 25, pop, Telemetry); keeps every emitted mask as a crop."""
    frames = core.EdinburghFrames(FRAMES_DIR[clip])
    mot = core.MotWriter()
    rle_p = osp.join(LOUT, "clips", clip, f"rle_{tag}.jsonl.gz")
    rle = core.RleWriter(rle_p)
    tel = core.Telemetry(arm.model)
    per = {}
    caps = {}
    hook_none = []

    def on_frame(fi, sess, out, mask_logits, binm, rows, live, pruned):
        crops = {r.obj_id: crop_of(binm, r) for r in rows}
        per[fi] = {"rows": {r.obj_id: (r.x0, r.y0, r.x1, r.y1, r.area, r.score) for r in rows}, "crops": crops,
                   "grid": coverage_grid(union_from_crops(list(crops.values())))}
        if capture_hook is not None:
            f = capture_hook.pop()
            if f is None:
                hook_none.append(fi)
            elif capture_every is None or fi % capture_every == 0:
                caps[fi] = f
    try:
        st = core.track_stream(arm, frames, protocol="A", prompts=prompts, keep_last_n=core.KEEP_LAST_N,
                               prune_every=core.PRUNE_EVERY, pop=core.POP_FRAMES, mot=mot, rle=rle, tel=tel,
                               max_frames=NF if NF < N_FRAMES else None, progress_every=P.progress_every, log=_log,
                               on_frame=on_frame)
    finally:
        rle.close()
        tsum = tel.summary()
        tel.detach()
    if st["frames"] != NF or sorted(per) != list(range(NF)):
        raise core.BenchAssertError(f"{clip} {tag}: {st['frames']} frames / {len(per)} outputs, expected {NF}")
    if hook_none:
        raise core.BenchAssertError(f"{clip} {tag}: backbone capture empty on frames {hook_none[:5]}")
    mot_p = mot.write(osp.join(LOUT, "clips", clip, f"mot_{tag}.txt"))
    return per, caps, st, tsum, mot_p, rle_p


class DummyCapture:
    """Observe the DummyBackbone output (= the student features the tracker used) -> [1,1024,72,72] bf16 on the CPU."""
    def __init__(self, module):
        self.captured = None
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        lhs = out.last_hidden_state
        B, N, C = lhs.shape
        self.captured = lhs.reshape(B, GRID, GRID, C).permute(0, 3, 1, 2).detach().cpu()

    def pop(self):
        f, self.captured = self.captured, None
        return f

    def remove(self):
        self.h.remove()


def fid_torch(s, t, regions):
    """GPU float32 mirror of fid_np. s, t: [C, 72, 72] float32 cuda. regions {name: (fg, bg) numpy bool}.
    -> (dict, cos map numpy float16)."""
    dot = (s * t).sum(0)
    ns = (s * s).sum(0)
    nt = (t * t).sum(0)
    cm = dot / torch.sqrt((ns + EPS_P) * (nt + EPS_P))
    flat = cm.reshape(-1)
    nan = torch.tensor(float("nan"), device=s.device)
    keys = list(GLOBAL_STATS)
    vals = [flat.mean(), torch.quantile(flat, 0.10), torch.quantile(flat, 0.50), (flat < COS_LOW).float().mean(),
            dot.sum() / torch.clamp(torch.sqrt(ns.sum()) * torch.sqrt(nt.sum()), min=EPS_G),
            s.std() / t.std(), ((s - t) ** 2).mean()]
    counts = {}
    for name, pair in regions.items():
        for part, m in zip(("fg", "bg"), pair):
            n = int(np.count_nonzero(m))
            counts[f"{name}_{part}_n"] = n
            keys += [f"{name}_{part}_{k}" for k in REG_STATS]
            if n == 0:
                vals += [nan] * len(REG_STATS)
                continue
            mt = torch.from_numpy(np.ascontiguousarray(m)).to(s.device)
            v = cm[mt]
            ss, tt = s[:, mt], t[:, mt]
            vals += [v.mean(), torch.quantile(v, 0.10), (v < COS_LOW).float().mean(), ss.std() / tt.std()]
    out = dict(zip(keys, torch.stack([x.float() for x in vals]).cpu().tolist()))
    out.update(counts)
    return out, cm.detach().half().cpu().numpy()


def xcheck(gpu, host, what):
    bad = xcheck_bad(gpu, host)
    if bad:
        raise core.BenchAssertError(f"GPU vs float64 host metrics differ ({what}): {dict(list(bad.items())[:6])}")


def load_target_file(p):
    t = torch.load(p, map_location="cpu", weights_only=True)
    if tuple(t.shape) != tuple(dc.TARGET_SHAPE) or t.dtype != torch.float16:
        raise RuntimeError(f"{p}: {tuple(t.shape)} {t.dtype}")
    return t


def fidelity_loader(clip, stored_dir=None, new_feat=None):
    """(fi, pil, descriptors, stored fp16 | None, new fp16 | None) in order; decoding on a prefetch thread."""
    def gen():
        prev = None
        for fi in range(NF):
            with Image.open(osp.join(FRAMES_DIR[clip], frame_name(fi))) as im:
                pil = im.convert("RGB")
            desc, prev = frame_descriptors(np.asarray(pil), prev)
            st = load_target_file(osp.join(stored_dir, target_name(fi))) if stored_dir else None
            nw = load_target_file(new_feat[(clip, fi)]) if new_feat is not None else None
            yield fi, pil, desc, st, nw
    return core.prefetch(gen(), qsize=6, name=f"tol79_fid_{clip}")


def regions_for(clip, fi, gridT=None, gridS=None):
    """-> ({name: (pig cells, background cells)}, {name: coverage grid}) for the regions available on this frame."""
    covs = {}
    if clip in GT_BY_F:
        covs["gt"] = coverage_grid(rasterize_boxes([GT_BY_F[clip][fi][k] for k in sorted(GT_BY_F[clip][fi])]))
    if gridT is not None:
        covs["t3"] = gridT
    if gridS is not None:
        covs["s3"] = gridS
    if fi in TXT.get(clip, {}):
        covs["txt"] = TXT[clip][fi]["grid"]
    return {k: region_masks(v) for k, v in covs.items()}, covs


def encode_grid(g):
    """coverage [72, 72] -> uint8 round(cov * 254); 255 = region not available on this frame."""
    return np.full((GRID, GRID), 255, np.uint8) if g is None else np.round(np.asarray(g) * 254).astype(np.uint8)


def txt_columns(clip, fi, regs):
    tx = TXT.get(clip, {}).get(fi)
    if tx is None:
        return {}
    out = {"txt_n": tx["n"], "txt_max_box_iou": tx["crowd"]["max_iou"], "txt_n_touch": tx["crowd"]["n_touch"]}
    if clip in GT_BY_F and GT_STATUS[clip] == "trusted":
        gtb = [[x, y, x + w, y + h] for (x, y, w, h) in (GT_BY_F[clip][fi][k] for k in sorted(GT_BY_F[clip][fi]))]
        out["txt_gt_matched"] = len(core.match_by_iou(tx["boxes"], gtb, 0.5))
    if "t3" in regs and "txt" in regs:                       # pig cells of the text prompt vs pig cells of T's masks
        a, b = regs["txt"][0], regs["t3"][0]
        un = int((a | b).sum())
        out["txt_t3_cell_iou"] = float((a & b).sum() / un) if un else float("nan")
    return out


FID_ROWS = []           # one row per (clip, fi), all 8 clips
XCHECKS = []
DETERMINISM = []
FIDE_ROWS = []
FRESH_VS_STORED = []
PIG_ROWS, TRACK_FRAME_ROWS, EVENT_ROWS = [], [], []
T_REPRO = {}
CLIP_TIMES = {}


def fidelity_pass(clip, fresh=None, stored_dir=None, new_feat=None, perT=None, perS=None, dummy_caps=None):
    """Student (bf16, the bench's S-Edi) on every frame; metrics vs the fresh and (benchmark clips) the stored teacher."""
    cos_f, cos_s = [], []
    grids = {k: [] for k in REGIONS}
    t0 = time.time()
    n_x = 0
    for fi, pil, desc, stored, new in fidelity_loader(clip, stored_dir, new_feat):
        x = ARM_S.student_input(pil)
        with torch.no_grad():
            sf = ARM_S.student(x)
        if not bool(torch.isfinite(sf).all()):
            raise core.BenchAssertError(f"{clip} f{fi}: non-finite student features")
        s = sf[0].float()
        gT = perT[fi]["grid"] if perT is not None else None
        gS = perS[fi]["grid"] if perS is not None else None
        regs, covs = regions_for(clip, fi, gT, gS)
        for k in REGIONS:
            grids[k].append(encode_grid(covs.get(k)))
        row = {"clip": clip, "role": ROLE[clip], "fi": fi, "gt_status": GT_STATUS[clip]}
        row.update(desc)
        row.update(txt_columns(clip, fi, regs))
        f_t = fresh[fi] if fresh is not None else new
        tf = f_t[0].float().to(DEV)
        mf, cmf = fid_torch(s, tf, regs)
        row.update({f"f_{k}": v for k, v in mf.items()})
        cos_f.append(cmf)
        if stored is not None:
            ts = stored[0].float().to(DEV)
            ms, cms = fid_torch(s, ts, regs)
            row.update({f"s_{k}": v for k, v in ms.items()})
            cos_s.append(cms)
            agree = {"cos_global": float(((tf * ts).sum() / (tf.norm() * ts.norm())).item()),
                     "max_abs": float((tf - ts).abs().max().item()),
                     "frac_equal": float((f_t.reshape(-1) == stored.reshape(-1)).float().mean())}
            row.update({f"fs_{k}": v for k, v in agree.items()})
            FRESH_VS_STORED.append(dict(clip=clip, fi=fi, **agree))
        if n_x < XCHECK_N:                             # hard gate: GPU float32 == float64 host (fid_np, fidelity_metrics)
            s64 = sf[0].float().cpu().numpy()
            hn, _ = fid_np(s64, f_t.float().numpy(), regs)
            xcheck(mf, hn, f"{clip} f{fi} fresh")
            ref = dc.fidelity_metrics(s64, f_t.float().numpy())
            for a, b in (("cos_mean", "cos_patch"), ("cos_global", "cos_global"), ("scale_ratio", "scale_ratio"),
                         ("mse", "mse"), ("cos_p50", "cos_patch_p50")):
                if abs(hn[a] - ref[b]) > 1e-9 * max(1.0, abs(ref[b])):
                    raise core.BenchAssertError(f"fid_np {a} != fidelity_metrics {b}: {hn[a]} {ref[b]}")
            XCHECKS.append({"clip": clip, "fi": fi, "max_abs_diff": max(abs(mf[k] - hn[k]) for k in hn if k in mf
                                                                        and not math.isnan(hn[k]))})
            n_x += 1
        if dummy_caps is not None and fi in dummy_caps:     # tracker's student features == this pass's (bf16)
            d = dummy_caps[fi].float()
            DETERMINISM.append({"clip": clip, "fi": fi, "max_abs": float((d - sf.float().cpu()).abs().max()),
                                "equal": bool(torch.equal(dummy_caps[fi], sf.cpu()))})
        if S32 is not None and stored is not None and (
                (clip == EDI_VAL[0] and fi % 5 == 0) or (clip == EDI_TRAIN[0] and fi < 20)):
            with torch.no_grad():
                y32 = S32(S32_TF(pil).unsqueeze(0).to(DEV)).float().cpu().numpy()
            m32 = dc.fidelity_metrics(y32, stored.float().numpy())
            FIDE_ROWS.append(dict(row_id=("val/" if clip == EDI_VAL[0] else "train20/") + clip, clip=clip, fi=fi,
                                  **{k: m32[k] for k in ("cos_patch", "cos_global", "scale_ratio", "mse")}))
        FID_ROWS.append(row)
        if (fi + 1) % P.progress_every == 0 or fi + 1 == NF:
            _log(f"  fidelity {clip}: {fi + 1}/{NF}, {time.time() - t0:.0f} s, mean cos (fresh) "
                 f"{np.mean([r['f_cos_mean'] for r in FID_ROWS if r['clip'] == clip]):.4f}")
    p = osp.join(LOUT, "clips", clip, "maps.npz")
    arrs = {"cos_fresh": np.stack(cos_f)}
    if cos_s:
        arrs["cos_stored"] = np.stack(cos_s)
    for k, v in grids.items():
        arrs[f"region_{k}"] = np.stack(v)
    np.savez_compressed(p, **arrs)
    publish(p, f"clips/{clip}/maps.npz")
    return round(time.time() - t0, 1)


def compare_tracks(clip, perT, perS):
    """Per frame and pig: S-Edi vs T mask IoU, swaps relative to T, box IoU vs the interpolated GT (frozen offset)."""
    pigs = sorted(KF[clip])
    series = {k: [] for k in pigs}
    rows_pf = []
    for fi in range(NF):
        T, S = perT[fi], perS[fi]
        gt = GT_BY_F[clip][fi]
        gtb = {k: [gt[k][0], gt[k][1], gt[k][0] + gt[k][2], gt[k][1] + gt[k][3]] for k in gt}
        fr = {"clip": clip, "fi": fi, "n_T": len(T["rows"]), "n_S": len(S["rows"])}
        pT, pS = pairwise_crop_stats(T["crops"]), pairwise_crop_stats(S["crops"])
        pG = pairwise_box_stats([gtb[k] for k in sorted(gtb)])
        fr.update({"t3_max_iou": pT["max_iou"], "t3_max_ios": pT["max_ios"], "t3_n_touch": pT["n_touch"],
                   "s3_max_iou": pS["max_iou"], "s3_max_ios": pS["max_ios"], "s3_n_touch": pS["n_touch"],
                   "gt_max_iou": pG["max_iou"], "gt_n_touch": pG["n_touch"], "gt_n_pairs_iou01": pG["n_pairs_iou01"]})
        ious, nswap, nlow = [], 0, 0
        for k in pigs:
            rT, rS = T["rows"].get(k), S["rows"].get(k)
            cT, cS = T["crops"].get(k), S["crops"].get(k)
            r = {"clip": clip, "fi": fi, "pig": k, "T_present": int(rT is not None), "S_present": int(rS is not None)}
            iou = float("nan")
            if cT is not None:
                iou, ios, _ = crop_overlap(cT, cS) if cS is not None else (0.0, 0.0, 0)
                r["iou_ST"], r["ios_ST"] = iou, ios
            best_j, best = None, 0.0
            if cS is not None:
                for j, cj in T["crops"].items():
                    if cj[0] <= cS[2] and cS[0] <= cj[2] and cj[1] <= cS[3] and cS[1] <= cj[3]:
                        v = crop_overlap(cS, cj)[0]
                        if v > best:
                            best_j, best = j, v
            r["S_best_T"], r["S_best_T_iou"] = best_j, best
            swap = bool(cS is not None and best_j is not None and best_j != k and best >= SWAP_IOU
                        and not (iou >= IOU_LOW))
            r["swap_vs_T"] = int(swap)
            for tag, rr in (("T", rT), ("S", rS)):
                if rr is not None:
                    x0, y0, x1, y1, area, score = rr
                    r.update({f"{tag}_x0": x0, f"{tag}_y0": y0, f"{tag}_x1": x1, f"{tag}_y1": y1, f"{tag}_area": area,
                              f"{tag}_score": score})
                    b = [x0 + XY_O, y0 + XY_O, x1 + 1 + XY_O, y1 + 1 + XY_O]
                    if k in gtb:
                        r[f"{tag}_gt_iou"] = core.box_iou(b, gtb[k])
                    best_g = max(((core.box_iou(b, gb), g) for g, gb in gtb.items()), default=(0.0, None))
                    r[f"{tag}_best_gt"], r[f"{tag}_best_gt_iou"] = best_g[1], best_g[0]
            if k in gt:
                r.update({"gt_x": gt[k][0], "gt_y": gt[k][1], "gt_w": gt[k][2], "gt_h": gt[k][3]})
            rows_pf.append(r)
            series[k].append(iou)
            if not math.isnan(iou):
                ious.append(iou)
                nlow += iou < IOU_LOW
            nswap += swap
        fr.update({"mean_iou_ST": float(np.mean(ious)) if ious else float("nan"),
                   "min_iou_ST": float(np.min(ious)) if ious else float("nan"), "n_iou_lt05": nlow, "n_swap": nswap})
        TRACK_FRAME_ROWS.append(fr)
    PIG_ROWS.extend(rows_pf)
    by = {(r["fi"], r["pig"]): r for r in rows_pf}
    for k in pigs:
        for (a, b) in runs_below(series[k], IOU_LOW, SUSTAIN_MIN):
            seg = [by[(fi, k)] for fi in range(a, b + 1)]
            fa = float(np.mean([1 - r["S_present"] for r in seg]))
            fs = float(np.mean([r["swap_vs_T"] for r in seg]))
            EVENT_ROWS.append({"clip": clip, "pig": k, "start_fi": a, "end_fi": b, "n_frames": b - a + 1,
                               "long": int(b - a + 1 >= SUSTAIN_LONG), "min_iou_ST": float(np.min(series[k][a:b + 1])),
                               "mean_iou_ST": float(np.mean(series[k][a:b + 1])), "frac_S_absent": fa, "frac_swap": fs,
                               "kind": classify_event(fa, fs),
                               "swap_partner": collections.Counter(r["S_best_T"] for r in seg
                                                                   if r["swap_vs_T"]).most_common(1)[0][0]
                               if fs > 0 else None,
                               "mean_T_gt_iou": describe([r.get("T_gt_iou") for r in seg])["mean"]
                               if describe([r.get("T_gt_iou") for r in seg]) else None,
                               "mean_S_gt_iou": describe([r.get("S_gt_iou") for r in seg])["mean"]
                               if describe([r.get("S_gt_iou") for r in seg]) else None})
    return series


def j4_reference(clip):
    base = R.path("paper1.bench", None, "edinburgh", "T__protA", "all")
    mot = read_bytes(guard_path(f"{base}/mot/{clip}.txt"))
    tele = json.loads(read_bytes(guard_path(f"{base}/telemetry/{clip}.json")).decode("utf-8"))
    return mot, tele


# ---- the 5 benchmark non-test clips ----------------------------------------------------------------------------------------
for clip in TRACK_CLIPS:
    tc = time.time()
    kf = KF[clip]
    prompts = core.protocol_a_edinburgh(kf)
    old = core.read_bbox_prompts_txt(SEED_LOCAL[clip])
    if {p.obj_id: list(p.box) for p in prompts} != old:
        raise core.BenchAssertError(f"{clip}: seeded run prompts differ from the published seeds")
    # T (teacher): fresh backbone features captured on every frame (observe-only hook)
    cap = BackboneFeatureHook(ARM_T.model)
    try:
        perT, fresh, stT, telT, motT, rleT = track_arm(ARM_T, clip, prompts, "T", capture_hook=cap)
    finally:
        cap.remove()
    core.settle_memory()
    dcap = DummyCapture(ARM_S.model.vision_encoder.backbone)
    try:
        perS, dcaps, stS, telS, motS, rleS = track_arm(ARM_S, clip, prompts, "S", capture_hook=dcap, capture_every=100)
    finally:
        dcap.remove()
    core.settle_memory()
    # reproduction control: T's MOT bytes / frame hashes vs J4 (same core, same GPU class)
    j4_mot, j4_tel = j4_reference(clip)
    with open(motT, "rb") as f:
        mine = f.read()
    rep = {"j4_sha256_expected": J4_T_MOT_SHA256[clip], "j4_sha256_read": hashlib.sha256(j4_mot).hexdigest(),
           "mine_sha256": hashlib.sha256(mine).hexdigest(), "frames": NF}
    rep["bytes_identical"] = rep["mine_sha256"] == rep["j4_sha256_read"] == rep["j4_sha256_expected"]
    rep["rows"] = mot_compare(mine.decode("utf-8"), j4_mot.decode("utf-8"), max_frame=NF)
    jh = {int(a): b for a, b in (j4_tel.get("stats") or {}).get("frame_hashes", [])}
    mh = {int(a): b for a, b in stT["frame_hashes"]}
    rep["frame_hashes"] = {str(fi): {"mine": mh[fi], "j4": jh.get(fi), "equal": mh[fi] == jh.get(fi)} for fi in mh}
    rep["j4_wall_s"] = (j4_tel.get("stats") or {}).get("wall_s")
    rep["mine_wall_s"] = stT["wall_s"]
    T_REPRO[clip] = rep
    if not rep["rows"]["identical"]:
        NOTES.append(f"{clip}: T rows differ from J4 over frames <= {NF}: {rep['rows']}")
    _log(f"{clip}: T vs J4 rows identical {rep['rows']['identical']} (bytes identical {rep['bytes_identical']}); "
         f"T {stT['wall_s']} s, S-Edi {stS['wall_s']} s")
    # control: Sam3Teacher's 20 control features vs the T-capture of the same frames (same pipeline, prompts differ)
    for (c, fi), feat in CONTROL_FEATS.items():
        if c == clip and fi in fresh:
            DETERMINISM.append({"clip": clip, "fi": fi, "what": "Sam3Teacher (dummy boxes) vs T-capture (GT prompts)",
                                **compare_features(feat, fresh[fi])})
    compare_tracks(clip, perT, perS)
    fut = STORED_FUT[clip]
    STAGE[f"targets_{clip}"] = fut.result()
    CLIP_TIMES[clip] = {"T_wall_s": stT["wall_s"], "S_wall_s": stS["wall_s"]}
    CLIP_TIMES[clip]["fidelity_s"] = fidelity_pass(clip, fresh=fresh, stored_dir=STORED_DIR[clip], perT=perT,
                                                   perS=perS, dummy_caps=dcaps)
    for tag, st, ts, mp, rp in (("T", stT, telT, motT, rleT), ("S", stS, telS, motS, rleS)):
        tp = write_json_local(osp.join(LOUT, "clips", clip, f"telemetry_{tag}.json"),
                              {"clip": clip, "arm": "T" if tag == "T" else "S-Edi", "stats": st, "telemetry": ts,
                               "notebook": NB_VERSION, "settings": {"keep_last_n": core.KEEP_LAST_N,
                                                                     "prune_every": core.PRUNE_EVERY,
                                                                     "pop": core.POP_FRAMES, "protocol": "A"}})
        for lp in (mp, rp, tp):
            publish(lp, f"clips/{clip}/{osp.basename(lp)}")
    del perT, perS, fresh, dcaps
    gc.collect()
    CLIP_TIMES[clip]["total_s"] = round(time.time() - tc, 1)
    _log(f"{clip} done in {CLIP_TIMES[clip]['total_s']} s")

# ---- the 3 unused clips: fidelity vs the new targets (= fresh teacher) -----------------------------------------------------
for clip in UNUSED:
    tc = time.time()
    CLIP_TIMES[clip] = {"fidelity_s": fidelity_pass(clip, new_feat=TT_LOCAL_FEAT)}
    CLIP_TIMES[clip]["total_s"] = round(time.time() - tc, 1)
TIMINGS["phase_track_fidelity_s"] = round(time.time() - t_ph, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tables, controls, teacher-target manifest, run record (published last)

# COMMAND ----------

t_ph = time.time()
# ---- per-frame tables ------------------------------------------------------------------------------------------------------
TXT_COLS = ["txt_n", "txt_max_box_iou", "txt_n_touch", "txt_gt_matched", "txt_t3_cell_iou"]
TF_BY = {(r["clip"], r["fi"]): r for r in TRACK_FRAME_ROWS}
for r in FID_ROWS:
    clip, fi = r["clip"], r["fi"]
    if (clip, fi) in TF_BY:
        for k in ("t3_max_iou", "t3_max_ios", "t3_n_touch", "s3_max_iou", "s3_max_ios", "s3_n_touch", "gt_max_iou",
                  "gt_n_touch", "gt_n_pairs_iou01", "mean_iou_ST", "min_iou_ST", "n_iou_lt05", "n_swap", "n_T", "n_S"):
            r[k] = TF_BY[(clip, fi)][k]
    elif clip in GT_BY_F:                                      # GT crowding of the unverified unused clips (flagged)
        gtb = [[x, y, x + w, y + h] for (x, y, w, h) in (GT_BY_F[clip][fi][k] for k in sorted(GT_BY_F[clip][fi]))]
        pg = pairwise_box_stats(gtb)
        r["gt_max_iou"], r["gt_n_touch"], r["gt_n_pairs_iou01"] = pg["max_iou"], pg["n_touch"], pg["n_pairs_iou01"]
DESC_COLS = ["brightness", "contrast", "saturation", "motion", "sharpness"]
FID_STAT_COLS = list(GLOBAL_STATS) + [f"{reg}_{part}_{k}" for reg in REGIONS for part in ("fg", "bg")
                                      for k in REG_STATS + ("n",)]
FID_HEADER = (["clip", "role", "fi", "gt_status"] + DESC_COLS + [f"f_{k}" for k in FID_STAT_COLS]
              + [f"s_{k}" for k in FID_STAT_COLS] + ["fs_cos_global", "fs_max_abs", "fs_frac_equal"]
              + ["n_T", "n_S", "mean_iou_ST", "min_iou_ST", "n_iou_lt05", "n_swap", "t3_max_iou", "t3_max_ios",
                 "t3_n_touch", "s3_max_iou", "s3_max_ios", "s3_n_touch", "gt_max_iou", "gt_n_touch", "gt_n_pairs_iou01"]
              + TXT_COLS)
PIG_HEADER = ["clip", "fi", "pig", "T_present", "S_present", "iou_ST", "ios_ST", "S_best_T", "S_best_T_iou", "swap_vs_T",
              "T_x0", "T_y0", "T_x1", "T_y1", "T_area", "T_score", "S_x0", "S_y0", "S_x1", "S_y1", "S_area", "S_score",
              "gt_x", "gt_y", "gt_w", "gt_h", "T_gt_iou", "S_gt_iou", "T_best_gt", "T_best_gt_iou", "S_best_gt",
              "S_best_gt_iou"]
TF_HEADER = ["clip", "fi", "n_T", "n_S", "mean_iou_ST", "min_iou_ST", "n_iou_lt05", "n_swap", "t3_max_iou", "t3_max_ios",
             "t3_n_touch", "s3_max_iou", "s3_max_ios", "s3_n_touch", "gt_max_iou", "gt_n_touch", "gt_n_pairs_iou01"]
EV_HEADER = ["clip", "pig", "start_fi", "end_fi", "n_frames", "long", "min_iou_ST", "mean_iou_ST", "frac_S_absent",
             "frac_swap", "kind", "swap_partner", "mean_T_gt_iou", "mean_S_gt_iou"]
TABLES = {"tables/per_frame_fidelity.csv": (FID_HEADER, FID_ROWS), "tables/per_pig_frame_track.csv": (PIG_HEADER, PIG_ROWS),
          "tables/per_frame_track.csv": (TF_HEADER, TRACK_FRAME_ROWS), "tables/events.csv": (EV_HEADER, EVENT_ROWS),
          "controls/fresh_vs_stored_teacher.csv": (["clip", "fi", "cos_global", "max_abs", "frac_equal"], FRESH_VS_STORED),
          "controls/gpu_vs_host_xcheck.csv": (["clip", "fi", "max_abs_diff"], XCHECKS),
          "controls/determinism.csv": (["clip", "fi", "what", "max_abs", "equal", "cos_global", "mean_abs", "frac_equal",
                                        "bit_identical"], DETERMINISM),
          "controls/fide_fp32_replication.csv": (["row_id", "clip", "fi", "cos_patch", "cos_global", "scale_ratio", "mse"],
                                                 FIDE_ROWS)}
for rel, (hdr, rows) in TABLES.items():
    publish(write_csv(osp.join(LOUT, rel), hdr, rows), rel)
publish(osp.join(LOUT, "controls", "teacher_recompute_20.csv"), "controls/teacher_recompute_20.csv")

# ---- per-clip main + controls summary ----------------------------------------------------------------------------------
SUMMARY = {}
for clip in CLIPS:
    fr = [r for r in FID_ROWS if r["clip"] == clip]
    d = {"role": ROLE[clip], "gt_status": GT_STATUS[clip], "frames": len(fr),
         "fresh": {k: describe([r.get(f"f_{k}") for r in fr]) for k in GLOBAL_STATS}}
    if clip in TRACK_CLIPS:
        d["stored"] = {k: describe([r.get(f"s_{k}") for r in fr]) for k in GLOBAL_STATS}
        pr = [r for r in PIG_ROWS if r["clip"] == clip]
        ious = [r.get("iou_ST") for r in pr]
        d["track"] = {"iou_ST": describe(ious),
                      "share_pig_frames_iou_lt05": float(np.mean([v < IOU_LOW for v in ious if v is not None])),
                      "pig_frames_swap_vs_T": int(sum(r["swap_vs_T"] for r in pr)),
                      "S_absent_pig_frames": int(sum(1 - r["S_present"] for r in pr)),
                      "T_absent_pig_frames": int(sum(1 - r["T_present"] for r in pr)),
                      "events": len([e for e in EVENT_ROWS if e["clip"] == clip]),
                      "events_long": len([e for e in EVENT_ROWS if e["clip"] == clip and e["long"]]),
                      "events_by_kind": dict(collections.Counter(e["kind"] for e in EVENT_ROWS if e["clip"] == clip)),
                      "T_gt_iou": describe([r.get("T_gt_iou") for r in pr]),
                      "S_gt_iou": describe([r.get("S_gt_iou") for r in pr]),
                      "T_repro_rows_identical": T_REPRO[clip]["rows"]["identical"],
                      "T_repro_bytes_identical": T_REPRO[clip]["bytes_identical"]}
        for reg in ("gt", "t3", "txt"):
            d[f"fresh_{reg}"] = {part: describe([r.get(f"f_{reg}_{part}_cos") for r in fr]) for part in ("fg", "bg")}
    else:
        for reg in ("gt", "txt"):
            d[f"fresh_{reg}"] = {part: describe([r.get(f"f_{reg}_{part}_cos") for r in fr]) for part in ("fg", "bg")}
    d["times"] = CLIP_TIMES.get(clip)
    SUMMARY[clip] = d
FIDE_SUMMARY = {}
for rid, ref in FIDE.items():
    sel = [r for r in FIDE_ROWS if r["row_id"] == rid]
    if not sel:
        continue
    ours = {k: float(np.mean([r[k] for r in sel])) for k in ref}
    FIDE_SUMMARY[rid] = {"n": len(sel), "complete": len(sel) == (120 if rid.startswith("val/") else 20),
                         "comparison": fide_compare(ours, ref)}
CONTROLS = {"teacher_recompute_20": CONTROL_SUMMARY, "prompt_independence": PI,
            "fresh_vs_stored_all_benchmark_frames": {k: describe([r[k] for r in FRESH_VS_STORED])
                                                     for k in ("cos_global", "max_abs", "frac_equal")},
            "sam3teacher_vs_T_capture": [d for d in DETERMINISM if "what" in d],
            "student_tracker_vs_fidelity_pass": [d for d in DETERMINISM if "what" not in d],
            "gpu_vs_host_xcheck": {"n": len(XCHECKS), "max_abs_diff": max([x["max_abs_diff"] for x in XCHECKS], default=None),
                                   "tolerance": XCHECK_TOL},
            "fide_fp32_replication": FIDE_SUMMARY, "t_reproduction_vs_j4": T_REPRO,
            "test_ids": {"count": len(_TEST_IDS), "sha256_of_sorted_ids": TEST_IDS_DIGEST,
                         "rule": "every path helper refuses them; every text output scanned before publishing"}}
try:
    BG_INFO = BG_FUT.result()
    for p in BG.values():
        publish(p, f"bg_median/{osp.basename(p)}")
except Exception as e:  # noqa: BLE001
    BG_INFO = None
    ERRORS.append(f"(background medians) {type(e).__name__}: {e}")

# ---- teacher-target manifest (after every clip's tree and record are on the Volume) -----------------------------------------
tt_err = []
for clip, f in TT_FUTS.items():
    try:
        f.result()
    except Exception as e:  # noqa: BLE001
        tt_err.append((clip, e))
TT_PUB.shutdown(wait=True)
if tt_err:
    raise RuntimeError(f"teacher-target publishing failed for {[c for c, _ in tt_err]}") from tt_err[0][1]
TT_MAN = dc.manifest_rows(TT_RECORDS, {c: TT_INFO for c in UNUSED}, [])
TT_MAN_SUMMARY = dc.validate_manifest(TT_MAN, plan={c: list(range(NF)) for c in UNUSED}, val_seqs=[],
                                      n_targets=NF * len(UNUSED))
def _vol_bad(r):
    out = []
    for k, n in (("feature", r["feature_bytes"]), ("image", r["image_bytes"])):
        p = f"{TT_OUT}/{r[k]}"
        if not osp.isfile(p) or osp.getsize(p) != n:
            out.append(r[k])
    return out


with ThreadPoolExecutor(max_workers=16) as _ex:
    _missing = [p for lst in _ex.map(_vol_bad, TT_MAN) for p in lst]
if _missing:
    raise RuntimeError(f"{len(_missing)} teacher-target files missing / wrong size on the Volume, e.g. {_missing[0]}")
for r in TT_MAN[:: max(1, len(TT_MAN) // 3)][:3]:
    if core.sha256_file(f"{TT_OUT}/{r['feature']}") != r["feature_sha256"]:
        raise RuntimeError(f"Volume copy of {r['feature']} differs from its record")
_lp = core.write_text_atomic(osp.join(LOCAL, "tt", "manifest.csv"), dc.manifest_csv(TT_MAN))
scan_file(_lp)
core.publish_no_overwrite(_lp, f"{TT_OUT}/manifest.csv", dbutils)
TT_MAN_SHA = core.sha256_file(_lp)
_ts = {"manifest": TT_MAN_SUMMARY, "manifest_sha256": TT_MAN_SHA, "per_clip": {
    c: {"n": TT_RECORDS[c]["n"], "dir": TT_RECORDS[c]["dir"], "extract_seconds": TT_RECORDS[c]["extract_seconds"],
        "feat_mean_of_stds": TT_RECORDS[c]["feat_mean_of_stds"], "gt_status": GT_STATUS[c]} for c in UNUSED},
    "prompt_independence": PI, "control_vs_stored": CONTROL_SUMMARY, "nb": NB_VERSION, "finished_utc": _now()}
_lp = write_json_local(osp.join(LOCAL, "tt", "TARGETS_SUMMARY.json"), _ts)
scan_file(_lp)
core.publish_no_overwrite(_lp, f"{TT_OUT}/TARGETS_SUMMARY.json", dbutils)
_stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:6]
_lp = write_json_local(osp.join(LOCAL, "tt", "runs", f"{_stamp}.json"),
                       {"nb": NB_VERSION, "code": CODE_SHAS, "gpu": GPU, "extract_seconds": TT_EXTRACT_S,
                        "publish": {c: TT_RECORDS[c]["publish"] for c in UNUSED}, "manifest_sha256": TT_MAN_SHA})
scan_file(_lp)
core.publish_no_overwrite(_lp, f"{TT_OUT}/runs/{_stamp}.json", dbutils)
_log(f"teacher targets: {TT_MAN_SUMMARY['rows']} rows, {TT_MAN_SUMMARY['bytes'] / 1e9:.2f} GB, manifest {TT_MAN_SHA[:12]}")

# ---- wait for every background publish, then the run record (LAST = done marker) -------------------------------------------
PUB.shutdown(wait=True)
_pub_err = [f.exception() for f in PUB_FUTS if f.exception() is not None]
if _pub_err:
    raise RuntimeError(f"{len(_pub_err)} background publish(es) failed") from _pub_err[0]
STAGER.shutdown(wait=True)
TIMINGS["finalize_s"] = round(time.time() - t_ph, 1)
TIMINGS["total_s"] = round(time.time() - T0, 1)
RECORD = {"notebook": NB_NAME, "nb_version": NB_VERSION, "mode": P.mode, "frames_per_clip": NF, "widgets": RAW,
          "created_utc": _now(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0)),
          "out": OUT, "targets": TT_OUT, "clips": {c: {"role": ROLE[c], "gt_status": GT_STATUS[c]} for c in CLIPS},
          "code": CODE_SHAS, "pip": PIP_SPEC, "versions": VERSIONS, "gpu": GPU,
          "registry": {"file": R.source, "sha256": R.sha256}, "revisions": REVISIONS, "mot_xy_offset": XY_O,
          "inputs": {"gt_sha256": GT_SHA, "staging": STAGE, "student_ckpt_sha256": CKPT_SHA256},
          "teacher": TEACHER_INFO, "text_prompt": TXT_INFO, "arms": ARMS_INFO,
          "settings": {"keep_last_n": core.KEEP_LAST_N, "prune_every": core.PRUNE_EVERY, "pop": core.POP_FRAMES,
                       "dtype": core.DTYPE_NAME, "cos_low": COS_LOW, "iou_low": IOU_LOW, "sustain_min": SUSTAIN_MIN,
                       "sustain_long": SUSTAIN_LONG, "swap_iou": SWAP_IOU, "fg_cov": FG_COV, "grid": GRID,
                       "region_grid_encoding": "uint8 round(coverage * 254); 255 = region not available on that frame",
                       "student_precision": "bf16 (bench_core.load_arm('S-Edi'), the bench's student and transform)",
                       "metric_precision": "GPU float32, cross-checked against float64 host code on the first "
                                           f"{XCHECK_N} frames of every clip (tolerance {XCHECK_TOL})"},
          "background_medians": BG_INFO, "controls": CONTROLS, "summary": SUMMARY, "timings_s": TIMINGS,
          "teacher_targets": {"manifest": TT_MAN_SUMMARY, "manifest_sha256": TT_MAN_SHA},
          "published": PUBLISHED, "notes": NOTES, "errors": ERRORS}
_lp = write_json_local(osp.join(LOUT, "RUN_RECORD.json"), RECORD)
_publish_now(_lp, "RUN_RECORD.json")
_log(f"published {len(PUBLISHED)} files to {OUT}; total {TIMINGS['total_s']} s; errors {ERRORS}")
HEAD = {c: {"cos_fresh": (SUMMARY[c]["fresh"]["cos_mean"] or {}).get("mean"),
            "cos_stored": ((SUMMARY[c].get("stored") or {}).get("cos_mean") or {}).get("mean"),
            "iou_ST": (((SUMMARY[c].get("track") or {}).get("iou_ST")) or {}).get("mean")} for c in CLIPS}
_log("main", json.dumps(HEAD, default=str))
if ERRORS:          # everything is published; the task still fails so a smoke error stops the full task
    raise RuntimeError(f"measure_edinburgh_hard_frames finished with errors (results published to {OUT}): {ERRORS}")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"ok": not ERRORS, "mode": P.mode, "out": OUT, "targets": TT_OUT, "main": HEAD,
                                  "t_repro_rows_identical": {c: T_REPRO[c]["rows"]["identical"] for c in T_REPRO},
                                  "fide": {k: {m: v["within"] for m, v in s["comparison"].items()}
                                           for k, s in FIDE_SUMMARY.items()},
                                  "errors": ERRORS, "total_s": TIMINGS["total_s"]}, default=str))
