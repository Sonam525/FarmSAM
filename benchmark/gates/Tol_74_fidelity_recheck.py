# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N74 — independent re-check of the Edinburgh feature fidelity [FID-E] (Paper 1, one **A10**)
# MAGIC
# MAGIC BENCH_PROTOCOL b1.5 item 2(a) (user, 2026-10-22: "Check that"); pre-registration `bench/_s15_pending/fid_prereg.md`.
# MAGIC [FID-E] = `Tol_68` tag `v1_fidelity`, stage (d): the published Edinburgh student (S-Edi, EMA epoch 75) against the
# MAGIC stored SAM 3 teacher targets on 620 frames (4 test clips at stride 5, the val clip at stride 5, the first 20 frames of
# MAGIC training clip 0). This notebook recomputes it with its OWN code: frame indexing and decoding, target reading, the
# MAGIC student build (architecture copied from the exported Fixed notebook, Cell 7, with `drop_path_rate=0.0`) and checkpoint
# MAGIC load, the input transform and the metrics. Nothing is imported from `Tol_68`, `tol_bench_core` or `tol_distill_core`;
# MAGIC only the path registry `tol_paths` (deployed RAW beside this notebook) and the `hf/token` secret are used.
# MAGIC
# MAGIC | row | teacher | student numerics | gated |
# MAGIC |---|---|---|---|
# MAGIC | **primary** | stored targets (`paper1.teacher_targets.edinburgh`) | fp32; matmul TF32 off, cuDNN TF32 on (as [FID-E] recorded) | yes: cosines \|Δ\| ≤ 0.005, scale ratio and MSE relative \|Δ\| ≤ 1 % |
# MAGIC | **B** | stored targets | fp32; cuDNN TF32 also off | no |
# MAGIC | **A** | FRESH: own forward of SAM 3 (`facebook/sam3` @ pinned commit, bf16) + own hook on `vision_encoder.backbone`, cast to fp16 | as primary | no |
# MAGIC
# MAGIC Outputs (new folder `paper1_distillation/evidence/edinburgh_audit/<tag>/`, refused if it exists; written once, at the
# MAGIC end): `RECHECK_CONFIG.json`, `fid_recheck.json` (rows, comparison, verdict), `per_frame.csv`,
# MAGIC `teacher_fresh_vs_stored.csv`. Local check (numpy only): `TOL74_SELFTEST=1 python Tol_74_fidelity_recheck.py`.

# COMMAND ----------

# ---- requirements for the %pip cell (FID-E's pins, minus the tracking-only scipy / motmetrics) -------------------------
import os as _os
_TOL74_REQ = "/local_disk0/tol74_env/requirements.txt"
_TOL74_PINS = ("numpy<2.0", "transformers==5.2.0", "Pillow==11.1.0", "huggingface_hub", "accelerate", "timm==1.0.25")
try:
    dbutils  # noqa: F821,B018
    _os.makedirs(_os.path.dirname(_TOL74_REQ), exist_ok=True)
    with open(_TOL74_REQ, "w", encoding="utf-8", newline="\n") as _f:
        _f.write("\n".join(_TOL74_PINS) + "\n")
    print(f"%pip requirements -> {_TOL74_REQ}")
except NameError:
    pass   # local selftest

# COMMAND ----------

# MAGIC %pip install -r /local_disk0/tol74_env/requirements.txt

# COMMAND ----------

try:
    dbutils.library.restartPython()  # noqa: F821
except NameError:
    pass

# COMMAND ----------

import csv
import gc
import hashlib
import io
import json
import math
import os
import os.path as osp
import re
import sys
import time
import traceback

import numpy as np

NB_VERSION = "tol74-2026-10-22.1"
IS_NB = True
try:
    dbutils  # noqa: F821,B018
except NameError:
    IS_NB = False
SELFTEST = os.environ.get("TOL74_SELFTEST") == "1"

# ---- pre-registered constants (fid_prereg.md; b1.5 item 2(a)) --------------------------------------------------------
EDI_TEST = ("2019_12_02_000005", "2019_12_02_000208", "2019_12_10_000060", "2019_12_10_000078")
EDI_VAL = "2019_11_28_000113"
EDI_TRAIN0 = "2019_11_05_000002"
CLIP_FRAMES = 600            # frames 0..599 of each clip are in [FID-E]'s plan
STRIDE = 5
TRAIN_BRIDGE_N = 20
CKPT_SHA256 = "9686b4c0f60e02251d8c2febee2029c1e7ebe74e85fbefda7e0c5daffc2075f2"
CKPT_EPOCH0 = 74
CKPT_EMA_VAL_LOSS = 0.25041554232438407
N_PARAMS = 40655408
GN_MAX_DIFF = 1e-5
FEAT_SHAPE = (1024, 72, 72)
SAM3_REPO = "facebook/sam3"
SAM3_REV_DEFAULT = "3c879f39826c281e95690f02c7821c4de09afae7"
TF_PIN = "5.2.0"
TIMM_PIN = "1.0.25"
TOL_PATHS_SHA256 = "f6603e639ec0e010640da401afc3837ae2e3923bf9c9dd62ff3e55bc9d9d3d46"
STORED_FID_SHA256 = "756e2d920f905480c746f66e1c3bd9c160e7dc454e2eddccd040848004be9ffe"
TOL_COS = 0.005              # |delta| for cos_patch and cos_global
TOL_REL = 0.01               # |delta| / stored for scale_ratio and mse
GATED = ("cos_patch", "cos_global", "scale_ratio", "mse")
COS_METRICS = ("cos_patch", "cos_global")
EPS_PATCH = 1e-12            # per-position cosine: s.t / sqrt((|s|^2 + eps)(|t|^2 + eps))
EPS_GLOBAL = 1e-8            # global cosine denominator floor
EPS_NORM = 1e-12             # channel L2-normalisation floor (mse_dir)
METRICS = ("cos_patch", "cos_patch_p05", "cos_patch_p50", "cos_global", "scale_ratio", "mse", "mse_dir", "ch_scale",
           "loss_total", "student_std", "teacher_std")
DUMMY_BOX = [100, 100, 300, 300]   # one box prompt for the one-frame teacher session (backbone is prompt-independent)
TAG_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# stored fp32 means of [FID-E] (fidelity.json sha256 756e2d92...); checked against the Volume copy at run time
STORED = {
    "test": {"cos_patch": 0.756984385534881, "cos_global": 0.7177551603170039,
             "scale_ratio": 0.9498849319104832, "mse": 1.1070626731998925},
    "test/2019_12_02_000005": {"cos_patch": 0.7578193606857242, "cos_global": 0.7149326255581239,
                               "scale_ratio": 0.9506466877387325, "mse": 1.1211110913382052},
    "test/2019_12_02_000208": {"cos_patch": 0.7636211533841663, "cos_global": 0.7257551568241192,
                               "scale_ratio": 0.9529285692056124, "mse": 1.0775507851450263},
    "test/2019_12_10_000060": {"cos_patch": 0.7498134913642479, "cos_global": 0.7107937820987014,
                               "scale_ratio": 0.9484759931514984, "mse": 1.1314775038365676},
    "test/2019_12_10_000078": {"cos_patch": 0.7566835367053852, "cos_global": 0.7195390767870709,
                               "scale_ratio": 0.9474884775460891, "mse": 1.098111312479771},
    "val/2019_11_28_000113": {"cos_patch": 0.7508357814739799, "cos_global": 0.6921778052278609,
                              "scale_ratio": 0.9557935679024919, "mse": 1.1941566283638172},
    "train20/2019_11_05_000002": {"cos_patch": 0.8172928978956694, "cos_global": 0.8077837131584718,
                                  "scale_ratio": 0.9488702276074521, "mse": 0.7630867285585043},
}
ROW_ORDER = ["test"] + [f"test/{c}" for c in EDI_TEST] + [f"val/{EDI_VAL}", f"train20/{EDI_TRAIN0}"]
# where each row's stored value sits inside fidelity.json["summary"]
STORED_LOCATION = {"test": ("groups", "test"),
                   **{f"test/{c}": ("per_clip", f"test/{c}") for c in EDI_TEST},
                   f"val/{EDI_VAL}": ("per_clip", f"val_bridge/{EDI_VAL}"),
                   f"train20/{EDI_TRAIN0}": ("per_clip", f"train20_bridge/{EDI_TRAIN0}")}


# =====================================================================================================================
# pure helpers (numpy + stdlib; exercised by the local selftest)
# =====================================================================================================================
def build_plan():
    """[(group, clip, fi)] in a fixed order: 4 test clips x 120, val x 120, training clip 0 x 20 (620)."""
    plan = []
    for clip in EDI_TEST:
        plan.extend(("test", clip, fi) for fi in range(0, CLIP_FRAMES, STRIDE))
    plan.extend(("val", EDI_VAL, fi) for fi in range(0, CLIP_FRAMES, STRIDE))
    plan.extend(("train20", EDI_TRAIN0, fi) for fi in range(TRAIN_BRIDGE_N))
    return plan


def frame_file(fi):
    """Decoded frames are 1-based JPEGs: frame index 0 is 0000001.jpg."""
    return "%07d.jpg" % (fi + 1)


def target_file(fi):
    return "backbone_%06d.pt" % fi


def rows_of(group, clip):
    if group == "test":
        return ("test", "test/" + clip)
    if group == "val":
        return ("val/" + clip,)
    if group == "train20":
        return ("train20/" + clip,)
    raise ValueError(group)


def to_chw(x, shape=FEAT_SHAPE):
    a = np.asarray(x, dtype=np.float64)
    while a.ndim > len(shape) and a.shape[0] == 1:
        a = a[0]
    if tuple(a.shape) != tuple(shape):
        raise ValueError(f"feature shape {tuple(np.shape(x))} is not {shape} (or 1 x {shape})")
    return a


def frame_metrics(student, teacher, shape=FEAT_SHAPE):
    """Every [FID-E] metric of one frame, float64. C = channels, P = H*W positions."""
    s = to_chw(student, shape)
    t = to_chw(teacher, shape)
    C = s.shape[0]
    s = s.reshape(C, -1)
    t = t.reshape(C, -1)
    st = np.einsum("cp,cp->p", s, t)
    ss = np.einsum("cp,cp->p", s, s)
    tt = np.einsum("cp,cp->p", t, t)
    cos_pos = st / np.sqrt((ss + EPS_PATCH) * (tt + EPS_PATCH))
    denom = math.sqrt(float(ss.sum())) * math.sqrt(float(tt.sum()))
    cos_global = float(st.sum()) / max(denom, EPS_GLOBAL)
    d = s - t
    mse = float(np.mean(d * d))
    s_unit = s / np.maximum(np.sqrt(ss), EPS_NORM)[None, :]
    t_unit = t / np.maximum(np.sqrt(tt), EPS_NORM)[None, :]
    du = s_unit - t_unit
    mse_dir = float(np.mean(du * du))
    ch_std = np.std(s, axis=1, ddof=1) - np.std(t, axis=1, ddof=1)
    ch_mean = np.mean(s, axis=1) - np.mean(t, axis=1)
    ch_scale = float(np.mean(ch_std * ch_std) + np.mean(ch_mean * ch_mean))
    s_sd = float(np.std(s, ddof=1))
    t_sd = float(np.std(t, ddof=1))
    cos_patch = float(np.mean(cos_pos))
    return {"cos_patch": cos_patch,
            "cos_patch_p05": float(np.percentile(cos_pos, 5)),
            "cos_patch_p50": float(np.percentile(cos_pos, 50)),
            "cos_global": cos_global,
            "scale_ratio": s_sd / t_sd if t_sd > 0 else float("nan"),
            "mse": mse, "mse_dir": mse_dir, "ch_scale": ch_scale,
            "loss_total": 1.0 * mse_dir + 0.5 * (1.0 - cos_patch) + 0.3 * ch_scale + 0.1 * mse,
            "student_std": s_sd, "teacher_std": t_sd}


def global_cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / max(float(np.linalg.norm(a)) * float(np.linalg.norm(b)), EPS_GLOBAL))


def teacher_agreement(fresh_fp16, stored_fp16):
    """Fresh vs stored teacher features (both fp16 arrays of the same shape)."""
    a = np.asarray(fresh_fp16)
    b = np.asarray(stored_fp16)
    if a.shape != b.shape:
        return {"shape_equal": False, "fresh_shape": list(a.shape), "stored_shape": list(b.shape)}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"shape_equal": True, "cos_global": global_cosine(a, b), "max_abs": float(d.max()),
            "mean_abs": float(d.mean()), "frac_equal": float(np.mean(a == b))}


def describe(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "median": float(np.median(v))}


def summarize_rows(records, key, metrics=METRICS):
    """records: per-frame dicts with 'group', 'clip' and record[key] = {metric: value}. -> {row: {metric: describe}}"""
    out = {}
    for row in ROW_ORDER:
        sel = [r for r in records if row in rows_of(r["group"], r["clip"]) and r.get(key) is not None]
        if not sel:
            continue
        out[row] = {"n": len(sel)}
        for m in metrics:
            vals = [r[key][m] for r in sel if m in r[key]]
            if vals:
                out[row][m] = describe(vals)
    return out


def compare(summary, stored=STORED, tol_cos=TOL_COS, tol_rel=TOL_REL):
    """Gated comparison of row means against the stored [FID-E] means. -> (table, all_within, n_values, n_within)"""
    table, n, n_ok = {}, 0, 0
    for row in ROW_ORDER:
        table[row] = {}
        for m in GATED:
            ref = stored[row][m]
            ours = (summary.get(row) or {}).get(m, {}) or {}
            ours = ours.get("mean")
            n += 1
            if ours is None or not math.isfinite(ours):
                table[row][m] = {"stored": ref, "ours": ours, "within": False, "note": "missing"}
                continue
            diff = ours - ref
            if m in COS_METRICS:
                within = abs(diff) <= tol_cos
                crit = {"abs_diff": abs(diff), "tolerance_abs": tol_cos}
            else:
                rel = abs(diff) / abs(ref)
                within = rel <= tol_rel
                crit = {"rel_diff": rel, "tolerance_rel": tol_rel}
            n_ok += bool(within)
            table[row][m] = dict({"stored": ref, "ours": ours, "diff": diff, "within": bool(within)}, **crit)
    return table, n_ok == n, n, n_ok


def stored_from_fidelity_json(fid):
    """The STORED-shaped dict read from [FID-E]'s fidelity.json (fp32 means)."""
    out = {}
    summ = fid["summary"]
    for row, (section, name) in STORED_LOCATION.items():
        node = summ[section][name]["fp32"]
        out[row] = {m: node[m]["mean"] for m in GATED}
        out[row]["n"] = node["n"]
    return out


def csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([repr(x) if isinstance(x, float) else x for x in r])
    return buf.getvalue()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# =====================================================================================================================
# local selftest (no torch, no GPU, no Volume)
# =====================================================================================================================
def _naive_metrics(s, t):
    """Loop reference for the four gated metrics (pure python over positions)."""
    C = s.shape[0]
    S = s.reshape(C, -1)
    T = t.reshape(C, -1)
    cps = []
    for p in range(S.shape[1]):
        a = [float(x) for x in S[:, p]]
        b = [float(x) for x in T[:, p]]
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a)
        nb = sum(y * y for y in b)
        cps.append(dot / math.sqrt((na + EPS_PATCH) * (nb + EPS_PATCH)))
    fa = [float(x) for x in S.ravel()]
    fb = [float(x) for x in T.ravel()]
    dot = sum(x * y for x, y in zip(fa, fb))
    cg = dot / max(math.sqrt(sum(x * x for x in fa)) * math.sqrt(sum(y * y for y in fb)), EPS_GLOBAL)

    def usd(v):
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    mse = sum((x - y) ** 2 for x, y in zip(fa, fb)) / len(fa)
    return {"cos_patch": sum(cps) / len(cps), "cos_global": cg, "scale_ratio": usd(fa) / usd(fb), "mse": mse}


def selftest():
    fails = []

    def check(name, cond, detail=""):
        print(("ok   " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    plan = build_plan()
    check("plan has 620 frames", len(plan) == 620, str(len(plan)))
    check("plan groups", [sum(g == x for g, _, _ in plan) for x in ("test", "val", "train20")] == [480, 120, 20])
    check("plan test frames 0..595 step 5", [fi for g, c, fi in plan if c == EDI_TEST[0]] == list(range(0, 600, 5)))
    check("plan train bridge 0..19", [fi for g, c, fi in plan if g == "train20"] == list(range(20)))
    check("file names", frame_file(0) == "0000001.jpg" and frame_file(595) == "0000596.jpg"
          and target_file(0) == "backbone_000000.pt" and target_file(595) == "backbone_000595.pt")
    check("rows_of", rows_of("test", "x") == ("test", "test/x") and rows_of("val", "v") == ("val/v",))
    check("ROW_ORDER covers STORED", sorted(ROW_ORDER) == sorted(STORED))

    rng = np.random.default_rng(0)
    shp = (8, 4, 5)
    t = rng.normal(size=shp)
    m = frame_metrics(t, t, shp)
    check("identical: cosines 1", abs(m["cos_patch"] - 1) < 1e-12 and abs(m["cos_global"] - 1) < 1e-12)
    check("identical: scale 1, mse 0, loss 0", abs(m["scale_ratio"] - 1) < 1e-12 and m["mse"] == 0 and
          abs(m["loss_total"]) < 1e-12 and m["ch_scale"] == 0)
    m = frame_metrics(2 * t, t, shp)
    check("2x: cos 1, scale 2, mse = mean t^2", abs(m["cos_global"] - 1) < 1e-12 and abs(m["scale_ratio"] - 2) < 1e-12
          and abs(m["mse"] - float(np.mean(t * t))) < 1e-12 and abs(m["mse_dir"]) < 1e-15)
    m = frame_metrics(-t, t[None], shp)
    check("negated: cosines -1 (leading 1 dims squeezed)", abs(m["cos_global"] + 1) < 1e-12
          and abs(m["cos_patch"] + 1) < 1e-12)
    s = rng.normal(size=shp) + 0.5 * t
    m = frame_metrics(s, t, shp)
    ref = _naive_metrics(s, t)
    check("random vs loop reference", all(abs(m[k] - ref[k]) < 1e-12 for k in ref),
          str({k: m[k] - ref[k] for k in ref}))
    try:
        frame_metrics(s, t[:4], shp)
        check("shape mismatch raises", False)
    except ValueError:
        check("shape mismatch raises", True)
    m16 = teacher_agreement(t.astype(np.float16), t.astype(np.float16))
    check("teacher agreement identical", m16["frac_equal"] == 1.0 and m16["max_abs"] == 0.0
          and abs(m16["cos_global"] - 1) < 1e-6)

    recs = [{"group": g, "clip": c, "fi": fi, "x": {mm: STORED[rows_of(g, c)[-1]][mm] for mm in GATED}}
            for g, c, fi in plan]
    summ = summarize_rows(recs, "x", GATED)
    check("summary n per row", summ["test"]["n"] == 480 and summ["test/" + EDI_TEST[0]]["n"] == 120
          and summ["val/" + EDI_VAL]["n"] == 120 and summ["train20/" + EDI_TRAIN0]["n"] == 20)
    tab, ok, n, n_ok = compare(summ)
    # pooled test mean of the 4 per-clip means differs from the stored pooled mean only by float rounding
    check("compare: stored values -> all within", ok and n == 28 and n_ok == 28, f"{n_ok}/{n}")
    edge = {r: dict(v) for r, v in STORED.items()}
    edge_s = {r: {m: {"mean": edge[r][m]} for m in GATED} for r in ROW_ORDER}
    edge_s["test"]["cos_patch"]["mean"] += 0.0049
    edge_s["val/" + EDI_VAL]["mse"]["mean"] *= 1.0099
    _, ok2, _, _ = compare(edge_s)
    check("compare: just inside tolerance", ok2)
    edge_s["test"]["cos_global"]["mean"] -= 0.0051
    tab3, ok3, _, n_ok3 = compare(edge_s)
    check("compare: cosine outside -> not within", (not ok3) and n_ok3 == 27
          and not tab3["test"]["cos_global"]["within"])
    edge_s["test"]["cos_global"]["mean"] += 0.0051
    edge_s["train20/" + EDI_TRAIN0]["scale_ratio"]["mean"] *= 1.0102
    _, ok4, _, _ = compare(edge_s)
    check("compare: scale ratio 1.02 % outside", not ok4)
    del edge_s["train20/" + EDI_TRAIN0]
    tab5, ok5, n5, _ = compare(edge_s)
    check("compare: a missing row fails", (not ok5) and n5 == 28
          and tab5["train20/" + EDI_TRAIN0]["mse"]["note"] == "missing")

    p = os.environ.get("TOL74_STORED_JSON")
    if p:
        with open(p, "rb") as f:
            raw = f.read()
        check("stored fidelity.json sha256", sha256_bytes(raw) == STORED_FID_SHA256)
        st = stored_from_fidelity_json(json.loads(raw.decode("utf-8")))
        check("embedded STORED == fidelity.json fp32 means (exact)",
              all(st[r][m] == STORED[r][m] for r in ROW_ORDER for m in GATED))
        check("fidelity.json row sizes", [st[r]["n"] for r in ROW_ORDER] == [480, 120, 120, 120, 120, 120, 20])
    else:
        print("skip stored fidelity.json check (set TOL74_STORED_JSON)")

    txt = csv_text(["a", "b"], [["x,y", 0.1], ["z", 1]])
    check("csv quoting + repr floats", txt == 'a,b\n"x,y",0.1\nz,1\n', repr(txt))
    print(f"selftest: {len(fails)} failure(s)")
    return not fails


if SELFTEST or not IS_NB:
    _ok = selftest()
    raise SystemExit(0 if _ok else 1)

# COMMAND ----------

# =====================================================================================================================
# Databricks: environment, registry, output folder, checks of the stored reference
# =====================================================================================================================
import torch                                   # noqa: E402
import torch.nn as nn                          # noqa: E402
import torch.nn.functional as F                # noqa: E402
import timm                                    # noqa: E402
import transformers                            # noqa: E402
import PIL                                     # noqa: E402
from PIL import Image                          # noqa: E402
import torchvision                             # noqa: E402
from torchvision import transforms             # noqa: E402

T_START = time.time()
ERRORS = []


def widget(name, default):
    try:
        dbutils.widgets.text(name, default)  # noqa: F821
    except Exception:  # noqa: BLE001
        pass
    v = dbutils.widgets.get(name)  # noqa: F821
    return default if v is None or str(v).strip() == "" else str(v).strip()


def as_bool(v):
    v = str(v).strip().lower()
    if v not in ("true", "false"):
        raise ValueError(f"expected true/false, got {v!r}")
    return v == "true"


W = {"tag": widget("tag", "fid_recheck_v1"), "fresh_teacher": widget("fresh_teacher", "true"),
     "numerics_row": widget("numerics_row", "true"), "require_gpu": widget("require_gpu", "A10"),
     "sam3_revision": widget("sam3_revision", SAM3_REV_DEFAULT), "progress_every": widget("progress_every", "50")}
TAG = W["tag"]
if not TAG_RX.match(TAG):
    raise ValueError(f"tag {TAG!r}")
FRESH = as_bool(W["fresh_teacher"])
ROW_B = as_bool(W["numerics_row"])
SAM3_REV = W["sam3_revision"]
if not re.match(r"^[0-9a-f]{40}$", SAM3_REV):
    raise ValueError("sam3_revision must be a 40-hex commit")
PROGRESS = max(1, int(W["progress_every"]))


def notebook_dir():
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # noqa: F821
    return "/Workspace" + osp.dirname(ctx.notebookPath().get())


CODE_DIR = notebook_dir()
if not osp.isfile(osp.join(CODE_DIR, "tol_paths.py")):
    raise RuntimeError(f"tol_paths.py is not beside this notebook ({CODE_DIR})")
sys.path.insert(0, CODE_DIR)
sys.modules.pop("tol_paths", None)
import tol_paths                               # noqa: E402
sys.path.remove(CODE_DIR)                      # nothing else is imported from the Workspace folder
with open(tol_paths.__file__, "rb") as _f:
    TOL_PATHS_SHA = sha256_bytes(_f.read())
if TOL_PATHS_SHA != TOL_PATHS_SHA256:
    raise RuntimeError(f"tol_paths.py sha256 {TOL_PATHS_SHA} != {TOL_PATHS_SHA256}")
R = tol_paths.load()
OUT = R.out("paper1.root", "evidence", "edinburgh_audit", TAG)
if osp.exists(OUT):
    raise RuntimeError(f"{OUT} exists: this notebook never writes into an existing folder (use a new tag)")
LOCAL = "/local_disk0/tol74/" + TAG
os.makedirs(LOCAL + "/out", exist_ok=True)
os.makedirs(LOCAL + "/cache", exist_ok=True)

P_FRAMES = R.path("datasets.edinburgh_pigs.decoded_frames")
P_TARGETS = R.path("paper1.teacher_targets.edinburgh")
P_CKPT = R.path("models.student_edinburgh")
P_STORED = R.path("paper1.root", None, "evidence", "edinburgh_audit", "v1_fidelity", "d_fidelity", "fidelity.json")

# versions and GPU
VERS = {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__,
        "transformers": transformers.__version__, "timm": timm.__version__, "Pillow": PIL.__version__,
        "numpy": np.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "databricks_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION")}
print("versions", VERS)
if not str(VERS["databricks_runtime"] or "").startswith("17.3"):
    raise RuntimeError(f"Databricks runtime {VERS['databricks_runtime']} is not 17.3 (BENCH_PROTOCOL §3)")
if transformers.__version__ != TF_PIN:
    raise RuntimeError(f"transformers {transformers.__version__} != {TF_PIN}")
if timm.__version__ != TIMM_PIN:
    raise RuntimeError(f"timm {timm.__version__} != {TIMM_PIN}")
if not torch.cuda.is_available():
    raise RuntimeError("no CUDA device")
GPU_NAME = torch.cuda.get_device_name(0)
if W["require_gpu"] == "A10" and not re.search(r"A10(?!0)", GPU_NAME):
    raise RuntimeError(f"GPU {GPU_NAME} is not an A10")
DEVICE = torch.device("cuda")
GPU = {"name": GPU_NAME, "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
       "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)}

# numerics = [FID-E]'s recorded state: matmul TF32 off ("highest"), cuDNN TF32 on (default), no cudnn.benchmark
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
NUMERICS = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}
print("numerics", NUMERICS)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


# the stored reference: sha256 and the embedded constants must both match it
_raw = read_bytes(P_STORED)
STORED_SHA = sha256_bytes(_raw)
if STORED_SHA != STORED_FID_SHA256:
    raise RuntimeError(f"stored fidelity.json sha256 {STORED_SHA} != {STORED_FID_SHA256}")
_st = stored_from_fidelity_json(json.loads(_raw.decode("utf-8")))
_bad = [(r, m) for r in ROW_ORDER for m in GATED if _st[r][m] != STORED[r][m]]
if _bad:
    raise RuntimeError(f"embedded STORED differs from the stored fidelity.json at {_bad}")
del _raw, _st
print(f"Tol_74 {NB_VERSION}: tag {TAG}, out {OUT}, GPU {GPU}, fresh_teacher {FRESH}, numerics_row {ROW_B}")

# COMMAND ----------

# =====================================================================================================================
# student: architecture copied from the exported Fixed notebook (Cell 7), drop_path_rate=0.0 (BENCH_PROTOCOL §3)
# =====================================================================================================================
def replace_bn_with_gn(module):
    """Recursively replace all BatchNorm layers with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nc = child.num_features
            ng = min(32, nc)
            while nc % ng != 0:
                ng -= 1
            gn = nn.GroupNorm(ng, nc, affine=child.affine)
            if child.affine and child.weight is not None:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child)


class StudentEncoderTinyViT(nn.Module):
    """TinyViT-21M-512 + multi-scale FPN, all GroupNorm (Fixed export Cell 7; + drop_path_rate=0.0)."""
    def __init__(self, out_channels=1024, pretrained=False):
        super().__init__()
        self.out_channels = out_channels
        self.backbone = timm.create_model('tiny_vit_21m_512', pretrained=pretrained, features_only=True,
                                          out_indices=(1, 2, 3), drop_path_rate=0.0)
        replace_bn_with_gn(self.backbone)
        self.lateral_s1 = nn.Sequential(nn.Conv2d(192, 256, 1, bias=False), nn.GroupNorm(32, 256), nn.GELU())
        self.lateral_s2 = nn.Sequential(nn.Conv2d(384, 256, 1, bias=False), nn.GroupNorm(32, 256), nn.GELU())
        self.lateral_s3 = nn.Sequential(nn.Conv2d(576, 256, 1, bias=False), nn.GroupNorm(32, 256), nn.GELU())
        self.channel_proj = nn.Sequential(nn.Conv2d(768, out_channels, 1, bias=False),
                                          nn.GroupNorm(32, out_channels), nn.GELU())
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(32, out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(32, out_channels))
        self.output_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.output_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        stages = self.backbone(x)
        f1 = self.lateral_s1(stages[0])
        f2 = self.lateral_s2(stages[1])
        f3 = self.lateral_s3(stages[2])
        f1 = F.interpolate(f1, size=(72, 72), mode='area')
        f2 = F.interpolate(f2, size=(72, 72), mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=(72, 72), mode='bilinear', align_corners=False)
        out = self.channel_proj(torch.cat([f1, f2, f3], dim=1))
        out = out + self.spatial_refine(out)
        return out * self.output_scale + self.output_bias


def load_student():
    """sha256 -> header -> strict EMA load (no forward before the load: TinyViT caches attention biases in eval)."""
    raw = read_bytes(P_CKPT)
    sha = sha256_bytes(raw)
    if sha != CKPT_SHA256:
        raise RuntimeError(f"student checkpoint sha256 {sha} != {CKPT_SHA256}")
    ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    del raw
    epoch0 = int(ck.get("epoch", -1))
    met = ck.get("metrics") or {}
    evl = float(met.get("ema_val_loss", float("nan")))
    if epoch0 != CKPT_EPOCH0 or not abs(evl - CKPT_EMA_VAL_LOSS) <= 1e-6:
        raise RuntimeError(f"checkpoint header epoch {epoch0} ema_val_loss {evl}: not the EMA epoch-75 checkpoint")
    ema = ck.get("ema_shadow")
    if not ema:
        raise RuntimeError("checkpoint has no ema_shadow")
    model = StudentEncoderTinyViT(out_channels=1024, pretrained=False)
    want = set(model.state_dict().keys())
    have = set(ema.keys())
    if want != have:
        raise RuntimeError(f"ema_shadow keys != model keys: missing {sorted(want - have)[:8]} extra {sorted(have - want)[:8]}")
    model.load_state_dict({k: v.detach().to(torch.float32) for k, v in ema.items()}, strict=True)
    n_msd = len(ck.get("model_state_dict") or {})
    del ck, ema
    model = model.to(DEVICE).float()
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_bn = sum(1 for mm in model.modules() if isinstance(mm, (nn.BatchNorm1d, nn.BatchNorm2d)))
    dps = [float(getattr(mm, "drop_prob", 0.0)) for mm in model.modules() if type(mm).__name__ == "DropPath"]
    if n_params != N_PARAMS:
        raise RuntimeError(f"{n_params} parameters != {N_PARAMS}")
    if n_bn:
        raise RuntimeError(f"{n_bn} BatchNorm layers left")
    if any(p > 0 for p in dps):
        raise RuntimeError(f"nonzero drop-path probabilities {sorted(set(dps))}")
    info = {"path": P_CKPT, "sha256": sha, "epoch0": epoch0, "ema_val_loss": evl, "n_ema_tensors": len(have),
            "n_model_state_dict": n_msd, "strict_load": True, "n_params": n_params, "n_batchnorm": n_bn,
            "n_droppath_modules": len(dps), "dtype": "float32"}
    return model, info


_tmp = timm.create_model("tiny_vit_21m_512", pretrained=False)
TV_MEAN = tuple(float(x) for x in _tmp.pretrained_cfg["mean"])
TV_STD = tuple(float(x) for x in _tmp.pretrained_cfg["std"])
del _tmp
STUDENT_TF = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                 transforms.Normalize(mean=TV_MEAN, std=TV_STD)])
print("student transform: Resize((1024,1024)) -> ToTensor -> Normalize", TV_MEAN, TV_STD)


def decode_rgb(b):
    with Image.open(io.BytesIO(b)) as im:
        return im.convert("RGB")


def load_target(b):
    t = torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)
    if not torch.is_tensor(t):
        raise RuntimeError(f"target is a {type(t).__name__}, not a tensor")
    if tuple(t.shape) not in ((1,) + FEAT_SHAPE, FEAT_SHAPE):
        raise RuntimeError(f"target shape {tuple(t.shape)}")
    if not bool(torch.isfinite(t).all()):
        raise RuntimeError("non-finite target")
    return t


def student_forward(model, x, cudnn_tf32=True):
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    try:
        with torch.no_grad():
            y = model(x)
    finally:
        torch.backends.cudnn.allow_tf32 = prev
    y = y.float().cpu()
    if tuple(y.shape) != (1,) + FEAT_SHAPE or not bool(torch.isfinite(y).all()):
        raise RuntimeError(f"student output shape {tuple(y.shape)} or non-finite")
    return y.numpy()


STUDENT, STUDENT_INFO = load_student()
PLAN = build_plan()
# GroupNorm train / eval check on one real frame (the first training-bridge frame), fp32
_x0 = STUDENT_TF(decode_rgb(read_bytes(osp.join(P_FRAMES, EDI_TRAIN0, "video1", frame_file(0))))).unsqueeze(0).to(DEVICE)
STUDENT.train()
with torch.no_grad():
    _ytr = STUDENT(_x0).float()
STUDENT.eval()
with torch.no_grad():
    _yev = STUDENT(_x0).float()
STUDENT_INFO["gn_train_eval_max_abs"] = float((_ytr - _yev).abs().max())
if not STUDENT_INFO["gn_train_eval_max_abs"] < GN_MAX_DIFF:
    raise RuntimeError(f"GroupNorm train/eval max diff {STUDENT_INFO['gn_train_eval_max_abs']} >= {GN_MAX_DIFF}")
del _x0, _ytr, _yev
print("student", STUDENT_INFO)

# COMMAND ----------

# =====================================================================================================================
# pass 1: primary (stored targets, FID-E numerics) and row B (cuDNN TF32 off); inputs hashed and cached locally
# =====================================================================================================================
RECORDS = []
t1 = time.time()
io_s = 0.0
for k, (grp, clip, fi) in enumerate(PLAN):
    t_io = time.time()
    fb = read_bytes(osp.join(P_FRAMES, clip, "video1", frame_file(fi)))
    tb = read_bytes(osp.join(P_TARGETS, clip, target_file(fi)))
    io_s += time.time() - t_io
    cdir = osp.join(LOCAL, "cache", clip)
    os.makedirs(cdir, exist_ok=True)
    with open(osp.join(cdir, frame_file(fi)), "wb") as f:
        f.write(fb)
    with open(osp.join(cdir, target_file(fi)), "wb") as f:
        f.write(tb)
    tgt = load_target(tb)
    t_np = tgt.float().numpy()
    x = STUDENT_TF(decode_rgb(fb)).unsqueeze(0).to(DEVICE)
    y = student_forward(STUDENT, x, cudnn_tf32=True)
    rec = {"group": grp, "clip": clip, "fi": fi, "frame_sha256": sha256_bytes(fb), "target_sha256": sha256_bytes(tb),
           "target_dtype": str(tgt.dtype).replace("torch.", ""), "primary": frame_metrics(y, t_np)}
    if ROW_B:
        y2 = student_forward(STUDENT, x, cudnn_tf32=False)
        rec["rowB"] = frame_metrics(y2, t_np)
        rec["rowB_cos_vs_primary_output"] = global_cosine(y2, y)
        rec["rowB_maxabs_vs_primary_output"] = float(np.abs(y2.astype(np.float64) - y).max())
    RECORDS.append(rec)
    del x, y, tgt, t_np, fb, tb
    if (k + 1) % PROGRESS == 0 or k + 1 == len(PLAN):
        print(f"  pass 1: {k + 1}/{len(PLAN)} frames, {time.time() - t1:.0f} s (volume reads {io_s:.0f} s)")
PASS1_S = round(time.time() - t1, 1)
SUMM = {"primary": summarize_rows(RECORDS, "primary")}
if ROW_B:
    SUMM["B_cudnn_tf32_off"] = summarize_rows(RECORDS, "rowB")
TABLE, ALL_OK, N_VAL, N_OK = compare(SUMM["primary"])
VERDICT = "CONFIRMED" if ALL_OK else "DISCREPANCY"
for row in ROW_ORDER:
    print(row, {m: (round(TABLE[row][m]["ours"], 6), TABLE[row][m]["within"]) for m in GATED})
print(f"primary: {N_OK}/{N_VAL} values within tolerance -> {VERDICT} ({PASS1_S} s)")

# COMMAND ----------

# =====================================================================================================================
# pass 2 (row A): FRESH teacher -- own SAM 3 forward + own hook on vision_encoder.backbone; not gated
# =====================================================================================================================
TEACHER_INFO = None
PASS2_S = None


class BackboneTap:
    """Forward hook on SAM 3's vision_encoder.backbone: tokens [B, H*W, C] -> [B, C, H, W], fp16 on the CPU."""
    def __init__(self, module):
        self.feat = None
        self.n_calls = 0
        self.handle = module.register_forward_hook(self._on_forward)

    def _on_forward(self, module, args, output):
        self.n_calls += 1
        h = getattr(output, "last_hidden_state", None)
        if h is None and torch.is_tensor(output):
            h = output
        if h is None:
            raise RuntimeError(f"backbone output {type(output).__name__} has no last_hidden_state")
        h = h.detach()
        if h.dim() == 3:
            b, n, c = h.shape
            side = math.isqrt(n)
            if side * side != n:
                raise RuntimeError(f"{n} tokens is not a square grid")
            h = h.transpose(1, 2).reshape(b, c, side, side)
        self.feat = h.cpu().to(torch.float16)

    def pop(self):
        f, self.feat = self.feat, None
        return f

    def remove(self):
        self.handle.remove()


def load_teacher():
    from huggingface_hub import login
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    secret = dbutils.secrets.get("hf", "token").strip()  # noqa: F821  (never printed or written)
    login(token=secret, add_to_git_credential=False)
    t0 = time.time()
    proc = Sam3TrackerVideoProcessor.from_pretrained(SAM3_REPO, revision=SAM3_REV, token=secret)
    model = Sam3TrackerVideoModel.from_pretrained(SAM3_REPO, revision=SAM3_REV, token=secret, dtype=torch.bfloat16)
    del secret
    model = model.to(DEVICE).eval()
    ip = getattr(proc, "image_processor", proc)
    size = getattr(ip, "size", None)
    info = {"repo": SAM3_REPO, "revision": SAM3_REV, "commit": getattr(model.config, "_commit_hash", None),
            "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "processor_size": dict(size) if isinstance(size, dict) else (str(size) if size is not None else None),
            "processor_class": type(proc).__name__, "load_s": round(time.time() - t0, 1),
            "hook": "forward hook on model.vision_encoder.backbone; last_hidden_state [1,N,C] -> [1,C,sqrt N,sqrt N] "
                    "-> fp16 on the CPU", "session": "one fresh one-frame tracking session per frame, one dummy box "
                    f"{DUMMY_BOX} on frame 0"}
    return proc, model, info


def fresh_teacher(proc, model, tap, pil):
    session = proc.init_video_session(inference_device=DEVICE, dtype=torch.bfloat16)
    inputs = proc(images=pil, device=DEVICE, return_tensors="pt")
    proc.add_inputs_to_inference_session(inference_session=session, frame_idx=0, obj_ids=[0],
                                         input_boxes=[[list(DUMMY_BOX)]], original_size=inputs.original_sizes[0])
    n0 = tap.n_calls
    with torch.no_grad():
        model(inference_session=session, frame=inputs.pixel_values[0])
    feat = tap.pop()
    calls = tap.n_calls - n0
    shape_in = list(inputs.pixel_values.shape)
    del session, inputs
    if feat is None or calls < 1:
        raise RuntimeError(f"backbone hook fired {calls} times (expected >= 1)")
    return feat, shape_in, calls


if FRESH:
    try:
        PROC, SAM3, TEACHER_INFO = load_teacher()
        TAP = BackboneTap(SAM3.vision_encoder.backbone)
        print("teacher", TEACHER_INFO)
        t2 = time.time()
        for k, (rec, (grp, clip, fi)) in enumerate(zip(RECORDS, PLAN)):
            cdir = osp.join(LOCAL, "cache", clip)
            fb = read_bytes(osp.join(cdir, frame_file(fi)))
            stored = load_target(read_bytes(osp.join(cdir, target_file(fi))))
            pil = decode_rgb(fb)
            feat, shape_in, n_calls = fresh_teacher(PROC, SAM3, TAP, pil)
            if tuple(feat.shape) != tuple(stored.shape) and tuple(feat.shape[1:]) != tuple(stored.shape):
                raise RuntimeError(f"fresh teacher shape {tuple(feat.shape)} vs stored {tuple(stored.shape)}")
            if not bool(torch.isfinite(feat).all()):
                raise RuntimeError(f"non-finite fresh teacher {clip} {fi}")
            fresh_np = feat.reshape(stored.shape).numpy()
            rec["teacher_agree"] = teacher_agreement(fresh_np, stored.numpy())
            rec["teacher_input_shape"] = shape_in
            rec["teacher_hook_calls"] = n_calls
            y = student_forward(STUDENT, STUDENT_TF(pil).unsqueeze(0).to(DEVICE), cudnn_tf32=True)
            rec["rowA"] = frame_metrics(y, fresh_np.astype(np.float32))
            # determinism: the re-run student output against the stored target reproduces pass 1
            rec["rerun_cos_global_diff"] = abs(global_cosine(y, stored.float().numpy()) - rec["primary"]["cos_global"])
            del feat, stored, pil, y, fresh_np
            if (k + 1) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            if (k + 1) % PROGRESS == 0 or k + 1 == len(PLAN):
                print(f"  pass 2: {k + 1}/{len(PLAN)} frames, {time.time() - t2:.0f} s")
        PASS2_S = round(time.time() - t2, 1)
        TAP.remove()
        SUMM["A_fresh_teacher"] = summarize_rows(RECORDS, "rowA")
        SUMM["teacher_fresh_vs_stored"] = {}
        for row in ROW_ORDER:
            sel = [r["teacher_agree"] for r in RECORDS if row in rows_of(r["group"], r["clip"]) and "teacher_agree" in r]
            SUMM["teacher_fresh_vs_stored"][row] = {m: describe([s[m] for s in sel])
                                                    for m in ("cos_global", "max_abs", "mean_abs", "frac_equal")}
            SUMM["teacher_fresh_vs_stored"][row]["n"] = len(sel)
        print("fresh vs stored teacher (test):", SUMM["teacher_fresh_vs_stored"]["test"])
    except Exception as e:  # noqa: BLE001 -- row A is supplementary: the primary result is still published
        ERRORS.append(f"(row A) {type(e).__name__}: {e}")
        print("row A failed:\n" + traceback.format_exc())

# COMMAND ----------

# =====================================================================================================================
# results: supplementary comparisons, files, publish (once, into the new folder)
# =====================================================================================================================
SUPP = {}
for key in ("B_cudnn_tf32_off", "A_fresh_teacher"):
    if key in SUMM:
        tab, ok, n, n_ok = compare(SUMM[key])
        SUPP[key] = {"table": tab, "all_within": ok, "n_values": n, "n_within": n_ok}
DET = [r["rerun_cos_global_diff"] for r in RECORDS if "rerun_cos_global_diff" in r]
EXTRA = {"rowB_output_vs_primary_output": {
             "cos_global": describe([r["rowB_cos_vs_primary_output"] for r in RECORDS if "rowB_cos_vs_primary_output" in r]),
             "max_abs": describe([r["rowB_maxabs_vs_primary_output"] for r in RECORDS
                                  if "rowB_maxabs_vs_primary_output" in r])},
         "student_rerun_determinism_cos_global_absdiff": describe(DET)}

NUMERICS_END = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}
if NUMERICS_END != NUMERICS:
    ERRORS.append(f"torch numerics changed during the run: {NUMERICS} -> {NUMERICS_END}")
CONFIG = {"notebook": NB_VERSION, "tag": TAG, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T_START)),
          "protocol": "BENCH_PROTOCOL b1.5 item 2(a); pre-registration bench/_s15_pending/fid_prereg.md",
          "widgets": W, "versions": VERS, "gpu": GPU, "numerics": NUMERICS, "numerics_end": NUMERICS_END,
          "student": STUDENT_INFO,
          "student_transform": {"resize": [1024, 1024], "to_tensor": True, "mean": TV_MEAN, "std": TV_STD},
          "teacher_fresh": TEACHER_INFO,
          "inputs": {"frames": P_FRAMES, "targets": P_TARGETS, "stored_fidelity_json": P_STORED,
                     "stored_fidelity_sha256": STORED_SHA},
          "registry": {"file": R.source, "sha256": R.sha256, "tol_paths_sha256": TOL_PATHS_SHA},
          "plan": {"n_frames": len(PLAN), "stride": STRIDE, "test_clips": list(EDI_TEST), "val_clip": EDI_VAL,
                   "train_bridge_clip": EDI_TRAIN0, "train_bridge_frames": TRAIN_BRIDGE_N},
          "tolerance": {"cos_abs": TOL_COS, "scale_ratio_mse_rel": TOL_REL, "gated": list(GATED)},
          "independence": "own frame indexing / decoding, target reading, student build and load, transform and "
                          "metrics; nothing imported from Tol_68, tol_bench_core or tol_distill_core"}
RESULT = {"notebook": NB_VERSION, "tag": TAG, "verdict": VERDICT, "n_values": N_VAL, "n_within": N_OK,
          "comparison_primary": TABLE, "summary": SUMM, "supplementary_comparison": SUPP, "extra": EXTRA,
          "definitions": frame_metrics.__doc__ + " cos_patch = mean over positions of s.t/sqrt((|s|^2+1e-12)(|t|^2+1e-12))"
                         " along C; cos_global = cosine of the flattened tensors (denominator floor 1e-8); scale_ratio ="
                         " std(s)/std(t) unbiased; mse = mean (s-t)^2; mse_dir = MSE of channel-L2-normalised maps;"
                         " ch_scale = MSE(per-channel unbiased std) + MSE(per-channel mean); loss_total = 1.0 mse_dir +"
                         " 0.5 (1-cos_patch) + 0.3 ch_scale + 0.1 mse; row value = mean over its frames",
          "timings_s": {"pass1": PASS1_S, "pass2": PASS2_S, "total": round(time.time() - T_START, 1)},
          "errors": ERRORS}

hdr = ["group", "clip", "fi", "frame_sha256", "target_sha256", "target_dtype"] + [f"primary_{m}" for m in METRICS]
if ROW_B:
    hdr += [f"rowB_{m}" for m in METRICS] + ["rowB_cos_vs_primary_output", "rowB_maxabs_vs_primary_output"]
hdr_a = [f"rowA_{m}" for m in METRICS] + ["rerun_cos_global_diff"]
any_a = any("rowA" in r for r in RECORDS)
rows_csv = []
for r in RECORDS:
    v = [r["group"], r["clip"], r["fi"], r["frame_sha256"], r["target_sha256"], r["target_dtype"]]
    v += [r["primary"][m] for m in METRICS]
    if ROW_B:
        v += [r["rowB"][m] for m in METRICS] + [r["rowB_cos_vs_primary_output"], r["rowB_maxabs_vs_primary_output"]]
    if "rowA" in r:
        v += [r["rowA"][m] for m in METRICS] + [r["rerun_cos_global_diff"]]
    elif any_a:
        v += [""] * len(hdr_a)          # row A stopped early: keep every primary row, leave row A empty
    rows_csv.append(v)
FILES = {"RECHECK_CONFIG.json": json.dumps(CONFIG, indent=1, sort_keys=True, default=str),
         "fid_recheck.json": json.dumps(RESULT, indent=1, sort_keys=True, default=str),
         "per_frame.csv": csv_text(hdr + (hdr_a if any_a else []), rows_csv)}
if any_a:
    FILES["teacher_fresh_vs_stored.csv"] = csv_text(
        ["group", "clip", "fi", "cos_global", "max_abs", "mean_abs", "frac_equal", "teacher_input_shape",
         "hook_calls"],
        [[r["group"], r["clip"], r["fi"], r["teacher_agree"].get("cos_global"), r["teacher_agree"].get("max_abs"),
          r["teacher_agree"].get("mean_abs"), r["teacher_agree"].get("frac_equal"),
          "x".join(map(str, r["teacher_input_shape"])), r["teacher_hook_calls"]]
         for r in RECORDS if "teacher_agree" in r])
for name, text in FILES.items():
    with open(osp.join(LOCAL, "out", name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
if osp.exists(OUT):
    raise RuntimeError(f"{OUT} appeared during the run: nothing published (results kept in {LOCAL}/out)")
dbutils.fs.mkdirs(OUT)  # noqa: F821
for name in FILES:
    dbutils.fs.cp("file:" + osp.join(LOCAL, "out", name), OUT + "/" + name)  # noqa: F821
PUBLISHED = {}
for name in FILES:
    PUBLISHED[name] = sha256_bytes(read_bytes(OUT + "/" + name))
    if PUBLISHED[name] != sha256_bytes(FILES[name].encode("utf-8")):
        ERRORS.append(f"published {name} differs from the local file")
print("published", OUT, PUBLISHED)
print(f"VERDICT {VERDICT}: {N_OK}/{N_VAL} primary values within tolerance; errors {ERRORS}")
if ERRORS:
    raise RuntimeError(f"Tol_74 finished with errors (results published to {OUT}): {ERRORS}")
dbutils.notebook.exit(json.dumps({"verdict": VERDICT, "n_within": N_OK, "n_values": N_VAL, "out": OUT,  # noqa: F821
                                  "published_sha256": PUBLISHED}))
