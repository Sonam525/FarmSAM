# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N60 — Paper 1 bench, gate G1 checks (job J0b)
# MAGIC
# MAGIC The G1 checks of BENCH_PROTOCOL.md §13 that the J0 smoke (Tol_63 smoke100) did not cover, on the **A10**, through the
# MAGIC SAME streaming loop as every bench run (`tol_bench_core.track_stream`, transformers 5.2.0, bf16, SDPA):
# MAGIC
# MAGIC | mode | check | what is compared | pass rule |
# MAGIC |---|---|---|---|
# MAGIC | `arm` | **G1-3** determinism floor | unpruned run twice (Edinburgh test clip, first `g13_frames`) | every frame's raw outputs bit-identical (sha256 of `pred_masks` + `object_score_logits`) |
# MAGIC | `arm` | **G1-3** k16 ≡ unpruned | `keep_last_n=16` + pop vs unpruned | bit-identical (or, if the floor itself is not bit-identical, no worse than the floor) |
# MAGIC | `arm` | **G1-3** pop on / off (T21, S21, ET) | k16 with vs without the `processed_frames` pop | bit-identical |
# MAGIC | `arm` | **G1-3** k8 delta | k8 + pop vs unpruned | measured (mask IoU), no gate |
# MAGIC | `arm` | **G1-5** birth at frame `birth_frame` | one pig prompted mid-stream vs a no-birth reference | existing objects after the birth: per-object mean mask IoU > 0.99; their prompts / cond outputs survive the birth's `add_inputs` (`clear_old_inputs=True`); the new object is tracked (emitted ≥ 90 % of frames, mean box IoU vs GT ≥ 0.5) |
# MAGIC | `arm` | **G1-4** memory slope | k16 + pop over the long PigTrack sequence (`pig_long`, 985 frames) and the full Edinburgh clip (600) | OLS slope of allocated memory (frames ≥ 32) within ±0.25 MiB/frame; growth explained by session-held vs other storages and a CUDA tensor census |
# MAGIC | `arm` | **leak** (J0 session retention) | allocated memory after every teardown vs the weights-only baseline; whether the session object outlived its teardown and who held it; one `release=False` run reproduces the J0 behaviour | first released teardown ≤ weights-only + `MEM_RETURN_TOL_MIB`, every later one ≤ the first + `MEM_DRIFT_TOL_MIB` (= Tol_63's rule) |
# MAGIC | `parity` | **G1-8** HF port vs official code | T21 / S21 (facebookresearch/sam2) and ET (facebookresearch/EdgeTAM), `pig_short` (98 frames), unpruned, same box prompts; official code in a subprocess with its own repo on PYTHONPATH (the two `sam2` packages never share a process) | mean mask IoU > 0.95 |
# MAGIC | `students` | **G1-9** student bf16 vs fp32 | S-Edi, S-v1, S-v2 on 10 frames (5 Edinburgh + 5 PigTrack) | global feature cosine ≥ 0.999 on every frame |
# MAGIC
# MAGIC Outputs (never overwritten; `publish_no_overwrite`): `paper1_distillation/evidence/g1/<tag>/<mode>_<arm>__<check>.json` and the
# MAGIC per-task summary `<mode>_<arm>.json`. A finished check file is reused on a re-run (resume).
# MAGIC G1-7 (`input_masks` on a calf window) needs `protocol/calf_windows_v1.json` (Tol_67) and is not part of this notebook.
# MAGIC
# MAGIC **Local check** (no torch / GPU / Volume): `TOL60_SELFTEST=1 python Tol_60_bench_g1.py` (stub tracker from
# MAGIC `test_tol_bench_core`, temporary registry root).

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
import contextlib
import copy
import gc
import glob
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

NB_NAME = "Tol_60_bench_g1"
NB_VERSION = "tol60-2026-10-14.4"
SELFTEST = os.environ.get("TOL60_SELFTEST") == "1"
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None


def _this_dir():
    env = os.environ.get("TOL60_CODE_DIR")
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


print(f"{NB_NAME} {NB_VERSION}: tol_bench_core {bc.BENCH_VERSION} (sha256 {_sha256(bc.__file__)[:16]})")
CORE_API = ("release_session", "check_memory_returned", "settle_memory", "session_tensor_bytes", "TEARDOWN_DIAG",
            "clear_finished_frames_under",
            "clear_frames_referencing")


def ensure_core(expect_sha="", wait_s=1800, poll_s=30, log=print):
    """The J0b core must be the one deployed beside this notebook: wait (re-importing) until the deployed
    tol_bench_core.py has the J0b API and, when expect_sha is given, that sha256 prefix -- so a job submitted while
    another job still runs on the previous core never mixes core versions. -> seconds waited."""
    global bc
    t0 = time.time()
    while True:
        sha = _sha256(bc.__file__)
        ok = all(hasattr(bc, a) for a in CORE_API) and (not expect_sha or sha.startswith(expect_sha))
        if ok:
            return round(time.time() - t0, 1)
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"deployed tol_bench_core sha {sha[:16]} lacks the J0b API or != {expect_sha}")
        log(f"waiting for the J0b tol_bench_core (deployed sha {sha[:16]}, want {expect_sha or 'J0b API'})")
        time.sleep(poll_s)
        bc = _import_exact("tol_bench_core", [CODE_DIR])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters and the pure comparison helpers (exercised by the selftest)

# COMMAND ----------

WIDGETS = collections.OrderedDict([
    ("mode", "arm"), ("arm", ""), ("arms", ""), ("checks", "g13,g15,g14,leak"), ("tag", "j0b_v1"),
    ("edi_clip", "2019_12_02_000005"), ("pig_long", "pigtrack0002"), ("pig_short", "pigtrack0055"),
    ("g13_frames", "300"), ("g15_frames", "200"), ("birth_frame", "50"), ("leak_frames", "50"),
    ("revisions", "protocol:hf_revisions_v1.json"), ("require_gpu", "A10"), ("progress_every", "250"),
    ("expect_core_sha", ""), ("g14_seqs", "short,edi,pig"),
])
MODES = ("arm", "parity", "students")
ARM_CHECKS = ("g13", "g15", "g14", "leak")
POP_CHECK_ARMS = ("T21", "S21", "ET")          # pop on/off identity (T and S-Edi: tol_v5_core E1 evidence)
PARITY_ARMS = ("T21", "S21", "ET")
STUDENT_ARMS = ("S-Edi", "S-v1", "S-v2")
G1_ARMS = ("T", "S-Edi", "T21", "S21", "ET")
BIRTH_IOU_MIN = 0.99                           # BENCH_PROTOCOL §13 G1-5
BIRTH_TRACKED_FRAC = 0.90                      # the new object is emitted on >= 90 % of frames after its birth ...
BIRTH_TRACKED_GT_IOU = 0.5                     # ... with mean box IoU vs its interpolated GT >= 0.5
SLOPE_TOL_MIB = 0.25                           # §11 "fits an A10" growth criterion, used as the G1-4 pass rule
PARITY_IOU_MIN = 0.95                          # §13 G1-8 (review R1.3)
STUDENT_COS_MIN = 0.999                        # §13 G1-9
TAG_RX = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")

# official code (G1-8): the facebookresearch repos, cloned per job; their commit is recorded. EdgeTAM's official
# predictor cannot encode the memory of more than one object per batch (perceiver.forward_2d: .view() on an
# expand()-ed tensor raises for B > 1; facebookresearch/EdgeTAM 7711e012, found locally 2026-10-14), so its objects are
# tracked one at a time (reset_state between objects). SAM 2.x tracks objects independently (per-object memory), so
# this is output-equivalent; it is recorded as official.per_object.
OFFICIAL = {
    "T21": {"repo": "sam2", "url": "https://github.com/facebookresearch/sam2", "ckpt_repo": "facebook/sam2.1-hiera-tiny",
            "cfg_name": "sam2.1_hiera_t.yaml", "pin_from": "facebook/sam2.1-hiera-tiny", "per_object": False},
    "S21": {"repo": "sam2", "url": "https://github.com/facebookresearch/sam2", "ckpt_repo": "facebook/sam2.1-hiera-small",
            "cfg_name": "sam2.1_hiera_s.yaml", "pin_from": "facebook/sam2.1-hiera-small", "per_object": False},
    "ET": {"repo": "EdgeTAM", "url": "https://github.com/facebookresearch/EdgeTAM", "ckpt_repo": "facebook/EdgeTAM",
           "cfg_name": "edgetam.yaml", "pin_from": None, "per_object": True},
}
OFFICIAL_DEPS = ("hydra-core==1.3.2", "omegaconf==2.3.0", "antlr4-python3-runtime==4.9.3", "iopath==0.1.10",
                 "portalocker")


def parse_params(raw):
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}; known: {list(WIDGETS)}")
    g = {k: str(raw.get(k) if raw.get(k) not in (None, "") else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace()
    P.mode = g["mode"]
    if P.mode not in MODES:
        raise ValueError(f"mode={P.mode!r}: expected one of {MODES}")
    P.arm = g["arm"]
    if P.mode == "arm" and P.arm not in G1_ARMS:
        raise ValueError(f"mode=arm needs arm in {G1_ARMS}, got {P.arm!r}")
    arms = [a for a in re.split(r"[,\s]+", g["arms"]) if a]
    if P.mode == "parity":
        P.arms = arms or list(PARITY_ARMS)
        bad = [a for a in P.arms if a not in OFFICIAL]
        if bad:
            raise ValueError(f"parity arms {bad}: official code is wired for {sorted(OFFICIAL)}")
    elif P.mode == "students":
        P.arms = arms or list(STUDENT_ARMS)
        bad = [a for a in P.arms if not bc.ARMS.get(a, {}).get("student_key")]
        if bad:
            raise ValueError(f"students: {bad} are not student arms")
    else:
        P.arms = [P.arm]
    P.checks = [c for c in re.split(r"[,\s]+", g["checks"]) if c]
    bad = [c for c in P.checks if c not in ARM_CHECKS]
    if bad:
        raise ValueError(f"checks {bad}: expected a subset of {ARM_CHECKS}")
    P.tag = g["tag"]
    if not TAG_RX.match(P.tag):
        raise ValueError(f"tag={P.tag!r}")
    P.edi_clip, P.pig_long, P.pig_short = g["edi_clip"], g["pig_long"], g["pig_short"]
    if P.edi_clip not in bc.EDI_TEST:
        raise ValueError(f"edi_clip {P.edi_clip} is not an Edinburgh test clip {bc.EDI_TEST}")
    for s in (P.pig_long, P.pig_short):
        if s not in bc.PIGTRACK_TEST:
            raise ValueError(f"{s} is not a PigTrack test sequence (val is T-only, train is distillation-only: §2.2)")
    P.g13_frames, P.g15_frames = int(g["g13_frames"]), int(g["g15_frames"])
    P.birth_frame, P.leak_frames = int(g["birth_frame"]), int(g["leak_frames"])
    if not (0 < P.birth_frame < P.g15_frames):
        raise ValueError("birth_frame must lie inside g15_frames")
    P.revisions_spec = g["revisions"]
    P.require_gpu = g["require_gpu"]
    if P.require_gpu not in ("A10", "A100", "any"):
        raise ValueError(f"require_gpu={P.require_gpu!r}")
    P.progress_every = int(g["progress_every"])
    P.g14_seqs = [w for w in re.split(r"[,\s]+", g["g14_seqs"]) if w]
    if not P.g14_seqs or any(w not in ("short", "edi", "pig") for w in P.g14_seqs):
        raise ValueError(f"g14_seqs={g['g14_seqs']!r}: a subset of short, edi, pig")
    P.expect_core_sha = g["expect_core_sha"].lower()
    if P.expect_core_sha and not re.match(r"^[0-9a-f]{8,64}$", P.expect_core_sha):
        raise ValueError(f"expect_core_sha={P.expect_core_sha!r}: a sha256 hex prefix")
    return P


# ---------------------------------------------------------------------------------------------------------------- capture
def _arr_bytes(x):
    """Exact bytes of an output tensor for a digest (bf16 -> fp32 is lossless)."""
    if bc._is_t(x):
        return x.detach().float().cpu().numpy().tobytes()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32)).tobytes()


class Capture:
    """on_frame recorder: per forwarded frame the sha256 of the raw outputs (pred_masks logits + object scores),
    the emitted rows, optionally the RLE of every emitted mask, and at prune frames the memory / session accounting."""

    def __init__(self, rle=False, mem=False, mem_every=0, census_at=(), census_fn=None, inspect_at=None,
                 inspect_fn=None):
        self.digest, self.rows, self.masks, self.mem, self.census, self.inspect = {}, {}, {}, [], {}, {}
        self.rle, self.track_mem, self.mem_every = rle, mem, int(mem_every)
        self.census_at, self.census_fn = set(census_at), census_fn
        self.inspect_at, self.inspect_fn = inspect_at, inspect_fn

    def __call__(self, fi, sess, out, mask_logits, binm, rows, live, pruned):
        h = hashlib.sha256()
        h.update(_arr_bytes(out.pred_masks))
        h.update(_arr_bytes(out.object_score_logits))
        self.digest[fi] = h.hexdigest()[:24]
        self.rows[fi] = [[r.obj_id, r.x0, r.y0, r.x1, r.y1, r.area] for r in rows]
        if self.rle:
            self.masks[fi] = {int(r.obj_id): bc.rle_encode(np.asarray(bc.to_numpy(binm[r.k]), dtype=bool)) for r in rows}
        if self.track_mem and (pruned or (self.mem_every and fi % self.mem_every == 0)):
            snap = bc.mem_snapshot() or {}
            acct = bc.session_tensor_bytes(sess)
            self.mem.append({"fi": int(fi), "pruned": bool(pruned), "n_obj": len(live), "alloc": snap.get("alloc"),
                             "reserved": snap.get("reserved"), "session_mib": acct["total_mib"],
                             "session_storages": acct["n_storages"], "by_group_mib": acct["by_group_mib"]})
        if fi in self.census_at and self.census_fn is not None:
            self.census[int(fi)] = self.census_fn()
        if self.inspect_at is not None and fi == self.inspect_at and self.inspect_fn is not None:
            self.inspect = self.inspect_fn(sess, live)

    def record(self):
        return {"digest": self.digest, "rows": self.rows, "masks": self.masks}


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


def compare_runs(a, b, *, frames=None, objects=None, iou_when_identical=False):
    """Two Capture records -> identity and mask-agreement statistics. objects restricts the mask comparison to those
    ids (G1-5: existing objects only). Masks are decoded only on frames whose digests differ (identical raw outputs
    give identical masks) unless iou_when_identical."""
    fa = sorted(set(a["digest"]) & set(b["digest"]))
    if frames is not None:
        fa = [f for f in fa if f in frames]
    only = sorted(set(a["digest"]) ^ set(b["digest"]))
    same = [f for f in fa if a["digest"][f] == b["digest"][f]]
    rows_same = [f for f in fa if a["rows"].get(f) == b["rows"].get(f)]
    ious, per_obj, emit_mismatch = [], collections.defaultdict(list), 0
    have_masks = bool(a.get("masks")) and bool(b.get("masks"))
    for f in fa:
        if have_masks and (iou_when_identical or a["digest"][f] != b["digest"][f] or objects is not None):
            ma, mb = a["masks"].get(f, {}), b["masks"].get(f, {})
            ids = set(ma) | set(mb)
            if objects is not None:
                ids &= set(objects)
            for oid in sorted(ids):
                if (oid in ma) != (oid in mb):
                    emit_mismatch += 1
                if a["digest"][f] == b["digest"][f] and oid in ma and oid in mb:
                    v = 1.0
                else:
                    v = rle_iou(ma.get(oid), mb.get(oid))
                ious.append(v)
                per_obj[oid].append(v)
        elif a["digest"][f] == b["digest"][f]:
            for oid in (objects if objects is not None else []):
                per_obj[oid].append(1.0)
    return {"frames_compared": len(fa), "frames_only_in_one": only[:10], "n_frames_only_in_one": len(only),
            "digest_identical_frames": len(same), "bit_identical": len(same) == len(fa) and not only,
            "first_digest_diff": next((f for f in fa if a["digest"][f] != b["digest"][f]), None),
            "rows_identical_frames": len(rows_same), "emit_mismatch": emit_mismatch,
            "mask_iou": _stats(ious) if ious else ({"n": 0, "mean": 1.0} if len(same) == len(fa) else {"n": 0}),
            "per_object_mean_iou": {str(k): round(float(np.mean(v)), 6) for k, v in sorted(per_obj.items())}}


def verdict_identity(cmp, floor=None):
    """PASS when bit-identical; when the determinism floor itself is not bit-identical, PASS_WITHIN_FLOOR when the
    comparison is no worse than the floor (mean mask IoU)."""
    if cmp["bit_identical"]:
        return "PASS"
    if floor is not None and not floor["bit_identical"]:
        fm = floor["mask_iou"].get("mean")
        cm = cmp["mask_iou"].get("mean")
        if fm is not None and cm is not None and cm >= fm - 1e-6:
            return "PASS_WITHIN_FLOOR"
    return "FAIL"


def pick_birth_object(gt_rows, frame):
    """The pig born mid-stream in G1-5: the one whose GT box at `frame` overlaps the other pigs least (lowest max box
    IoU; ties -> smallest id), so the birth does not start on top of another pig's mask."""
    r = gt_rows[gt_rows[:, 0] == frame]
    boxes = {int(x[1]): (x[2], x[3], x[2] + x[4], x[3] + x[5]) for x in r}
    best = None
    for oid in sorted(boxes):
        m = max([bc.box_iou(boxes[oid], boxes[o]) for o in boxes if o != oid] or [0.0])
        if best is None or m < best[1] - 1e-12:
            best = (oid, m)
    return best[0], boxes[best[0]], round(best[1], 4)


def slope_verdict(fis, mib, tol=SLOPE_TOL_MIB, warmup=None):
    g = bc.growth_slope(fis, mib, warmup=bc.WARMUP_FRAMES if warmup is None else warmup)
    ols = g.get("ols_mib_per_frame")
    g["pass"] = None if ols is None else bool(abs(ols) <= tol)
    g["tol_mib_per_frame"] = tol
    if ols is not None:
        y = np.asarray([m for f, m in zip(fis, mib) if f >= (bc.WARMUP_FRAMES if warmup is None else warmup)])
        g["range_mib"] = round(float(y.max() - y.min()), 3)
    return g


def explain_growth(mem):
    """Split the allocated-memory trend into the session-held part and the rest (OLS slopes, MiB/frame), and list the
    session groups that changed most between the first and last post-warm-up samples. Post-prune samples only (the
    same point of the 25-frame prune cycle); fewer than 3 -> every sample from frame 0 (short sequences)."""
    s = [m for m in mem if m["fi"] >= bc.WARMUP_FRAMES and m.get("alloc") is not None and m.get("pruned", True)]
    if len(s) < 3:
        s = [m for m in mem if m.get("alloc") is not None]
    if len(s) < 3:
        return {"n": len(s)}
    fis = [m["fi"] for m in s]
    alloc = [m["alloc"] for m in s]
    sess = [m["session_mib"] for m in s]
    other = [a - b for a, b in zip(alloc, sess)]
    out = {"n": len(s), "alloc": bc.growth_slope(fis, alloc, 0), "session": bc.growth_slope(fis, sess, 0),
           "outside_session": bc.growth_slope(fis, other, 0)}
    g0, g1 = s[0]["by_group_mib"], s[-1]["by_group_mib"]
    deltas = {k: round(g1.get(k, 0.0) - g0.get(k, 0.0), 3) for k in sorted(set(g0) | set(g1))}
    out["session_group_delta_mib"] = dict(sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:8])
    out["first"] = {k: s[0][k] for k in ("fi", "alloc", "session_mib", "n_obj")}
    out["last"] = {k: s[-1][k] for k in ("fi", "alloc", "session_mib", "n_obj")}
    return out


def census_diff(c0, c1, top=10):
    if not c0 or not c1:
        return None
    g0, g1 = c0.get("groups", {}), c1.get("groups", {})
    d = {}
    for k in set(g0) | set(g1):
        a, b = g0.get(k, [0, 0.0]), g1.get(k, [0, 0.0])
        if a != b:
            d[k] = {"count": b[0] - a[0], "mib": round(b[1] - a[1], 3)}
    return {"visible_mib": [c0.get("visible_mib"), c1.get("visible_mib")],
            "alloc_mib": [c0.get("alloc_mib"), c1.get("alloc_mib")],
            "invisible_mib": [c0.get("invisible_mib"), c1.get("invisible_mib")],
            "top_changes": dict(sorted(d.items(), key=lambda kv: -abs(kv[1]["mib"]))[:top])}


def parity_compare(hf, off):
    """HF record vs official record ({fi: {oid: rle}}): mask IoU over object-frames where either emitted."""
    ious, per_obj, f0 = [], collections.defaultdict(list), {}
    frames = sorted(set(hf) | set(off))
    for f in frames:
        a, b = hf.get(f, {}), off.get(f, {})
        for oid in sorted(set(a) | set(b)):
            v = rle_iou(a.get(oid), b.get(oid))
            ious.append(v)
            per_obj[oid].append(v)
            if f == 0:
                f0[str(oid)] = round(v, 4)
    st = _stats(ious)
    st["frac_ge_0.95"] = round(float(np.mean(np.asarray(ious) >= 0.95)), 6) if ious else None
    return {"frames": len(frames), "object_frames": len(ious), "mask_iou": st, "frame0_iou": f0,
            "per_object_mean_iou": {str(k): round(float(np.mean(v)), 5) for k, v in sorted(per_obj.items())},
            "pass": bool(ious) and st["mean"] > PARITY_IOU_MIN}


def feature_cosines(f32, f16):
    """[1, C, H, W] fp32 and bf16-cast-to-fp32 features -> global cosine, per-patch cosine (mean / min), rel L2."""
    a = np.asarray(f32, dtype=np.float64).reshape(f32.shape[1], -1)
    b = np.asarray(f16, dtype=np.float64).reshape(f16.shape[1], -1)
    g = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    num = (a * b).sum(0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-30
    pp = num / den
    return {"cos_global": round(g, 7), "cos_patch_mean": round(float(pp.mean()), 7),
            "cos_patch_min": round(float(pp.min()), 7),
            "rel_l2": round(float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30)), 7)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Staging, runs, checks

# COMMAND ----------


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_local(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp_{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, default=_jd) + "\n")
    os.replace(tmp, path)
    return path


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def out_path(ctx, name):
    if not TAG_RX.match(name.replace(".json", "")):
        raise ValueError(f"output name {name!r}")
    return ctx.R.out("paper1.root", "evidence", "g1", ctx.P.tag, name)


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


def stage_edinburgh(ctx, clip):
    if clip in ctx.staged:
        return ctx.staged[clip]
    R = ctx.R
    date, num = re.match(r"^(\d{4}_\d{2}_\d{2})_(\d{6})$", clip).groups()
    d = os.path.join(ctx.local, "stage", "edinburgh", clip, "video1")
    bc.stage_tree(R.path("datasets.edinburgh_pigs.decoded_frames", None, clip, "video1"), d, ctx.dbutils)
    gt, _ = bc.stage_file(R.path("datasets.edinburgh_pigs.annotated", None, date, num, "output.json"),
                          os.path.join(ctx.local, "stage", "edinburgh", clip, "output.json"), ctx.dbutils)
    txt, _ = bc.stage_file(R.path("datasets.edinburgh_pigs.initial_prompts", None, clip, "video1",
                                  f"BoundingBoxes_{clip}.txt"),
                           os.path.join(ctx.local, "stage", "edinburgh", clip, "BoundingBoxes.txt"), ctx.dbutils)
    kf = bc.read_edinburgh_keyframes(gt)
    ps = bc.protocol_a_edinburgh(kf)
    if {p.obj_id: list(p.box) for p in ps} != bc.read_bbox_prompts_txt(txt):
        raise bc.BenchAssertError(f"{clip}: GT-derived prompts differ from the published seeds (§4)")
    ctx.staged[clip] = {"frames_dir": d, "kf": kf, "prompts": ps}
    return ctx.staged[clip]


def stage_pig(ctx, seq):
    if seq in ctx.staged:
        return ctx.staged[seq]
    d = os.path.join(ctx.local, "stage", "pigtrack", "test", seq)
    bc.stage_tree(ctx.R.path("datasets.pigtrack.sequences", None, "test", seq), d, ctx.dbutils)
    src = bc.PigTrackFrames(d)
    ps, census = bc.protocol_a_from_mot(bc.read_mot_txt(src.gt_path))
    ctx.staged[seq] = {"seq_dir": d, "prompts": ps, "census": census, "n": len(src), "fps": src.fps}
    return ctx.staged[seq]


def edi_frames(ctx, clip, n=None):
    st = stage_edinburgh(ctx, clip)
    src = bc.EdinburghFrames(st["frames_dir"])
    return src if n is None else _limited(src, n)


def pig_frames(ctx, seq, n=None):
    st = stage_pig(ctx, seq)
    src = bc.PigTrackFrames(st["seq_dir"])
    return src if n is None else _limited(src, n)


def _limited(src, n):
    it = iter(src)
    try:
        for fi, pil, meta in it:
            if fi >= n:
                break
            yield fi, pil, meta
    finally:
        c = getattr(it, "close", None)
        if callable(c):
            c()


def run(ctx, arm, frames, prompts, label, *, keep=bc.KEEP_LAST_N, pop=True, capture=None, tel=None, release=True):
    """One track_stream run + the between-sequence memory check (recorded, never raised here: the G1 report states it)."""
    t0 = time.time()
    st = bc.track_stream(arm, frames, protocol="A", prompts=prompts, keep_last_n=keep, pop=pop, mot=bc.MotWriter(),
                         tel=tel, on_frame=capture, release=release, progress_every=ctx.P.progress_every,
                         log=ctx.log)
    mem = bc.check_memory_returned(ctx.mem_base, ctx.hooks["mem_settle"](), what=f"after {label}",
                                   raise_on_fail=False)
    td = st.get("teardown") or {}
    rec = {"label": label, "frames": st["frames"], "forwards": st["forwards"], "objects": st["objects"],
           "keep_last_n": keep, "pop": pop, "release": release, "wall_s": round(time.time() - t0, 2),
           "max_processed_frames": st["max_processed_frames"], "noncond_after_prune_max": st["noncond_after_prune_max"],
           "memory_after": mem, "session_alive_after_teardown": td.get("session_alive_after_teardown"),
           "holders": td.get("holders"), "frames_cleared": td.get("frames_cleared"),
           "session_alive_after_frame_clear": td.get("session_alive_after_frame_clear"),
           "released_mib": (td.get("release") or {}).get("released_mib") if isinstance(td.get("release"), dict) else None,
           "alloc_after_teardown_mib": td.get("alloc_after_mib")}
    ctx.runs.append(rec)
    ctx.log(f"  run {label}: {st['frames']} frames, {rec['wall_s']} s, after teardown {mem.get('alloc_mib')} MiB "
            f"(excess {mem.get('excess_mib')}), session alive {rec['session_alive_after_teardown']}")
    return st, rec


# ------------------------------------------------------------------------------------------------------------ G1-3
def check_g13(ctx, arm):
    P = ctx.P
    st = stage_edinburgh(ctx, P.edi_clip)
    n = P.g13_frames
    caps, recs = {}, {}
    plan = [("none_a", None, True), ("none_b", None, True), ("k16", bc.KEEP_LAST_N, True), ("k8", 8, True)]
    if arm.name in POP_CHECK_ARMS:
        plan.append(("k16_nopop", bc.KEEP_LAST_N, False))
    for label, keep, pop in plan:
        cap = Capture(rle=True)
        _, recs[label] = run(ctx, arm, edi_frames(ctx, P.edi_clip, n), st["prompts"], f"g13_{label}", keep=keep, pop=pop,
                             capture=cap)
        caps[label] = cap.record()
    floor = compare_runs(caps["none_a"], caps["none_b"])
    k16 = compare_runs(caps["k16"], caps["none_a"])
    k8 = compare_runs(caps["k8"], caps["none_a"], iou_when_identical=False)
    res = {"check": "G1-3", "clip": P.edi_clip, "frames": n, "objects": len(st["prompts"]),
           "determinism_floor": {"compare": floor, "verdict": "PASS" if floor["bit_identical"] else "FAIL"},
           "k16_vs_none": {"compare": k16, "verdict": verdict_identity(k16, floor)},
           "k8_vs_none": {"compare": k8, "verdict": "MEASURED"}, "runs": recs}
    if "k16_nopop" in caps:
        pp = compare_runs(caps["k16_nopop"], caps["k16"])
        res["pop_on_vs_off"] = {"compare": pp, "verdict": verdict_identity(pp)}
        res["pop_on_vs_off"]["max_processed_frames"] = {"pop": recs["k16"]["max_processed_frames"],
                                                        "no_pop": recs["k16_nopop"]["max_processed_frames"]}
    return res


# ------------------------------------------------------------------------------------------------------------ G1-5
def _birth_inspector(born, existing, birth_fi):
    def fn(sess, live):
        idx = getattr(sess, "_obj_id_to_idx", None)
        if idx is None:                                    # stub sessions (selftest)
            idx = getattr(sess, "_id2idx", {})
        pin = getattr(sess, "point_inputs_per_obj", None)
        if pin is None:                                    # stub: inputs live in sess.inputs
            pin = getattr(sess, "inputs", {})
        rec = {"existing_prompt_kept": {}, "existing_cond_kept": {}}
        for oid in existing:
            i = idx.get(oid)
            rec["existing_prompt_kept"][str(oid)] = bool(i is not None and 0 in (pin.get(i) or {}))
            rec["existing_cond_kept"][str(oid)] = bool(
                i is not None and 0 in sess.output_dict_per_obj[i].get("cond_frame_outputs", {}))
        j = idx.get(born)
        rec["born_prompt_at_birth"] = bool(j is not None and birth_fi in (pin.get(j) or {}))
        rec["born_cond_at_birth"] = bool(
            j is not None and birth_fi in sess.output_dict_per_obj[j].get("cond_frame_outputs", {}))
        rec["live"] = list(live)
        return rec
    return fn


@contextlib.contextmanager
def per_object_memory_flags(model):
    """G1-5 mechanism probe (NOT a protocol setting). transformers 5.2.0 Sam2Video / Sam3TrackerVideo encode the memory of
    all objects of a frame in ONE batch with is_mask_from_pts = any(per-object flags): on a frame where one object is
    prompted (a birth), every other object's memory is encoded from its BINARISED mask instead of the sigmoid. This
    patch splits the batch by flag so each object keeps its own flag (EdgeTamVideo already encodes per object).
    Yields True when the model has the batched encoder."""
    orig = getattr(type(model), "_batch_encode_memories", None)
    if orig is None:
        yield False
        return
    bound = orig.__get__(model)

    def patched(**kw):
        groups = collections.OrderedDict()
        for i, f in enumerate(kw["is_mask_from_pts_per_obj"]):
            groups.setdefault(bool(f), []).append(i)
        for f, idx in groups.items():
            bound(inference_session=kw["inference_session"], frame_idx=kw["frame_idx"],
                  objects_needing_memory_encoding=[kw["objects_needing_memory_encoding"][i] for i in idx],
                  high_res_masks_for_memory=[kw["high_res_masks_for_memory"][i] for i in idx],
                  object_score_logits_for_memory=[kw["object_score_logits_for_memory"][i] for i in idx],
                  is_mask_from_pts_per_obj=[f] * len(idx))
    model.__dict__["_batch_encode_memories"] = patched
    try:
        yield True
    finally:
        model.__dict__.pop("_batch_encode_memories", None)


def check_g15(ctx, arm):
    P = ctx.P
    st = stage_edinburgh(ctx, P.edi_clip)
    n, fb = P.g15_frames, P.birth_frame
    gt = bc.edinburgh_gt_rows(st["kf"])
    born, box, max_iou = pick_birth_object(gt, fb)
    existing = [p.obj_id for p in st["prompts"] if p.obj_id != born]
    ref_ps = [p for p in st["prompts"] if p.obj_id != born]
    birth_ps = ref_ps + [bc.Prompt(fb, born, box=tuple(float(v) for v in box))]
    cap_ref = Capture(rle=True)
    _, rec_ref = run(ctx, arm, edi_frames(ctx, P.edi_clip, n), ref_ps, "g15_ref", capture=cap_ref)
    cap_b = Capture(rle=True, inspect_at=fb, inspect_fn=_birth_inspector(born, existing, fb))
    _, rec_b = run(ctx, arm, edi_frames(ctx, P.edi_clip, n), birth_ps, "g15_birth", capture=cap_b)
    a, b = cap_b.record(), cap_ref.record()
    pre = compare_runs(a, b, frames=set(range(0, fb)))
    post = compare_runs(a, b, frames=set(range(fb, n)), objects=existing)
    probe = {"model_has_batched_memory_encoder": False}
    with per_object_memory_flags(arm.model) as patched:
        if patched:
            cap_p = Capture(rle=True)
            _, rec_p = run(ctx, arm, edi_frames(ctx, P.edi_clip, n), birth_ps, "g15_birth_perobj_flags", capture=cap_p)
            cmp_p = compare_runs(cap_p.record(), b, frames=set(range(fb, n)), objects=existing)
            probe = {"model_has_batched_memory_encoder": True, "existing_after_birth": cmp_p, "run": rec_p,
                     "explains": bool(cmp_p["per_object_mean_iou"]) and all(
                         v > BIRTH_IOU_MIN for v in cmp_p["per_object_mean_iou"].values())}
    per = {k: v for k, v in post["per_object_mean_iou"].items()}
    ok_exist = bool(per) and all(v > BIRTH_IOU_MIN for v in per.values()) and len(per) == len(existing)
    # the new object: emitted fraction and box IoU vs its interpolated GT after the birth
    emitted, gt_ious = 0, []
    for f in range(fb, n):
        row = next((r for r in a["rows"].get(f, []) if r[0] == born), None)
        g = gt[(gt[:, 0] == f) & (gt[:, 1] == born)]
        if row is not None:
            emitted += 1
            if len(g):
                gb = (g[0, 2], g[0, 3], g[0, 2] + g[0, 4], g[0, 3] + g[0, 5])
                gt_ious.append(bc.box_iou((row[1], row[2], row[3] + 1, row[4] + 1), gb))
    r0 = next((r for r in a["rows"].get(fb, []) if r[0] == born), None)
    frac = emitted / max(1, n - fb)
    new_ok = frac >= BIRTH_TRACKED_FRAC and bool(gt_ious) and float(np.mean(gt_ious)) >= BIRTH_TRACKED_GT_IOU
    ins = cap_b.inspect or {}
    prompts_ok = bool(ins) and all(ins.get("existing_prompt_kept", {}).values()) and \
        all(ins.get("existing_cond_kept", {}).values()) and ins.get("born_prompt_at_birth") and \
        ins.get("born_cond_at_birth")
    return {"check": "G1-5", "clip": P.edi_clip, "frames": n, "birth_frame": fb, "born_id": born,
            "born_box": [round(float(v), 2) for v in box], "born_max_iou_with_others": max_iou,
            "before_birth": {"bit_identical": pre["bit_identical"], "frames": pre["frames_compared"]},
            "existing_after_birth": post, "existing_verdict": "PASS" if ok_exist else "FAIL",
            "new_object": {"emitted_frac": round(frac, 4), "mean_box_iou_vs_gt": round(float(np.mean(gt_ious)), 4)
                           if gt_ious else None,
                           "frame_of_birth_box_iou_vs_prompt": None if r0 is None else round(
                               bc.box_iou((r0[1], r0[2], r0[3] + 1, r0[4] + 1), box), 4),
                           "verdict": "PASS" if new_ok else "FAIL"},
            "clear_old_inputs": {"inspect": ins, "verdict": "PASS" if prompts_ok else "FAIL"},
            "mechanism_probe_per_object_memory_flags": probe,
            "verdict": "PASS" if (ok_exist and new_ok and prompts_ok) else "FAIL", "runs": [rec_ref, rec_b]}


# ------------------------------------------------------------------------------------------------------------ G1-4
def cuda_tensor_census(exclude_ptrs=frozenset(), top=40):
    """gc-visible CUDA tensors (unique storages, model parameters / buffers excluded) grouped by shape and dtype, and
    the allocated memory nobody visible holds ('invisible': C++-held storages, caches, workspaces)."""
    if not (bc.TORCH_OK and bc.torch.cuda.is_available()):
        return None
    torch = bc.torch
    groups, seen, total = {}, set(), 0
    for o in gc.get_objects():
        try:
            if not (issubclass(type(o), torch.Tensor) and o.is_cuda):     # type(): no __class__ lookups
                continue
            s = o.untyped_storage()
            key = s.data_ptr()
            if key in exclude_ptrs or key in seen:
                continue
            seen.add(key)
            nb = s.nbytes()
            total += nb
            k = f"{tuple(o.shape)}|{str(o.dtype).replace('torch.', '')}"
            c = groups.get(k, [0, 0])
            groups[k] = [c[0] + 1, c[1] + nb]
        except Exception:  # noqa: BLE001
            continue
    alloc = bc.mem_allocated_mib()
    m = 2 ** 20
    top_g = dict(sorted(((k, [v[0], round(v[1] / m, 3)]) for k, v in groups.items()), key=lambda kv: -kv[1][1])[:top])
    return {"visible_mib": round(total / m, 3), "alloc_mib": round(alloc, 3),
            "invisible_mib": round(alloc - total / m - ctx_weights_mib(), 3), "groups": top_g}


_WEIGHTS = {"mib": 0.0}


def ctx_weights_mib():
    return _WEIGHTS["mib"]


def model_ptrs(arm):
    if not bc.TORCH_OK:
        return frozenset()
    ptrs, nb = set(), 0
    for mod in (arm.model, arm.student):
        if mod is None or not hasattr(mod, "parameters"):
            continue
        for t in list(mod.parameters()) + list(mod.buffers()):
            try:
                s = t.untyped_storage()
                if s.data_ptr() not in ptrs:
                    ptrs.add(s.data_ptr())
                    nb += s.nbytes()
            except Exception:  # noqa: BLE001
                pass
    _WEIGHTS["mib"] = nb / 2 ** 20
    return frozenset(ptrs)


def check_g14(ctx, arm, which):
    """which: 'pig' = pig_long (985 frames, the G1-4 slope), 'edi' = the full Edinburgh clip (the J0 T growth),
    'short' = pig_short (98 frames, the J0 S-Edi growth on pigtrack0055). Every 5th frame is also sampled (session
    accounting) to resolve the 25-frame prune cycle."""
    P = ctx.P
    if which == "pig":
        stg = stage_pig(ctx, P.pig_long)
        frames, prompts, n, name = pig_frames(ctx, P.pig_long), stg["prompts"], stg["n"], P.pig_long
    elif which == "short":
        stg = stage_pig(ctx, P.pig_short)
        frames, prompts, n, name = pig_frames(ctx, P.pig_short), stg["prompts"], stg["n"], P.pig_short
    else:
        stg = stage_edinburgh(ctx, P.edi_clip)
        frames, prompts, n, name = edi_frames(ctx, P.edi_clip), stg["prompts"], bc.EDI_N_FRAMES, P.edi_clip
    excl = model_ptrs(arm)
    last_prune = (n // bc.PRUNE_EVERY) * bc.PRUNE_EVERY - 1
    census_at = (99, last_prune) if last_prune > 99 else ()
    cap = Capture(mem=True, mem_every=0 if which == "pig" else 5, census_at=census_at,
                  census_fn=lambda: cuda_tensor_census(excl))
    tel = bc.Telemetry(arm.model, use_cuda=ctx.hooks.get("use_cuda"))
    bc_reset_peaks()
    try:
        st, rec = run(ctx, arm, frames, prompts, f"g14_{which}", capture=cap, tel=tel)
        tsum = tel.summary()
    finally:
        tel.detach()
    fis = [m["fi"] for m in tsum.get("mem_samples", [])]
    alloc = [m.get("alloc") for m in tsum.get("mem_samples", [])]
    if not alloc or alloc[0] is None:
        slope = {"pass": None, "note": "no CUDA"}
    elif sum(1 for f in fis if f >= bc.WARMUP_FRAMES) < 3:
        slope = dict(bc.growth_slope(fis, alloc, 0), **{"pass": None, "note": "< 3 post-warm-up prune samples: "
                                                                               "diagnostic run, no slope verdict"})
    else:
        slope = slope_verdict(fis, alloc)
    expl = explain_growth(cap.mem)
    cd = census_diff(cap.census.get(census_at[0]) if census_at else None,
                     cap.census.get(census_at[1]) if census_at else None)
    return {"check": "G1-4", "sequence": name, "frames": st["frames"], "objects": st["objects"],
            "slope": slope, "verdict": "DIAGNOSTIC" if which == "short" else
            {True: "PASS", False: "FAIL", None: "NOT_MEASURED"}[slope.get("pass")],
            "growth_explained": expl, "census_diff": cd, "census_frames": list(census_at),
            "mem_samples": tsum.get("mem_samples"), "session_samples": cap.mem,
            "gpu_ms_model_median": (tsum.get("gpu_ms", {}).get("model") or {}).get("median"),
            "noncond_after_prune_max": st["noncond_after_prune_max"], "max_processed_frames": st["max_processed_frames"],
            "run": rec}


def bc_reset_peaks():
    if bc.TORCH_OK and bc.torch.cuda.is_available():
        bc.torch.cuda.reset_peak_memory_stats()


# ------------------------------------------------------------------------------------------------------------ leak
def leak_rule(excess, tol, drift_tol):
    """The Tol_63 between-sequence rule on a list of post-teardown excesses (MiB above weights-only): the first may
    carry the lazily built caches (<= tol), every later one must stay within drift_tol of the first. -> (ok, drift_tol)"""
    if not excess:
        return None, drift_tol
    first = excess[0]
    return bool(first <= tol and all(e <= first + drift_tol for e in excess[1:])), drift_tol


def check_leak(ctx, arm):
    """The J0 behaviour, measured both ways on the same Edinburgh prefix: released (the fix) and release=False (the
    old teardown: drop the reference + gc). Runs LAST in the arm task: an unreleased session may stay allocated."""
    P = ctx.P
    st = stage_edinburgh(ctx, P.edi_clip)
    _, rel = run(ctx, arm, edi_frames(ctx, P.edi_clip, P.leak_frames), st["prompts"], "leak_released")
    _, old = run(ctx, arm, edi_frames(ctx, P.edi_clip, P.leak_frames), st["prompts"], "leak_unreleased",
                 release=False)
    all_runs = [r for r in ctx.runs if r.get("release")]
    excess = [r["memory_after"].get("excess_mib") for r in all_runs]
    measured = [e for e in excess if e is not None]
    ok, drift_tol = leak_rule(measured, bc.MEM_RETURN_TOL_MIB, getattr(bc, "MEM_DRIFT_TOL_MIB", 64.0))
    holders = sorted({h for r in ctx.runs for h in (r.get("holders") or [])})
    return {"check": "leak", "released_runs": len(all_runs),
            "excess_after_teardown_mib": excess, "max_excess_mib": max([e for e in excess if e is not None],
                                                                        default=None),
            "tol_mib": bc.MEM_RETURN_TOL_MIB, "drift_tol_mib": drift_tol,
            "rule": "first released teardown <= weights-only + tol_mib; every later one <= the first + drift_tol_mib",
            "verdict": {True: "PASS", False: "FAIL", None: "NOT_MEASURED"}[ok],
            "session_alive_after_teardown": [r.get("session_alive_after_teardown") for r in ctx.runs],
            "session_alive_after_frame_clear": [r.get("session_alive_after_frame_clear") for r in ctx.runs],
            "frames_cleared": sorted({c for r in ctx.runs for c in (r.get("frames_cleared") or [])}),
            "holders_seen": holders[:60],
            "unreleased_run": {"excess_mib": old["memory_after"].get("excess_mib"),
                               "session_alive": old.get("session_alive_after_teardown"), "holders": old.get("holders")},
            "released_run": {"excess_mib": rel["memory_after"].get("excess_mib"),
                             "session_alive": rel.get("session_alive_after_teardown")}}


# ------------------------------------------------------------------------------------------------------------ G1-8
OFFICIAL_SCRIPT = r'''
import json, os, sys, time, contextlib
import numpy as np, torch
cfg_name, ckpt, jpeg_dir, prompts_path, out_path, per_object = sys.argv[1:7]
per_object = per_object == "1"
from sam2.build_sam import build_sam2_video_predictor
import sam2
{rle_src}
prompts = json.load(open(prompts_path))
t0 = time.time()
DEV = os.environ.get("OFFICIAL_DEVICE", "cuda")
predictor = build_sam2_video_predictor(cfg_name, ckpt, device=DEV)
info = {{"sam2_file": sam2.__file__, "torch": torch.__version__, "cfg": cfg_name,
         "fill_hole_area": getattr(predictor, "fill_hole_area", None),
         "non_overlap_masks": getattr(predictor, "non_overlap_masks", None), "per_object": per_object}}
recs = {{}}
amp = torch.autocast("cuda", dtype=torch.bfloat16) if DEV == "cuda" else contextlib.nullcontext()
groups = [[p] for p in prompts] if per_object else [prompts]
with torch.inference_mode(), amp:
    state = predictor.init_state(video_path=jpeg_dir)
    for g in groups:
        predictor.reset_state(state)
        for oid, x1, y1, x2, y2 in g:
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=int(oid),
                                            box=np.asarray([x1, y1, x2, y2], np.float32))
        for f_idx, ids, logits in predictor.propagate_in_video(state):
            m = (logits[:, 0] > 0.0).cpu().numpy()
            d = recs.setdefault(int(f_idx), {{}})
            for k, i in enumerate(ids):
                if m[k].any():
                    d[str(int(i))] = rle_encode(m[k])
info["seconds"] = round(time.time() - t0, 2)
info["frames"] = len(recs)
json.dump({{"info": info, "masks": recs}}, open(out_path, "w"))
print("OFFICIAL_DONE", info["frames"], info["seconds"])
'''


def _git_head(repo):
    try:
        r = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def fetch_repo(ctx, name, url):
    """git clone --depth 1 (fallback: the GitHub zip of the default branch). -> (repo dir, commit or None, how)."""
    root = os.path.join(ctx.local, "official", name + "_repo")
    if os.path.isdir(os.path.join(root, "sam2")):
        return root, _git_head(root), "cached"
    shutil.rmtree(root, ignore_errors=True)
    r = subprocess.run(["git", "clone", "--depth", "1", url, root], capture_output=True, text=True, timeout=600)
    if r.returncode == 0 and os.path.isdir(os.path.join(root, "sam2")):
        return root, _git_head(root), "git"
    ctx.log(f"git clone {url} failed ({(r.stderr or '')[-300:]}); trying the zip archive")
    import urllib.request
    zp = root + ".zip"
    urllib.request.urlretrieve(url.replace("https://github.com/", "https://codeload.github.com/") + "/zip/HEAD", zp)
    with zipfile.ZipFile(zp) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(os.path.dirname(root))
    os.replace(os.path.join(os.path.dirname(root), top), root)
    return root, None, "zip"


def install_deps(ctx):
    d = os.path.join(ctx.local, "official", "deps")
    if os.path.isfile(os.path.join(d, ".ok")):
        return d
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", d,
                        *OFFICIAL_DEPS], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"pip install of the official-code deps failed: {(r.stderr or r.stdout)[-800:]}")
    open(os.path.join(d, ".ok"), "w").close()
    return d


def find_config(repo, cfg_name):
    hits = sorted(glob.glob(os.path.join(repo, "sam2", "configs", "**", cfg_name), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{cfg_name} not under {repo}/sam2/configs")
    rel = os.path.relpath(hits[0], os.path.join(repo, "sam2")).replace("\\", "/")
    return rel                                            # e.g. configs/sam2.1/sam2.1_hiera_t.yaml


def official_checkpoint(ctx, spec):
    from huggingface_hub import HfApi, hf_hub_download
    rev = ctx.revisions.get(spec["pin_from"]) if spec["pin_from"] else None
    files = HfApi().list_repo_files(spec["ckpt_repo"], revision=rev)
    pts = [f for f in files if f.endswith(".pt")]
    if len(pts) != 1:
        raise RuntimeError(f"{spec['ckpt_repo']}: expected one .pt file, found {pts}")
    p = hf_hub_download(spec["ckpt_repo"], pts[0], revision=rev)
    return p, pts[0], rev


def run_official(ctx, arm_name, jpeg_dir, prompts):
    spec = OFFICIAL[arm_name]
    repo, commit, how = fetch_repo(ctx, spec["repo"], spec["url"])
    deps = install_deps(ctx)
    cfg = find_config(repo, spec["cfg_name"])
    ckpt, ckpt_file, rev = official_checkpoint(ctx, spec)
    run_dir = os.path.join(ctx.local, "official", "run_" + arm_name)
    os.makedirs(run_dir, exist_ok=True)
    script = os.path.join(run_dir, "official_track.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(OFFICIAL_SCRIPT.format(rle_src=inspect.getsource(bc.rle_encode)))
    pj = write_json_local(os.path.join(run_dir, "prompts.json"),
                          [[int(p.obj_id)] + [float(v) for v in p.box] for p in prompts])
    out = os.path.join(run_dir, "official_masks.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([repo, deps]), HYDRA_FULL_ERROR="1", SAM2_BUILD_CUDA="0")
    t0 = time.time()
    r = subprocess.run([sys.executable, script, cfg, ckpt, jpeg_dir, pj, out, "1" if spec["per_object"] else "0"],
                       capture_output=True, text=True, timeout=3600, cwd=run_dir, env=env)
    log_tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-3000:]
    if r.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError(f"official {arm_name} failed (rc {r.returncode}): {log_tail}")
    with open(out, encoding="utf-8") as f:
        res = json.load(f)
    masks = {int(k): {int(o): v for o, v in d.items()} for k, d in res["masks"].items()}
    meta = dict(res["info"], repo=spec["url"], commit=commit, fetched_by=how, config=cfg, checkpoint=ckpt_file,
                checkpoint_repo=spec["ckpt_repo"], checkpoint_revision=rev, checkpoint_sha256=_sha256(ckpt),
                wall_s=round(time.time() - t0, 1),
                post_processing_skipped="Skipping the post-processing" in (r.stderr or ""),
                stderr_tail=(r.stderr or "")[-1200:])
    return masks, meta


def jpeg_dir_for_official(ctx, seq):
    """The official loaders sort frames by int(file stem): PigTrack's img1/00000001.jpg.. qualifies as staged."""
    st = stage_pig(ctx, seq)
    d = os.path.join(st["seq_dir"], "img1")
    names = sorted(os.listdir(d))
    if [int(os.path.splitext(n)[0]) for n in names] != list(range(1, len(names) + 1)):
        raise ValueError(f"{d}: frames are not 1..N")
    return d


def parity_one(ctx, arm_name):
    P = ctx.P
    st = stage_pig(ctx, P.pig_short)
    if any(p.frame != 0 for p in st["prompts"]):
        raise ValueError(f"{P.pig_short}: G1-8 needs every id prompted at frame 0 (official prompts on frame 0)")
    arm = ctx.hooks["arm_loader"](ctx, arm_name)
    ctx.mem_base = ctx.hooks["mem_settle"]()
    cap = Capture(rle=True)
    try:
        _, rec = run(ctx, arm, pig_frames(ctx, P.pig_short), st["prompts"], f"g18_{arm_name}_hf", keep=None,
                     capture=cap)
    finally:
        info = dict(arm.info)
        del arm
        ctx.hooks["mem_settle"]()
    hf_masks = cap.record()["masks"]
    off_masks, meta = ctx.hooks["official_runner"](ctx, arm_name, jpeg_dir_for_official(ctx, P.pig_short),
                                                   st["prompts"])
    cmp = parity_compare(hf_masks, off_masks)
    return {"check": "G1-8", "arm": arm_name, "sequence": P.pig_short, "frames": st["n"], "objects": len(st["prompts"]),
            "hf": {"hf_id": info.get("hf_id"), "commit": info.get("commit_hash"), "keep_last_n": None, "pop": True,
                   "dtype": "bfloat16 weights", "run": rec},
            "official": meta, "compare": cmp, "verdict": "PASS" if cmp["pass"] else "FAIL",
            "note": "official = facebookresearch code, bf16 autocast, init_state(video_path) (all frames preloaded, "
                    "PIL resize) vs the HF processor's resize: small pixel differences are expected"}


# ------------------------------------------------------------------------------------------------------------ G1-9
def student_frames(ctx):
    P = ctx.P
    out = []
    for fi, pil, _m in edi_frames(ctx, P.edi_clip):
        if fi in (0, 120, 240, 360, 480):
            out.append((f"{P.edi_clip}#{fi}", pil.copy()))
    for fi, pil, _m in pig_frames(ctx, P.pig_short):
        if fi in (0, 20, 40, 60, 80):
            out.append((f"{P.pig_short}#{fi}", pil.copy()))
    return out


def students_one(ctx, name, frames):
    torch = bc.torch
    spec = bc.ARMS[name]
    src = ctx.R.path(spec["student_key"])
    pin = ctx.R.entry(spec["student_key"]).get("sha256")
    local, sha = bc.stage_file(src, os.path.join(ctx.local, "models", name, os.path.basename(src)), ctx.dbutils, pin)
    s32, meta = bc.load_student(local, device="cuda", dtype=torch.float32)
    s16 = copy.deepcopy(s32).to(torch.bfloat16).eval()
    tf = bc.make_student_transform()
    rows = []
    with torch.no_grad():
        for label, pil in frames:
            x = tf(pil).unsqueeze(0).to("cuda")
            f32 = s32(x).float().cpu().numpy()
            f16 = s16(x.to(torch.bfloat16)).float().cpu().numpy()
            if not (np.isfinite(f32).all() and np.isfinite(f16).all()):
                raise bc.BenchAssertError(f"{name} {label}: non-finite student features")
            r = feature_cosines(f32, f16)
            r["frame"] = label
            rows.append(r)
    del s32, s16
    ctx.hooks["mem_settle"]()
    cmin = min(r["cos_global"] for r in rows)
    return {"check": "G1-9", "arm": name, "student": {"sha256": sha, "epoch": meta.get("epoch"),
                                                      "format": meta.get("format"), "file": os.path.basename(src)},
            "frames": rows, "cos_global_min": cmin, "cos_global_mean": round(float(np.mean([r["cos_global"] for r in rows])), 7),
            "cos_patch_mean": round(float(np.mean([r["cos_patch_mean"] for r in rows])), 7),
            "rel_l2_max": max(r["rel_l2"] for r in rows), "threshold": STUDENT_COS_MIN,
            "verdict": "PASS" if cmin >= STUDENT_COS_MIN else "FAIL",
            "note": "frames: 5 Edinburgh test + 5 PigTrack test frames for every student (a precision check, not a "
                    "domain check)"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production hooks and the driver

# COMMAND ----------


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cuda_probe(retries=4, sleep_s=60):
    """= Tol_63.cuda_probe: a subprocess CUDA start-up probe BEFORE this process touches CUDA (A10 boots occasionally
    fail at CUDA init -- J0b run 1086943349058786 task g1_arm_T: 'CUDA unknown error'; a process whose CUDA init
    failed cannot recover, a fresh subprocess can)."""
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


PROD_HOOKS = {"local_root": "/local_disk0/tol60", "arm_loader": _prod_arm_loader, "gpu_preflight": gpu_preflight,
              "hf_token": lambda dbu: bc.hf_token_from_dbutils(dbu), "mem_settle": lambda: bc.settle_memory(),
              "official_runner": run_official, "use_cuda": None, "log": _log}


def main(raw, *, dbutils_=None, R=None, hooks=None):
    h = dict(PROD_HOOKS)
    h.update(hooks or {})
    P = parse_params(raw)
    core_wait_s = ensure_core(P.expect_core_sha, log=h["log"])
    R = R if R is not None else tol_paths.load(version=1)
    t0 = time.time()
    ctx = types.SimpleNamespace(P=P, R=R, dbutils=dbutils_, hooks=h, log=h["log"], local=h["local_root"], staged={},
                                runs=[], mem_base=None, token=None, revisions=resolve_revisions(R, P.revisions_spec))
    key = P.arm if P.mode == "arm" else "all"
    summary_name = f"{P.mode}_{key}.json"
    done = load_done(ctx, summary_name)
    if done is not None:
        ctx.log(f"{summary_name} already published: nothing to do")
        return {"summary": summary_name, "status": "done", "verdicts": done.get("verdicts")}
    probe = h["gpu_preflight"](P) if h.get("gpu_preflight") else {"skipped": True}
    ctx.token = h["hf_token"](dbutils_) if h.get("hf_token") else None
    results, verdicts = {}, {}

    def do(name, fn):
        prev = load_done(ctx, name)
        if prev is not None:
            ctx.log(f"{name}: reused (already published)")
            return prev
        t1 = time.time()
        r = fn()
        r["wall_s"] = round(time.time() - t1, 1)
        r["notebook"] = f"{NB_NAME} {NB_VERSION}"
        r["core_sha256"] = _sha256(bc.__file__)
        r["created_utc"] = _now()
        publish(ctx, name, r)
        return r

    if P.mode == "arm":
        arm = h["arm_loader"](ctx, P.arm)
        ctx.mem_base = h["mem_settle"]()
        weights = (arm.info.get("weights_mem_mib") or {}).get("alloc") if isinstance(arm.info, dict) else None
        ctx.log(f"{P.arm}: weights-only baseline {ctx.mem_base} MiB (load-time snapshot {weights})")
        try:
            if "g13" in P.checks:
                results["g13"] = do(f"arm_{P.arm}__g13.json", lambda: check_g13(ctx, arm))
                verdicts["G1-3 determinism floor"] = results["g13"]["determinism_floor"]["verdict"]
                verdicts["G1-3 k16 == unpruned"] = results["g13"]["k16_vs_none"]["verdict"]
                if "pop_on_vs_off" in results["g13"]:
                    verdicts["G1-3 pop on/off"] = results["g13"]["pop_on_vs_off"]["verdict"]
            if "g15" in P.checks:
                results["g15"] = do(f"arm_{P.arm}__g15.json", lambda: check_g15(ctx, arm))
                verdicts["G1-5 birth"] = results["g15"]["verdict"]
            if "g14" in P.checks:
                for which in P.g14_seqs:
                    results[f"g14_{which}"] = do(f"arm_{P.arm}__g14_{which}.json", lambda w=which: check_g14(ctx, arm, w))
                    verdicts[f"G1-4 slope {which}"] = results[f"g14_{which}"]["verdict"]
            if "leak" in P.checks:
                results["leak"] = do(f"arm_{P.arm}__leak.json", lambda: check_leak(ctx, arm))
                verdicts["leak (memory returns between sequences)"] = results["leak"]["verdict"]
        finally:
            info = dict(arm.info) if isinstance(arm.info, dict) else {}
            del arm
            h["mem_settle"]()
        arm_info = {k: info.get(k) for k in ("hf_id", "commit_hash", "attn_request", "params", "weights_mem_mib")}
    elif P.mode == "parity":
        for a in P.arms:
            results[a] = do(f"parity_{a}.json", lambda a=a: parity_one(ctx, a))
            verdicts[f"G1-8 parity {a}"] = results[a]["verdict"]
        arm_info = None
    else:
        frames = None
        for a in P.arms:
            if load_done(ctx, f"students_{a}.json") is None and frames is None:
                frames = student_frames(ctx)
            results[a] = do(f"students_{a}.json", lambda a=a: students_one(ctx, a, frames))
            verdicts[f"G1-9 {a} bf16 vs fp32"] = results[a]["verdict"]
        arm_info = None
    summary = {"notebook": f"{NB_NAME} {NB_VERSION}", "mode": P.mode, "arm": P.arm, "arms": P.arms, "tag": P.tag,
               "core_wait_s": core_wait_s,
               "verdicts": verdicts, "probe": probe, "arm_info": arm_info, "mem_baseline_mib": ctx.mem_base,
               "runs": ctx.runs, "params": vars(P), "core_sha256": _sha256(bc.__file__),
               "versions": bc.versions() if bc.TORCH_OK else None, "gpu": bc.gpu_info() if bc.TORCH_OK else None,
               "created_utc": _now(), "wall_s": round(time.time() - t0, 1)}
    publish(ctx, summary_name, summary)
    ctx.log("VERDICTS " + json.dumps(verdicts))
    return {"summary": summary_name, "status": "done", "verdicts": verdicts, "wall_s": summary["wall_s"]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local selftest (`TOL60_SELFTEST=1`)

# COMMAND ----------


def selftest():
    sys.path.insert(0, CODE_DIR)
    import test_tol_bench_core as tb
    from PIL import Image
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

    # ---- pure helpers
    m1 = np.zeros((10, 10), bool)
    m1[2:6, 2:6] = True
    m2 = np.zeros((10, 10), bool)
    m2[2:6, 2:8] = True
    r1, r2 = bc.rle_encode(m1), bc.rle_encode(m2)
    check("rle_iou: 16/24, missing side = 0, both missing = None",
          abs(rle_iou(r1, r2) - 16 / 24) < 1e-12 and rle_iou(r1, None) == 0.0 and rle_iou(None, None) is None)
    a = {"digest": {0: "x", 1: "y", 2: "z"}, "rows": {0: [[0]], 1: [[0]], 2: [[0]]},
         "masks": {0: {0: r1}, 1: {0: r1}, 2: {0: r1}}}
    b = {"digest": {0: "x", 1: "q", 2: "z"}, "rows": {0: [[0]], 1: [[0]], 2: [[0]]},
         "masks": {0: {0: r1}, 1: {0: r2}, 2: {0: r1}}}
    c = compare_runs(a, b)
    check("compare_runs: one differing frame -> not identical, IoU decoded there only",
          not c["bit_identical"] and c["first_digest_diff"] == 1 and c["mask_iou"]["n"] == 1
          and abs(c["mask_iou"]["mean"] - 16 / 24) < 1e-6)
    check("compare_runs: identical runs", compare_runs(a, a)["bit_identical"] and verdict_identity(compare_runs(a, a))
          == "PASS")
    floor = {"bit_identical": False, "mask_iou": {"mean": 0.6}}
    check("verdict_identity: within a non-identical determinism floor",
          verdict_identity(c, floor) == "PASS_WITHIN_FLOOR" and verdict_identity(c) == "FAIL"
          and verdict_identity(c, {"bit_identical": False, "mask_iou": {"mean": 0.9}}) == "FAIL")
    cobj = compare_runs(a, b, frames={1, 2}, objects=[0])
    check("compare_runs: frames / objects restriction", cobj["frames_compared"] == 2
          and cobj["per_object_mean_iou"]["0"] == round((16 / 24 + 1.0) / 2, 6))
    gt = np.array([[50, 0, 0, 0, 10, 10], [50, 1, 5, 5, 10, 10], [50, 2, 100, 100, 10, 10]], float)
    check("pick_birth_object: the least-overlapping pig", pick_birth_object(gt, 50)[0] == 2)
    fis = list(range(24, 1000, 25))
    check("slope_verdict: flat passes, +2 MiB/frame fails",
          slope_verdict(fis, [1000.0] * len(fis))["pass"] is True
          and slope_verdict(fis, [1000.0 + 2 * f for f in fis])["pass"] is False)
    mem = [{"fi": f, "alloc": 1000 + 2.0 * f, "session_mib": 500.0, "n_obj": 8, "by_group_mib": {"noncond:x": 500.0}}
           for f in fis]
    ex = explain_growth(mem)
    check("explain_growth: growth outside the session is attributed outside",
          abs(ex["outside_session"]["ols_mib_per_frame"] - 2.0) < 1e-9 and abs(ex["session"]["ols_mib_per_frame"]) < 1e-9)
    cd = census_diff({"groups": {"(1,)|bf16": [1, 1.0]}, "visible_mib": 1}, {"groups": {"(1,)|bf16": [5, 5.0]},
                                                                              "visible_mib": 5})
    check("census_diff: per-shape count / MiB change", cd["top_changes"]["(1,)|bf16"] == {"count": 4, "mib": 4.0})
    pc = parity_compare({0: {0: r1, 1: r1}, 1: {0: r1}}, {0: {0: r1, 1: r1}, 1: {0: r1}})
    check("parity_compare: identical -> IoU 1, pass", pc["pass"] and pc["mask_iou"]["mean"] == 1.0
          and pc["frame0_iou"] == {"0": 1.0, "1": 1.0})
    check("parity_compare: a missing object counts as IoU 0", not parity_compare({0: {0: r1}}, {0: {}})["pass"])
    f = np.random.default_rng(0).normal(size=(1, 8, 4, 4)).astype(np.float32)
    fc = feature_cosines(f, f + 1e-3)
    check("feature_cosines: near-identical features", fc["cos_global"] > 0.999 and fc["cos_patch_min"] > 0.99)
    class _BatchedEnc:
        def __init__(self):
            self.calls = []

        def _batch_encode_memories(self, inference_session, frame_idx, objects_needing_memory_encoding,
                                   high_res_masks_for_memory, object_score_logits_for_memory, is_mask_from_pts_per_obj):
            self.calls.append((list(objects_needing_memory_encoding), list(is_mask_from_pts_per_obj),
                               list(high_res_masks_for_memory)))

        def forward(self):
            self._batch_encode_memories(inference_session=None, frame_idx=4, objects_needing_memory_encoding=[0, 1, 2],
                                        high_res_masks_for_memory=["m0", "m1", "m2"],
                                        object_score_logits_for_memory=["s0", "s1", "s2"],
                                        is_mask_from_pts_per_obj=[False, False, True])
    fm = _BatchedEnc()
    with per_object_memory_flags(fm) as patched:
        fm.forward()
    fm.forward()
    check("G1-5 probe: the batch is split by each object's own flag, the class method is restored afterwards",
          patched and fm.calls[:2] == [([0, 1], [False, False], ["m0", "m1"]), ([2], [True], ["m2"])]
          and fm.calls[2] == ([0, 1, 2], [False, False, True], ["m0", "m1", "m2"]), str(fm.calls))
    with per_object_memory_flags(types.SimpleNamespace()) as patched2:
        pass
    check("G1-5 probe: a model without a batched encoder (EdgeTAM, stub) is left alone", patched2 is False)
    check("leak_rule: first <= tol, later <= first + drift",
          leak_rule([209.0, 209.0, 230.0], 512, 64) == (True, 64) and leak_rule([600.0], 512, 64)[0] is False
          and leak_rule([67.0, 170.0], 512, 64)[0] is False and leak_rule([], 512, 64)[0] is None)
    check("ensure_core: the J0b core passes at once; a wrong expected sha stops after the wait",
          ensure_core("", wait_s=0) == 0.0 and raises(lambda: ensure_core("deadbeef", wait_s=0, log=lambda *a: None),
                                                       RuntimeError, "lacks the J0b API"))
    check("params: mode / arm / checks / tags validated",
          raises(lambda: parse_params({"mode": "arm", "arm": "ES-L"}), ValueError)
          and raises(lambda: parse_params({"mode": "arm", "arm": "T", "checks": "g99"}), ValueError)
          and raises(lambda: parse_params({"mode": "parity", "arms": "T"}), ValueError)
          and raises(lambda: parse_params({"mode": "arm", "arm": "T", "pig_long": "pigtrack0001"}), ValueError)
          and parse_params({"mode": "students"}).arms == list(STUDENT_ARMS)
          and parse_params({"mode": "arm", "arm": "T", "g14_seqs": "pig, short"}).g14_seqs == ["pig", "short"]
          and raises(lambda: parse_params({"mode": "arm", "arm": "T", "g14_seqs": "long"}), ValueError))

    # ---- end to end on the stub tracker (arm mode, parity with a fake official runner)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = tmp.replace("\\", "/")
        reg_yaml = os.environ.get("TOL60_REGISTRY_YAML") or os.path.join(os.path.dirname(tol_paths.__file__),
                                                                         "paths_v1.yaml")
        R0 = tol_paths.load(path=reg_yaml)
        data = json.loads(json.dumps(R0.data))
        data["roots"].update(volume=tmp + "/vol", mount=tmp + "/mnt", mount_uri="file:" + tmp + "/mnt")
        R = tol_paths.Registry(data, source=reg_yaml, sha256=R0.sha256)
        H, W = 100, 160
        edi_world = {f"p{k}": tb.moving((10 + 18 * k, 10, 24 + 18 * k, 30), 0.05 * (k + 1)) for k in range(4)}
        pig_world = {f"q{k}": tb.static((10 + 30 * k, 50, 30 + 30 * k, 80)) for k in range(3)}
        clip = bc.EDI_TEST[0]
        date, num = clip[:10], clip[11:]

        def jpeg(p, v=0):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray(np.full((12, 16, 3), v % 256, np.uint8)).save(p)
        fd = R.path("datasets.edinburgh_pigs.decoded_frames", None, clip, "video1")
        for k in range(600):
            jpeg(f"{fd}/{k + 1:07d}.jpg", k)
        objs = []
        for k, name in enumerate(sorted(edi_world)):
            frs = []
            for fr in (0, 100, 300, 599):
                x1, y1, x2, y2 = edi_world[name](fr)
                frs.append({"frameNumber": fr, "visible": True,
                            "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}})
            objs.append({"id": k, "frames": frs})
        gtp = R.path("datasets.edinburgh_pigs.annotated", None, date, num, "output.json")
        os.makedirs(os.path.dirname(gtp), exist_ok=True)
        with open(gtp, "w", encoding="utf-8") as fh:
            json.dump({"objects": objs}, fh)
        txt = R.path("datasets.edinburgh_pigs.initial_prompts", None, clip, "video1", f"BoundingBoxes_{clip}.txt")
        os.makedirs(os.path.dirname(txt), exist_ok=True)
        with open(txt, "w", encoding="utf-8") as fh:
            for o in objs:
                b = o["frames"][0]["bbox"]
                fh.write(f"id {o['id']}: {b['x']},{b['y']},{b['width']},{b['height']}\n")
        kfw = bc.read_edinburgh_keyframes(gtp)
        gtrows = bc.edinburgh_gt_rows(kfw)
        world_by_id = {}
        for oid in sorted(kfw):                                 # the stub follows the INTERPOLATED GT boxes exactly
            rr = gtrows[gtrows[:, 1] == oid]
            boxes = {int(r[0]): (r[2], r[3], r[2] + r[4], r[3] + r[5]) for r in rr}
            world_by_id[f"p{oid}"] = (lambda bx: (lambda fi: bx.get(fi)))(boxes)
        for seq, n in (("pigtrack0002", 120), ("pigtrack0055", 98)):
            d = R.path("datasets.pigtrack.sequences", None, "test", seq)
            for k in range(n):
                jpeg(f"{d}/img1/{k + 1:08d}.jpg", 3 * k)
            with open(f"{d}/seqinfo.ini", "w", encoding="utf-8") as fh:
                fh.write(f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=10\nseqLength={n}\nimWidth=16\nimHeight=12\n"
                         "imExt=.jpg\npenID=3\nnight=0\n")
            os.makedirs(f"{d}/gt", exist_ok=True)
            with open(f"{d}/gt/gt.txt", "w", encoding="utf-8") as fh:
                for k, name in enumerate(sorted(pig_world)):
                    for fi in range(n):
                        x1, y1, x2, y2 = pig_world[name](fi)
                        fh.write(f"{fi + 1},{k},{x1},{y1},{x2 - x1},{y2 - y1},1,1,1\n")
        world = dict(world_by_id)
        world.update(pig_world)
        revp = bc.protocol_file_path(R, "hf_revisions_v1.json")
        os.makedirs(os.path.dirname(revp), exist_ok=True)
        with open(revp, "w", encoding="utf-8") as fh:
            json.dump({"revisions": {"facebook/sam2.1-hiera-tiny": "de431c4043854a71d8101e17995dfe596bf101a5"}}, fh)
        loads = collections.Counter()

        def loader(ctx_, name):
            loads[name] += 1
            arm = tb.stub_arm(world, H=H, W=W)
            arm.name = name
            arm.info = {"hf_id": f"stub/{name}", "commit_hash": "0" * 40, "weights_mem_mib": {"alloc": 1.0}}
            return arm

        def fake_official(ctx_, arm_name, jpeg_dir, prompts):
            arm = tb.stub_arm(world, H=H, W=W)
            cap = Capture(rle=True)
            bc.track_stream(arm, bc.PigTrackFrames(os.path.dirname(jpeg_dir)), prompts=prompts, on_frame=cap)
            return cap.record()["masks"], {"commit": "stub", "frames": len(cap.masks)}

        mem_vals = iter([1000.0] + [1010.0] * 100)
        hooks = {"local_root": tmp + "/local", "arm_loader": loader, "gpu_preflight": None, "hf_token": None,
                 "mem_settle": lambda: next(mem_vals), "official_runner": fake_official, "use_cuda": False,
                 "log": lambda *a: None}
        raw = {"mode": "arm", "arm": "T21", "g13_frames": "60", "g15_frames": "80", "birth_frame": "50",
               "leak_frames": "10", "require_gpu": "any", "progress_every": "0"}
        s = main(raw, R=R, hooks=hooks)
        v = s["verdicts"]
        g1 = R.out("paper1.root", "evidence", "g1", "j0b_v1")
        with open(f"{g1}/arm_T21__g15.json", encoding="utf-8") as fh:
            g15 = json.load(fh)
        with open(f"{g1}/arm_T21__g14_pig.json", encoding="utf-8") as fh:
            g14 = json.load(fh)
        with open(f"{g1}/arm_T21__leak.json", encoding="utf-8") as fh:
            lk = json.load(fh)
        check("arm mode (stub): G1-3 floor / k16 / pop all PASS on an ideal deterministic tracker",
              v.get("G1-3 determinism floor") == "PASS" and v.get("G1-3 k16 == unpruned") == "PASS"
              and v.get("G1-3 pop on/off") == "PASS", str(v))
        check("arm mode (stub): G1-5 birth PASS (existing ids unchanged, prompts kept, new id tracked)",
              v.get("G1-5 birth") == "PASS" and g15["clear_old_inputs"]["verdict"] == "PASS"
              and g15["new_object"]["emitted_frac"] == 1.0, json.dumps(g15)[:400])
        check("arm mode (stub): G1-4 runs the three sequences (no CUDA -> slope NOT_MEASURED, short = DIAGNOSTIC)",
              v.get("G1-4 slope pig") == "NOT_MEASURED" and v.get("G1-4 slope short") == "DIAGNOSTIC"
              and v.get("G1-4 slope edi") == "NOT_MEASURED" and g14["frames"] == 120 and g14["objects"] == 3)
        check("arm mode (stub): leak check uses the injected memory reader (+10 MiB <= tolerance -> PASS) and records "
              "that the stub processor keeps the session", v.get("leak (memory returns between sequences)") == "PASS"
              and g15["mechanism_probe_per_object_memory_flags"] == {"model_has_batched_memory_encoder": False}
              and lk["max_excess_mib"] == 10.0 and lk["unreleased_run"]["session_alive"] is True
              and any("StubProcessor" in hh for hh in lk["holders_seen"]), json.dumps(lk)[:400])
        n_loads = loads["T21"]
        s2 = main(raw, R=R, hooks=hooks)
        check("resume: a published summary is not recomputed", s2["status"] == "done" and loads["T21"] == n_loads)
        os.remove(f"{g1}/arm_T21.json")
        s3 = main(raw, R=R, hooks=hooks)
        check("resume: published check files are reused, the summary is rebuilt", s3["verdicts"] == v
              and loads["T21"] == n_loads + 1)
        sp = main({"mode": "parity", "arms": "T21,ET", "require_gpu": "any"}, R=R, hooks=hooks)
        check("parity mode (stub, fake official = same tracker): IoU 1, PASS",
              sp["verdicts"] == {"G1-8 parity T21": "PASS", "G1-8 parity ET": "PASS"}, str(sp))

    # ---- static audit
    with open(os.path.abspath(__file__), "rb") as fh:
        src = fh.read().decode("utf-8")
    try:
        compile(OFFICIAL_SCRIPT.format(rle_src=inspect.getsource(bc.rle_encode)), "official_track.py", "exec")
        ok_script = True
    except SyntaxError as e:
        ok_script = repr(e)
    check("official-code script renders and compiles", ok_script is True, str(ok_script))
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
            out[name] = os.environ.get("TOL60_" + name.upper(), default)
    return out


RAW = read_widgets()
print("widgets:", json.dumps(RAW, sort_keys=True))
SUMMARY = main(RAW, dbutils_=dbutils)
print(json.dumps(SUMMARY, indent=1, default=str))

# COMMAND ----------

if IS_NOTEBOOK:
    dbutils.notebook.exit(json.dumps(SUMMARY, default=str))
