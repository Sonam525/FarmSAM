# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N80 — Edinburgh frame-to-frame similarity in SAM 3 teacher-feature space (EXPLORATORY; CPU; non-test clips only)
# MAGIC
# MAGIC Part of the Edinburgh hard-case mining ANALYSIS stage (user request 2026-10-29). **No student is trained, no test clip
# MAGIC is read, nothing is appended to the benchmark protocol.** Question answered: how close is each frame of the 3 clips the
# MAGIC benchmark does not use (the candidate pool) to its nearest TRAINING frame, compared with (i) adjacent frames of one
# MAGIC clip, (ii) a training clip vs the other training clips, (iii) the held-out validation clip vs the training clips?
# MAGIC
# MAGIC Reads only existing SAM 3 backbone features `[1,1024,72,72]` fp16:
# MAGIC - the stored 2026-02 targets of the 4 `EDI_TRAIN` clips and `EDI_VAL` (`paper1_distillation/teacher_targets/edinburgh/<clip>/backbone_{fi:06d}.pt`);
# MAGIC - the Tol_79 targets of the 3 unused clips (`paper1_distillation/edi_mining/teacher_targets_unused_v1/seqs/<clip>/features/<clip>_f{fi:06d}.pt`),
# MAGIC   sha256-checked against that folder's `manifest.csv`.
# MAGIC Every `stride`-th decoded frame. Per frame three descriptors, each a set of unit vectors:
# MAGIC `glob` = mean over the 72x72 grid (1 vector); `g8` = 8x8 adaptive average pool (64 cell vectors); `g18` = 4x4 average
# MAGIC pool (18x18 = 324 cell vectors). Similarity of two frames = mean over cells of the cosine of the cell vectors at the SAME
# MAGIC position (so it measures layout AND appearance; `glob` measures appearance only).
# MAGIC Outputs (NEW folder `paper1_distillation/edi_mining/<out_tag>/`, refused if it exists): `index.csv`, `sim_{glob,g8,g18}.npy`
# MAGIC (float32 N x N), `desc_glob.npy` (float32 N x 1024), `RUN_RECORD.json` (last). The 4 EDI_TEST ids are refused by every path.

# COMMAND ----------

import csv
import hashlib
import io
import json
import os
import os.path as osp
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

NB_NAME, NB_VERSION = "Tol_80_edinburgh_nn_teacher", "tol80-2026-10-30.1"
T0 = time.time()
VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
STORED_ROOT = f"{VOL}/paper1_distillation/teacher_targets/edinburgh"
UNUSED_ROOT = f"{VOL}/paper1_distillation/edi_mining/teacher_targets_unused_v1"
EM_ROOT = f"{VOL}/paper1_distillation/edi_mining"
EDI_TRAIN = ("2019_11_05_000002", "2019_11_11_000028", "2019_11_11_000036", "2019_11_22_000010")
EDI_VAL = ("2019_11_28_000113",)
UNUSED = ("2019_11_05_000009", "2019_11_11_000016", "2019_11_15_000033")
CLIPS = EDI_TRAIN + EDI_VAL + UNUSED
ROLE = dict([(c, "train") for c in EDI_TRAIN] + [(c, "val") for c in EDI_VAL] + [(c, "unused") for c in UNUSED])
_TEST_IDS = ("2019_12_02_000005", "2019_12_02_000208", "2019_12_10_000060", "2019_12_10_000078")
NF = 600
SHAPE = (1, 1024, 72, 72)


class LeakError(RuntimeError):
    pass


def guard_path(p):
    s = str(p)
    if any(t in s for t in _TEST_IDS) or "2019_12_" in s:
        raise LeakError("refused: a test-clip path")
    return p


if set(CLIPS) & set(_TEST_IDS):
    raise LeakError("a test clip is among the clips")


def _w(name, default):
    try:
        dbutils.widgets.text(name, default)  # noqa: F821
        return dbutils.widgets.get(name).strip()  # noqa: F821
    except NameError:
        return os.environ.get("TOL80_" + name.upper(), default)


STRIDE = int(_w("stride", "2"))
OUT_TAG = _w("out_tag", "nn_teacher_v1")
THREADS = int(_w("threads", "32"))
if not OUT_TAG or "/" in OUT_TAG or OUT_TAG in ("measure_v1", "teacher_targets_unused_v1", "smoke_measure_v1",
                                                "smoke_teacher_targets_unused_v1"):
    raise ValueError(f"bad out_tag {OUT_TAG!r}")
if not (1 <= STRIDE <= 100):
    raise ValueError("stride")
OUT = guard_path(f"{EM_ROOT}/{OUT_TAG}")
if osp.exists(OUT):
    raise RuntimeError(f"{OUT} exists: this notebook writes only NEW folders")
LOCAL = f"/local_disk0/tol80/{OUT_TAG}"
shutil.rmtree(LOCAL, ignore_errors=True)
os.makedirs(LOCAL, exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count() or 1))
print(NB_NAME, NB_VERSION, "stride", STRIDE, "out", OUT, "threads", THREADS, "torch", torch.__version__,
      "cpus", os.cpu_count(), "runtime", os.environ.get("DATABRICKS_RUNTIME_VERSION"))

# COMMAND ----------

# ---- the frame list and the source files ---------------------------------------------------------------------------------
man_path = guard_path(f"{UNUSED_ROOT}/manifest.csv")
with open(man_path, "rb") as f:
    man_raw = f.read()
MAN_SHA = hashlib.sha256(man_raw).hexdigest()
MAN = {}
for r in csv.DictReader(io.StringIO(man_raw.decode("utf-8"))):
    MAN[(r["seq"], int(r["fi"]))] = (r["feature"], int(r["feature_bytes"]), r["feature_sha256"])
if len(MAN) != 3 * NF or {k[0] for k in MAN} != set(UNUSED):
    raise RuntimeError(f"unused manifest: {len(MAN)} rows, seqs {sorted({k[0] for k in MAN})}")

ROWS = []   # (row, clip, role, fi, path, expected bytes or None, expected sha or None)
for clip in CLIPS:
    for fi in range(0, NF, STRIDE):
        if ROLE[clip] == "unused":
            rel, nb, sh = MAN[(clip, fi)]
            p, eb, es = f"{UNUSED_ROOT}/{rel}", nb, sh
        else:
            p, eb, es = f"{STORED_ROOT}/{clip}/backbone_{fi:06d}.pt", None, None
        ROWS.append((len(ROWS), clip, ROLE[clip], fi, guard_path(p), eb, es))
for clip in EDI_TRAIN + EDI_VAL:
    n = len([x for x in os.listdir(guard_path(f"{STORED_ROOT}/{clip}")) if x.startswith("backbone_") and x.endswith(".pt")])
    if n != NF:
        raise RuntimeError(f"{clip}: {n} stored targets, expected {NF}")
N = len(ROWS)
print("frames", N, "per clip", N // len(CLIPS))

# COMMAND ----------

# ---- load every feature, build the three descriptors ---------------------------------------------------------------------
G8, G18 = 8, 18
D_GLOB = np.zeros((N, 1024), np.float32)
D_G8 = np.zeros((N, G8 * G8 * 1024), np.float32)
D_G18 = np.zeros((N, G18 * G18 * 1024), np.float32)
STATS = [None] * N


def unit_cells(x):
    """[1024, g, g] -> flattened [g*g*1024] with every cell vector L2-normalised."""
    v = x.permute(1, 2, 0).reshape(-1, 1024)
    return F.normalize(v, dim=1).reshape(-1)


def load_one(row):
    i, clip, role, fi, p, eb, es = row
    for attempt in range(4):
        try:
            with open(p, "rb") as f:
                raw = f.read()
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(5)
    sha = hashlib.sha256(raw).hexdigest()
    if eb is not None and (len(raw) != eb or sha != es):
        raise IOError(f"{osp.basename(p)}: {len(raw)} B sha {sha[:12]} differs from the manifest")
    t = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if tuple(t.shape) != SHAPE or t.dtype != torch.float16:
        raise RuntimeError(f"{osp.basename(p)}: {tuple(t.shape)} {t.dtype}")
    x = t[0].float()
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{osp.basename(p)}: non-finite values")
    D_GLOB[i] = F.normalize(x.mean(dim=(1, 2)), dim=0).numpy()
    D_G8[i] = unit_cells(F.adaptive_avg_pool2d(x[None], G8)[0]).numpy()
    D_G18[i] = unit_cells(F.avg_pool2d(x[None], 4)[0]).numpy()
    STATS[i] = {"bytes": len(raw), "sha256": sha, "mean": float(x.mean()), "std": float(x.std())}
    return i


torch.set_num_threads(1)      # many loader threads; intra-op parallelism only for the matrix products below
t_load = time.time()
done = 0
with ThreadPoolExecutor(max_workers=THREADS) as ex:
    for _ in ex.map(load_one, ROWS):
        done += 1
        if done % 300 == 0:
            print(f"loaded {done}/{N} in {time.time() - t_load:.1f} s")
T_LOAD = time.time() - t_load
if any(s is None for s in STATS):
    raise RuntimeError("a frame was not loaded")
print(f"load + descriptors {T_LOAD:.1f} s")

# COMMAND ----------

# ---- similarity matrices (mean over cells of the same-position cosine) ---------------------------------------------------
torch.set_num_threads(max(1, os.cpu_count() or 1))
t_sim = time.time()
SIMS = {}
for name, D, ncell in (("glob", D_GLOB, 1), ("g8", D_G8, G8 * G8), ("g18", D_G18, G18 * G18)):
    d = torch.from_numpy(D)
    S = (d @ d.T) / float(ncell)
    SIMS[name] = S.numpy().astype(np.float32)
    dg = np.diag(SIMS[name])
    if not np.allclose(dg, 1.0, atol=1e-3):
        raise RuntimeError(f"{name}: self-similarity {dg.min()}..{dg.max()} is not 1")
    if not np.allclose(SIMS[name], SIMS[name].T, atol=1e-4):
        raise RuntimeError(f"{name}: not symmetric")
    print(name, "sim range", float(SIMS[name].min()), float(SIMS[name].max()))
T_SIM = time.time() - t_sim

# ---- controls: per-frame stats vs the Tol_79 manifest (unused: feat_mean / feat_std recorded there) ----------------------
man_stats = {}
for r in csv.DictReader(io.StringIO(man_raw.decode("utf-8"))):
    man_stats[(r["seq"], int(r["fi"]))] = (float(r["feat_mean"]), float(r["feat_std"]))
dev = []
for i, clip, role, fi, *_ in ROWS:
    if role == "unused":
        m, s = man_stats[(clip, fi)]
        dev.append(max(abs(STATS[i]["mean"] - m), abs(STATS[i]["std"] - s)))
CONTROL = {"unused_sha256_checked": sum(1 for r in ROWS if r[2] == "unused"),
           "unused_stats_max_abs_dev_vs_manifest": float(max(dev)) if dev else None}
print("controls", CONTROL)

# COMMAND ----------

# ---- write locally, publish (never overwrite), RUN_RECORD last -------------------------------------------------------------
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


files = {}
with open(osp.join(LOCAL, "index.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["row", "clip", "role", "fi", "source", "bytes", "sha256", "feat_mean", "feat_std"])
    for i, clip, role, fi, p, *_ in ROWS:
        w.writerow([i, clip, role, fi, "unused_v1" if role == "unused" else "stored_2026_02", STATS[i]["bytes"],
                    STATS[i]["sha256"], f"{STATS[i]['mean']:.6f}", f"{STATS[i]['std']:.6f}"])
files["index.csv"] = None
for name, S in SIMS.items():
    np.save(osp.join(LOCAL, f"sim_{name}.npy"), S)
    files[f"sim_{name}.npy"] = None
np.save(osp.join(LOCAL, "desc_glob.npy"), D_GLOB)
files["desc_glob.npy"] = None
for k in list(files):
    with open(osp.join(LOCAL, k), "rb") as f:
        b = f.read()
    if any(t.encode() in b for t in _TEST_IDS) or b"2019_12_" in b:
        raise LeakError(f"{k} contains a test id")
    files[k] = {"sha256": sha_file(osp.join(LOCAL, k)), "bytes": osp.getsize(osp.join(LOCAL, k))}


def publish(name):
    dst = guard_path(f"{OUT}/{name}")
    if osp.exists(dst):
        raise FileExistsError(dst)
    for attempt in range(4):
        try:
            dbutils.fs.cp("file:" + osp.join(LOCAL, name), "dbfs:" + dst)  # noqa: F821
            if osp.getsize(dst) == osp.getsize(osp.join(LOCAL, name)):
                return
        except Exception as e:  # noqa: BLE001
            print("publish retry", name, repr(e)[:200])
        time.sleep(5)
    raise IOError(f"publish {name} failed")


os.makedirs(OUT, exist_ok=False)
t_pub = time.time()
for k in files:
    publish(k)
REC = {"notebook": NB_NAME, "version": NB_VERSION, "stride": STRIDE, "n_frames": N, "clips": list(CLIPS), "roles": ROLE,
       "sources": {"stored": STORED_ROOT, "unused": UNUSED_ROOT, "unused_manifest_sha256": MAN_SHA},
       "descriptors": {"glob": "L2(mean over 72x72)", "g8": "8x8 adaptive avg pool, each cell L2, sim = mean cell cosine",
                       "g18": "4x4 avg pool -> 18x18, each cell L2, sim = mean cell cosine"},
       "files": files, "controls": CONTROL,
       "timings_s": {"load_descriptors": round(T_LOAD, 1), "similarity": round(T_SIM, 1),
                     "publish": round(time.time() - t_pub, 1), "total": round(time.time() - T0, 1)},
       "versions": {"torch": torch.__version__, "numpy": np.__version__,
                    "runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION")},
       "test_ids": {"count": len(_TEST_IDS), "rule": "every path refused; every output scanned"}}
with open(osp.join(LOCAL, "RUN_RECORD.json"), "w", encoding="utf-8") as f:
    json.dump(REC, f, indent=1)
publish("RUN_RECORD.json")
print("done", REC["timings_s"])
dbutils.notebook.exit(json.dumps({"ok": True, "out": OUT, "n": N, "timings_s": REC["timings_s"]}))  # noqa: F821
