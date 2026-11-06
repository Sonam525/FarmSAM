# Databricks notebook source
# MAGIC %md
# MAGIC # Train the RF-DETR pig detectors
# MAGIC
# MAGIC Trains the box detectors used when tracking with `protocol=D` (`rfdetr==1.10.1`, Apache-2.0):
# MAGIC
# MAGIC | target | data | selection | published to |
# MAGIC |---|---|---|---|
# MAGIC | `d_pt_v1` | PigTrack train split (35 sequences) | best EMA val mAP50:95 on PigTrack val | `models.detector_pig_pt` |
# MAGIC | `d_edi_v1` | `d_pt_v1` fine-tuned on Edinburgh train clips | Edinburgh validation clip | `models.detector_pig_edi` |
# MAGIC
# MAGIC `d_pt_v1` starts as RF-DETR Small and moves to Medium only if Small's val AP50 is below 0.85.
# MAGIC
# MAGIC `track_bench` reads `<model dir>/checkpoint_best_total.pth` and `<model dir>/train_summary.json` (`resolution`,
# MAGIC `class_names`, `op_threshold`).
# MAGIC
# MAGIC | widget | values | meaning |
# MAGIC |---|---|---|
# MAGIC | `stage` | `train`, `bench`, `coco` | train (A100); bench = single-forward memory and latency (A10); coco = exports only |
# MAGIC | `target` | `d_pt_v1`, `d_edi_v1` | |
# MAGIC | `variant` | `small`, `medium` | `d_pt_v1` only |
# MAGIC | `mode` | `smoke`, `full`, `resume` | full and resume need this run's `smoke.json` |
# MAGIC | `attempt` | tag | run tag suffix; use a new one after a failed run |
# MAGIC | `run_tag` / `small_run_tag` | tag | override the derived run tag |
# MAGIC | `epochs` `patience` `train_stride` `sel_val_stride` `max_hours` | int / float | empty = the target default |
# MAGIC | `batch_size` `grad_accum` `num_workers` `seed` `smoke_train_n` `smoke_valid_n` `ckpt_sync_min` | int | |
# MAGIC | `require_gpu` | `A100`, `A10`, `any` | empty = A100 (train), A10 (bench), any (coco) |
# MAGIC | `bench_tags` / `bench_iters` | `d_pt_v1,d_edi_v1` / int | stage=bench |
# MAGIC
# MAGIC To continue after a crash, re-run the same task with `mode=resume`. A run with a `train_record.json` is complete.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm" "scipy" "rfdetr==1.10.1" "pytorch_lightning>=2.6,!=2.6.2,!=2.6.3,<3" "torchmetrics[detection]>=1.8.2,<1.9" "faster-coco-eval>=1.7.2" "pycocotools" "peft" "torch-hungarian==0.1.0rc0"

# COMMAND ----------

# MAGIC %pip uninstall -y opencv-python opencv-contrib-python

# COMMAND ----------

# MAGIC %pip install --no-deps --force-reinstall "opencv-python-headless==4.11.0.86"

# COMMAND ----------

try:
    dbutils.library.restartPython()   # noqa: F821
except NameError:
    pass

# COMMAND ----------

import collections
import contextlib
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
import types
import zipfile

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np

NB_NAME = "train_rfdetr_pig"
NB_VERSION = "train_rfdetr_pig-2026-10-14.3"
# %pip line known to work for transformers 5.2.0 + rfdetr 1.10.1 training on the A100, + headless OpenCV + the
# rfdetr[train] pin torch-hungarian==0.1.0rc0 (import name torch_linear_assignment): BOX-model training on CUDA solves the
# Hungarian matching with it (rfdetr/models/_assignment.py:74); J11 run 593021960562654 stopped on its absence.
PIP_SPEC = ('"numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm" "scipy" '
            '"rfdetr==1.10.1" "pytorch_lightning>=2.6,!=2.6.2,!=2.6.3,<3" "torchmetrics[detection]>=1.8.2,<1.9" '
            '"faster-coco-eval>=1.7.2" "pycocotools" "peft" "torch-hungarian==0.1.0rc0"; '
            'opencv-python-headless==4.11.0.86 (--no-deps)')
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None
if not IS_NOTEBOOK:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _this_dir():
    """Where bench_core.py / bench_core.py / farm_paths.py live (= track_bench's rule): env FARMSAM_CODE_DIR, else this
    notebook's workspace directory, else this file's directory (plain python)."""
    env = os.environ.get("FARMSAM_CODE_DIR")
    if env:
        return env
    if not IS_NOTEBOOK:
        return os.path.dirname(os.path.abspath(__file__))
    try:
        nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"cannot read this notebook's workspace path ({e!r}); set env FARMSAM_CODE_DIR") from e
    return "/Workspace" + os.path.dirname(nbp)


def _import_exact(name, dirs):
    """Import `name` from the FIRST of `dirs` holding `<name>.py`; nothing else on sys.path may shadow it (= track_bench)."""
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
os.environ.setdefault("FARMSAM_CODE_DIR", CODE_DIR)          # bc.load_rfdetr_detector -> bench_core.run_with_precision
bc = _import_exact("bench_core", [CODE_DIR])
_PATHS_DIRS = [CODE_DIR, os.environ.get("FARMSAM_PATHS_DIR")]
farm_paths = _import_exact("farm_paths", _PATHS_DIRS)


def _sha256(path):
    return bc.sha256_file(path)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


print(f"{NB_NAME} {NB_VERSION}: bench_core {bc.BENCH_VERSION} (sha256 {_sha256(bc.__file__)[:16]}), "
      f"farm_paths {farm_paths.__file__} (sha256 {_sha256(farm_paths.__file__)[:16]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Constants and pure helpers (no widget / dbutils / GPU)

# COMMAND ----------

REGISTRY_VERSION = 1                       # _admin/registry/paths_v1.yaml
CLASS_NAMES = ["pig"]
CAT_ID = 1
CATEGORIES = [{"id": CAT_ID, "name": "pig", "supercategory": "none"}]   # rfdetr: 1 class -> label 0 (rfdetr_api.md §c)
AP50_RULE = 0.85                           # Medium only if Small val AP50 < 0.85
SWEEP_TAUS = [round(0.10 + 0.01 * i, 2) for i in range(86)]   # 0.10 .. 0.95 (rounded to 0.01, §5.4); ties -> lower
SWEEP_IOU = 0.5                            # TP iff Hungarian-matched at box IoU >= 0.5 (TrackEval CLEAR rule)
AP_KEEP_TOP = 100                          # pycocotools maxDets=100: only the top-100 scores per image can count
PRED_PROBE_THR = 0.0                       # predict(threshold=0.0): every non-background row (strict > 0)
SMOKE_EPOCHS = 2
PROJ_OVERHEAD = 1.15                       
T_SETUP_FALLBACK_S = 60.0
TVAL_PROBE_N = 64                          # images timed with predict() to measure the per-image validation cost
EVAL_BATCH = 8
CHECKPOINT_INTERVAL = 5                    # != 1 so rfdetr writes last.ckpt every epoch (trainer.py:320-333)
RESUME_EMA_EPS = 1e-9
SYNC_SETTLE_S, SYNC_SETTLE_TRIES = 1.0, 8
SNAPSHOT_SET_TRIES, SNAPSHOT_SET_WAIT_S = 6, 20.0
RESUMABLE = ("last.ckpt", "checkpoint_best_ema.pth", "metrics.csv")
UPLOAD_ORDER = ("metrics.csv", "checkpoint_best_ema.pth", "last.ckpt")
GEN_COMPLETE = "COMPLETE.json"
GEN_RX = re.compile(r"^g(\d{6})$")
TAG_RX = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
PIGTRACK_SPLIT_N = {"train": 35, "val": 20, "test": 25}         # official split.txt
EDI_N_EXPECT = bc.EDI_N_FRAMES                                  # decoded frames per Edinburgh clip (600)
EDI_SPLITS = {"train": bc.EDI_TRAIN, "val": bc.EDI_VAL, "test": bc.EDI_TEST}
COCO_DIR = "coco_v1"
BENCH_DIR = "a10_forward_v1"
LOCAL_ROOT = "/local_disk0/tol71"
RF_HOME = "/local_disk0/rf_home"
BENCH_FRAMES = (("pigtrack", "test", "pigtrack0055", 1), ("edinburgh", "test", "2019_12_02_000005", 0))

VARIANTS = {   # rfdetr config.py:813-835 (box models: patch 16, 2 windows -> resolution % 32 == 0)
    "small": {"cls": "RFDETRSmall", "resolution": 512, "block": 32, "patch": 16, "pretrain": "rf-detr-small.pth"},
    "medium": {"cls": "RFDETRMedium", "resolution": 576, "block": 32, "patch": 16, "pretrain": "rf-detr-medium.pth"},
}
TARGETS = {
    "d_pt_v1": {"registry_key": "models.detector_pig_pt", "dataset": "pigtrack", "epochs": 40, "patience": 8,
                "train_stride": 2, "sel_val_stride": 4, "max_hours": 6.0,
                "train_set": "pigtrack_train", "sel_set": "pigtrack_val",
                "eval_sets": ("pigtrack_val", "pigtrack_test", "edinburgh_test")},
    "d_edi_v1": {"registry_key": "models.detector_pig_edi", "dataset": "edinburgh", "epochs": 30, "patience": 8,
                 "train_stride": 1, "sel_val_stride": 1, "max_hours": 2.0,
                 "train_set": "edinburgh_train_clips0-3", "sel_set": "edinburgh_val",
                 "eval_sets": ("edinburgh_val", "edinburgh_test")},
}
COCO_SETS = {   # name -> (dataset, split): the deterministic exports under detector_runs/coco_v1/
    "pigtrack_train": ("pigtrack", "train"), "pigtrack_val": ("pigtrack", "val"), "pigtrack_test": ("pigtrack", "test"),
    "edinburgh_train_clips0-3": ("edinburgh", "train"), "edinburgh_val": ("edinburgh", "val"),
    "edinburgh_test": ("edinburgh", "test"),
}
# TrainConfig whitelist (rfdetr config.py:1011-1270 + the two kwargs train() pops)
TRAIN_KW_WHITELIST = frozenset({
    "lr", "lr_encoder", "batch_size", "grad_accum_steps", "eval_batch_size", "auto_batch_target_effective",
    "auto_batch_max_targets_per_image", "auto_batch_ema_headroom", "epochs", "resume", "ema_decay", "ema_tau",
    "lr_drop", "checkpoint_interval", "skip_best_epochs", "smooth_alpha", "warmup_epochs", "lr_vit_layer_decay",
    "lr_component_decay", "drop_path", "cls_loss_coef", "dataset_file", "square_resize_div_64", "dataset_dir",
    "output_dir", "multi_scale", "expanded_scales", "do_random_resize_via_padding", "use_ema", "ema_update_interval",
    "eval_base_model", "num_workers", "weight_decay", "amp_dtype", "best_model_metric", "early_stopping",
    "early_stopping_patience", "early_stopping_min_delta", "early_stopping_use_ema", "progress_bar", "tensorboard",
    "wandb", "mlflow", "clearml", "project", "run", "class_names", "run_test", "eval_max_dets", "eval_interval",
    "log_per_class_metrics", "aug_config", "scale_jitter", "augmentation_backend", "notes", "accelerator",
    "clip_max_norm", "seed", "sync_bn", "strategy", "devices", "num_nodes", "lr_scheduler", "lr_scheduler_kwargs",
    "num_sanity_val_steps", "pin_memory", "persistent_workers", "prefetch_factor", "device", "resolution"})
RESUME_BOUND_HPARAMS = ("epochs", "batch_size", "grad_accum_steps", "lr", "lr_encoder", "weight_decay", "ema_decay",
                        "early_stopping_patience", "checkpoint_interval", "seed", "multi_scale", "expanded_scales",
                        "scale_jitter")
TS_REQUIRED = ("tag", "run_tag", "mode", "variant", "model_class", "resolution", "class_names", "num_classes",
               "checkpoint_best_total_sha256", "op_threshold", "val", "rfdetr_version", "train_data")


class T71Error(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- MOT / Edinburgh -> COCO
def mot_gt_census(rows, n_images):
    """What TrackEval's MOT17 preprocessing does with this GT (= the scorer census_gt): kept iff int(conf) != 0 and
    class == 1. A fractional conf, a class != 1 row with conf != 0, duplicates, negative ids or frame < 1 STOP the
    conversion (an amendment, never a silent filter, ); rows beyond the image count are dropped and
    counted (no image exists for them)."""
    n = len(rows)
    conf, cls = rows[:, 6], rows[:, 7]
    fr = rows[:, 0]
    pairs = collections.Counter((int(a), int(b)) for a, b in rows[:, :2])
    keep = (conf.astype(int) != 0) & (cls.astype(int) == 1)
    c = {"rows": int(n), "rows_evaluated": int(keep.sum()), "rows_conf0": int((conf.astype(int) == 0).sum()),
         "rows_conf_fractional": int((conf != np.round(conf)).sum()),
         "rows_class_not_1": int(((conf.astype(int) != 0) & (cls.astype(int) != 1)).sum()),
         "dup_frame_id": int(sum(v - 1 for v in pairs.values() if v > 1)),
         "negative_ids": int((rows[:, 1] < 0).sum()) if n else 0,
         "frame_lt_1": int((fr < 1).sum()) if n else 0,
         "noninteger_frame_or_id": int(((fr != np.round(fr)) | (rows[:, 1] != np.round(rows[:, 1]))).sum()) if n else 0,
         "rows_beyond_images": int(((fr > n_images) & keep).sum()) if n else 0,
         "conf_values": sorted({float(x) for x in conf})[:10], "class_values": sorted({float(x) for x in cls})[:10]}
    problems = [k for k in ("rows_conf_fractional", "rows_class_not_1", "dup_frame_id", "negative_ids", "frame_lt_1",
                            "noninteger_frame_or_id") if c[k]]
    return c, keep, problems


def clamp_box(x, y, w, h, W, H):
    """MOT box [x, y, w, h] -> COCO box clamped to the image [0, W] x [0, H]; None when the clamped area is 0."""
    x0, y0, x1, y1 = max(0.0, float(x)), max(0.0, float(y)), min(float(W), float(x) + float(w)), min(float(H), float(y) + float(h))
    if x1 <= x0 or y1 <= y0:
        return None, True
    clamped = (x0, y0, x1, y1) != (float(x), float(y), float(x) + float(w), float(y) + float(h))
    return [x0, y0, x1 - x0, y1 - y0], clamped


def _new_coco(name):
    return {"info": {"description": f"train_rfdetr_pig {name}", "version": COCO_DIR}, "images": [], "annotations": [],
            "categories": [dict(c) for c in CATEGORIES]}


def coco_from_mot(name, seq_records):
    """PigTrack MOT -> COCO (every frame; strides are applied later by subset_coco). seq_records: [{seq, n_images, width,
    height, night, rows (float64 [R,10] = bc.read_mot_txt)}]. Image ids 1.. in (seq, frame) order; file_name
    '<seq>/img1/<frame:08d>.jpg'. -> (coco, census)."""
    coco = _new_coco(name)
    cen = {"name": name, "n_seqs": 0, "n_images": 0, "n_boxes": 0, "n_clamped": 0, "n_zero_area_dropped": 0,
           "n_night_images": 0, "max_boxes_per_image": 0, "per_seq": {}}
    iid = aid = 0
    for rec in sorted(seq_records, key=lambda r: r["seq"]):
        seq, n_img, W, H = rec["seq"], int(rec["n_images"]), int(rec["width"]), int(rec["height"])
        c, keep, problems = mot_gt_census(rec["rows"], n_img)
        if problems:
            raise T71Error(f"{name}/{seq}: GT {problems} ({ {k: c[k] for k in problems} }) — TrackEval would not "
                           "evaluate these rows as the converter would; an amendment is needed, not a silent filter")
        by_frame = collections.defaultdict(list)
        for r in rec["rows"][keep]:
            by_frame[int(r[0])].append(r)
        n_box = 0
        for k in range(n_img):
            fr = k + 1
            iid += 1
            coco["images"].append({"id": iid, "file_name": f"{seq}/img1/{fr:08d}.jpg", "width": W, "height": H,
                                   "seq": seq, "frame": fr, "night": int(rec["night"])})
            nb = 0
            for r in sorted(by_frame.get(fr, []), key=lambda z: int(z[1])):
                box, clamped = clamp_box(r[2], r[3], r[4], r[5], W, H)
                if box is None:
                    cen["n_zero_area_dropped"] += 1
                    continue
                cen["n_clamped"] += int(bool(clamped))
                aid += 1
                coco["annotations"].append({"id": aid, "image_id": iid, "category_id": CAT_ID, "bbox": box,
                                            "area": box[2] * box[3], "iscrowd": 0, "track_id": int(r[1])})
                nb += 1
            n_box += nb
            cen["max_boxes_per_image"] = max(cen["max_boxes_per_image"], nb)
        cen["n_seqs"] += 1
        cen["n_images"] += n_img
        cen["n_boxes"] += n_box
        cen["n_night_images"] += n_img * int(rec["night"])
        cen["per_seq"][seq] = dict(c, n_images=n_img, width=W, height=H, night=int(rec["night"]), n_boxes=n_box)
    return coco, cen


def coco_from_edinburgh(name, clip_records):
    """Edinburgh keyframes -> COCO with the INTERPOLATED per-frame GT of (bc.edinburgh_gt_rows).
    clip_records: [{clip, keyframes, n_images, width, height}]; frame k = file '<clip>/video1/<k+1:07d>.jpg'."""
    coco = _new_coco(name)
    cen = {"name": name, "n_seqs": 0, "n_images": 0, "n_boxes": 0, "n_clamped": 0, "n_zero_area_dropped": 0,
           "n_night_images": 0, "max_boxes_per_image": 0, "per_seq": {}, "gt": "interpolated (edinburgh_gt_rows)"}
    iid = aid = 0
    for rec in sorted(clip_records, key=lambda r: r["clip"]):
        clip, n, W, H = rec["clip"], int(rec["n_images"]), int(rec["width"]), int(rec["height"])
        rows = bc.edinburgh_gt_rows(rec["keyframes"], n_frames=n, interpolate=True)
        by_frame = collections.defaultdict(list)
        for r in rows:
            by_frame[int(r[0])].append(r)
        n_box = 0
        for k in range(n):
            iid += 1
            coco["images"].append({"id": iid, "file_name": f"{clip}/video1/{k + 1:07d}.jpg", "width": W, "height": H,
                                   "seq": clip, "frame": k, "night": 0})
            nb = 0
            for r in sorted(by_frame.get(k, []), key=lambda z: int(z[1])):
                box, clamped = clamp_box(r[2], r[3], r[4], r[5], W, H)
                if box is None:
                    cen["n_zero_area_dropped"] += 1
                    continue
                cen["n_clamped"] += int(bool(clamped))
                aid += 1
                coco["annotations"].append({"id": aid, "image_id": iid, "category_id": CAT_ID, "bbox": box,
                                            "area": box[2] * box[3], "iscrowd": 0, "track_id": int(r[1])})
                nb += 1
            n_box += nb
            cen["max_boxes_per_image"] = max(cen["max_boxes_per_image"], nb)
        cen["n_seqs"] += 1
        cen["n_images"] += n
        cen["n_boxes"] += n_box
        cen["per_seq"][clip] = {"n_images": n, "width": W, "height": H, "n_ids": len(rec["keyframes"]),
                                "n_keyframes": int(sum(len(v) for v in rec["keyframes"].values())), "n_boxes": n_box}
    return coco, cen


def subset_coco(coco, keep_ids, name=None):
    keep_ids = set(keep_ids)
    out = _new_coco(name or coco["info"]["description"])
    out["images"] = [im for im in coco["images"] if im["id"] in keep_ids]
    out["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
    return out


def stride_ids(coco, stride):
    """Every `stride`-th frame of every sequence, starting at its first frame (deterministic)."""
    out, k_by_seq = [], collections.Counter()
    for im in coco["images"]:
        if k_by_seq[im["seq"]] % stride == 0:
            out.append(im["id"])
        k_by_seq[im["seq"]] += 1
    return out


def coco_bytes(coco):
    return (json.dumps(coco, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def ids_sha(ids):
    return hashlib.sha256(json.dumps(sorted(int(i) for i in ids)).encode("utf-8")).hexdigest()


def zip_root(names):
    """Common top folder inside a zip ('' if members sit at the root)."""
    tops = {n.split("/", 1)[0] for n in names if n and not n.startswith("__MACOSX")}
    if len(tops) == 1:
        t = next(iter(tops))
        if all(n.startswith(t + "/") or n == t + "/" for n in names if n and not n.startswith("__MACOSX")):
            return t + "/"
    return ""


def read_md5sums(text):
    out = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            h, _, name = ln.partition(" ")
            out[name.strip().lstrip("*")] = h.strip().lower()
    return out


def read_split_txt(text):
    out = collections.defaultdict(list)
    for ln in text.splitlines():
        p = ln.split()
        if len(p) == 2:
            out[p[1]].append(p[0])
    return {k: sorted(v) for k, v in out.items()}


def read_seqmap(text):
    return [ln.strip() for ln in text.splitlines()[1:] if ln.strip()]


# ---------------------------------------------------------------------------------------------- evaluation (pure)
def iou_xyxy(A, B):
    A, B = np.asarray(A, np.float64).reshape(-1, 4), np.asarray(B, np.float64).reshape(-1, 4)
    if not len(A) or not len(B):
        return np.zeros((len(A), len(B)))
    ix0 = np.maximum(A[:, None, 0], B[None, :, 0])
    iy0 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix1 = np.minimum(A[:, None, 2], B[None, :, 2])
    iy1 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _lsa(cost):
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except ImportError:                                        # the core's deterministic Hungarian (no scipy)
        pairs = bc.linear_assignment(cost)
        return (np.asarray([p[0] for p in pairs], int), np.asarray([p[1] for p in pairs], int))


def n_matches(iou, thr=SWEEP_IOU):
    """Max-sum-IoU Hungarian restricted to pairs with IoU >= thr (TrackEval CLEAR); -> number of such pairs."""
    if iou.size == 0:
        return 0
    m = np.where(iou >= thr - 1e-12, iou, 0.0)
    if not m.any():
        return 0
    r, c = _lsa(-m)
    return int((m[r, c] >= thr - 1e-12).sum())


def gt_xyxy_by_image(coco):
    out = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        out[a["image_id"]].append([x, y, x + w, y + h])
    return {k: np.asarray(v, np.float64).reshape(-1, 4) for k, v in out.items()}


def dets_by_image(dets):
    """dets: float [N, 6] (image_id, x, y, w, h, score) -> {image_id: (xyxy [n,4], scores [n]) sorted by -score}."""
    out = {}
    d = np.asarray(dets, np.float64).reshape(-1, 6)
    if not len(d):
        return out
    d = d[np.lexsort((-d[:, 5], d[:, 0]))]                  # by image id, then score descending (O(N log N))
    ids, starts = np.unique(d[:, 0], return_index=True)
    ends = list(starts[1:]) + [len(d)]
    for iid, a, b in zip(ids, starts, ends):
        s = d[a:b]
        out[int(iid)] = (np.stack([s[:, 1], s[:, 2], s[:, 1] + s[:, 3], s[:, 2] + s[:, 4]], 1), s[:, 5])
    return out


def f1_sweep(coco, dets, taus=SWEEP_TAUS):
    """Detections with score >= tau (= ProtocolD's score gate, bench_core: `float(s) >= tau` on the cached float32
    scores, rule 2) vs GT, per image; -> {tau: [tp, fp, fn]}."""
    G = gt_xyxy_by_image(coco)
    D = dets_by_image(dets)
    acc = {t: [0, 0, 0] for t in taus}
    for iid, g in G.items():
        boxes, scores = D.get(iid, (np.zeros((0, 4)), np.zeros(0)))
        iou = iou_xyxy(g, boxes)
        cache = {}
        for t in taus:
            k = int((scores >= t).sum())                 # scores sorted desc: the kept set is a prefix
            if k not in cache:
                cache[k] = n_matches(iou[:, :k]) if (k and len(g)) else 0
            tp = cache[k]
            acc[t][0] += tp
            acc[t][1] += k - tp
            acc[t][2] += len(g) - tp
    return acc


def op_threshold_from_sweep(acc):
    f1 = {t: (2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0) for t, (tp, fp, fn) in acc.items()}
    best = max(f1.values())
    op = min(t for t, v in f1.items() if v >= best - 1e-12)          # ties -> the lower threshold
    return round(op, 2), f1


def coco_ap(coco, dets, img_ids=None):
    """pycocotools bbox AP: GT = the COCO export (TrackEval-evaluated rows / interpolated
    Edinburgh GT), detections = top-100 per image. -> {AP50_95, AP50, AP75, AR100, n_images, n_gt, n_dets}."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    sel = set(img_ids) if img_ids is not None else {im["id"] for im in coco["images"]}
    n_gt = sum(1 for a in coco["annotations"] if a["image_id"] in sel)
    d = np.asarray(dets, np.float64).reshape(-1, 6)
    d = d[np.isin(d[:, 0].astype(int), list(sel))] if len(d) else d
    out = {"n_images": len(sel), "n_gt": n_gt, "n_dets": int(len(d))}
    if not len(d) or not n_gt:
        out.update(AP50_95=0.0, AP50=0.0, AP75=0.0, AR100=0.0)
        return out
    gt = COCO()
    gt.dataset = {"images": [{"id": im["id"], "width": im["width"], "height": im["height"]} for im in coco["images"]],
                  "annotations": [dict(a) for a in coco["annotations"]], "categories": [dict(c) for c in CATEGORIES]}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gt.createIndex()
        res = np.concatenate([d, np.full((len(d), 1), CAT_ID, np.float64)], 1)
        dt = gt.loadRes(res)
        E = COCOeval(gt, dt, "bbox")
        E.params.imgIds = sorted(sel)
        E.evaluate()
        E.accumulate()
        E.summarize()
    s = [float(x) for x in E.stats]
    out.update(AP50_95=s[0], AP50=s[1], AP75=s[2], AR100=s[8])
    return out


# ---------------------------------------------------------------------------------------------- training bookkeeping
def check_train_kwargs(kw):
    bad = sorted(k for k in kw if k not in TRAIN_KW_WHITELIST)
    if bad:
        raise T71Error(f"train() kwargs not in the TrainConfig whitelist: {bad}")
    if "dataset_dir" not in kw:
        raise T71Error("dataset_dir is required (config.py:1062)")
    if kw.get("amp_dtype") != "bf16":
        raise T71Error(f"amp_dtype must be 'bf16', got {kw.get('amp_dtype')!r}")
    if kw.get("batch_size") == "auto":
        raise T71Error("batch_size='auto' is forbidden (pinned ints)")
    return True


def resolve_precision_replica(amp, amp_dtype, cuda_available, bf16_supported):
    """See rfdetr training/trainer.py:558-621."""
    if not amp or not cuda_available:
        return "32-true"
    if amp_dtype == "fp16":
        return "16-mixed"
    return "bf16-mixed" if bf16_supported else "16-mixed"


def read_metrics_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize_metrics_csv(rows, key="val/ema_mAP_50_95", key50="val/ema_mAP_50"):
    """Best epoch / value of the EMA track BestModelCallback monitors (val/ema_mAP_50_95 for a box model,
    trainer.py:351-357); falls back to the mirrored val/mAP_50_95 when the ema key is absent. Strict '>' = the
    callback's own improvement rule (best_model.py:570)."""
    used = key
    if rows and not any(r.get(key, "") not in ("", None) for r in rows):
        k2 = key.replace("val/ema_", "val/")
        if any(r.get(k2, "") not in ("", None) for r in rows):
            used, key = k2, k2
    keys50 = [key50, key50.replace("val/ema_", "val/")] if used.startswith("val/ema_") else [key50.replace("val/ema_", "val/")]
    best, best_ep, best50, epochs = -1.0, None, None, set()
    for r in rows:
        v = r.get(key, "")
        if v in ("", None):
            continue
        try:
            v = float(v)
            ep = int(float(r.get("epoch", "nan")))
        except ValueError:
            continue
        epochs.add(ep)
        if v > best:
            best, best_ep = v, ep
            v50 = [r.get(k, "") for k in keys50 if r.get(k, "") not in ("", None)]
            best50 = float(v50[0]) if v50 else None
    return {"key": used, "epochs_run": (max(epochs) + 1) if epochs else 0, "best_epoch": best_ep,
            "best_val_mAP_50_95": (best if best >= 0 else None), "best_val_mAP_50": best50}


def project_hours(T_total, T_setup, t_val_img, smoke_train_n, smoke_valid_n, N_train, N_valid, N_eval, epochs,
                  smoke_epochs=SMOKE_EPOCHS, overhead=PROJ_OVERHEAD):
    """ips = smoke_epochs * smoke_train_n / (T_total - T_setup - smoke_epochs * smoke_valid_n * t_val_img);
    train h = epochs * (N_train / ips + N_valid * t_val_img) * overhead; eval h = N_eval * t_val_img * overhead."""
    denom = T_total - T_setup - smoke_epochs * smoke_valid_n * t_val_img
    if denom <= 0:
        raise T71Error(f"projection denominator {denom:.1f} s <= 0 (T_total {T_total:.1f}, T_setup {T_setup:.1f}, "
                       f"val {smoke_epochs * smoke_valid_n * t_val_img:.1f}): the smoke timing is inconsistent")
    ips = smoke_epochs * smoke_train_n / denom
    h_train = epochs * (N_train / ips + N_valid * t_val_img) * overhead / 3600.0
    h_eval = N_eval * t_val_img * overhead / 3600.0
    return {"ips": ips, "t_val_img": t_val_img, "hours_train": h_train, "hours_eval": h_eval,
            "hours_full": h_train + h_eval, "denom_s": denom}


def stop_gate(hours, max_hours, epochs):
    if hours > max_hours:
        raise T71Error(f"STOP GATE: projected {hours:.2f} h > max_hours {max_hours:.2f} h. Cheapest fallbacks (your call, "
                       f"never auto-applied): epochs {epochs} -> {max(1, round(epochs * 25 / 40))}, or a larger "
                       "train_stride; re-run mode=smoke with a new attempt")
    return True


def resume_highwater_action(best_ema_in_ckpt, csv_best, csv_best_epoch, ema_epoch, ema_exists, ckpt_epoch=None,
                            eps=RESUME_EMA_EPS):
    """Resume high-water check: 'ok' | 'patch' (in-flight snapshot: last.ckpt one epoch
    behind the csv best epoch) | raise (inconsistent triple)."""
    if not ema_exists:
        if (csv_best is not None and csv_best > 0.0) or (best_ema_in_ckpt is not None and best_ema_in_ckpt > 0.0):
            raise T71Error(f"resume: checkpoint_best_ema.pth absent but the EMA metric already improved (csv {csv_best} @ "
                           f"{csv_best_epoch}, mark {best_ema_in_ckpt}); rfdetr would backfill the FINAL EMA weights as best")
        return "ok"
    if csv_best is None:
        raise T71Error("resume: checkpoint_best_ema.pth exists but metrics.csv has no validation row")
    if best_ema_in_ckpt is None:
        raise T71Error("resume: last.ckpt carries no BestModelCallback state (_best_ema)")
    if ema_epoch is not None and csv_best_epoch is not None and int(ema_epoch) != int(csv_best_epoch):
        raise T71Error(f"resume: checkpoint_best_ema.pth epoch {ema_epoch} != metrics.csv best epoch {csv_best_epoch}")
    if best_ema_in_ckpt > csv_best + eps:
        raise T71Error(f"resume: last.ckpt mark {best_ema_in_ckpt} > metrics.csv best {csv_best} (csv truncated)")
    if best_ema_in_ckpt >= csv_best - eps:
        return "ok"
    if ckpt_epoch is None or csv_best_epoch is None or int(ckpt_epoch) != int(csv_best_epoch) - 1:
        raise T71Error(f"resume: mark {best_ema_in_ckpt} < csv best {csv_best} @ {csv_best_epoch} but last.ckpt is epoch "
                       f"{ckpt_epoch} (only an in-flight snapshot one epoch behind may carry a lower mark)")
    return "patch"


def gen_name(n):
    return f"g{int(n):06d}"


def list_generations(root):
    """[(n, path, complete)] newest first."""
    if not os.path.isdir(root):
        return []
    out = []
    for nm in os.listdir(root):
        m = GEN_RX.match(nm)
        if m:
            p = os.path.join(root, nm)
            out.append((int(m.group(1)), p, os.path.exists(os.path.join(p, GEN_COMPLETE))))
    return sorted(out, key=lambda z: -z[0])


def file_sig(path):
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


def stable_snapshot(src, dst, settle_s=SYNC_SETTLE_S, tries=SYNC_SETTLE_TRIES, sleep=time.sleep):
    """Copy only when (size, mtime) is unchanged across settle_s before AND after the copy."""
    for _ in range(tries):
        a = file_sig(src)
        if a is None:
            return None
        sleep(settle_s)
        if file_sig(src) != a:
            continue
        shutil.copyfile(src, dst)
        if file_sig(src) == a and os.path.getsize(dst) == a[0]:
            return a
    raise T71Error(f"{src} kept changing across {tries} x {settle_s:.0f} s")


def validate_zip_file(path):
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
    except zipfile.BadZipFile as e:
        raise T71Error(f"{path} is not a complete zip container: {e}")
    if bad is not None:
        raise T71Error(f"{path}: CRC failure in member {bad!r}")
    return True


def snapshot_set(src_dir, stg_dir, consistency, files=RESUMABLE, required=("last.ckpt", "metrics.csv"),
                 tries=SNAPSHOT_SET_TRIES, wait_s=SNAPSHOT_SET_WAIT_S, sleep=time.sleep, snap=None):
    """All local snapshots back to back, retaken until no source moved and consistency() passes."""
    snap = snap or stable_snapshot
    last_err = None
    for k in range(tries):
        if k:
            sleep(wait_s)
        sigs = {fn: snap(os.path.join(src_dir, fn), os.path.join(stg_dir, fn)) for fn in files}
        if any(sigs[fn] is None for fn in required):
            return None
        moved = [fn for fn in files if file_sig(os.path.join(src_dir, fn)) != sigs[fn]]
        if moved:
            last_err = f"sources changed while the set was taken: {moved}"
            continue
        paths = {fn: (os.path.join(stg_dir, fn) if sigs[fn] is not None else None) for fn in files}
        try:
            return sigs, consistency(paths)
        except T71Error as e:
            last_err = f"inconsistent set: {e}"
    raise T71Error(f"no consistent snapshot set after {tries} attempts ({last_err})")


def publish_generation(stg_paths, shas, vol_root, n, record, copier):
    """One snapshot set into a NEW directory vol_root/g<n>/ (never over an existing one, never pruned: the Volume
    rule forbids deleting); files, then GEN_COMPLETE last."""
    gdir = os.path.join(vol_root, gen_name(n))
    if os.path.exists(gdir):
        raise T71Error(f"generation {gdir} already exists — generations are never rewritten")
    os.makedirs(gdir)
    for fn in UPLOAD_ORDER:
        if stg_paths.get(fn) is None:
            continue
        copier(stg_paths[fn], os.path.join(gdir, fn))
    rec = dict(record)
    rec.update({"generation": n, "files": {fn: shas[fn] for fn in UPLOAD_ORDER if stg_paths.get(fn) is not None}})
    tmp = os.path.join(os.path.dirname(stg_paths["metrics.csv"]), GEN_COMPLETE)
    bc.write_json_atomic(tmp, rec)
    copier(tmp, os.path.join(gdir, GEN_COMPLETE))
    return gdir


def resume_mismatches(rec_hp, cur_hp, rec_prov, cur_prov, prov_keys=("run_tag", "target", "variant", "resolution",
                                                                       "train_ids_sha256", "sel_ids_sha256")):
    out = []
    for k in prov_keys:
        if (rec_prov or {}).get(k) != cur_prov.get(k):
            out.append((k, (rec_prov or {}).get(k), cur_prov.get(k)))
    for k in RESUME_BOUND_HPARAMS:
        if (rec_hp or {}).get(k) != cur_hp.get(k):
            out.append(("train." + k, (rec_hp or {}).get(k), cur_hp.get(k)))
    return out


# ---------------------------------------------------------------------------------------------- publish guard (models/)
def publish_roots(R):
    return [R.out("paper1.root"), R.out("models.detector_pig_pt"), R.out("models.detector_pig_edi")]


def assert_t71_target(dst, roots):
    """train_rfdetr_pig writes ONLY under paper1_distillation/ and models/rfdetr_box_pig/{d_pt_v1,d_edi_v1}/ (registry R.out),
    never under Tol/, the mount or anywhere else; '..' refused."""
    p = str(dst).replace("\\", "/")
    seg = [s for s in p.split("/") if s]
    if ".." in seg:
        raise PermissionError(f"{dst}: '..' in a publish target")
    if p.startswith(("/dbfs/mnt/", "dbfs:/mnt/", "/mnt/")) or "playbehavior" in seg:
        raise PermissionError(f"{dst}: publishing to the blob mount is forbidden")
    for r in roots:
        r = str(r).replace("\\", "/").rstrip("/")
        if p.startswith(r + "/"):
            if "Tol" in [s for s in p[len(r):].split("/") if s]:
                raise PermissionError(f"{dst}: never under Tol/")
            return dst
    raise PermissionError(f"{dst}: train_rfdetr_pig publishes only under {roots}")


def publish_new(src, dst, roots, dbutils_=None, attempts=bc.CP_ATTEMPTS, sleep_s=bc.CP_SLEEP_S):
    """publish_no_overwrite semantics for every train_rfdetr_pig target (identical -> 'identical'; different -> FileExistsError).
    paper1_distillation/ targets go through bc.publish_no_overwrite itself; models/ targets (which the core's guard
    reserves for paper1_distillation/) through this replica of it, behind assert_t71_target."""
    assert_t71_target(dst, roots)
    if "/" + bc.PAPER1_SEGMENT + "/" in str(dst).replace("\\", "/"):
        return bc.publish_no_overwrite(src, dst, dbutils_, attempts, sleep_s)
    bc.assert_no_secrets(src)
    if os.path.exists(dst):
        if bc.sha256_file(dst) == bc.sha256_file(src):
            return "identical"
        raise FileExistsError(f"{dst} exists with different content; refusing to overwrite")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    want = os.path.getsize(src)
    if dbutils_ is not None:
        t_start = time.time() - 2.0

        def _cp():
            if os.path.exists(dst):
                if bc.sha256_file(dst) == bc.sha256_file(src):
                    return
                if not (os.path.getsize(dst) < want and os.path.getmtime(dst) >= t_start):
                    raise FileExistsError(f"{dst} appeared during the publish with different content")
            dbutils_.fs.cp("file:" + os.path.abspath(src), bc.dbfs_uri(dst))
            got = os.path.getsize(dst) if os.path.exists(dst) else -1
            if got != want:
                raise IOError(f"published {dst}: {got} B vs {want} B local")
        bc._retry(_cp, f"publish {os.path.basename(dst)}", attempts, sleep_s)
        return "copied"
    part = dst + ".part"
    shutil.copyfile(src, part)
    os.replace(part, dst)
    return "copied"


def train_summary_check(ts):
    """The contract track_bench (detector_resolution / load_rfdetr_detector) and J6 (op_threshold) read."""
    miss = [k for k in TS_REQUIRED if k not in ts]
    if miss:
        raise T71Error(f"train_summary.json lacks {miss}")
    if not isinstance(ts["resolution"], int) or ts["resolution"] <= 0 or ts["resolution"] % VARIANTS[ts["variant"]]["block"]:
        raise T71Error(f"train_summary resolution {ts['resolution']!r} invalid for {ts['variant']}")
    if not re.match(r"^[0-9a-f]{64}$", str(ts["checkpoint_best_total_sha256"])):
        raise T71Error("checkpoint_best_total_sha256 is not a sha256")
    if list(ts["class_names"]) != CLASS_NAMES or int(ts["num_classes"]) != 1:
        raise T71Error(f"class_names {ts['class_names']} / num_classes {ts['num_classes']} != {CLASS_NAMES} / 1")
    if not (0.10 <= float(ts["op_threshold"]) <= 0.95) or round(float(ts["op_threshold"]), 2) != float(ts["op_threshold"]):
        raise T71Error(f"op_threshold {ts['op_threshold']} is not a 0.01-rounded value in [0.10, 0.95]")
    if ts["mode"] not in ("full", "resume"):
        raise T71Error(f"mode {ts['mode']!r}")
    return True


def decide_medium(small_val_ap50, rule=AP50_RULE):
    return bool(small_val_ap50 < rule)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

WIDGETS = collections.OrderedDict([
    ("stage", "train"), ("target", "d_pt_v1"), ("variant", "small"), ("mode", "smoke"), ("attempt", "a1"),
    ("run_tag", ""), ("small_run_tag", ""), ("epochs", ""), ("patience", ""), ("train_stride", ""),
    ("sel_val_stride", ""), ("max_hours", ""), ("batch_size", "8"), ("grad_accum", "2"), ("num_workers", "12"),
    ("seed", "0"), ("smoke_train_n", "1000"), ("smoke_valid_n", "200"), ("ckpt_sync_min", "45"),
    ("require_gpu", ""), ("bench_tags", "d_pt_v1,d_edi_v1"), ("bench_iters", "50"),
])


def read_widgets():
    out = {}
    for k, v in WIDGETS.items():
        if IS_NOTEBOOK:
            try:
                dbutils.widgets.text(k, v)
            except Exception:  # noqa: BLE001
                pass
            out[k] = dbutils.widgets.get(k)
        else:
            out[k] = os.environ.get("TOL71_" + k.upper(), v)
    return out


def _int(v, name, lo=None):
    try:
        x = int(str(v).strip())
    except ValueError:
        raise ValueError(f"{name}={v!r} is not an int")
    if lo is not None and x < lo:
        raise ValueError(f"{name}={x} < {lo}")
    return x


def parse_params(raw):
    g = {k: str(raw.get(k, d) if raw.get(k) is not None else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace()
    P.stage = g["stage"].lower()
    if P.stage not in ("train", "bench", "coco"):
        raise ValueError(f"stage={P.stage!r}: train | bench | coco")
    P.target = g["target"]
    if P.target not in TARGETS:
        raise ValueError(f"target={P.target!r}: {sorted(TARGETS)}")
    P.variant = g["variant"].lower()
    if P.variant not in VARIANTS:
        raise ValueError(f"variant={P.variant!r}: {sorted(VARIANTS)}")
    if P.target == "d_edi_v1" and P.stage == "train" and P.variant != "small" and g["variant"]:
        _log(f"note: variant={P.variant} ignored for d_edi_v1 (inherits d_pt_v1's variant)")
    P.mode = g["mode"].lower()
    if P.mode not in ("smoke", "full", "resume"):
        raise ValueError(f"mode={P.mode!r}: smoke | full | resume")
    for k in ("attempt", "run_tag", "small_run_tag"):
        if g[k] and not TAG_RX.match(g[k]):
            raise ValueError(f"{k}={g[k]!r} is not one path segment")
    P.attempt, P.run_tag, P.small_run_tag = g["attempt"] or "a1", g["run_tag"], g["small_run_tag"]
    t = TARGETS[P.target]
    P.epochs = _int(g["epochs"], "epochs", 1) if g["epochs"] else t["epochs"]
    P.patience = _int(g["patience"], "patience", 1) if g["patience"] else t["patience"]
    P.train_stride = _int(g["train_stride"], "train_stride", 1) if g["train_stride"] else t["train_stride"]
    P.sel_val_stride = _int(g["sel_val_stride"], "sel_val_stride", 1) if g["sel_val_stride"] else t["sel_val_stride"]
    P.max_hours = float(g["max_hours"]) if g["max_hours"] else t["max_hours"]
    P.batch_size = _int(g["batch_size"], "batch_size", 1)
    P.grad_accum = _int(g["grad_accum"], "grad_accum", 1)
    P.num_workers = _int(g["num_workers"], "num_workers", 0)
    P.seed = _int(g["seed"], "seed", 0)
    P.smoke_train_n = _int(g["smoke_train_n"], "smoke_train_n", 8)
    P.smoke_valid_n = _int(g["smoke_valid_n"], "smoke_valid_n", 8)
    P.ckpt_sync_min = float(g["ckpt_sync_min"])
    P.require_gpu = g["require_gpu"] or {"train": "A100", "bench": "A10", "coco": "any"}[P.stage]
    if P.require_gpu not in ("A100", "A10", "any"):
        raise ValueError(f"require_gpu={P.require_gpu!r}")
    P.bench_tags = [s.strip() for s in g["bench_tags"].split(",") if s.strip()]
    if any(s not in TARGETS for s in P.bench_tags):
        raise ValueError(f"bench_tags {P.bench_tags}: subset of {sorted(TARGETS)}")
    P.bench_iters = _int(g["bench_iters"], "bench_iters", 1)
    return P

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production hooks (rfdetr, CUDA, staging, listing)

# COMMAND ----------


def _prod_rf():
    os.environ["RF_HOME"] = RF_HOME
    os.makedirs(RF_HOME, exist_ok=True)
    import torch
    mp_before = torch.get_float32_matmul_precision()
    import importlib.metadata as md
    import rfdetr  # noqa: F401  (import side effects: matmul 'high', RF_HOME read)
    import torch_linear_assignment  # noqa: F401  (rfdetr[train]: box matching on CUDA; fail before any work)
    from rfdetr import RFDETRMedium, RFDETRSmall, from_checkpoint
    from rfdetr.assets.model_weights import ModelWeights

    def registry_md5(fn):
        a = ModelWeights.from_filename(fn)
        return None if a is None else a.md5_hash

    def build(variant, resolution, pretrain_weights=None):
        cls = {"small": RFDETRSmall, "medium": RFDETRMedium}[variant]
        kw = {"resolution": int(resolution), "num_classes": 1}
        if pretrain_weights:
            kw["pretrain_weights"] = pretrain_weights
        return cls(**kw)

    return types.SimpleNamespace(build=build, from_checkpoint=lambda p, resolution: from_checkpoint(p, resolution=int(resolution)),
                                 registry_md5=registry_md5, version=md.version("rfdetr"), matmul_before=mp_before)


def _prod_cuda():
    import torch
    return types.SimpleNamespace(
        available=torch.cuda.is_available(), name=lambda: torch.cuda.get_device_name(0),
        bf16=torch.cuda.is_bf16_supported, alloc=torch.cuda.memory_allocated, peak=torch.cuda.max_memory_allocated,
        reserved=torch.cuda.memory_reserved, peak_reserved=torch.cuda.max_memory_reserved,
        reset=torch.cuda.reset_peak_memory_stats, sync=torch.cuda.synchronize, empty=torch.cuda.empty_cache,
        oom=torch.cuda.OutOfMemoryError)


def _prod_ckpt_read(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _prod_ckpt_write(obj, path):
    import torch
    torch.save(obj, path)


def _prod_list_remote(path):
    """{relative path: size} of every file under a Volume directory (dbutils.fs.ls, recursive)."""
    out = {}

    def rec(p, rel):
        for fi in dbutils.fs.ls(bc.dbfs_uri(p)):
            nm = fi.name.rstrip("/")
            if fi.name.endswith("/") or (getattr(fi, "isDir", None) and fi.isDir()):
                rec(p.rstrip("/") + "/" + nm, rel + nm + "/")
            else:
                out[rel + nm] = int(fi.size)
    rec(path, "")
    return out


def _prod_detector_loader(ckpt_local, resolution):
    return bc.load_rfdetr_detector(ckpt_local, resolution=resolution)


def _prod_param_count(ckpt_local):
    sd = _prod_ckpt_read(ckpt_local)
    m = sd.get("model", sd)
    return int(sum(int(v.numel()) for v in m.values() if hasattr(v, "numel")))


def _prod_gpu_preflight(P, cuda):
    if P.require_gpu == "any" and not cuda.available:
        return {"name": None, "class": "CPU"}
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError("PYTORCH_CUDA_ALLOC_CONF lacks expandable_segments:True (job cluster spec)")
    if not cuda.available:
        raise RuntimeError("no CUDA device")
    name = cuda.name()
    up = name.upper()
    cls = "A100" if "A100" in up else ("A10" if re.search(r"A10(?!\d)", up) else name)
    if P.require_gpu != "any" and cls != P.require_gpu:
        raise RuntimeError(f"GPU {name} (class {cls}) but require_gpu={P.require_gpu} (user GPU policy: A100 for detector "
                           "training, A10 for the forward benchmark)")
    if P.stage == "train" and not cuda.bf16():
        raise RuntimeError("no bf16 support (bf16 only, never fp16)")
    return {"name": name, "class": cls}


def _notebook_source():
    if not IS_NOTEBOOK:
        with open(os.path.abspath(__file__), "rb") as f:
            return f.read()
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    for cand in ("/Workspace" + nbp, "/Workspace" + nbp + ".py"):
        try:
            if os.path.isfile(cand):
                with open(cand, "rb") as f:
                    b = f.read()
                if b.lstrip(b"\xef\xbb\xbf").startswith(b"# Databricks notebook source"):
                    return b
        except Exception:  # noqa: BLE001
            pass
    import base64
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ExportFormat
    return base64.b64decode(WorkspaceClient().workspace.export(nbp, format=ExportFormat.SOURCE).content)


PROD_HOOKS = {
    "local_root": LOCAL_ROOT, "rf": _prod_rf, "cuda": _prod_cuda, "ckpt_read": _prod_ckpt_read,
    "ckpt_write": _prod_ckpt_write, "list_remote": _prod_list_remote, "detector_loader": _prod_detector_loader,
    "param_count": _prod_param_count, "gpu_preflight": _prod_gpu_preflight, "notebook_source": _notebook_source,
    "log": _log, "sleep": time.sleep, "stage_threads": 4, "settle_s": SYNC_SETTLE_S, "snapshot_wait_s": SNAPSHOT_SET_WAIT_S,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data: staging (verified against the catalog), COCO exports, rfdetr dataset directories

# COMMAND ----------


def _read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def pigtrack_lists(ctx):
    """split.txt (registry) cross-checked against seqmaps/pigtrack-<split>.txt; counts 35/20/25; test = PIGTRACK_TEST."""
    R = ctx.R
    sp = read_split_txt(_read_text(R.path("datasets.pigtrack.split_file")))
    out = {}
    for split, n in PIGTRACK_SPLIT_N.items():
        names = sp.get(split, [])
        sm = R.path("datasets.pigtrack.seqmaps", None, f"pigtrack-{split}.txt")
        if os.path.isfile(sm) and sorted(read_seqmap(_read_text(sm))) != names:
            raise T71Error(f"seqmaps/pigtrack-{split}.txt differs from split.txt")
        if len(names) != n:
            raise T71Error(f"PigTrack {split}: {len(names)} sequences in split.txt, expected {n}")
        out[split] = names
    if out["test"] != sorted(bc.PIGTRACK_TEST):
        raise T71Error("PigTrack test list differs from bench_core.PIGTRACK_TEST")
    return out


def stage_pigtrack_seq(ctx, split, seq, md5s):
    """raw/<seq>.zip -> /local_disk0 (ONE large-file dbutils.fs.cp; a per-file copy of the frames runs at ~5 MB/s),
    md5 vs MD5SUMS.txt, unzip, then verify names + sizes against the catalog's sequences/<split>/<seq>/ (the frames
    track_bench tracks). Marker makes it idempotent."""
    R, h = ctx.R, ctx.h
    d = os.path.join(h["local_root"], "stage", "pigtrack", split, seq)
    marker = os.path.join(d, ".t71_staged.json")
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8") as f:
            return d, json.load(f)
    t0 = time.time()
    zl = os.path.join(h["local_root"], "stage", "zips", seq + ".zip")
    bc.stage_file(R.path("datasets.pigtrack.raw", None, seq + ".zip"), zl, ctx.dbutils)
    t_copy = time.time() - t0
    md5 = hashlib.md5()
    with open(zl, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            md5.update(chunk)
    if md5s.get(seq + ".zip") != md5.hexdigest():
        raise T71Error(f"{seq}.zip md5 {md5.hexdigest()} != MD5SUMS.txt {md5s.get(seq + '.zip')}")
    tmp = d + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zl) as zf:
        names = zf.namelist()
        root = zip_root(names)
        for m in zf.infolist():
            if m.is_dir() or m.filename.startswith("__MACOSX"):
                continue
            rel = m.filename[len(root):]
            if not rel or ".." in rel.split("/"):
                raise T71Error(f"{seq}.zip: unexpected member {m.filename!r}")
            dst = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(m) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
    local = {}
    for dp, _dn, fn in os.walk(tmp):
        for name in fn:
            p = os.path.join(dp, name)
            local[os.path.relpath(p, tmp).replace(os.sep, "/")] = os.path.getsize(p)
    cat = h["list_remote"](R.path("datasets.pigtrack.sequences", None, split, seq))
    cat = {k: v for k, v in cat.items() if not k.split("/")[-1].startswith(".")}
    local = {k: v for k, v in local.items() if not k.split("/")[-1].startswith(".")}
    if local != cat:
        miss, extra = sorted(set(cat) - set(local))[:5], sorted(set(local) - set(cat))[:5]
        diff = sorted(k for k in set(cat) & set(local) if cat[k] != local[k])[:5]
        raise T71Error(f"{seq}: unzipped files differ from the catalog sequences/{split}/{seq} (missing {miss}, extra "
                       f"{extra}, size {diff})")
    shutil.rmtree(d, ignore_errors=True)                     # a partial copy of an earlier, interrupted stage (local)
    os.replace(tmp, d)
    os.remove(zl)
    rec = {"seq": seq, "split": split, "zip_md5": md5.hexdigest(), "zip_root": root, "files": len(local),
           "bytes": int(sum(local.values())), "copy_s": round(t_copy, 1), "total_s": round(time.time() - t0, 1),
           "verified_against": "catalog sequences/ (names + sizes)"}
    bc.write_json_atomic(marker, rec)
    return d, rec


def stage_edinburgh_clip(ctx, clip):
    R = ctx.R
    root = os.path.join(ctx.h["local_root"], "stage", "edinburgh")
    d = os.path.join(root, clip, "video1")
    rec = bc.stage_tree(R.path("datasets.edinburgh_pigs.decoded_frames", None, clip, "video1"), d, ctx.dbutils)
    date, num = re.match(r"^(\d{4}_\d{2}_\d{2})_(\d{6})$", clip).groups()
    gt, gsha = bc.stage_file(R.path("datasets.edinburgh_pigs.annotated", None, date, num, "output.json"),
                             os.path.join(root, clip, "output.json"), ctx.dbutils)
    return {"clip": clip, "frames_dir": d, "gt": gt, "gt_sha256": gsha, "staged": rec}


def _parallel(fn, items, n):
    from concurrent.futures import ThreadPoolExecutor
    if n <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, items))


def build_sets(ctx, names):
    """Stage what the named COCO sets need and build them. -> {name: {coco, census, root, staging}}."""
    from PIL import Image
    out, t0 = {}, time.time()
    need_pt = sorted({COCO_SETS[n][1] for n in names if COCO_SETS[n][0] == "pigtrack"})
    need_edi = sorted({COCO_SETS[n][1] for n in names if COCO_SETS[n][0] == "edinburgh"})
    if need_pt:
        lists = pigtrack_lists(ctx)
        md5s = read_md5sums(_read_text(ctx.R.path("datasets.pigtrack.raw", None, "MD5SUMS.txt")))
        for split in need_pt:
            seqs = lists[split]
            ctx.log(f"staging PigTrack {split}: {len(seqs)} zips")
            staged = _parallel(lambda s: stage_pigtrack_seq(ctx, split, s, md5s), seqs, ctx.h["stage_threads"])
            recs = []
            for seq, (d, srec) in zip(seqs, staged):
                fr = bc.PigTrackFrames(d)
                with Image.open(fr.files[0]) as im:
                    if im.size != (fr.width, fr.height):
                        raise T71Error(f"{seq}: image {im.size} != seqinfo {(fr.width, fr.height)}")
                gt_cat = ctx.R.path("datasets.pigtrack.sequences", None, split, seq, "gt", "gt.txt")
                if _sha256(gt_cat) != _sha256(fr.gt_path):
                    raise T71Error(f"{seq}: staged gt.txt differs from the catalog's")
                recs.append({"seq": seq, "n_images": fr.n, "width": fr.width, "height": fr.height, "night": fr.night,
                             "rows": bc.read_mot_txt(gt_cat), "warnings": fr.warnings})
            name = f"pigtrack_{split}"
            coco, cen = coco_from_mot(name, recs)
            cen["warnings"] = {r["seq"]: r["warnings"] for r in recs if r["warnings"]}
            cen["gt_source"] = "catalog datasets/pigtrack/sequences/<split>/<seq>/gt/gt.txt"
            out[name] = {"coco": coco, "census": cen, "root": os.path.join(ctx.h["local_root"], "stage", "pigtrack", split),
                         "staging": {s: r for s, (_d, r) in zip(seqs, staged)}}
    if need_edi:
        clips = sorted({c for sp in need_edi for c in EDI_SPLITS[sp]})
        ctx.log(f"staging Edinburgh clips {clips}")
        staged = {r["clip"]: r for r in _parallel(lambda c: stage_edinburgh_clip(ctx, c), clips, ctx.h["stage_threads"])}
        for split in need_edi:
            recs = []
            for clip in EDI_SPLITS[split]:
                s = staged[clip]
                fr = bc.EdinburghFrames(s["frames_dir"], expect_n=EDI_N_EXPECT)
                with Image.open(fr.files[0]) as im:
                    W, H = im.size
                recs.append({"clip": clip, "keyframes": bc.read_edinburgh_keyframes(s["gt"]), "n_images": fr.n,
                             "width": W, "height": H})
            name = {"train": "edinburgh_train_clips0-3", "val": "edinburgh_val", "test": "edinburgh_test"}[split]
            coco, cen = coco_from_edinburgh(name, recs)
            cen["gt_sha256"] = {c: staged[c]["gt_sha256"] for c in EDI_SPLITS[split]}
            out[name] = {"coco": coco, "census": cen, "root": os.path.join(ctx.h["local_root"], "stage", "edinburgh"),
                         "staging": {c: staged[c]["staged"] for c in EDI_SPLITS[split]}}
    ctx.log(f"built {sorted(out)} in {time.time() - t0:.0f} s: " +
            ", ".join(f"{n} {out[n]['census']['n_images']} img / {out[n]['census']['n_boxes']} boxes" for n in sorted(out)))
    return out


def publish_coco(ctx, sets):
    """Deterministic exports -> detector_runs/coco_v1/<name>.json + census_<name>.json (identical = skip)."""
    rec = {}
    for name, s in sorted(sets.items()):
        loc = os.path.join(ctx.h["local_root"], "pub", COCO_DIR)
        os.makedirs(loc, exist_ok=True)
        for fn, b in ((f"{name}.json", coco_bytes(s["coco"])),
                      (f"census_{name}.json", (json.dumps(s["census"], sort_keys=True, indent=1) + "\n").encode())):
            p = os.path.join(loc, fn)
            with open(p, "wb") as f:
                f.write(b)
            st = publish_new(p, ctx.R.out("paper1.detector_runs", COCO_DIR, fn), ctx.roots, ctx.dbutils)
            rec[fn] = {"sha256": hashlib.sha256(b).hexdigest(), "status": st}
    return rec


def write_rf_dataset(ds_dir, split_dirname, coco, src_root):
    """<ds_dir>/<train|valid>/_annotations.coco.json (standard keys) + one symlink per sequence to the staged frames."""
    d = os.path.join(ds_dir, split_dirname)
    os.makedirs(d, exist_ok=True)
    tops = sorted({im["file_name"].split("/", 1)[0] for im in coco["images"]})
    for t in tops:
        link, target = os.path.join(d, t), os.path.join(src_root, t)
        if not os.path.isdir(target):
            raise T71Error(f"staged directory {target} missing")
        if not os.path.lexists(link):
            try:
                os.symlink(target, link, target_is_directory=True)
            except OSError:
                if os.name != "nt":                      # Linux (Databricks): a symlink failure is a real error
                    raise
                shutil.copytree(target, link)            # Windows without the symlink privilege
    miss = [im["file_name"] for im in coco["images"] if not os.path.isfile(os.path.join(d, im["file_name"]))]
    if miss:
        raise T71Error(f"{len(miss)} images referenced but absent under {d}, e.g. {miss[:3]}")
    slim = {"images": [{"id": im["id"], "file_name": im["file_name"], "width": im["width"], "height": im["height"]}
                       for im in coco["images"]],
            "annotations": [{k: a[k] for k in ("id", "image_id", "category_id", "bbox", "area", "iscrowd")}
                            for a in coco["annotations"]],
            "categories": [dict(c) for c in CATEGORIES]}
    with open(os.path.join(d, "_annotations.coco.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f)
    return d

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction, evaluation, checkpoint reload

# COMMAND ----------


def det_rows(image_id, d, num_classes, keep_top=AP_KEEP_TOP):
    """supervision.Detections -> float rows (image_id, x, y, w, h, score), background rows (class_id >= num_classes,
    rfdetr_api.md §d trap) dropped, top `keep_top` by score."""
    if d is None or len(d) == 0:
        return np.zeros((0, 6))
    k = np.asarray(d.class_id) < num_classes
    xy = np.asarray(d.xyxy, np.float64)[k]
    sc = np.asarray(d.confidence, np.float64)[k]
    o = np.argsort(-sc, kind="stable")[:keep_top]
    xy, sc = xy[o], sc[o]
    return np.column_stack([np.full(len(sc), image_id, np.float64), xy[:, 0], xy[:, 1], xy[:, 2] - xy[:, 0],
                            xy[:, 3] - xy[:, 1], sc]) if len(sc) else np.zeros((0, 6))


def predict_images(model, images, root, batch=EVAL_BATCH, log=_log, what=""):
    """predict() in batches (prefetching the next batch's decode on threads) at threshold 0 -> float [N, 6]."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    nc = int(model.model_config.num_classes)

    def load(im):
        with Image.open(os.path.join(root, im["file_name"])) as x:
            return np.asarray(x.convert("RGB"))
    out, t0 = [], time.time()
    chunks = [images[i:i + batch] for i in range(0, len(images), batch)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = [ex.submit(load, im) for im in chunks[0]] if chunks else []
        for j, ch in enumerate(chunks):
            arrs = [f.result() for f in fut]
            fut = [ex.submit(load, im) for im in chunks[j + 1]] if j + 1 < len(chunks) else []
            res = model.predict(arrs, threshold=PRED_PROBE_THR, include_source_image=False)
            if not isinstance(res, list):
                res = [res]
            for im, d in zip(ch, res):
                out.append(det_rows(im["id"], d, nc))
            if log and (j + 1) % 250 == 0:
                log(f"  predict {what}: {(j + 1) * batch}/{len(images)} ({(time.time() - t0) / 60:.1f} min)")
    dt = time.time() - t0
    rows = np.concatenate(out, 0) if out else np.zeros((0, 6))
    if not np.isfinite(rows).all():
        raise T71Error(f"non-finite detector output on {what}")
    return rows, {"seconds": round(dt, 1), "img_per_s": round(len(images) / max(dt, 1e-9), 2)}


def evaluate_set(name, s, rows):
    coco = s["coco"]
    res = {"all": coco_ap(coco, rows)}
    if name.startswith("pigtrack"):
        for lab, flag in (("day", 0), ("night", 1)):
            ids = [im["id"] for im in coco["images"] if int(im["night"]) == flag]
            if ids:
                res[lab] = coco_ap(coco, rows, ids)
    return res


def reload_best(ctx, out_dir, variant, res):
    """from_checkpoint with the resolution PINNED (strip_checkpoint drops model_config); class,
    num_classes, PE checked. The in-memory model after train() holds LIVE weights: always reload."""
    p = os.path.join(out_dir, "checkpoint_best_total.pth")
    if not os.path.exists(p):
        raise T71Error(f"{p} missing after train()")
    m = ctx.rf.from_checkpoint(p, res)
    mc = m.model_config
    if int(mc.num_classes) != 1:
        raise T71Error(f"reloaded num_classes {mc.num_classes} != 1")
    if list(m.class_names) != CLASS_NAMES:
        raise T71Error(f"reloaded class_names {m.class_names} != {CLASS_NAMES}")
    if int(mc.resolution) != res:
        raise T71Error(f"reloaded resolution {mc.resolution} != {res}")
    if int(mc.positional_encoding_size) != res // int(mc.patch_size):
        raise T71Error(f"reloaded PE {mc.positional_encoding_size} != {res}//{mc.patch_size}")
    return m, p


def read_ckpt_meta(ctx, path):
    try:
        ck = ctx.h["ckpt_read"](path)
    except Exception as e:  # noqa: BLE001
        ctx.log(f"could not read checkpoint metadata from {path}: {e}")
        return {}
    out = {k: ck.get(k) for k in ("epoch", "best_total_source", "global_step", "model_name", "rfdetr_version")}
    del ck
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training stage (smoke / full / resume)

# COMMAND ----------


class FirstStepProbe:
    """T_setup = time to the first CUDA allocation jump >= 1 GB above the pre-train() baseline (a seg run used
    2 GB for Seg at 624 / 2304x1296; a box Small batch at <= 672^2 allocates less)."""

    def __init__(self, cuda, jump_gb=1.0):
        self.cuda, self.jump = cuda, jump_gb * 1e9
        self.t_first = None
        self._stop = threading.Event()

    def start(self):
        self.t0, self.base = time.perf_counter(), self.cuda.alloc()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def _run(self):
        while not self._stop.is_set() and self.t_first is None:
            try:
                if self.cuda.alloc() - self.base >= self.jump:
                    self.t_first = time.perf_counter() - self.t0
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)

    def stop(self):
        self._stop.set()
        self.th.join(timeout=2)
        return self.t_first


class CkptSync:
    """Background resume-state publisher, without pruning (Volume content is never deleted): every
    `every_min` one consistent (last.ckpt, checkpoint_best_ema.pth, metrics.csv) set -> resume_state/g<NNNNNN>/."""

    def __init__(self, ctx, out_dir, vol_root, every_min, record, consistency, copier):
        self.ctx, self.out_dir, self.vol_root, self.every = ctx, out_dir, vol_root, every_min * 60.0
        self.record, self.consistency, self.copier = record, consistency, copier
        self.stg = os.path.join(ctx.h["local_root"], "sync", os.path.basename(out_dir))
        shutil.rmtree(self.stg, ignore_errors=True)
        os.makedirs(self.stg, exist_ok=True)
        self._stop, self._lock = threading.Event(), threading.Lock()
        self.log, self.last_sigs, self.last_gen = [], None, None

    def start(self):
        if self.every > 0:
            self.th = threading.Thread(target=self._run, daemon=True)
            self.th.start()
        return self

    def sync_once(self):
        with self._lock:
            snap = lambda s, d: stable_snapshot(s, d, settle_s=self.ctx.h["settle_s"], sleep=self.ctx.h["sleep"])  # noqa: E731
            res = snapshot_set(self.out_dir, self.stg, self.consistency, sleep=self.ctx.h["sleep"],
                               wait_s=self.ctx.h["snapshot_wait_s"], snap=snap)
            if res is None:
                return None
            sigs, info = res
            if sigs == self.last_sigs:
                return self.last_gen
            stg = {fn: (os.path.join(self.stg, fn) if sigs[fn] is not None else None) for fn in RESUMABLE}
            shas = {fn: bc.sha256_file(p) for fn, p in stg.items() if p is not None}
            os.makedirs(self.vol_root, exist_ok=True)
            gens = list_generations(self.vol_root)
            n = (gens[0][0] + 1) if gens else 1
            rec = dict(self.record)
            rec.update({"tick_utc": _now(), "state": info})
            t0 = time.time()
            publish_generation(stg, shas, self.vol_root, n, rec, self.copier)
            self.last_sigs, self.last_gen = sigs, gen_name(n)
            self.log.append({"t": rec["tick_utc"], "generation": gen_name(n), "s": round(time.time() - t0, 1),
                             "ckpt_epoch": info.get("ckpt_epoch"), "action": info.get("action")})
            self.ctx.log(f"  resume generation {gen_name(n)} published (ckpt epoch {info.get('ckpt_epoch')}, "
                         f"{info.get('action')}, {time.time() - t0:.0f} s)")
            return gen_name(n)

    def _run(self):
        while not self._stop.wait(self.every):
            try:
                self.sync_once()
            except Exception as e:  # noqa: BLE001
                self.ctx.log(f"  resume sync tick skipped (retry next tick): {type(e).__name__}: {str(e)[:200]}")
                self.log.append({"error": str(e)[:200]})

    def stop(self):
        self._stop.set()
        if hasattr(self, "th"):
            self.th.join()


def make_copier(ctx):
    def copier(src, dst):
        assert_t71_target(dst, ctx.roots)
        if os.path.exists(dst):
            raise FileExistsError(f"{dst} exists (generation files are never rewritten)")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if ctx.dbutils is not None:
            bc._retry(lambda: ctx.dbutils.fs.cp("file:" + os.path.abspath(src), bc.dbfs_uri(dst)),
                      f"generation {os.path.basename(dst)}")
            if os.path.getsize(dst) != os.path.getsize(src):
                raise IOError(f"{dst}: size differs after copy")
        else:
            shutil.copyfile(src, dst)
    return copier


def resume_consistency_factory(ctx):
    def resume_consistency(paths):
        for fn, p in paths.items():
            if p is not None and (fn.endswith(".ckpt") or fn.endswith(".pth")):
                validate_zip_file(p)
        ck = ctx.h["ckpt_read"](paths["last.ckpt"])
        marks = [float(s["_best_ema"]) for s in (ck.get("callbacks") or {}).values() if isinstance(s, dict) and "_best_ema" in s]
        info = {"ckpt_epoch": ck.get("epoch"), "global_step": ck.get("global_step"), "best_ema_in_ckpt": max(marks) if marks else None}
        del ck
        ema_p = paths.get("checkpoint_best_ema.pth")
        ema = read_ckpt_meta(ctx, ema_p) if ema_p else {}
        if ema_p and ema.get("epoch") is None:
            raise T71Error(f"{ema_p}: no readable 'epoch'")
        ms = summarize_metrics_csv(read_metrics_csv(paths["metrics.csv"]))
        info.update(csv_best=ms["best_val_mAP_50_95"], csv_best_epoch=ms["best_epoch"], csv_epochs_run=ms["epochs_run"],
                    ema_epoch=ema.get("epoch"))
        info["action"] = resume_highwater_action(info["best_ema_in_ckpt"], info["csv_best"], info["csv_best_epoch"],
                                                 info["ema_epoch"], ema_p is not None, ckpt_epoch=info["ckpt_epoch"])
        return info
    return resume_consistency


def train_kwargs(P, dataset_dir, output_dir, epochs, bs, ga, prov, resume=None):
    """rfdetr TrainConfig. Every literal is an rfdetr default (cited) or a stated project choice."""
    kw = dict(dataset_dir=dataset_dir, output_dir=output_dir, dataset_file="roboflow",
              epochs=epochs, batch_size=bs, grad_accum_steps=ga,       # 8 x 2 = effective 16 (rfdetr's recommended total)
              eval_batch_size=EVAL_BATCH,                              # project: box validation is light
              lr=1e-4, lr_encoder=1.5e-4, weight_decay=1e-4,           # rfdetr defaults (config.py:1011, 1012, 1095)
              clip_max_norm=0.1,                                       # rfdetr default (config.py:1230)
              amp_dtype="bf16",                                        # project: bf16 only, never fp16
              multi_scale=True, expanded_scales=True, scale_jitter=True,   # rfdetr defaults, pinned explicitly
              augmentation_backend="torchvision",                      # project: never auto-pick Albumentations
              use_ema=True, ema_decay=0.993,                           # rfdetr defaults
              early_stopping=True, early_stopping_patience=P.patience, early_stopping_min_delta=0.001,
              best_model_metric="map",                                 # -> val/ema_mAP_50_95 for a box model
              checkpoint_interval=CHECKPOINT_INTERVAL, eval_interval=1,
              num_workers=P.num_workers, seed=P.seed, tensorboard=False, wandb=False, mlflow=False,
              progress_bar=None, run_test=False, class_names=list(CLASS_NAMES), notes=json.dumps(prov), device="cuda")
    if resume:
        kw["resume"] = resume
    check_train_kwargs(kw)
    return kw


def medium_gate(ctx, small_tag):
    p = ctx.R.path("paper1.detector_runs", None, small_tag, "decision.json")
    if not os.path.isfile(p):
        raise T71Error(f"variant=medium needs the Small run's decision ({p}); run the Small full task first")
    with open(p, encoding="utf-8") as f:
        dec = json.load(f)
    return dec


def init_from_dpt(ctx):
    """d_edi_v1 starts from the PUBLISHED d_pt_v1 (registry models.detector_pig_pt): checkpoint sha == summary."""
    R = ctx.R
    ts_p = R.path("models.detector_pig_pt", None, "train_summary.json")
    if not os.path.isfile(ts_p):
        raise T71Error(f"d_edi_v1 needs the published d_pt_v1 ({ts_p} missing)")
    with open(ts_p, encoding="utf-8") as f:
        ts = json.load(f)
    train_summary_check(ts)
    loc = os.path.join(ctx.h["local_root"], "models", "d_pt_v1", "checkpoint_best_total.pth")
    local, sha = bc.stage_file(R.path("models.detector_pig_pt", None, "checkpoint_best_total.pth"), loc, ctx.dbutils,
                               ts["checkpoint_best_total_sha256"])
    return {"summary": ts, "local": local, "sha256": sha, "variant": ts["variant"], "resolution": int(ts["resolution"]),
            "run_tag": ts["run_tag"]}


def gpu_mem_gb(cuda):
    return round(cuda.peak() / 1e9, 3) if cuda is not None and cuda.available else None


def _publish_json(ctx, obj, dst, local_name):
    p = os.path.join(ctx.pub, local_name)
    bc.write_json_atomic(p, obj)
    return publish_new(p, dst, ctx.roots, ctx.dbutils)


def run_train(ctx):
    P, R, h, log = ctx.P, ctx.R, ctx.h, ctx.log
    T = TARGETS[P.target]
    init = init_from_dpt(ctx) if P.target == "d_edi_v1" else None
    variant = init["variant"] if init else P.variant
    V = VARIANTS[variant]
    RES = init["resolution"] if init else V["resolution"]
    if RES % V["block"]:
        raise T71Error(f"resolution {RES} is not a multiple of {V['block']}")
    run_tag = P.run_tag or f"{P.target}_{variant}_r{RES}_{P.attempt}"
    RUN = R.out("paper1.detector_runs", run_tag)
    ctx.pub = os.path.join(h["local_root"], "pub", run_tag)
    os.makedirs(ctx.pub, exist_ok=True)
    summary = {"stage": "train", "target": P.target, "variant": variant, "mode": P.mode, "run_tag": run_tag, "run_dir": RUN}
    rec_path = os.path.join(RUN, "train_record.json")
    if os.path.isfile(rec_path):
        with open(rec_path, encoding="utf-8") as f:
            rec = json.load(f)
        log(f"{run_tag} is COMPLETE (train_record.json exists): nothing to do")
        summary.update(status="complete", record={k: rec.get(k) for k in ("published_model", "val", "hours_actual")})
        return summary
    small_tag = None
    if P.target == "d_pt_v1" and variant == "medium":
        small_tag = P.small_run_tag or f"d_pt_v1_small_r{VARIANTS['small']['resolution']}_{P.attempt}"
        dec = medium_gate(ctx, small_tag)
        if not dec.get("medium_needed"):
            log(f"variant=medium SKIPPED: the Small run {small_tag} passed (val AP50 {dec.get('small_val_AP50')} >= "
                f"{dec.get('rule_threshold')})")
            summary.update(status="skipped", small_decision=dec)
            return summary
        log(f"variant=medium RUNS: Small {small_tag} val AP50 {dec.get('small_val_AP50')} < {dec.get('rule_threshold')}")
    probe = h["gpu_preflight"](P, ctx.cuda)
    ctx.rf = h["rf"]()                                  # rfdetr + torch_linear_assignment imported BEFORE staging
    # ---------------------------------------------------------------- data
    sets = build_sets(ctx, sorted({T["train_set"], T["sel_set"], *T["eval_sets"]}))
    coco_pub = publish_coco(ctx, sets)
    tr_full, sel_full = sets[T["train_set"]], sets[T["sel_set"]]
    train_ids = stride_ids(tr_full["coco"], P.train_stride)
    sel_ids = stride_ids(sel_full["coco"], P.sel_val_stride)
    train_coco = subset_coco(tr_full["coco"], train_ids, T["train_set"] + f"_stride{P.train_stride}")
    sel_coco = subset_coco(sel_full["coco"], sel_ids, T["sel_set"] + f"_stride{P.sel_val_stride}")
    N_train, N_valid = len(train_coco["images"]), len(sel_coco["images"])
    N_eval = sum(len(sets[n]["coco"]["images"]) for n in T["eval_sets"])
    ds_dir = os.path.join(h["local_root"], "ds", run_tag)
    shutil.rmtree(ds_dir, ignore_errors=True)
    write_rf_dataset(ds_dir, "train", train_coco, tr_full["root"])
    write_rf_dataset(ds_dir, "valid", sel_coco, sel_full["root"])
    prov = {"notebook": NB_NAME, "notebook_version": NB_VERSION, "notebook_sha256": ctx.nb_sha, "core_sha256": ctx.core_sha,
            "bench_version": bc.BENCH_VERSION, "run_tag": run_tag, "target": P.target, "variant": variant,
            "resolution": RES, "train_set": T["train_set"], "train_stride": P.train_stride,
            "train_ids_sha256": ids_sha(train_ids), "sel_set": T["sel_set"], "sel_val_stride": P.sel_val_stride,
            "sel_ids_sha256": ids_sha(sel_ids), "coco_exports": {k: v["sha256"] for k, v in coco_pub.items()},
            "init_from": None if init is None else {"tag": "d_pt_v1", "run_tag": init["run_tag"], "sha256": init["sha256"]},
            "registry": R.provenance([T["registry_key"], "paper1.detector_runs"])}
    data_rec = {"train": {"set": T["train_set"], "stride": P.train_stride, "n_images": N_train,
                          "n_boxes": len(train_coco["annotations"]), "n_seqs": tr_full["census"]["n_seqs"]},
                "selection_val": {"set": T["sel_set"], "stride": P.sel_val_stride, "n_images": N_valid,
                                  "n_boxes": len(sel_coco["annotations"])},
                "eval_sets": {n: {"n_images": sets[n]["census"]["n_images"], "n_boxes": sets[n]["census"]["n_boxes"]}
                              for n in T["eval_sets"]},
                "census": {n: {k: v for k, v in s["census"].items() if k != "per_seq"} for n, s in sets.items()}}
    log(f"{run_tag}: train {N_train} images ({len(train_coco['annotations'])} boxes, stride {P.train_stride}), selection val "
        f"{N_valid} (stride {P.sel_val_stride}), final eval {N_eval} images")
    if P.mode == "smoke":
        out = run_smoke(ctx, run_tag, RUN, variant, RES, init, sets, ds_dir, train_coco, sel_coco, tr_full, sel_full,
                        N_train, N_valid, N_eval, prov, data_rec, probe)
    else:
        out = run_full(ctx, run_tag, RUN, variant, RES, init, sets, ds_dir, N_train, N_valid, N_eval, prov, data_rec,
                       probe, small_tag, train_coco)
    summary.update(out)
    return summary


def _build_model(ctx, variant, res, init):
    m = ctx.rf.build(variant, res, init["local"] if init else None)
    if init:
        info = {"source": "d_pt_v1", "path": init["local"], "sha256": init["sha256"]}
    else:
        path = str(m.model_config.pretrain_weights)
        if not os.path.exists(path):
            raise T71Error(f"pretrained weights absent after construction: {path} (needs egress to storage.googleapis.com)")
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        want = ctx.rf.registry_md5(VARIANTS[variant]["pretrain"])
        if want is not None and want != md5:
            raise T71Error(f"{VARIANTS[variant]['pretrain']} md5 {md5} != rfdetr registry {want}")
        info = {"source": "rfdetr COCO", "file": VARIANTS[variant]["pretrain"], "md5": md5, "registry_md5": want}
    return m, info


def _check_precision(ctx, out_dir):
    with open(os.path.join(out_dir, "training_config.json"), encoding="utf-8") as f:
        tc = json.load(f)
    prec = resolve_precision_replica(tc["model_config"].get("amp", True), tc["train_config"].get("amp_dtype"),
                                     True, ctx.cuda.bf16())
    if prec != "bf16-mixed" or tc["train_config"].get("amp_dtype") != "bf16":
        raise T71Error(f"resolved precision {prec} / amp_dtype {tc['train_config'].get('amp_dtype')} — not bf16-mixed")
    aug = {k: tc["train_config"].get(k) for k in ("multi_scale", "expanded_scales", "scale_jitter",
                                                  "augmentation_backend", "aug_config", "square_resize_div_64")}
    return prec, aug, tc


def measure_t_val_img(model, coco, root, n=TVAL_PROBE_N):
    """Seconds per validation image: predict() (batches of EVAL_BATCH) timed after one warm-up batch."""
    ims = coco["images"][:n]
    if len(ims) < 2:
        raise T71Error(f"need >= 2 valid images to time validation, have {len(ims)}")
    w = min(EVAL_BATCH, len(ims) // 2)
    predict_images(model, ims[:w], root, log=None)                                 # warm-up batch
    t0 = time.perf_counter()
    predict_images(model, ims[w:], root, log=None)
    return (time.perf_counter() - t0) / len(ims[w:]), len(ims[w:])


def run_smoke(ctx, run_tag, RUN, variant, RES, init, sets, ds_dir, train_coco, sel_coco, tr_full, sel_full,
              N_train, N_valid, N_eval, prov, data_rec, probe):
    P, h, log, cuda = ctx.P, ctx.h, ctx.log, ctx.cuda
    rng = random.Random(P.seed)
    sm_tr = sorted(rng.sample([im["id"] for im in train_coco["images"]], min(P.smoke_train_n, N_train)))
    sm_va = sorted(random.Random(P.seed + 1).sample([im["id"] for im in sel_coco["images"]], min(P.smoke_valid_n, N_valid)))
    sm_dir = os.path.join(h["local_root"], "ds_smoke", run_tag)
    shutil.rmtree(sm_dir, ignore_errors=True)
    write_rf_dataset(sm_dir, "train", subset_coco(train_coco, sm_tr), tr_full["root"])
    sm_va_coco = subset_coco(sel_coco, sm_va)
    write_rf_dataset(sm_dir, "valid", sm_va_coco, sel_full["root"])
    out_dir = os.path.join(h["local_root"], "out", run_tag + "_smoke")
    bs, ga, ooms = P.batch_size, P.grad_accum, []
    while True:
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        kw = train_kwargs(P, sm_dir, out_dir, SMOKE_EPOCHS, bs, ga, prov)
        model, pre = _build_model(ctx, variant, RES, init)
        cuda.reset()
        probe_fs = FirstStepProbe(cuda).start()
        t0 = time.perf_counter()
        try:
            model.train(**kw)
            T_total = time.perf_counter() - t0
            break
        except cuda.oom as e:
            probe_fs.stop()
            ooms.append({"batch_size": bs, "grad_accum": ga, "error": str(e)[:200]})
            del model
            cuda.empty()
            if bs // 2 < 2:
                raise T71Error(f"OOM down to batch 2: {ooms}")
            bs, ga = bs // 2, ga * 2
            log(f"OOM -> batch {bs} x accum {ga}")
    t_first = probe_fs.stop()
    if t_first is not None:
        T_setup, T_setup_src = t_first, "cuda allocation jump"
    else:
        T_setup = min(T_SETUP_FALLBACK_S, 0.25 * T_total)
        T_setup_src = f"assumed min(60 s, 0.25 x T_total) = {T_setup:.1f} s (first step not observed)"
    peak = gpu_mem_gb(cuda)
    prec, aug, _tc = _check_precision(ctx, out_dir)
    ms = summarize_metrics_csv(read_metrics_csv(os.path.join(out_dir, "metrics.csv")))
    sm_score = ms["best_val_mAP_50"] if ms["best_val_mAP_50"] is not None else ms["best_val_mAP_50_95"]
    if not sm_score or sm_score <= 0:
        raise T71Error(f"val mAP50 = {ms['best_val_mAP_50']} / mAP50:95 = {ms['best_val_mAP_50_95']} after "
                       f"{SMOKE_EPOCHS} epochs: labels are not being read")
    best, _p = reload_best(ctx, out_dir, variant, RES)
    t_val_img, n_timed = measure_t_val_img(best, sm_va_coco, sel_full["root"])
    proj = project_hours(T_total, T_setup, t_val_img, len(sm_tr), len(sm_va), N_train, N_valid, N_eval, P.epochs)
    log(f"smoke: {T_total:.0f} s (setup {T_setup:.0f} s, {T_setup_src}), {proj['ips']:.1f} img/s train, "
        f"t_val_img {t_val_img * 1e3:.1f} ms -> projected {proj['hours_train']:.2f} h train + {proj['hours_eval']:.2f} h eval "
        f"= {proj['hours_full']:.2f} h (cap {P.max_hours}); smoke val mAP50 {ms['best_val_mAP_50']}")
    rec = {"smoke": True, "run_tag": run_tag, "target": P.target, "variant": variant, "model_class": VARIANTS[variant]["cls"],
           "resolution": RES, "smoke_train_n": len(sm_tr), "smoke_valid_n": len(sm_va), "smoke_epochs": SMOKE_EPOCHS,
           "batch_size_requested": P.batch_size, "grad_accum_requested": P.grad_accum, "batch_size": bs, "grad_accum": ga,
           "oom_steps": ooms, "T_total_s": T_total, "T_setup_s": T_setup, "T_setup_source": T_setup_src,
           "t_val_img": t_val_img, "t_val_source": f"predict() on {n_timed} valid images after a warm-up batch",
           "ips": round(proj["ips"], 3), "N_train": N_train, "N_valid": N_valid, "N_eval": N_eval, "epochs": P.epochs,
           "hours_train": round(proj["hours_train"], 3), "hours_eval": round(proj["hours_eval"], 3),
           "hours_full": round(proj["hours_full"], 3), "max_hours": P.max_hours, "peak_vram_gb": peak,
           "gpu": probe, "precision_resolved": prec, "augmentation_resolved": aug, "smoke_val": ms, "pretrained": pre,
           "provenance": prov, "data": data_rec, "versions": bc.versions(), "rfdetr_version": ctx.rf.version,
           "created_utc": _now()}
    st = _publish_json(ctx, rec, os.path.join(RUN, "smoke.json"), "smoke.json")
    log(f"smoke record {st}: {RUN}/smoke.json")
    stop_gate(proj["hours_full"], P.max_hours, P.epochs)
    return {"status": "smoke_passed", "hours_projected": round(proj["hours_full"], 2), "ips": round(proj["ips"], 2),
            "smoke_val_mAP50": ms["best_val_mAP_50"], "batch": [bs, ga]}


def run_full(ctx, run_tag, RUN, variant, RES, init, sets, ds_dir, N_train, N_valid, N_eval, prov, data_rec, probe,
             small_tag, train_coco):
    P, R, h, log, cuda = ctx.P, ctx.R, ctx.h, ctx.log, ctx.cuda
    smk_p = os.path.join(RUN, "smoke.json")
    if not os.path.isfile(smk_p):
        raise T71Error(f"mode={P.mode} refused: no smoke projection at {smk_p}")
    with open(smk_p, encoding="utf-8") as f:
        smk = json.load(f)
    cur = {"variant": variant, "resolution": RES, "N_train": N_train, "N_valid": N_valid,
           "batch_size_requested": P.batch_size, "grad_accum_requested": P.grad_accum}
    mism = [(k, smk.get(k), v) for k, v in cur.items() if smk.get(k) != v]
    if mism:
        raise T71Error(f"the smoke record does not describe this run: {mism}")
    bs, ga = int(smk["batch_size"]), int(smk["grad_accum"])
    notes = []
    if (bs, ga) != (P.batch_size, P.grad_accum):
        notes.append(f"batch {P.batch_size}x{P.grad_accum} -> {bs}x{ga} (the smoke's OOM policy)")
    proj = project_hours(smk["T_total_s"], smk["T_setup_s"], smk["t_val_img"], smk["smoke_train_n"], smk["smoke_valid_n"],
                         N_train, N_valid, N_eval, P.epochs)
    log(f"re-projected {proj['hours_full']:.2f} h at epochs={P.epochs} (cap {P.max_hours})")
    stop_gate(proj["hours_full"], P.max_hours, P.epochs)
    out_dir = os.path.join(h["local_root"], "out", run_tag)
    rs_vol = os.path.join(RUN, "resume_state")
    kw_plan = train_kwargs(P, ds_dir, out_dir, P.epochs, bs, ga, prov)
    hp = {k: kw_plan[k] for k in RESUME_BOUND_HPARAMS}
    resumed, resume_path = None, None
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    consistency = resume_consistency_factory(ctx)
    if P.mode == "full":
        old = list_generations(rs_vol)
        if old:
            notes.append(f"mode=full started over beside {len(old)} earlier resume generation(s) (kept, never deleted)")
            log("WARNING " + notes[-1])
    else:
        tried = []
        for n, gdir, complete in list_generations(rs_vol):
            if not complete:
                continue
            with open(os.path.join(gdir, GEN_COMPLETE), encoding="utf-8") as f:
                grec = json.load(f)
            mm = resume_mismatches(grec.get("train_hparams"), hp, grec.get("provenance"), prov)
            if mm:
                raise T71Error(f"mode=resume refused: {gen_name(n)} belongs to a different run: {mm}")
            paths = {fn: None for fn in RESUMABLE}
            try:
                for fn in RESUMABLE:
                    if fn not in (grec.get("files") or {}):
                        continue
                    lp, sha = bc.stage_file(os.path.join(gdir, fn), os.path.join(out_dir, fn), ctx.dbutils, grec["files"][fn])
                    paths[fn] = lp
                if paths["last.ckpt"] is None or paths["metrics.csv"] is None:
                    raise T71Error(f"{gen_name(n)} lacks last.ckpt or metrics.csv")
                resumed = consistency(paths)
                resumed["generation"] = gen_name(n)
                break
            except (T71Error, RuntimeError) as e:
                tried.append(f"{gen_name(n)}: {e}")
                log(f"resume generation {gen_name(n)} unusable: {e}")
                for fn in RESUMABLE:
                    if os.path.exists(os.path.join(out_dir, fn)):
                        os.remove(os.path.join(out_dir, fn))
        if resumed is None:
            raise T71Error(f"mode=resume: no usable complete generation under {rs_vol} ({tried}); use mode=full with a new attempt")
        if tried:
            notes.append(f"resume fell back past {tried}")
        resume_path = os.path.join(out_dir, "last.ckpt")
        if resumed["action"] == "patch":
            ck = h["ckpt_read"](resume_path)
            for s in (ck.get("callbacks") or {}).values():
                if isinstance(s, dict) and "_best_ema" in s:
                    s["_best_ema"] = float(resumed["csv_best"])
            h["ckpt_write"](ck, resume_path)
            notes.append(f"resume: last.ckpt _best_ema patched to the csv best {resumed['csv_best']} (in-flight snapshot)")
        log(f"resuming from {resumed['generation']} (ckpt epoch {resumed['ckpt_epoch']}, csv best {resumed['csv_best']} @ "
            f"{resumed['csv_best_epoch']}, {resumed['action']})")
    kw = train_kwargs(P, ds_dir, out_dir, P.epochs, bs, ga, prov, resume=resume_path)
    model, pre = _build_model(ctx, variant, RES, init)
    cuda.reset()
    sync = CkptSync(ctx, out_dir, rs_vol, P.ckpt_sync_min,
                    {"provenance": prov, "train_hparams": hp, "mode": P.mode, "started_utc": _now(),
                     "resumed_from": (resumed or {}).get("generation")}, consistency, make_copier(ctx)).start()
    t0 = time.perf_counter()
    try:
        model.train(**kw)
    except Exception as err:
        sync.stop()
        try:
            g = sync.sync_once()
            log(f"train() raised {type(err).__name__}; crash-time resume generation: {g}")
        except Exception as e2:  # noqa: BLE001
            log(f"train() raised {type(err).__name__}; crash-time generation NOT published: {e2}")
        raise
    finally:
        sync.stop()
    hours = (time.perf_counter() - t0) / 3600.0
    peak_train = gpu_mem_gb(cuda)
    final_gen = None
    try:
        final_gen = sync.sync_once()
    except Exception as e:  # noqa: BLE001
        notes.append(f"final resume generation not published: {str(e)[:160]}")
    log(f"train() returned after {hours:.2f} h (projected {proj['hours_train']:.2f} h train); peak {peak_train} GB")
    del model
    cuda.empty()
    prec, aug, tc = _check_precision(ctx, out_dir)
    ms = summarize_metrics_csv(read_metrics_csv(os.path.join(out_dir, "metrics.csv")))
    best, best_path = reload_best(ctx, out_dir, variant, RES)
    ckm = read_ckpt_meta(ctx, best_path)
    ema_meta = read_ckpt_meta(ctx, os.path.join(out_dir, "checkpoint_best_ema.pth")) \
        if os.path.exists(os.path.join(out_dir, "checkpoint_best_ema.pth")) else {}
    best_epoch = ema_meta.get("epoch") if ema_meta.get("epoch") is not None else ms["best_epoch"]
    if ckm.get("best_total_source") == "ema" and ema_meta.get("epoch") is not None and ms["best_epoch"] is not None \
            and int(ema_meta["epoch"]) != int(ms["best_epoch"]):
        raise T71Error(f"checkpoint_best_total.pth (EMA epoch {ema_meta['epoch']}) is not the metrics.csv best epoch "
                       f"{ms['best_epoch']} — rfdetr's backfill or an inconsistent resume replaced the true best")
    log(f"selected checkpoint: {ckm.get('best_total_source')} epoch {best_epoch}; csv best {ms['best_val_mAP_50_95']} @ "
        f"{ms['best_epoch']} ({ms['key']}), {ms['epochs_run']} epochs run")
    # ---------------------------------------------------------------- evaluation (all frames of every eval set)
    ev, preds, ev_speed = {}, {}, {}
    cuda.reset()
    for name in TARGETS[P.target]["eval_sets"]:
        rows, sp = predict_images(best, sets[name]["coco"]["images"], sets[name]["root"], log=log, what=name)
        preds[name], ev_speed[name] = rows, sp
        ev[name] = evaluate_set(name, sets[name], rows)
        log(f"  {name}: AP50 {ev[name]['all']['AP50']:.4f}  AP50:95 {ev[name]['all']['AP50_95']:.4f} "
            f"({sp['img_per_s']} img/s)")
    peak_eval = gpu_mem_gb(cuda)
    sel_name = TARGETS[P.target]["sel_set"]
    acc = f1_sweep(sets[sel_name]["coco"], preds[sel_name])
    op, f1 = op_threshold_from_sweep(acc)
    log(f"val-F1 operating point {op} (F1 {f1[op]:.4f}) on {sel_name}")
    # sanity: a black frame and a train frame with >= 3 pigs
    sanity = sanity_checks(ctx, best, train_coco, sets[TARGETS[P.target]["train_set"]]["root"])
    # ---------------------------------------------------------------- publish the run files
    pub = {}
    for fn in ("metrics.csv", "training_config.json", "checkpoint_best_total.pth"):
        pub[fn] = publish_new(os.path.join(out_dir, fn), os.path.join(RUN, fn), ctx.roots, ctx.dbutils)
    best_sha = bc.sha256_file(best_path)
    eval_rec = {"eval": ev, "predict_speed": ev_speed, "peak_vram_gb_eval": peak_eval, "ap_tool": "pycocotools bbox, maxDets 100",
                "pred_threshold": PRED_PROBE_THR, "keep_top": AP_KEEP_TOP, "checkpoint_sha256": best_sha}
    pub["eval.json"] = _publish_json(ctx, eval_rec, os.path.join(RUN, "eval.json"), "eval.json")
    sweep_rec = {"set": sel_name, "taus": SWEEP_TAUS, "counts": {str(t): v for t, v in acc.items()},
                 "f1": {str(t): round(v, 6) for t, v in f1.items()}, "op_threshold": op, "rule": "argmax F1, ties -> lower; "
                 "Hungarian (max IoU sum) at box IoU >= 0.5; kept iff score >= tau (= ProtocolD's gate on the cached scores)"}
    pub["f1_sweep.json"] = _publish_json(ctx, sweep_rec, os.path.join(RUN, "f1_sweep.json"), "f1_sweep.json")
    fig = None
    if P.target == "d_pt_v1":
        try:
            fig = render_val_panel(ctx, sets[sel_name], preds[sel_name], op, os.path.join(RUN, "figures", "val_preds.png"))
        except Exception as e:  # noqa: BLE001
            notes.append(f"val panel not rendered: {type(e).__name__}: {str(e)[:160]}")
    # ---------------------------------------------------------------- decision + model publication
    val_ap50 = ev[sel_name]["all"]["AP50"]
    decision, publish_model = None, True
    if P.target == "d_pt_v1" and variant == "small":
        need = decide_medium(val_ap50)
        decision = {"rule": "Small; Medium only if Small val AP50 < 0.85 (pycocotools, all val frames)",
                    "rule_threshold": AP50_RULE, "small_run_tag": run_tag, "small_val_AP50": val_ap50,
                    "medium_needed": need, "created_utc": _now()}
        pub["decision.json"] = _publish_json(ctx, decision, os.path.join(RUN, "decision.json"), "decision.json")
        publish_model = not need
        log(f"DECISION: Small val AP50 {val_ap50:.4f} {'<' if need else '>='} {AP50_RULE} -> "
            f"{'MEDIUM needed; d_pt_v1 is NOT published from this run' if need else 'd_pt_v1 = this Small run'}")
    model_dir = R.out(TARGETS[P.target]["registry_key"])
    ts = None
    if publish_model:
        ts = {"tag": P.target, "run_tag": run_tag, "mode": P.mode, "variant": variant, "model_class": VARIANTS[variant]["cls"],
              "resolution": RES, "class_names": list(CLASS_NAMES), "num_classes": 1, "category_ids": {"pig": CAT_ID},
              "checkpoint_best_total_sha256": best_sha, "best_total_source": ckm.get("best_total_source"),
              "best_epoch": best_epoch, "epochs": P.epochs, "epochs_run": ms["epochs_run"],
              "early_stopped": bool(ms["epochs_run"] and ms["epochs_run"] < P.epochs),
              "selection": {"metric": ms["key"], "split": f"{TARGETS[P.target]['sel_set']} every {P.sel_val_stride}th frame",
                            "best_val_mAP_50_95_rfdetr": ms["best_val_mAP_50_95"], "best_val_mAP_50_rfdetr": ms["best_val_mAP_50"],
                            "small_decision": decision if decision else (None if small_tag is None else
                                                                          medium_gate(ctx, small_tag))},
              "op_threshold": op, "op_threshold_f1": round(f1[op], 6),
              "op_threshold_rule": "val-F1-optimal score threshold, 0.10..0.95 step 0.01, ties -> lower",
              "val": ev[sel_name], "eval": ev, "train_data": data_rec, "pretrained": pre, "init_from": prov["init_from"],
              "hours_actual": round(hours, 3), "hours_projected": round(proj["hours_full"], 3), "peak_vram_gb_train": peak_train,
              "gpu": probe, "precision_resolved": prec, "augmentation_resolved": aug, "batch_size": bs, "grad_accum": ga,
              "train_kwargs": {k: v for k, v in kw.items() if k not in ("notes", "dataset_dir", "output_dir")},
              "rfdetr_version": ctx.rf.version, "versions": bc.versions(), "provenance": prov,
              "loader": "bench_core.load_rfdetr_detector(ckpt, resolution=<resolution>) — from_checkpoint pins it",
              "licence": ("RF-DETR Apache-2.0; trained on PigTrack (CC BY 4.0)" if P.target == "d_pt_v1" else
                          "RF-DETR Apache-2.0; fine-tuned on Edinburgh pigs (CC BY-NC): non-commercial use only"),
              "notes": notes, "created_utc": _now()}
        train_summary_check(ts)
        for fn in ("checkpoint_best_total.pth", "metrics.csv", "training_config.json"):
            pub["model/" + fn] = publish_new(os.path.join(out_dir, fn), os.path.join(model_dir, fn), ctx.roots, ctx.dbutils)
        if bc.sha256_file(os.path.join(model_dir, "checkpoint_best_total.pth")) != best_sha:
            raise T71Error("the published checkpoint does not hash to the evaluated one")
        pub["model/train_summary.json"] = _publish_json(ctx, ts, os.path.join(model_dir, "train_summary.json"), "train_summary.json")
        contract = verify_published(ctx, model_dir, sets[sel_name], preds[sel_name])
    else:
        contract = None
    record = {"run_tag": run_tag, "target": P.target, "variant": variant, "resolution": RES, "mode": P.mode,
              "published_model": model_dir if publish_model else None, "decision": decision,
              "val": ev[sel_name], "eval": ev, "op_threshold": op, "best_epoch": best_epoch, "epochs_run": ms["epochs_run"],
              "hours_actual": round(hours, 3), "hours_projected": round(proj["hours_full"], 3), "peak_vram_gb_train": peak_train,
              "peak_vram_gb_eval": peak_eval, "checkpoint_sha256": best_sha, "resumed": resumed, "final_generation": final_gen,
              "sync_log": sync.log, "published": pub, "contract_check": contract, "sanity": sanity, "figure": fig,
              "notes": notes, "provenance": prov, "created_utc": _now()}
    _publish_json(ctx, record, os.path.join(RUN, "train_record.json"), "train_record.json")
    log(f"{run_tag} COMPLETE -> {RUN}/train_record.json" + (f"; model {model_dir}" if publish_model else ""))
    return {"status": "complete", "val_AP50": val_ap50, "op_threshold": op, "published_model": record["published_model"],
            "hours_actual": round(hours, 3), "eval": {k: v["all"] for k, v in ev.items()}}


def sanity_checks(ctx, model, train_coco, root):
    """Recorded (not gates): detections > 0.5 on a black frame; on a seeded train frame with >= 3 GT pigs the model must
    keep >= 1 detection at 0.5 (raise on 0: a catastrophically broken checkpoint)."""
    from PIL import Image  # noqa: F401
    ims = [im for im in train_coco["images"]]
    ngt = collections.Counter(a["image_id"] for a in train_coco["annotations"])
    cand = [im for im in ims if ngt[im["id"]] >= 3]
    out = {}
    if cand:
        im = random.Random(ctx.P.seed).choice(cand)
        rows, _ = predict_images(model, [im], root, log=None)
        k = int((rows[:, 5] > 0.5).sum())
        out["train_frame"] = {"file": im["file_name"], "n_gt": ngt[im["id"]], "n_det_gt_0.5": k}
        if k == 0:
            raise T71Error(f"the reloaded checkpoint keeps 0 detections at 0.5 on train frame {im['file_name']} "
                           f"({ngt[im['id']]} GT pigs)")
    W, H = ims[0]["width"], ims[0]["height"]
    d = model.predict(np.zeros((H, W, 3), np.uint8), threshold=0.5, include_source_image=False)
    out["black_frame_n_det_gt_0.5"] = int((np.asarray(d.class_id) < 1).sum()) if len(d) else 0
    return out


def load_rgb_writable(path):
    """HWC uint8 RGB that cv2 may draw on: np.asarray(PIL) is READ-ONLY and cv2.rectangle refuses it (J11 run
    593021960562654: 'img marked as output argument, but provided NumPy array'), so copy."""
    from PIL import Image
    with Image.open(path) as x:
        return np.array(x.convert("RGB"), dtype=np.uint8, copy=True, order="C")


def render_val_panel(ctx, s, rows, thr, dst, n=8):
    """8 seeded PigTrack val frames (CC BY 4.0; never Edinburgh, ) with GT (white) and detections
    above the operating point (green)."""
    import cv2
    from PIL import Image
    ims = s["coco"]["images"]
    pick = sorted(random.Random(ctx.P.seed).sample(range(len(ims)), min(n, len(ims))))
    G = gt_xyxy_by_image(s["coco"])
    D = dets_by_image(rows)
    panels = []
    for i in pick:
        im = ims[i]
        a = load_rgb_writable(os.path.join(s["root"], im["file_name"]))
        for b in G[im["id"]]:
            cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 2)
        boxes, sc = D.get(im["id"], (np.zeros((0, 4)), np.zeros(0)))
        for b, v in zip(boxes, sc):
            if v > thr:
                cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (40, 220, 40), 2)
        a = cv2.resize(a, (640, int(640 * a.shape[0] / a.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.putText(a, f"{im['seq']} f{im['frame']} thr {thr}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        panels.append(a)
    hmax = max(p.shape[0] for p in panels)
    panels = [np.pad(p, ((0, hmax - p.shape[0]), (0, 0), (0, 0))) for p in panels]
    while len(panels) % 4:
        panels.append(np.zeros_like(panels[0]))
    grid = np.concatenate([np.concatenate(panels[i:i + 4], 1) for i in range(0, len(panels), 4)], 0)
    loc = os.path.join(ctx.pub, "val_preds.png")
    Image.fromarray(grid).save(loc)
    publish_new(loc, dst, ctx.roots, ctx.dbutils)
    return dst


def verify_published(ctx, model_dir, sel, sel_rows, n_img=3):
    """Re-read the published train_summary.json, stage the published checkpoint fresh and load it EXACTLY as track_bench
    does (bc.load_rfdetr_detector with the summary's resolution); its detections on 3 val frames must reproduce the
    evaluation's (score > 0.30: every row matched at IoU >= 0.9 with |d score| < 0.05)."""
    from PIL import Image
    with open(os.path.join(model_dir, "train_summary.json"), encoding="utf-8") as f:
        ts = json.load(f)
    train_summary_check(ts)
    loc = os.path.join(ctx.h["local_root"], "verify", ts["tag"], "checkpoint_best_total.pth")
    local, sha = bc.stage_file(os.path.join(model_dir, "checkpoint_best_total.pth"), loc, ctx.dbutils,
                               ts["checkpoint_best_total_sha256"])
    det = ctx.h["detector_loader"](local, ts["resolution"])
    D = dets_by_image(sel_rows)
    ims = sel["coco"]["images"][:: max(1, len(sel["coco"]["images"]) // n_img)][:n_img]
    n_ref = n_ok = 0
    for fi, im in enumerate(ims):
        with Image.open(os.path.join(sel["root"], im["file_name"])) as x:
            pil = x.convert("RGB")
        got = [(b, s) for b, s in det(fi, pil) if s > 0.30]
        ref_b, ref_s = D.get(im["id"], (np.zeros((0, 4)), np.zeros(0)))
        keep = ref_s > 0.30
        ref_b, ref_s = ref_b[keep], ref_s[keep]
        n_ref += len(ref_s)
        if len(ref_s) and got:
            iou = iou_xyxy(ref_b, np.asarray([g[0] for g in got]))
            gs = np.asarray([g[1] for g in got])
            for i in range(len(ref_s)):
                j = int(np.argmax(iou[i]))
                n_ok += int(iou[i, j] >= 0.9 and abs(gs[j] - ref_s[i]) < 0.05)
    frac = n_ok / n_ref if n_ref else 1.0
    out = {"loader": "bench_core.load_rfdetr_detector", "resolution": ts["resolution"], "sha256": sha,
           "det_info": {k: v for k, v in getattr(det, "info", {}).items()}, "n_ref": n_ref, "n_matched": n_ok,
           "fraction": round(frac, 4)}
    del det
    if frac < 0.95:
        raise T71Error(f"track_bench's loader does not reproduce the evaluated detections: {out}")
    ctx.log(f"contract check OK: track_bench's loader reproduces {n_ok}/{n_ref} evaluation detections")
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bench stage (A10): single-forward peak memory and latency of the published detectors via track_bench's loader

# COMMAND ----------


def run_bench(ctx):
    P, R, h, log, cuda = ctx.P, ctx.R, ctx.h, ctx.log, ctx.cuda
    from PIL import Image
    probe = h["gpu_preflight"](P, cuda)
    frames = {}
    for ds, split, seq, k in BENCH_FRAMES:
        if ds == "pigtrack":
            src = R.path("datasets.pigtrack.sequences", None, split, seq, "img1", f"{k + 0:08d}.jpg")
        else:
            src = R.path("datasets.edinburgh_pigs.decoded_frames", None, seq, "video1", f"{k + 1:07d}.jpg")
        loc, sha = bc.stage_file(src, os.path.join(h["local_root"], "bench_frames", ds, os.path.basename(src)), ctx.dbutils)
        with Image.open(loc) as im:
            frames[f"{ds}:{seq}:{os.path.basename(src)}"] = (im.convert("RGB"), sha)
    out_all = {}
    for tag in P.bench_tags:
        key = TARGETS[tag]["registry_key"]
        with open(R.path(key, None, "train_summary.json"), encoding="utf-8") as f:
            ts = json.load(f)
        train_summary_check(ts)
        local, sha = bc.stage_file(R.path(key, None, "checkpoint_best_total.pth"),
                                   os.path.join(h["local_root"], "models", tag, "checkpoint_best_total.pth"), ctx.dbutils,
                                   ts["checkpoint_best_total_sha256"])
        cuda.empty()
        cuda.reset()
        base = cuda.alloc()
        det = h["detector_loader"](local, ts["resolution"])
        after_load = cuda.alloc() - base
        res = {"tag": tag, "ckpt_sha256": sha, "resolution": ts["resolution"], "variant": ts["variant"],
               "model_class": ts["model_class"], "params": h["param_count"](local), "gpu": probe,
               "baseline_alloc_mib": round(base / 2 ** 20, 1), "alloc_after_load_mib": round(after_load / 2 ** 20, 1),
               "frames": {}}
        for fk, (pil, fsha) in frames.items():
            cuda.reset()
            t0 = time.perf_counter()
            det(0, pil)
            cuda.sync()
            cold_ms = (time.perf_counter() - t0) * 1e3
            cold_peak = cuda.peak() - base
            weights = cuda.alloc() - base
            for _ in range(5):
                det(0, pil)
            cuda.sync()
            cuda.reset()
            dets = det(0, pil)
            cuda.sync()
            single_peak = cuda.peak() - base
            lat = []
            for _ in range(P.bench_iters):
                t0 = time.perf_counter()
                det(0, pil)
                cuda.sync()
                lat.append((time.perf_counter() - t0) * 1e3)
            lat = np.asarray(lat)
            res["frames"][fk] = {
                "size_wh": list(pil.size), "frame_sha256": fsha, "n_det_gt_0.10": len(dets),
                "resident_after_first_forward_mib": round(weights / 2 ** 20, 1),
                "cold_first_forward_peak_mib": round(cold_peak / 2 ** 20, 1), "cold_first_forward_ms": round(cold_ms, 1),
                "single_forward_peak_alloc_mib": round(single_peak / 2 ** 20, 1),
                "single_forward_transient_mib": round((single_peak - weights) / 2 ** 20, 1),
                "peak_reserved_mib": round(cuda.peak_reserved() / 2 ** 20, 1),
                "latency_ms": {"n": int(len(lat)), "median": round(float(np.median(lat)), 2),
                               "p25": round(float(np.percentile(lat, 25)), 2), "p75": round(float(np.percentile(lat, 75)), 2)}}
            log(f"  {tag} {fk}: single-forward peak {res['frames'][fk]['single_forward_peak_alloc_mib']} MiB "
                f"(resident {res['frames'][fk]['resident_after_first_forward_mib']} MiB), "
                f"median {res['frames'][fk]['latency_ms']['median']} ms")
        res.update({"what": "predict() through bench_core.load_rfdetr_detector (threshold 0.10, TF32 inside predict); "
                            "peaks = torch.cuda.max_memory_allocated above the pre-load baseline; latency includes the "
                            "PIL->tensor preprocessing and supervision post-processing", "nb_version": NB_VERSION,
                    "notebook_sha256": ctx.nb_sha, "core_sha256": ctx.core_sha, "versions": bc.versions(),
                    "created_utc": _now()})
        del det
        cuda.empty()
        st = _publish_json(ctx, res, ctx.R.out("paper1.detector_runs", BENCH_DIR, f"{tag}.json"), f"bench_{tag}.json")
        out_all[tag] = {"status": st, "frames": {k: {kk: v[kk] for kk in ("single_forward_peak_alloc_mib",
                                                                         "resident_after_first_forward_mib", "latency_ms")}
                                                 for k, v in res["frames"].items()}}
    return {"stage": "bench", "results": out_all}


def run_coco(ctx):
    sets = build_sets(ctx, sorted(COCO_SETS))
    return {"stage": "coco", "published": publish_coco(ctx, sets)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main

# COMMAND ----------


def main(raw, *, dbutils_=None, R=None, hooks=None):
    h = dict(PROD_HOOKS)
    h.update(hooks or {})
    P = parse_params(raw)
    R = R if R is not None else farm_paths.load(version=REGISTRY_VERSION)
    src = h["notebook_source"]() if callable(h["notebook_source"]) else h["notebook_source"]
    ctx = types.SimpleNamespace(P=P, R=R, dbutils=dbutils_, h=h, log=h["log"], roots=publish_roots(R),
                                nb_sha=hashlib.sha256(src).hexdigest(), core_sha=_sha256(bc.__file__), rf=None,
                                cuda=h["cuda"]() if P.stage != "coco" else None, pub=os.path.join(h["local_root"], "pub", "_"))
    os.makedirs(ctx.pub, exist_ok=True)
    ctx.log(f"{NB_NAME} {NB_VERSION} stage={P.stage} target={P.target} variant={P.variant} mode={P.mode} "
            f"attempt={P.attempt} (notebook sha {ctx.nb_sha[:12]}, core {ctx.core_sha[:12]})")
    if P.stage == "coco":
        return run_coco(ctx)
    if P.stage == "bench":
        return run_bench(ctx)
    return run_train(ctx)

# COMMAND ----------


if not IS_NOTEBOOK:
    print("This notebook needs a GPU cluster and the path registry: see docs/REPRODUCE.md")
    raise SystemExit(0)
