# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N69 — Behaviour classifier re-split BY CLIP (Paper 1, job **J10**, CPU)
# MAGIC
# MAGIC PLAN §A5 `Tol_69` / decision D9 ("re-split by clip, accept a lower accuracy": accepted by the user) / harness.md
# MAGIC Finding 12 / BENCH_PROTOCOL §15 (b1.1 clarification 2026-10-15, J10). The published DINOv3-ViT-S + BiLSTM behaviour
# MAGIC classifier (97.34 % accuracy, 91.67 macro-F1, 97.37 weighted-F1) split stride-1 overlapping 3-frame windows **at
# MAGIC random** across all 9 Edinburgh clips, so almost every test window shares 2 of its 3 crops with a training window.
# MAGIC This notebook re-trains the SAME model with the SAME recipe on a **clip-level split** that matches the distillation
# MAGIC split: train = clips 0–3 (`EDI_TRAIN`), val = clip 4 (`EDI_VAL`), test = clips 5–8 (`EDI_TEST`).
# MAGIC
# MAGIC **Recipe = `DinoV3_ViTS_Behavior_Classification_1` cells 15/17/19/21/23, verbatim** (exported in
# MAGIC `scratchpad/catalog/paper1_ws/dinov3_behavior.txt`): windows per (clip, pig) sorted by frame, length 3, stride 1,
# MAGIC majority label (`Counter.most_common(1)`), windows whose majority is not one of the 9 classes dropped; BiLSTM
# MAGIC (384 → 128 × 2, 1 layer, last time step) + Linear(256,128)–ReLU–Dropout(0.3)–Linear(128,9); class weights
# MAGIC `1/(n+1)` normalised to sum 9 on the TRAIN split; weighted cross-entropy; Adam lr 1e-3, weight decay 1e-5; batch 128
# MAGIC (train shuffled); ≤ 60 epochs; checkpoint on strict val-loss improvement; early stop after 8 epochs without one;
# MAGIC test on the best checkpoint. **Only the split changes.** Not part of the recipe and recorded: torch seeds (the
# MAGIC original set none), CPU (the original trained on an A10), `torch.use_deterministic_algorithms(True)`, and each
# MAGIC stored embedding is copied into a compact tensor (the 2026-03 `.pt` files hold the whole 256-row batch storage;
# MAGIC the 384 values are unchanged).
# MAGIC
# MAGIC **Arms** (widget `arms`):
# MAGIC | arm | what | role |
# MAGIC |---|---|---|
# MAGIC | `bridge` | the published checkpoint (`models.behaviour_bilstm`) evaluated on the RE-CREATED window-level test split (`train_test_split`, stratified, `random_state=42`, exactly as cell 15) | proves the catalog embeddings + window logic are the published ones: must give 4,178 / 4,292 correct (= 97.34 %) within ±2 windows; the window census must equal the published one exactly |
# MAGIC | `leaky` | the recipe retrained on the original window-level split, seeds 0–4 | separates the harness / seed effect from the split effect |
# MAGIC | `clip` | **the recipe on the clip-level split, seeds 0–4 = the J10 result** | headline |
# MAGIC
# MAGIC **Headline (pre-registered, §15):** mean ± SD (ddof 1) over seeds 0–4 of test accuracy, macro-F1 (sklearn default =
# MAGIC classes present in y_true ∪ y_pred, as cell 23) and weighted-F1; beside them macro-F1 over the classes WITH test
# MAGIC support, balanced accuracy, per-class P/R/F1 (mean ± SD), a pooled confusion matrix (sum over seeds), per-test-clip
# MAGIC accuracy and the train-majority-class baseline. No verdict: behaviour carries no pre-registered contrast.
# MAGIC
# MAGIC **Outputs** (`paper1_distillation/behaviour_resplit/<tag>/`, registry `paper1.root`, each file published once with
# MAGIC `publish_no_overwrite`): `RESPLIT_CONFIG.json` (a resume must match it) · `census.json` (window counts per split /
# MAGIC class / clip, leakage statistics, published-census gates) · `data/embeddings_digest.json` · `bridge/{metrics.json,
# MAGIC compare.json, predictions.csv}` · `<arm>/seed_<s>/{metrics.json, history.json, predictions.csv, best_bilstm.pt}` ·
# MAGIC `.done_<unit>` markers (written last; they carry the timings) · `SUMMARY.{json,md}` (once every unit exists) ·
# MAGIC `runs/<utc>.json`. The published classifier is never replaced; promoting a re-split checkpoint to `models/` needs the
# MAGIC user's OK (and a new registry version).
# MAGIC
# MAGIC **Rules.** Paths only through the registry (`tol_paths`, `_admin/registry/paths_v1.yaml`); inputs only from the
# MAGIC catalog (never the blob mount: the metadata's mount paths are mapped through relocation E05 onto
# MAGIC `datasets.edinburgh_pigs.embeddings`); writes only under `paper1_distillation/`. A reduced run (fewer epochs, seeds
# MAGIC or arms) needs a tag starting with `smoke` and is labelled SMOKE. Before touching real data the notebook runs its
# MAGIC selftest (synthetic data, the whole flow incl. training, resume and determinism) and stops if it fails.
# MAGIC Deploy: this file SOURCE; `tol_bench_core.py` and the registry's `tol_paths.py` RAW in the same Workspace folder.
# MAGIC Local check (no Volume; torch parts run only where torch is installed): `TOL69_SELFTEST=1 python Tol_69_behaviour_resplit.py`.

# COMMAND ----------

import csv
import hashlib
import io
import json
import math
import os
import os.path as osp
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

NB_VERSION = "tol69-2026-10-15.1"
NB_NAME = "Tol_69_behaviour_resplit"
IS_NOTEBOOK = True
try:
    dbutils  # noqa: F821,B018
except NameError:
    IS_NOTEBOOK = False
SELFTEST = os.environ.get("TOL69_SELFTEST") == "1"
if not IS_NOTEBOOK and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_OK = True
except Exception:  # noqa: BLE001 -- local selftest without torch: the torch parts are skipped and reported
    torch = nn = optim = DataLoader = Dataset = None
    TORCH_OK = False


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
if tol_paths is None and IS_NOTEBOOK:
    raise RuntimeError(f"tol_paths.py is not in {CODE_DIR}: deploy the registry's tol_paths.py (RAW) beside this notebook")
print(f"{NB_NAME} {NB_VERSION}; tol_bench_core {core.BENCH_VERSION} sha256 {core.sha256_file(core.__file__)[:16]}"
      + (f"; tol_paths sha256 {core.sha256_file(tol_paths.__file__)[:16]}" if tol_paths else "; tol_paths absent")
      + f"; torch {'yes' if TORCH_OK else 'NO'}")

# =====================================================================================================================
# Pre-registered constants. RECIPE = cells 15/17/19/21/23 of DinoV3_ViTS_Behavior_Classification_1 (verbatim values);
# a change is a new NB_VERSION + a BENCH_PROTOCOL §15 entry.
# =====================================================================================================================
WANTED_BEH = ['standing', 'lying', 'eat', 'drink', 'sitting', 'sleep', 'run', 'playwithtoy', 'nose-to-nose']
NUM_CLASSES = len(WANTED_BEH)
EMB_DIM = 384                       # DINOv3-ViT-S pooler_output
SEQ_LEN = 3
SEQ_STRIDE = 1
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 60
PATIENCE = 8
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.3
LEAKY_TEST_SIZE = 0.30              # cell 15: train_test_split(paths, labels, test_size=0.30, stratify, random_state=42)
LEAKY_TEST_HALF = 0.50              #          then (tmp) test_size=0.50, stratify, random_state=42
LEAKY_RANDOM_STATE = 42
RECIPE = {"wanted_beh": WANTED_BEH, "emb_dim": EMB_DIM, "seq_len": SEQ_LEN, "seq_stride": SEQ_STRIDE,
          "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "num_epochs": NUM_EPOCHS,
          "patience": PATIENCE, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
          "bidirectional": True, "readout": "last time step", "optimizer": "Adam", "loss": "CrossEntropyLoss(weight)",
          "class_weights": "inv=1/(train_count+1); w=inv/sum(inv)*9", "checkpoint": "strict val-loss improvement",
          "early_stop": "no_improve >= patience", "window_label": "Counter(3 behaviours).most_common(1)",
          "source": "DinoV3_ViTS_Behavior_Classification_1 cells 15/17/19/21/23 (paper1_ws/dinov3_behavior export)"}

META_FILE = "metadata_DinoV3.csv"   # in datasets.edinburgh_pigs.behaviour_metadata (relocation E04)
OLD_EMB_DIR = "All Embeddings_DinoV3_21M"   # the metadata's mount folder -> datasets.edinburgh_pigs.embeddings (E05)
PUB_REPORT_REL = ("evidence", "behaviour_2026-03", "figures", "dinov3_vits_classification_report.json")  # E06
CLIP_RX = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{6}$")
TAG_RX = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
ARMS_ALL = ("bridge", "leaky", "clip")
SEEDS_DEFAULT = (0, 1, 2, 3, 4)
SPLIT_CLIPS = {"train": tuple(core.EDI_TRAIN), "val": tuple(core.EDI_VAL), "test": tuple(core.EDI_TEST)}
BRIDGE_TOL_WINDOWS = 2              # |correct(bridge) - 4,178| <= 2 windows (published accuracy is given to 1e-16)
REGISTRY_VERSION = 1
REGISTRY_KEYS = ("datasets.edinburgh_pigs.behaviour_metadata", "datasets.edinburgh_pigs.embeddings",
                 "models.behaviour_bilstm", "paper1.root")

# the published run (cell 15 / 23 outputs; the report JSON copy is asserted equal at run time)
PUBLISHED = {
    "test_accuracy": 0.9734389561975769, "macro_f1": 0.9166823433050134, "weighted_f1": 0.9737190672909913,
    "test_loss_4dp": 0.2155,
    "per_class_precision": {"standing": 0.9907192575406032, "lying": 0.9111969111969112, "eat": 0.9820143884892086,
                            "drink": 0.912621359223301, "sitting": 0.7868852459016393, "sleep": 0.9968652037617555,
                            "run": 0.8333333333333334, "playwithtoy": 0.8571428571428571,
                            "nose-to-nose": 0.810126582278481},
    "per_class_recall": {"standing": 0.9008438818565401, "lying": 0.989517819706499, "eat": 1.0,
                         "drink": 0.9894736842105263, "sitting": 0.9795918367346939, "sleep": 0.9763157894736842,
                         "run": 0.7142857142857143, "playwithtoy": 0.9473684210526315,
                         "nose-to-nose": 0.9846153846153847},
    "per_class_f1": {"standing": 0.943646408839779, "lying": 0.9487437185929648, "eat": 0.9909255898366606,
                     "drink": 0.9494949494949495, "sitting": 0.8727272727272727, "sleep": 0.9864834921338356,
                     "run": 0.7692307692307693, "playwithtoy": 0.9, "nose-to-nose": 0.8888888888888888},
}
PUB_CENSUS = {
    "rows": 43200, "windows": 28609, "unique_embeddings": 29246,
    "leaky": {"train": {"n": 20026, "per_class": [2209, 2225, 3821, 444, 231, 10640, 64, 88, 304]},
              "val": {"n": 4291, "per_class": [473, 476, 819, 95, 50, 2280, 14, 19, 65]},
              "test": {"n": 4292, "per_class": [474, 477, 819, 95, 49, 2280, 14, 19, 65]}},
    "class_weights_3dp": [0.108, 0.107, 0.062, 0.536, 1.029, 0.022, 3.671, 2.681, 0.782],
}


# =====================================================================================================================
# pure helpers (stdlib + numpy + pandas + sklearn)
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


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def add_keys(df_meta):
    """VERBATIM cell 15: grouping keys + sort. Plus integrity checks the original did not make (no missing key, one row
    per (clip, pig, frame), one row per embedding file)."""
    df_meta = df_meta.copy()
    df_meta['clip'] = df_meta['raw_frame'].str.extract(r'Decoded Frames/(.*?)/video1')
    df_meta['pig'] = df_meta['cropped_frame'].str.extract(r'pig_(\d+)')
    df_meta['frame'] = df_meta['raw_frame'].str.extract(r'(\d{7})\.jpg').astype(int)
    df_meta = df_meta.sort_values(['clip', 'pig', 'frame']).reset_index(drop=True)
    if df_meta[['clip', 'pig']].isna().any().any():
        raise ValueError("metadata rows without a clip or pig key")
    if df_meta.duplicated(['clip', 'pig', 'frame']).any():
        raise ValueError("duplicate (clip, pig, frame) rows in the metadata")
    if df_meta['embedding'].duplicated().any():
        raise ValueError("duplicate embedding paths in the metadata")
    return df_meta


def build_windows(df_meta):
    """VERBATIM cell 15 window loop (+ per-window clip / pig / frames, which the original did not keep)."""
    label2idx = {b: i for i, b in enumerate(WANTED_BEH)}
    all_windows, meta = [], []
    for (clip, pig), grp in df_meta.groupby(['clip', 'pig'], sort=False):
        emb_paths = grp['embedding'].tolist()
        behaviors = grp['behavior'].tolist()
        frames = [int(f) for f in grp['frame'].tolist()]
        n = len(emb_paths)
        for start in range(0, n - SEQ_LEN + 1, SEQ_STRIDE):
            window_paths = emb_paths[start:start + SEQ_LEN]
            window_behs = behaviors[start:start + SEQ_LEN]
            maj_label = Counter(window_behs).most_common(1)[0][0]
            if maj_label in label2idx:
                all_windows.append((window_paths, label2idx[maj_label]))
                meta.append({"clip": clip, "pig": str(pig), "frames": tuple(frames[start:start + SEQ_LEN])})
    return all_windows, meta


def leaky_split(all_windows):
    """VERBATIM cell 15 split, applied to window INDICES (train_test_split draws one permutation from n_samples and the
    stratify labels and indexes every array with it, so this equals splitting the path tuples; asserted against
    leaky_split_paths)."""
    from sklearn.model_selection import train_test_split
    labels = [lab for _, lab in all_windows]
    idx = list(range(len(all_windows)))
    i_tr, i_tmp, l_tr, l_tmp = train_test_split(
        idx, labels, test_size=LEAKY_TEST_SIZE, stratify=labels, random_state=LEAKY_RANDOM_STATE)
    i_val, i_test, l_val, l_test = train_test_split(
        i_tmp, l_tmp, test_size=LEAKY_TEST_HALF, stratify=l_tmp, random_state=LEAKY_RANDOM_STATE)
    return {"train": [int(i) for i in i_tr], "val": [int(i) for i in i_val], "test": [int(i) for i in i_test]}


def leaky_split_paths(all_windows):
    """The literal cell-15 call (paths, labels) -> (p_tr, p_val, p_test), for the equivalence check only."""
    from sklearn.model_selection import train_test_split
    paths, labels = zip(*all_windows)
    p_tr, p_tmp, l_tr, l_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=42)
    p_val, p_test, l_val, l_test = train_test_split(
        p_tmp, l_tmp, test_size=0.50, stratify=l_tmp, random_state=42)
    return {"train": p_tr, "val": p_val, "test": p_test}


def clip_split(meta, split_clips):
    """Clip-level split: a window belongs to the split that holds its clip (windows never cross clips)."""
    owner = {}
    for s, clips in split_clips.items():
        for c in clips:
            if c in owner:
                raise ValueError(f"clip {c} is in two splits ({owner[c]}, {s})")
            owner[c] = s
    out = {s: [] for s in split_clips}
    missing = sorted({m["clip"] for m in meta if m["clip"] not in owner})
    if missing:
        raise ValueError(f"clips {missing} are in no split")
    for i, m in enumerate(meta):
        out[owner[m["clip"]]].append(i)
    return out


def check_partition(split, n):
    """Every window index in exactly one split."""
    seen = Counter(i for idx in split.values() for i in idx)
    if len(seen) != n or any(v != 1 for v in seen.values()) or set(seen) != set(range(n)):
        raise AssertionError(f"split is not a partition of the {n} windows")


def census(all_windows, meta, split):
    out = {}
    for s, idx in split.items():
        cnt = Counter(all_windows[i][1] for i in idx)
        out[s] = {"n": len(idx), "per_class": {WANTED_BEH[k]: int(cnt.get(k, 0)) for k in range(NUM_CLASSES)},
                  "per_clip": dict(sorted(Counter(meta[i]["clip"] for i in idx).items())),
                  "unique_crops": len({p for i in idx for p in all_windows[i][0]})}
    return out


def overlap_stats(all_windows, meta, split):
    """Leakage between splits: shared crops / clips per pair, and for val and test the share of windows that contain
    >= 1, >= 2 or all 3 crops also used by a TRAIN window. Consecutive-frame check of every window."""
    crops = {s: {p for i in idx for p in all_windows[i][0]} for s, idx in split.items()}
    clips = {s: {meta[i]["clip"] for i in idx} for s, idx in split.items()}
    names = list(split)
    pairs = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = names[a], names[b]
            pairs[f"{x}|{y}"] = {"shared_crops": len(crops[x] & crops[y]), "shared_clips": sorted(clips[x] & clips[y])}
    vs_train = {}
    for s in names:
        if s == "train" or "train" not in split:
            continue
        ref = crops["train"]
        k = Counter(sum(p in ref for p in all_windows[i][0]) for i in split[s])
        n = max(1, len(split[s]))
        vs_train[s] = {"windows": len(split[s]),
                       "share_ge1_crop_in_train": sum(v for c, v in k.items() if c >= 1) / n,
                       "share_ge2_crops_in_train": sum(v for c, v in k.items() if c >= 2) / n,
                       "share_all3_crops_in_train": k.get(SEQ_LEN, 0) / n}
    nonconsec = sum(1 for m in meta if any(m["frames"][j + 1] != m["frames"][j] + 1 for j in range(SEQ_LEN - 1)))
    return {"pairs": pairs, "vs_train": vs_train, "non_consecutive_windows": nonconsec}


def assert_no_leak(ov):
    """Hard gate for the clip-level split: no crop and no clip shared by two splits."""
    bad = {k: v for k, v in ov["pairs"].items() if v["shared_crops"] or v["shared_clips"]}
    if bad:
        raise AssertionError(f"clip-level split leaks: {bad}")


def published_census_gates(n_rows, all_windows, cen_leaky):
    """The re-created data must equal the published run's census exactly (cell 15 output)."""
    got = {"rows": n_rows, "windows": len(all_windows),
           "unique_embeddings": len({p for w, _ in all_windows for p in w}),
           "leaky": {s: {"n": cen_leaky[s]["n"], "per_class": [cen_leaky[s]["per_class"][b] for b in WANTED_BEH]}
                     for s in ("train", "val", "test")}}
    want = {k: PUB_CENSUS[k] for k in ("rows", "windows", "unique_embeddings", "leaky")}
    checks = {k: got[k] == want[k] for k in want}
    return {"pass": all(checks.values()), "checks": checks, "got": got, "published": want}


def class_weights_np(l_tr):
    """VERBATIM cell 21 (numpy part)."""
    train_counts = np.bincount(l_tr, minlength=NUM_CLASSES)
    inv_freq = 1.0 / (train_counts + 1)
    weights = inv_freq / inv_freq.sum() * NUM_CLASSES
    return train_counts, weights


def top_confused(cm, cm_pct, k=10):
    """VERBATIM cell 23 loop (on the labels=range(9) matrix)."""
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append({'true': WANTED_BEH[i], 'predicted': WANTED_BEH[j], 'count': int(cm[i, j]),
                              'pct_of_true': float(cm_pct[i, j])})
    pairs.sort(key=lambda x: x['count'], reverse=True)
    return pairs[:k]


def metrics_block(y_true, y_pred, clips=None):
    """Cell-23 metrics (accuracy, sklearn-default macro-F1 and weighted-F1, per-class P/R/F1/support, confusion matrix,
    top confused pairs) on labels=range(9) so absent classes keep their row, + macro-F1 over the classes with test
    support, balanced accuracy, per-clip accuracy."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if yt.shape != yp.shape or yt.size == 0:
        raise ValueError("y_true / y_pred empty or of different length")
    labels = list(range(NUM_CLASSES))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # sklearn UndefinedMetricWarning for absent classes (scored 0, as cell 23)
        acc = float(accuracy_score(yt, yp))
        macro = float(f1_score(yt, yp, average='macro'))
        weighted = float(f1_score(yt, yp, average='weighted'))
        bal = float(balanced_accuracy_score(yt, yp))
        p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    supported = [k for k in labels if s[k] > 0]
    present = sorted(set(yt.tolist()) | set(yp.tolist()))
    out = {"n": int(yt.size), "correct": int((yt == yp).sum()), "accuracy": acc, "macro_f1": macro,
           "macro_f1_label_set": [WANTED_BEH[k] for k in present],
           "macro_f1_supported": float(np.mean([f[k] for k in supported])) if supported else None,
           "supported_classes": [WANTED_BEH[k] for k in supported],
           "weighted_f1": weighted, "balanced_accuracy": bal,
           "per_class": {WANTED_BEH[k]: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                                         "support": int(s[k]), "predicted": int(cm[:, k].sum())} for k in labels},
           "confusion_matrix": cm.tolist(),
           "confusion_matrix_pct": [[None if not np.isfinite(v) else float(v) for v in row] for row in cm_pct],
           "top_confused_pairs": top_confused(cm, cm_pct)}
    if clips is not None:
        cl = np.asarray(clips)
        out["per_clip"] = {c: {"n": int((cl == c).sum()), "accuracy": float((yt[cl == c] == yp[cl == c]).mean())}
                           for c in sorted(set(cl.tolist()))}
    return out


def majority_baseline(l_train, y_test, clips=None):
    """Predict the TRAIN split's most frequent class for every test window."""
    maj = int(np.bincount(l_train, minlength=NUM_CLASSES).argmax())
    m = metrics_block(y_test, [maj] * len(y_test), clips)
    return {"class": WANTED_BEH[maj], **{k: m[k] for k in ("accuracy", "macro_f1", "macro_f1_supported",
                                                            "weighted_f1", "balanced_accuracy")}}


def emb_rel(old_path):
    """Metadata 'embedding' (a blob-mount path, 2026-03) -> path relative to datasets.edinburgh_pigs.embeddings
    (relocation E05 'All Embeddings_DinoV3_21M/' -> 'dinov3_vits16_embeddings/'): <clip>/pig_<k>/<frame>_pig<k>.pt."""
    p = str(old_path).replace("\\", "/")
    key = "/" + OLD_EMB_DIR + "/"
    if p.count(key) != 1:
        raise ValueError(f"embedding path without exactly one '{key}': {old_path!r}")
    parts = p.split(key, 1)[1].split("/")
    if (len(parts) != 3 or not CLIP_RX.match(parts[0]) or not re.match(r"^pig_\d+$", parts[1])
            or not re.match(r"^\d{7}_pig\d+\.pt$", parts[2])):
        raise ValueError(f"unexpected embedding path layout: {old_path!r}")
    if parts[2].split("_pig", 1)[1][:-3] != parts[1][4:]:
        raise ValueError(f"pig folder / file name disagree: {old_path!r}")
    return "/".join(parts)


def check_emb_keys(df_k):
    """Each row's embedding path names the row's own clip, pig and frame (the windows key on the path)."""
    bad = 0
    for e, c, pg, fr in zip(df_k["embedding"], df_k["clip"], df_k["pig"], df_k["frame"]):
        clip_, pigdir, fname = emb_rel(e).split("/")
        if clip_ != c or pigdir != f"pig_{pg}" or int(fname[:7]) != int(fr):
            bad += 1
    if bad:
        raise ValueError(f"{bad} metadata rows whose embedding path disagrees with their clip / pig / frame")
    return True


def mean_sd(vals):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0}
    return {"mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else None,
            "min": min(v), "max": max(v), "n": len(v)}


def published_cm_counts(report, support):
    """Published confusion matrix in counts = row % x row support / 100 (rounded), from the report JSON."""
    pct = np.asarray(report["confusion_matrix_pct"], dtype=float)
    sup = np.asarray(support, dtype=float)[:, None]
    cm = np.rint(pct * sup / 100.0).astype(int)
    if not np.array_equal(cm.sum(axis=1), np.asarray(support, dtype=int)):
        raise ValueError("published confusion-matrix rows do not add up to the supports")
    return cm


def check_published_report(rep):
    """The catalog copy of the 2026-03 report JSON must hold the published numbers this notebook compares against."""
    bad = []
    for k in ("test_accuracy", "macro_f1", "weighted_f1"):
        if abs(float(rep[k]) - PUBLISHED[k]) > 1e-12:
            bad.append(k)
    for k in ("per_class_precision", "per_class_recall", "per_class_f1"):
        for b in WANTED_BEH:
            if abs(float(rep[k][b]) - PUBLISHED[k][b]) > 1e-12:
                bad.append(f"{k}.{b}")
    if bad:
        raise ValueError(f"published report differs from the pinned values: {bad}")
    return True


def compare_bridge(m, pub_cm):
    """Bridge verdict: the published checkpoint on the re-created leaky test split vs the published run."""
    cm = np.asarray(m["confusion_matrix"], dtype=int)
    pub_correct = int(np.trace(pub_cm))
    d_corr = int(m["correct"]) - pub_correct
    diff = cm - pub_cm
    out = {"published_correct": pub_correct, "published_n": int(pub_cm.sum()), "correct": int(m["correct"]),
           "n": int(m["n"]), "delta_correct": d_corr, "tol_windows": BRIDGE_TOL_WINDOWS,
           "cm_cells_differing": int((diff != 0).sum()), "cm_abs_diff_sum": int(np.abs(diff).sum()),
           "accuracy": m["accuracy"], "published_accuracy": PUBLISHED["test_accuracy"],
           "macro_f1": m["macro_f1"], "published_macro_f1": PUBLISHED["macro_f1"],
           "weighted_f1": m["weighted_f1"], "published_weighted_f1": PUBLISHED["weighted_f1"],
           "test_loss": m.get("test_loss"), "published_test_loss_4dp": PUBLISHED["test_loss_4dp"]}
    out["status"] = "PASS" if (abs(d_corr) <= BRIDGE_TOL_WINDOWS and int(m["n"]) == int(pub_cm.sum())) else "FAIL"
    return out


def config_diff(a, b, prefix=""):
    """Paths of differing leaves between two JSON-like objects."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += config_diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{prefix}.{k}" if prefix else str(k))
        return out
    return [] if json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str) else [prefix]


def jsonable(o):
    return json.loads(json.dumps(o, sort_keys=True, default=core._json_default))


# =====================================================================================================================
# torch parts: VERBATIM cells 17 / 19 / 21 / 23
# =====================================================================================================================
if TORCH_OK:
    class WindowDataset(Dataset):
        """Window dataset that reads from in-memory cache — zero disk I/O."""   # VERBATIM cell 17
        def __init__(self, windows, labels, cache):
            self.windows = windows
            self.labels = labels
            self.cache = cache

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            seq = torch.stack(
                [self.cache[p] for p in self.windows[idx]], dim=0
            )  # [SEQ_LEN, EMB_DIM] — instant, no disk access
            return seq, torch.tensor(self.labels[idx], dtype=torch.long)

    class BiLSTMClassifier(nn.Module):                                          # VERBATIM cell 19
        def __init__(self, emb_dim, hidden_dim=128, num_layers=1,
                     num_classes=9, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]  # Last timestep
            return self.classifier(last)


def _require_torch():
    if not TORCH_OK:
        raise RuntimeError("torch is not importable")


def new_model(device):
    _require_torch()
    return BiLSTMClassifier(emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
                            dropout=DROPOUT).to(device)


def _make_eval(model, criterion, device):
    def eval_epoch(loader, dataset_len):                                      # VERBATIM cell 21
        model.eval()
        total_loss, preds, labs = 0, [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                total_loss += loss.item() * X.size(0)
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(y.cpu().numpy())
        from sklearn.metrics import accuracy_score
        return total_loss / dataset_len, accuracy_score(labs, preds), labs, preds
    return eval_epoch


def train_recipe(seed, data, cache, device, workdir, epochs=NUM_EPOCHS, log=log):
    """Cells 17 (loaders) / 19 (model) / 21 (training) / 23 (best-checkpoint test pass), verbatim, seeded.
    data = {'train': (paths, labels), 'val': (...), 'test': (...)}. Returns the run record + test predictions."""
    _require_torch()
    from sklearn.metrics import accuracy_score
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    (p_tr, l_tr), (p_val, l_val), (p_test, l_test) = data["train"], data["val"], data["test"]
    train_ds = WindowDataset(p_tr, l_tr, cache)
    val_ds = WindowDataset(p_val, l_val, cache)
    test_ds = WindowDataset(p_test, l_test, cache)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    sample_x, sample_y = next(iter(train_loader))            # cell 17 shape check (draws one shuffle, as the original)
    if tuple(sample_x.shape[1:]) != (SEQ_LEN, EMB_DIM):
        raise AssertionError(f"batch shape {tuple(sample_x.shape)}")
    del sample_x, sample_y
    model = new_model(device)                                # cell 19
    train_counts, weights = class_weights_np(l_tr)          # cell 21
    class_weights = torch.tensor(weights, dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def train_epoch():
        model.train()
        total_loss, preds, labs = 0, [], []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            preds.extend(logits.argmax(1).cpu().numpy())
            labs.extend(y.cpu().numpy())
        return total_loss / len(train_ds), accuracy_score(labs, preds)

    eval_epoch = _make_eval(model, criterion, device)
    os.makedirs(workdir, exist_ok=True)
    best_model_path = osp.join(workdir, "best_bilstm.pt")
    best_val_loss = float('inf')
    no_improve = 0
    history = []
    best_epoch = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch()
        val_loss, val_acc, _, _ = eval_epoch(val_loader, len(val_ds))
        if not (math.isfinite(tr_loss) and math.isfinite(val_loss)):
            raise FloatingPointError(f"seed {seed} epoch {epoch}: non-finite loss (train {tr_loss}, val {val_loss})")
        # no wall times in the published history (a resumed unit must republish identical bytes); timings -> marker
        history.append({'epoch': epoch, 'tr_loss': tr_loss, 'tr_acc': float(tr_acc), 'val_loss': val_loss,
                        'val_acc': float(val_acc)})
        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch
            marker = ' *'
        else:
            no_improve += 1
        if epoch % 5 == 0 or epoch == 1 or marker:
            log(f"  seed {seed} E{epoch:02d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{marker}")
        if no_improve >= PATIENCE:
            log(f"  seed {seed}: early stopping at epoch {epoch} (patience={PATIENCE})")
            break
    train_s = time.time() - t0
    model.load_state_dict(torch.load(best_model_path, map_location=device))   # cell 23
    model.eval()
    test_loss, test_acc, y_true, y_pred = eval_epoch(test_loader, len(test_ds))
    vbest_loss, vbest_acc, _, _ = eval_epoch(val_loader, len(val_ds))
    return {"seed": seed, "best_epoch": best_epoch, "stopped_epoch": history[-1]["epoch"],
            "best_val_loss": best_val_loss, "val_loss_at_best": vbest_loss, "val_acc_at_best": float(vbest_acc),
            "test_loss": test_loss, "test_acc": float(test_acc), "train_counts": train_counts.tolist(),
            "class_weights": weights.tolist(), "history": history, "train_s": train_s,
            "y_true": [int(v) for v in y_true], "y_pred": [int(v) for v in y_pred], "ckpt": best_model_path}


def eval_state_dict(state_dict, data, cache, device):
    """The published checkpoint (or any state dict) on data['test'], with the cell-21 criterion of data['train']."""
    _require_torch()
    model = new_model(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _, weights = class_weights_np(data["train"][1])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))
    test_ds = WindowDataset(data["test"][0], data["test"][1], cache)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    loss, acc, y_true, y_pred = _make_eval(model, criterion, device)(loader, len(test_ds))
    return {"test_loss": loss, "test_acc": float(acc), "y_true": [int(v) for v in y_true],
            "y_pred": [int(v) for v in y_pred]}


def load_embedding_cache(paths, root, threads, cache_file=None, reader=None, log=log):
    """Cell 17's RAM cache {metadata path -> float32 [384]}, loaded once (each file -> torch.load(...).float(), exactly
    as cell 17, then a compact copy: the 2026-03 files hold the whole batch storage). Any failed, mis-shaped or
    non-finite embedding stops the run (the original silently dropped failures; it reported 0)."""
    _require_torch()
    uniq = sorted(set(paths))
    paths_sha = sha256_bytes("\n".join(uniq).encode("utf-8"))
    if cache_file and osp.isfile(cache_file):
        blob = torch.load(cache_file, map_location="cpu")
        if blob.get("paths_sha256") == paths_sha and list(blob["paths"]) == uniq:
            M = blob["matrix"]
            log(f"embeddings: local cache {cache_file} ({M.shape[0]} x {M.shape[1]})")
            return {p: M[i] for i, p in enumerate(uniq)}, _matrix_info(M, uniq, paths_sha, "local_cache", 0.0)
    t0 = time.time()
    read = reader or (lambda p: torch.load(root + "/" + emb_rel(p), map_location='cpu'))

    def _one(p):
        t = read(p).float()
        if tuple(t.shape) != (EMB_DIM,):
            raise ValueError(f"{p}: shape {tuple(t.shape)} != ({EMB_DIM},)")
        return t.clone()

    out = []
    chunk = 2000
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for k in range(0, len(uniq), chunk):
            out += list(pool.map(_one, uniq[k:k + chunk]))
            done = min(len(uniq), k + chunk)
            el = time.time() - t0
            log(f"embeddings: {done}/{len(uniq)} loaded ({el:.0f} s, {done / max(el, 1e-6):.0f} files/s)")
    M = torch.stack(out, 0).contiguous()
    if not torch.isfinite(M).all():
        raise ValueError("non-finite values in the embeddings")
    if float(M.std(0).max()) <= 1e-6:
        raise ValueError("collapsed embeddings (no variance)")
    secs = time.time() - t0
    if cache_file:
        os.makedirs(osp.dirname(cache_file), exist_ok=True)
        tmp = cache_file + ".tmp"
        torch.save({"paths": uniq, "paths_sha256": paths_sha, "matrix": M}, tmp)
        os.replace(tmp, cache_file)
    return {p: M[i] for i, p in enumerate(uniq)}, _matrix_info(M, uniq, paths_sha, "read", secs)


def _matrix_info(M, uniq, paths_sha, how, secs):
    a = M.numpy().astype("<f4", copy=False)
    return {"n": int(M.shape[0]), "dim": int(M.shape[1]), "paths_sha256": paths_sha,
            "matrix_sha256_f32le": sha256_bytes(a.tobytes()), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()), "how": how, "seconds": round(secs, 1)}


# =====================================================================================================================
# I/O (local file, then publish_no_overwrite to the Volume) and the flow
# =====================================================================================================================
class IO:
    def __init__(self, out_root, local_root, dbu=None):
        self.out, self.local, self.dbu = out_root.rstrip("/"), local_root.rstrip("/"), dbu
        os.makedirs(self.local, exist_ok=True)

    def vol(self, rel):
        return self.out + "/" + rel

    def loc(self, rel):
        p = self.local + "/" + rel
        os.makedirs(osp.dirname(p), exist_ok=True)
        return p

    def exists(self, rel):
        return osp.exists(self.vol(rel))

    def read_json(self, rel):
        with open(self.vol(rel), encoding="utf-8") as f:
            return json.load(f)

    def publish_file(self, rel, local_path):
        return core.publish_no_overwrite(local_path, self.vol(rel), dbutils=self.dbu)

    def publish_json(self, rel, obj):
        p = self.loc(rel)
        core.write_json_atomic(p, obj)
        return self.publish_file(rel, p)

    def publish_text(self, rel, text):
        p = self.loc(rel)
        core.write_text_atomic(p, text)
        return self.publish_file(rel, p)

    def publish_ckpt(self, rel, local_path):
        """publish_no_overwrite for a state dict whose zip container may differ in bytes only (zip entry times): an
        existing file with the SAME tensors (same keys, torch.equal) counts as identical; anything else still raises."""
        dst = self.vol(rel)
        if osp.exists(dst) and core.sha256_file(dst) != core.sha256_file(local_path):
            a = torch.load(dst, map_location="cpu")
            b = torch.load(local_path, map_location="cpu")
            if list(a.keys()) == list(b.keys()) and all(torch.equal(a[k], b[k]) for k in a):
                return "identical_tensors"
            raise FileExistsError(f"{dst} exists with different tensors; refusing to overwrite (use a new tag)")
        return self.publish_file(rel, local_path)


def write_predictions_csv(path, idx, meta, y_true, y_pred):
    os.makedirs(osp.dirname(path), exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["window", "clip", "pig", "frame_start", "y_true", "y_pred", "true", "pred"])
    for i, t, p in zip(idx, y_true, y_pred):
        m = meta[i]
        w.writerow([i, m["clip"], m["pig"], m["frames"][0], t, p, WANTED_BEH[t], WANTED_BEH[p]])
    core.write_text_atomic(path, buf.getvalue())
    return path


def split_data(all_windows, idx_split):
    return {s: ([all_windows[i][0] for i in idx], [all_windows[i][1] for i in idx]) for s, idx in idx_split.items()}


def unit_list(arms, seeds):
    out = []
    for a in ARMS_ALL:
        if a not in arms:
            continue
        if a == "bridge":
            out.append(("bridge", None))
        else:
            out += [(a, s) for s in seeds]
    return out


def unit_dir(arm, seed):
    return "bridge" if arm == "bridge" else f"{arm}/seed_{seed}"


def unit_marker(arm, seed):
    return ".done_bridge" if arm == "bridge" else f".done_{arm}_seed{seed}"


def aggregate_arm(ms):
    """Seed aggregation for one arm: mean / SD of the headline metrics, per class, per clip; pooled confusion matrix."""
    keys = ("accuracy", "macro_f1", "macro_f1_supported", "weighted_f1", "balanced_accuracy", "test_loss",
            "best_epoch", "stopped_epoch")
    out = {"seeds": [m["seed"] for m in ms], "n_test": ms[0]["test"]["n"]}
    for k in keys:
        out[k] = mean_sd([(m["test"].get(k) if k in m["test"] else m.get(k)) for m in ms])
    out["per_class"] = {b: {q: mean_sd([m["test"]["per_class"][b][q] for m in ms]) for q in ("precision", "recall", "f1")}
                        | {"support": ms[0]["test"]["per_class"][b]["support"]} for b in WANTED_BEH}
    cm = np.sum([np.asarray(m["test"]["confusion_matrix"], dtype=int) for m in ms], axis=0)
    out["pooled_confusion_matrix"] = cm.tolist()
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    out["pooled_confusion_matrix_pct"] = [[None if not np.isfinite(v) else float(v) for v in r] for r in pct]
    out["pooled_top_confused_pairs"] = top_confused(cm, np.nan_to_num(pct))
    if "per_clip" in ms[0]["test"]:
        out["per_clip_accuracy"] = {c: mean_sd([m["test"]["per_clip"][c]["accuracy"] for m in ms])
                                    | {"n": ms[0]["test"]["per_clip"][c]["n"]} for c in ms[0]["test"]["per_clip"]}
    return out


def _pct(x, d=2):
    return "–" if x is None else f"{100 * x:.{d}f}"


def _ms(a, d=2):
    if a is None or a.get("mean") is None:
        return "–"
    return f"{100 * a['mean']:.{d}f}" + ("" if a.get("sd") is None else f" ± {100 * a['sd']:.{d}f}")


def build_summary(units_done, cen, ov, smoke, cfg_brief):
    """SUMMARY.json + SUMMARY.md from every unit's metrics (pure)."""
    S = {"nb_version": NB_VERSION, "status": "SMOKE" if smoke else "COMPLETE", "config": cfg_brief,
         "published_leaky_2026_03": {k: PUBLISHED[k] for k in ("test_accuracy", "macro_f1", "weighted_f1")},
         "census": {k: {s: {"n": v["n"], "per_class": v["per_class"]} for s, v in cen[k].items()} for k in cen},
         "leakage": ov, "arms": {}}
    if "bridge" in units_done:
        b = units_done["bridge"]
        S["bridge"] = {k: b["compare"][k] for k in ("status", "correct", "n", "published_correct", "delta_correct",
                                                   "cm_cells_differing", "accuracy", "macro_f1", "weighted_f1",
                                                   "test_loss")}
    for arm in ("leaky", "clip"):
        ms = [units_done[k] for k in sorted(units_done) if k.startswith(arm + "_seed")]
        if ms:
            S["arms"][arm] = aggregate_arm(ms)
            S["arms"][arm]["per_seed"] = {str(m["seed"]): {k: m["test"][k] for k in ("accuracy", "macro_f1",
                                                                                     "macro_f1_supported",
                                                                                     "weighted_f1")}
                                          | {"best_epoch": m["best_epoch"], "stopped_epoch": m["stopped_epoch"]}
                                          for m in ms}
            S["arms"][arm]["majority_baseline"] = ms[0]["majority_baseline"]
    c, lk = S["arms"].get("clip"), S["arms"].get("leaky")
    if c:
        S["delta_clip_vs_published"] = {k: c[k]["mean"] - PUBLISHED[p] for k, p in
                                        (("accuracy", "test_accuracy"), ("macro_f1", "macro_f1"),
                                         ("weighted_f1", "weighted_f1"))}
    if c and lk:
        S["delta_clip_vs_leaky_retrain"] = {k: c[k]["mean"] - lk[k]["mean"]
                                            for k in ("accuracy", "macro_f1", "weighted_f1")}
    # ---- markdown
    L = [f"# J10 behaviour re-split by clip — {S['status']} ({NB_VERSION})", "",
         "DINOv3-ViT-S embeddings (Edinburgh, 2026-03) + BiLSTM, recipe verbatim (cells 15–23); only the split changes. "
         "Mean ± SD over seeds " + ", ".join(str(s) for s in cfg_brief["seeds"]) + " (test set, %).", "",
         "| model / split | test windows | accuracy | macro-F1 | weighted-F1 | macro-F1 (supported classes) |",
         "|---|---|---|---|---|---|",
         f"| published, window-level random split (2026-03) | 4,292 | {_pct(PUBLISHED['test_accuracy'])} | "
         f"{_pct(PUBLISHED['macro_f1'])} | {_pct(PUBLISHED['weighted_f1'])} | {_pct(PUBLISHED['macro_f1'])} |"]
    if "bridge" in S:
        b = S["bridge"]
        L.append(f"| published checkpoint, re-created window-level test split (bridge {b['status']}, "
                 f"Δ {b['delta_correct']:+d} windows) | {b['n']:,} | {_pct(b['accuracy'])} | {_pct(b['macro_f1'])} | "
                 f"{_pct(b['weighted_f1'])} | – |")
    if lk:
        L.append(f"| retrained here, window-level random split | {lk['n_test']:,} | {_ms(lk['accuracy'])} | "
                 f"{_ms(lk['macro_f1'])} | {_ms(lk['weighted_f1'])} | {_ms(lk['macro_f1_supported'])} |")
    if c:
        L.append(f"| **retrained here, clip-level split (train 0–3 / val 4 / test 5–8)** | {c['n_test']:,} | "
                 f"**{_ms(c['accuracy'])}** | **{_ms(c['macro_f1'])}** | **{_ms(c['weighted_f1'])}** | "
                 f"{_ms(c['macro_f1_supported'])} |")
        mb = c["majority_baseline"]
        L.append(f"| majority class of the clip-level train split ({mb['class']}) | {c['n_test']:,} | "
                 f"{_pct(mb['accuracy'])} | {_pct(mb['macro_f1'])} | {_pct(mb['weighted_f1'])} | "
                 f"{_pct(mb['macro_f1_supported'])} |")
    L += ["", "Macro-F1 = sklearn's default (classes in y_true ∪ y_pred), as the published cell 23; "
          "'supported classes' averages only classes with test windows."]
    if c:
        L += ["", "## Per class, clip-level split (mean ± SD over seeds; published = window-level split)", "",
              "| class | train windows | test windows | precision | recall | F1 | published F1 |",
              "|---|---|---|---|---|---|---|"]
        for bname in WANTED_BEH:
            pc = c["per_class"][bname]
            L.append(f"| {bname} | {cen['clip']['train']['per_class'][bname]:,} | {pc['support']:,} | "
                     f"{_ms(pc['precision'], 1)} | {_ms(pc['recall'], 1)} | {_ms(pc['f1'], 1)} | "
                     f"{_pct(PUBLISHED['per_class_f1'][bname], 1)} |")
        if "per_clip_accuracy" in c:
            L += ["", "## Accuracy per test clip (clip-level split)", "", "| clip | windows | accuracy |", "|---|---|---|"]
            for cl, v in c["per_clip_accuracy"].items():
                L.append(f"| {cl} | {v['n']:,} | {_ms(v, 2)} |")
        L += ["", "## Per seed (clip-level split)", "", "| seed | accuracy | macro-F1 | weighted-F1 | best / stop epoch |",
              "|---|---|---|---|---|"]
        for s, v in c["per_seed"].items():
            L.append(f"| {s} | {_pct(v['accuracy'])} | {_pct(v['macro_f1'])} | {_pct(v['weighted_f1'])} | "
                     f"{v['best_epoch']} / {v['stopped_epoch']} |")
        L += ["", "Pooled confusion matrix (sum over seeds; rows = true, columns = predicted, order "
              + ", ".join(WANTED_BEH) + "):", "", "```"]
        L += [" ".join(f"{v:6d}" for v in row) for row in c["pooled_confusion_matrix"]]
        L += ["```"]
    lv = ov.get("leaky", {}).get("vs_train", {}).get("test")
    if lv:
        L += ["", f"Leakage of the window-level split: {100 * lv['share_ge1_crop_in_train']:.1f} % of its test windows "
              f"share ≥ 1 crop with a training window, {100 * lv['share_ge2_crops_in_train']:.1f} % share ≥ 2. "
              "Clip-level split: no crop and no clip shared between train, val and test (asserted)."]
    if c:
        L += ["", "Clip-level test windows per class: " + ", ".join(
            f"{b} {n:,}" for b, n in cen["clip"]["test"]["per_class"].items()) + "."]
    return S, "\n".join(L) + "\n"


def run_resplit(io_, *, df_meta, split_clips, arms, seeds, epochs, device, emb_loader, published_sd=None,
                published_report=None, expect_published=True, cfg_base=None, smoke=False, log=log):
    """The whole J10 flow (also driven by the selftest with synthetic data). Returns the run summary dict."""
    t_flow = time.time()
    rec = {"units": {}, "timings": {}}
    dfk = add_keys(df_meta)
    check_emb_keys(dfk)
    all_windows, meta = build_windows(dfk)
    n = len(all_windows)
    splits = {"leaky": leaky_split(all_windows), "clip": clip_split(meta, split_clips)}
    for s in splits.values():
        check_partition(s, n)
    lp = leaky_split_paths(all_windows)
    for s in ("train", "val", "test"):
        if [tuple(all_windows[i][0]) for i in splits["leaky"][s]] != [tuple(p) for p in lp[s]]:
            raise AssertionError(f"index-based leaky split differs from the verbatim path-based split ({s})")
    cen = {k: census(all_windows, meta, v) for k, v in splits.items()}
    ov = {k: overlap_stats(all_windows, meta, v) for k, v in splits.items()}
    assert_no_leak(ov["clip"])
    if ov["clip"]["non_consecutive_windows"]:
        raise AssertionError(f"{ov['clip']['non_consecutive_windows']} windows are not 3 consecutive frames")
    data_clips = sorted({m["clip"] for m in meta})
    if data_clips != sorted(c for v in split_clips.values() for c in v):
        raise AssertionError(f"metadata clips {data_clips} != split clips {split_clips}")
    gates = published_census_gates(len(df_meta), all_windows, cen["leaky"]) if expect_published else None
    if gates is not None and not gates["pass"]:
        raise AssertionError(f"re-created windows differ from the published census: {gates['checks']}")
    cfg = dict(cfg_base or {})
    cfg.update({"nb_version": NB_VERSION, "recipe": RECIPE, "arms": list(arms), "seeds": list(seeds),
                "epochs": int(epochs), "device": str(device), "smoke": bool(smoke),
                "split": {"clip": {k: list(v) for k, v in split_clips.items()},
                          "leaky": {"test_size": LEAKY_TEST_SIZE, "then_test_size": LEAKY_TEST_HALF,
                                    "random_state": LEAKY_RANDOM_STATE, "stratify": "window label"}},
                "bridge_tol_windows": BRIDGE_TOL_WINDOWS})
    cfg = jsonable(cfg)
    if io_.exists("RESPLIT_CONFIG.json"):
        d = config_diff(io_.read_json("RESPLIT_CONFIG.json"), cfg)
        if d:
            raise RuntimeError(f"RESPLIT_CONFIG.json of this tag differs ({d[:12]}); use a new tag")
    else:
        io_.publish_json("RESPLIT_CONFIG.json", cfg)
    io_.publish_json("census.json", jsonable({"windows": n, "rows": len(df_meta), "census": cen, "leakage": ov,
                                              "published_gates": gates,
                                              "class_weights": {k: class_weights_np([all_windows[i][1] for i in
                                                                                     v["train"]])[1].tolist()
                                                                for k, v in splits.items()}}))
    log(f"windows {n}; clip split {[cen['clip'][s]['n'] for s in ('train', 'val', 'test')]}; leaky split "
        f"{[cen['leaky'][s]['n'] for s in ('train', 'val', 'test')]}; leaky test windows sharing >= 2 crops with "
        f"train: {100 * ov['leaky']['vs_train']['test']['share_ge2_crops_in_train']:.1f} %")
    units = unit_list(arms, seeds)
    todo = [u for u in units if not io_.exists(unit_marker(*u))]
    cache = None
    if todo:
        cache, info = emb_loader(sorted({p for w, _ in all_windows for p in w}))
        rec["embeddings"] = info
        io_.publish_json("data/embeddings_digest.json", {k: v for k, v in info.items() if k not in ("how", "seconds")})
        log(f"embeddings: {info['n']} x {info['dim']}, sha256 {info['matrix_sha256_f32le'][:16]} ({info['how']}, "
            f"{info['seconds']} s)")
    done = {}
    for arm, seed in units:
        key = "bridge" if arm == "bridge" else f"{arm}_seed{seed}"
        ud = unit_dir(arm, seed)
        if io_.exists(unit_marker(arm, seed)):
            done[key] = io_.read_json(ud + "/metrics.json")
            if arm == "bridge":
                done[key]["compare"] = io_.read_json("bridge/compare.json")
            rec["units"][key] = "resumed"
            continue
        t_u = time.time()
        sp = splits["leaky" if arm in ("bridge", "leaky") else "clip"]
        data = split_data(all_windows, sp)
        test_clips = [meta[i]["clip"] for i in sp["test"]]
        if arm == "bridge":
            if published_sd is None or published_report is None:
                raise RuntimeError("bridge needs the published checkpoint and report")
            r = eval_state_dict(published_sd, data, cache, device)
            m = {"arm": "bridge", "seed": None, "test": metrics_block(r["y_true"], r["y_pred"], test_clips)}
            m["test"]["test_loss"] = r["test_loss"]
            sup = [m["test"]["per_class"][b]["support"] for b in WANTED_BEH]
            cmp_ = compare_bridge(m["test"], published_cm_counts(published_report, sup))
            io_.publish_json(ud + "/metrics.json", jsonable(m))
            io_.publish_file(ud + "/predictions.csv",
                             write_predictions_csv(io_.loc(ud + "/predictions.csv"), sp["test"], meta,
                                                   r["y_true"], r["y_pred"]))
            io_.publish_json("bridge/compare.json", jsonable(cmp_))
            m["compare"] = cmp_
            log(f"bridge {cmp_['status']}: {cmp_['correct']}/{cmp_['n']} correct vs published "
                f"{cmp_['published_correct']} (Δ {cmp_['delta_correct']:+d}); acc {100 * m['test']['accuracy']:.2f} %")
        else:
            r = train_recipe(seed, data, cache, device, io_.loc(ud), epochs=epochs, log=log)
            if r["y_true"] != data["test"][1]:
                raise AssertionError("test labels came back in a different order")
            m = {"arm": arm, "seed": seed, "best_epoch": r["best_epoch"], "stopped_epoch": r["stopped_epoch"],
                 "best_val_loss": r["best_val_loss"], "val_loss_at_best": r["val_loss_at_best"],
                 "val_acc_at_best": r["val_acc_at_best"], "train_counts": r["train_counts"],
                 "class_weights": r["class_weights"],
                 "test": metrics_block(r["y_true"], r["y_pred"], test_clips),
                 "majority_baseline": majority_baseline(data["train"][1], data["test"][1], test_clips)}
            m["test"]["test_loss"] = r["test_loss"]
            if abs(m["test"]["accuracy"] - r["test_acc"]) > 1e-12:
                raise AssertionError("metrics_block accuracy != eval_epoch accuracy")
            io_.publish_json(ud + "/metrics.json", jsonable(m))
            io_.publish_json(ud + "/history.json", jsonable(r["history"]))
            io_.publish_file(ud + "/predictions.csv",
                             write_predictions_csv(io_.loc(ud + "/predictions.csv"), sp["test"], meta,
                                                   r["y_true"], r["y_pred"]))
            io_.publish_ckpt(ud + "/best_bilstm.pt", r["ckpt"])
            rec["timings"][key + "_train_s"] = round(r["train_s"], 1)
            log(f"{arm} seed {seed}: acc {100 * m['test']['accuracy']:.2f} macro-F1 {100 * m['test']['macro_f1']:.2f} "
                f"weighted-F1 {100 * m['test']['weighted_f1']:.2f} (best epoch {r['best_epoch']}, stop "
                f"{r['stopped_epoch']}, {r['train_s']:.0f} s)")
        io_.publish_json(unit_marker(arm, seed), {"unit": key, "seconds": round(time.time() - t_u, 1),
                                                  "train_s": rec["timings"].get(key + "_train_s"),
                                                  "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        done[key] = m
        rec["units"][key] = "ran"
    if all(("bridge" if a == "bridge" else f"{a}_seed{s}") in done for a, s in units):
        brief = {"arms": list(arms), "seeds": list(seeds), "epochs": int(epochs), "device": str(device),
                 "split_clips": {k: list(v) for k, v in split_clips.items()}}
        S, md = build_summary(done, cen, ov, smoke, brief)
        io_.publish_json("SUMMARY.json", jsonable(S))
        io_.publish_text("SUMMARY.md", md)
        rec["summary"] = S
        rec["summary_md"] = md
    rec["timings"]["flow_s"] = round(time.time() - t_flow, 1)
    return rec


# =====================================================================================================================
# selftest (synthetic data; torch parts only where torch is installed -- always on the cluster)
# =====================================================================================================================
def _synthetic_meta(clips, pigs=2, frames=40, seed=0, extra_beh=("walk", "investigating")):
    """Metadata rows with mount-style paths, rows shuffled; behaviours in runs; returns (df, {path: emb})."""
    import pandas as pd
    rng = np.random.RandomState(seed)
    protos = rng.normal(0, 1, (NUM_CLASSES + len(extra_beh), EMB_DIM)).astype(np.float32)
    names = WANTED_BEH + list(extra_beh)
    rows, emb = [], {}
    for clip in clips:
        for pig in range(pigs):
            f = 1
            while f <= frames:
                b = int(rng.randint(len(names)))
                run = int(rng.randint(3, 9))
                for _ in range(run):
                    if f > frames:
                        break
                    e = (f"/dbfs/mnt/playbehavior/Pig Behavior Edinburgh/{OLD_EMB_DIR}/{clip}/pig_{pig}/"
                         f"{f:07d}_pig{pig}.pt")
                    rows.append({"raw_frame": f"/dbfs/mnt/playbehavior/Pig Behavior Edinburgh/Decoded Frames/{clip}/"
                                              f"video1/{f:07d}.jpg",
                                 "cropped_frame": f"/dbfs/mnt/playbehavior/Pig Behavior Edinburgh/All Cropped Frames/"
                                                  f"{clip}/pig_{pig}/{f:07d}_pig{pig}.jpg",
                                 "embedding": e, "behavior": names[b]})
                    emb[e] = protos[b] + 0.3 * rng.normal(0, 1, EMB_DIM).astype(np.float32)
                    f += 1
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df, emb


def _selftest():
    import pandas as pd
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))

    def raises(fn, exc=Exception):
        try:
            fn()
        except exc:
            return True
        return False

    # 1. verbatim window rule: sort by (clip, pig-as-string, frame); majority; ties -> first behaviour; drop non-classes
    df = pd.DataFrame({
        "raw_frame": [f"x/Decoded Frames/2019_01_01_000001/video1/{f:07d}.jpg" for f in (3, 1, 2, 4, 5, 1, 2, 3)],
        "cropped_frame": [f"x/All Cropped Frames/2019_01_01_000001/pig_{p}/a.jpg" for p in (0, 0, 0, 0, 0, 1, 1, 1)],
        "embedding": [f"x/{OLD_EMB_DIR}/2019_01_01_000001/pig_{p}/{f:07d}_pig{p}.pt"
                      for p, f in ((0, 3), (0, 1), (0, 2), (0, 4), (0, 5), (1, 1), (1, 2), (1, 3))],
        "behavior": ["eat", "sleep", "eat", "walk", "walk", "run", "drink", "lying"]})
    dk = add_keys(df)
    check("add_keys sorts by clip, pig, frame", dk["frame"].tolist() == [1, 2, 3, 4, 5, 1, 2, 3]
          and dk["pig"].tolist() == ["0"] * 5 + ["1"] * 3)
    aw, mt = build_windows(dk)
    # pig 0: [sleep,eat,eat]->eat, [eat,eat,walk]->eat, [eat,walk,walk]->walk (dropped); pig 1: [run,drink,lying]->run
    check("windows: majority, tie -> first, non-class dropped",
          [lab for _, lab in aw] == [2, 2, 6] and [m["frames"] for m in mt] == [(1, 2, 3), (2, 3, 4), (1, 2, 3)],
          str([(lab, m) for (_, lab), m in zip(aw, mt)]))
    check("check_emb_keys accepts consistent rows", check_emb_keys(dk))
    bad = dk.copy()
    bad.loc[0, "embedding"] = bad.loc[1, "embedding"].replace("0000002", "0000009")
    check("check_emb_keys refuses a row whose path names another frame", raises(lambda: check_emb_keys(bad)))
    check("add_keys refuses duplicate rows", raises(lambda: add_keys(pd.concat([df, df.iloc[:1]]))))

    # 2. embedding path mapping (E05)
    e = f"/dbfs/mnt/playbehavior/Pig Behavior Edinburgh/{OLD_EMB_DIR}/2019_11_05_000002/pig_3/0000001_pig3.pt"
    check("emb_rel maps the mount path", emb_rel(e) == "2019_11_05_000002/pig_3/0000001_pig3.pt")
    check("emb_rel refuses a foreign layout", raises(lambda: emb_rel("/x/Other/2019_11_05_000002/pig_3/1.pt")))
    check("emb_rel refuses '..'", raises(lambda: emb_rel(f"/x/{OLD_EMB_DIR}/../pig_3/0000001_pig3.pt")))
    check("emb_rel refuses pig folder / file mismatch",
          raises(lambda: emb_rel(f"/x/{OLD_EMB_DIR}/2019_11_05_000002/pig_3/0000001_pig4.pt")))

    # 3. class weights = the published cell-21 printout
    lab = np.repeat(np.arange(NUM_CLASSES), PUB_CENSUS["leaky"]["train"]["per_class"])
    _, w = class_weights_np(lab)
    check("class weights reproduce the published printout", [round(float(x), 3) for x in w]
          == PUB_CENSUS["class_weights_3dp"], str([round(float(x), 3) for x in w]))

    # 4. metrics: absent classes, macro definitions, cm shape, per clip, published report arithmetic
    yt, yp = [0, 0, 1, 1, 2, 5, 5, 5], [0, 1, 1, 1, 2, 5, 5, 8]
    m = metrics_block(yt, yp, clips=["a"] * 4 + ["b"] * 4)
    f_manual = np.mean([2 / 3, 0.8, 1.0, 0.8, 0.0])     # classes 0,1,2,5 + predicted-only 8 (F1 0)
    check("macro-F1 = sklearn default (union of y_true and y_pred)", abs(m["macro_f1"] - f_manual) < 1e-12,
          f"{m['macro_f1']} vs {f_manual}")
    check("macro-F1 over supported classes", abs(m["macro_f1_supported"] - np.mean([2 / 3, 0.8, 1.0, 0.8])) < 1e-12)
    check("confusion matrix is 9 x 9 and counts every window", np.asarray(m["confusion_matrix"]).shape == (9, 9)
          and int(np.asarray(m["confusion_matrix"]).sum()) == 8 and m["correct"] == 6)
    check("per-clip accuracy", m["per_clip"]["a"]["accuracy"] == 0.75 and m["per_clip"]["b"]["accuracy"] == 0.75)
    check("absent class keeps a zero row", m["per_class"]["run"]["support"] == 0
          and m["confusion_matrix_pct"][6][0] is None)
    mb = majority_baseline([5, 5, 0], yt)
    check("majority baseline predicts the train majority", mb["class"] == "sleep" and abs(mb["accuracy"] - 3 / 8) < 1e-12)
    sup = PUB_CENSUS["leaky"]["test"]["per_class"]
    diag = [round(PUBLISHED["per_class_recall"][b] * s) for b, s in zip(WANTED_BEH, sup)]
    check("published recall x support adds up to the published accuracy (4,178 / 4,292)",
          sum(diag) == 4178 and abs(4178 / 4292 - PUBLISHED["test_accuracy"]) < 1e-12)
    check("published macro-F1 = mean of the per-class F1",
          abs(np.mean(list(PUBLISHED["per_class_f1"].values())) - PUBLISHED["macro_f1"]) < 1e-12)
    pct = [[100.0 * (i == j) for j in range(9)] for i in range(9)]
    check("published_cm_counts from row % x support", np.array_equal(
        published_cm_counts({"confusion_matrix_pct": pct}, sup), np.diag(sup)))
    fake_m = {"confusion_matrix": np.diag(sup).tolist(), "correct": sum(sup) - 1, "n": sum(sup), "accuracy": 0,
              "macro_f1": 0, "weighted_f1": 0}
    pub_cm = np.diag(sup)
    check("bridge PASS within tolerance", compare_bridge(fake_m, pub_cm)["status"] == "PASS")
    fake_m["correct"] = sum(sup) - 3
    check("bridge FAIL beyond tolerance", compare_bridge(fake_m, pub_cm)["status"] == "FAIL")
    rep = {"test_accuracy": PUBLISHED["test_accuracy"], "macro_f1": PUBLISHED["macro_f1"],
           "weighted_f1": PUBLISHED["weighted_f1"], "per_class_precision": dict(PUBLISHED["per_class_precision"]),
           "per_class_recall": dict(PUBLISHED["per_class_recall"]), "per_class_f1": dict(PUBLISHED["per_class_f1"])}
    check("check_published_report accepts the pinned values", check_published_report(rep))
    rep["macro_f1"] += 1e-6
    check("check_published_report refuses a changed value", raises(lambda: check_published_report(rep)))

    # 5. splits: clip split partition + no leak; leaky split index == verbatim paths; leak detector fires
    clips9 = [f"2019_01_0{k + 1}_00000{k}" for k in range(9)]
    sc = {"train": tuple(clips9[0:4]), "val": (clips9[4],), "test": tuple(clips9[5:9])}
    dfs, embs = _synthetic_meta(clips9, pigs=2, frames=100, seed=1)
    aw, mt = build_windows(add_keys(dfs))
    cs = clip_split(mt, sc)
    check_partition(cs, len(aw))
    ovc = overlap_stats(aw, mt, cs)
    check("clip split: no shared crops / clips", all(not v["shared_crops"] and not v["shared_clips"]
                                                    for v in ovc["pairs"].values()))
    check("clip split: every window from its split's clips", all(mt[i]["clip"] in sc[s] for s, idx in cs.items()
                                                                 for i in idx))
    ls_ = leaky_split(aw)
    lp = leaky_split_paths(aw)
    check("leaky split on indices == verbatim split on paths", all(
        [tuple(aw[i][0]) for i in ls_[s]] == [tuple(p) for p in lp[s]] for s in ("train", "val", "test")))
    ovl = overlap_stats(aw, mt, ls_)
    check("leak detector: the window-level split shares crops", ovl["pairs"]["train|test"]["shared_crops"] > 0
          and ovl["vs_train"]["test"]["share_ge1_crop_in_train"] > 0.5)
    check("assert_no_leak fires on the window-level split", raises(lambda: assert_no_leak(ovl), AssertionError))
    check("clip_split refuses a clip in two splits", raises(lambda: clip_split(mt, {"a": (clips9[0],),
                                                                                 "b": (clips9[0],)})))
    check("clip_split refuses an unassigned clip", raises(lambda: clip_split(mt, {"a": tuple(clips9[:8])})))
    check("unit list order", unit_list(["clip", "bridge"], [3, 4]) == [("bridge", None), ("clip", 3), ("clip", 4)])
    check("config_diff", config_diff({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 3}}) == ["b.c"])

    # 6. real metadata (if available locally): census = the published one
    real = os.environ.get("TOL69_META") or osp.join(osp.dirname(CODE_DIR), "t69", "_vol", META_FILE)
    if osp.isfile(real):
        dfr = pd.read_csv(real)
        dkr = add_keys(dfr)
        check_emb_keys(dkr)
        awr, mtr = build_windows(dkr)
        lsr = leaky_split(awr)
        g = published_census_gates(len(dfr), awr, census(awr, mtr, lsr))
        check("REAL metadata: census = the published cell-15 output", g["pass"], str(g["checks"]))
        csr = clip_split(mtr, SPLIT_CLIPS)
        ovr = overlap_stats(awr, mtr, csr)
        check("REAL metadata: clip split leak-free", all(not v["shared_crops"] and not v["shared_clips"]
                                                         for v in ovr["pairs"].values())
              and ovr["non_consecutive_windows"] == 0)
        cr = census(awr, mtr, csr)
        check("REAL metadata: clip split sizes 9,845 / 3,105 / 15,659",
              [cr[s]["n"] for s in ("train", "val", "test")] == [9845, 3105, 15659],
              str([cr[s]["n"] for s in ("train", "val", "test")]))
        ol = overlap_stats(awr, mtr, lsr)["vs_train"]["test"]
        print(f"      real leaky test windows sharing >=1 / >=2 crops with train: "
              f"{100 * ol['share_ge1_crop_in_train']:.1f} % / {100 * ol['share_ge2_crops_in_train']:.1f} %")
    else:
        print(f"SKIP real-metadata census ({real} absent)")

    # 7. torch: verbatim model size, training on separable synthetic data, determinism, the whole flow + resume
    if TORCH_OK:
        torch.use_deterministic_algorithms(True)
        nparam = sum(p.numel() for p in new_model("cpu").parameters())
        check("BiLSTM parameter count = 560,393 (published '560.4K')", nparam == 560393, str(nparam))
        cache = {p: torch.from_numpy(v) for p, v in embs.items()}
        data = split_data(aw, cs)
        tmp = tempfile.mkdtemp(prefix="tol69_st_")
        try:
            r1 = train_recipe(0, data, cache, "cpu", osp.join(tmp, "a"), epochs=12, log=lambda *a: None)
            r2 = train_recipe(0, data, cache, "cpu", osp.join(tmp, "b"), epochs=12, log=lambda *a: None)
            r3 = train_recipe(1, data, cache, "cpu", osp.join(tmp, "c"), epochs=12, log=lambda *a: None)
            sd1, sd2, sd3 = (torch.load(r["ckpt"], map_location="cpu") for r in (r1, r2, r3))
            check("training learns the synthetic task (test acc > 0.6; chance ~0.11)", r1["test_acc"] > 0.6,
                  str(r1["test_acc"]))
            check("same seed -> identical predictions, history and tensors", r1["y_pred"] == r2["y_pred"]
                  and r1["history"] == r2["history"] and all(torch.equal(sd1[k], sd2[k]) for k in sd1))
            print(f"      (info) same-seed checkpoint bytes identical: "
                  f"{core.sha256_file(r1['ckpt']) == core.sha256_file(r2['ckpt'])}")
            check("different seed -> different weights", not all(torch.equal(sd1[k], sd3[k]) for k in sd1))
            iok = IO(osp.join(tmp, "VOLK"), osp.join(tmp, "LOCALK"))
            os.makedirs(iok.out, exist_ok=True)
            torch.save(sd1, iok.vol("x.pt"))                                  # the "already published" file
            legacy = osp.join(tmp, "other_name.pt")    # same tensors; the zip's archive folder = file stem -> new bytes
            torch.save(sd2, legacy)
            check("publish_ckpt: same tensors in different bytes count as identical",
                  core.sha256_file(legacy) != core.sha256_file(iok.vol("x.pt"))
                  and iok.publish_ckpt("x.pt", legacy) == "identical_tensors")
            check("publish_ckpt: different tensors are refused", raises(lambda: iok.publish_ckpt("x.pt", r3["ckpt"]),
                                                                       FileExistsError))
            check("best epoch = argmin val loss", r1["best_epoch"] == 1 + int(np.argmin(
                [h["val_loss"] for h in r1["history"]])))
            sd = torch.load(r1["ckpt"], map_location="cpu")
            e1 = eval_state_dict(sd, data, cache, "cpu")
            check("eval_state_dict reproduces the trained run's test pass", e1["y_pred"] == r1["y_pred"]
                  and abs(e1["test_loss"] - r1["test_loss"]) < 1e-9)
            pat = train_recipe(0, data, cache, "cpu", osp.join(tmp, "d"), epochs=60, log=lambda *a: None)
            h = pat["history"]
            check("early stopping: stop = best + patience (or the epoch limit)",
                  pat["stopped_epoch"] == min(60, pat["best_epoch"] + PATIENCE), f"{pat['best_epoch']} {pat['stopped_epoch']}")
            check("no improvement after the best epoch", all(x["val_loss"] >= pat["best_val_loss"]
                                                             for x in h[pat["best_epoch"]:]))
            # the whole flow: published = the leaky seed-0 model, so the bridge must PASS with delta 0
            lsd = split_data(aw, ls_)
            rl = train_recipe(0, lsd, cache, "cpu", osp.join(tmp, "pub"), epochs=4, log=lambda *a: None)
            pm = metrics_block(rl["y_true"], rl["y_pred"])
            pub_rep = {"confusion_matrix_pct": pm["confusion_matrix_pct"]}
            pub_rep["confusion_matrix_pct"] = [[0.0 if v is None else v for v in row] for row in
                                               pub_rep["confusion_matrix_pct"]]

            def loader(paths):
                return load_embedding_cache(paths, None, 4, cache_file=osp.join(tmp, "cache", "emb.pt"),
                                            reader=lambda p: torch.from_numpy(embs[p]), log=lambda *a: None)
            iod = IO(osp.join(tmp, "VOL", "behaviour_resplit", "st"), osp.join(tmp, "LOCAL"))
            kw = dict(df_meta=dfs, split_clips=sc, arms=["bridge", "leaky", "clip"], seeds=[0, 1], epochs=4,
                      device="cpu", emb_loader=loader, published_sd=torch.load(rl["ckpt"], map_location="cpu"),
                      published_report=pub_rep, expect_published=False, cfg_base={"selftest": True}, smoke=True,
                      log=lambda *a: None)
            out1 = run_resplit(iod, **kw)
            S = out1.get("summary") or {}
            check("flow: every unit ran and SUMMARY exists", set(out1["units"].values()) == {"ran"}
                  and iod.exists("SUMMARY.json") and iod.exists("SUMMARY.md") and len(out1["units"]) == 5)
            check("flow: bridge PASS with delta 0 on its own published model",
                  S.get("bridge", {}).get("status") == "PASS" and S["bridge"]["delta_correct"] == 0)
            check("flow: leaky seed 0 (4 epochs) == the 'published' model's predictions",
                  iod.read_json("leaky/seed_0/metrics.json")["test"]["correct"] == pm["correct"])
            check("flow: summary aggregates both seeds", S["arms"]["clip"]["accuracy"]["n"] == 2
                  and S["arms"]["leaky"]["accuracy"]["n"] == 2)
            check("flow: embeddings cached locally", osp.isfile(osp.join(tmp, "cache", "emb.pt")))
            out2 = run_resplit(iod, **kw)
            check("flow: re-run resumes every unit and republishes identical files",
                  set(out2["units"].values()) == {"resumed"})
            os.remove(iod.vol(".done_clip_seed1"))
            out3 = run_resplit(iod, **kw)
            check("flow: a unit without its marker is re-run bit-identically (publish says identical)",
                  out3["units"]["clip_seed1"] == "ran")
            kw2 = dict(kw, epochs=5)
            check("flow: a changed config is refused for the same tag", raises(lambda: run_resplit(iod, **kw2),
                                                                               RuntimeError))
            iod2 = IO(osp.join(tmp, "VOL2"), osp.join(tmp, "LOCAL2"))
            check("flow: the published census gate fires on non-Edinburgh data",
                  raises(lambda: run_resplit(iod2, **dict(kw, expect_published=True)), AssertionError))
            md = iod.read_json("SUMMARY.json")
            check("flow: summary deltas present", "delta_clip_vs_published" in md and "delta_clip_vs_leaky_retrain" in md)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("SKIP torch selftests (torch not importable here; they run on the cluster before any real work)")
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"{len(results) - n_fail}/{len(results)} selftests passed" + ("" if TORCH_OK else " (torch parts skipped)"))
    return n_fail == 0


if not IS_NOTEBOOK:
    ok = _selftest() if SELFTEST else True
    if not SELFTEST:
        print("not a notebook and TOL69_SELFTEST != 1: nothing to run")
    raise SystemExit(0 if ok else 1)

# COMMAND ----------

# ---- widgets, registry, selftest, inputs, run ------------------------------------------------------------------------
def P(name, default):
    try:
        dbutils.widgets.text(name, default)  # noqa: F821
    except Exception:  # noqa: BLE001
        pass
    v = dbutils.widgets.get(name)  # noqa: F821
    return default if v is None or v == "" else v


W = {"tag": P("tag", "v1"),
     "arms": P("arms", ",".join(ARMS_ALL)),
     "seeds": P("seeds", ",".join(str(s) for s in SEEDS_DEFAULT)),
     "epochs": P("epochs", str(NUM_EPOCHS)),
     "device": P("device", "cpu"),
     "io_threads": P("io_threads", "32")}
TAG = W["tag"].strip()
if not TAG_RX.match(TAG):
    raise ValueError(f"tag {TAG!r}")
ARMS = [a for a in ARMS_ALL if a in parse_csv(W["arms"], set(ARMS_ALL), "arms")]
SEEDS = parse_ints(W["seeds"], "seeds")
EPOCHS = int(W["epochs"])
if not 1 <= EPOCHS <= NUM_EPOCHS:
    raise ValueError(f"epochs must be in 1..{NUM_EPOCHS}")
SMOKE = EPOCHS != NUM_EPOCHS or tuple(SEEDS) != SEEDS_DEFAULT or tuple(ARMS) != ARMS_ALL
if SMOKE and not TAG.startswith("smoke"):
    raise ValueError("a reduced run (epochs < 60, other seeds or a subset of arms) needs a tag starting with 'smoke'")
if TAG.startswith("smoke") and not SMOKE:
    print("note: smoke tag with the full configuration (labelled SMOKE anyway)")
SMOKE = SMOKE or TAG.startswith("smoke")
DEVICE = W["device"].strip()
if DEVICE not in ("cpu", "cuda"):
    raise ValueError("device must be cpu or cuda")
_require_torch()
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("device=cuda but no GPU")
torch.use_deterministic_algorithms(True)
RUN = {"nb_version": NB_VERSION, "tag": TAG, "widgets": W, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                        time.gmtime()),
       "errors": []}

# selftest first (synthetic data, local temp dirs only): stop before touching real data if anything fails
_st_buf = io.StringIO()
_stdout = sys.stdout
sys.stdout = _st_buf
try:
    _st_ok = _selftest()
finally:
    sys.stdout = _stdout
print(_st_buf.getvalue())
RUN["selftest"] = {"ok": _st_ok, "lines": _st_buf.getvalue().splitlines()[-3:]}
if not _st_ok:
    raise RuntimeError("Tol_69 selftest failed on the cluster; nothing was read or published")

R = tol_paths.load(version=REGISTRY_VERSION)
OUT = R.out("paper1.root", "behaviour_resplit", TAG)
LOCAL = "/local_disk0/tol69/" + TAG
os.makedirs(LOCAL, exist_ok=True)
_vol_tp = R.path("admin.registry", None, "tol_paths.py")
if core.sha256_file(_vol_tp) != core.sha256_file(tol_paths.__file__):
    raise RuntimeError("the deployed tol_paths.py differs from the registry's copy: re-deploy it beside this notebook")


def _notebook_source():
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
    errs = []
    for cand in ("/Workspace" + nbp, "/Workspace" + nbp + ".py"):
        try:
            if os.path.isfile(cand):
                with open(cand, "rb") as f:
                    b = f.read()
                if b.lstrip(b"\xef\xbb\xbf").startswith(b"# Databricks notebook source"):
                    return b
                errs.append(f"{cand}: not SOURCE format")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{cand}: {e!r}")
    try:
        import base64
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.workspace import ExportFormat
        return base64.b64decode(WorkspaceClient().workspace.export(nbp, format=ExportFormat.SOURCE).content)
    except Exception as e:  # noqa: BLE001
        errs.append(f"sdk export: {e!r}")
    raise RuntimeError(f"{NB_NAME} cannot fingerprint its own source: " + "; ".join(errs))


# inputs (catalog only): metadata CSV + published checkpoint staged to local disk and hashed; report JSON read once
META_SRC = R.path("datasets.edinburgh_pigs.behaviour_metadata", None, META_FILE)
META_LOCAL, META_SHA = core.stage_file(META_SRC, LOCAL + "/inputs/" + META_FILE, dbutils=dbutils)  # noqa: F821
EMB_ROOT = R.path("datasets.edinburgh_pigs.embeddings")
CKPT_SRC = R.path("models.behaviour_bilstm")
REPORT_SRC = R.path("paper1.root", None, *PUB_REPORT_REL)
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
DF_META = pd.read_csv(META_LOCAL)
PUB_SD, PUB_REPORT, CKPT_SHA, REPORT_SHA = None, None, None, None
if "bridge" in ARMS:
    CKPT_LOCAL, CKPT_SHA = core.stage_file(CKPT_SRC, LOCAL + "/inputs/best_bilstm_dinov3_vits.pt",
                                           dbutils=dbutils)  # noqa: F821
    PUB_SD = torch.load(CKPT_LOCAL, map_location="cpu")
    with open(REPORT_SRC, "rb") as _f:
        _rb = _f.read()
    REPORT_SHA = sha256_bytes(_rb)
    PUB_REPORT = json.loads(_rb.decode("utf-8"))
    check_published_report(PUB_REPORT)

VERSIONS = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
CFG_BASE = {
    "nb_name": NB_NAME, "tag": TAG,
    "code": {"notebook": core.source_shas(_notebook_source()), "tol_bench_core": core.code_sha256(),
             "tol_paths": core.source_shas(tol_paths.__file__), "bench_version": core.BENCH_VERSION},
    "inputs": {"metadata": {"path": META_SRC, "sha256": META_SHA},
               "embeddings_root": EMB_ROOT,
               "published_checkpoint": {"path": CKPT_SRC, "sha256": CKPT_SHA},
               "published_report": {"path": REPORT_SRC, "sha256": REPORT_SHA}},
    "registry": R.provenance(REGISTRY_KEYS),
    "versions": VERSIONS, "databricks_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
    "torch_threads": torch.get_num_threads(), "deterministic_algorithms": True,
    "hardware": core.gpu_info() if DEVICE == "cuda" else {"class": "CPU", "cpus": os.cpu_count()},
}
RUN["config_base"] = CFG_BASE
print(json.dumps({k: CFG_BASE[k] for k in ("inputs", "versions", "databricks_runtime", "torch_threads")}, indent=1))
IO_ = IO(OUT, LOCAL, dbutils)  # noqa: F821
EMB_CACHE_FILE = f"/local_disk0/tol69/_cache/emb_{META_SHA[:16]}.pt"     # shared by the tags of one cluster


def _emb_loader(paths):
    return load_embedding_cache(paths, EMB_ROOT, int(W["io_threads"]), cache_file=EMB_CACHE_FILE)


try:
    REC = run_resplit(IO_, df_meta=DF_META, split_clips=SPLIT_CLIPS, arms=ARMS, seeds=SEEDS, epochs=EPOCHS,
                      device=DEVICE, emb_loader=_emb_loader, published_sd=PUB_SD, published_report=PUB_REPORT,
                      expect_published=True, cfg_base=CFG_BASE, smoke=SMOKE)
except Exception as e:  # noqa: BLE001
    RUN["errors"].append(f"{type(e).__name__}: {e}")
    RUN["traceback"] = traceback.format_exc()
    REC = {}
    print(RUN["traceback"])
RUN["record"] = {k: v for k, v in REC.items() if k not in ("summary_md",)}
RUN["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
IO_.publish_json(f"runs/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json", jsonable(RUN))
if REC.get("summary_md"):
    print(REC["summary_md"])
log(f"outputs: {OUT}")
if RUN["errors"]:
    raise RuntimeError("Tol_69 finished with errors (completed units are published): " + " | ".join(RUN["errors"]))

# COMMAND ----------

_S = (REC.get("summary") or {})
dbutils.notebook.exit(json.dumps({"tag": TAG, "status": _S.get("status"), "bridge": _S.get("bridge"),  # noqa: F821
                                  "clip": {k: (_S.get("arms", {}).get("clip", {}).get(k) or {}).get("mean")
                                           for k in ("accuracy", "macro_f1", "weighted_f1")},
                                  "leaky": {k: (_S.get("arms", {}).get("leaky", {}).get(k) or {}).get("mean")
                                            for k in ("accuracy", "macro_f1", "weighted_f1")},
                                  "out": OUT}, default=str)[:60000])
