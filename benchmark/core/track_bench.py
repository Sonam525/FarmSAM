# Databricks notebook source
# MAGIC %md
# MAGIC # Tracking benchmark
# MAGIC
# MAGIC Runs one model over one dataset split and writes one run directory:
# MAGIC
# MAGIC ```
# MAGIC paper1_distillation/bench/<dataset>/<arm>__prot<A|D>/<split>[/<variant>]/
# MAGIC     BENCH_CONFIG.json      written once; a later job into this folder must match it
# MAGIC     mot/<seq>.txt          MOT17 tracker file for TrackEval, 0-based pixel boxes
# MAGIC     rle/<seq>.jsonl.gz     per-frame masks (always for calves; for pigs with rle=true)
# MAGIC     telemetry/<seq>.json   timings, memory and frame hashes
# MAGIC     .done_<seq>            written last
# MAGIC paper1_distillation/bench/<dataset>/_detections/<detector>/<split>/<seq>.json   detector cache, shared by all models
# MAGIC ```
# MAGIC
# MAGIC `protocol=A` seeds tracks from reference boxes at the first frame. `protocol=D` runs an RF-DETR detector in the loop.
# MAGIC
# MAGIC | widget | values | meaning |
# MAGIC |---|---|---|
# MAGIC | `dataset` | `edinburgh`, `pigtrack`, `calf` | `calf` = the calf windows |
# MAGIC | `split` | edinburgh `test`, `val`, `all`; pigtrack `test`, `val`; calf `windows_v1`, `windows_v2` | empty = `test` (pigs) / `windows_v1` |
# MAGIC | `seqs` | comma list | empty = every sequence of the split |
# MAGIC | `arm` | `T` `S-Edi` `S-Edi-mined` `S-PT` `S-PT-mined` `S-v2` `T21` `S21` `ET` `ES-L` | model key: see benchmark/README.md |
# MAGIC | `protocol` | `A`, `D` | see above |
# MAGIC | `detector_ckpt` | `<models.* registry key>[:<file inside it>]` | `protocol=D` only; empty = the pig detector of the dataset |
# MAGIC | `detector_resolution` | int | `protocol=D` only, required |
# MAGIC | `tau` | float | `protocol=D`: on `val`, required; on `test`, read from `protocol/detector_threshold.json` |
# MAGIC | `keep_last_n` / `pop` | `16` / `true` | anything else goes to a variant folder such as `k8` or `none` |
# MAGIC | `out_root` | empty or the run folder | a guard, not a redirect |
# MAGIC | `smoke` | int (max frames per sequence) | > 0 writes to variant `smoke<N>` |
# MAGIC | `variant` | tag | extra folder level; `explore_*` admits other model and dataset pairs |
# MAGIC | `rle` | `auto`, `true`, `false` | auto = calves only |
# MAGIC | `revisions` | empty, JSON `{hf_id: commit}`, or `protocol:hf_revisions.json` | Hugging Face commit pins |
# MAGIC | `require_gpu` | `A10`, `A100`, `any` | |
# MAGIC | `progress_every` | int | progress log cadence in frames |
# MAGIC
# MAGIC A sequence with `.done_<seq>` is skipped, so a job can resume after a crash.
# MAGIC The `calf` dataset reads a window file (`protocol/calf_windows_v1.json`, `calf_windows_v2.json`) with the clip, start
# MAGIC frame, frame count and seed masks of each window. That file is not in this repository.

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
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types

# allocator config BEFORE anything touches CUDA (the job cluster sets it too; asserted before the first CUDA call)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

NB_NAME = "track_bench"
NB_VERSION = "track_bench-2026-10-20.1"     # + calf split windows_v2; nothing else changed
# the %pip cell above, verbatim
PIP_SPEC = ('"numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm" "scipy" '
            '"rfdetr==1.10.1" "pycocotools" "peft"; opencv-python-headless==4.11.0.86 (--no-deps)')
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None


def _this_dir():
    """Where bench_core.py / bench_core.py / farm_paths.py live: env FARMSAM_CODE_DIR, else this notebook's own
    workspace directory (job / notebook mode), else the directory of this file (plain python)."""
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
    """Import module `name` from the FIRST of `dirs` that holds `<name>.py` and assert that this file is what runs
    (nothing else on sys.path may shadow it: its sha256 goes into BENCH_CONFIG)."""
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
os.environ.setdefault("FARMSAM_CODE_DIR", CODE_DIR)      # bench_core finds bench_core (calf decoder contract) there
bc = _import_exact("bench_core", [CODE_DIR])
_PATHS_DIRS = [CODE_DIR, os.environ.get("FARMSAM_PATHS_DIR")]
farm_paths = _import_exact("farm_paths", _PATHS_DIRS)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print(f"{NB_NAME} {NB_VERSION}: bench_core {bc.BENCH_VERSION} ({bc.__file__}, sha256 {_sha256(bc.__file__)[:16]}), "
      f"farm_paths {farm_paths.__file__} (sha256 {_sha256(farm_paths.__file__)[:16]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters, matrix, variants, sequences (pure Python)

# COMMAND ----------

WIDGETS = collections.OrderedDict([
    ("dataset", ""), ("split", ""), ("seqs", ""), ("arm", ""), ("protocol", "A"),
    ("detector_ckpt", ""), ("detector_resolution", ""), ("tau", ""),
    ("keep_last_n", str(bc.KEEP_LAST_N)), ("pop", "true" if bc.POP_FRAMES else "false"),
    ("out_root", ""), ("smoke", "0"), ("variant", ""), ("rle", "auto"), ("revisions", ""),
    ("require_gpu", "A10"), ("progress_every", "500"),
])
CORE_DATASET = {"edinburgh": "edinburgh", "pigtrack": "pigtrack", "calf": "calves_tol"}
SPLITS = {"edinburgh": ("test", "val", "all"), "pigtrack": ("test", "val"), "calf": ("windows_v1", "windows_v2")}
DEFAULT_SPLIT = {"edinburgh": "test", "pigtrack": "test", "calf": "windows_v1"}
# calf split -> (windows-file version, file name). v1 is the registry key paper1.calf_windows_protocol; v2 is
# paper1.root/protocol/calf_windows_v2.json (bench_core.protocol_file_path), written by the calf window selection mode snap_v2.
CALF_WINDOWS_FILES = {"windows_v1": ("calf_windows_v1", "calf_windows_v1.json"),
                      "windows_v2": ("calf_windows_v2", "calf_windows_v2.json")}
# Which models run on which dataset, protocol and split. Other pairs need variant=explore_*.
_NONSTUDENT = {"T", "T21", "S21", "ET", "ES-L"}
MATRIX = {
    ("edinburgh", "A"): {"test": _NONSTUDENT | {"S-Edi", "S-Edi-mined"}, "val": {"T"}, "all": set(_NONSTUDENT)},
    ("edinburgh", "D"): {"test": _NONSTUDENT | {"S-Edi"}},
    ("pigtrack", "A"): {"test": _NONSTUDENT | {"S-PT", "S-PT-mined"}, "val": {"T"}},
    ("pigtrack", "D"): {"test": _NONSTUDENT | {"S-PT"}, "val": {"T"}},
    ("calf", "A"): {"windows_v1": _NONSTUDENT | {"S-v2"},
                    "windows_v2": _NONSTUDENT | {"S-v2"}},
}
DEFAULT_DETECTOR = {"pigtrack": "models.detector_pig_pt:checkpoint_best_total.pth",
                    "edinburgh": "models.detector_pig_edi:checkpoint_best_total.pth"}
TAU_FILE = "detector_threshold.json"
REGISTRY_VERSION = 1          # env CALF_PATHS points at your copy of benchmark/paths_example.yaml
TAG_RX = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")          # one path segment: never "." / ".."
# user variants may not impersonate a derived variant
RESERVED_VARIANT_RX = re.compile(r"^(oldprot.*|tau_.*|det_.*|smoke.*|none(pop)?|k\d+(pop)?)$", re.I)
SEQ_RX = {"edinburgh": re.compile(r"^(\d{4}_\d{2}_\d{2})_(\d{6})$"), "pigtrack": re.compile(r"^pigtrack\d{4}$"),
          "calf": re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")}
CALF_CLIP_RX = re.compile(r"^Tol_ch01[-_](\d{8})[-_](\d{6})[-_](\d{6})\.mp4$")   # ch01 clips, no " (1)" duplicates
CV_CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_HEIGHT = 3, 4                          # OpenCV enum values (stable in 4.x)


def _bool(v, name):
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={v!r}: expected true|false")


def _int(v, name, lo=None):
    try:
        x = int(str(v).strip())
    except ValueError:
        raise ValueError(f"{name}={v!r}: expected an integer") from None
    if lo is not None and x < lo:
        raise ValueError(f"{name}={x} < {lo}")
    return x


def parse_params(raw):
    """Widget strings -> typed, validated parameters. Unknown names and malformed values RAISE (never a silent default)."""
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}; known: {list(WIDGETS)}")
    g = {k: str(raw.get(k, d) if raw.get(k) is not None else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace()
    ds = g["dataset"].lower()
    ds = "calf" if ds in ("calf", "calves", "calves_tol") else ds
    if ds not in CORE_DATASET:
        raise ValueError(f"dataset={g['dataset']!r}: expected edinburgh | pigtrack | calf")
    P.dataset, P.core_dataset = ds, CORE_DATASET[ds]
    P.split = g["split"] or DEFAULT_SPLIT[ds]
    if P.split not in SPLITS[ds]:
        raise ValueError(f"split={P.split!r} for {ds}: expected one of {SPLITS[ds]}")
    P.arm = g["arm"]
    if P.arm not in bc.ARMS:
        raise ValueError(f"arm={P.arm!r}: expected one of {sorted(bc.ARMS)}")
    P.protocol = g["protocol"].upper()
    if P.protocol not in bc.PROTOCOLS:
        raise ValueError(f"protocol={g['protocol']!r}: expected one of {bc.PROTOCOLS}")
    P.seqs = [s for s in re.split(r"[,\s]+", g["seqs"]) if s]
    if len(set(P.seqs)) != len(P.seqs):
        raise ValueError(f"seqs lists a sequence twice: {P.seqs}")
    bad = [s for s in P.seqs if not SEQ_RX[ds].match(s)]
    if bad:
        raise ValueError(f"seqs {bad} are not {ds} sequence names")
    k = g["keep_last_n"].lower()
    P.keep_last_n = None if k in ("none", "off") else _int(k, "keep_last_n", 1)
    P.pop = _bool(g["pop"], "pop")
    P.smoke = _int(g["smoke"] or "0", "smoke", 0)
    r = g["rle"].lower()
    if r not in ("auto", "true", "false"):
        raise ValueError(f"rle={g['rle']!r}: expected auto | true | false")
    P.rle = (ds == "calf") if r == "auto" else (r == "true")
    if ds == "calf" and not P.rle:
        raise ValueError("calf runs always write masks")
    if g["tau"]:
        try:
            t = float(g["tau"])
        except ValueError:
            raise ValueError(f"tau={g['tau']!r}: expected a number") from None
        if not (0.0 < t < 1.0) or abs(round(t, 2) - t) > 1e-9:
            raise ValueError(f"tau={t}: expected a value in (0, 1) with at most 2 decimals")
        P.tau = round(t, 2)
    else:
        P.tau = None
    P.detector_ckpt = g["detector_ckpt"]
    P.detector_resolution = _int(g["detector_resolution"], "detector_resolution", 1) if g["detector_resolution"] else None
    if P.protocol == "A" and (P.detector_ckpt or P.detector_resolution or P.tau is not None):
        raise ValueError("detector_ckpt / detector_resolution / tau are detector run parameters (protocol=A runs no detector)")
    P.out_root = g["out_root"]
    P.user_variant = g["variant"]
    if P.user_variant and not TAG_RX.match(P.user_variant):
        raise ValueError(f"variant={P.user_variant!r}: letters, digits, '_', '.', '-' only")
    if P.user_variant and (RESERVED_VARIANT_RX.match(P.user_variant) or "__" in P.user_variant):
        raise ValueError(f"variant={P.user_variant!r} is reserved: 'oldprot' names the J3(c) OLD-protocol runs (the fidelity audit "
                         "writes them under evidence/edinburgh_audit/; this notebook runs only the new protocol), and "
                         "tau_* / det_* / smoke* / none / k<N>[pop] / '__' are derived by this notebook itself")
    P.revisions_spec = g["revisions"]
    P.require_gpu = g["require_gpu"] or "A10"
    if P.require_gpu not in ("A10", "A100", "any"):
        raise ValueError(f"require_gpu={P.require_gpu!r}: expected A10 | A100 | any")
    P.progress_every = _int(g["progress_every"] or "0", "progress_every", 0)
    return P


def check_matrix(P, log=print):
    """an arm runs on a dataset / split / protocol only where the protocol lists it. Anything else
    needs variant=explore_* (it then lands in its own directory and never in a paper table)."""
    if P.dataset == "calf" and P.protocol == "D":
        raise ValueError("calf windows under detector run are efficiency-only (segv1d + S-v2, one window): that is efficiency_bench "
                         "(job J13), not this notebook")
    allowed = MATRIX.get((P.dataset, P.protocol), {}).get(P.split, set())
    if P.arm in allowed:
        return "in_matrix"
    if P.user_variant.startswith("explore_"):
        log(f"WARNING: {P.arm} on {P.dataset}/{P.split} protocol {P.protocol} is OFF the matrix; "
            f"exploratory run in variant {P.user_variant!r} (never a paper number)")
        return "explore"
    raise ValueError(f"{P.arm} is not run on {P.dataset}/{P.split} under protocol {P.protocol} ("
                     f"allows {sorted(allowed) or 'nothing'}); use variant=explore_<tag> for an exploratory run")


def detector_spec(R, spec):
    """'<models.* registry key>[:<file inside it>]' -> source path, tag (= the checkpoint folder name), registry pin."""
    key, _, rel = spec.partition(":")
    key, rel = key.strip(), rel.strip().strip("/")
    if not re.match(r"^models\.[A-Za-z0-9_.]+$", key):
        raise ValueError(f"detector_ckpt {spec!r}: expected '<models.* registry key>[:<file inside it>]' (registry only)")
    parts = [p for p in rel.split("/") if p]
    if any(p in ("..", ".") for p in parts):
        raise ValueError(f"detector_ckpt {spec!r}: '..' not allowed")
    base = R.path(key)
    src = R.path(key, None, *parts) if parts else base
    tag = base.rstrip("/").rsplit("/", 1)[-1] if parts else base.rstrip("/").rsplit("/", 2)[-2]
    if not TAG_RX.match(tag):
        raise ValueError(f"detector tag {tag!r} derived from {spec!r} is not a valid directory name")
    return {"spec": spec, "registry_key": key, "file": rel or None, "src": src, "tag": tag,
            "sha256_pin": None if parts else R.entry(key).get("sha256")}


def resolve_detector(R, P):
    if P.protocol != "D":
        return None
    det = detector_spec(R, P.detector_ckpt or DEFAULT_DETECTOR[P.dataset])
    det["default_tag"] = detector_spec(R, DEFAULT_DETECTOR[P.dataset])["tag"]
    return det


def detector_resolution(det, widget):
    """RF-DETR's input resolution is PINNED at load (strip_checkpoint drops model_config, so an unpinned load silently
    uses the class default). Source: the train_summary.json beside the checkpoint (train_rfdetr_pig contract: keys
    `resolution`, `checkpoint_best_total_sha256`) and/or the widget; when both exist they must agree. A wrong value would
    otherwise publish a wrong SHARED detection cache that no later run could replace. -> (resolution, source)."""
    ts_path = det["src"].rsplit("/", 1)[0] + "/train_summary.json"
    ts = None
    if os.path.isfile(ts_path):
        with open(ts_path, encoding="utf-8") as f:
            ts = json.load(f)
        pinned = ts.get("checkpoint_best_total_sha256")
        if det["src"].rsplit("/", 1)[-1] == "checkpoint_best_total.pth" and pinned and pinned != det["sha256"]:
            raise RuntimeError(f"train_summary.json beside {det['tag']} names checkpoint {pinned[:16]}, the staged file "
                               f"is {det['sha256'][:16]}")
    res_ts = None if ts is None or ts.get("resolution") is None else int(ts["resolution"])
    if res_ts is not None and widget is not None and res_ts != int(widget):
        raise ValueError(f"detector_resolution={widget} but {det['tag']}/train_summary.json says {res_ts}")
    if widget is None and res_ts is None:
        raise ValueError(f"detector_resolution is required for detector run: no train_summary.json with a resolution "
                         f"beside {det['tag']} (RF-DETR's resolution is pinned at load)")
    src = "widget+train_summary" if (widget is not None and res_ts is not None) else (
        "widget" if widget is not None else "train_summary")
    return int(widget if widget is not None else res_ts), src


def derive_variant(P, det=None):
    """Run-dir variant: user tag, tau_<τ> (D val grid), det_<tag> (non-default detector,
    e.g. the d_pt_v1 published-weights sensitivity row on Edinburgh), the pruning factorial (k8 / none / k16 / ...), smoke<N>."""
    parts = []
    if P.user_variant:
        parts.append(P.user_variant)
    if P.protocol == "D":
        if P.split == "val":
            if P.tau is None:
                raise ValueError("detector run on val is the tau grid: give tau (one job per value of D_TAU_GRID "
                                 f"{bc.D_TAU_GRID} plus D-PT's val-F1 operating point)")
            parts.append(f"tau_{P.tau:.2f}")
        if det is not None and det["tag"] != det["default_tag"]:
            parts.append(f"det_{det['tag']}")
    if (P.keep_last_n, P.pop) != (bc.KEEP_LAST_N, bc.POP_FRAMES):
        parts.append(("none" if P.keep_last_n is None else f"k{P.keep_last_n}") + ("pop" if P.pop else ""))
    if P.smoke:
        parts.append(f"smoke{P.smoke}")
    v = "__".join(parts)
    if v and not TAG_RX.match(v):
        raise ValueError(f"variant {v!r} is not a valid directory name")
    return v or None


def check_out_root(P, R, run_dir):
    """out_root is a guard, never a redirect: empty / 'auto', or a path naming the derived run dir (or its split or
    arm directory), absolute or as a 'paper1_distillation/bench/...' suffix."""
    if not P.out_root or P.out_root.lower() == "auto":
        return run_dir
    want = P.out_root.replace("\\", "/").rstrip("/")
    split_dir = bc.bench_run_dir(R, P.core_dataset, P.arm, P.protocol, P.split)
    arm_dir = split_dir.rsplit("/", 1)[0]
    for c in (run_dir, split_dir, arm_dir):
        if c == want or c.endswith("/" + want.lstrip("/")):
            return run_dir
    raise ValueError(f"out_root={P.out_root!r} does not name this run's directory {run_dir} (out_root is a guard: "
                     "paths come only from the registry)")


def _select(names, wanted, what):
    if not wanted:
        return list(names)
    missing = [s for s in wanted if s not in names]
    if missing:
        raise ValueError(f"{what}: {missing} not in this split ({len(names)} available: {list(names)[:8]}...)")
    return [n for n in names if n in wanted]


def read_seqmap(path):
    names = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s and s.lower() != "name":
                names.append(s)
    return names


def parse_calf_windows(obj, pins, version="calf_windows_v1"):
    """Strict reader of calf_windows_v1.json / calf_windows_v2.json (the contract in the header cell). Returns
    OrderedDict name -> window (v2: + run_split)."""
    if not isinstance(obj, dict) or obj.get("version") != version:
        raise ValueError(f"calf windows file: version {obj.get('version') if isinstance(obj, dict) else None!r} "
                         f"!= {version!r}")
    fps = float(obj.get("track_fps", bc.CALF_TRACK_FPS))
    if fps != bc.CALF_TRACK_FPS:
        raise ValueError(f"calf windows track_fps {fps} != {bc.CALF_TRACK_FPS}")
    if not pins:
        raise ValueError("registry has no paper1.calf_windows_gt_pins")
    ws = obj.get("windows")
    if isinstance(ws, list):
        ws = collections.OrderedDict((w.get("name"), w) for w in ws)
    if not isinstance(ws, dict) or not ws:
        raise ValueError("calf windows file has no windows")
    out = collections.OrderedDict()
    # the union of what this notebook (clip, seed_masks) and the scorer's calf scorer (stride, fps, animals, gt_file) read
    need = ("day", "phase", "clip", "start_fidx", "n_frames", "stride", "fps", "animals", "gt_version", "gt_file",
            "seed_masks")
    for name, w in ws.items():
        if not isinstance(name, str) or not SEQ_RX["calf"].match(name):
            raise ValueError(f"calf window name {name!r}")
        miss = [k for k in need if k not in w]
        if miss:
            raise ValueError(f"calf window {name}: missing {miss} (the file is read by track_bench AND the scorer)")
        if float(w["fps"]) != bc.CALF_TRACK_FPS or int(w["stride"]) < 1:
            raise ValueError(f"calf window {name}: fps {w['fps']} / stride {w['stride']} (the v5 contract is 5 fps)")
        day = str(w["day"])
        farm_paths.Registry.check_day(day)
        if pins.get(day) != w["gt_version"]:
            raise ValueError(f"calf window {name}: gt_version {w['gt_version']!r} != registry pin {pins.get(day)!r} "
                             f"for {day}")
        m = CALF_CLIP_RX.match(str(w["clip"]))
        if not m or m.group(1) != day:
            raise ValueError(f"calf window {name}: clip {w['clip']!r} is not a Tol_ch01 clip of {day}")
        start, n = int(w["start_fidx"]), int(w["n_frames"])
        if start < 0 or n <= 0:
            raise ValueError(f"calf window {name}: start_fidx {start} / n_frames {n}")
        seeds = w["seed_masks"]
        if not isinstance(seeds, dict) or not seeds:
            raise ValueError(f"calf window {name}: seed_masks must be a non-empty {{gt id: rle}} map")
        for gid, r in seeds.items():
            if not re.match(r"^\d+$", str(gid)) or not isinstance(r, dict) or "size" not in r or "counts" not in r:
                raise ValueError(f"calf window {name}: seed {gid!r} is not a column-major RLE {{size, counts}}")
        animals = sorted(int(a) for a in w["animals"])
        if animals != sorted(int(g) for g in seeds):
            raise ValueError(f"calf window {name}: animals {animals} != seed_masks ids {sorted(int(g) for g in seeds)} "
                             "(seeding contract: session obj_id == GT animal id, scored by the scorer)")
        out[name] = {"name": name, "day": day, "phase": str(w["phase"]), "clip": str(w["clip"]), "start_fidx": start,
                     "n_frames": n, "stride": int(w["stride"]), "fps": float(w["fps"]), "animals": animals,
                     "gt_version": str(w["gt_version"]), "gt_file": str(w["gt_file"]), "seed_masks": seeds,
                     "clip_sha256": w.get("clip_sha256")}
        if version != "calf_windows_v1":
            rs = w.get("run_split")
            if rs not in CALF_WINDOWS_FILES:
                raise ValueError(f"calf window {name}: run_split {rs!r} not in {sorted(CALF_WINDOWS_FILES)} "
                                 "(calf_windows_v2: the run-dir split that holds the window's runs)")
            out[name]["run_split"] = rs
    return out


def resolve_sequences(R, P):
    """-> ([spec], provenance dict). Every source path comes from the registry; requested names must exist."""
    specs = []
    if P.dataset == "edinburgh":
        names = {"test": bc.EDI_TEST, "val": bc.EDI_VAL, "all": bc.EDI_TRAIN + bc.EDI_VAL + bc.EDI_TEST}[P.split]
        for clip in _select(list(names), P.seqs, "edinburgh clips"):
            date, num = SEQ_RX["edinburgh"].match(clip).groups()
            spec = {"seq": clip, "kind": "edinburgh",
                    "frames_src": R.path("datasets.edinburgh_pigs.decoded_frames", None, clip, "video1"),
                    "gt_src": R.path("datasets.edinburgh_pigs.annotated", None, date, num, "output.json"),
                    # catalog layout = the mount's 'Initial Prompts/<clip>/video1/BoundingBoxes_<clip>.txt' (op E03)
                    "prompts_txt_src": R.path("datasets.edinburgh_pigs.initial_prompts", None, clip, "video1",
                                              f"BoundingBoxes_{clip}.txt")}
            if not os.path.isdir(spec["frames_src"]):
                raise FileNotFoundError(f"{clip}: decoded frames are not in the catalog ({spec['frames_src']})")
            if P.protocol == "A" and not os.path.isfile(spec["gt_src"]):
                raise FileNotFoundError(f"{clip}: GT output.json is not in the catalog ({spec['gt_src']})")
            if P.protocol == "A" and not os.path.isfile(spec["prompts_txt_src"]):
                raise FileNotFoundError(f"{clip}: published seed file is not in the catalog ({spec['prompts_txt_src']}); "
                                        "seeded run must prove its GT-derived prompts equal the published seeds")
            specs.append(spec)
        return specs, {"names": "bench_core.EDI_* constants", "n_split": len(names)}
    if P.dataset == "pigtrack":
        seqmap = R.path("datasets.pigtrack.seqmaps", None, f"pigtrack-{P.split}.txt")
        root = R.path("datasets.pigtrack.sequences", None, P.split)
        if os.path.isfile(seqmap):
            names, prov = read_seqmap(seqmap), {"seqmap": f"pigtrack-{P.split}.txt", "seqmap_sha256": _sha256(seqmap)}
        else:
            names = sorted(n for n in os.listdir(root) if SEQ_RX["pigtrack"].match(n))
            prov = {"seqmap": None, "listing": True}
        if P.split == "test" and sorted(names) != sorted(bc.PIGTRACK_TEST):
            raise ValueError(f"PigTrack test split has {len(names)} sequences that differ from PIGTRACK_TEST "
                             f": {sorted(set(names) ^ set(bc.PIGTRACK_TEST))[:6]}")
        for s in _select(names, P.seqs, "pigtrack sequences"):
            src = R.path("datasets.pigtrack.sequences", None, P.split, s)
            if not os.path.isfile(src + "/seqinfo.ini"):
                raise FileNotFoundError(f"{s}: not staged in the catalog ({src})")
            specs.append({"seq": s, "kind": "pigtrack", "seq_src": src})
        prov["n_split"] = len(names)
        return specs, prov
    version, fname = CALF_WINDOWS_FILES[P.split]
    wpath = (R.path("paper1.calf_windows_protocol") if P.split == "windows_v1"
             else bc.protocol_file_path(R, fname))
    if not os.path.isfile(wpath):
        raise FileNotFoundError(f"{wpath} does not exist yet: the calf window selection must snap and publish the windows "
                                f"({'mode snap' if P.split == 'windows_v1' else 'mode snap_v2'}) before any calf run "
                                "")
    with open(wpath, "rb") as f:
        raw = f.read()
    pins = (R.data.get("paper1") or {}).get("calf_windows_gt_pins") or {}
    windows = parse_calf_windows(json.loads(raw.decode("utf-8")), {str(k): str(v) for k, v in pins.items()}, version)
    names = list(windows)
    if P.split != "windows_v1":                  # v2: only the windows whose runs live in THIS split directory
        names = [n for n, w in windows.items() if w["run_split"] == P.split]
        elsewhere = [s for s in P.seqs if s in windows and windows[s]["run_split"] != P.split]
        if elsewhere:
            raise ValueError(f"calf windows {elsewhere}: run_split {[windows[s]['run_split'] for s in elsewhere]} in "
                             f"{fname} (carried over; their runs are in that split directory, not {P.split})")
        if not names:
            raise ValueError(f"{fname} has no window with run_split {P.split!r}")
    for name in _select(names, P.seqs, "calf windows"):
        w = windows[name]
        src = R.path("datasets.calves.videos", None, w["clip"])
        if not os.path.isfile(src):
            raise FileNotFoundError(f"window {name}: clip {w['clip']} not found under datasets.calves.videos")
        specs.append({"seq": name, "kind": "calf", "window": w, "clip_src": src})
    return specs, {"calf_windows_file": fname, "calf_windows_sha256": hashlib.sha256(raw).hexdigest(),
                   "n_windows": len(windows), "n_windows_in_split": len(names),
                   "gt_pins": {str(k): str(v) for k, v in pins.items()}}


def resolve_revisions(R, P):
    """HF commit pins: '' | JSON | protocol:<file>.json."""
    s = P.revisions_spec
    if not s:
        return {}
    if s.startswith("protocol:"):
        p = bc.protocol_file_path(R, s[len("protocol:"):])
        with open(p, encoding="utf-8") as f:
            obj = json.load(f)
        obj = obj.get("revisions", obj)
    else:
        obj = json.loads(s)
    if not isinstance(obj, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items()):
        raise ValueError("revisions must map hf_id -> commit hash")
    bad = {k: v for k, v in obj.items() if not re.match(r"^[0-9a-f]{7,40}$", v)}
    if bad:
        raise ValueError(f"revisions: not commit hashes: {bad}")
    return dict(sorted(obj.items()))


def load_tau(R, P):
    """D only. val: the widget value (the grid). test: the value frozen ONCE in protocol/detector_threshold.json; a widget value must equal it. A smoke run may use the widget before the file exists."""
    if P.protocol != "D":
        return None, None
    if P.split == "val":
        return P.tau, {"kind": "val_grid", "in_D_TAU_GRID": P.tau in bc.D_TAU_GRID}
    path = bc.protocol_file_path(R, TAU_FILE)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            raw = f.read()
        obj = json.loads(raw.decode("utf-8"))
        if "tau" not in obj:
            raise ValueError(f"{TAU_FILE} has no 'tau' key: {sorted(obj)}")
        tau = round(float(obj["tau"]), 2)
        if P.tau is not None and P.tau != tau:
            raise ValueError(f"tau widget {P.tau} != frozen tau {tau} ({TAU_FILE}); the frozen value is never changed")
        return tau, {"kind": "frozen", "file": TAU_FILE, "sha256": hashlib.sha256(raw).hexdigest()}
    if P.smoke and P.tau is not None:
        return P.tau, {"kind": "smoke_widget", "note": f"{TAU_FILE} not frozen yet"}
    raise RuntimeError(f"{TAU_FILE} not found: copy it from benchmark/config/ before any protocol=D test job")


def protd_config_params(tau):
    """The fps-independent detector run constants for BENCH_CONFIG (period / death frames depend on each sequence's fps
    and are recorded per sequence in its telemetry)."""
    p = dict(bc.ProtocolD(fps=1.0, tau=tau).params)
    for k in ("fps", "period_frames", "death_frames"):
        p.pop(k, None)
    p.update(period_rule="max(1, round(fps * pass_period_s))", death_rule="round(death_s * fps)",
             detection_cache_threshold=bc.DET_PROBE_THRESHOLD)
    return p


_FLAG_RX = re.compile(r"(non_overlap|multimask|fill_hole|max_cond_frame|num_maskmem|max_object_pointers|binarize|"
                      r"mask_threshold|sigmoid_scale|sigmoid_bias|image_size)", re.I)


def arm_flags(info):
    """The post-processing / memory defaults that differ between arms (recorded,
    never overridden) + the attention backend, pulled out of arm.info for the arm table."""
    mc = {k: v for k, v in (info.get("model_config") or {}).items() if _FLAG_RX.search(k)}
    pc = {k: v for k, v in (info.get("processor") or {}).items() if _FLAG_RX.search(k)}
    return {"model_config": mc, "processor": pc, "attn_request": info.get("attn_request"),
            "attn_resolved": info.get("attn_resolved"), "dtype": info.get("dtype"),
            "torch_sdp_backends": _sdp_backends()}


def _sdp_backends():
    if not bc.TORCH_OK:
        return None
    try:
        import torch
        b = torch.backends.cuda
        out = {"flash": bool(b.flash_sdp_enabled()), "mem_efficient": bool(b.mem_efficient_sdp_enabled()),
               "math": bool(b.math_sdp_enabled())}
        if hasattr(b, "cudnn_sdp_enabled"):
            out["cudnn"] = bool(b.cudnn_sdp_enabled())
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


VOLATILE_ARM_KEYS = ("weights_mem_mib",)                         # measured at load: may differ by bytes between loads
VOLATILE_STUDENT_KEYS = ("gn_train_eval_max_diff_fp32",)        # GPU float, not bit-stable across processes


def split_arm_info(info):
    """arm.info -> (checked part for BENCH_CONFIG, volatile measurements for BENCH_CONFIG.extra). A resume must not be
    refused because a load-time measurement moved in its last bit; the measurement is still recorded."""
    checked = json.loads(json.dumps(info, default=str))
    vol = {}
    for k in VOLATILE_ARM_KEYS:
        if k in checked:
            vol[k] = checked.pop(k)
    st = checked.get("student")
    if isinstance(st, dict):
        for k in VOLATILE_STUDENT_KEYS:
            if k in st:
                vol["student." + k] = st.pop(k)
    return checked, vol


def registry_keys_for(P, det):
    keys = ["paper1.bench"]
    keys += {"edinburgh": ["datasets.edinburgh_pigs.decoded_frames", "datasets.edinburgh_pigs.annotated",
                           "datasets.edinburgh_pigs.initial_prompts"],
             "pigtrack": ["datasets.pigtrack.sequences", "datasets.pigtrack.seqmaps"],
             "calf": ["datasets.calves.videos", "paper1.calf_windows_protocol" if P.split == "windows_v1"
                      else "paper1.root"]}[P.dataset]
    sk = bc.ARMS[P.arm].get("student_key")
    if sk:
        keys.append(sk)
    if det is not None:
        keys.append(det["registry_key"])
    return keys

# COMMAND ----------

# MAGIC %md
# MAGIC ## Staging, sources, prompts, detection caches, one sequence

# COMMAND ----------


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json_local(path, obj):
    """LOCAL disk only (UC Volumes refuse seek-writes / appends): temp file + os.replace."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp_{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, default=_jd) + "\n")
    os.replace(tmp, path)
    return path


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stage_spec(ctx, spec):
    """Volume / mount -> /local_disk0 once (dbutils.fs.cp; never read frames through FUSE inside the loop)."""
    root, dbu = ctx.stage_root, ctx.dbutils
    if spec["kind"] == "edinburgh":
        clip = spec["seq"]
        d = os.path.join(root, "edinburgh", clip, "video1")
        rec = bc.stage_tree(spec["frames_src"], d, dbu)
        out = {"frames_dir": d, "staged": rec, "gt": None, "gt_sha256": None, "prompts_txt": None}
        if os.path.isfile(spec["gt_src"]):
            out["gt"], out["gt_sha256"] = bc.stage_file(spec["gt_src"], os.path.join(root, "edinburgh", clip, "output.json"),
                                                        dbu)
        if os.path.isfile(spec["prompts_txt_src"]):
            out["prompts_txt"], _ = bc.stage_file(spec["prompts_txt_src"],
                                                  os.path.join(root, "edinburgh", clip, "BoundingBoxes.txt"), dbu)
        return out
    if spec["kind"] == "pigtrack":
        d = os.path.join(root, "pigtrack", ctx.P.split, spec["seq"])
        return {"seq_dir": d, "staged": bc.stage_tree(spec["seq_src"], d, dbu)}
    w = spec["window"]
    mp4, sha = bc.stage_file(spec["clip_src"], os.path.join(root, "calves", w["clip"]), dbu, w.get("clip_sha256"))
    return {"mp4": mp4, "clip_sha256": sha}


def open_source(ctx, spec, local):
    if spec["kind"] == "edinburgh":
        return bc.EdinburghFrames(local["frames_dir"])
    if spec["kind"] == "pigtrack":
        return bc.PigTrackFrames(local["seq_dir"])
    w = spec["window"]
    cap = ctx.hooks["calf_cap_factory"](local["mp4"]) if ctx.hooks.get("calf_cap_factory") else None
    src = bc.CalfWindowFrames(local["mp4"], w["start_fidx"], w["n_frames"], cap=cap, name=spec["seq"])
    if src.fps != bc.CALF_TRACK_FPS:
        src.close()
        raise RuntimeError(f"{spec['seq']}: calf source runs at {src.fps} fps, not {bc.CALF_TRACK_FPS}")
    if "stride" in w and int(w["stride"]) != src.stride:      # the GT window file (the scorer) is keyed on this grid
        src.close()
        raise ValueError(f"{spec['seq']}: window stride {w['stride']} != the v5 decoder's round(native {src.native_fps:g} "
                         f"/ 5) = {src.stride}; the RLE fidx would never line up with the GT window file")
    return src


def source_info(frames):
    d = {"kind": getattr(frames, "kind", None), "name": getattr(frames, "name", None), "n_frames": len(frames),
         "fps": frames.fps, "warnings": list(getattr(frames, "warnings", []) or [])}
    for k in ("night", "pen", "width", "height", "seq_length_ini", "native_fps", "stride"):
        if hasattr(frames, k):
            d[k] = getattr(frames, k)
    if hasattr(frames, "fidxs") and frames.fidxs:
        d["native_fidx_first"], d["native_fidx_last"] = int(frames.fidxs[0]), int(frames.fidxs[-1])
    return d


def build_prompts(ctx, spec, local, frames):
    """seeded run prompts: byte-identical for every arm."""
    if spec["kind"] == "edinburgh":
        kf = bc.read_edinburgh_keyframes(local["gt"])
        ps = bc.protocol_a_edinburgh(kf)
        info = {"n_ids": len(ps), "gt_sha256": local["gt_sha256"]}
        if not local["prompts_txt"]:               # resolve_sequences already requires it for seeded run
            raise FileNotFoundError(f"{spec['seq']}: published seed file not staged ({spec['prompts_txt_src']})")
        old = bc.read_bbox_prompts_txt(local["prompts_txt"])
        if {p.obj_id: list(p.box) for p in ps} != old:
            raise bc.BenchAssertError(f"{spec['seq']}: seeded run prompts from the GT JSON differ from the published "
                                      "seeds -- the GT or seed file changed")
        info["seed_txt_check"] = "identical"
        info["seed_txt_sha256"] = _sha256(local["prompts_txt"])
    elif spec["kind"] == "pigtrack":
        gt = bc.read_mot_txt(frames.gt_path)
        ps, census = bc.protocol_a_from_mot(gt)
        info = {"n_ids": len(ps), "gt_sha256": _sha256(frames.gt_path), "census": census}
    else:
        w = spec["window"]
        masks = {int(k): bc.rle_decode(v).astype(bool) for k, v in w["seed_masks"].items()}
        cap = getattr(frames, "cap", None)
        wh = (int(cap.get(CV_CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(CV_CAP_PROP_FRAME_HEIGHT) or 0)) if cap else (0, 0)
        for gid, m in masks.items():
            if wh[0] and m.shape != (wh[1], wh[0]):
                raise ValueError(f"{spec['seq']}: seed mask {gid} is {m.shape}, the clip is {wh[1]}x{wh[0]}")
        ps = bc.prompts_from_masks(masks, frame=0)
        info = {"n_ids": len(ps), "gt_version": w["gt_version"], "seed_ids": sorted(masks),
                "seed_areas": {str(k): int(m.sum()) for k, m in sorted(masks.items())}}
    bc.schedule_prompts(ps)                     # validate now (track_stream re-validates)
    info["prompt_frames"] = sorted({int(p.frame) for p in ps})
    info["kinds"] = sorted({p.kind for p in ps})
    return ps, info


def _frames_limited(frames, max_frames, hashes=None):
    """Iterate a source up to max_frames, hashing every FRAME_HASH_EVERY-th frame; closes the source iterator (joins
    its decode thread) on early exit."""
    it = iter(frames)
    try:
        for fi, pil, meta in it:
            if max_frames is not None and fi >= max_frames:
                break
            if hashes is not None and fi % bc.FRAME_HASH_EVERY == 0:
                hashes.append([int(fi), bc.frame_sha256(pil)])
            yield fi, pil, meta
    finally:
        c = getattr(it, "close", None)
        if callable(c):
            c()


def validate_cache(cache, det, seq, period, n_frames, allow_partial=False):
    m, bad = cache.meta, []
    if m.get("ckpt_sha256") != det["sha256"]:
        bad.append(f"ckpt_sha256 {str(m.get('ckpt_sha256'))[:12]} != {det['sha256'][:12]}")
    if int(m.get("resolution", -1)) != int(det["resolution"]):
        bad.append(f"resolution {m.get('resolution')} != {det['resolution']}")
    if float(m.get("probe_threshold", -1)) != bc.DET_PROBE_THRESHOLD:
        bad.append(f"probe_threshold {m.get('probe_threshold')} != {bc.DET_PROBE_THRESHOLD}")
    if int(m.get("period", -1)) != int(period):
        bad.append(f"period {m.get('period')} != {period}")
    if m.get("seq") != seq:
        bad.append(f"seq {m.get('seq')} != {seq}")
    if m.get("partial") and not allow_partial:
        bad.append("partial (smoke) cache")
    if int(m.get("n_frames", -1)) != int(n_frames):
        bad.append(f"n_frames {m.get('n_frames')} != {n_frames}")
    missing = [fi for fi in range(0, int(n_frames), int(period)) if fi not in cache.dets]
    if missing:
        bad.append(f"pass frames missing: {missing[:5]}")
    if bad:
        raise bc.BenchAssertError(f"detection cache for {seq} does not fit this run: {'; '.join(bad)}")


def ensure_detection_caches(ctx, specs, tau):
    """detector run detections, ONCE per sequence at DET_PROBE_THRESHOLD, shared by every arm and every tau.
    A published cache is validated and reused; a missing one is built with the live detector (loaded once, freed
    before tracking) and published (a concurrent publisher wins: its cache is used). Smoke builds stay local."""
    P, R, det = ctx.P, ctx.R, ctx.det
    caches, live = {}, None
    try:
        for spec in specs:
            seq = spec["seq"]
            local = stage_spec(ctx, spec)
            frames = open_source(ctx, spec, local)
            n_full = len(frames)
            period = bc.ProtocolD(frames.fps, tau).period
            vol = bc.detection_cache_path(R, P.core_dataset, det["tag"], P.split, seq)
            if os.path.isfile(vol):
                with open(vol, encoding="utf-8") as f:
                    cache = bc.DetectionCache.from_json(json.load(f))
                validate_cache(cache, det, seq, period, n_full)
                caches[seq] = (cache, {"source": "published", "path": vol, "sha256": _sha256(vol)})
                continue
            if live is None:
                ctx.log(f"detector run: loading detector {det['tag']} (resolution {det['resolution']}) for missing caches")
                live = ctx.hooks["detector_loader"](det["local"], det["resolution"])
                if live.info.get("ckpt_sha256") != det["sha256"]:
                    raise RuntimeError("the loaded detector is not the staged checkpoint")
            n = min(n_full, P.smoke) if P.smoke else n_full
            hashes = []
            t0 = time.perf_counter()
            meta = {"dataset": P.core_dataset, "split": P.split, "seq": seq, "detector_tag": det["tag"],
                    "registry_key": det["registry_key"], "file": det["file"], "ckpt_sha256": det["sha256"],
                    "resolution": det["resolution"], "num_classes": live.info.get("num_classes"),
                    "class_names": live.info.get("class_names"), "fps": frames.fps, "n_frames": n,
                    "partial": n < n_full, "built_by": f"{NB_NAME} {NB_VERSION}", "created_utc": _now()}
            cache = bc.DetectionCache.build(live, _frames_limited(frames, n, hashes), period, meta=meta)
            cache.meta["frame_hashes"] = hashes
            cache.meta["build_s"] = round(time.perf_counter() - t0, 2)
            validate_cache(cache, det, seq, period, n, allow_partial=True)
            loc = write_json_local(os.path.join(ctx.local_run, "_detections", f"{seq}.json"), cache.to_json())
            if cache.meta["partial"]:
                caches[seq] = (cache, {"source": "local_smoke", "path": None, "sha256": _sha256(loc)})
                continue
            try:
                bc.publish_no_overwrite(loc, vol, ctx.dbutils)
                caches[seq] = (cache, {"source": "built", "path": vol, "sha256": _sha256(loc)})
            except FileExistsError:            # another arm's job published first: every arm must use THAT cache
                with open(vol, encoding="utf-8") as f:
                    cache = bc.DetectionCache.from_json(json.load(f))
                validate_cache(cache, det, seq, period, n_full)
                caches[seq] = (cache, {"source": "published_race", "path": vol, "sha256": _sha256(vol)})
            ctx.log(f"  cache {seq}: {len(cache.dets)} passes, {sum(len(v) for v in cache.dets.values())} detections "
                    f"(> {bc.DET_PROBE_THRESHOLD}), {caches[seq][1]['source']}")
    finally:
        if live is not None:
            del live
            _free_gpu()
    return caches


def _free_gpu():
    gc.collect()
    if bc.TORCH_OK:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _reset_peaks():
    if bc.TORCH_OK:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def check_frame_hashes(ctx, seq, hashes, own_tel):
    """frame hashes must agree across arms on the same sequence. Compares with every other run of
    this dataset/split already published (any arm, protocol, variant). Returns the number of runs compared."""
    base = ctx.R.out("paper1.bench", ctx.P.core_dataset)
    mine = {int(f): h for f, h in hashes}
    found = set()
    for pat in (f"{base}/*__prot*/{ctx.P.split}/telemetry/{seq}.json",
                f"{base}/*__prot*/{ctx.P.split}/*/telemetry/{seq}.json"):
        found.update(glob.glob(pat))
    n = 0
    for p in sorted(found):
        if os.path.normcase(os.path.normpath(p)) == os.path.normcase(os.path.normpath(own_tel)):
            continue
        with open(p, encoding="utf-8") as f:
            other = ((json.load(f).get("stats") or {}).get("frame_hashes")) or []
        for fi, h in other:
            if int(fi) in mine and mine[int(fi)] != h:
                raise bc.BenchAssertError(f"{seq}: frame {fi} hash {mine[int(fi)]} != {h} in {p} (identical pixels "
                                          "across arms violated: run INVALID)")
        n += 1
    return n


def _publish(ctx, loc, dst):
    try:
        return bc.publish_no_overwrite(loc, dst, ctx.dbutils)
    except FileExistsError as e:
        raise RuntimeError(f"{e} -- an incomplete earlier attempt published a DIFFERENT file; "
                           "re-run this sequence in a new variant directory (variant=<tag>)") from e


def vol_paths(run_dir, seq):
    return {"done": f"{run_dir}/.done_{seq}", "mot": f"{run_dir}/mot/{seq}.txt",
            "tel": f"{run_dir}/telemetry/{seq}.json", "rle": f"{run_dir}/rle/{seq}.jsonl.gz"}


def recover_if_complete(ctx, seq):
    """Telemetry is published only after MOT (and RLE), and .done after telemetry: a sequence whose telemetry exists
    without a marker crashed between the last two publishes. Its outputs are complete -> write the marker, no re-run."""
    v = vol_paths(ctx.run_dir, seq)
    if os.path.isfile(v["done"]) or not os.path.isfile(v["tel"]):
        return None
    need = ["mot"] + (["rle"] if ctx.P.rle else [])
    missing = [k for k in need if not os.path.isfile(v[k])]
    if missing:
        raise RuntimeError(f"{seq}: telemetry published but {missing} absent -- not a known crash state; inspect "
                           f"{ctx.run_dir} and re-run in a new variant directory")
    rec = {"seq": seq, "recovered_from_partial": True, "utc": _now(), "notebook": f"{NB_NAME} {NB_VERSION}",
           "files": {k: _sha256(v[k]) for k in need + ["tel"]}}
    loc = write_json_local(os.path.join(ctx.local_run, f".done_{seq}"), rec)
    _publish(ctx, loc, v["done"])
    ctx.log(f"{seq}: outputs complete from an earlier attempt -> .done written (no re-run)")
    return {"seq": seq, "status": "recovered"}


def run_sequence(ctx, arm, spec):
    """Track ONE sequence with track_stream and publish mot -> rle -> telemetry -> .done (marker LAST)."""
    P, seq = ctx.P, spec["seq"]
    v = vol_paths(ctx.run_dir, seq)
    if os.path.isfile(v["done"]):
        return {"seq": seq, "status": "done"}
    local = stage_spec(ctx, spec)
    frames = open_source(ctx, spec, local)
    tel = None
    try:
        n_full = len(frames)
        n_expected = min(n_full, P.smoke) if P.smoke else n_full
        prompts, prompt_info, protd, detector, cache_rec = (), None, None, None, None
        if P.protocol == "A":
            prompts, prompt_info = build_prompts(ctx, spec, local, frames)
        else:
            cache, cache_rec = ctx.caches[seq]
            protd = bc.ProtocolD(frames.fps, ctx.tau)
            detector = cache.detector()
        mot = bc.MotWriter()
        rle = bc.RleWriter(os.path.join(ctx.local_run, "rle", f"{seq}.jsonl.gz")) if P.rle else None
        tel = bc.Telemetry(arm.model, use_cuda=ctx.hooks.get("use_cuda"))
        # J0b: the previous sequence's session must be gone before this one starts:
        # vs the weights-only baseline (+ lazily built caches) before the first sequence of this job, then vs the level
        # the first teardown left (+ drift tolerance)
        if ctx.mem_ref is None:
            mem_start = bc.check_memory_returned(ctx.mem_base, ctx.hooks["mem_settle"](), what=f"{seq}: start")
        else:
            mem_start = bc.check_memory_returned(ctx.mem_ref, ctx.hooks["mem_settle"](), tol_mib=bc.MEM_DRIFT_TOL_MIB,
                                                 what=f"{seq}: start (vs the first teardown)")
        _reset_peaks()
        ctx.log(f"{seq}: {n_expected} frames @ {frames.fps:g} fps, protocol {P.protocol}"
                + (f", {len(prompts)} prompts" if P.protocol == "A" else f", tau {ctx.tau}, period {protd.period}"))
        try:
            st = bc.track_stream(arm, frames, protocol=P.protocol, prompts=prompts, detector=detector, protd=protd,
                                 keep_last_n=P.keep_last_n, prune_every=bc.PRUNE_EVERY, pop=P.pop, mot=mot, rle=rle,
                                 tel=tel, max_frames=P.smoke or None, progress_every=P.progress_every, log=ctx.log)
        finally:
            if rle is not None:
                rle.close()
        peak = bc.mem_snapshot()
        tsum = tel.summary()
        src_info = source_info(frames)
    finally:
        if tel is not None:
            tel.detach()
        if spec["kind"] == "calf":
            frames.close()
    if st["frames"] != n_expected:
        raise bc.BenchAssertError(f"{seq}: {st['frames']} frames processed, expected {n_expected}")
    mem_end = bc.check_memory_returned(ctx.mem_base, ctx.hooks["mem_settle"](), what=f"{seq}: after teardown",
                                       raise_on_fail=False)       # recorded here; the NEXT sequence's start check raises
    if ctx.mem_ref is None and mem_end.get("ok") is True:
        ctx.mem_ref = mem_end["alloc_mib"]                         # the level later sequences must return to
    memory = {"baseline_mib": ctx.mem_base, "baseline_source": ctx.mem_base_source, "start": mem_start,
              "after_teardown": mem_end, "first_teardown_ref_mib": ctx.mem_ref, "teardown": st.get("teardown"),
              "peak_alloc_mib": None if not peak else round(peak["peak_alloc"], 3),
              "peak_above_baseline_mib": None if not peak or ctx.mem_base is None
              else round(peak["peak_alloc"] - ctx.mem_base, 3)}
    if mem_end.get("ok") is False:
        ctx.log(f"WARNING {seq}: {mem_end['excess_mib']:.0f} MiB above the baseline after teardown "
                f"(session alive: {(st.get('teardown') or {}).get('session_alive_after_teardown')})")
    if cache_rec is not None:
        cmap = {int(f): h for f, h in (ctx.caches[seq][0].meta.get("frame_hashes") or [])}
        for fi, h in st["frame_hashes"]:
            if int(fi) in cmap and cmap[int(fi)] != h:
                raise bc.BenchAssertError(f"{seq}: frame {fi} differs from the frame the detection cache saw")
    n_cmp = check_frame_hashes(ctx, seq, st["frame_hashes"], v["tel"])
    if ctx.cfg_status != "matched":           # BENCH_CONFIG is created only once a sequence has VALID outputs, so a
        ctx.cfg_status = ensure_bench_config(ctx, ctx.cfg)   # failing input never freezes a config into the run dir
        ctx.cfg_created = ctx.cfg_created or ctx.cfg_status == "created"
        ctx.cfg_status = "matched"
    mot_loc = mot.write(os.path.join(ctx.local_run, "mot", f"{seq}.txt"))
    tele = {"seq": seq, "dataset": P.core_dataset, "split": P.split, "arm": P.arm, "protocol": P.protocol,
            "variant": ctx.variant, "notebook": f"{NB_NAME} {NB_VERSION}", "created_utc": _now(),
            "source": src_info, "staged": {k: val for k, val in local.items() if k in ("staged", "clip_sha256")},
            "prompts": prompt_info, "stats": st, "telemetry": tsum, "frame_hash_runs_compared": n_cmp,
            "mot_rows": len(mot), "smoke_max_frames": P.smoke or None, "memory": memory}
    if protd is not None:
        tele["protocol_D"] = {"params": protd.params, "events": protd.events, "cache": cache_rec}
    tel_loc = write_json_local(os.path.join(ctx.local_run, "telemetry", f"{seq}.json"), tele)
    files = {"mot": _sha256(mot_loc)}
    _publish(ctx, mot_loc, v["mot"])
    if rle is not None:
        files["rle"] = _sha256(rle.path)
        _publish(ctx, rle.path, v["rle"])
    files["tel"] = _sha256(tel_loc)
    _publish(ctx, tel_loc, v["tel"])
    done = {"seq": seq, "utc": _now(), "frames": st["frames"], "forwards": st["forwards"], "rows": st["rows"],
            "objects": st["objects"], "wall_s": st["wall_s"], "files": files, "notebook": f"{NB_NAME} {NB_VERSION}",
            "detection_cache": cache_rec}
    _publish(ctx, write_json_local(os.path.join(ctx.local_run, f".done_{seq}"), done), v["done"])
    fps = st["frames"] / max(st["wall_s"], 1e-9)
    ctx.log(f"{seq}: done -- {st['frames']} frames, {st['forwards']} forwards, {st['objects']} objects, "
            f"{st['rows']} rows, {st['wall_s']:.0f} s ({fps:.2f} fps), compared with {n_cmp} other run(s)")
    return {"seq": seq, "status": "tracked", "frames": st["frames"], "rows": st["rows"], "objects": st["objects"],
            "wall_s": st["wall_s"], "fps": round(fps, 3)}


def check_existing_bench_config(ctx, cfg):
    """Fail fast: an existing BENCH_CONFIG.json must match this job exactly (config_check: code, versions, GPU class,
    arm, protocol, stream settings, run flags; created_utc and extra excluded). -> 'matched' | 'absent'."""
    dst = f"{ctx.run_dir}/BENCH_CONFIG.json"
    if not os.path.isfile(dst):
        return "absent"
    with open(dst, encoding="utf-8") as f:
        bc.config_check(json.load(f), cfg)
    return "matched"


def ensure_bench_config(ctx, cfg):
    """BENCH_CONFIG.json is written ONCE per run dir (called just before the first sequence publishes)."""
    dst = f"{ctx.run_dir}/BENCH_CONFIG.json"
    if check_existing_bench_config(ctx, cfg) == "matched":
        return "matched"
    loc = write_json_local(os.path.join(ctx.local_run, "BENCH_CONFIG.json"), cfg)
    try:
        bc.publish_no_overwrite(loc, dst, ctx.dbutils)
        return "created"
    except FileExistsError:                    # a concurrent job of the same run dir published first
        with open(dst, encoding="utf-8") as f:
            bc.config_check(json.load(f), cfg)
        return "matched"

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPU / HF / arm (production hooks) and the driver

# COMMAND ----------


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cuda_probe(retries=3, sleep_s=60):
    """Subprocess CUDA start-up probe BEFORE this process touches CUDA (A10 boots occasionally fail at CUDA init,
    see the cluster notes). Raises after `retries` failures so the job retry lands on a fresh cluster."""
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
        raise RuntimeError(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r} lacks "
                           "expandable_segments:True (job cluster spec)")
    probe = cuda_probe()
    import torch
    name = torch.cuda.get_device_name(0)
    up = name.upper()
    cls = "A100" if "A100" in up else ("A10" if re.search(r"A10(?!\d)", up) else name)
    if P.require_gpu != "any" and cls != P.require_gpu:
        raise RuntimeError(f"GPU {name} (class {cls}) but require_gpu={P.require_gpu}: tracking / baselines run on the "
                           "A10 (user GPU policy; a run dir never mixes GPU classes)")
    probe["class"] = cls
    return probe


def _prod_arm_loader(P, R, hf_token, revisions, dbutils_, log):
    return bc.load_arm(P.arm, device="cuda", hf_token=hf_token, revisions=revisions, registry=R, dbutils=dbutils_,
                       log=log)


def _prod_detector_loader(ckpt_local, resolution):
    return bc.load_rfdetr_detector(ckpt_local, resolution=resolution)


def _notebook_source():
    """The bytes of the code that is RUNNING (its sha256 goes into BENCH_CONFIG): this file in plain-python mode, else
    the notebook read through the workspace FUSE (SOURCE format), else exported with the Databricks SDK."""
    if not IS_NOTEBOOK:
        with open(os.path.abspath(__file__), "rb") as f:
            return f.read()
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
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
    raise RuntimeError(f"{NB_NAME} cannot fingerprint its own source, so BENCH_CONFIG would be blind to a code change: "
                       + "; ".join(errs))


PROD_HOOKS = {
    "local_root": "/local_disk0/tol63",
    "arm_loader": _prod_arm_loader,
    "detector_loader": _prod_detector_loader,
    "calf_cap_factory": None,                  # CalfWindowFrames opens the staged clip with cv2
    "notebook_source": None,                   # None -> _notebook_source()
    "gpu_preflight": gpu_preflight,
    "hf_token": lambda dbu: bc.hf_token_from_dbutils(dbu),
    "use_cuda": None,                          # Telemetry: CUDA events when CUDA is available
    "mem_settle": bc.settle_memory,            # gc + empty_cache -> allocated MiB (None without CUDA)
    "log": _log,
}


def main(raw, *, dbutils_=None, R=None, hooks=None):
    """One job = one arm x one dataset split x one protocol -> one run directory. Returns a JSON-able summary."""
    h = dict(PROD_HOOKS)
    h.update(hooks or {})
    log = h["log"]
    t_start = time.time()
    P = parse_params(raw)
    R = R if R is not None else farm_paths.load(version=REGISTRY_VERSION)
    matrix = check_matrix(P, log)
    det = resolve_detector(R, P)
    variant = derive_variant(P, det)
    run_dir = bc.bench_run_dir(R, P.core_dataset, P.arm, P.protocol, P.split, variant)
    check_out_root(P, R, run_dir)
    local_run = os.path.join(h["local_root"], "runs", P.core_dataset, f"{P.arm}__prot{P.protocol}", P.split,
                             variant or "_main")
    ctx = types.SimpleNamespace(P=P, R=R, dbutils=dbutils_, hooks=h, log=log, run_dir=run_dir, local_run=local_run,
                                stage_root=os.path.join(h["local_root"], "stage"), variant=variant, det=det,
                                tau=None, caches={}, cfg=None, cfg_status="absent", cfg_created=False,
                                mem_base=None, mem_base_source=None, mem_ref=None)
    log(f"run dir {run_dir} (variant {variant or '-'}; {matrix})")
    specs, seq_prov = resolve_sequences(R, P)
    results, todo = [], []
    for spec in specs:
        if os.path.isfile(vol_paths(run_dir, spec["seq"])["done"]):
            results.append({"seq": spec["seq"], "status": "done"})
            continue
        rec = recover_if_complete(ctx, spec["seq"])
        if rec:
            results.append(rec)
        else:
            todo.append(spec)
    summary = {"notebook": f"{NB_NAME} {NB_VERSION}", "run_dir": run_dir, "variant": variant, "arm": P.arm,
               "dataset": P.core_dataset, "split": P.split, "protocol": P.protocol, "matrix": matrix,
               "n_sequences": len(specs)}
    if not todo:
        log(f"nothing to do: all {len(specs)} sequences have .done markers")
        summary.update(results=results, bench_config="untouched", wall_s=round(time.time() - t_start, 1))
        return summary
    tau, tau_src = load_tau(R, P)
    ctx.tau = tau
    revisions = resolve_revisions(R, P)
    if not revisions:
        log("WARNING: no HF revision pins")
    if det is not None:
        det["local"], det["sha256"] = bc.stage_file(det["src"], os.path.join(h["local_root"], "models", det["tag"],
                                                                             det["src"].rsplit("/", 1)[-1]),
                                                    dbutils_, det["sha256_pin"])
        det["resolution"], det["resolution_source"] = detector_resolution(det, P.detector_resolution)
    probe = h["gpu_preflight"](P) if h.get("gpu_preflight") else {"skipped": True}
    token = h["hf_token"](dbutils_) if h.get("hf_token") else None
    arm = h["arm_loader"](P, R, token, revisions, dbutils_, log)
    del token
    for hf_id, rev in revisions.items():
        if arm.info.get("hf_id") == hf_id and len(rev) == 40 and arm.info.get("commit_hash") not in (None, rev):
            raise RuntimeError(f"{hf_id}: loaded commit {arm.info.get('commit_hash')} != pinned {rev}")
    arm_checked, arm_volatile = split_arm_info(arm.info)
    det_info = None if det is None else {
        "tag": det["tag"], "registry_key": det["registry_key"], "file": det["file"], "ckpt_sha256": det["sha256"],
        "resolution": det["resolution"], "probe_threshold": bc.DET_PROBE_THRESHOLD, "default_tag": det["default_tag"],
        "cache": f"bench/{P.core_dataset}/_detections/{det['tag']}/{P.split}/<seq>.json (shared by all arms and taus)"}
    src_bytes = h["notebook_source"] if h.get("notebook_source") is not None else _notebook_source()
    cfg = bc.bench_config(
        arm_checked, dataset=P.core_dataset, split=P.split, protocol=P.protocol, variant=variant,
        protd_params=protd_config_params(tau) if P.protocol == "D" else None, keep_last_n=P.keep_last_n,
        prune_every=bc.PRUNE_EVERY, pop=P.pop, detector_info=det_info, registry=R,
        registry_keys=registry_keys_for(P, det), notebook_source=src_bytes,
        extra={"arm_measurements": arm_volatile, "seqs_requested": P.seqs, "started_utc": _now(), "probe": probe,
               "cluster_id": os.environ.get("DB_CLUSTER_ID"),
               "detector_resolution_source": None if det is None else det["resolution_source"]})
    cfg["run"] = {"notebook": NB_NAME, "notebook_version": NB_VERSION, "dataset_widget": P.dataset, "matrix": matrix,
                  "smoke_max_frames": P.smoke or None, "rle": P.rle, "require_gpu": P.require_gpu, "tau": tau,
                  "tau_source": tau_src, "sequences": seq_prov, "flags": arm_flags(arm.info), "revisions": revisions,
                  "frame_hash_every": bc.FRAME_HASH_EVERY, "mot_writer": "MotWriter() default xy offset (0-based)",
                  "pip": PIP_SPEC}
    cfg = json.loads(json.dumps(cfg, default=_jd))
    if P.require_gpu != "any" and (cfg.get("gpu") or {}).get("class") != P.require_gpu:
        raise RuntimeError(f"BENCH_CONFIG gpu class {(cfg.get('gpu') or {}).get('class')} != {P.require_gpu}")
    ctx.cfg = cfg
    ctx.cfg_status = check_existing_bench_config(ctx, cfg)        # mismatch -> stop BEFORE any work
    log(f"BENCH_CONFIG {ctx.cfg_status}; {len(todo)} of {len(specs)} sequences to run")
    if P.protocol == "D":
        ctx.caches = ensure_detection_caches(ctx, todo, tau)
        _reset_peaks()
    # weights-only baseline for the between-sequence memory check (after the arm load and, for detector run, after the
    # live detector that built missing caches was freed); recorded per sequence in telemetry["memory"]
    ctx.mem_base = h["mem_settle"]()
    ctx.mem_base_source = "after load" + (" + detection caches" if P.protocol == "D" else "")
    log(f"memory baseline: {ctx.mem_base if ctx.mem_base is None else round(ctx.mem_base, 1)} MiB allocated "
        f"({ctx.mem_base_source}; tolerance {bc.MEM_RETURN_TOL_MIB:.0f} MiB before the first sequence, then "
        f"{bc.MEM_DRIFT_TOL_MIB:.0f} MiB above the first teardown)")
    for spec in todo:
        results.append(run_sequence(ctx, arm, spec))
    order = {s["seq"]: k for k, s in enumerate(specs)}
    summary["bench_config"] = "created" if ctx.cfg_created else ctx.cfg_status
    summary.update(results=sorted(results, key=lambda r: order[r["seq"]]), wall_s=round(time.time() - t_start, 1))
    del arm
    _free_gpu()
    return summary

# COMMAND ----------





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
            out[name] = os.environ.get("TOL63_" + name.upper(), default)
    return out


RAW = read_widgets()
print("widgets:", json.dumps(RAW, sort_keys=True))
SUMMARY = main(RAW, dbutils_=dbutils)
print(json.dumps(SUMMARY, indent=1, default=str))

# COMMAND ----------

if IS_NOTEBOOK:
    dbutils.notebook.exit(json.dumps(SUMMARY, default=str))
