# Databricks notebook source
# MAGIC %md
# MAGIC # Distil the Edinburgh student with mined frames
# MAGIC
# MAGIC Trains one student on 3,027 targets: the 2,400 Edinburgh base targets plus the 627 frames chosen by
# MAGIC `select_targets_edinburgh`. Validation uses all 600 targets of `2019_11_28_000113`. The recipe matches
# MAGIC `distill_pigtrack_mined`: AdamW, warm restarts, EMA, early stopping, stochastic depth 0.1 in `train()` only.
# MAGIC Use `eval_stage=only` on an A10 to compute held-out fidelity from a finished run.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "timm==1.0.25" "Pillow==11.1.0" "huggingface_hub"

# COMMAND ----------

try:
    dbutils.library.restartPython()  # noqa: F821
except NameError:
    pass

# COMMAND ----------

import collections
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import os.path as osp
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
import uuid

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

NB_NAME = "distill_edinburgh_mined"
NB_VERSION = "distill_edinburgh_mined-2026-10-30.1"
CONFIG_VERSION = "distill_edinburgh_mined-distill-v1"
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
_PATHS_DIRS = [CODE_DIR, os.environ.get("FARMSAM_PATHS_DIR")]
if not IS_NOTEBOOK:
    _PATHS_DIRS.append(osp.join(osp.dirname(CODE_DIR), "catalog", "build"))
farm_paths = _import_exact("farm_paths", _PATHS_DIRS)
# Full sha256 of the shared modules this notebook may run against: the Workspace-root core 5e705fff (the core S-PT trained with),
# _pinned/core_bc46ecfb or _pinned/core_8689f125 (the base of the b1.11 scoring core); every symbol used here is AST-identical in the three
# (bench/pt_mining/_work/ast_core_check.json; bench/edi_mining/train_build/ast_core_check.json for 8689f125).
RID = uuid.uuid4().hex[:6]
print(f"{NB_NAME} {NB_VERSION} rid {RID}: bench_core {core.BENCH_VERSION} "
      f"sha256 {core.sha256_file(core.__file__)[:16]}, distill_core sha256 {core.sha256_file(dc.__file__)[:16]}, "
      f"farm_paths sha256 {core.sha256_file(farm_paths.__file__)[:16]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters, recipe constants and the pure-Python driver

# COMMAND ----------

WIDGETS = collections.OrderedDict([
    ("run", "edinburgh_mined_v1"), ("max_epochs", "100"), ("resume_every", "10"), ("seed", "42"), ("num_workers", "8"),
    ("smoke_steps", "0"), ("smoke_val_batches", "0"), ("fid_max_frames", "0"), ("stop_after_epoch", "0"),
    ("publish_model", "false"), ("require_gpu", "A100"), ("stage_threads", "16"),
    ("drop_path_rate", "0.1"), ("eval_stage", "inline"),
    ("base_manifest_sha256", ""), ("extra_manifest_sha256", ""), ("summary_sha256", ""), ("expect_ckpt_sha256", ""),
])
CANONICAL_RUN = "pigtrack_v2_dp01"                 # the registry key models.student_pigtrack names this run
# b1.6: NO run of this notebook publishes a model (models/ is written only after test scoring), so PUBLISH_RUNS is empty and the
# distill_student_pigtrack publish paths below stay inert (parse_params refuses publish_model=true).
PUBLISH_RUNS = {}
# The b1.11 arm: run tag -> (target set, training seed, arm name). drop_path_rate 0.1 = S-Edi's (timm default). ONE student; no base,
# random or second-seed arm exists in this experiment.
ARM_RUNS = {"edinburgh_mined_v1": ("mined", 42, "S-Edi-mined")}
ARM_DROP_PATH = 0.1
TARGET_SETS = tuple(v[0] for v in ARM_RUNS.values())
ARM_BY_SET = {v[0]: v[2] for v in ARM_RUNS.values()}
ARM_SEED_BY_SET = {v[0]: v[1] for v in ARM_RUNS.values()}
REF_RUN = "pigtrack_v2_dp01"                       # = S-PT: recipe / training / base-data / versions reference of every b1.6 run
SPT_SHA256 = "c07140f2de80a5e9faf175bdb4fc071bb1064e4757b91c3eefe55e271e5da69c"   # pigtrack_v2_dp01 EMA epoch 32 (S-PT)
TIMM_DEFAULT_DROP_PATH = 0.1                       # timm 1.0.25 tiny_vit_21m_512 default (tiny_vit.py, S-Edi's training)
DROP_PATH_MAX = 0.5
DP_RECIPE_KEYS = ("name", "student", "drop_path_rate", "drop_path_scope")   # the only recipe keys a dp run may change
EVAL_STAGES = ("inline", "defer", "only")
EVAL_ONLY_IGNORE = ("gpu_class", "versions")       # an eval_stage=only run may differ from the training run in these
TARGETS_KEY = "paper1.teacher_targets.edinburgh"            # <clip>/backbone_{fi:06d}.pt (stored 2026-02)
MODEL_KEY = "models.student_pigtrack"
REF_STUDENT_KEY = "models.student_edinburgh"       # published-weights reference on the same val frames (read-only)
REF_STUDENT_SHA256 = "9686b4c0f60e02251d8c2febee2029c1e7ebe74e85fbefda7e0c5daffc2075f2"   # J0: EMA epoch 75 (S-Edi)
REGISTRY_VERSION = 1
CKPT_NAME = "best_backbone_student_edinburgh.pt"   # b1.11: final_<rid>/<CKPT_NAME> under paper1.distillation_runs
HISTORY_NAME = "backbone_distill_history.json"     # = the Edinburgh model folder's history name
# ---- distill_edinburgh_mined: the Edinburgh data identity. Every value is registered in b1.11 (items 0, 3, 4) or recorded in
# the INVENTORY (stored targets). PINS is a literal that submit_edi_guarded.py reads with ast (nothing is executed).
EDI_TRAIN = ("2019_11_05_000002", "2019_11_11_000028", "2019_11_11_000036", "2019_11_22_000010")   # the 4 base clips (S-Edi's training clips)
EDI_VAL = ("2019_11_28_000113",)                                                                   # the validation clip (all 600 frames)
POOL_CLIPS = ("2019_11_05_000009", "2019_11_11_000016", "2019_11_15_000033")                       # pool P3
EDI_TEST = ("2019_12_02_000005", "2019_12_02_000208", "2019_12_10_000060", "2019_12_10_000078")    # NEVER read: refused in every text and row
CLIP_PREFIX = "2019_11_"
CLIP_RX = re.compile(r"^2019_11_\d{2}_\d{6}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FRAMES_KEY = "datasets.edinburgh_pigs.decoded_frames"        # <clip>/video1/{fi+1:07d}.jpg
SELECTION_PARTS = ("edi_mining", "selection_v1")             # under registry paper1.root: select_targets_edinburgh's output folder
UNUSED_PARTS = ("edi_mining", "teacher_targets_unused_v1")   # under registry paper1.root: measure_edinburgh_hard_frames's pool targets (read only)
SAMPLE_BASE_FEATURE = ("2019_11_05_000002", 0, "9706704cd9fd853b13d775a1f72f7c2e8a2c64dc7057f3ff68d2d8793673c6bc")   # INVENTORY 4
EDI_EXPECT = {"train_clips": EDI_TRAIN, "val_clips": EDI_VAL, "pool_clips": POOL_CLIPS, "n_frames": 600, "n_base": 2400, "n_val": 600,
              "n_extra": 627, "per_clip_extra": 209, "canon": "53668699a31324cf009cc8c8aef50309a9413cc0316509ecbb15d1cc552b2702",
              "unused_sha": "3b9f6f59db4bcf6899f9f96968c31efe8925e383ec6aae7bfa8708d9c1ea7f6f", "sample_feature": SAMPLE_BASE_FEATURE,
              "selection_sha": "11e629fef34a85ad35d5c237887c43f4b8b1531907dd7cecdb2bff7d80e4fa75", "fide_n": 120}
PINS = {"experiment": "b1.11_edinburgh_mining", "run": "edinburgh_mined_v1", "arm": "S-Edi-mined", "pool": "P3", "M": 627,
        "q_per_clip": 209, "n_base": 2400, "n_val": 600, "seed": 42, "drop_path_rate": 0.1, "max_epochs": 100,
        "canon_sha256_M": "53668699a31324cf009cc8c8aef50309a9413cc0316509ecbb15d1cc552b2702",
        "canon_sha256_M_lexicographic_WRONG": "6eac1185df849bdd9995d6ee75f8079b3ae32202d88839f4fb8c65594d4e19c5",
        "selection_file_sha256_M": "11e629fef34a85ad35d5c237887c43f4b8b1531907dd7cecdb2bff7d80e4fa75",
        "unused_manifest_sha256": "3b9f6f59db4bcf6899f9f96968c31efe8925e383ec6aae7bfa8708d9c1ea7f6f",
        "sedi_sha256": "9686b4c0f60e02251d8c2febee2029c1e7ebe74e85fbefda7e0c5daffc2075f2",
        "sedi_ema_val_loss": 0.25041554232438407, "gm_factor": 0.1, "a100_limit_h": 8.0}
SEDI_EMA_VAL_LOSS = 0.25041554232438407       # S-Edi's recorded `ema_val_loss`, EMA epoch 75 of 90 (backbone_distill_history.json)
GM_FACTOR = 0.1
# b1.11 item 8(e): where the targets come from (recorded in DISTILL_CONFIG.data)
TARGET_SOURCES = {"base_and_val": "stored 2026-02 extraction (teacher_targets/edinburgh; SAM 3 unpinned revision, bf16, GT-box prompts)",
                  "pool": "measure_edinburgh_hard_frames 2026-10 extraction (teacher_targets_unused_v1; SAM 3 3c879f39..., dummy boxes)"}
# `training` keys that a smoke run changes by construction; the S-PT gate does not compare them in a smoke run
SMOKE_TRAINING_KEYS = ("max_epochs", "resume_every", "smoke_steps", "smoke_val_batches", "fid_max_frames")
# [FID-E] fp32 means of S-Edi on the stride-5 val subset (the fidelity check stored, fidelity.json sha256 756e2d92...; measure_edinburgh_hard_frames reproduced them with diff 0.0)
FIDE_VAL = {"cos_patch": 0.7508357814739799, "cos_global": 0.6921778052278609, "scale_ratio": 0.9557935679024919, "mse": 1.1941566283638172}
FIDE_VAL_4DP = {"cos_patch": 0.7508, "cos_global": 0.6922, "scale_ratio": 0.9558, "mse": 1.1942}
FIDE_PASS = ("identical", "within_4dp")
# status "within_4dp" = every |ours - stored| <= FIDE_TOL (an ABSOLUTE tolerance, not a rounding comparison: the stored MSE 1.1941566 lies 6.6e-6 above the
# rounding edge 1.19415 of its registered 4-decimal value, so a rounding test could flip on a harmless 1e-5). 1e-4 = about 2x the recorded A10-vs-A100
# difference (4.5e-5). This is a gross-breakage control (how far a wrong frame subset or preprocessing would move these means was not measured). The status NAME is kept: Tol_82 reads it.
FIDE_TOL = 1e-4
# The published Edinburgh recipe (Fixed cells 13/15/17/19). Recorded in DISTILL_CONFIG; the code below implements it.
RECIPE = {
    "name": "edinburgh_fixed_v1 + drop_path_rate=0.0",
    "source": "Student+TeacherDecoder_TinyViT_Fixed cells 13/15/17/19 (catalog/paper1_ws/student_fixed.txt)",
    "student": "bench_core.StudentEncoderTinyViT(out_channels=1024, pretrained=True) (verbatim export + [bench-C1])",
    "timm_model": "tiny_vit_21m_512", "pretrained": True, "drop_path_rate": 0.0, "n_params": 40655408,
    "loss": {"dir_w": 1.0, "cos_w": 0.5, "scale_w": 0.3, "raw_mse_w": 0.1, "scale": "MSE(ch std) + MSE(ch mean)"},
    "optimizer": {"type": "AdamW", "lr_backbone": 1e-4, "lr_heads": 3e-4, "lr_scale_bias": 3e-4,
                  "wd": 1e-4, "wd_scale_bias": 0.0, "groups": 7},
    "scheduler": {"type": "CosineAnnealingWarmRestarts", "T_0": 15, "T_mult": 2, "eta_min": 1e-6,
                  "step": "scheduler.step(epoch) once per epoch"},
    "precision": "torch.autocast('cuda', float16) + GradScaler (= the deprecated torch.cuda.amp pair), loss in autocast",
    "grad_clip": 1.0, "ema_decay": 0.9995, "ema_over": "named_parameters (requires_grad)",
    "early_stopping": {"patience": 15, "min_delta": 1e-5, "on": "EMA val loss"},
    "checkpoint": "save when EMA val loss < best (strict); payload {epoch, model_state_dict, ema_shadow, optimizer, "
                  "metrics}",
    "batch": 4, "image_size": [1024, 1024], "drop_last_train": True,
    "augment": "ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02) on the full-res frame, then "
               "h-flip image + feature (torch.rand(1) > 0.5), then Resize((1024,1024)), ToTensor, TinyViT-512 mean/std",
    "val": "Resize/ToTensor/Normalize, batch 4; val loss = mean over batches; EMA val each epoch",
}
RECIPE_V1_SHA256 = "b1f4457d554dd70eada8db06d2b99fe4e79fd2d6e826043778495a1b20aeb255"   # json_dumps
FID_BANDS = {"cos_patch_min": 0.70, "scale_ratio": [0.8, 1.2]}        # distill_student's a-priori bands (informative only)


def recipe_for(drop_path_rate):
    """The recipe dict recorded in DISTILL_CONFIG. drop_path_rate 0.0 -> RECIPE itself, byte-identical to pigtrack_v1's
    (RECIPE_V1_SHA256, checked at load), so that run still resumes / reproduces. Otherwise the four keys of
    DP_RECIPE_KEYS change and nothing else."""
    dp = float(drop_path_rate)
    if dp == 0.0:
        return copy.deepcopy(RECIPE)
    r = copy.deepcopy(RECIPE)
    r["name"] = f"edinburgh_fixed_v1, drop_path_rate={dp:g} in training (timm default for tiny_vit_21m_512 = S-Edi's)"
    r["student"] = (RECIPE["student"] + f"; built with drop_path_rate={dp:g} in the class's own timm.create_model call "
                    "(make_student; the export never passed the argument, so S-Edi trained with timm's default)")
    r["drop_path_rate"] = dp
    r["drop_path_scope"] = ("stochastic depth active only in train(); every val / EMA-val pass, the fidelity and the "
                            "released checkpoint run in eval() (DropPath = identity); the checkpoint has no DropPath "
                            "state and loads unchanged into the bench's drop_path_rate=0.0 class")
    return r


def recipe_match_diffs(ref_cfg, cfg):
    """b1.2 item 4: a publishing run other than pigtrack_v1 must equal it in `data`, `training`,
    `version` and `publish_model`, and differ in the recipe only in DP_RECIPE_KEYS. Returns the offending diffs."""
    bad = []
    for k in ("data", "training", "version", "publish_model"):
        bad += [(f"{k}{'.' + p if p else ''}", a, b) for p, a, b in core.config_diff(ref_cfg.get(k), cfg.get(k), ())]
    for p, a, b in core.config_diff(ref_cfg.get("recipe"), cfg.get("recipe"), ()):
        if p.split(".")[0] not in DP_RECIPE_KEYS:
            bad.append((f"recipe.{p}", a, b))
    return bad


def model_dir_for(R, run):
    """models/sam3_student_tinyvit/<run>/. pigtrack_v1 = the registry key's own directory; a later publishing run
    lies beside it (paths_v1 is immutable and has no key for it; the path is derived from the key, never a literal)."""
    v1_dir = osp.dirname(R.out(MODEL_KEY))
    if osp.basename(v1_dir) != CANONICAL_RUN:
        raise RuntimeError(f"registry {MODEL_KEY} -> {v1_dir}: expected a '{CANONICAL_RUN}' directory")
    if run == CANONICAL_RUN:
        return v1_dir
    if run not in PUBLISH_RUNS:
        raise ValueError(f"run {run!r} does not publish a model")
    return osp.dirname(v1_dir).replace("\\", "/") + "/" + dc.check_run_tag(run)


class _TimmDropPathShim:
    """Stands in for `core.timm` while the verbatim student class is constructed: forwards everything to timm and gives
    the class's own `timm.create_model('tiny_vit_21m_512', ..., drop_path_rate=0.0)` call the requested rate instead.
    Nothing else of the class changes, so the module tree is timm's own construction for that rate."""

    def __init__(self, real, rate):
        self._real, self._rate, self.calls = real, float(rate), 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def create_model(self, name, *a, **kw):
        if name != "tiny_vit_21m_512" or kw.get("drop_path_rate") != 0.0:
            raise RuntimeError(f"unexpected create_model({name!r}, drop_path_rate={kw.get('drop_path_rate')!r}): the "
                               "verbatim class changed; re-check make_student")
        self.calls += 1
        return self._real.create_model(name, *a, **dict(kw, drop_path_rate=self._rate))


def build_with_drop_path(build, mod, rate):
    """build() with `mod.timm` (the bench core's timm global, read by the verbatim class at construction) replaced by
    the shim for a non-zero rate; restored afterwards, also on error. rate 0.0 -> build() untouched."""
    rate = float(rate)
    if rate == 0.0:
        return build()
    real = mod.timm
    shim = _TimmDropPathShim(real, rate)
    mod.timm = shim
    try:
        m = build()
    finally:
        mod.timm = real
    if shim.calls != 1:
        raise RuntimeError(f"build_with_drop_path: {shim.calls} create_model calls through the shim, expected 1")
    return m


# ---- the held-out fidelity table, the FID-E positive control and gate G-M (pure Python)
FID_CSV_KEYS = ("cos_patch", "cos_global", "scale_ratio", "mse")


def student_order(arm):
    """The students of the held-out fidelity, in column order: this run's arm, then S-Edi (the only reference of b1.11, sha-pinned)."""
    return (arm, "S-Edi")


def fidelity_csv(rows, students):
    """per_frame_fidelity.csv (the pigtrack_v1 column layout, one column block per student present)."""
    cols = ["seq", "fi", "night"] + [f"{s}_{p}_{k}" for s in students for p in ("fp32", "bf16") for k in FID_CSV_KEYS]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join([r["seq"], str(r["fi"]), str(r["night"])] + [repr(r[s][p][k]) for s in students
                                                                          for p in ("fp32", "bf16")
                                                                          for k in FID_CSV_KEYS]))
    return "\n".join(lines) + "\n"


def fide_control(rows, val_clip=EDI_VAL[0], stride=5, n_expect=EDI_EXPECT["fide_n"]):
    """Positive control: the fp32 means of S-Edi over the frames fi % stride == 0 of the val clip must reproduce FID-E
    (0.7508 / 0.6922 / 0.9558 / 1.1942; measure_edinburgh_hard_frames reproduced the stored values with a difference of exactly 0.0). Pure function of the
    per-frame rows of prod_fidelity. status: identical (every difference exactly 0.0) | within_4dp (every |difference| <= FIDE_TOL = 1e-4, an
    absolute tolerance on the STORED full-precision values; NOT a comparison of rounded numbers, which could flip on a harmless 1e-5 because the stored
    MSE lies 6.6e-6 above its rounding edge) | differs | incomplete (not all n_expect frames present: a smoke subset)."""
    sel = [r for r in rows if r["seq"] == val_clip and r["fi"] % stride == 0 and "S-Edi" in r]
    out = {"val_clip": val_clip, "stride": stride, "n": len(sel), "n_expected": n_expect, "complete": len(sel) == n_expect,
           "stored": dict(FIDE_VAL), "registered_4dp": dict(FIDE_VAL_4DP), "tolerance": FIDE_TOL, "ours": {}, "diff": {}}
    if not sel:
        out["status"] = "incomplete"
        return out
    for k in FIDE_VAL:
        out["ours"][k] = float(np.mean([r["S-Edi"]["fp32"][k] for r in sel]))
        out["diff"][k] = out["ours"][k] - FIDE_VAL[k]
    if not out["complete"]:
        out["status"] = "incomplete"
    elif not all(math.isfinite(v) for v in out["ours"].values()):
        out["status"] = "differs"
    elif all(d == 0.0 for d in out["diff"].values()):
        out["status"] = "identical"
    elif all(abs(d) <= FIDE_TOL for d in out["diff"].values()):
        out["status"] = "within_4dp"
    else:
        out["status"] = "differs"
    return out


def gate_gm(v):
    """Gate G-M, a PURE function; the notebook only records its result, the controller and submit_edi_guarded.py enforce it.
    ok iff v is a finite real number (bool, str, None, NaN and +-inf refuse) with
    abs(v - 0.25041554232438407) <= 0.1 * 0.25041554232438407, i.e. v in [0.225374, 0.275457]. Returns (ok, record)."""
    real = isinstance(v, (int, float)) and not isinstance(v, bool)
    ok = bool(real and math.isfinite(v) and abs(v - SEDI_EMA_VAL_LOSS) <= GM_FACTOR * SEDI_EMA_VAL_LOSS)
    rec = {"value": (float(v) if real and math.isfinite(v) else repr(v)), "reference_S-Edi_ema_val_loss": SEDI_EMA_VAL_LOSS,
           "factor": GM_FACTOR, "band": [SEDI_EMA_VAL_LOSS - GM_FACTOR * SEDI_EMA_VAL_LOSS, SEDI_EMA_VAL_LOSS + GM_FACTOR * SEDI_EMA_VAL_LOSS],
           "rel_deviation": ((v - SEDI_EMA_VAL_LOSS) / SEDI_EMA_VAL_LOSS if real and math.isfinite(v) else None), "inside_band": ok,
           "note": "recorded by the notebook, enforced by the controller / submit_edi_guarded.py; gross-breakage gate only"}
    return ok, rec


def gate_gm_train_done(td):
    """G-M on a parsed train_done.json: a non-dict, a missing key or a non-numeric value refuses (fail closed)."""
    if not isinstance(td, dict) or "best_ema_val_loss" not in td:
        return False, {"value": None, "inside_band": False, "note": "train_done.json missing or without best_ema_val_loss"}
    return gate_gm(td["best_ema_val_loss"])


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _int(v, name, lo=0):
    x = int(str(v).strip())
    if x < lo:
        raise ValueError(f"{name}={x} < {lo}")
    return x


def parse_params(raw):
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}; known: {list(WIDGETS)}")
    g = {k: str(raw.get(k) if raw.get(k) not in (None, "") else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace()
    P.run = dc.check_run_tag(g["run"])
    P.smoke = dc.is_smoke(P.run)
    P.max_epochs = _int(g["max_epochs"], "max_epochs", 1)
    P.resume_every = _int(g["resume_every"], "resume_every", 1)
    P.seed = _int(g["seed"], "seed")
    P.num_workers = _int(g["num_workers"], "num_workers")
    P.smoke_steps = _int(g["smoke_steps"], "smoke_steps")
    P.smoke_val_batches = _int(g["smoke_val_batches"], "smoke_val_batches")
    P.fid_max_frames = _int(g["fid_max_frames"], "fid_max_frames")
    P.stop_after_epoch = _int(g["stop_after_epoch"], "stop_after_epoch")
    P.stage_threads = _int(g["stage_threads"], "stage_threads", 1)
    reduced = P.smoke_steps or P.smoke_val_batches or P.fid_max_frames
    if reduced and not P.smoke:
        raise ValueError("smoke_steps / smoke_val_batches / fid_max_frames need a run tag starting with 'smoke'")
    if P.smoke and not reduced:
        raise ValueError("a smoke run must set smoke_steps (it would otherwise be a full training run)")
    if not P.smoke and P.run in PUBLISH_RUNS and P.max_epochs != 100:
        raise ValueError("the canonical run uses the recipe's 100-epoch cap")
    try:
        P.drop_path_rate = float(g["drop_path_rate"])
    except ValueError:
        raise ValueError(f"drop_path_rate={g['drop_path_rate']!r}") from None
    if not (0.0 <= P.drop_path_rate <= DROP_PATH_MAX) or not math.isfinite(P.drop_path_rate):
        raise ValueError(f"drop_path_rate={P.drop_path_rate} outside [0, {DROP_PATH_MAX}]")
    if P.run in PUBLISH_RUNS and not P.smoke and P.drop_path_rate != PUBLISH_RUNS[P.run]:
        raise ValueError(f"run={P.run} trains with drop_path_rate={PUBLISH_RUNS[P.run]}, "
                         f"not {P.drop_path_rate}")
    P.eval_stage = g["eval_stage"].lower()
    if P.eval_stage not in EVAL_STAGES:
        raise ValueError(f"eval_stage={P.eval_stage!r}; one of {EVAL_STAGES}")
    pm = g["publish_model"].lower()
    if pm not in ("auto", "true", "false"):
        raise ValueError(f"publish_model={pm!r}")
    P.publish_model = (P.run in PUBLISH_RUNS and not P.smoke) if pm == "auto" else pm == "true"
    if P.publish_model and (P.run not in PUBLISH_RUNS or P.smoke):
        raise ValueError(f"only the runs {sorted(PUBLISH_RUNS)} (not smoke) may publish a model")
    P.require_gpu = g["require_gpu"]
    if P.require_gpu not in ("A100", "A10", "any"):
        raise ValueError(f"require_gpu={P.require_gpu!r}")
    # ---- distill_edinburgh_mined: the three pinned manifests, the checkpoint pin of the fidelity stage, the single full-run rule of the arm
    P.target_set = "mined"
    for k in ("base_manifest_sha256", "extra_manifest_sha256", "summary_sha256", "expect_ckpt_sha256"):
        v = g[k].lower()
        if v and not HEX64.match(v):
            raise ValueError(f"{k} must be 64 hex characters (a sha256) or empty")
        setattr(P, k, v)
    if not (P.base_manifest_sha256 and P.extra_manifest_sha256 and P.summary_sha256):
        raise ValueError("a run needs base_manifest_sha256, extra_manifest_sha256 and summary_sha256 (the sha256 values of "
                         "manifest_base.csv, manifest_mined.csv and SELECTION_SUMMARY.json, filled in by finalize_edi_payload.py)")
    if P.eval_stage == "only" and not P.expect_ckpt_sha256:
        raise ValueError("eval_stage=only needs expect_ckpt_sha256 (the pin of the trained checkpoint)")
    if P.eval_stage != "only" and P.expect_ckpt_sha256:
        raise ValueError("expect_ckpt_sha256 belongs to eval_stage=only (the checkpoint does not exist before the training stage)")
    if not P.smoke:
        if P.run not in ARM_RUNS:
            raise ValueError(f"run={P.run!r}: one of {sorted(ARM_RUNS)} (or a 'smoke...' tag)")
        want_set, want_seed, _arm = ARM_RUNS[P.run]
        if P.seed != want_seed:
            raise ValueError(f"run={P.run} trains with seed {want_seed}, not {P.seed}")
        if P.max_epochs != 100:
            raise ValueError("the b1.11 run uses the recipe's 100-epoch cap")
        if P.drop_path_rate != ARM_DROP_PATH:
            raise ValueError(f"the b1.11 run trains with drop_path_rate={ARM_DROP_PATH} (S-Edi's), not {P.drop_path_rate}")
    P.arm = ARM_RUNS[P.run][2] if P.run in ARM_RUNS else ARM_BY_SET[P.target_set]
    return P


def _read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def build_config(P, data_cfg, versions, gpu_class):
    cfg = {"version": CONFIG_VERSION, "run": P.run, "arm": P.arm, "recipe": recipe_for(P.drop_path_rate),
           "data": data_cfg,
           "training": {"max_epochs": P.max_epochs, "seed": P.seed, "resume_every": P.resume_every,
                        "smoke_steps": P.smoke_steps, "smoke_val_batches": P.smoke_val_batches,
                        "fid_max_frames": P.fid_max_frames},
           "versions": {k: versions.get(k) for k in ("python", "torch", "torchvision", "timm", "numpy", "Pillow",
                                                     "cuda", "cudnn")},
           "gpu_class": gpu_class, "publish_model": P.publish_model, "created_utc": None,
           "extra": {"versions_all": versions}}
    return json.loads(dc.json_dumps(cfg))


def stage_targets(rows, tgt_root, local_root, reuse_root, threads, log=print):
    """{relpath: local path} for every image / feature of the manifest. A extract_teacher_targets_pigtrack local copy (same job cluster) is used
    when its sha256 equals the manifest's; anything else is copied from the Volume and sha-verified."""
    want = {}
    for r in rows:
        for k in ("image", "feature"):
            want[r[k]] = (r[f"{k}_bytes"], r[f"{k}_sha256"])
    out, cands = {}, []
    for rel, (n, _sha) in want.items():
        c = osp.join(reuse_root, rel) if reuse_root else None
        if c and osp.isfile(c) and osp.getsize(c) == n:
            cands.append((rel, c))
    shas = dc.sha256_many([c for _, c in cands], threads=threads) if cands else {}
    reused = 0
    for rel, c in cands:
        if shas[c] == want[rel][1]:
            out[rel] = c
            reused += 1
    jobs = [(f"{tgt_root}/{rel}", osp.join(local_root, rel), n, sha) for rel, (n, sha) in want.items() if rel not in out]
    st = dc.copy_files_parallel(jobs, threads=threads, log=log) if jobs else {"files": 0, "copied": 0,
                                                                             "bytes_copied": 0, "seconds": 0.0}
    for _src, dst, _n, _sha in jobs:
        out[osp.relpath(dst, local_root).replace("\\", "/")] = dst
    if set(out) != set(want):
        raise RuntimeError(f"staging incomplete: {len(set(want) - set(out))} files")
    st.update({"reused_tol66_local": reused, "total": len(want)})
    return out, st


STAGE_MIN_MBPS = 100.0      # a staging slower than this is FLAGGED (stats['slow'] + a log line); no method is switched on a rate (the other path's rate on Volume paths is unmeasured)
STAGE_RATE_MIN_BYTES = 256e6   # a rate is only meaningful (and only flagged) once at least this many bytes were copied
STAGE_METHODS = ("auto", "shutil", "dbutils")


def stage_targets_dbutils(rows, tgt_root, local_root, dbutils, threads, log=print):
    """FALLBACK staging path: dbutils.fs.cp per file (core.stage_file: retried, size- and sha256-verified, one thread per file, a verified local
    copy is kept and not copied again). NOT measured on /Volumes paths with many threads (the 15 MB/s per copy is for 1.3 GB video clips on
    the /dbfs mount), so it is only used when the proven shutil path (`stage_targets`) raised. Returns ({relpath: local path}, stats)."""
    from concurrent.futures import ThreadPoolExecutor
    want = {}
    for r in rows:
        for k in ("image", "feature"):
            want[r[k]] = (r[f"{k}_bytes"], r[f"{k}_sha256"])
    t0 = time.time()

    def one(item):
        rel, (n, sha) = item
        dst = osp.join(local_root, rel)
        had = osp.isfile(dst)
        core.stage_file(f"{tgt_root}/{rel}", dst, dbutils, sha)          # raises on a sha256 mismatch; keeps a verified local copy
        size = osp.getsize(dst)
        if n is not None and size != n:
            raise RuntimeError(f"{rel}: staged {size} B, manifest {n} B")
        return (0 if had else 1), (0 if had else size)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        res = list(ex.map(one, list(want.items())))
    dt = time.time() - t0
    b = sum(r[1] for r in res)
    out = {rel: osp.join(local_root, rel) for rel in want}
    st = {"files": len(want), "copied": sum(r[0] for r in res), "bytes_copied": b, "seconds": round(dt, 1),
          "reused_tol66_local": 0, "total": len(want), "method": "dbutils.fs.cp"}
    return out, st


def stage_targets_fs(rows, tgt_root, local_root, reuse_root, dbutils, threads, log=print, method="auto"):
    """Staging of the manifest's images / features to local disk, TIMED (stats['seconds'] = wall time of this staging alone, failed attempt
    included; stats['mb_per_s'] = bytes copied / that time; stats['slow'] flags a rate below STAGE_MIN_MBPS; stats['method'] names the path that
    delivered the files; stats['fallback'] holds the error that made the primary path give up).
      method 'auto' (production): the PROVEN path first = `stage_targets` (threaded shutil.copyfile over the /Volumes FUSE path, sha256- and
        size-verified; S-PT staged 34.25 GB over this same Volume in 48.6 s = 704 MB/s with it). Only if it RAISES, and a dbutils exists, the files
        still missing are staged by `stage_targets_dbutils` (verified copies are kept); both failing raises. Without dbutils:
        the shutil path alone, its error propagates.
      method 'shutil' / 'dbutils': that path alone (tests)."""
    if method not in STAGE_METHODS:
        raise ValueError(f"stage method {method!r} not in {STAGE_METHODS}")
    if method == "dbutils" and dbutils is None:
        raise ValueError("stage method 'dbutils' needs a dbutils")
    t0 = time.perf_counter()          # high-resolution clock: time.time() ticks in ~16 ms steps on Windows, so a fast stub staging measured 0.0 s
    fallback = None
    if method == "dbutils":
        out, st = stage_targets_dbutils(rows, tgt_root, local_root, dbutils, threads, log)
    else:
        try:
            out, st = stage_targets(rows, tgt_root, local_root, reuse_root, threads, log)
            st["method"] = "shutil"
        except Exception as e:  # noqa: BLE001
            if method == "shutil" or dbutils is None:
                raise
            fallback = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"WARNING stage_targets (shutil) failed ({fallback}); falling back to dbutils.fs.cp for the files still missing")
            out, st = stage_targets_dbutils(rows, tgt_root, local_root, dbutils, threads, log)
    dt = time.perf_counter() - t0
    st["seconds"], st["seconds_exact"] = round(dt, 1), dt
    st["mb_per_s"] = round(st["bytes_copied"] / 1e6 / dt, 1) if dt > 0 and st["bytes_copied"] else None
    st["slow"] = bool(st["mb_per_s"] is not None and st["bytes_copied"] >= STAGE_RATE_MIN_BYTES and st["mb_per_s"] < STAGE_MIN_MBPS)
    if fallback:
        st["fallback"] = fallback
    if st["slow"]:
        log(f"WARNING staging rate {st['mb_per_s']} MB/s is below {STAGE_MIN_MBPS:g} MB/s ({st['method']}; S-PT measured 704 MB/s with shutil)")
    log(f"stage_targets_fs: {st['total']} files, {st['copied']} copied ({st['bytes_copied'] / 1e9:.2f} GB) in {st['seconds']} s via {st['method']}"
        + (f" = {st['mb_per_s']} MB/s" if st["mb_per_s"] else ""))
    return out, st


def stage_union(rows, roots, local_root, reuse_root, threads, log=print, dbutils=None, method="auto"):
    """stage_targets_fs per root: {(root tag, relpath): local path}. Rows carry `_root`. distill_edinburgh_mined: ONE root tag ('base') = the Volume,
    every manifest path is Volume-relative after load_edi_data, so the two target kinds (decoded frames, stored / pool targets) share it."""
    out = {}
    tot = {"files": 0, "copied": 0, "bytes_copied": 0, "seconds": 0.0, "reused_tol66_local": 0, "total": 0}
    exact = 0.0
    for tag, root in roots.items():
        sub = [r for r in rows if r["_root"] == tag]
        if not sub:
            continue
        loc, st = stage_targets_fs(sub, root, osp.join(local_root, "targets" if tag == "base" else "targets_extra"),
                                   reuse_root if tag == "base" else None, dbutils, threads, log, method)
        out.update({(tag, rel): p for rel, p in loc.items()})
        for k in ("files", "copied", "bytes_copied", "reused_tol66_local", "total"):
            tot[k] += st[k]
        exact += st["seconds_exact"]
        tot["seconds"] = round(exact, 1)
        tot["method"] = st["method"]
        if st.get("fallback"):
            tot["fallback"] = st["fallback"]
    tot["mb_per_s"] = round(tot["bytes_copied"] / 1e6 / exact, 1) if exact > 0 and tot["bytes_copied"] else None
    tot["slow"] = bool(tot["mb_per_s"] is not None and tot["bytes_copied"] >= STAGE_RATE_MIN_BYTES and tot["mb_per_s"] < STAGE_MIN_MBPS)
    return out, tot
# ---- distill_edinburgh_mined data layer: the pinned manifests of select_targets_edinburgh, the stored 2026-02 base / val targets, the measure_edinburgh_hard_frames pool targets ---------
def read_pinned(path, pin, what):
    """(text, sha256) of a file read as BYTES (the pins are sha256 values of the file bytes); a non-empty pin that differs raises."""
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    if pin and sha != pin:
        raise RuntimeError(f"{what} sha256 {sha[:16]} != the pinned {pin[:16]}")
    return raw.decode("utf-8"), sha


def canonical_digest(pairs):
    """sha256 of "{clip},{fi}\\n" per row, rows ordered by clip ascending, then by INTEGER fi ascending. Sorting the same
    lines as strings (',10' before ',2') gives another digest (PINS['canon_sha256_M_lexicographic_WRONG']) which is NOT the registered one."""
    rows = sorted((str(c), int(fi)) for c, fi in pairs)
    return hashlib.sha256("".join(f"{c},{fi}\n" for c, fi in rows).encode("utf-8")).hexdigest()


def parse_edi_manifest(text, what="manifest"):
    """Rows of an Edinburgh target manifest. Accepted layouts: the dc layout (distill_core.MANIFEST_COLUMNS, as written by measure_edinburgh_hard_frames) or a
    lean layout with the columns {seq | clip}, fi, image, image_sha256, feature, feature_sha256 (+ optional role, image_bytes,
    feature_bytes). Paths must be relative (no leading '/', no '..', no ':', no backslash); sha256 values 64 lowercase hex; no duplicate
    (clip, fi). Returns [{seq, fi, role (or None), image, image_sha256, feature, feature_sha256, image_bytes, feature_bytes}]."""
    rd = csv.DictReader(io.StringIO(text))
    head = list(rd.fieldnames or [])
    idc = "seq" if "seq" in head else ("clip" if "clip" in head else None)
    need = ("fi", "image", "image_sha256", "feature", "feature_sha256")
    missing = [c for c in need if c not in head]
    if idc is None or missing:
        raise ValueError(f"{what}: columns {head} lack {missing or ['seq|clip']}")
    rows, seen = [], set()
    for r in rd:
        seq, fi = (r[idc] or "").strip(), int(r["fi"])
        if (seq, fi) in seen:
            raise ValueError(f"{what}: duplicate (clip, fi) {(seq, fi)}")
        seen.add((seq, fi))
        row = {"seq": seq, "fi": fi, "role": (r.get("role") or "").strip() or None}
        for k in ("image", "feature"):
            p = (r[k] or "").strip()
            if not p or p.startswith("/") or ".." in p.split("/") or ":" in p or "\\" in p:
                raise ValueError(f"{what}: {k} path must be relative and plain: {p!r}")
            row[k] = p
            s = (r[f"{k}_sha256"] or "").strip()
            if not HEX64.match(s):
                raise ValueError(f"{what}: {(seq, fi)} {k}_sha256 {s!r}")
            row[f"{k}_sha256"] = s
            b = (r.get(f"{k}_bytes") or "").strip()
            row[f"{k}_bytes"] = int(b) if b else None
        rows.append(row)
    return rows


def recorded_sha_for(obj, fname):
    """Every sha256 that a parsed JSON document records for the file `fname`: a key equal to fname (or ending in '/' + fname) whose value
    is a 64-hex string or an object with a 64-hex 'sha256', or an object with a name / file / path / filename field ending in fname and a
    64-hex 'sha256'. Returns a set (empty = nothing recorded). Shared by distill_edinburgh_mined and finalize_edi_payload.py."""
    out = set()

    def hit(v):
        if isinstance(v, str) and HEX64.match(v):
            out.add(v)
        elif isinstance(v, dict) and isinstance(v.get("sha256"), str) and HEX64.match(v["sha256"]):
            out.add(v["sha256"])

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and (k == fname or k.endswith("/" + fname)):
                    hit(v)
            for nk in ("name", "file", "path", "filename"):
                nv = o.get(nk)
                if isinstance(nv, str) and (nv == fname or nv.endswith("/" + fname)) and isinstance(o.get("sha256"), str) \
                        and HEX64.match(o["sha256"]):
                    out.add(o["sha256"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return out


def summary_strings(obj):
    """Every string leaf of a parsed JSON document."""
    out = set()

    def walk(o):
        if isinstance(o, str):
            out.add(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return out


def pairing_problems(rows, pool):
    """Rows whose image and feature do not belong to the same (clip, fi) by the NAMING CONVENTION of the two target sources (the sha256 pins carry
    the files, not which image goes with which feature; an image swapped between two frames of a clip would pass every count and uniqueness check).
    Base / val rows (stored 2026-02 extraction): image .../{clip}/video1/{fi + 1:07d}.jpg, feature .../{clip}/backbone_{fi:06d}.pt.
    Pool rows (measure_edinburgh_hard_frames): image .../frames/{clip}_f{fi:06d}.jpg, feature .../features/{clip}_f{fi:06d}.pt. The clip id must also be a path component of
    both. Returns [(clip, fi, image, feature)] of the offenders."""
    bad = []
    for r in rows:
        seq, fi = r["seq"], r["fi"]
        img_name, feat_name = r["image"].rsplit("/", 1)[-1], r["feature"].rsplit("/", 1)[-1]
        want = (f"{seq}_f{fi:06d}.jpg", f"{seq}_f{fi:06d}.pt") if pool else (f"{fi + 1:07d}.jpg", f"backbone_{fi:06d}.pt")
        in_clip_dir = f"/{seq}/" in "/" + r["image"] and f"/{seq}/" in "/" + r["feature"]
        if (img_name, feat_name) != want or not in_clip_dir:
            bad.append((seq, fi, r["image"], r["feature"]))
    return bad


def check_edi_rows(base_rows, mined_rows, expect=None, texts=()):
    """The b1.11 data assertions on the parsed manifests. base_rows = manifest_base.csv (4 train clips + the val clip); mined_rows =
    manifest_mined.csv (the pool rows, or the 2,400 base rows plus the pool rows; val rows are refused there). Checks: no EDI_TEST id in
    any text or row; every clip id starts with `2019_11_` and has the clip shape; manifest_base.csv holds exactly the 4 train clips + the val
    clip with fi = 0..n_frames-1 each; counts n_base / n_val / n_extra, per pool clip per_clip_extra; no (clip, fi) in two of
    train / val / extra; the canonical digest of the extra rows equals the registered one (rows ordered by clip, INTEGER fi); roles
    consistent where given; the image / feature pairing convention of every row (pairing_problems). Returns (train_rows, val_rows, extra_rows, canon), each
    list sorted by (clip, fi)."""
    ex = dict(EDI_EXPECT, **(expect or {}))
    for name, txt in texts:
        hits = [t for t in EDI_TEST if t in txt]
        if hits:
            raise RuntimeError(f"{name} names the test clip(s) {hits}: no EDI_TEST clip may appear anywhere")
    allr = list(base_rows) + list(mined_rows)
    bad = sorted({r["seq"] for r in allr if not (r["seq"].startswith(CLIP_PREFIX) and CLIP_RX.match(r["seq"]))})
    if bad:
        raise RuntimeError(f"clip ids {bad[:4]} do not start with {CLIP_PREFIX} (every training row must be a November clip)")
    hit = sorted({r["seq"] for r in allr} & set(EDI_TEST))
    if hit:
        raise RuntimeError(f"rows of the test clip(s) {hit}")
    train_clips, val_clips, pool = tuple(ex["train_clips"]), tuple(ex["val_clips"]), tuple(ex["pool_clips"])
    nf = ex["n_frames"]
    by = {}
    for r in base_rows:
        by.setdefault(r["seq"], []).append(r)
    if set(by) != set(train_clips) | set(val_clips):
        raise RuntimeError(f"manifest_base.csv clips {sorted(by)} != the 4 base clips + the val clip {sorted(set(train_clips) | set(val_clips))}")
    for c, rs in by.items():
        if sorted(r["fi"] for r in rs) != list(range(nf)):
            raise RuntimeError(f"manifest_base.csv {c}: frames are not 0..{nf - 1}")
        roles = {r["role"] for r in rs if r["role"]}
        want_role = "val" if c in val_clips else "train"
        if roles and roles != {want_role}:
            raise RuntimeError(f"manifest_base.csv {c}: role {sorted(roles)}, expected {want_role}")
    train_rows = [r for r in base_rows if r["seq"] in train_clips]
    val_rows = [r for r in base_rows if r["seq"] in val_clips]
    if len(train_rows) != ex["n_base"] or len(val_rows) != ex["n_val"]:
        raise RuntimeError(f"base {len(train_rows)} / val {len(val_rows)} rows, expected {ex['n_base']} / {ex['n_val']}")
    extra_rows = [r for r in mined_rows if r["seq"] in pool]
    others = [r for r in mined_rows if r["seq"] not in pool]
    if any(r["seq"] in val_clips for r in others):
        raise RuntimeError("manifest_mined.csv holds validation rows (the val rows stay in manifest_base.csv)")
    if others:
        if any(r["seq"] not in train_clips for r in others):
            raise RuntimeError("manifest_mined.csv holds a clip that is neither a pool clip nor a base clip")
        fields = ("seq", "fi", "image", "image_sha256", "feature", "feature_sha256")
        if sorted(tuple(r[f] for f in fields) for r in others) != sorted(tuple(r[f] for f in fields) for r in train_rows):
            raise RuntimeError("the base rows of manifest_mined.csv differ from the train rows of manifest_base.csv")
    if len(extra_rows) != ex["n_extra"]:
        raise RuntimeError(f"manifest_mined.csv holds {len(extra_rows)} pool rows, expected {ex['n_extra']}")
    per = collections.Counter(r["seq"] for r in extra_rows)
    if set(per) != set(pool) or any(n != ex["per_clip_extra"] for n in per.values()):
        raise RuntimeError(f"pool rows per clip {dict(per)}, expected {ex['per_clip_extra']} for each of {list(pool)}")
    if any(not 0 <= r["fi"] < nf for r in extra_rows):
        raise RuntimeError("a pool row has a frame index outside the clip")
    keys = [{(r["seq"], r["fi"]) for r in rs} for rs in (train_rows, val_rows, extra_rows)]
    if sum(len(k) for k in keys) != len(keys[0] | keys[1] | keys[2]):
        raise RuntimeError("a (clip, fi) occurs in two of base / val / extra")
    canon = canonical_digest((r["seq"], r["fi"]) for r in extra_rows)
    if canon != ex["canon"]:
        raise RuntimeError(f"the pool rows have canonical digest {canon[:16]}, the registered set M has {ex['canon'][:16]}"
                           + (" (the lexicographic, known-wrong ordering)" if canon == PINS["canon_sha256_M_lexicographic_WRONG"] else ""))
    sample = ex.get("sample_feature")
    if sample:
        got = [r for r in train_rows if (r["seq"], r["fi"]) == (sample[0], sample[1])]
        if len(got) != 1 or got[0]["feature_sha256"] != sample[2]:
            raise RuntimeError(f"the stored target {sample[0]}/backbone_{sample[1]:06d}.pt is not the recorded sample (sha256 {sample[2][:16]})")
    for what, rs, pool_rows in (("train", train_rows, False), ("val", val_rows, False), ("pool", extra_rows, True)):
        bad = pairing_problems(rs, pool_rows)
        if bad:
            raise RuntimeError(f"{len(bad)} {what} rows pair an image and a feature of different frames (naming convention; e.g. {bad[:2]})")
    key = lambda r: (r["seq"], r["fi"])      # noqa: E731
    return sorted(train_rows, key=key), sorted(val_rows, key=key), sorted(extra_rows, key=key), canon


def crosscheck_unused(extra_rows, unused_text):
    """Every pool row must be a row of measure_edinburgh_hard_frames's teacher_targets_unused_v1/manifest.csv with the same image and feature sha256 (content
    identity, independent of the path form)."""
    by = {(u["seq"], u["fi"]): u for u in dc.read_manifest_csv(unused_text)}
    miss = [(r["seq"], r["fi"]) for r in extra_rows if (r["seq"], r["fi"]) not in by]
    if miss:
        raise RuntimeError(f"{len(miss)} pool rows are not in teacher_targets_unused_v1/manifest.csv (e.g. {miss[:3]})")
    diff = [(r["seq"], r["fi"]) for r in extra_rows
            if by[(r["seq"], r["fi"])]["image_sha256"] != r["image_sha256"] or by[(r["seq"], r["fi"])]["feature_sha256"] != r["feature_sha256"]]
    if diff:
        raise RuntimeError(f"{len(diff)} pool rows differ in sha256 from teacher_targets_unused_v1/manifest.csv (e.g. {diff[:3]})")
    return len(extra_rows)


def selection_file_text(rows):
    """The registered selection file of M: header + one line per row ordered by clip, then INTEGER fi, LF line
    ends, image / feature paths relative to teacher_targets_unused_v1/. Rebuilt from the pool rows so that the paths, the per-file sha256 values
    and the (clip, fi) set are all pinned by ONE registered digest."""
    lines = ["clip,fi,image,image_sha256,feature,feature_sha256"]
    for r in sorted(rows, key=lambda r: (r["seq"], r["fi"])):
        lines.append(f"{r['seq']},{r['fi']},{r['image']},{r['image_sha256']},{r['feature']},{r['feature_sha256']}")
    return chr(10).join(lines) + chr(10)


def pick_root(cands, rels):
    """The first candidate root under which a sample of `rels` (first, last and up to 24 evenly spaced) exists as files; raises if none."""
    n = len(rels)
    if not n:
        raise RuntimeError("pick_root: no paths")
    sample = [rels[i] for i in sorted({0, n - 1} | {int(i * (n - 1) / 23) for i in range(24)})]
    counts = {}
    for c in cands:
        counts[c] = sum(1 for p in sample if osp.isfile(f"{c}/{p}"))
        if counts[c] == len(sample):
            return c
    raise RuntimeError(f"no candidate root holds the manifest's files (sample of {len(sample)}): {counts}")


def vol_rel(path, volume):
    """The Volume-relative form of an absolute path under `volume` (raises for anything outside it)."""
    p, v = str(path).replace("\\", "/"), str(volume).replace("\\", "/").rstrip("/")
    if not p.startswith(v + "/"):
        raise RuntimeError(f"{path} is not under the Volume {volume}")
    return p[len(v) + 1:]


def load_edi_data(P, sel_dir, unused_dir, tgt_root, frames_root, volume, h):
    """Read, pin and check the three select_targets_edinburgh outputs and measure_edinburgh_hard_frames's unused-target manifest; resolve every row to a Volume-relative image and
    feature path. Returns {rows, n_train, n_val, info, cfg_data}; rows carry role, _root='base', night=0, image / feature (Volume-relative),
    *_bytes, *_sha256. Row order: train (clip, fi), then the pool rows (clip, fi), then val (clip, fi)."""
    ex = dict(EDI_EXPECT, **(h.get("expect") or {}))
    base_text, base_sha = read_pinned(f"{sel_dir}/manifest_base.csv", P.base_manifest_sha256, "manifest_base.csv")
    mined_text, mined_sha = read_pinned(f"{sel_dir}/manifest_mined.csv", P.extra_manifest_sha256, "manifest_mined.csv")
    summ_text, summ_sha = read_pinned(f"{sel_dir}/SELECTION_SUMMARY.json", P.summary_sha256, "SELECTION_SUMMARY.json")
    unused_text, unused_sha = read_pinned(f"{unused_dir}/manifest.csv", ex["unused_sha"], "teacher_targets_unused_v1/manifest.csv")
    summary = json.loads(summ_text)
    rr_p = f"{sel_dir}/RUN_RECORD.json"          # select_targets_edinburgh's run record: it records the sha256 of every file it wrote (read unpinned: a cross-check only)
    run_record = json.loads(_read(rr_p)) if osp.isfile(rr_p) else {}
    for fname, sha in (("manifest_base.csv", base_sha), ("manifest_mined.csv", mined_sha), ("SELECTION_SUMMARY.json", summ_sha)):
        rec = recorded_sha_for(summary, fname) | recorded_sha_for(run_record, fname)
        if rec != {sha} and not (fname == "SELECTION_SUMMARY.json" and not rec):
            raise RuntimeError(f"SELECTION_SUMMARY.json / RUN_RECORD.json record {sorted(x[:16] for x in rec) or 'no sha256'} for {fname}; "
                               f"the pinned file is {sha[:16]}")
    if ex["canon"] not in summary_strings(summary):
        raise RuntimeError("SELECTION_SUMMARY.json does not record the registered canonical digest of M")
    want_fields = {"M": ex["n_extra"], "q_per_clip": ex["per_clip_extra"], "pool_clips": list(ex["pool_clips"]),
                   "training_total": ex["n_base"] + ex["n_extra"]}
    wrong = {k: summary[k] for k, v in want_fields.items() if k in summary and summary[k] != v}
    if "counts" in summary and summary["counts"] != {"train": ex["n_base"], "val": ex["n_val"], "extra": ex["n_extra"]}:
        wrong["counts"] = summary["counts"]
    if wrong:
        raise RuntimeError(f"SELECTION_SUMMARY.json disagrees with the registered design: {wrong}")
    train_rows, val_rows, extra_rows, canon = check_edi_rows(
        parse_edi_manifest(base_text, "manifest_base.csv"), parse_edi_manifest(mined_text, "manifest_mined.csv"), ex,
        texts=(("manifest_base.csv", base_text), ("manifest_mined.csv", mined_text), ("SELECTION_SUMMARY.json", summ_text),
               ("teacher_targets_unused_v1/manifest.csv", unused_text)))
    crosscheck_unused(extra_rows, unused_text)

    def to_vol(rs, role, img_roots, feat_roots):
        ri = pick_root(img_roots, [r["image"] for r in rs])
        rf = pick_root(feat_roots, [r["feature"] for r in rs])
        return [dict(r, role=role, image=vol_rel(f"{ri}/{r['image']}", volume), feature=vol_rel(f"{rf}/{r['feature']}", volume),
                     _root="base", night=0) for r in rs]

    rows_train = to_vol(train_rows, "train", [frames_root, volume], [tgt_root, volume])
    rows_val = to_vol(val_rows, "val", [frames_root, volume], [tgt_root, volume])
    rows_extra = to_vol(extra_rows, "train", [unused_dir, volume], [unused_dir, volume])
    urel = vol_rel(unused_dir, volume) + "/"
    if not all(r["image"].startswith(urel) and r["feature"].startswith(urel) for r in rows_extra):
        raise RuntimeError("the pool rows do not lie under teacher_targets_unused_v1/")
    sel_text = selection_file_text([dict(r, image=r["image"][len(urel):], feature=r["feature"][len(urel):]) for r in rows_extra])
    sel_sha = hashlib.sha256(sel_text.encode("utf-8")).hexdigest()
    sel_rec = (summary.get("selection_file") or {}).get("sha256")
    if sel_sha != ex["selection_sha"] or (sel_rec is not None and sel_rec != sel_sha):
        raise RuntimeError(f"the pool rows reproduce the selection file {sel_sha[:16]}, the registered one is {ex['selection_sha'][:16]}"
                           + (f" (the summary records {sel_rec[:16]})" if sel_rec is not None else ""))
    rows = rows_train + rows_extra + rows_val
    if len({(r["_root"], r["image"]) for r in rows}) != len(rows) or len({(r["_root"], r["feature"]) for r in rows}) != len(rows):
        raise RuntimeError("two rows share an image or a feature file")
    per = collections.Counter(r["seq"] for r in rows_extra)
    info = {"dir": "/".join(SELECTION_PARTS), "pool": PINS["pool"], "q_per_clip": ex["per_clip_extra"], "n_base": len(rows_train),
            "n_extra": len(rows_extra), "n_val": len(rows_val), "per_clip_extra": dict(sorted(per.items())),
            "base_manifest_sha256": base_sha, "extra_manifest_sha256": mined_sha, "summary_sha256": summ_sha,
            "unused_manifest_sha256": unused_sha, "canon_sha256": canon, "selection_file_sha256": sel_sha,
            "mined_manifest_holds_base_rows": len(parse_edi_manifest(mined_text, "manifest_mined.csv")) != len(rows_extra)}
    cfg_data = {"targets_key": TARGETS_KEY, "frames_key": FRAMES_KEY, "n_train": len(rows_train) + len(rows_extra), "n_val": len(rows_val),
                "val_seqs": sorted(ex["val_clips"]), "extra": info, "target_sources": TARGET_SOURCES}
    return {"rows": rows, "n_train": cfg_data["n_train"], "n_val": cfg_data["n_val"], "info": info, "cfg_data": cfg_data}


def ckpt_pin_diffs(td, pin):
    """eval_stage=only pin check: train_done.json must name the registered checkpoint file and its sha256 must equal the widget pin."""
    bad = []
    if td.get("ckpt") != CKPT_NAME:
        bad.append(("ckpt", CKPT_NAME, td.get("ckpt")))
    if not str(td.get("final_dir", "")).startswith("final_"):
        bad.append(("final_dir", "final_<rid>", td.get("final_dir")))
    if td.get("ckpt_sha256") != pin:
        bad.append(("ckpt_sha256", pin, td.get("ckpt_sha256")))
    return bad


def spt_reference_diffs(ref_cfg, cfg, expected_seed, smoke=False):
    """b1.11 recipe gate: a new student's DISTILL_CONFIG equals pigtrack_v2_dp01's (S-PT; its recipe is S-Edi's algorithm, INVENTORY 3.3) in
    the whole `recipe` block key by key (fatal) and in `training` except `seed` (fatal; `seed` must equal expected_seed, 42), and uses the
    same timm version (versions gate, fatal). `data` is NOT compared: Edinburgh and PigTrack data differ by design (the Edinburgh data
    identity is checked by load_edi_data). Other `versions` / extra.versions_all differences are not fatal: they are returned to be
    recorded. smoke=True: the `training` keys in SMOKE_TRAINING_KEYS are not compared (a smoke run shortens the schedule by
    construction); recipe, seed and versions are checked exactly as in a full run. Returns (fatal diffs, recorded diffs)."""
    bad = []
    for p, a, b in core.config_diff(ref_cfg.get("recipe"), cfg.get("recipe"), ()):
        bad.append((f"recipe.{p}" if p else "recipe", a, b))
    for p, a, b in core.config_diff(ref_cfg.get("training"), cfg.get("training"), ()):
        if p != "seed" and not (smoke and p.split(".")[0] in SMOKE_TRAINING_KEYS):
            bad.append((f"training.{p}", a, b))
    if cfg["training"]["seed"] != expected_seed:
        bad.append(("training.seed", expected_seed, cfg["training"]["seed"]))
    ref_timm, cur_timm = (ref_cfg.get("versions") or {}).get("timm"), (cfg.get("versions") or {}).get("timm")
    if ref_timm != cur_timm:
        bad.append(("versions.timm", ref_timm, cur_timm))
    recorded = [(f"versions.{p}", a, b) for p, a, b in core.config_diff(ref_cfg.get("versions"), cfg.get("versions"), ())]
    recorded += [(f"extra.versions_all.{p}", a, b) for p, a, b in
                 core.config_diff((ref_cfg.get("extra") or {}).get("versions_all"), (cfg.get("extra") or {}).get("versions_all"), ())]
    return bad, recorded


def resume_listing(run_dir):
    d = f"{run_dir}/resume"
    if not osp.isdir(d):
        return d, [], {}, {}
    names = sorted(os.listdir(d))
    sizes = {n: osp.getsize(f"{d}/{n}") for n in names if n.endswith(".pt")}
    marks = {}
    for n in names:
        if n.endswith(".json"):
            try:
                marks[n] = json.loads(_read(f"{d}/{n}"))
            except Exception:  # noqa: BLE001
                marks[n] = {}
    return d, names, sizes, marks


def student_labels(P):
    return {P.arm: f"{P.run}: this run's student ({PINS['n_base']:,} base targets + {PINS['M']} pool targets of the set M; "
                   f"drop_path_rate={P.drop_path_rate:g} in training; seed {P.seed})",
            "S-Edi": "published Edinburgh student (models.student_edinburgh, EMA epoch 75, sha256 9686b4c0...), read-only reference"}


def main(raw, *, hooks):
    h = dict(hooks)
    log = h.get("log", _log)
    dbu = h.get("dbutils")
    rid = h.get("rid", RID)
    t_run = time.time()
    P = parse_params(raw)
    if recipe_for(0.0) != RECIPE or hashlib.sha256(dc.json_dumps(RECIPE).encode("utf-8")).hexdigest() != RECIPE_V1_SHA256:
        raise RuntimeError("RECIPE no longer equals pigtrack_v1's recorded recipe (RECIPE_V1_SHA256)")
    R = h.get("registry") or farm_paths.load(version=REGISTRY_VERSION)
    TGT = R.path(TARGETS_KEY)
    FRAMES = R.path(FRAMES_KEY)
    SEL = R.path("paper1.root", None, *SELECTION_PARTS)
    UNUSED = R.path("paper1.root", None, *UNUSED_PARTS)
    RUN_DIR = R.out("paper1.distillation_runs", P.run)
    MODEL_DIR = model_dir_for(R, P.run) if P.publish_model else None
    model_dst = f"{MODEL_DIR}/{CKPT_NAME}" if MODEL_DIR else None
    LOCAL = osp.join(h["local_root"], P.run)
    os.makedirs(LOCAL, exist_ok=True)

    def pub(local_path, rel):
        return core.publish_no_overwrite(local_path, f"{RUN_DIR}/{rel}", dbu)

    def pub_json(obj, rel):
        lp = core.write_text_atomic(osp.join(LOCAL, "pub", rel), dc.json_dumps(obj))
        return pub(lp, rel)

    # ---- data identity: the pinned select_targets_edinburgh manifests + the stored 2026-02 base / val targets + the measure_edinburgh_hard_frames pool targets
    data = load_edi_data(P, SEL, UNUSED, TGT, FRAMES, R.volume, h)
    rows, extra_info = data["rows"], data["info"]
    summ = {"train": data["n_train"], "val": data["n_val"], "val_seqs": list(EDI_VAL)}
    log(f"data: {extra_info['n_base']} base + {extra_info['n_extra']} pool = {summ['train']} train, {summ['val']} val targets "
        f"(canonical digest of M {extra_info['canon_sha256'][:12]}; manifests {extra_info['base_manifest_sha256'][:12]} / "
        f"{extra_info['extra_manifest_sha256'][:12]})")
    gpu = h["gpu_preflight"](P)
    td_p = f"{RUN_DIR}/train_done.json"
    ed_p = f"{RUN_DIR}/eval_done.json"
    need_train = not osp.isfile(td_p)
    need_eval = P.eval_stage != "defer" and not osp.isfile(ed_p)
    if need_train and P.eval_stage == "only":
        raise RuntimeError(f"eval_stage=only: {P.run} has no train_done.json (train it first)")
    if P.eval_stage == "only":                  # b1.11: the checkpoint this evaluation reads is the pinned one
        pin_bad = ckpt_pin_diffs(json.loads(_read(td_p)), P.expect_ckpt_sha256)
        if pin_bad:
            raise RuntimeError("eval_stage=only: train_done.json does not match the pinned checkpoint: "
                               + "; ".join(f"{k}: {a!r} -> {b!r}" for k, a, b in pin_bad))
    if need_train and P.run in ARM_RUNS and gpu["class"] != "A100":
        raise RuntimeError(f"{P.run} trains on the A100 (recipe), not {gpu['class']}")
    cfg = build_config(P, data["cfg_data"], h["versions"](), gpu["class"])
    spt_ref = None
    if P.run in ARM_RUNS or P.smoke:        # a smoke run runs the gate too (schedule keys excepted): first live read of S-PT's config
        ref_p = R.path("paper1.distillation_runs", None, REF_RUN, "DISTILL_CONFIG.json")
        ref_cfg = json.loads(_read(ref_p))
        bad, recorded = spt_reference_diffs(ref_cfg, cfg, ARM_SEED_BY_SET[P.target_set], smoke=P.smoke)
        if bad:
            raise RuntimeError(f"{P.run} is not {REF_RUN}'s recipe / training / timm: "
                               + "; ".join(f"{p}: {a!r} -> {b!r}" for p, a, b in bad[:12]))
        spt_ref = {"run": REF_RUN, "config_sha256": hashlib.sha256(_read(ref_p).encode("utf-8")).hexdigest(),
                   "version_diffs": [list(d) for d in recorded]}
        cfg.setdefault("extra", {})["spt_reference"] = spt_ref
        log(f"gate vs {REF_RUN}{' (smoke: schedule keys not compared)' if P.smoke else ''}: recipe block equal key by key, training "
            f"equal except seed ({P.seed}), timm equal (data not compared: Edinburgh vs PigTrack); {len(recorded)} version difference(s) recorded: {recorded[:4]}")
    cfg_p = f"{RUN_DIR}/DISTILL_CONFIG.json"
    if osp.isfile(cfg_p):
        stored = json.loads(_read(cfg_p))
        if P.eval_stage == "only":
            d = core.config_diff(stored, cfg, tuple(core.CONFIG_VOLATILE) + EVAL_ONLY_IGNORE)
            if d:
                raise RuntimeError("DISTILL_CONFIG mismatch (eval_stage=only): "
                                   + "; ".join(f"{p}: {a!r} -> {b!r}" for p, a, b in d[:12]))
            if (stored.get("versions") or {}).get("timm") != cfg["versions"].get("timm"):
                raise RuntimeError(f"eval_stage=only: timm {cfg['versions'].get('timm')} != the training run's "
                                   f"{(stored.get('versions') or {}).get('timm')}")
            log(f"DISTILL_CONFIG matches (eval_stage=only on {gpu['class']}; trained on {stored.get('gpu_class')})")
        else:
            core.config_check(stored, cfg)
            log("DISTILL_CONFIG.json matches: resuming this run")
        cfg = stored
    elif P.eval_stage == "only":
        raise RuntimeError(f"eval_stage=only: {P.run} has no DISTILL_CONFIG.json (train it first)")
    else:
        cfg = dict(cfg, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        pub_json(cfg, "DISTILL_CONFIG.json")
    eval_env = {"gpu": gpu, "versions": h["versions"](), "eval_stage": P.eval_stage, "nb": NB_VERSION,
                "training_gpu_class": cfg.get("gpu_class")}
    runlog = {"nb": NB_VERSION, "rid": rid, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "gpu": gpu, "code": h["code_shas"](), "params": vars(P), "spt_reference": spt_ref}
    if osp.isfile(f"{RUN_DIR}/DONE.json"):
        done = json.loads(_read(f"{RUN_DIR}/DONE.json"))
        log(f"{P.run}: DONE.json present -- nothing to do")
        return dict(done, skipped=True)

    # ---- stage the targets (only what the stages still to run need: all rows to train, the val rows to evaluate)
    stage_rows = rows if need_train else ([r for r in rows if r["role"] == "val"] if need_eval else [])
    local_of = {}
    if stage_rows:
        t_stage = time.time()
        local_of, st = stage_union(stage_rows, {"base": R.volume}, LOCAL, h.get("reuse_root"), P.stage_threads, log, dbutils=dbu)
        st["wall_seconds"] = round(time.time() - t_stage, 1)          # staging is timed apart from the compute
        runlog["staging"] = st
        log(f"staged {st['total']} files ({st.get('method', 'shutil')}): {st['copied']} copied ({st['bytes_copied'] / 1e9:.1f} GB) in "
            f"{st['wall_seconds']} s wall = {st.get('mb_per_s')} MB/s{' [slow]' if st.get('slow') else ''}"
            f"{' [FALLBACK after: ' + st['fallback'] + ']' if st.get('fallback') else ''} (staging time is not training time)")

    def pairs(role):
        return [{"image": local_of[(r["_root"], r["image"])], "feature": local_of[(r["_root"], r["feature"])],
                 "seq": r["seq"], "fi": r["fi"], "night": r["night"]} for r in rows if r["role"] == role]

    ctx = {"P": P, "cfg": cfg, "rid": rid, "log": log, "local": LOCAL, "pub": pub, "pub_json": pub_json,
           "run_dir": RUN_DIR, "R": R, "dbutils": dbu, "gpu": gpu}

    # ---- train (or reuse a finished training)
    if need_train:
        rdir, names, sizes, marks = resume_listing(RUN_DIR)
        ep, name = dc.latest_state(names, sizes, marks)
        resume_local = None
        if name:
            resume_local = osp.join(LOCAL, "resume", name)
            core.stage_file(f"{rdir}/{name}", resume_local, dbu, marks[dc.state_marker(name)]["sha256"])
            log(f"resuming from {name} (epoch {ep})")
        res = h["train"](ctx, pairs("train"), pairs("val"), resume_local)
        runlog["train"] = {k: v for k, v in res.items() if k not in ("history", "best_local")}
        if res["stopped"] == "stop_after_epoch":
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            pub_json(dict(runlog, seconds=round(time.time() - t_run, 1)), f"runs/{stamp}_{rid}.json")
            log(f"stopped after epoch {res['last_epoch']} as asked (stop_after_epoch); re-run to continue")
            return {"status": "stopped_after_epoch", "last_epoch": res["last_epoch"], "run_dir": RUN_DIR}
        hist_local = core.write_text_atomic(osp.join(LOCAL, f"final_{rid}", HISTORY_NAME), dc.json_dumps(res["history"]))
        pub(res["best_local"], f"final_{rid}/{CKPT_NAME}")
        pub(hist_local, f"final_{rid}/{HISTORY_NAME}")
        best = res["best_metrics"]
        td = {"final_dir": f"final_{rid}", "ckpt": CKPT_NAME, "history": HISTORY_NAME,
              "ckpt_sha256": core.sha256_file(res["best_local"]), "history_sha256": core.sha256_file(hist_local),
              "best_epoch": best["epoch"], "best_ema_val_loss": best["ema_val_loss"],
              "epochs_run": len(res["history"]), "stopped": res["stopped"],
              "train_seconds": round(sum(e.get("sec", 0.0) for e in res["history"]), 1),
              "resumed_from": res.get("resumed_from"), "rid": rid, "nb": NB_VERSION,
              "drop_path_rate": P.drop_path_rate, "droppath_checks": res.get("droppath_checks"),
              "gpu_class": gpu["class"], "gate_gm": gate_gm(best["ema_val_loss"])[1],
              "staging_seconds": runlog.get("staging", {}).get("wall_seconds")}
        pub_json(td, "train_done.json")
    td = json.loads(_read(td_p))
    best_local = osp.join(LOCAL, td["final_dir"], td["ckpt"])
    core.stage_file(f"{RUN_DIR}/{td['final_dir']}/{td['ckpt']}", best_local, dbu, td["ckpt_sha256"])
    hist_local = osp.join(LOCAL, td["final_dir"], td["history"])
    core.stage_file(f"{RUN_DIR}/{td['final_dir']}/{td['history']}", hist_local, dbu, td["history_sha256"])
    history = json.loads(_read(hist_local))

    def publish_model(fid_local=None):
        """The model release (publishing runs only) into models/sam3_student_tinyvit/<run>/, never overwriting; then
        the bench's own loader + gates on the released file."""
        cfg_local = osp.join(LOCAL, "DISTILL_CONFIG.json")
        core.stage_file(cfg_p, cfg_local, dbu)
        out = {}
        for src, name in ((best_local, CKPT_NAME), (hist_local, HISTORY_NAME), (fid_local, "fidelity_heldout.json"),
                          (cfg_local, "DISTILL_CONFIG.json")):
            if src is not None:
                out[name] = dc.publish_file_guarded(src, f"{MODEL_DIR}/{name}", MODEL_DIR, dbu)
        if core.sha256_file(model_dst) != td["ckpt_sha256"]:
            raise RuntimeError(f"{model_dst} does not hold the trained checkpoint")
        h["verify_published"](model_dst, ctx)
        return out

    model_dir_rel = (MODEL_DIR[len(R.volume) + 1:] if MODEL_DIR and MODEL_DIR.startswith(R.volume + "/")
                     else MODEL_DIR)
    if P.eval_stage == "defer":
        published = publish_model() if P.publish_model else {}
        h["verify_published"](best_local, ctx)          # b1.11: bench class load + GroupNorm gate; train_done.json is already published
        info = {"run": P.run, "rid": rid, "ckpt_sha256": td["ckpt_sha256"], "best_epoch": td["best_epoch"],
                "epochs_run": td["epochs_run"], "stopped": td["stopped"], "train_seconds": td["train_seconds"],
                "drop_path_rate": P.drop_path_rate, "model_published": published, "model_dir": model_dir_rel,
                "loads_in_bench_class": True, "gate_gm": td.get("gate_gm")}
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        pub_json(dict(runlog, seconds=round(time.time() - t_run, 1), deferred=info), f"runs/{stamp}_{rid}.json")
        log(f"{P.run}: trained (best epoch {td['best_epoch']} of {td['epochs_run']}); fidelity deferred "
            f"(eval_stage=only on the A10)")
        return dict(info, status="trained_eval_deferred", run_dir=RUN_DIR)

    # ---- held-out fidelity (EMA best checkpoint + S-Edi on the same val frames; the FID-E positive control is inside the fidelity JSON)
    if need_eval:
        val = pairs("val")
        if P.fid_max_frames:
            val = val[:P.fid_max_frames]
        fid, csv_text = h["fidelity"](ctx, best_local, val)
        best_h = next(e for e in history if e["epoch"] == td["best_epoch"])
        fid["recipe_val_at_best_epoch"] = {
            "epoch": td["best_epoch"], "ema_val_loss": best_h["ema_val_loss"], "ema_cosine_loss": best_h["ema_cosine"],
            "one_minus_ema_cosine_loss": 1.0 - best_h["ema_cosine"],
            "note": "the training loop's own EMA val numbers (fp16 autocast, batch-of-4 means): comparable with the "
                    "Edinburgh history's val cosine-embedding loss 0.2493 (per-patch cos ~0.751) for S-Edi"}
        spt = fid["summary"][P.arm]["fp32"]["overall"]
        fid["bands"] = {"cos_patch_ok": spt["cos_patch"]["mean"] >= FID_BANDS["cos_patch_min"],
                        "scale_ok": FID_BANDS["scale_ratio"][0] <= spt["scale_ratio"]["mean"] <= FID_BANDS["scale_ratio"][1],
                        "bands": FID_BANDS, "note": "distill_student's a-priori bands; informative, no decision rests on them"}
        fid["training"] = {k: td[k] for k in ("best_epoch", "best_ema_val_loss", "epochs_run", "stopped",
                                              "train_seconds", "ckpt_sha256")}
        fid["labels"] = student_labels(P)
        fid["drop_path_rate_in_training"] = {P.arm: P.drop_path_rate,
                                             "S-Edi": f"timm default ({TIMM_DEFAULT_DROP_PATH}, export did not set it)"}
        fid["gate_gm"] = td.get("gate_gm")
        fid["pins"] = dict(PINS)
        fid["eval_env"] = eval_env
        fl = core.write_text_atomic(osp.join(LOCAL, f"eval_{rid}", "fidelity_heldout.json"), dc.json_dumps(fid))
        cl = core.write_text_atomic(osp.join(LOCAL, f"eval_{rid}", "per_frame_fidelity.csv"), csv_text)
        pub(fl, f"eval_{rid}/fidelity_heldout.json")
        pub(cl, f"eval_{rid}/per_frame_fidelity.csv")
        pub_json({"eval_dir": f"eval_{rid}", "fidelity_sha256": core.sha256_file(fl),
                  "csv_sha256": core.sha256_file(cl), "n_frames": fid["n_frames"], "rid": rid,
                  "gpu_class": gpu["class"], "eval_stage": P.eval_stage}, "eval_done.json")
    ed = json.loads(_read(ed_p))
    fid_local = osp.join(LOCAL, ed["eval_dir"], "fidelity_heldout.json")
    core.stage_file(f"{RUN_DIR}/{ed['eval_dir']}/fidelity_heldout.json", fid_local, dbu, ed["fidelity_sha256"])
    fid = json.loads(_read(fid_local))

    # ---- the model release (publishing runs only), never overwriting
    published = publish_model(fid_local) if P.publish_model else {}

    def heads(st, grp="overall"):      # None for an empty group (a smoke subset may hold no night frame)
        return {pr: ({k: st[pr][grp][k]["mean"] for k in ("cos_patch", "cos_global", "scale_ratio", "mse")}
                     if st[pr][grp].get("n") else None) for pr in st}

    s = fid["summary"][P.arm]
    ref = fid["summary"].get("S-Edi", {})
    done = {"run": P.run, "rid": rid, "ckpt_sha256": td["ckpt_sha256"], "best_epoch": td["best_epoch"],
            "epochs_run": td["epochs_run"], "stopped": td["stopped"], "train_seconds": td["train_seconds"],
            "heldout": {pr: {k: s[pr]["overall"][k]["mean"] for k in ("cos_patch", "cos_global", "scale_ratio", "mse")}
                        for pr in ("fp32", "bf16")},
            "heldout_S-Edi": {pr: {k: ref[pr]["overall"][k]["mean"] for k in ("cos_patch", "cos_global", "scale_ratio",
                                                                               "mse")} for pr in ref},
            "n_val_frames": fid["n_frames"], "model_published": published,
            "model_path_registry_key": MODEL_KEY if (P.publish_model and P.run == CANONICAL_RUN) else None,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if P.run != CANONICAL_RUN:      # distill_edinburgh_mined: the b1.11 fields (the arm, G-M, the checkpoint pin, the positive control)
        done.update({"drop_path_rate": P.drop_path_rate, "model_dir": model_dir_rel,
                     "eval_gpu_class": ed.get("gpu_class"),
                     "heldout_by_light": {st: {grp: heads(fid["summary"][st], grp) for grp in ("day", "night")}
                                          for st in student_order(P.arm) if st in fid["summary"]}})
        done["arm"] = P.arm
        done["gate_gm"] = td.get("gate_gm")
        done["ckpt_pin_checked"] = P.eval_stage == "only"
        pc = fid.get("positive_control") or {}
        done["positive_control_status"] = pc.get("status")
        if not P.smoke and pc.get("status") not in FIDE_PASS:
            raise core.BenchAssertError(f"positive control failed: S-Edi on the stride-5 val subset does not reproduce FID-E "
                                        f"(status {pc.get('status')}, diffs {pc.get('diff')}); no DONE.json is written")
    pub_json(done, "DONE.json")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    pub_json(dict(runlog, seconds=round(time.time() - t_run, 1), done=done), f"runs/{stamp}_{rid}.json")
    log(f"DONE {P.run}: best epoch {td['best_epoch']}, held-out fp32 cos_patch "
        f"{done['heldout']['fp32']['cos_patch']:.4f} scale {done['heldout']['fp32']['scale_ratio']:.4f}")
    return dict(done, status="done", run_dir=RUN_DIR)


# COMMAND ----------

if not IS_NOTEBOOK:
    print("This notebook needs a GPU cluster and the path registry: see docs/REPRODUCE.md")
    raise SystemExit(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The recipe (GPU): verbatim Fixed cells 15 / 17 / 19 around the bench's verbatim student class

# COMMAND ----------

import torch                                   # noqa: E402
import torch.nn as nn                          # noqa: E402
import torch.nn.functional as F                # noqa: E402
from torch.utils.data import DataLoader, Dataset   # noqa: E402
from torchvision import transforms             # noqa: E402
from PIL import Image                          # noqa: E402
import timm                                    # noqa: E402
from timm.data import resolve_data_config      # noqa: E402

core._require_torch()
StudentEncoderTinyViT = core.StudentEncoderTinyViT      # VERBATIM export cell 13 + [bench-C1] drop_path_rate=0.0
device = torch.device("cuda")

# >>> Fixed cell 15 (dataset + augmentation), verbatim except that the pair list comes from the manifest
_tmp = timm.create_model('tiny_vit_21m_512', pretrained=False)
_data_cfg = resolve_data_config(_tmp.pretrained_cfg)
del _tmp
TINYVIT_MEAN = _data_cfg['mean']
TINYVIT_STD = _data_cfg['std']


class BackboneFeatureDataset(Dataset):
    """Paired dataset: RGB frame + pre-extracted backbone feature [1024, 72, 72]; synchronized h-flip augmentation."""
    def __init__(self, pairs, image_size=(1024, 1024), augment=False):
        self.image_size = image_size
        self.augment = augment
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=TINYVIT_MEAN, std=TINYVIT_STD),
        ])
        # Color jitter (image only -- teacher features already encode original colors)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02
        ) if augment else None
        self.pairs = [(p["image"], p["feature"]) for p in pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fp, bp = self.pairs[idx]
        img = Image.open(fp).convert('RGB')
        feat = torch.load(bp, map_location='cpu').float()
        if feat.dim() == 4:
            feat = feat.squeeze(0)  # [1024, 72, 72]
        if self.augment:
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            if torch.rand(1).item() > 0.5:
                img = transforms.functional.hflip(img)
                feat = feat.flip(-1)  # Flip width dimension of [C, H, W]
        img = self.base_transform(img)
        return img, feat
# <<< Fixed cell 15


# >>> Fixed cell 17 (loss + EMA + early stopping), verbatim
class BackboneDistillationLoss(nn.Module):
    def __init__(self, dir_w=1.0, cos_w=0.5, scale_w=0.3, raw_mse_w=0.1):
        super().__init__()
        self.dir_w = dir_w
        self.cos_w = cos_w
        self.scale_w = scale_w
        self.raw_mse_w = raw_mse_w

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        mse_dir = F.mse_loss(pred_norm, target_norm)
        pf = pred.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)
        tf = target.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)
        cosine = F.cosine_embedding_loss(
            pf, tf, torch.ones(pf.size(0), device=pred.device))
        pred_ch_std = pred.std(dim=(2, 3))
        target_ch_std = target.std(dim=(2, 3))
        pred_ch_mean = pred.mean(dim=(2, 3))
        target_ch_mean = target.mean(dim=(2, 3))
        scale = (F.mse_loss(pred_ch_std, target_ch_std) +
                 F.mse_loss(pred_ch_mean, target_ch_mean))
        mse_raw = F.mse_loss(pred, target)
        total = (self.dir_w * mse_dir + self.cos_w * cosine +
                 self.scale_w * scale + self.raw_mse_w * mse_raw)
        details = {
            'total_loss': total.item(),
            'mse_dir': mse_dir.item(),
            'cosine': cosine.item(),
            'scale': scale.item(),
            'mse_raw': mse_raw.item(),
            'pred_std': pred.std().item(),
            'target_std': target.std().item(),
            'scale_ratio': pred.std().item() / (target.std().item() + 1e-8),
        }
        return total, details


class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_loss, self.best_epoch, self.early_stop = 0, None, 0, False

    def __call__(self, val_loss, epoch):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch, self.counter = val_loss, epoch, 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
# <<< Fixed cell 17


def make_optimizer(student):
    """Fixed cell 17, verbatim."""
    return torch.optim.AdamW([
        {'params': student.backbone.parameters(),       'lr': 1e-4,  'weight_decay': 1e-4},
        {'params': student.lateral_s1.parameters(),     'lr': 3e-4,  'weight_decay': 1e-4},
        {'params': student.lateral_s2.parameters(),     'lr': 3e-4,  'weight_decay': 1e-4},
        {'params': student.lateral_s3.parameters(),     'lr': 3e-4,  'weight_decay': 1e-4},
        {'params': student.channel_proj.parameters(),   'lr': 3e-4,  'weight_decay': 1e-4},
        {'params': student.spatial_refine.parameters(), 'lr': 3e-4,  'weight_decay': 1e-4},
        {'params': [student.output_scale,
                    student.output_bias],                'lr': 3e-4,  'weight_decay': 0.0},
    ], lr=3e-4)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---- stochastic depth: training only, never at evaluation / export
def make_student(drop_path_rate, pretrained=True):
    """core.StudentEncoderTinyViT (verbatim export class + [bench-C1]) with `drop_path_rate` in its timm call. 0.0 = the
    class exactly as before. The parameters / state_dict do not depend on the rate (DropPath holds none),
    so a checkpoint loads into either form, in particular into the bench's drop_path_rate=0.0 class."""
    return build_with_drop_path(lambda: StudentEncoderTinyViT(out_channels=1024, pretrained=pretrained), core,
                                drop_path_rate)


def droppath_modules(model, strip=""):
    """[(module name, drop_prob)] of every timm DropPath (Identity where the rate is 0, so absent)."""
    from timm.layers import DropPath
    return [(n[len(strip):] if strip and n.startswith(strip) else n, float(m.drop_prob))
            for n, m in model.named_modules() if isinstance(m, DropPath)]


def set_droppath_training(model, flag):
    from timm.layers import DropPath
    for m in model.modules():
        if isinstance(m, DropPath):
            m.train(flag)


def assert_eval_mode(model, where):
    """Every val / EMA-val / fidelity / export forward runs with the whole model in eval(): stochastic depth inactive."""
    bad = [n for n, m in model.named_modules() if m.training]
    if bad:
        raise core.BenchAssertError(f"{where}: {len(bad)} modules in train mode (e.g. {bad[:3]}); stochastic depth "
                                    "would be active")


def _rng_get():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _rng_set(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])


DP_TRAIN_DIFF_MIN = 1e-3        # train() vs eval() max |diff| with stochastic depth on (kept paths are scaled by 1/keep)
DP_EVAL_REPEAT_MAX = 1e-6       # two eval() forwards
DP_RELEASED_MAX = 1e-6          # eval() of the drop_path_rate=0.0 class with the same weights
DP_N_SEEDS = 8                  # seeded train() forwards: >= 2 distinct outputs = the drop pattern follows the RNG


def droppath_checks(student, drop_path_rate, x):
    """A check before training (GPU, real timm). RNG-neutral: every RNG state is restored,
    so drop_path_rate=0.0 trains exactly as pigtrack_v1 did. Checks:
      schedule  dp=0: no DropPath module; dp>0: max drop_prob = dp, and for dp = timm's default the per-block schedule
                equals timm's own default construction of tiny_vit_21m_512 (= S-Edi's training);
      GN        train() with every DropPath in eval() vs eval(): max |diff| < core.GN_TRAIN_EVAL_MAX (the GroupNorm
                train/eval identity, measured with drop path INACTIVE);
      eval      two eval() forwards: max |diff| <= DP_EVAL_REPEAT_MAX (bit-identity recorded);
      train     dp>0: full train() vs eval() max |diff| > DP_TRAIN_DIFF_MIN and the DP_N_SEEDS seeded train() forwards
                give >= 2 distinct outputs; dp=0: train() == eval() within the GN gate;
      released  a drop_path_rate=0.0 class with the same state_dict (same keys) gives the same eval() output."""
    dp = float(drop_path_rate)
    rng = _rng_get()
    was_training = student.training
    fails = []
    try:
        mods = droppath_modules(student, "backbone.")
        ref = timm.create_model('tiny_vit_21m_512', pretrained=False, features_only=True, out_indices=(1, 2, 3))
        dflt = droppath_modules(ref)
        del ref
        res = {"drop_path_rate": dp, "n_droppath": len(mods), "drop_probs": [round(p, 6) for _, p in mods],
               "max_drop_prob": max((p for _, p in mods), default=0.0), "timm_default_n": len(dflt),
               "timm_default_max": max((p for _, p in dflt), default=0.0), "matches_timm_default": mods == dflt}
        if dp == 0.0:
            if mods:
                fails.append(f"{len(mods)} DropPath modules although drop_path_rate=0")
        else:
            if not mods or abs(res["max_drop_prob"] - dp) > 1e-6:
                fails.append(f"DropPath max {res['max_drop_prob']} != {dp}")
            if abs(dp - TIMM_DEFAULT_DROP_PATH) < 1e-12 and mods != dflt:
                fails.append("DropPath schedule differs from timm's default construction")
        with torch.no_grad():
            student.train()
            set_droppath_training(student, False)
            a = student(x).float()
            student.eval()
            assert_eval_mode(student, "droppath_checks eval")
            b = student(x).float()
            b2 = student(x).float()
            res["gn_train_eval_max_diff_dp_inactive"] = float((a - b).abs().max())
            res["eval_repeat_max_diff"] = float((b - b2).abs().max())
            res["eval_repeat_bit_identical"] = bool(torch.equal(b, b2))
            student.train()
            outs = []
            for k in range(DP_N_SEEDS):
                torch.manual_seed(1000 + k)
                outs.append(student(x).float())
            res["train_vs_eval_max_diff"] = max(float((o - b).abs().max()) for o in outs)
            res["train_vs_eval_min_diff"] = min(float((o - b).abs().max()) for o in outs)
            distinct = []
            for o in outs:
                if not any(torch.equal(o, d) for d in distinct):
                    distinct.append(o)
            res["train_distinct_outputs"] = len(distinct)
            torch.manual_seed(1000)
            res["train_same_seed_bit_identical"] = bool(torch.equal(student(x).float(), outs[0]))
            del outs, distinct, a, b2
            s0 = StudentEncoderTinyViT(out_channels=1024, pretrained=False).to(device=x.device, dtype=x.dtype)
            sd = student.state_dict()
            res["state_dict_keys_equal"] = set(sd) == set(s0.state_dict())
            s0.load_state_dict(sd, strict=True)
            s0.eval()
            res["released_class_max_diff"] = float((s0(x).float() - b).abs().max())
            del s0, sd, b
        if not res["gn_train_eval_max_diff_dp_inactive"] < core.GN_TRAIN_EVAL_MAX:
            fails.append(f"GroupNorm train/eval (drop path inactive) {res['gn_train_eval_max_diff_dp_inactive']:.3g}")
        if not res["eval_repeat_max_diff"] <= DP_EVAL_REPEAT_MAX:
            fails.append(f"eval() not deterministic: {res['eval_repeat_max_diff']:.3g}")
        if dp > 0.0:
            if not res["train_vs_eval_min_diff"] > DP_TRAIN_DIFF_MIN:
                fails.append(f"train() output equals eval() ({res['train_vs_eval_min_diff']:.3g}): stochastic depth off")
            if res["train_distinct_outputs"] < 2:
                fails.append(f"{DP_N_SEEDS} seeded train() forwards gave one output: no stochastic depth")
        elif not res["train_vs_eval_max_diff"] < core.GN_TRAIN_EVAL_MAX:
            fails.append(f"drop_path_rate=0 but train() != eval() ({res['train_vs_eval_max_diff']:.3g})")
        if not res["state_dict_keys_equal"]:
            fails.append("state_dict keys differ from the drop_path_rate=0.0 class")
        if not res["released_class_max_diff"] <= DP_RELEASED_MAX:
            fails.append(f"drop_path_rate=0.0 class with the same weights differs in eval(): "
                         f"{res['released_class_max_diff']:.3g}")
    finally:
        student.train(was_training)
        _rng_set(rng)
    res["pass"] = not fails
    res["fails"] = fails
    if fails:
        raise core.BenchAssertError("drop_path checks failed: " + "; ".join(fails))
    return res


def _cpu(d):
    return {k: v.detach().cpu().clone() for k, v in d.items()}


def prod_train(ctx, train_pairs, val_pairs, resume_local):
    """Fixed cell 19 (the training loop), verbatim in substance; additions are bookkeeping only: smoke caps, skipped-step
    counting (GradScaler), finite guards, timing, progress / resume publishing, stop_after_epoch."""
    P, log, rid = ctx["P"], ctx["log"], ctx["rid"]
    seed_all(P.seed)
    student = make_student(P.drop_path_rate, pretrained=True).to(device)
    bn_count = sum(1 for m in student.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)))
    gn_count = sum(1 for m in student.modules() if isinstance(m, nn.GroupNorm))
    g = torch.Generator(device="cpu").manual_seed(0)
    dummy = torch.randn(1, 3, 1024, 1024, generator=g).to(device)
    # b1.2 item 5: the GroupNorm train/eval gate is measured with drop path INACTIVE; stochastic depth is checked to be
    # active in train() only; RNG-neutral (drop_path_rate=0.0 trains exactly as pigtrack_v1)
    dpc = droppath_checks(student, P.drop_path_rate, dummy)
    gn_diff = dpc["gn_train_eval_max_diff_dp_inactive"]
    n_par = student.get_num_parameters()
    del dummy
    log(f"student: {n_par} params, BatchNorm {bn_count}, GroupNorm {gn_count}, train/eval max diff (drop path "
        f"inactive) {gn_diff:.2e}; drop_path_rate {P.drop_path_rate}: {dpc['n_droppath']} DropPath modules "
        f"(max {dpc['max_drop_prob']:.3f}, timm default schedule {dpc['matches_timm_default']}), train()-vs-eval() "
        f"{dpc['train_vs_eval_min_diff']:.3g}..{dpc['train_vs_eval_max_diff']:.3g}, {dpc['train_distinct_outputs']} "
        f"distinct seeded train() outputs, eval() repeat {dpc['eval_repeat_max_diff']:.1e} "
        f"(bit-identical {dpc['eval_repeat_bit_identical']}), drop_path_rate=0.0 class {dpc['released_class_max_diff']:.1e}")
    if bn_count or n_par != RECIPE["n_params"] or not gn_diff < core.GN_TRAIN_EVAL_MAX:
        raise core.BenchAssertError(f"student gates: bn {bn_count}, params {n_par}, train/eval diff {gn_diff}")
    pcfg = getattr(student.backbone, "pretrained_cfg", None) or {}
    pre_info = {k: str(pcfg.get(k)) for k in ("hf_hub_id", "tag", "architecture") if k in pcfg}

    train_dataset = BackboneFeatureDataset(train_pairs, augment=True)
    val_dataset = BackboneFeatureDataset(val_pairs, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=P.num_workers,
                              pin_memory=True, drop_last=True, persistent_workers=P.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=min(4, P.num_workers),
                            pin_memory=True, persistent_workers=P.num_workers > 0)
    log(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | batches {len(train_loader)} / {len(val_loader)}")

    criterion = BackboneDistillationLoss(dir_w=1.0, cos_w=0.5, scale_w=0.3, raw_mse_w=0.1)
    optimizer = make_optimizer(student)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ema = ModelEMA(student, decay=0.9995)
    early_stopping = EarlyStopping(patience=15)
    best_val_loss = float('inf')
    history = []
    start_epoch = 0
    best_local = osp.join(ctx["local"], f"best_{rid}.pt")
    best_metrics = None
    resumed_from = None
    if resume_local:
        st = torch.load(resume_local, map_location="cpu", weights_only=False)     # our own sha-verified state
        if st.get("config_sha256") != hashlib.sha256(dc.json_dumps(ctx["cfg"]).encode()).hexdigest():
            raise RuntimeError("resume state was written under a different DISTILL_CONFIG")
        student.load_state_dict(st["model_state_dict"])
        ema.shadow = {k: v.to(device) for k, v in st["ema_shadow"].items()}
        optimizer.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        scheduler.load_state_dict(st["scheduler"])
        for k, v in st["early_stopping"].items():
            setattr(early_stopping, k, v)
        best_val_loss = st["best_val_loss"]
        history = st["history"]
        start_epoch = st["epoch"] + 1
        if st.get("best") is not None:
            torch.save(st["best"], best_local)
            best_metrics = st["best"]["metrics"]
        rng = st["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        resumed_from = osp.basename(resume_local)
        log(f"resumed at epoch {start_epoch + 1} (best EMA val {best_val_loss:.4f}, early stop {early_stopping.counter})")
        del st
    cfg_sha = hashlib.sha256(dc.json_dumps(ctx["cfg"]).encode()).hexdigest()

    def save_state(epoch):
        name = dc.state_name(epoch + 1, rid)
        lp = osp.join(ctx["local"], "resume", name)
        os.makedirs(osp.dirname(lp), exist_ok=True)
        best_payload = torch.load(best_local, map_location="cpu", weights_only=False) if osp.isfile(best_local) else None
        torch.save({"epoch": epoch, "model_state_dict": _cpu(student.state_dict()), "ema_shadow": _cpu(ema.shadow),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(), "early_stopping": dict(vars(early_stopping)),
                    "best_val_loss": best_val_loss, "best": best_payload, "history": history,
                    "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                    "config_sha256": cfg_sha, "rid": rid}, lp)
        ctx["pub"](lp, f"resume/{name}")
        ctx["pub_json"]({"bytes": osp.getsize(lp), "sha256": core.sha256_file(lp), "epoch": epoch + 1, "rid": rid},
                        f"resume/{dc.state_marker(name)}")
        os.remove(lp)
        log(f"  resume state {name} published")

    for epoch in range(start_epoch, P.max_epochs):
        t_ep = time.time()
        # ── Train ──
        student.train()
        run = collections.defaultdict(float)
        n_steps = skipped = nonfinite = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if P.smoke_steps and batch_idx >= P.smoke_steps:
                break
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                pred = student(images)
                loss, ld = criterion(pred, targets)
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(student)
            skipped += int(scaler.get_scale() < scale_before)
            nonfinite += int(not math.isfinite(ld['total_loss']))
            for k, v in ld.items():
                run[k] += v
            n_steps += 1
            if (batch_idx + 1) % 100 == 0:
                n = batch_idx + 1
                log(f"  [E{epoch + 1}] B{n}/{len(train_loader)} Loss:{run['total_loss'] / n:.4f} "
                    f"DirMSE:{run['mse_dir'] / n:.4f} Cos:{run['cosine'] / n:.4f} ScaleR:{run['scale_ratio'] / n:.3f}")
            del images, targets, pred, loss
        train_loss = run['total_loss'] / max(n_steps, 1)
        # Step scheduler once per epoch (warm restarts) -- Fixed cell 19 passes the epoch explicitly
        scheduler.step(epoch)
        # ── Validate ──
        student.eval()
        assert_eval_mode(student, f"epoch {epoch + 1} val")          # stochastic depth inactive
        vrun = collections.defaultdict(float)
        nvb = 0
        with torch.no_grad():
            for vb, (images, targets) in enumerate(val_loader):
                if P.smoke_val_batches and vb >= P.smoke_val_batches:
                    break
                images = images.to(device)
                targets = targets.to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = student(images)
                    loss, ld = criterion(pred, targets)
                for k, v in ld.items():
                    vrun[k] += v
                nvb += 1
                del images, targets, pred, loss
        val_loss = vrun['total_loss'] / max(nvb, 1)
        # ── EMA validate ──
        ema.apply_shadow(student)
        assert_eval_mode(student, f"epoch {epoch + 1} EMA val")      # the EMA checkpoint is selected in eval()
        ema_val = 0.0
        ema_cos = 0.0
        with torch.no_grad():
            for vb, (images, targets) in enumerate(val_loader):
                if P.smoke_val_batches and vb >= P.smoke_val_batches:
                    break
                images = images.to(device)
                targets = targets.to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = student(images)
                    loss, ld_ema = criterion(pred, targets)
                ema_val += loss.item()
                ema_cos += ld_ema['cosine']
                del images, targets, pred, loss
        ema_val /= max(nvb, 1)
        ema_cos /= max(nvb, 1)
        ema.restore(student)
        if not (math.isfinite(val_loss) and math.isfinite(ema_val)) or nonfinite > max(1, 0.01 * n_steps) \
                or skipped > 0.5 * n_steps:
            raise core.BenchAssertError(f"epoch {epoch + 1}: val {val_loss} ema {ema_val}, non-finite train losses "
                                        f"{nonfinite}/{n_steps}, skipped fp16 steps {skipped}/{n_steps}")
        epoch_log = {
            'epoch': epoch + 1, 'train_loss': train_loss,
            'val_loss': val_loss, 'ema_val_loss': ema_val,
            'val_mse_dir': vrun['mse_dir'] / max(nvb, 1),
            'val_cosine': vrun['cosine'] / max(nvb, 1),
            'val_scale': vrun['scale'] / max(nvb, 1),
            'val_mse_raw': vrun['mse_raw'] / max(nvb, 1),
            'val_pred_std': vrun['pred_std'] / max(nvb, 1),
            'val_target_std': vrun['target_std'] / max(nvb, 1),
            'val_scale_ratio': vrun['scale_ratio'] / max(nvb, 1),
            'ema_cosine': ema_cos,
            'lr': optimizer.param_groups[0]['lr'],
            # bookkeeping (not in the Fixed log)
            'lr_heads': optimizer.param_groups[1]['lr'], 'sec': time.time() - t_ep, 'steps': n_steps,
            'skipped_fp16_steps': skipped, 'nonfinite_train_losses': nonfinite, 'grad_scale': scaler.get_scale(),
            'val_batches': nvb,
        }
        history.append(epoch_log)
        flag = ""
        # ── Checkpoint (Fixed: strict improvement of the EMA val loss) ──
        if ema_val < best_val_loss:
            best_val_loss = ema_val
            best_metrics = epoch_log
            torch.save({
                'epoch': epoch,
                'model_state_dict': _cpu(student.state_dict()),
                'ema_shadow': _cpu(ema.shadow),
                'optimizer': optimizer.state_dict(),
                'metrics': epoch_log,
                'provenance': {'recipe': ctx["cfg"]["recipe"]['name'], 'run': P.run, 'rid': rid, 'nb': NB_VERSION,
                               'config_sha256': cfg_sha, 'timm_pretrained': pre_info,
                               'drop_path_rate_in_training': P.drop_path_rate},
            }, best_local)
            flag = "  <- best"
        stop = early_stopping(ema_val, epoch)
        log(f"Epoch {epoch + 1}: train={train_loss:.4f} val={val_loss:.4f} ema_val={ema_val:.4f} "
            f"EMA_Cos={ema_cos:.4f} ScaleRatio={epoch_log['val_scale_ratio']:.3f} lr={epoch_log['lr']:.2e} "
            f"skipped={skipped} {epoch_log['sec'] / 60:.1f} min{flag}")
        ctx["pub_json"](epoch_log, f"progress/ep{epoch + 1:03d}_{rid}.json")
        stop_here = P.stop_after_epoch and epoch + 1 == P.stop_after_epoch
        if ((epoch + 1) % P.resume_every == 0 and not stop and epoch + 1 < P.max_epochs) or stop_here:
            save_state(epoch)
        if stop:
            log(f"EARLY STOPPING at epoch {epoch + 1}")
            break
        if stop_here:
            return {"stopped": "stop_after_epoch", "last_epoch": epoch + 1, "resumed_from": resumed_from,
                    "droppath_checks": dpc}
        gc.collect()
        torch.cuda.empty_cache()
    if best_metrics is None or not osp.isfile(best_local):
        raise RuntimeError("training produced no checkpoint")
    return {"stopped": "early_stopping" if early_stopping.early_stop else "max_epochs", "best_local": best_local,
            "history": history, "best_metrics": best_metrics, "resumed_from": resumed_from,
            "timm_pretrained": pre_info, "last_epoch": history[-1]["epoch"], "droppath_checks": dpc}


def to_bf16(m32):
    """A bf16 copy of an fp32 student. timm 1.0.25's TinyViT Attention caches `attention_biases[:, idxs]` per device on
    its first eval forward (the fp32 GroupNorm gate of core.load_student fills it); Module.to() does not convert that
    plain-dict cache, so a bf16 forward then fails in SDPA ("invalid dtype for bias", J5 smoke 2026-10-14). The cache is
    dropped after the cast (timm itself drops it on train(True)); it is rebuilt from the bf16 parameters."""
    m = copy.deepcopy(m32).to(torch.bfloat16).eval()
    for mod in m.modules():
        if isinstance(getattr(mod, "attention_bias_cache", None), dict):
            mod.attention_bias_cache = {}
    return m


def _load_students(best_local, ctx):
    """The trained file and S-Edi (registry models.student_edinburgh, sha256 pinned to 9686b4c0..., read-only; the only reference of b1.11),
    both through the bench's own loader and gates (the drop_path_rate=0.0 class, eval())."""
    out = {}
    s32, meta = core.load_student(best_local, device="cuda", dtype=torch.float32, check_gn=True)
    out[ctx["P"].arm] = (s32, to_bf16(s32), dict(meta, sha256=core.sha256_file(best_local)))
    R = ctx["R"]
    want = R.entry(REF_STUDENT_KEY).get("sha256") or REF_STUDENT_SHA256
    if want != PINS["sedi_sha256"]:
        raise core.BenchAssertError(f"the S-Edi pin {want} is not the registered {PINS['sedi_sha256']}")
    src = R.path(REF_STUDENT_KEY)
    ref_local, sha = core.stage_file(src, osp.join(ctx["local"], "ref", "S-Edi", osp.basename(src)), ctx["dbutils"], want)
    e32, emeta = core.load_student(ref_local, device="cuda", dtype=torch.float32, check_gn=True)
    out["S-Edi"] = (e32, to_bf16(e32), dict(emeta, sha256=sha, registry_key=REF_STUDENT_KEY))
    for name, (a, b, _m) in out.items():
        assert_eval_mode(a, f"fidelity {name} fp32")
        assert_eval_mode(b, f"fidelity {name} bf16")
    return out


def prod_fidelity(ctx, best_local, val):
    log = ctx["log"]
    students = _load_students(best_local, ctx)
    names = [s for s in student_order(ctx["P"].arm) if s in students]
    tf = core.make_student_transform()
    rows = []
    t0 = time.time()
    for k, v in enumerate(val):
        t = torch.load(v["feature"], map_location="cpu").float().numpy()
        with Image.open(v["image"]) as im:
            x = tf(im.convert("RGB")).unsqueeze(0)
        row = {"seq": v["seq"], "fi": v["fi"], "night": v["night"]}
        for name in names:
            m32, m16, _meta = students[name]
            with torch.no_grad():
                f32 = m32(x.to(device)).float().cpu().numpy()
                f16 = m16(x.to(device, torch.bfloat16)).float().cpu().numpy()
            if not (np.isfinite(f32).all() and np.isfinite(f16).all()):
                raise core.BenchAssertError(f"{name}: non-finite features on {v['seq']} {v['fi']}")
            row[name] = {"fp32": dc.fidelity_metrics(f32, t), "bf16": dc.fidelity_metrics(f16, t)}
        rows.append(row)
        if (k + 1) % 100 == 0:
            log(f"  fidelity {k + 1}/{len(val)} frames ({time.time() - t0:.0f} s)")
    summary = dc.fidelity_groups(rows, tuple(names))
    res = {"definitions": dc.fidelity_metrics.__doc__, "n_frames": len(rows),
           "val_seqs": sorted({r["seq"] for r in rows}), "summary": summary,
           "students": {n: students[n][2] for n in names},
           "input": "core.make_student_transform() (= the export transform) on the byte-identical decoded Edinburgh JPEG",
           "teacher_targets": "the stored fp16 targets of the 2026-02 extraction (val clip, teacher_targets/edinburgh)",
           "seconds": round(time.time() - t0, 1), "gpu": core.gpu_info()}
    if "S-Edi" in students:
        res["positive_control"] = fide_control(rows)
        pc = res["positive_control"]
        log(f"  positive control (S-Edi fp32, stride-5 val subset vs FID-E): {pc['status']} ({pc['n']} frames; diffs {pc['diff']})")
    for _n, (a, b, _m) in students.items():
        del a, b
    students.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return res, fidelity_csv(rows, names)


def verify_published(path, ctx):
    """Load the RELEASED file exactly as the bench's student arms do (core.load_student: the drop_path_rate=0.0 class,
    bf16, GroupNorm gate in fp32 with drop path inactive)."""
    s, meta = core.load_student(path, device="cuda", dtype=torch.bfloat16, check_gn=True)
    ctx["log"](f"released checkpoint passes the bench load gates: {meta}")
    del s
    gc.collect()
    torch.cuda.empty_cache()


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


def gpu_preflight(P):
    probe = cuda_probe()
    g = core.gpu_info()
    if P.require_gpu != "any" and g["class"] != P.require_gpu:
        raise RuntimeError(f"GPU {g['name']} (class {g['class']}) but require_gpu={P.require_gpu}: J5 runs on the A100")
    g["probe"] = probe
    return g


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


def code_shas():
    try:
        nb = hashlib.sha256(_notebook_source()).hexdigest()
    except Exception as e:  # noqa: BLE001
        nb = f"unavailable: {e!r}"[:200]
    return {"notebook": nb, "distill_core": core.sha256_file(dc.__file__),
            "bench_core": core.sha256_file(core.__file__), "farm_paths": core.sha256_file(farm_paths.__file__),
            "nb_version": NB_VERSION, "dc_version": dc.DISTILL_CORE_VERSION, "core_version": core.BENCH_VERSION}


PROD_HOOKS = {
    "local_root": "/local_disk0/tol02d",
    "reuse_root": None,
    "dbutils": dbutils,
    "log": _log,
    "gpu_preflight": gpu_preflight,
    "versions": core.versions,
    "code_shas": code_shas,
    "train": prod_train,
    "fidelity": prod_fidelity,
    "verify_published": verify_published,
}

# COMMAND ----------

for _k, _v in WIDGETS.items():
    dbutils.widgets.text(_k, _v)
RAW = {k: dbutils.widgets.get(k) for k in WIDGETS}
print("widgets:", RAW)
RESULT = main(RAW, hooks=PROD_HOOKS)
print(json.dumps(RESULT, indent=1, default=str))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({k: RESULT.get(k) for k in ("status", "run", "best_epoch", "epochs_run", "stopped",
                                                              "train_seconds", "heldout", "heldout_S-Edi",
                                                              "arm", "positive_control_status", "gate_gm", "loads_in_bench_class", "drop_path_rate", "model_dir",
                                                              "eval_gpu_class",
                                                              "ckpt_sha256", "last_epoch", "skipped")}, default=str))
