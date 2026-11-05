# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N02c — Distil a b1.6 PigTrack student: base targets + extra (hard-mined / random) targets (Paper 1, job **J15**, **A100**)
# MAGIC
# MAGIC **`Tol_02b_distill_student_pigtrack` with the data layer, the paths and the run names changed and nothing else** (the training
# MAGIC recipe, the loss, the augmentation, the optimiser, the schedule, the EMA, the early stopping, the drop-path gates and the
# MAGIC held-out fidelity code are byte-identical; `bench/pt_mining/train_diff.md` lists every hunk). Training set = the pinned J5
# MAGIC base manifest's 2,626 train targets (`teacher_targets/pigtrack_train/manifest.csv`, sha256 `6eff15e8...`) UNION the 2,125
# MAGIC extra targets of `manifest_<target_set>.csv` in `teacher_targets/pigtrack_mined_v1/` (written by `Tol_75_pigtrack_mine_targets`,
# MAGIC sha256 pinned by the `extra_manifest_sha256` widget); validation = the SAME 374 base val targets (checkpoint selection and
# MAGIC held-out fidelity). Init = fresh ImageNet TinyViT (calf v2's rule), seed 42 (R2: 43). The four runs:
# MAGIC `pigtrack_mined_v1` (set `mined`, S-PT-mined), `pigtrack_random_v1` (`random`, S-PT-rand), `pigtrack_random2_v1`
# MAGIC (`random2`, seed 43, S-PT-rand2), `pigtrack_comp_v1` (`comp`, S-PT-comp; only if trigger T1 of `MINING_SUMMARY.json` fired).
# MAGIC No run publishes to `models/` (`publish_model` must be `auto`/`false`; `PUBLISH_RUNS` is empty). Fidelity (`eval_stage=only`, A10)
# MAGIC evaluates the new student beside S-PT (`c07140f2...`), S-PT-dp0 and S-Edi (all sha-pinned) on the 374 val targets.
# MAGIC Local check: `TOL02C_SELFTEST=1 python Tol_02c_distill_pigtrack_mined.py`. The cell below is Tol_02b's documentation.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC # (Tol_02b) Distil the S-PT student on the PigTrack TRAIN targets (Paper 1, job **J5b**, **A100**)
# MAGIC
# MAGIC A path-isolated copy of `Tol_02_distill_student` (which writes into `Tol/`, so it is not edited) that runs **the
# MAGIC recipe of the published Edinburgh student** (BENCH_PROTOCOL §1: S-PT = "same recipe"), i.e. the
# MAGIC `Student+TeacherDecoder_TinyViT_Fixed` notebook, cells 13 / 15 / 17 / 19 (export `catalog/paper1_ws/student_fixed.txt`),
# MAGIC with the single change **`drop_path_rate=0.0`** (the verbatim class in `tol_bench_core`, marked `[bench-C1]`).
# MAGIC `Tol_02` itself departs from that recipe in seven places; S-PT follows the published one:
# MAGIC
# MAGIC | item | published Edinburgh recipe (= this notebook) | `Tol_02` (calf students) |
# MAGIC |---|---|---|
# MAGIC | student init | verbatim class incl. `_init_weights` (kaiming fan-out, last GroupNorm of the residual zero) | no `_init_weights` |
# MAGIC | scale term | `MSE(per-channel std) + MSE(per-channel mean)` | L1 |
# MAGIC | optimiser | AdamW, 7 groups: backbone 1e-4, heads 3e-4; weight decay 1e-4 (0 for `output_scale/bias`) | 2 groups, wd 0.01 |
# MAGIC | schedule | `CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-6)`, `scheduler.step(epoch)` | eta_min 0, `step()` |
# MAGIC | precision | `autocast` fp16 + `GradScaler`, loss inside autocast | bf16 autocast, no scaler |
# MAGIC | augmentation | `ColorJitter(0.2, 0.2, 0.1, 0.02)` on the full-res frame, then synced h-flip (`rand > 0.5`), then `Resize((1024,1024))` | saturation 0.2, resize first |
# MAGIC | checkpoint / stop | save when EMA val < best; `EarlyStopping(15, min_delta 1e-5)` on EMA val; val loss = mean over batches | `best - 1e-5` for both |
# MAGIC
# MAGIC Unchanged in both: TinyViT-21M-512 + FPN + GroupNorm (40,655,408 params), loss `1.0 dirMSE + 0.5 cos + 0.3 scale +
# MAGIC 0.1 rawMSE`, grad clip 1.0, EMA 0.9995 (reported), patience 15, ≤ 100 epochs, batch 4 @ 1024², TinyViT-512 mean/std.
# MAGIC Not part of the recipe (recorded): `num_workers` 8 (train) / 4 (val), seed 42, timm 1.0.25, Pillow 11.1.0.
# MAGIC
# MAGIC **Data:** `paper1_distillation/teacher_targets/pigtrack_train/manifest.csv` (Tol_66): `role=train` rows train, `role=val`
# MAGIC rows (4 whole held-out train sequences) select the checkpoint and give the held-out fidelity. Staged to local NVMe;
# MAGIC on the Tol_66 job cluster the local copies are verified by sha256 instead of being copied again.
# MAGIC
# MAGIC **Outputs:** run dir `paper1_distillation/distillation_runs/<run>/` (registry `paper1.distillation_runs`):
# MAGIC ```
# MAGIC DISTILL_CONFIG.json                    recipe + data identity (manifest sha) + versions + GPU class; a resume must match
# MAGIC progress/ep<NNN>_<rid>.json            one per epoch (the Fixed epoch_log keys + timing, skipped fp16 steps)
# MAGIC resume/state_ep<NNN>_<rid>.pt (+.json) every `resume_every` epochs: model, EMA, optimiser, scaler, scheduler, early
# MAGIC                                        stopping, best checkpoint, history, RNG states (the .json marks it complete)
# MAGIC final_<rid>/best_backbone_student_pigtrack.pt, backbone_distill_history.json      + train_done.json (points to it)
# MAGIC eval_<rid>/fidelity_heldout.json, per_frame_fidelity.csv                         + eval_done.json  (points to it)
# MAGIC DONE.json, runs/<utc>_<rid>.json
# MAGIC ```
# MAGIC `<rid>` = one id per notebook run, so a resumed attempt never collides with files of a crashed one. Only
# MAGIC `run=pigtrack_v1` (not smoke) publishes the model to the registry key `models.student_pigtrack`
# MAGIC (`models/sam3_student_tinyvit/pigtrack_v1/best_backbone_student_pigtrack.pt`, + `backbone_distill_history.json`,
# MAGIC `fidelity_heldout.json`, `DISTILL_CONFIG.json`), never overwriting.
# MAGIC
# MAGIC **Held-out fidelity** (EMA best epoch, `tol_distill_core.fidelity_metrics` = Tol_68 stage d definitions): per-patch
# MAGIC cosine, global cosine, scale ratio, MSE … on every val target, in fp32 and in bf16 (the bench precision), overall / day /
# MAGIC night / per sequence; the same for the published Edinburgh student (S-Edi, zero-shot) on the same frames.
# MAGIC
# MAGIC **Resume:** re-run with the same widgets: a finished stage (train_done / eval_done / DONE) is skipped; training restarts
# MAGIC from the newest complete `resume/` state. Local check: `TOL02B_SELFTEST=1 python Tol_02b_distill_student_pigtrack.py`.
# MAGIC
# MAGIC **`drop_path_rate` (BENCH_PROTOCOL §15 b1.2, `tol02b-2026-10-15.1`).** Default `0.0`, which reproduces `pigtrack_v1`
# MAGIC byte for byte. `0.1` gives timm 1.0.25's default stochastic depth for `tiny_vit_21m_512`, the setting S-Edi trained
# MAGIC under, and is used by run `pigtrack_v2_dp01` (the headline S-PT; `pigtrack_v1` becomes the ablation row S-PT-dp0).
# MAGIC Stochastic depth is active **only in `train()`**: every val / EMA-val pass, the fidelity and the released checkpoint
# MAGIC use `eval()`, and the checkpoint loads unchanged into the bench's `drop_path_rate=0.0` class. Gates before
# MAGIC training (`droppath_checks`, recorded in `train_done.json`):
# MAGIC - `train()` outputs differ from `eval()` outputs;
# MAGIC - two `eval()` forwards agree;
# MAGIC - the GroupNorm train/eval check is made with every DropPath module in `eval()`;
# MAGIC - the DropPath schedule equals timm's default;
# MAGIC - a `drop_path_rate=0.0` copy gives the same `eval()` output.
# MAGIC
# MAGIC A canonical run other than `pigtrack_v1` must equal it in `data` and `training`, and may differ in the recipe only
# MAGIC in `name` / `student` / `drop_path_rate` / `drop_path_scope`.
# MAGIC
# MAGIC **`eval_stage`:**
# MAGIC - `inline` (default): train → fidelity → publish → DONE, as before.
# MAGIC - `defer`: train, then publish the checkpoint, history and DISTILL_CONFIG; no fidelity yet. Used on the A100.
# MAGIC - `only`: fidelity of a finished training, then publish `fidelity_heldout.json` and DONE. Used on the A10. The
# MAGIC   DISTILL_CONFIG must match apart from the GPU class and versions, and timm must be equal.
# MAGIC
# MAGIC Every run other than `pigtrack_v1` also evaluates S-PT-dp0 (`models.student_pigtrack`, sha-pinned) beside S-Edi.
# MAGIC Both are then compared with `pigtrack_v1`'s published per-frame evaluation (`reference_reproduction`).

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
import gc
import hashlib
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

NB_NAME = "Tol_02c_distill_pigtrack_mined"
NB_VERSION = "tol02c-2026-10-24.2"
CONFIG_VERSION = "tol02c-distill-v1"
SELFTEST = os.environ.get("TOL02C_SELFTEST") == "1"
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None
if not IS_NOTEBOOK and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _code_dir():
    env = os.environ.get("TOL02C_CODE_DIR")
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
core = _import_exact("tol_bench_core", [CODE_DIR])
dc = _import_exact("tol_distill_core", [CODE_DIR])
_PATHS_DIRS = [CODE_DIR, os.environ.get("TOL_PATHS_DIR")]
if not IS_NOTEBOOK:
    _PATHS_DIRS.append(osp.join(osp.dirname(CODE_DIR), "catalog", "build"))
tol_paths = _import_exact("tol_paths", _PATHS_DIRS)
# Full sha256 of the shared modules this notebook may run against: the Workspace-root core 5e705fff (the core S-PT trained with) or
# _pinned/core_bc46ecfb; every symbol used here is AST-identical in the two (bench/pt_mining/_work/ast_core_check.json).
CODE_PINS = {
    "tol_bench_core": {"5e705fffcb1711a427239a6bcec3e3622b9b8be69dc3e1b459643c663d460e58": "Workspace root core (S-PT training core)",
                       "bc46ecfb9eca56af5fc991d5ea0e3460776cc2dc1f9780c9a8c756c060132b16": "_pinned/core_bc46ecfb (bench core)"},
    "tol_distill_core": {"16522b6c738bac6fa6cbc3097481b9e970acd11538e495d40def92371761fe02": "dc1.0 2026-10-14"},
    "tol_paths": {"f6603e639ec0e010640da401afc3837ae2e3923bf9c9dd62ff3e55bc9d9d3d46": "registry v1 loader"},
}
for _mod, _name in ((core, "tol_bench_core"), (dc, "tol_distill_core"), (tol_paths, "tol_paths")):
    _sha = core.sha256_file(_mod.__file__)
    if _sha not in CODE_PINS[_name]:
        raise RuntimeError(f"{_name} sha256 {_sha} is not one of the pinned versions {sorted(CODE_PINS[_name])}")
RID = uuid.uuid4().hex[:6]
print(f"{NB_NAME} {NB_VERSION} rid {RID}: tol_bench_core {core.BENCH_VERSION} "
      f"sha256 {core.sha256_file(core.__file__)[:16]}, tol_distill_core sha256 {core.sha256_file(dc.__file__)[:16]}, "
      f"tol_paths sha256 {core.sha256_file(tol_paths.__file__)[:16]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters, recipe constants and the pure-Python driver (exercised by the selftest with a stub trainer)

# COMMAND ----------

WIDGETS = collections.OrderedDict([
    ("run", "pigtrack_mined_v1"), ("max_epochs", "100"), ("resume_every", "10"), ("seed", "42"), ("num_workers", "8"),
    ("smoke_steps", "0"), ("smoke_val_batches", "0"), ("fid_max_frames", "0"), ("stop_after_epoch", "0"),
    ("publish_model", "auto"), ("require_gpu", "A100"), ("stage_threads", "16"),
    ("drop_path_rate", "0.0"), ("eval_stage", "inline"),
    ("target_set", "mined"), ("extra_manifest_sha256", ""),
])
CANONICAL_RUN = "pigtrack_v1"                      # = S-PT-dp0: its model dir and published evaluation are read-only references
# b1.6: NO run of this notebook publishes a model (models/ is written only after test scoring), so PUBLISH_RUNS is empty and the
# Tol_02b publish paths below stay inert (parse_params refuses publish_model=true).
PUBLISH_RUNS = {}
# The b1.6 arms: run tag -> (target set, training seed, arm name). drop_path_rate 0.1 = S-PT's (timm default = S-Edi's).
ARM_RUNS = {"pigtrack_mined_v1": ("mined", 42, "S-PT-mined"), "pigtrack_random_v1": ("random", 42, "S-PT-rand"),
            "pigtrack_random2_v1": ("random2", 43, "S-PT-rand2"), "pigtrack_comp_v1": ("comp", 42, "S-PT-comp")}
ARM_DROP_PATH = 0.1
TARGET_SETS = tuple(v[0] for v in ARM_RUNS.values())
ARM_BY_SET = {v[0]: v[2] for v in ARM_RUNS.values()}
ARM_SEED_BY_SET = {v[0]: v[1] for v in ARM_RUNS.values()}
REF_RUN = "pigtrack_v2_dp01"                       # = S-PT: recipe / training / base-data / versions reference of every b1.6 run
SPT_SHA256 = "c07140f2de80a5e9faf175bdb4fc071bb1064e4757b91c3eefe55e271e5da69c"   # pigtrack_v2_dp01 EMA epoch 32 (S-PT)
SPT_DP0_SHA256 = "c4edf8805520b83fe4c1fe094f70287d97864e45bde341fb7a641c1fdcb777cd"   # pigtrack_v1 EMA epoch 28
EXPECT_MANIFEST_SHA256 = "6eff15e8350e2b155a2f3a16a7edede7883c95b8a105f340fb1faf38089d8a27"   # Tol_66, J5 (published)
TIMM_DEFAULT_DROP_PATH = 0.1                       # timm 1.0.25 tiny_vit_21m_512 default (tiny_vit.py, S-Edi's training)
DROP_PATH_MAX = 0.5
DP_RECIPE_KEYS = ("name", "student", "drop_path_rate", "drop_path_scope")   # the only recipe keys a dp run may change
EVAL_STAGES = ("inline", "defer", "only")
EVAL_ONLY_IGNORE = ("gpu_class", "versions")       # an eval_stage=only run may differ from the training run in these
TARGETS_KEY = "paper1.teacher_targets.pigtrack_train"
MODEL_KEY = "models.student_pigtrack"
REF_STUDENT_KEY = "models.student_edinburgh"       # zero-shot reference on the same val frames (read-only)
REF_STUDENT_SHA256 = "9686b4c0f60e02251d8c2febee2029c1e7ebe74e85fbefda7e0c5daffc2075f2"   # J0: EMA epoch 75 (S-Edi)
REGISTRY_VERSION = 1
CKPT_NAME = "best_backbone_student_pigtrack.pt"    # = basename of the registry key models.student_pigtrack
HISTORY_NAME = "backbone_distill_history.json"     # = the Edinburgh model folder's history name
TOL66_REUSE_ROOT = "/local_disk0/tol66/pigtrack_train/out"
# ---- b1.6 extra targets: teacher_targets/pigtrack_mined_v1/ (Tol_75), read only here
EXTRA_DIR = "pigtrack_mined_v1"
EXTRA_N, EXTRA_NIGHT = 2125, 747                   # every set: M = 2,125 frames, 747 of them night (DESIGN s.3)
# Canonical sha256 (sorted "seq,fi" lines, one per line) of the PRE-REGISTERED random extra sets R (`random`) and R2 (`random2`),
# DESIGN s.3 / the Prereg amendment. mined / comp are known only after J14: for them load_extra_targets compares
# MINING_SUMMARY.json's set_canon_sha256.
RANDOM_CANON_SHA256 = {"random": "afaf2371b85111d9696c6c48da3620ce99bd4080ead479624ac298c48902b0a8",
                       "random2": "5f1b04a3756e0ff64120f092bad0affbbe37fba80438ec08b265c0ddf2298e9a"}
# `training` keys that a smoke run changes by construction; the S-PT gate does not compare them in a smoke run
SMOKE_TRAINING_KEYS = ("max_epochs", "resume_every", "smoke_steps", "smoke_val_batches", "fid_max_frames")
WHITELIST31 = (
    "pigtrack0003", "pigtrack0009", "pigtrack0012", "pigtrack0013", "pigtrack0014", "pigtrack0016", "pigtrack0019",
    "pigtrack0023", "pigtrack0024", "pigtrack0025", "pigtrack0026", "pigtrack0027", "pigtrack0028", "pigtrack0035",
    "pigtrack0036", "pigtrack0039", "pigtrack0043", "pigtrack0044", "pigtrack0045", "pigtrack0046", "pigtrack0052",
    "pigtrack0053", "pigtrack0057", "pigtrack0059", "pigtrack0061", "pigtrack0063", "pigtrack0073", "pigtrack0075",
    "pigtrack0076", "pigtrack0079", "pigtrack0080")
WHITELIST_SHA256 = "163dbf117668955ab0bed54e183dfc2f02f39e61404bb910d80107513b9392bd"   # sorted, newline-joined, final newline
VAL4 = ("pigtrack0041", "pigtrack0050", "pigtrack0064", "pigtrack0070")
OFFICIAL_VAL20 = ("pigtrack0001", "pigtrack0006", "pigtrack0007", "pigtrack0008", "pigtrack0015", "pigtrack0020",
                  "pigtrack0022", "pigtrack0029", "pigtrack0031", "pigtrack0032", "pigtrack0033", "pigtrack0034",
                  "pigtrack0037", "pigtrack0047", "pigtrack0048", "pigtrack0051", "pigtrack0056", "pigtrack0062",
                  "pigtrack0066", "pigtrack0069")
TEST25 = ("pigtrack0002", "pigtrack0004", "pigtrack0005", "pigtrack0010", "pigtrack0011", "pigtrack0017", "pigtrack0018",
          "pigtrack0021", "pigtrack0030", "pigtrack0038", "pigtrack0040", "pigtrack0042", "pigtrack0049", "pigtrack0054",
          "pigtrack0055", "pigtrack0058", "pigtrack0060", "pigtrack0065", "pigtrack0067", "pigtrack0068", "pigtrack0071",
          "pigtrack0072", "pigtrack0074", "pigtrack0077", "pigtrack0078")
# The published Edinburgh recipe (Fixed cells 13/15/17/19). Recorded in DISTILL_CONFIG; the code below implements it.
RECIPE = {
    "name": "edinburgh_fixed_v1 + drop_path_rate=0.0",
    "source": "Student+TeacherDecoder_TinyViT_Fixed cells 13/15/17/19 (catalog/paper1_ws/student_fixed.txt)",
    "student": "tol_bench_core.StudentEncoderTinyViT(out_channels=1024, pretrained=True) (verbatim export + [bench-C1])",
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
RECIPE_V1_SHA256 = "b1f4457d554dd70eada8db06d2b99fe4e79fd2d6e826043778495a1b20aeb255"   # json_dumps(pigtrack_v1 recipe)
FID_BANDS = {"cos_patch_min": 0.70, "scale_ratio": [0.8, 1.2]}        # Tol_02's a-priori bands (informative only)


def recipe_for(drop_path_rate):
    """The recipe dict recorded in DISTILL_CONFIG. drop_path_rate 0.0 -> RECIPE itself, byte-identical to pigtrack_v1's
    (RECIPE_V1_SHA256, asserted by the selftest), so that run still resumes / reproduces. Otherwise the four keys of
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
    """BENCH_PROTOCOL §15 b1.2 item 4: a publishing run other than pigtrack_v1 must equal it in `data`, `training`,
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


# ---- the S-PT-dp0 / S-Edi reproduction check against pigtrack_v1's published evaluation (pure Python)
FID_CSV_KEYS = ("cos_patch", "cos_global", "scale_ratio", "mse")
REPRO_MAP = (("S-PT-dp0", "S-PT"), ("S-Edi", "S-Edi"))   # (student here, its column prefix in pigtrack_v1's evaluation)
REPRO_TOL = {"cos_patch": 1e-3, "cos_global": 1e-3, "scale_ratio": 1e-3, "mse": 1e-3}   # abs; mse relative
REPRO_MAP_V2 = (("S-PT", "S-PT"), ("S-PT-dp0", "S-PT-dp0"), ("S-Edi", "S-Edi"))   # vs pigtrack_v2_dp01's published evaluation


def student_order(arm):
    """The students of the held-out fidelity, in column order: this run's arm, then the sha-pinned references."""
    return (arm, "S-PT", "S-PT-dp0", "S-Edi")


def fidelity_csv(rows, students):
    """per_frame_fidelity.csv (the pigtrack_v1 column layout, one column block per student present)."""
    cols = ["seq", "fi", "night"] + [f"{s}_{p}_{k}" for s in students for p in ("fp32", "bf16") for k in FID_CSV_KEYS]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join([r["seq"], str(r["fi"]), str(r["night"])] + [repr(r[s][p][k]) for s in students
                                                                          for p in ("fp32", "bf16")
                                                                          for k in FID_CSV_KEYS]))
    return "\n".join(lines) + "\n"


def parse_fidelity_csv(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    head = lines[0].split(",")
    out = {}
    for ln in lines[1:]:
        v = ln.split(",")
        d = dict(zip(head, v, strict=True))
        out[(d["seq"], int(d["fi"]))] = {k: float(x) for k, x in d.items() if k not in ("seq", "fi", "night")}
    return out


def reference_reproduction(rows, ref_csv_text, ref_summary=None, cur_summary=None, repro_map=None):
    """Re-computed S-PT-dp0 / S-Edi vs the published pigtrack_v1 evaluation: per-frame max |diff| (mse relative) for
    every student / precision / metric on the frames present in both, and, when both summaries cover the same frame
    count, the overall / day / night means. status: identical | within_tolerance | differs (reported, never fatal)."""
    ref = parse_fidelity_csv(ref_csv_text)
    out = {"tolerance": REPRO_TOL, "mse": "relative", "per_frame": {}, "summary_mean_abs_diff": {}, "missing": 0}
    worst_ok, all_zero, n_cmp = True, True, 0
    for cur, old in (repro_map or REPRO_MAP):
        for pr in ("fp32", "bf16"):
            for k in FID_CSV_KEYS:
                ds = []
                for r in rows:
                    if cur not in r:
                        continue
                    o = ref.get((r["seq"], int(r["fi"])))
                    if o is None:
                        out["missing"] += 1
                        continue
                    a, b = float(r[cur][pr][k]), o[f"{old}_{pr}_{k}"]
                    ds.append(abs(a - b) / (abs(b) if k == "mse" and b else 1.0))
                if not ds:
                    continue
                n_cmp = max(n_cmp, len(ds))
                mx = max(ds)
                out["per_frame"][f"{cur}/{pr}/{k}"] = {"max": mx, "mean": sum(ds) / len(ds), "n": len(ds)}
                all_zero &= mx == 0.0
                worst_ok &= mx <= REPRO_TOL[k]
    if ref_summary and cur_summary:
        for cur, old in (repro_map or REPRO_MAP):
            if cur not in cur_summary or old not in ref_summary:
                continue
            for pr in ("fp32", "bf16"):
                for grp in ("overall", "day", "night"):
                    a, b = cur_summary[cur][pr][grp], ref_summary[old][pr][grp]
                    if a.get("n") != b.get("n"):
                        continue
                    for k in FID_CSV_KEYS:
                        d = abs(a[k]["mean"] - b[k]["mean"]) / (abs(b[k]["mean"]) if k == "mse" else 1.0)
                        out["summary_mean_abs_diff"][f"{cur}/{pr}/{grp}/{k}"] = d
                        all_zero &= d == 0.0
                        worst_ok &= d <= REPRO_TOL[k]
    out["n_frames_compared"] = n_cmp
    out["status"] = ("no_overlap" if not out["per_frame"] else
                     "identical" if all_zero else "within_tolerance" if worst_ok else "differs")
    return out


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
        raise ValueError(f"run={P.run} trains with drop_path_rate={PUBLISH_RUNS[P.run]} (BENCH_PROTOCOL §15 b1.2), "
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
    # ---- b1.6 (Tol_02c): target set, its pinned manifest, the full-run rules of the four arms
    P.target_set = g["target_set"]
    if P.target_set not in TARGET_SETS:
        raise ValueError(f"target_set={P.target_set!r}; one of {TARGET_SETS}")
    P.extra_manifest_sha256 = g["extra_manifest_sha256"].lower()
    if P.extra_manifest_sha256 and not re.match(r"^[0-9a-f]{64}$", P.extra_manifest_sha256):
        raise ValueError("extra_manifest_sha256 must be 64 hex characters (the sha256 of manifest_<target_set>.csv) or empty")
    if not P.smoke:
        if P.run not in ARM_RUNS:
            raise ValueError(f"run={P.run!r}: one of {sorted(ARM_RUNS)} (or a 'smoke...' tag)")
        want_set, want_seed, _arm = ARM_RUNS[P.run]
        if P.target_set != want_set or P.seed != want_seed:
            raise ValueError(f"run={P.run} trains on target_set={want_set} with seed {want_seed} (BENCH b1.6), "
                             f"not {P.target_set} / {P.seed}")
        if P.max_epochs != 100:
            raise ValueError("the b1.6 runs use the recipe's 100-epoch cap")
        if P.drop_path_rate != ARM_DROP_PATH:
            raise ValueError(f"the b1.6 runs train with drop_path_rate={ARM_DROP_PATH} (S-PT's), not {P.drop_path_rate}")
        if not P.extra_manifest_sha256:
            raise ValueError("a full run needs extra_manifest_sha256 (the pinned sha256 of manifest_<target_set>.csv)")
    P.arm = ARM_RUNS[P.run][2] if P.run in ARM_RUNS else ARM_BY_SET[P.target_set]
    return P


def _read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def build_config(P, man_sha, tcfg, tcfg_sha, summ, versions, gpu_class, extra):
    cfg = {"version": CONFIG_VERSION, "run": P.run, "arm": P.arm, "recipe": recipe_for(P.drop_path_rate),
           "data": {"targets_key": TARGETS_KEY, "manifest_sha256": man_sha, "targets_config_sha256": tcfg_sha,
                    "n_train": summ["train"], "n_val": summ["val"], "val_seqs": summ["val_seqs"],
                    "selection_plan_sha256": tcfg["selection"]["plan_sha256"],
                    "teacher_revision": tcfg["teacher"]["revision"], "extra": extra},
           "training": {"max_epochs": P.max_epochs, "seed": P.seed, "resume_every": P.resume_every,
                        "smoke_steps": P.smoke_steps, "smoke_val_batches": P.smoke_val_batches,
                        "fid_max_frames": P.fid_max_frames},
           "versions": {k: versions.get(k) for k in ("python", "torch", "torchvision", "timm", "numpy", "Pillow",
                                                     "cuda", "cudnn")},
           "gpu_class": gpu_class, "publish_model": P.publish_model, "created_utc": None,
           "extra": {"versions_all": versions}}
    return json.loads(dc.json_dumps(cfg))


def stage_targets(rows, tgt_root, local_root, reuse_root, threads, log=print):
    """{relpath: local path} for every image / feature of the manifest. A Tol_66 local copy (same job cluster) is used
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


def stage_union(rows, roots, local_root, reuse_root, threads, log=print):
    """stage_targets per root: {(root tag, relpath): local path}. Rows carry `_root` ('base' | 'extra'); only the base can reuse
    a Tol_66 local copy. The two roots stage into separate local folders (their relative paths cannot collide)."""
    out = {}
    tot = {"files": 0, "copied": 0, "bytes_copied": 0, "seconds": 0.0, "reused_tol66_local": 0, "total": 0}
    for tag, root in roots.items():
        sub = [r for r in rows if r["_root"] == tag]
        if not sub:
            continue
        loc, st = stage_targets(sub, root, osp.join(local_root, "targets" if tag == "base" else "targets_extra"),
                                reuse_root if tag == "base" else None, threads, log)
        out.update({(tag, rel): p for rel, p in loc.items()})
        for k in ("files", "copied", "bytes_copied", "reused_tol66_local", "total"):
            tot[k] += st[k]
        tot["seconds"] = round(tot["seconds"] + st["seconds"], 1)
    return out, tot


def load_extra_targets(P, extra_root, base_rows, h):
    """The b1.6 extra targets: manifest_<target_set>.csv of the mining folder, pinned by sha256 (widget), checked row by row:
    role=train, whitelisted sequences only, none of the 4 val / 20 official-val / 25 test sequences, no (seq, fi) overlap with the
    base manifest, exactly EXTRA_N rows of which EXTRA_NIGHT night; the comp set only if trigger T1 fired; the sha256 must equal
    the one MINING_SUMMARY.json recorded; the (seq, fi) canonical sha256 of the rows must equal MINING_SUMMARY.json's
    set_canon_sha256 and, for random / random2, the pre-registered one (RANDOM_CANON_SHA256). Returns (rows, info)."""
    exp = h.get("extra_expect") or {}
    n_exp, night_exp = exp.get("n", EXTRA_N), exp.get("night", EXTRA_NIGHT)
    whitelist = tuple(exp.get("whitelist", WHITELIST31))
    if "whitelist" not in exp and hashlib.sha256("".join(f"{s}\n" for s in sorted(whitelist)).encode("utf-8")).hexdigest() != WHITELIST_SHA256:
        raise RuntimeError("WHITELIST31 differs from its pinned sha256")
    forbidden = set(VAL4) | set(OFFICIAL_VAL20) | set(TEST25)
    text = _read(f"{extra_root}/manifest_{P.target_set}.csv")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if P.extra_manifest_sha256 and sha != P.extra_manifest_sha256:
        raise RuntimeError(f"manifest_{P.target_set}.csv sha256 {sha[:16]} != the pinned extra_manifest_sha256 {P.extra_manifest_sha256[:16]}")
    mining = json.loads(_read(f"{extra_root}/MINING_SUMMARY.json"))
    if mining["file_sha256"].get(f"manifest_{P.target_set}.csv") != sha:
        raise RuntimeError(f"manifest_{P.target_set}.csv differs from the sha256 recorded in MINING_SUMMARY.json")
    if P.target_set == "comp" and not mining["T1"]["fired"]:
        raise RuntimeError("target_set=comp is trained only if trigger T1 fired (MINING_SUMMARY.json T1.fired is false)")
    rows = dc.read_manifest_csv(text)
    dc.validate_manifest(rows)
    bad_role = [r for r in rows if r["role"] != "train"]
    bad_seq = sorted({r["seq"] for r in rows if r["seq"] not in whitelist})
    hit = sorted({r["seq"] for r in rows} & forbidden)
    overlap = sorted({(r["seq"], r["fi"]) for r in rows} & {(r["seq"], r["fi"]) for r in base_rows})
    night = sum(1 for r in rows if r["night"])
    if bad_role or bad_seq or hit or overlap or len(rows) != n_exp or night != night_exp:
        raise RuntimeError(f"extra targets {P.target_set}: {len(rows)} rows (want {n_exp}), {night} night (want {night_exp}), "
                           f"{len(bad_role)} non-train rows, non-whitelisted {bad_seq[:3]}, forbidden {hit[:3]}, "
                           f"{len(overlap)} (seq, fi) shared with the base")
    # set identity, independent of the manifest bytes (defence in depth): sha256 of the sorted "seq,fi" lines = Tol_75's canon_sha256
    canon = hashlib.sha256("".join(f"{s_},{fi_}\n" for s_, fi_ in sorted((r["seq"], r["fi"]) for r in rows)).encode("utf-8")).hexdigest()
    prereg = (exp["canon"] if "canon" in exp else RANDOM_CANON_SHA256).get(P.target_set)
    if prereg is not None and canon != prereg:
        raise RuntimeError(f"manifest_{P.target_set}.csv holds another (seq, fi) set (canonical sha256 {canon[:16]}) than the "
                           f"pre-registered {P.target_set} draw ({prereg[:16]})")
    recorded_canon = (mining.get("set_canon_sha256") or {}).get(P.target_set)
    if recorded_canon != canon:
        raise RuntimeError(f"manifest_{P.target_set}.csv (seq, fi) canonical sha256 {canon[:16]} != MINING_SUMMARY.json's "
                           f"set_canon_sha256 {str(recorded_canon)[:16]}")
    return rows, {"dir": EXTRA_DIR, "target_set": P.target_set, "manifest_sha256": sha, "canon_sha256": canon, "n": len(rows), "night": night,
                  "mining_summary_sha256": hashlib.sha256(_read(f"{extra_root}/MINING_SUMMARY.json").encode("utf-8")).hexdigest(),
                  "t1_fired": bool(mining["T1"]["fired"]), "n_sequences": len({r["seq"] for r in rows})}


def spt_reference_diffs(ref_cfg, cfg, expected_seed, smoke=False):
    """b1.6 gate (DESIGN s.4): a new student's DISTILL_CONFIG equals pigtrack_v2_dp01's (S-PT) in the whole `recipe` block, in
    `training` except `seed` (which must be expected_seed), and in the BASE part of `data` (manifest, targets config, plan, teacher
    revision, val sequences, n_val). `run`, `arm`, `version`, `publish_model`, `data.extra`, `data.n_train` differ by design.
    `versions` / extra.versions_all differences are not fatal: they are returned to be recorded (DESIGN s.4, J15).
    smoke=True (a smoke run shortens the schedule by construction): the `training` keys in SMOKE_TRAINING_KEYS are not compared;
    the recipe, the seed, the base data and the versions are checked exactly as in a full run, so the first live read of S-PT's
    DISTILL_CONFIG.json on the Volume happens in the smoke task, not after the staging of the full task.
    Returns (fatal diffs, recorded diffs)."""
    bad = []
    for p, a, b in core.config_diff(ref_cfg.get("recipe"), cfg.get("recipe"), ()):
        bad.append((f"recipe.{p}" if p else "recipe", a, b))
    for p, a, b in core.config_diff(ref_cfg.get("training"), cfg.get("training"), ()):
        if p != "seed" and not (smoke and p.split(".")[0] in SMOKE_TRAINING_KEYS):
            bad.append((f"training.{p}", a, b))
    if cfg["training"]["seed"] != expected_seed:
        bad.append(("training.seed", expected_seed, cfg["training"]["seed"]))
    for k in ("targets_key", "manifest_sha256", "targets_config_sha256", "selection_plan_sha256", "teacher_revision", "val_seqs", "n_val"):
        if ref_cfg["data"].get(k) != cfg["data"].get(k):
            bad.append((f"data.{k}", ref_cfg["data"].get(k), cfg["data"].get(k)))
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
    lab = {P.arm: f"{P.run}: this run's student (base 2,626 train targets + {EXTRA_N:,} {P.target_set} extra targets; "
                  f"drop_path_rate={P.drop_path_rate:g} in training; seed {P.seed})",
           "S-PT": f"{REF_RUN} (models/sam3_student_tinyvit/{REF_RUN}), the headline base student, sha-pinned, read-only",
           "S-Edi": "published Edinburgh student (models.student_edinburgh, EMA epoch 75), zero-shot on PigTrack"}
    if P.run != CANONICAL_RUN:
        lab["S-PT-dp0"] = "pigtrack_v1 (models.student_pigtrack, drop_path_rate=0.0) = the ablation row S-PT-dp0"
    return lab


def main(raw, *, hooks):
    h = dict(hooks)
    log = h.get("log", _log)
    dbu = h.get("dbutils")
    rid = h.get("rid", RID)
    t_run = time.time()
    P = parse_params(raw)
    if recipe_for(0.0) != RECIPE or hashlib.sha256(dc.json_dumps(RECIPE).encode("utf-8")).hexdigest() != RECIPE_V1_SHA256:
        raise RuntimeError("RECIPE no longer equals pigtrack_v1's recorded recipe (RECIPE_V1_SHA256)")
    R = h.get("registry") or tol_paths.load(version=REGISTRY_VERSION)
    if h.get("check_deployed_paths", True):
        if core.sha256_file(R.path("admin.registry", None, "tol_paths.py")) != core.sha256_file(tol_paths.__file__):
            raise RuntimeError("the deployed tol_paths.py differs from the registry's copy: re-deploy it")
    TGT = R.path(TARGETS_KEY)
    EXTRA = R.path("paper1.root", None, "teacher_targets", EXTRA_DIR)
    RUN_DIR = R.out("paper1.distillation_runs", P.run)
    if osp.basename(R.out(MODEL_KEY)) != CKPT_NAME:
        raise RuntimeError(f"registry {MODEL_KEY} -> {R.out(MODEL_KEY)}: expected the file {CKPT_NAME}")
    MODEL_DIR = model_dir_for(R, P.run) if P.publish_model else None
    model_dst = f"{MODEL_DIR}/{CKPT_NAME}" if MODEL_DIR else None
    LOCAL = osp.join(h["local_root"], P.run)
    os.makedirs(LOCAL, exist_ok=True)

    def pub(local_path, rel):
        return core.publish_no_overwrite(local_path, f"{RUN_DIR}/{rel}", dbu)

    def pub_json(obj, rel):
        lp = core.write_text_atomic(osp.join(LOCAL, "pub", rel), dc.json_dumps(obj))
        return pub(lp, rel)

    # ---- data identity (a publishing run reuses the published J5 targets: b1.2 item 4)
    man_text = _read(f"{TGT}/manifest.csv")
    man_sha = hashlib.sha256(man_text.encode("utf-8")).hexdigest()
    expect_man = h.get("expect_manifest_sha256", EXPECT_MANIFEST_SHA256)
    if man_sha != expect_man:
        raise RuntimeError(f"manifest sha256 {man_sha[:16]} != the published J5 targets' {str(expect_man)[:16]}")
    rows = dc.read_manifest_csv(man_text)
    summ = dc.validate_manifest(rows)
    tcfg_text = _read(f"{TGT}/TARGETS_CONFIG.json")
    tcfg = json.loads(tcfg_text)
    if sorted(tcfg["val"]["seqs"]) != summ["val_seqs"]:
        raise RuntimeError(f"manifest val sequences {summ['val_seqs']} != TARGETS_CONFIG {tcfg['val']['seqs']}")
    if summ["rows"] != tcfg["selection"]["n_targets"]:
        raise RuntimeError(f"manifest rows {summ['rows']} != TARGETS_CONFIG n_targets {tcfg['selection']['n_targets']}")
    # ---- b1.6: base (pinned J5 manifest) UNION extra (pinned manifest of the mining folder); everything below sees the union
    base_rows, base_summ = rows, summ
    extra_rows, extra_info = load_extra_targets(P, EXTRA, base_rows, h)
    rows = [dict(r, _root="base") for r in base_rows] + [dict(r, _root="extra") for r in extra_rows]
    summ = dc.validate_manifest(rows)
    if (summ["train"], summ["val"], summ["val_seqs"]) != (base_summ["train"] + len(extra_rows), base_summ["val"], base_summ["val_seqs"]):
        raise RuntimeError(f"union manifest {summ['train']} train / {summ['val']} val != base {base_summ['train']} + "
                           f"{len(extra_rows)} extra / {base_summ['val']} (the val rows must be the base val rows only)")
    gpu = h["gpu_preflight"](P)
    td_p = f"{RUN_DIR}/train_done.json"
    ed_p = f"{RUN_DIR}/eval_done.json"
    need_train = not osp.isfile(td_p)
    need_eval = P.eval_stage != "defer" and not osp.isfile(ed_p)
    if need_train and P.eval_stage == "only":
        raise RuntimeError(f"eval_stage=only: {P.run} has no train_done.json (train it first)")
    if need_train and P.run in ARM_RUNS and gpu["class"] != "A100":
        raise RuntimeError(f"{P.run} trains on the A100 (recipe), not {gpu['class']}")
    cfg = build_config(P, man_sha, tcfg, hashlib.sha256(tcfg_text.encode("utf-8")).hexdigest(), summ,
                       h["versions"](), gpu["class"], extra_info)
    spt_ref = None
    if P.run in ARM_RUNS or P.smoke:        # a smoke run runs the gate too (schedule keys excepted): first live read of S-PT's config
        ref_p = R.path("paper1.distillation_runs", None, REF_RUN, "DISTILL_CONFIG.json")
        ref_cfg = json.loads(_read(ref_p))
        bad, recorded = spt_reference_diffs(ref_cfg, cfg, ARM_SEED_BY_SET[P.target_set], smoke=P.smoke)
        if bad:
            raise RuntimeError(f"{P.run} is not {REF_RUN}'s recipe / training / base data (b1.6 DESIGN s.4): "
                               + "; ".join(f"{p}: {a!r} -> {b!r}" for p, a, b in bad[:12]))
        spt_ref = {"run": REF_RUN, "config_sha256": hashlib.sha256(_read(ref_p).encode("utf-8")).hexdigest(),
                   "version_diffs": [list(d) for d in recorded]}
        cfg.setdefault("extra", {})["spt_reference"] = spt_ref
        log(f"gate vs {REF_RUN}{' (smoke: schedule keys not compared)' if P.smoke else ''}: recipe block equal key by key, training "
            f"equal except seed ({P.seed}), base data equal; {len(recorded)} version difference(s) recorded: {recorded[:4]}")
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
        local_of, st = stage_union(stage_rows, {"base": TGT, "extra": EXTRA}, LOCAL, h.get("reuse_root"),
                                   P.stage_threads, log)
        runlog["staging"] = st
        log(f"staged {st['total']} files: {st['reused_tol66_local']} verified Tol_66 local copies, {st['copied']} "
            f"copied ({st['bytes_copied'] / 1e9:.1f} GB, {st['seconds']} s)")

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
              "gpu_class": gpu["class"]}
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
        info = {"run": P.run, "rid": rid, "ckpt_sha256": td["ckpt_sha256"], "best_epoch": td["best_epoch"],
                "epochs_run": td["epochs_run"], "stopped": td["stopped"], "train_seconds": td["train_seconds"],
                "drop_path_rate": P.drop_path_rate, "model_published": published, "model_dir": model_dir_rel}
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        pub_json(dict(runlog, seconds=round(time.time() - t_run, 1), deferred=info), f"runs/{stamp}_{rid}.json")
        log(f"{P.run}: trained (best epoch {td['best_epoch']} of {td['epochs_run']}); fidelity deferred "
            f"(eval_stage=only on the A10)")
        return dict(info, status="trained_eval_deferred", run_dir=RUN_DIR)

    # ---- held-out fidelity (EMA best checkpoint, + S-PT-dp0 and the zero-shot Edinburgh student on the same frames)
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
                        "bands": FID_BANDS, "note": "Tol_02's a-priori bands; informative, no decision rests on them"}
        fid["training"] = {k: td[k] for k in ("best_epoch", "best_ema_val_loss", "epochs_run", "stopped",
                                              "train_seconds", "ckpt_sha256")}
        fid["labels"] = student_labels(P)
        fid["drop_path_rate_in_training"] = {P.arm: P.drop_path_rate, "S-PT": ARM_DROP_PATH, "S-PT-dp0": 0.0,
                                             "S-Edi": f"timm default ({TIMM_DEFAULT_DROP_PATH}, export did not set it)"}
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
    if P.run != CANONICAL_RUN:      # fields added by tol02b-2026-10-15.1 (pigtrack_v1's DONE.json predates them)
        done.update({"drop_path_rate": P.drop_path_rate, "model_dir": model_dir_rel,
                     "eval_gpu_class": ed.get("gpu_class"),
                     "heldout_by_light": {st: {grp: heads(fid["summary"][st], grp) for grp in ("day", "night")}
                                          for st in student_order(P.arm) if st in fid["summary"]}})
        done["arm"] = P.arm
        if "S-PT" in fid["summary"]:
            done["heldout_S-PT"] = heads(fid["summary"]["S-PT"])
        if "S-PT-dp0" in fid["summary"]:
            done["heldout_S-PT-dp0"] = heads(fid["summary"]["S-PT-dp0"])
        if fid.get("reference_reproduction"):
            done["reference_reproduction_status"] = fid["reference_reproduction"]["status"]
        if fid.get("reference_reproduction_spt"):
            done["reference_reproduction_spt_status"] = fid["reference_reproduction_spt"]["status"]
    pub_json(done, "DONE.json")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    pub_json(dict(runlog, seconds=round(time.time() - t_run, 1), done=done), f"runs/{stamp}_{rid}.json")
    log(f"DONE {P.run}: best epoch {td['best_epoch']}, held-out fp32 cos_patch "
        f"{done['heldout']['fp32']['cos_patch']:.4f} scale {done['heldout']['fp32']['scale_ratio']:.4f}")
    return dict(done, status="done", run_dir=RUN_DIR)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Selftest (local only): fake targets + registry, stub trainer / fidelity, resume + idempotence + publish guards

# COMMAND ----------

def _selftest():
    results = []

    def run(name, fn):
        try:
            fn()
            results.append((name, "PASS", ""))
        except Exception:  # noqa: BLE001
            results.append((name, "FAIL", traceback.format_exc()))

    reg_src = next((osp.join(d, "paths_v1.yaml") for d in (CODE_DIR, osp.join(CODE_DIR, "_vol"),
                                                           osp.join(osp.dirname(CODE_DIR), "catalog", "build"))
                    if osp.isfile(osp.join(d, "paths_v1.yaml"))), None)
    td = tempfile.mkdtemp(prefix="tol02c_st_")
    STUB_WL = ("pigtrack0003", "pigtrack0009")                       # the selftest's "whitelist" (extra_expect)
    STUB_V = {"torch": "stub", "timm": "1.0.25"}
    EXPECT = {"n": 4, "night": 2, "whitelist": STUB_WL, "canon": {}}     # canon {}: the stub sets are not the pre-registered R / R2

    def canon_of(pairs):                                                  # = Tol_75.canon_sha256
        return hashlib.sha256("".join(f"{a},{b}\n" for a, b in sorted(pairs)).encode("utf-8")).hexdigest()

    def raises(fn, exc, text=""):
        try:
            fn()
        except exc as e:
            assert text in str(e), (text, str(e))
            return
        raise AssertionError(f"no {exc.__name__} ({text})")

    def write(path, text_or_bytes):
        os.makedirs(osp.dirname(path), exist_ok=True)
        if isinstance(text_or_bytes, bytes):
            with open(path, "wb") as f:
                f.write(text_or_bytes)
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text_or_bytes)

    def make_world(tag):
        vol = osp.join(td, tag, "Volumes", "cat", "sch", "vol")
        text = re.sub(r'(?m)^  volume: ".*"$', f'  volume: "{vol.replace(chr(92), "/")}"', _read(reg_src))
        reg = osp.join(td, tag + "_paths_v1.yaml")
        write(reg, text)
        R = tol_paths.load(path=reg)
        TGT = R.out(TARGETS_KEY)
        EXTRA = R.path("paper1.root", None, "teacher_targets", EXTRA_DIR)
        base_rows = []
        for s, role, night in (("pigtrack0003", "train", 1), ("pigtrack0009", "train", 0), ("pigtrack0041", "val", 0)):
            for fi in (1, 4, 7):
                b = dc.target_basename(s, fi)
                relf, reli = f"seqs/{s}/features/{b}.pt", f"seqs/{s}/frames/{b}.jpg"
                write(osp.join(TGT, relf), os.urandom(64))
                write(osp.join(TGT, reli), b"jpeg" + bytes([fi]))
                base_rows.append({"seq": s, "role": role, "fi": fi, "mot_frame": fi + 1, "src_image": dc.src_image_name(fi),
                                  "image": reli, "feature": relf, "image_bytes": osp.getsize(osp.join(TGT, reli)),
                                  "image_sha256": core.sha256_file(osp.join(TGT, reli)), "feature_bytes": 64,
                                  "feature_sha256": core.sha256_file(osp.join(TGT, relf)), "feat_mean": 0.0, "feat_std": 1.4,
                                  "night": night, "pen": "2", "fps": 10.0, "width": 64, "height": 40, "n_images_seq": 9})
        write(f"{TGT}/manifest.csv", dc.manifest_csv(base_rows))
        tcfg_text = json.dumps({"val": {"seqs": ["pigtrack0041"]}, "selection": {"n_targets": 9, "plan_sha256": "p" * 64},
                                "teacher": {"revision": "3c879f39826c281e95690f02c7821c4de09afae7"}})
        write(f"{TGT}/TARGETS_CONFIG.json", tcfg_text)
        sh = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()
        man_sha = sh(_read(f"{TGT}/manifest.csv"))
        # extra targets: 4 rows per set (2 night from 0003, 2 day from 0009), different frames per set
        man_sha_extra, canon_extra = {}, {}
        sets = {"mined": (2, 5, 3, 6), "random": (6, 8, 2, 5), "random2": (0, 2, 5, 8), "comp": (2, 5, 3, 6)}
        for k, (a, b_, c, d) in sets.items():
            rows = []
            for s, fis, night in (("pigtrack0003", (a, b_), 1), ("pigtrack0009", (c, d), 0)):
                for fi in fis:
                    b = dc.target_basename(s, fi)
                    relf, reli = f"seqs/{s}/features/{b}.pt", f"seqs/{s}/frames/{b}.jpg"
                    if not osp.isfile(osp.join(EXTRA, relf)):
                        write(osp.join(EXTRA, relf), os.urandom(64))
                        write(osp.join(EXTRA, reli), b"jpeg" + bytes([fi, 9]))
                    rows.append({"seq": s, "role": "train", "fi": fi, "mot_frame": fi + 1, "src_image": dc.src_image_name(fi),
                                 "image": reli, "feature": relf, "image_bytes": osp.getsize(osp.join(EXTRA, reli)),
                                 "image_sha256": core.sha256_file(osp.join(EXTRA, reli)), "feature_bytes": 64,
                                 "feature_sha256": core.sha256_file(osp.join(EXTRA, relf)), "feat_mean": 0.0, "feat_std": 1.4,
                                 "night": night, "pen": "2", "fps": 10.0, "width": 64, "height": 40, "n_images_seq": 9})
            rows.sort(key=lambda r: (r["seq"], r["fi"]))
            write(f"{EXTRA}/manifest_{k}.csv", dc.manifest_csv(rows))
            man_sha_extra[k] = sh(_read(f"{EXTRA}/manifest_{k}.csv"))
            canon_extra[k] = canon_of((r["seq"], r["fi"]) for r in rows)
        write(f"{EXTRA}/MINING_SUMMARY.json", json.dumps({"file_sha256": {f"manifest_{k}.csv": v for k, v in man_sha_extra.items()},
                                                          "set_canon_sha256": canon_extra,
                                                          "T1": {"fired": False, "top_share": 0.3}}))
        # S-PT's published config (the reference of the recipe gate)
        ref_cfg = {"version": "tol02b-distill-v1", "run": REF_RUN, "recipe": recipe_for(0.1),
                   "data": {"targets_key": TARGETS_KEY, "manifest_sha256": man_sha, "targets_config_sha256": sh(tcfg_text),
                            "n_train": 6, "n_val": 3, "val_seqs": ["pigtrack0041"], "selection_plan_sha256": "p" * 64,
                            "teacher_revision": "3c879f39826c281e95690f02c7821c4de09afae7"},
                   "training": {"max_epochs": 100, "seed": 42, "resume_every": 10, "smoke_steps": 0, "smoke_val_batches": 0,
                                "fid_max_frames": 0},
                   "versions": {k: STUB_V.get(k) for k in ("python", "torch", "torchvision", "timm", "numpy", "Pillow", "cuda", "cudnn")},
                   "gpu_class": "A100", "publish_model": True, "extra": {"versions_all": dict(STUB_V)}}
        write(R.path("paper1.distillation_runs", None, REF_RUN, "DISTILL_CONFIG.json"), dc.json_dumps(ref_cfg))
        return {"R": R, "TGT": TGT, "EXTRA": EXTRA, "man_sha": man_sha, "extra_sha": man_sha_extra, "canon": canon_extra, "base_rows": base_rows,
                "ref_cfg": ref_cfg, "ref_path": R.path("paper1.distillation_runs", None, REF_RUN, "DISTILL_CONFIG.json"), "tag": tag}

    calls = {"train": 0, "fid": 0, "verify": 0, "pairs": []}

    def stub_train(ctx, train_pairs, val_pairs, resume_local):
        calls["train"] += 1
        P = ctx["P"]
        calls["pairs"].append((len(train_pairs), len(val_pairs), sorted({p["seq"] for p in val_pairs})))
        assert len(train_pairs) == 10 and len(val_pairs) == 3, (len(train_pairs), len(val_pairs))
        assert all(osp.isfile(p["image"]) and osp.isfile(p["feature"]) for p in train_pairs + val_pairs)
        assert all(p["seq"] != "pigtrack0041" for p in train_pairs) and all(p["seq"] == "pigtrack0041" for p in val_pairs)
        assert len({p["feature"] for p in train_pairs}) == 10, "union rows must be distinct files"
        start, hist = 0, []
        if resume_local:
            st = json.loads(_read(resume_local))
            start, hist = st["epoch"], st["history"]
        for ep in range(start, P.max_epochs):
            hist.append({"epoch": ep + 1, "ema_val_loss": 1.0 / (ep + 1), "ema_cosine": 0.2, "sec": 1.0})
            ctx["pub_json"](hist[-1], f"progress/ep{ep + 1:03d}_{ctx['rid']}.json")
            if (ep + 1) % P.resume_every == 0 or ep + 1 == P.stop_after_epoch:
                name = dc.state_name(ep + 1, ctx["rid"])
                lp = core.write_text_atomic(osp.join(ctx["local"], "st", name), json.dumps({"epoch": ep + 1, "history": hist}))
                ctx["pub"](lp, f"resume/{name}")
                ctx["pub_json"]({"bytes": osp.getsize(lp), "sha256": core.sha256_file(lp), "epoch": ep + 1}, f"resume/{dc.state_marker(name)}")
            if ep + 1 == P.stop_after_epoch:
                return {"stopped": "stop_after_epoch", "last_epoch": ep + 1}
        best = core.write_text_atomic(osp.join(ctx["local"], "best.pt"), "weights-" + str(len(hist)))
        return {"stopped": "max_epochs", "best_local": best, "history": hist, "best_metrics": hist[-1],
                "resumed_from": resume_local and osp.basename(resume_local)}

    def stub_fid(ctx, best_local, val):
        calls["fid"] += 1
        m = {k: 0.8 for k in dc.FID_KEYS}
        m["scale_ratio"] = 0.95
        names = list(student_order(ctx["P"].arm))
        rows_ = [dict({"seq": v["seq"], "night": v["night"], "fi": v["fi"]}, **{n: {"fp32": m, "bf16": m} for n in names}) for v in val]
        return {"summary": dc.fidelity_groups(rows_, tuple(names)), "n_frames": len(rows_)}, fidelity_csv(rows_, names)

    def hooks(w, rid, **kw):
        h = {"registry": w["R"], "check_deployed_paths": False, "local_root": osp.join(td, "local_" + w["tag"]), "dbutils": None,
             "log": lambda *a: None, "rid": rid, "reuse_root": None, "extra_expect": EXPECT,
             "gpu_preflight": lambda P: {"class": "A100", "name": "stub"},
             "versions": lambda: dict(STUB_V), "code_shas": lambda: {"nb": "stub"},
             "train": stub_train, "fidelity": stub_fid, "expect_manifest_sha256": w["man_sha"],
             "verify_published": lambda p, ctx: calls.__setitem__("verify", calls["verify"] + 1)}
        h.update(kw)
        return h

    def hooks_a10(w, rid, timm_v="1.0.25"):
        return hooks(w, rid, gpu_preflight=lambda P: {"class": "A10", "name": "stub"},
                     versions=lambda: {"torch": "stub", "timm": timm_v, "cuda": "other"})

    def arm_params(w, name="mined", **kw):
        run_, (set_, seed_, _arm) = {"mined": "pigtrack_mined_v1", "random": "pigtrack_random_v1", "random2": "pigtrack_random2_v1",
                                     "comp": "pigtrack_comp_v1"}[name], ARM_RUNS[{"mined": "pigtrack_mined_v1", "random": "pigtrack_random_v1",
                                                                              "random2": "pigtrack_random2_v1", "comp": "pigtrack_comp_v1"}[name]]
        d = {"run": run_, "target_set": set_, "seed": str(seed_), "drop_path_rate": "0.1", "extra_manifest_sha256": w["extra_sha"][set_]}
        d.update(kw)
        return d

    SM = {"run": "smoke_st", "smoke_steps": "2", "max_epochs": "3", "resume_every": "1", "fid_max_frames": "2", "drop_path_rate": "0.1"}

    W1 = make_world("w1")

    # -------------------------------------------------------------------------------------------------------- params
    def t_params():
        ok = arm_params(W1)
        P = parse_params(ok)
        assert P.arm == "S-PT-mined" and P.target_set == "mined" and P.seed == 42 and P.drop_path_rate == 0.1 and not P.publish_model
        assert parse_params(arm_params(W1, "random2")).arm == "S-PT-rand2" and parse_params(arm_params(W1, "random2")).seed == 43
        assert parse_params(arm_params(W1, "random")).arm == "S-PT-rand" and parse_params(arm_params(W1, "comp")).arm == "S-PT-comp"
        S = parse_params(dict(SM, target_set="random"))
        assert S.smoke and S.arm == "S-PT-rand" and not S.publish_model
        for bad in (dict(ok, run="pigtrack_v1"), dict(ok, run="pigtrack_v2_dp01"), dict(ok, target_set="random"),
                    dict(ok, seed="43"), dict(arm_params(W1, "random2"), seed="42"), dict(ok, max_epochs="99"),
                    dict(ok, drop_path_rate="0.0"), dict(ok, drop_path_rate="0.2"), dict(ok, extra_manifest_sha256=""),
                    dict(ok, extra_manifest_sha256="xyz"), dict(ok, target_set="nope"), dict(ok, publish_model="true"),
                    dict(ok, bogus="1"), dict(ok, run="../x"), dict(SM, smoke_steps="0", fid_max_frames="0"), dict(ok, smoke_steps="3"),
                    dict(SM, publish_model="true"), dict(SM, target_set="nope"), {}):
            raises(lambda bad=bad: parse_params(bad), ValueError)
        assert parse_params(dict(ok, publish_model="auto")).publish_model is False and parse_params(dict(ok, publish_model="false")).publish_model is False

    # -------------------------------------------------------------------------------------------------------- pure helpers
    def t_extra_checks():
        P = parse_params(arm_params(W1))
        base = dc.read_manifest_csv(_read(f"{W1['TGT']}/manifest.csv"))
        rows, info = load_extra_targets(P, W1["EXTRA"], base, {"extra_expect": EXPECT})
        assert len(rows) == 4 and info["night"] == 2 and info["manifest_sha256"] == W1["extra_sha"]["mined"] and not info["t1_fired"]
        assert info["canon_sha256"] == W1["canon"]["mined"] == canon_of((r["seq"], r["fi"]) for r in rows)
        # canonical (seq, fi) identity (defence in depth): the pre-registered sha256 for random / random2, MINING_SUMMARY's for all four
        assert set(RANDOM_CANON_SHA256) == {"random", "random2"} and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in RANDOM_CANON_SHA256.values())
        prod = {k: v for k, v in EXPECT.items() if k != "canon"}              # production: the typed pre-registered values apply
        for nm in ("random", "random2"):
            raises(lambda nm=nm: load_extra_targets(parse_params(arm_params(W1, nm)), W1["EXTRA"], base, {"extra_expect": prod}),
                   RuntimeError, f"pre-registered {nm} draw")
        raises(lambda: load_extra_targets(P, W1["EXTRA"], base, {"extra_expect": dict(EXPECT, canon={"mined": "0" * 64})}),
               RuntimeError, "pre-registered mined draw")
        assert load_extra_targets(P, W1["EXTRA"], base, {"extra_expect": dict(EXPECT, canon={"mined": W1["canon"]["mined"]})})[1]["n"] == 4
        wc = make_world("w_canon")
        smc = json.loads(_read(f"{wc['EXTRA']}/MINING_SUMMARY.json"))
        smc["set_canon_sha256"]["mined"] = "0" * 64
        write(f"{wc['EXTRA']}/MINING_SUMMARY.json", json.dumps(smc))
        raises(lambda: load_extra_targets(parse_params(dict(SM, target_set="mined")), wc["EXTRA"], base, {"extra_expect": EXPECT}),
               RuntimeError, "set_canon_sha256")
        del smc["set_canon_sha256"]
        write(f"{wc['EXTRA']}/MINING_SUMMARY.json", json.dumps(smc))
        raises(lambda: load_extra_targets(parse_params(dict(SM, target_set="mined")), wc["EXTRA"], base, {"extra_expect": EXPECT}),
               RuntimeError, "set_canon_sha256")
        # the manifest bytes AND its recorded sha256 edited consistently (a different row set under a matching file sha256): the
        # canonical identity still refuses it
        wd = make_world("w_canon2")
        txd = _read(f"{wd['EXTRA']}/manifest_mined.csv")
        rows_d = dc.read_manifest_csv(txd)
        alt = dict(rows_d[0], fi=rows_d[0]["fi"] + 20, mot_frame=rows_d[0]["fi"] + 21, src_image=dc.src_image_name(rows_d[0]["fi"] + 20))
        txd2 = dc.manifest_csv([alt] + rows_d[1:])
        write(f"{wd['EXTRA']}/manifest_mined.csv", txd2)
        smd = json.loads(_read(f"{wd['EXTRA']}/MINING_SUMMARY.json"))
        smd["file_sha256"]["manifest_mined.csv"] = hashlib.sha256(txd2.encode("utf-8")).hexdigest()
        write(f"{wd['EXTRA']}/MINING_SUMMARY.json", json.dumps(smd))
        raises(lambda: load_extra_targets(parse_params(dict(SM, target_set="mined")), wd["EXTRA"], base, {"extra_expect": EXPECT}),
               RuntimeError, "set_canon_sha256")
        # wrong pinned sha / wrong n / wrong night / non-whitelisted sequence / forbidden name / overlap with the base / comp without T1
        raises(lambda: load_extra_targets(parse_params(arm_params(W1, extra_manifest_sha256="0" * 64)), W1["EXTRA"], base,
                                          {"extra_expect": EXPECT}), RuntimeError, "pinned extra_manifest_sha256")
        raises(lambda: load_extra_targets(P, W1["EXTRA"], base, {"extra_expect": dict(EXPECT, n=5)}), RuntimeError, "want 5")
        raises(lambda: load_extra_targets(P, W1["EXTRA"], base, {"extra_expect": dict(EXPECT, night=3)}), RuntimeError, "want 3")
        raises(lambda: load_extra_targets(P, W1["EXTRA"], base, {"extra_expect": dict(EXPECT, whitelist=("pigtrack0003",))}),
               RuntimeError, "non-whitelisted")
        shifted = [dict(r, fi=2) if (r["seq"], r["fi"]) == ("pigtrack0003", 1) else r for r in base]       # base now holds (0003, 2)
        raises(lambda: load_extra_targets(P, W1["EXTRA"], shifted, {"extra_expect": EXPECT}), RuntimeError, "shared with the base")
        raises(lambda: load_extra_targets(parse_params(arm_params(W1, "comp")), W1["EXTRA"], base, {"extra_expect": EXPECT}),
               RuntimeError, "T1 fired")
        # a manifest edited after mining no longer matches MINING_SUMMARY.json
        w = make_world("w_extra")
        write(f"{w['EXTRA']}/manifest_mined.csv", _read(f"{w['EXTRA']}/manifest_mined.csv") + "\n")
        raises(lambda: load_extra_targets(parse_params(dict(SM, target_set="mined")), w["EXTRA"], base, {"extra_expect": EXPECT}),
               RuntimeError, "MINING_SUMMARY.json")
        # a duplicated (seq, fi) row inside the extra manifest is refused (dedup)
        w3 = make_world("w_dup")
        lines = _read(f"{w3['EXTRA']}/manifest_mined.csv").splitlines()
        tx3 = "\n".join(lines + [lines[1]]) + "\n"
        write(f"{w3['EXTRA']}/manifest_mined.csv", tx3)
        sm3 = json.loads(_read(f"{w3['EXTRA']}/MINING_SUMMARY.json"))
        sm3["file_sha256"]["manifest_mined.csv"] = hashlib.sha256(tx3.encode("utf-8")).hexdigest()
        write(f"{w3['EXTRA']}/MINING_SUMMARY.json", json.dumps(sm3))
        raises(lambda: load_extra_targets(parse_params(dict(SM, target_set="mined")), w3["EXTRA"], base, {"extra_expect": EXPECT}),
               ValueError, "duplicate")
        # forbidden names (val / official-val / test) in the extra manifest are refused even when whitelisted by mistake
        w2 = make_world("w_forbid")
        tx = _read(f"{w2['EXTRA']}/manifest_mined.csv").replace("pigtrack0009", "pigtrack0002")
        write(f"{w2['EXTRA']}/manifest_mined.csv", tx)
        sm = json.loads(_read(f"{w2['EXTRA']}/MINING_SUMMARY.json"))
        sm["file_sha256"]["manifest_mined.csv"] = hashlib.sha256(tx.encode("utf-8")).hexdigest()
        write(f"{w2['EXTRA']}/MINING_SUMMARY.json", json.dumps(sm))
        raises(lambda: load_extra_targets(parse_params(dict(SM, target_set="mined")), w2["EXTRA"], base,
                                          {"extra_expect": dict(EXPECT, whitelist=STUB_WL + ("pigtrack0002",))}),
               RuntimeError, "forbidden ['pigtrack0002']")

    def t_staging_union():
        rows = [dict(r, _root="base") for r in W1["base_rows"]]
        xr = dc.read_manifest_csv(_read(f"{W1['EXTRA']}/manifest_mined.csv"))
        rows += [dict(r, _root="extra") for r in xr]
        out, st = stage_union(rows, {"base": W1["TGT"], "extra": W1["EXTRA"]}, osp.join(td, "st_local"), None, 4)
        assert st["total"] == 2 * (9 + 4) and st["copied"] == 2 * 13, st
        for r in rows:
            assert core.sha256_file(out[(r["_root"], r["feature"])]) == r["feature_sha256"]
            assert core.sha256_file(out[(r["_root"], r["image"])]) == r["image_sha256"]
        assert len({p for p in out.values()}) == len(out), "two roots must not share a local file"
        # val-only staging (eval_stage=only) touches the base root only
        out2, st2 = stage_union([r for r in rows if r["role"] == "val"], {"base": W1["TGT"], "extra": W1["EXTRA"]}, osp.join(td, "st_local2"), None, 4)
        assert st2["total"] == 6 and all(k[0] == "base" for k in out2)

    def t_gate_unit():
        P = parse_params(arm_params(W1))
        cfg = build_config(P, W1["man_sha"], json.loads(_read(f"{W1['TGT']}/TARGETS_CONFIG.json")),
                           W1["ref_cfg"]["data"]["targets_config_sha256"], {"train": 10, "val": 3, "val_seqs": ["pigtrack0041"]},
                           dict(STUB_V), "A100", {"x": 1})
        bad, rec = spt_reference_diffs(W1["ref_cfg"], cfg, 42)
        assert bad == [] and rec == [], (bad, rec)
        # recorded, not fatal: a different torch
        cfg2 = copy.deepcopy(cfg)
        cfg2["versions"]["torch"] = "2.8"
        cfg2["extra"] = {"versions_all": dict(STUB_V, transformers="5.2.0")}
        bad, rec = spt_reference_diffs(W1["ref_cfg"], cfg2, 42)
        assert bad == [] and {p for p, _a, _b in rec} == {"versions.torch", "extra.versions_all.transformers"}, rec
        # fatal: recipe, training, base data, seed
        r2 = copy.deepcopy(cfg)
        r2["recipe"]["loss"]["cos_w"] = 0.4
        assert [p for p, _a, _b in spt_reference_diffs(W1["ref_cfg"], r2, 42)[0]] == ["recipe.loss.cos_w"]
        r3 = copy.deepcopy(cfg)
        r3["training"]["max_epochs"] = 99
        assert [p for p, _a, _b in spt_reference_diffs(W1["ref_cfg"], r3, 42)[0]] == ["training.max_epochs"]
        assert [p for p, _a, _b in spt_reference_diffs(W1["ref_cfg"], cfg, 43)[0]] == ["training.seed"]
        r4 = copy.deepcopy(cfg)
        r4["training"]["seed"] = 43
        assert spt_reference_diffs(W1["ref_cfg"], r4, 43)[0] == [] and [p for p, *_ in spt_reference_diffs(W1["ref_cfg"], r4, 42)[0]] == ["training.seed"]
        r5 = copy.deepcopy(cfg)
        r5["data"]["manifest_sha256"] = "0" * 64
        r5["data"]["val_seqs"] = ["pigtrack0050"]
        assert sorted(p for p, *_ in spt_reference_diffs(W1["ref_cfg"], r5, 42)[0]) == ["data.manifest_sha256", "data.val_seqs"]
        assert cfg["arm"] == "S-PT-mined" and cfg["data"]["extra"] == {"x": 1} and cfg["publish_model"] is False
        # smoke mode: the schedule keys are not compared, everything else is
        r6 = copy.deepcopy(cfg)
        r6["training"].update({"max_epochs": 3, "resume_every": 1, "smoke_steps": 20, "smoke_val_batches": 5, "fid_max_frames": 8})
        assert spt_reference_diffs(W1["ref_cfg"], r6, 42, smoke=True)[0] == []
        assert sorted(p for p, *_ in spt_reference_diffs(W1["ref_cfg"], r6, 42)[0]) == sorted(
            f"training.{k}" for k in SMOKE_TRAINING_KEYS), "a full run still compares every training key"
        assert [p for p, *_ in spt_reference_diffs(W1["ref_cfg"], r6, 43, smoke=True)[0]] == ["training.seed"]
        r6["training"]["unknown_key"] = 1
        assert [p for p, *_ in spt_reference_diffs(W1["ref_cfg"], r6, 42, smoke=True)[0]] == ["training.unknown_key"]
        r7 = copy.deepcopy(cfg)
        r7["recipe"]["loss"]["cos_w"] = 0.4
        r7["data"]["manifest_sha256"] = "0" * 64
        assert sorted(p for p, *_ in spt_reference_diffs(W1["ref_cfg"], r7, 42, smoke=True)[0]) == ["data.manifest_sha256", "recipe.loss.cos_w"]
        assert ARM_SEED_BY_SET == {"mined": 42, "random": 42, "random2": 43, "comp": 42}

    # -------------------------------------------------------------------------------------------------------- end to end (stubs)
    def t_smoke_stop_and_resume():
        S = dict(SM, target_set="mined")
        out = main(dict(S, stop_after_epoch="2"), hooks=hooks(W1, "aaaaaa"))
        assert out["status"] == "stopped_after_epoch" and out["last_epoch"] == 2, out
        rd = W1["R"].out("paper1.distillation_runs", "smoke_st")
        assert dc.latest_state(os.listdir(f"{rd}/resume"))[0] == 2 and not osp.isfile(f"{rd}/train_done.json")
        out = main(S, hooks=hooks(W1, "bbbbbb"))
        assert out["status"] == "done" and out["epochs_run"] == 3, out
        tdn = json.loads(_read(f"{rd}/train_done.json"))
        assert tdn["resumed_from"] == "state_ep002_aaaaaa.pt" and tdn["final_dir"] == "final_bbbbbb", tdn
        assert not osp.exists(osp.join(W1["R"].volume, "models")), "a b1.6 run must never write models/"
        n = calls["train"]
        assert main(S, hooks=hooks(W1, "cccccc")).get("skipped") and calls["train"] == n
        cfg = json.loads(_read(f"{rd}/DISTILL_CONFIG.json"))
        assert cfg["arm"] == "S-PT-mined" and cfg["data"]["extra"]["target_set"] == "mined" and cfg["data"]["n_train"] == 10
        assert cfg["data"]["extra"]["manifest_sha256"] == W1["extra_sha"]["mined"]
        assert cfg["data"]["extra"]["canon_sha256"] == W1["canon"]["mined"]
        # the smoke task runs the S-PT gate (schedule keys excepted) and records it, like a full run
        assert cfg["extra"]["spt_reference"]["run"] == REF_RUN and cfg["extra"]["spt_reference"]["version_diffs"] == []
        assert calls["pairs"][-1] == (10, 3, ["pigtrack0041"])

    def t_smoke_gate():
        """A smoke run reads S-PT's DISTILL_CONFIG and refuses a different recipe / base data / seed BEFORE writing anything."""
        w = make_world("w_sgate")
        ref = copy.deepcopy(w["ref_cfg"])
        ref["recipe"]["loss"]["cos_w"] = 0.4
        write(w["ref_path"], dc.json_dumps(ref))
        S = dict(SM, run="smoke_sg", target_set="mined")
        raises(lambda: main(S, hooks=hooks(w, "e1e1e1")), RuntimeError, "recipe.loss.cos_w")
        assert not osp.exists(w["R"].out("paper1.distillation_runs", "smoke_sg") + "/DISTILL_CONFIG.json"), "a refused smoke must not write its config"
        w2 = make_world("w_sgate2")                                              # random2 trains with seed 43: a smoke with seed 42 is refused
        raises(lambda: main(dict(SM, run="smoke_sg2", target_set="random2"), hooks=hooks(w2, "e2e2e2")), RuntimeError, "training.seed")
        out = main(dict(SM, run="smoke_sg2", target_set="random2", seed="43", stop_after_epoch="1"), hooks=hooks(w2, "e3e3e3"))
        assert out["status"] == "stopped_after_epoch", out
        cfg = json.loads(_read(w2["R"].out("paper1.distillation_runs", "smoke_sg2") + "/DISTILL_CONFIG.json"))
        assert cfg["training"]["seed"] == 43 and cfg["extra"]["spt_reference"]["run"] == REF_RUN
        w3 = make_world("w_sgate3")                                              # a missing S-PT config also stops the smoke
        os.remove(w3["ref_path"])
        raises(lambda: main(dict(SM, run="smoke_sg3", target_set="mined"), hooks=hooks(w3, "e4e4e4")), (OSError, FileNotFoundError))

    def t_config_guard():
        raises(lambda: main(dict(SM, target_set="mined", max_epochs="4"), hooks=hooks(W1, "ffffff")), RuntimeError, "mismatch")
        raises(lambda: main(dict(SM, target_set="random"), hooks=hooks(W1, "ffffff")), RuntimeError, "mismatch")     # another set = another data identity

    def t_mined_defer_then_eval():
        """The full-run rules: A100 train (deferred) -> A10 fidelity -> DONE; the gate vs S-PT; nothing in models/."""
        P = arm_params(W1)
        rd = W1["R"].out("paper1.distillation_runs", "pigtrack_mined_v1")
        raises(lambda: main(dict(P, eval_stage="defer", require_gpu="A10"), hooks=hooks_a10(W1, "a1a1a1")), RuntimeError, "trains on the A100")
        raises(lambda: main(dict(P, eval_stage="only", require_gpu="A10"), hooks=hooks_a10(W1, "a1a1a1")), RuntimeError, "train it first")
        raises(lambda: main(dict(P, eval_stage="defer"), hooks=hooks(W1, "a1a1a1", expect_manifest_sha256="0" * 64)), RuntimeError, "manifest sha256")
        raises(lambda: main(dict(P, eval_stage="defer", extra_manifest_sha256=W1["extra_sha"]["random"]), hooks=hooks(W1, "a1a1a1")),
               RuntimeError, "pinned extra_manifest_sha256")
        assert not osp.exists(f"{rd}/DISTILL_CONFIG.json"), "a refused run must not write its config"
        n_t, n_f = calls["train"], calls["fid"]
        out = main(dict(P, eval_stage="defer"), hooks=hooks(W1, "a2a2a2"))
        assert out["status"] == "trained_eval_deferred" and out["drop_path_rate"] == 0.1 and out["model_published"] == {} and out["model_dir"] is None, out
        assert (calls["train"], calls["fid"]) == (n_t + 1, n_f)
        cfg = json.loads(_read(f"{rd}/DISTILL_CONFIG.json"))
        assert cfg["arm"] == "S-PT-mined" and cfg["recipe"] == W1["ref_cfg"]["recipe"] and cfg["training"] == W1["ref_cfg"]["training"]
        assert cfg["data"]["n_train"] == 10 and cfg["data"]["n_val"] == 3 and cfg["data"]["val_seqs"] == ["pigtrack0041"]
        assert cfg["extra"]["spt_reference"]["run"] == REF_RUN and cfg["extra"]["spt_reference"]["version_diffs"] == []
        assert cfg["publish_model"] is False and cfg["gpu_class"] == "A100"
        raises(lambda: main(dict(P, eval_stage="only", require_gpu="A10"), hooks=hooks_a10(W1, "a4a4a4", "1.0.30")), RuntimeError, "timm")
        out = main(dict(P, eval_stage="only", require_gpu="A10"), hooks=hooks_a10(W1, "a5a5a5"))
        assert out["status"] == "done" and out["arm"] == "S-PT-mined" and out["eval_gpu_class"] == "A10" and "heldout_S-PT" in out and "heldout_S-PT-dp0" in out, out
        assert out["heldout_by_light"]["S-PT-mined"]["night"] == {"fp32": None, "bf16": None}
        assert set(out["heldout_by_light"]) == {"S-PT-mined", "S-PT", "S-PT-dp0", "S-Edi"}
        fid = json.loads(_read(f"{rd}/{json.loads(_read(rd + '/eval_done.json'))['eval_dir']}/fidelity_heldout.json"))
        assert set(fid["summary"]) == {"S-PT-mined", "S-PT", "S-PT-dp0", "S-Edi"}
        assert fid["drop_path_rate_in_training"]["S-PT-mined"] == 0.1 and fid["drop_path_rate_in_training"]["S-PT"] == 0.1
        assert set(fid["labels"]) == {"S-PT-mined", "S-PT", "S-PT-dp0", "S-Edi"} and "2,125 mined" in fid["labels"]["S-PT-mined"]
        assert not osp.exists(osp.join(W1["R"].volume, "models")), "a b1.6 run must never write models/"
        assert main(dict(P, eval_stage="only", require_gpu="A10"), hooks=hooks_a10(W1, "a6a6a6")).get("skipped")

    def t_random2_seed():
        P = arm_params(W1, "random2")
        out = main(dict(P, eval_stage="defer"), hooks=hooks(W1, "b1b1b1"))
        assert out["status"] == "trained_eval_deferred"
        cfg = json.loads(_read(W1["R"].out("paper1.distillation_runs", "pigtrack_random2_v1") + "/DISTILL_CONFIG.json"))
        assert cfg["training"]["seed"] == 43 and cfg["arm"] == "S-PT-rand2" and cfg["data"]["extra"]["target_set"] == "random2"
        assert cfg["recipe"] == W1["ref_cfg"]["recipe"]

    def t_gate_blocks_a_run():
        w = make_world("w_gate")
        ref = copy.deepcopy(w["ref_cfg"])
        ref["recipe"]["loss"]["cos_w"] = 0.4
        write(w["ref_path"], dc.json_dumps(ref))
        raises(lambda: main(dict(arm_params(w), eval_stage="defer"), hooks=hooks(w, "c1c1c1")), RuntimeError, "recipe.loss.cos_w")
        assert not osp.exists(w["R"].out("paper1.distillation_runs", "pigtrack_mined_v1") + "/DISTILL_CONFIG.json")
        # version differences are recorded, not fatal
        w2 = make_world("w_gate2")
        ref2 = copy.deepcopy(w2["ref_cfg"])
        ref2["versions"]["torch"] = "other"
        write(w2["ref_path"], dc.json_dumps(ref2))
        out = main(dict(arm_params(w2), eval_stage="defer"), hooks=hooks(w2, "c2c2c2"))
        assert out["status"] == "trained_eval_deferred"
        cfg = json.loads(_read(w2["R"].out("paper1.distillation_runs", "pigtrack_mined_v1") + "/DISTILL_CONFIG.json"))
        assert [d[0] for d in cfg["extra"]["spt_reference"]["version_diffs"]] == ["versions.torch"]

    def t_comp_gate():
        w = make_world("w_comp")
        raises(lambda: main(dict(arm_params(w, "comp"), eval_stage="defer"), hooks=hooks(w, "d1d1d1")), RuntimeError, "T1 fired")
        sm = json.loads(_read(f"{w['EXTRA']}/MINING_SUMMARY.json"))
        sm["T1"]["fired"] = True
        write(f"{w['EXTRA']}/MINING_SUMMARY.json", json.dumps(sm))
        out = main(dict(arm_params(w, "comp"), eval_stage="defer"), hooks=hooks(w, "d2d2d2"))
        assert out["status"] == "trained_eval_deferred"
        cfg = json.loads(_read(w["R"].out("paper1.distillation_runs", "pigtrack_comp_v1") + "/DISTILL_CONFIG.json"))
        assert cfg["arm"] == "S-PT-comp" and cfg["data"]["extra"]["t1_fired"] is True and cfg["training"]["seed"] == 42

    def t_model_dirs_inert():
        assert PUBLISH_RUNS == {}
        raises(lambda: model_dir_for(W1["R"], "pigtrack_v2_dp01"), ValueError)
        raises(lambda: model_dir_for(W1["R"], "pigtrack_mined_v1"), ValueError)
        assert model_dir_for(W1["R"], "pigtrack_v1").endswith("/models/sam3_student_tinyvit/pigtrack_v1")

    # ---- unchanged Tol_02b checks that still apply to the shared recipe code
    def t_recipe():
        assert hashlib.sha256(dc.json_dumps(recipe_for(0.0)).encode("utf-8")).hexdigest() == RECIPE_V1_SHA256
        r1 = recipe_for(0.1)
        assert {p.split(".")[0] for p, _a, _b in core.config_diff(recipe_for(0.0), r1, ())} == set(DP_RECIPE_KEYS)
        assert r1["drop_path_rate"] == 0.1 and RECIPE["drop_path_rate"] == 0.0 and "drop_path_scope" not in RECIPE
        base = {"version": CONFIG_VERSION, "run": "pigtrack_v1", "recipe": recipe_for(0.0), "data": {"m": 1},
                "training": {"max_epochs": 100}, "publish_model": True, "gpu_class": "A100", "versions": {"timm": "1.0.25"}}
        new = dict(base, run="pigtrack_v2_dp01", recipe=r1, gpu_class="A10", versions={"timm": "x"})
        assert recipe_match_diffs(base, new) == []
        bad = copy.deepcopy(r1)
        bad["loss"]["cos_w"] = 0.4
        assert [p for p, _a, _b in recipe_match_diffs(base, dict(new, recipe=bad))] == ["recipe.loss.cos_w"]

    def t_shim():
        class FakeTimm:
            other = "attr"

            def create_model(self, name, *a, **kw):
                return ("model", name, kw)

        fm = types.SimpleNamespace(timm=FakeTimm())
        real = fm.timm

        def build():
            return fm.timm.create_model("tiny_vit_21m_512", pretrained=True, features_only=True, out_indices=(1, 2, 3), drop_path_rate=0.0)

        m = build_with_drop_path(build, fm, 0.1)
        assert m[2] == {"pretrained": True, "features_only": True, "out_indices": (1, 2, 3), "drop_path_rate": 0.1}, m
        assert fm.timm is real
        assert build_with_drop_path(build, fm, 0.0)[2]["drop_path_rate"] == 0.0 and fm.timm is real
        for b in (lambda: fm.timm.create_model("vit_x", drop_path_rate=0.0), lambda: fm.timm.create_model("tiny_vit_21m_512"),
                  lambda: "no call", lambda: (build(), build())):
            raises(lambda b=b: build_with_drop_path(b, fm, 0.1), RuntimeError)
            assert fm.timm is real, "the core's timm global must be restored after a failure"
        csrc = _read(core.__file__)
        blk = csrc[csrc.index("# >>> VERBATIM student_fixed.txt cell 13"):csrc.index("# <<< VERBATIM student")]
        assert blk.count("timm.create_model(") == 1 and blk.count("drop_path_rate=0.0,") == 1 and "'tiny_vit_21m_512'" in blk

    def t_reproduction():
        def met(v):
            d = {k: v for k in dc.FID_KEYS}
            d["scale_ratio"], d["mse"] = 0.9 + v / 10, 1.0 + v
            return d

        old_rows, cur = [], []
        for i, (s, night) in enumerate((("pigtrack0041", 0), ("pigtrack0041", 0), ("pigtrack0064", 1))):
            a, b, c = met(0.6 + i / 100), met(0.5 + i / 100), met(0.7 + i / 100)
            old_rows.append({"seq": s, "fi": i, "night": night, "S-PT": {"fp32": a, "bf16": a}, "S-PT-dp0": {"fp32": c, "bf16": c},
                             "S-Edi": {"fp32": b, "bf16": b}})
            cur.append({"seq": s, "fi": i, "night": night, "S-PT-mined": {"fp32": met(0.9), "bf16": met(0.9)}, "S-PT": {"fp32": dict(a), "bf16": dict(a)},
                        "S-PT-dp0": {"fp32": dict(c), "bf16": dict(c)}, "S-Edi": {"fp32": dict(b), "bf16": dict(b)}})
        ref_csv = fidelity_csv(old_rows, ["S-PT", "S-PT-dp0", "S-Edi"])
        ref_sum = dc.fidelity_groups(old_rows, ("S-PT", "S-PT-dp0", "S-Edi"))
        cur_sum = dc.fidelity_groups(cur, student_order("S-PT-mined"))
        rep_ = reference_reproduction(cur, ref_csv, ref_sum, cur_sum, repro_map=REPRO_MAP_V2)
        assert rep_["status"] == "identical" and rep_["n_frames_compared"] == 3 and rep_["missing"] == 0, rep_
        assert len(rep_["per_frame"]) == 24 and len(rep_["summary_mean_abs_diff"]) == 3 * 2 * 3 * 4
        # the default map is Tol_02b's: S-PT-dp0 (here = old column S-PT) is NOT comparable -> differs, proving the map matters
        assert reference_reproduction(cur, ref_csv)["status"] == "differs"
        cur[1]["S-PT"]["fp32"]["mse"] *= 1.01
        rep_ = reference_reproduction(cur, ref_csv, repro_map=REPRO_MAP_V2)
        assert rep_["status"] == "differs" and abs(rep_["per_frame"]["S-PT/fp32/mse"]["max"] - 0.01) < 1e-9
        assert student_order("S-PT-rand2") == ("S-PT-rand2", "S-PT", "S-PT-dp0", "S-Edi")

    try:
        for name, fn in (("params: arm / set / seed / epochs / drop path / pinned sha / no publish", t_params),
                         ("extra-targets checks: sha, n, night, whitelist, forbidden, overlap, T1, MINING_SUMMARY, canonical sha", t_extra_checks),
                         ("stage_union: two roots, val-only", t_staging_union),
                         ("S-PT gate: recipe, training, base data, seed; versions recorded", t_gate_unit),
                         ("smoke: union pairs, stop, resume, done no-op, no models/", t_smoke_stop_and_resume),
                         ("smoke runs the S-PT gate (schedule keys excepted) before writing anything", t_smoke_gate),
                         ("config guard (epochs / target set)", t_config_guard),
                         ("pigtrack_mined_v1: A100 train (deferred) -> A10 eval -> DONE", t_mined_defer_then_eval),
                         ("pigtrack_random2_v1: seed 43 passes the gate", t_random2_seed),
                         ("S-PT gate blocks a run; version diffs recorded", t_gate_blocks_a_run),
                         ("comp only if T1 fired", t_comp_gate), ("publish paths inert", t_model_dirs_inert),
                         ("recipe: dp 0.0 = pigtrack_v1 byte for byte", t_recipe), ("timm drop_path shim", t_shim),
                         ("reproduction vs S-PT's evaluation (map V2)", t_reproduction)):
            run(name, fn)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    for name, status, detail in results:
        print(f"{status}  {name}")
        if status == "FAIL":
            print(detail)
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"Tol_02c SELFTEST: {len(results) - n_fail}/{len(results)} passed")
    return n_fail == 0


# COMMAND ----------

if SELFTEST and IS_NOTEBOOK:
    raise RuntimeError("the selftest is local-only: TOL02C_SELFTEST=1 python Tol_02c_distill_pigtrack_mined.py")
if SELFTEST or not IS_NOTEBOOK:
    ok = _selftest() if SELFTEST else True
    if not SELFTEST:
        print("not a notebook and TOL02C_SELFTEST != 1: nothing to run")
    raise SystemExit(0 if ok else 1)

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


# ---- stochastic depth (BENCH_PROTOCOL §15 b1.2): training only, never at evaluation / export
def make_student(drop_path_rate, pretrained=True):
    """core.StudentEncoderTinyViT (verbatim export class + [bench-C1]) with `drop_path_rate` in its timm call. 0.0 = the
    class exactly as before (pigtrack_v1). The parameters / state_dict do not depend on the rate (DropPath holds none),
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
    """The b1.2 item 5 selftest, a hard gate before training (GPU, real timm). RNG-neutral: every RNG state is restored,
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
        assert_eval_mode(student, f"epoch {epoch + 1} val")          # stochastic depth inactive (b1.2 item 5)
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
    """S-PT (the trained file), S-PT-dp0 (pigtrack_v1 = models.student_pigtrack, sha-pinned, read-only; every run other
    than pigtrack_v1) and S-Edi (registry, read-only), all through the bench's own loader and gates (the
    drop_path_rate=0.0 class, eval())."""
    out = {}
    s32, meta = core.load_student(best_local, device="cuda", dtype=torch.float32, check_gn=True)
    out[ctx["P"].arm] = (s32, to_bf16(s32), dict(meta, sha256=core.sha256_file(best_local)))
    R = ctx["R"]
    v1_dir = osp.dirname(R.path(MODEL_KEY))
    refs = [("S-PT", osp.dirname(v1_dir).replace("\\", "/") + f"/{REF_RUN}/{CKPT_NAME}", SPT_SHA256),
            ("S-PT-dp0", MODEL_KEY, R.entry(MODEL_KEY).get("sha256") or SPT_DP0_SHA256),
            ("S-Edi", REF_STUDENT_KEY, R.entry(REF_STUDENT_KEY).get("sha256") or REF_STUDENT_SHA256)]
    for name, key, want in refs:
        src = R.path(key) if "/" not in key else key
        ref_local, sha = core.stage_file(src, osp.join(ctx["local"], "ref", name, osp.basename(src)), ctx["dbutils"],
                                         want)
        e32, emeta = core.load_student(ref_local, device="cuda", dtype=torch.float32, check_gn=True)
        out[name] = (e32, to_bf16(e32), dict(emeta, sha256=sha, registry_key=key))
    for name, (a, b, _m) in out.items():
        assert_eval_mode(a, f"fidelity {name} fp32")
        assert_eval_mode(b, f"fidelity {name} bf16")
    return out


def load_v1_reference(ctx):
    """pigtrack_v1's published evaluation (per-frame CSV + summary), sha-checked against its eval_done.json."""
    R = ctx["R"]
    rd = R.path("paper1.distillation_runs", None, CANONICAL_RUN)
    ed = json.loads(_read(f"{rd}/eval_done.json"))
    with open(f"{rd}/{ed['eval_dir']}/per_frame_fidelity.csv", "rb") as f:
        csv_b = f.read()
    with open(f"{rd}/{ed['eval_dir']}/fidelity_heldout.json", "rb") as f:
        fid_b = f.read()
    if hashlib.sha256(csv_b).hexdigest() != ed["csv_sha256"] or hashlib.sha256(fid_b).hexdigest() != ed["fidelity_sha256"]:
        raise RuntimeError(f"{CANONICAL_RUN} evaluation files differ from its eval_done.json sha256")
    fid = json.loads(fid_b.decode("utf-8"))
    return csv_b.decode("utf-8"), fid["summary"], {"run": CANONICAL_RUN, "eval_dir": ed["eval_dir"],
                                                   "csv_sha256": ed["csv_sha256"], "n_frames": ed["n_frames"],
                                                   "gpu_class": "A100 (the pigtrack_v1 run)"}


def load_v2_reference(ctx):
    """pigtrack_v2_dp01's (S-PT) published evaluation (per-frame CSV + summary), sha-checked against its eval_done.json."""
    R = ctx["R"]
    rd = R.path("paper1.distillation_runs", None, REF_RUN)
    ed = json.loads(_read(f"{rd}/eval_done.json"))
    with open(f"{rd}/{ed['eval_dir']}/per_frame_fidelity.csv", "rb") as f:
        csv_b = f.read()
    with open(f"{rd}/{ed['eval_dir']}/fidelity_heldout.json", "rb") as f:
        fid_b = f.read()
    if hashlib.sha256(csv_b).hexdigest() != ed["csv_sha256"] or hashlib.sha256(fid_b).hexdigest() != ed["fidelity_sha256"]:
        raise RuntimeError(f"{REF_RUN} evaluation files differ from its eval_done.json sha256")
    fid = json.loads(fid_b.decode("utf-8"))
    return csv_b.decode("utf-8"), fid["summary"], {"run": REF_RUN, "eval_dir": ed["eval_dir"],
                                                   "csv_sha256": ed["csv_sha256"], "n_frames": ed["n_frames"],
                                                   "gpu_class": ed.get("gpu_class")}


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
           "input": "core.make_student_transform() (= the export transform) on the byte-identical PigTrack JPEG",
           "teacher_targets": "the stored fp16 targets of Tol_66 (val role)", "seconds": round(time.time() - t0, 1),
           "gpu": core.gpu_info()}
    if "S-PT-dp0" in students:
        ref_csv, ref_summary, ref_info = load_v1_reference(ctx)
        rep = reference_reproduction(rows, ref_csv, ref_summary, summary)
        rep["reference"] = ref_info
        res["reference_reproduction"] = rep
        log(f"  S-PT-dp0 / S-Edi vs the published {CANONICAL_RUN} evaluation: {rep['status']} "
            f"({rep['n_frames_compared']} frames; worst per-frame "
            f"{max((d['max'] for d in rep['per_frame'].values()), default=float('nan')):.2e})")
    if "S-PT" in students:
        ref_csv, ref_summary, ref_info = load_v2_reference(ctx)
        rep2 = reference_reproduction(rows, ref_csv, ref_summary, summary, repro_map=REPRO_MAP_V2)
        rep2["reference"] = ref_info
        res["reference_reproduction_spt"] = rep2
        log(f"  S-PT / S-PT-dp0 / S-Edi vs the published {REF_RUN} evaluation: {rep2['status']} "
            f"({rep2['n_frames_compared']} frames; worst per-frame "
            f"{max((d['max'] for d in rep2['per_frame'].values()), default=float('nan')):.2e})")
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
    return {"notebook": nb, "tol_distill_core": core.sha256_file(dc.__file__),
            "tol_bench_core": core.sha256_file(core.__file__), "tol_paths": core.sha256_file(tol_paths.__file__),
            "nb_version": NB_VERSION, "dc_version": dc.DISTILL_CORE_VERSION, "core_version": core.BENCH_VERSION}


PROD_HOOKS = {
    "local_root": "/local_disk0/tol02c",
    "reuse_root": TOL66_REUSE_ROOT,
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
                                                              "heldout_S-PT", "heldout_S-PT-dp0", "arm", "reference_reproduction_spt_status", "drop_path_rate", "model_dir",
                                                              "eval_gpu_class", "reference_reproduction_status",
                                                              "ckpt_sha256", "last_epoch", "skipped")}, default=str))
