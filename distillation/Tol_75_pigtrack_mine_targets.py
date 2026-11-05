# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N75 — hard-mined distillation targets for the PigTrack student (Paper 1, BENCH b1.6, job **J14**, **A100**)
# MAGIC
# MAGIC Mirrors `Tol_26_mine_targets` (calf v2) on PigTrack and `Tol_66_pigtrack_teacher_targets` (target format). For every
# MAGIC **eligible** frame of the 31 whitelisted PigTrack TRAIN sequences (= the 35 train sequences minus the 4 distillation-val
# MAGIC sequences, minus the 2,626 frames the base student S-PT already trains on) it runs the SAM 3 teacher backbone AND the
# MAGIC CURRENT student (S-PT = `pigtrack_v2_dp01`, sha-pinned) and scores the frame by the **per-position feature cosine**
# MAGIC (mean over the 72 x 72 positions of the 1024-d cosine, 5 decimals; ascending = hardest first). Then:
# MAGIC
# MAGIC | set | rule | file |
# MAGIC |---|---|---|
# MAGIC | **mined** (M = 2,125) | per stratum (night, then day): order by (cos, seq, frame), skip a frame whose sequence holds `floor(0.79 x eligible)` selected frames, stop at the stratum quota (747 night / 1,378 day) | `SELECTION_mined.csv` |
# MAGIC | **random** (R) | same quotas / caps / pool; hash order `sha256("20261024|seq|fi")`, NO cosine | `SELECTION_random.csv` |
# MAGIC | **random2** (R2) | as R with seed 20261025 | `SELECTION_random2.csv` |
# MAGIC | **comp** (R_comp, audit copy; trained only if trigger T1 fires) | per-sequence counts of M; inside a sequence the first n_s frames in hash order `sha256("20261026|seq|fi")` | `SELECTION_comp.csv` |
# MAGIC
# MAGIC The teacher features and frames of **ALL 6,094 eligible frames** are published (Tol_66 format: `[1,1024,72,72]` fp16
# MAGIC `torch.save` + the byte-identical JPEG), so that no later step needs a second extraction. **Positive control:** two base
# MAGIC train frames per whitelisted sequence (62) are re-extracted and compared with the stored Tol_66 targets (`cos_global >=
# MAGIC 0.9999` required, bit identity expected). They are forwarded TWICE: first in a **pre-pass** (before anything is published; a
# MAGIC failure stops the job with nothing published at all), then again inside each sequence's production session (a sequence whose
# MAGIC control fails there is NOT published and the job stops).
# MAGIC The GPU cosine is cross-checked against `tol_distill_core.fidelity_metrics` (float64, host) on the first 50 frames (< 1e-4).
# MAGIC
# MAGIC **Leakage rules (hard).** Only the explicit `whitelist` widget (31 names, sha256 pinned, asserted equal to
# MAGIC `train35 - val4` computed from `split.txt` + the val rule, and disjoint from the 4 val / 20 official-val / 25 test sequences)
# MAGIC is ever opened as a frame source. No test or official-val frame is read. No GT file is read. The output files are scanned
# MAGIC for forbidden sequence names before they are published (the 4 val names may appear only as `role=val` rows of the
# MAGIC `union_listing_*.csv` audit files).
# MAGIC
# MAGIC **Outputs** (NEW folder `paper1_distillation/teacher_targets/pigtrack_mined_v1/`; every file published once, never
# MAGIC overwritten; no append / seek on the Volume):
# MAGIC ```
# MAGIC MINING_CONFIG.json  candidates.csv  control_base.csv  MINING_SUMMARY.json  MINING_REPORT.json
# MAGIC SELECTION_{mined,random,random2,comp}.csv   manifest_{mined,random,random2,comp}.csv   union_listing_{...}.csv
# MAGIC records/<seq>.json (= the per-sequence done marker)   runs/<utc>.json
# MAGIC seqs/<seq>/frames/<seq>_f<fi:06d>.jpg   seqs/<seq>/features/<seq>_f<fi:06d>.pt        (ALL eligible frames)
# MAGIC ```
# MAGIC `manifest_<set>.csv` has the `tol_distill_core.MANIFEST_COLUMNS` of the Tol_66 manifest (paths relative to THIS folder,
# MAGIC `role=train`); `Tol_02c` forms base + extra. `union_listing_<set>.csv` = `seq,fi,role,source,cos,night` for base + extra rows.
# MAGIC
# MAGIC **Resume:** a sequence whose `records/<seq>.json` exists is skipped; a resume must match `MINING_CONFIG.json`.
# MAGIC **Smoke:** `smoke_seqs=<seq,...>` with `out_name=smoke_*` runs the same pass on those sequences into a separate folder and
# MAGIC stops before the selection. Local check (no torch / GPU / Volume): `TOL75_SELFTEST=1 python Tol_75_pigtrack_mine_targets.py`.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "Pillow==11.1.0" "huggingface_hub" "accelerate" "timm==1.0.25"

# COMMAND ----------

try:
    dbutils.library.restartPython()  # noqa: F821
except NameError:
    pass

# COMMAND ----------

import collections
import gc
import hashlib
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

NB_NAME = "Tol_75_pigtrack_mine_targets"
NB_VERSION = "tol75-2026-10-24.2"
CONFIG_VERSION = "tol75-mining-v1"
SELFTEST = os.environ.get("TOL75_SELFTEST") == "1"
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None
if not IS_NOTEBOOK and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _code_dir():
    env = os.environ.get("TOL75_CODE_DIR")
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


# Full sha256 of the shared modules this notebook may run against (DESIGN s.5 "pinned code only"). The Workspace root holds
# tol_bench_core 5e705fff (the core S-PT / Tol_66 / Tol_02b ran with); _pinned/core_bc46ecfb holds bc46ecfb. Every symbol this
# notebook uses is AST-identical in the two (checked when the notebook was built: bench/pt_mining/_work/ast_core_check.json).
CODE_PINS = {
    "tol_bench_core": {"5e705fffcb1711a427239a6bcec3e3622b9b8be69dc3e1b459643c663d460e58": "Workspace root core (S-PT training core)",
                       "bc46ecfb9eca56af5fc991d5ea0e3460776cc2dc1f9780c9a8c756c060132b16": "_pinned/core_bc46ecfb (bench core)"},
    "tol_distill_core": {"16522b6c738bac6fa6cbc3097481b9e970acd11538e495d40def92371761fe02": "dc1.0 2026-10-14"},
    "tol_paths": {"f6603e639ec0e010640da401afc3837ae2e3923bf9c9dd62ff3e55bc9d9d3d46": "registry v1 loader"},
}

CODE_DIR = _code_dir()
core = _import_exact("tol_bench_core", [CODE_DIR])
dc = _import_exact("tol_distill_core", [CODE_DIR])
_PATHS_DIRS = [CODE_DIR, os.environ.get("TOL_PATHS_DIR")]
if not IS_NOTEBOOK:
    _PATHS_DIRS.append(osp.join(osp.dirname(CODE_DIR), "catalog", "build"))
tol_paths = _import_exact("tol_paths", _PATHS_DIRS)
for _mod, _name in ((core, "tol_bench_core"), (dc, "tol_distill_core"), (tol_paths, "tol_paths")):
    _sha = core.sha256_file(_mod.__file__)
    if _sha not in CODE_PINS[_name]:
        raise RuntimeError(f"{_name} sha256 {_sha} is not one of the pinned versions {sorted(CODE_PINS[_name])}")
print(f"{NB_NAME} {NB_VERSION}: tol_bench_core {core.BENCH_VERSION} sha256 {core.sha256_file(core.__file__)[:16]}, "
      f"tol_distill_core {dc.DISTILL_CORE_VERSION} sha256 {core.sha256_file(dc.__file__)[:16]}, "
      f"tol_paths sha256 {core.sha256_file(tol_paths.__file__)[:16]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pinned design constants, parameters and the pure-Python functions (exercised by the selftest and by the real-data test)

# COMMAND ----------

# ---- pinned design constants (BENCH b1.6 DESIGN s.2 - s.3). Every value is asserted against its source at run time.
WHITELIST31 = (
    "pigtrack0003", "pigtrack0009", "pigtrack0012", "pigtrack0013", "pigtrack0014", "pigtrack0016", "pigtrack0019",
    "pigtrack0023", "pigtrack0024", "pigtrack0025", "pigtrack0026", "pigtrack0027", "pigtrack0028", "pigtrack0035",
    "pigtrack0036", "pigtrack0039", "pigtrack0043", "pigtrack0044", "pigtrack0045", "pigtrack0046", "pigtrack0052",
    "pigtrack0053", "pigtrack0057", "pigtrack0059", "pigtrack0061", "pigtrack0063", "pigtrack0073", "pigtrack0075",
    "pigtrack0076", "pigtrack0079", "pigtrack0080")
VAL4 = ("pigtrack0041", "pigtrack0050", "pigtrack0064", "pigtrack0070")           # the distillation-val sequences (374 targets)
OFFICIAL_VAL20 = ("pigtrack0001", "pigtrack0006", "pigtrack0007", "pigtrack0008", "pigtrack0015", "pigtrack0020",
                  "pigtrack0022", "pigtrack0029", "pigtrack0031", "pigtrack0032", "pigtrack0033", "pigtrack0034",
                  "pigtrack0037", "pigtrack0047", "pigtrack0048", "pigtrack0051", "pigtrack0056", "pigtrack0062",
                  "pigtrack0066", "pigtrack0069")
TEST25 = ("pigtrack0002", "pigtrack0004", "pigtrack0005", "pigtrack0010", "pigtrack0011", "pigtrack0017", "pigtrack0018",
          "pigtrack0021", "pigtrack0030", "pigtrack0038", "pigtrack0040", "pigtrack0042", "pigtrack0049", "pigtrack0054",
          "pigtrack0055", "pigtrack0058", "pigtrack0060", "pigtrack0065", "pigtrack0067", "pigtrack0068", "pigtrack0071",
          "pigtrack0072", "pigtrack0074", "pigtrack0077", "pigtrack0078")
PINS = {
    "whitelist_sha256": "163dbf117668955ab0bed54e183dfc2f02f39e61404bb910d80107513b9392bd",      # sorted, newline-joined, final newline
    "n_pool": 8720, "n_base_train": 2626, "n_eligible": 6094, "n_eligible_night": 2142,
    "eligible_canon_sha256": "b8369e3e33e18a1333e61dadf602a96751f56fa2f4a04130a16fbf29085a14ac",   # sorted "seq,fi\n" lines
    "M": 2125, "night_q": 747, "day_q": 1378,
    "R_canon_sha256": "afaf2371b85111d9696c6c48da3620ce99bd4080ead479624ac298c48902b0a8",
    "R2_canon_sha256": "5f1b04a3756e0ff64120f092bad0affbbe37fba80438ec08b265c0ddf2298e9a",
    "base_manifest_sha256": "6eff15e8350e2b155a2f3a16a7edede7883c95b8a105f340fb1faf38089d8a27",      # Tol_66 manifest.csv (J5)
    "base_targets_config_sha256": "1d8bbed385bb1b37512157f0ce45673fa5bbf566c35465c771a1cad811f78335",
    "base_plan_sha256": "f33cb9c50963d14e1d45c03f249bd24d157f1e4b307fdabce0e854a60747a140",
    "split_sha256": "3f75399d240fc79f9371c9edf598b7a9428914d2e266a3cbd4dc41227f4948e5",
    "student_sha256": "c07140f2de80a5e9faf175bdb4fc071bb1064e4757b91c3eefe55e271e5da69c",           # S-PT = pigtrack_v2_dp01 (EMA ep 32)
    "teacher_revision": "3c879f39826c281e95690f02c7821c4de09afae7",
}
SEED_R, SEED_R2, SEED_COMP = 20261024, 20261025, 20261026
CAP_FRAC = 0.79                         # Tol_26: per-clip cap 400 of 288 - 514 candidates per clip = 0.78 - 0.80
RATIO_NUM, RATIO_DEN = 8000, 22934      # calf mining: 8,000 selected of 22,934 candidates
T1_TOPK, T1_THRESHOLD = 6, 0.5          # trigger T1: the six largest per-sequence counts of M sum to MORE than 50 % of M
N_CONTROL_PER_SEQ = 2
CONTROL_COS_GLOBAL_MIN = 0.9999
CROSSCHECK_N, CROSSCHECK_TOL = 50, 1e-4
SET_NAMES = ("mined", "random", "random2", "comp")
SET_SOURCE = {"mined": "mined", "random": "random", "random2": "random2", "comp": "comp"}
MODEL_KEY = "models.student_pigtrack"            # models/sam3_student_tinyvit/pigtrack_v1/...; S-PT lies beside it
SPT_RUN = "pigtrack_v2_dp01"
CKPT_NAME = "best_backbone_student_pigtrack.pt"
BASE_TARGETS_KEY = "paper1.teacher_targets.pigtrack_train"
OUT_NAME_DEFAULT = "pigtrack_mined_v1"
REGISTRY_VERSION = 1
HF_SAM3 = "facebook/sam3"
PRUNE_EVERY_T01, KEEP_LAST_N_T01 = 25, 8        # Tol_01's session hygiene (the backbone output does not depend on it)
PROMPT_INDEP_MIN_COS = 0.9999                    # same frame, two sessions, two prompt sets -> the same backbone output
SEQ_TOKEN_RX = re.compile(r"pigtrack\d{4}")

WIDGETS = collections.OrderedDict([
    ("whitelist", ",".join(WHITELIST31)), ("out_name", OUT_NAME_DEFAULT), ("smoke_seqs", ""),
    ("sam3_revision", "protocol:hf_revisions_v1.json"), ("require_gpu", "A100"), ("copy_threads", "16"),
    ("progress_every", "5"), ("min_free_gb", "120"),
])


class WhitelistError(ValueError):
    pass


class SelectionError(RuntimeError):
    pass


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _sha_text(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def parse_params(raw):
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}; known: {list(WIDGETS)}")
    g = {k: str(raw.get(k) if raw.get(k) not in (None, "") else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace()
    P.whitelist = [s for s in re.split(r"[,\s]+", g["whitelist"]) if s]
    P.out_name = g["out_name"]
    P.smoke_seqs = [s for s in re.split(r"[,\s]+", g["smoke_seqs"]) if s]
    if P.smoke_seqs:
        if not re.match(r"^smoke_[a-z0-9_]+$", P.out_name):
            raise ValueError(f"smoke_seqs needs an out_name 'smoke_...', got {P.out_name!r}")
    elif P.out_name != OUT_NAME_DEFAULT:
        raise ValueError(f"out_name={P.out_name!r}: a full run writes {OUT_NAME_DEFAULT!r} only")
    P.sam3_revision = g["sam3_revision"]
    if P.sam3_revision and not (P.sam3_revision.startswith("protocol:") or re.match(r"^[0-9a-f]{40}$", P.sam3_revision)):
        raise ValueError(f"sam3_revision={P.sam3_revision!r}: '' | protocol:<file>.json | a 40-hex commit")
    P.require_gpu = g["require_gpu"]
    if P.require_gpu not in ("A100", "A10", "any"):
        raise ValueError(f"require_gpu={P.require_gpu!r}")
    P.copy_threads = int(g["copy_threads"])
    P.progress_every = int(g["progress_every"])
    P.min_free_gb = float(g["min_free_gb"])
    return P


# ---------------------------------------------------------------------------------------------------------------------
# whitelist, forbidden names
# ---------------------------------------------------------------------------------------------------------------------
def whitelist_sha256(names):
    return hashlib.sha256("".join(f"{s}\n" for s in sorted(names)).encode("utf-8")).hexdigest()


def forbidden_from_split(split, val_seqs):
    """{'val4', 'official_val', 'test'} derived from split.txt and the pre-registered val rule."""
    return {"val4": sorted(val_seqs),
            "official_val": sorted(s for s, v in split.items() if v == "val"),
            "test": sorted(s for s, v in split.items() if v == "test")}


def check_whitelist(names, *, train, val_seqs, split, expect_sha256=None):
    """The explicit whitelist must be exactly train35 - val4 (from split.txt + the val rule), disjoint from the 4 val / the
    official-val / the test sequences, duplicate-free, and have the pinned sha256. Returns the sorted list or raises."""
    names = list(names)
    if not names:
        raise WhitelistError("empty whitelist")
    if len(set(names)) != len(names):
        raise WhitelistError(f"duplicate whitelist entries {sorted(n for n in set(names) if names.count(n) > 1)}")
    bad = [s for s in names if not dc.SEQ_RX.match(s)]
    if bad:
        raise WhitelistError(f"malformed sequence names {bad[:5]}")
    fb = forbidden_from_split(split, val_seqs)
    for label in ("val4", "official_val", "test"):
        hit = sorted(set(names) & set(fb[label]))
        if hit:
            raise WhitelistError(f"whitelist contains {label} sequence(s) {hit}: refused")
    not_train = sorted(s for s in names if split.get(s) != "train")
    if not_train:
        raise WhitelistError(f"whitelist names that are not PigTrack TRAIN sequences: {not_train[:5]}")
    want = sorted(set(train) - set(val_seqs))
    if sorted(names) != want:
        raise WhitelistError(f"whitelist != train - val4: missing {sorted(set(want) - set(names))[:5]}, "
                             f"extra {sorted(set(names) - set(want))[:5]}")
    sha = whitelist_sha256(names)
    if expect_sha256 is not None and sha != expect_sha256:
        raise WhitelistError(f"whitelist sha256 {sha} != pinned {expect_sha256}")
    return sorted(names)


def forbidden_name_hits(text, names):
    """Sequence-name tokens of `text` that are in `names` (sorted, unique)."""
    return sorted(set(SEQ_TOKEN_RX.findall(text)) & set(names))


def scan_outputs(files, fb, val_listing_ok=()):
    """files: {relative name: text}. No test / official-val name may appear anywhere; the 4 val names only in the files named
    in `val_listing_ok` and there only on `role=val` rows. Returns a list of problems (empty = clean)."""
    problems = []
    for name, text in sorted(files.items()):
        for label in ("test", "official_val"):
            hit = forbidden_name_hits(text, fb[label])
            if hit:
                problems.append(f"{name}: {label} sequence(s) {hit}")
        hit4 = forbidden_name_hits(text, fb["val4"])
        if not hit4:
            continue
        if name not in val_listing_ok:
            problems.append(f"{name}: distillation-val sequence(s) {hit4} (only allowed in {sorted(val_listing_ok)})")
            continue
        for ln in text.splitlines()[1:]:
            c = ln.split(",")
            if c and c[0] in fb["val4"] and not (len(c) > 3 and c[2] == "val" and c[3] == "base"):
                problems.append(f"{name}: a val sequence row that is not role=val/source=base: {ln[:80]}")
    return problems


# ---------------------------------------------------------------------------------------------------------------------
# pool, quotas, selection (pure; tested on the real design data by test_tol75_real.py)
# ---------------------------------------------------------------------------------------------------------------------
def eligible_pool(infos, plan, whitelist):
    """[(seq, fi, night)] sorted: every frame of a whitelisted sequence that is not a base train target (plan frame)."""
    out = []
    for s in sorted(whitelist):
        n = int(infos[s]["n"])
        night = int(infos[s]["night"])
        base = set(plan[s])
        out += [(s, fi, night) for fi in range(n) if fi not in base]
    return out


def canon_list(pairs):
    return "".join(f"{s},{fi}\n" for s, fi in sorted(pairs))


def canon_sha256(pairs):
    return hashlib.sha256(canon_list(pairs).encode("utf-8")).hexdigest()


def caps_for(pool, cap_frac=CAP_FRAC):
    cnt = collections.Counter(s for s, _fi, _nt in pool)
    return {s: int(math.floor(cap_frac * c)) for s, c in sorted(cnt.items())}


def derive_M(n_eligible):
    """Largest M with M / eligible <= 8000 / 22934 (the calf ratio): floor(8000 * n / 22934)."""
    return (RATIO_NUM * int(n_eligible)) // RATIO_DEN


def derive_quotas(M, n_eligible, n_night):
    """Exact stratum counts = the eligible pool's night share (DESIGN s.0 item 5): night = round(M * n_night / n_eligible)."""
    nq = int(round(M * n_night / n_eligible))
    return nq, M - nq


def _fill(order, quota, cnt, caps, out, what):
    got = 0
    for s, fi in order:
        if got >= quota:
            break
        if cnt[s] >= caps[s]:
            continue
        out.append((s, fi))
        cnt[s] += 1
        got += 1
    if got != quota:
        raise SelectionError(f"{what}: only {got} of {quota} frames available under the per-sequence caps")


def select_mined(cands, caps, night_q, day_q):
    """cands: [(seq, fi, night, cos)]. Per stratum (night, then day): ascending (cos, seq, fi); skip a frame whose sequence
    already holds `caps[seq]` selected frames; stop at the quota. Returns (sorted [(seq, fi)], Counter per sequence)."""
    cnt, out = collections.Counter(), []
    for flag, quota, nm in ((1, night_q, "night"), (0, day_q, "day")):
        order = [(s, fi) for s, fi, _nt, _c in sorted((c for c in cands if int(c[2]) == flag),
                                                      key=lambda c: (c[3], c[0], c[1]))]
        _fill(order, quota, cnt, caps, out, f"mined {nm}")
    return sorted(out), cnt


def hash_key(seed, seq, fi):
    return hashlib.sha256(f"{seed}|{seq}|{fi}".encode("utf-8")).hexdigest()


def draw_random(pool, caps, seed, night_q, day_q):
    """The pre-registered cosine-free draw: per stratum hash order, same caps and quotas as the mined selection."""
    cnt, out = collections.Counter(), []
    for flag, quota, nm in ((1, night_q, "night"), (0, day_q, "day")):
        order = sorted(((s, fi) for s, fi, nt in pool if int(nt) == flag), key=lambda t: hash_key(seed, t[0], t[1]))
        _fill(order, quota, cnt, caps, out, f"random({seed}) {nm}")
    return sorted(out), cnt


def draw_comp(pool, counts, seed):
    """R_comp: per-sequence counts = those of M; inside a sequence the first n_s frames in hash order (seed|seq|fi)."""
    by_seq = collections.defaultdict(list)
    for s, fi, _nt in pool:
        by_seq[s].append(fi)
    out = []
    for s in sorted(counts):
        k = int(counts[s])
        if k == 0:
            continue
        order = sorted(by_seq[s], key=lambda fi: hash_key(seed, s, fi))
        if k > len(order):
            raise SelectionError(f"comp: {k} frames wanted from {s}, {len(order)} eligible")
        out += [(s, fi) for fi in order[:k]]
    return sorted(out)


def control_frames(plan, whitelist, n_per_seq=N_CONTROL_PER_SEQ):
    """The positive-control frames: per whitelisted sequence the n base train frames with the smallest
    sha256("control|seq|fi") (deterministic, cosine-free). {seq: [fi, ...]} sorted by fi."""
    out = {}
    for s in sorted(whitelist):
        if len(plan[s]) < n_per_seq:
            raise SelectionError(f"{s}: only {len(plan[s])} base train frames")
        pick = sorted(plan[s], key=lambda fi: hashlib.sha256(f"control|{s}|{fi}".encode("utf-8")).hexdigest())[:n_per_seq]
        out[s] = sorted(pick)
    return out


def top_share(cnt, k=T1_TOPK):
    v = sorted(cnt.values(), reverse=True)
    return sum(v[:k]) / max(sum(v), 1)


def n_for_share(cnt, share):
    v = sorted(cnt.values(), reverse=True)
    tot, c = sum(v), 0
    for i, x in enumerate(v, 1):
        c += x
        if c >= share * tot - 1e-9:
            return i
    return len(v)


def t1_decision(cnt, k=T1_TOPK, threshold=T1_THRESHOLD):
    """T1 fires iff the k largest per-sequence counts sum to MORE than `threshold` of all selected frames (exact integers)."""
    v = sorted(cnt.values(), reverse=True)
    top, tot = sum(v[:k]), sum(v)
    fired = top * 1000 > int(round(threshold * 1000)) * tot
    return {"topk": k, "top_sum": top, "total": tot, "top_share": top / max(tot, 1), "threshold": threshold,
            "fired": bool(fired), "n_sequences_for_50pct": n_for_share(cnt, 0.5)}


def adjacent_share(sel):
    S = set(sel)
    return sum(1 for s, fi in sel if (s, fi - 1) in S or (s, fi + 1) in S) / max(len(sel), 1)


def overlap(a, b):
    return len(set(a) & set(b))


def describe(vals):
    v = np.asarray(list(vals), dtype=np.float64)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "p05": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
            "p50": float(np.percentile(v, 50)), "p75": float(np.percentile(v, 75)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def csv_text(header, rows):
    return ",".join(header) + "\n" + "".join(",".join(str(c) for c in r) + "\n" for r in rows)


def fmt_cos(c):
    return f"{float(c):.5f}"


def set_manifest_rows(records, infos, sel_pairs):
    """Tol_66-format manifest rows (role=train, paths relative to the mining folder) for the selected (seq, fi) pairs."""
    want = collections.defaultdict(set)
    for s, fi in sel_pairs:
        want[s].add(int(fi))
    sub = {}
    for s, rec in records.items():
        files = [f for f in rec["files"] if int(f["fi"]) in want.get(s, ())]
        if files:
            sub[s] = dict(rec, files=files)
    rows = dc.manifest_rows(sub, infos, [])
    if sorted((r["seq"], r["fi"]) for r in rows) != sorted((s, int(fi)) for s, fi in sel_pairs):
        raise SelectionError("manifest rows != selection (a selected frame has no published target)")
    return rows


def build_diagnostics(*, pool, caps, cos_of, sets, counts, control_rows, night_of):
    """D1 / D2 (DESIGN s.3), no verdict except T1 (computed separately)."""
    def cos_stats(pairs):
        pairs = list(pairs)
        return {"all": describe(cos_of[p] for p in pairs),
                "night": describe(cos_of[p] for p in pairs if night_of[p[0]]),
                "day": describe(cos_of[p] for p in pairs if not night_of[p[0]])}
    elig_pairs = [(s, fi) for s, fi, _nt in pool]
    out = {"cos_eligible": cos_stats(elig_pairs),
           "cos_selected": {k: cos_stats(v) for k, v in sets.items()},
           "cos_control_base": {"all": describe(r["cos"] for r in control_rows),
                                "night": describe(r["cos"] for r in control_rows if r["night"]),
                                "day": describe(r["cos"] for r in control_rows if not r["night"])}}
    seqs = sorted(caps)
    per_seq = {}
    for s in seqs:
        ps = [p for p in elig_pairs if p[0] == s]
        per_seq[s] = {"night": int(night_of[s]), "eligible": len(ps), "cap": caps[s], "cos_eligible_mean": float(np.mean([cos_of[p] for p in ps])),
                      "cos_eligible_p10": float(np.percentile([cos_of[p] for p in ps], 10)),
                      **{f"n_{k}": int(counts[k].get(s, 0)) for k in sets},
                      **{f"cos_{k}_mean": (float(np.mean([cos_of[p] for p in sets[k] if p[0] == s]))
                                           if counts[k].get(s, 0) else None) for k in sets}}
    out["per_sequence"] = per_seq
    out["selected_vs_pool"] = {k: {"n": len(v), "cos_mean": float(np.mean([cos_of[p] for p in v])),
                                   "pool_cos_mean": float(np.mean([cos_of[p] for p in elig_pairs])),
                                   "cos_max_selected": float(max(cos_of[p] for p in v)),
                                   "n_sequences": len({s for s, _ in v}),
                                   "top6_share": top_share(counts[k]), "n_sequences_for_50pct": n_for_share(counts[k], 0.5),
                                   "adjacent_share": adjacent_share(v),
                                   "caps_binding": int(sum(1 for s in seqs if counts[k].get(s, 0) >= caps[s]))}
                               for k, v in sets.items()}
    names = list(sets)
    out["overlaps"] = {f"{a}-{b}": overlap(sets[a], sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# inputs: PigTrack train discovery, teacher revision (VERBATIM Tol_66)
# ---------------------------------------------------------------------------------------------------------------------
def resolve_revision(R, spec):
    if not spec:
        return None, "unpinned"
    if spec.startswith("protocol:"):
        name = spec.split(":", 1)[1]
        d = json.loads(_read(R.path("paper1.root", None, "protocol", name)))
        rev = (d.get("revisions") or {}).get(HF_SAM3)
        if not rev or not re.match(r"^[0-9a-f]{40}$", rev):
            raise ValueError(f"{name} has no 40-hex pin for {HF_SAM3}")
        return rev, spec
    return spec, "widget"


def discover(R, expect_train=dc.EXPECT_TRAIN_SEQS):
    """The PigTrack TRAIN sequences (split.txt == seqmap == directory listing) and their seq_info records. GT never read."""
    split_p = R.path("datasets.pigtrack.split_file")
    seqmap_p = R.path("datasets.pigtrack.seqmaps", None, "pigtrack-train.txt")
    root = R.path("datasets.pigtrack.sequences", None, "train")
    split = dc.parse_split_text(_read(split_p))
    train = sorted(s for s, sp in split.items() if sp == "train")
    seqmap = sorted(dc.parse_seqmap_text(_read(seqmap_p)))
    listed = sorted(n for n in os.listdir(root) if dc.SEQ_RX.match(n))
    if not (train == seqmap == listed) or len(train) != expect_train:
        raise RuntimeError(f"train sequences disagree: split.txt {len(train)}, seqmap {len(seqmap)}, directory "
                           f"{len(listed)} (expected {expect_train}); first diffs {sorted(set(train) ^ set(listed))[:5]}")
    infos = {}
    for s in train:
        si = dc.parse_seqinfo_text(_read(osp.join(root, s, "seqinfo.ini")))
        infos[s] = dc.seq_info(_read(osp.join(root, s, "seqinfo.ini")), os.listdir(osp.join(root, s, si["imDir"])), s)
    src = {"split_sha256": _sha_text(split_p), "seqmap_sha256": _sha_text(seqmap_p)}
    return train, infos, root, src


def student_ckpt_path(R):
    """models/sam3_student_tinyvit/pigtrack_v2_dp01/<ckpt>, derived from the registry key (read only)."""
    v1_dir = osp.dirname(R.path(MODEL_KEY))
    if osp.basename(v1_dir) != "pigtrack_v1":
        raise RuntimeError(f"registry {MODEL_KEY} -> {v1_dir}: expected a 'pigtrack_v1' directory")
    return osp.dirname(v1_dir).replace("\\", "/") + f"/{SPT_RUN}/{CKPT_NAME}"


def out_dir_for(R, out_name):
    out = R.out("paper1.root", "teacher_targets", out_name)
    if osp.dirname(out) != osp.dirname(R.out(BASE_TARGETS_KEY)):
        raise RuntimeError(f"{out} does not lie beside {R.out(BASE_TARGETS_KEY)}")
    return out


def build_config(P, *, pins, whitelist, val_seqs, pool, caps, sets_canon, infos, rev, rev_src, versions, gpu_class,
                 student_meta, src_sha, base_sha, code):
    n_night = sum(1 for _s, _f, nt in pool if nt)
    cfg = {
        "version": CONFIG_VERSION, "nb": NB_VERSION, "out_name": P.out_name, "smoke_seqs": P.smoke_seqs,
        "inputs": dict(src_sha, **base_sha, student_sha256=pins["student_sha256"], student_run=SPT_RUN),
        "whitelist": {"seqs": whitelist, "sha256": whitelist_sha256(whitelist), "rule": "train35 - val4 (split.txt + val rule)",
                      "forbidden_checked": {"val4_n": len(val_seqs), "val4_sha256": whitelist_sha256(val_seqs),
                                            "disjoint_from_official_val_and_test": True}},
        "pool": {"n_pool": sum(infos[s]["n"] for s in whitelist), "n_eligible": len(pool), "n_eligible_night": n_night,
                 "eligible_canon_sha256": canon_sha256([(s, fi) for s, fi, _ in pool]),
                 "per_seq": {s: {"n": infos[s]["n"], "night": int(infos[s]["night"]), "eligible": sum(1 for p in pool if p[0] == s),
                                 "cap": caps[s]} for s in whitelist}},
        "selection": {"M": pins["M"], "night_q": pins["night_q"], "day_q": pins["day_q"], "cap_frac": CAP_FRAC,
                      "mined": "per stratum (night, day): ascending (cos, seq, fi), skip a sequence at its cap, stop at the quota",
                      "seeds": {"R": SEED_R, "R2": SEED_R2, "R_comp": SEED_COMP},
                      "canon_sha256": sets_canon,
                      "T1": {"topk": T1_TOPK, "threshold": T1_THRESHOLD, "rule": "top-k per-sequence counts of M > threshold * M"}},
        "score": {"metric": "mean over 72x72 positions of the 1024-d cosine(teacher fp16 feature, student fp32 feature), 5 decimals",
                  "student": {"run": SPT_RUN, "sha256": pins["student_sha256"], "load": "core.load_student(dtype=float32, check_gn=True)",
                              "meta": student_meta},
                  "crosscheck": {"n": CROSSCHECK_N, "tol": CROSSCHECK_TOL, "vs": "tol_distill_core.fidelity_metrics cos_patch (float64, host)"}},
        "control": {"per_seq": N_CONTROL_PER_SEQ, "rule": "base train frames with the smallest sha256('control|seq|fi')",
                    "gate": f"cos_global >= {CONTROL_COS_GLOBAL_MIN} vs the stored Tol_66 target; bit identity expected"},
        "teacher": {"hf_id": HF_SAM3, "revision": rev, "revision_source": rev_src, "dtype": "bfloat16",
                    "transformers": core.TRANSFORMERS_PIN, "attention": "model default (resolved recorded; FA2 refused)",
                    "capture": "forward hook on vision_encoder.backbone; last_hidden_state [1,5184,1024] -> "
                               "permute/reshape [1,1024,72,72] -> .cpu().half() (Tol_01 BackboneFeatureHook, = Tol_66)",
                    "input": "Sam3TrackerVideoProcessor (1008x1008), one session per sequence, implicit frame index",
                    "prompt": {"dummy_box_fracs": dc.DUMMY_BOX_FRACS, "note": "backbone output is prompt-independent"},
                    "prune": {"every": PRUNE_EVERY_T01, "keep_last_n": KEEP_LAST_N_T01}},
        "target": {"shape": list(dc.TARGET_SHAPE), "dtype": dc.TARGET_DTYPE, "format": "torch.save(tensor)",
                   "frames": "img1 JPEG copied byte for byte"},
        "code": code,
        "versions": {k: versions.get(k) for k in ("python", "torch", "transformers", "numpy", "Pillow", "cuda", "cudnn")},
        "gpu_class": gpu_class, "created_utc": None, "extra": {"versions_all": versions},
    }
    return json.loads(dc.json_dumps(cfg))


def check_or_publish_config(OUT, cfg, local, dbutils_, log):
    p = f"{OUT}/MINING_CONFIG.json"
    if osp.isfile(p):
        stored = json.loads(_read(p))
        core.config_check(stored, cfg)
        log("MINING_CONFIG.json matches: resuming")
        return stored
    cfg = dict(cfg, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    lp = core.write_text_atomic(osp.join(local, "MINING_CONFIG.json"), dc.json_dumps(cfg))
    core.publish_no_overwrite(lp, p, dbutils_)
    return cfg


def main(raw, *, hooks):
    """One run: inputs + pins -> pool / R / R2 -> stage -> teacher + student -> per-sequence extract (controls gate) + publish
    -> (full run) selection, manifests, diagnostics, T1 -> publish."""
    h = dict(hooks)
    log = h.get("log", _log)
    dbu = h.get("dbutils")
    t_run = time.time()
    P = parse_params(raw)
    pins = dict(PINS, **(h.get("pins") or {}))
    R = h.get("registry") or tol_paths.load(version=REGISTRY_VERSION)
    if h.get("check_deployed_paths", True):
        if core.sha256_file(R.path("admin.registry", None, "tol_paths.py")) != core.sha256_file(tol_paths.__file__):
            raise RuntimeError("the deployed tol_paths.py differs from the registry's copy: re-deploy it")
    BASE = R.path(BASE_TARGETS_KEY)
    OUT = out_dir_for(R, P.out_name)
    LOCAL = h["local_root"]
    os.makedirs(LOCAL, exist_ok=True)
    target_shape = tuple(h.get("target_shape", dc.TARGET_SHAPE))

    # ---- 1. inputs, whitelist, pins (all before any GPU work; only train/ metadata and split.txt are read)
    train, infos, seq_root, src_sha = discover(R, h.get("expect_train_seqs", dc.EXPECT_TRAIN_SEQS))
    split_p = R.path("datasets.pigtrack.split_file")
    split = dc.parse_split_text(_read(split_p))
    if src_sha["split_sha256"] != pins["split_sha256"]:
        raise RuntimeError(f"split.txt sha256 {src_sha['split_sha256']} != pinned {pins['split_sha256']}")
    val_seqs, _val_work = dc.choose_val_sequences(infos)
    whitelist = check_whitelist(P.whitelist, train=train, val_seqs=val_seqs, split=split, expect_sha256=pins["whitelist_sha256"])
    fb = forbidden_from_split(split, val_seqs)
    if h.get("check_typed_forbidden", True):
        if (sorted(VAL4), sorted(OFFICIAL_VAL20), sorted(TEST25), sorted(WHITELIST31)) != \
                (fb["val4"], fb["official_val"], fb["test"], whitelist):
            raise RuntimeError("the typed val / official-val / test / whitelist constants differ from split.txt + the val rule")
    if P.smoke_seqs:
        bad = sorted(set(P.smoke_seqs) - set(whitelist))
        if bad or len(set(P.smoke_seqs)) != len(P.smoke_seqs):
            raise WhitelistError(f"smoke_seqs must be distinct whitelisted sequences, got {P.smoke_seqs}")
    tcfg_text = _read(f"{BASE}/TARGETS_CONFIG.json")
    tcfg = json.loads(tcfg_text)
    man_text = _read(f"{BASE}/manifest.csv")
    base_sha = {"base_manifest_sha256": hashlib.sha256(man_text.encode("utf-8")).hexdigest(),
                "base_targets_config_sha256": hashlib.sha256(tcfg_text.encode("utf-8")).hexdigest(),
                "base_plan_sha256": tcfg["selection"]["plan_sha256"]}
    for k in ("base_manifest_sha256", "base_targets_config_sha256", "base_plan_sha256"):
        if base_sha[k] != pins[k]:
            raise RuntimeError(f"{k} {base_sha[k]} != pinned {pins[k]}")
    base_rows = dc.read_manifest_csv(man_text)
    dc.validate_manifest(base_rows)
    if sorted(tcfg["val"]["seqs"]) != val_seqs:
        raise RuntimeError(f"TARGETS_CONFIG val sequences {tcfg['val']['seqs']} != the val rule's {val_seqs}")
    plan = dc.selection_plan(infos, tcfg["selection"]["n_targets"])
    if dc.plan_digest(plan) != pins["base_plan_sha256"]:
        raise RuntimeError("the re-derived selection plan differs from the pinned plan digest")
    base_train_pairs = sorted((r["seq"], r["fi"]) for r in base_rows if r["role"] == "train")
    if sorted((s, fi) for s in train if s not in val_seqs for fi in plan[s]) != base_train_pairs:
        raise RuntimeError("the base manifest's train rows are not the selection plan's non-val frames")
    pool = eligible_pool(infos, plan, whitelist)
    n_night = sum(1 for _s, _f, nt in pool if nt)
    if not (len(pool) == pins["n_eligible"] and n_night == pins["n_eligible_night"]
            and sum(infos[s]["n"] for s in whitelist) == pins["n_pool"]
            and sum(len(plan[s]) for s in whitelist) == pins["n_base_train"]):
        raise RuntimeError(f"pool counts ({len(pool)} eligible, {n_night} night) differ from the pins")
    if canon_sha256([(s, fi) for s, fi, _ in pool]) != pins["eligible_canon_sha256"]:
        raise RuntimeError("the eligible pool differs from the pinned eligible-pool sha256")
    if set((s, fi) for s, fi, _ in pool) & set(base_train_pairs):
        raise RuntimeError("an eligible frame is a base train target")
    M = derive_M(len(pool))
    night_q, day_q = derive_quotas(M, len(pool), n_night)
    if (M, night_q, day_q) != (pins["M"], pins["night_q"], pins["day_q"]):
        raise RuntimeError(f"derived (M, night, day) = {(M, night_q, day_q)} != pinned {(pins['M'], pins['night_q'], pins['day_q'])}")
    caps = caps_for(pool)
    sel_R, cnt_R = draw_random(pool, caps, SEED_R, night_q, day_q)
    sel_R2, cnt_R2 = draw_random(pool, caps, SEED_R2, night_q, day_q)
    if canon_sha256(sel_R) != pins["R_canon_sha256"] or canon_sha256(sel_R2) != pins["R2_canon_sha256"]:
        raise RuntimeError("the recomputed random sets R / R2 differ from the pre-registered sha256 values")
    controls = control_frames(plan, whitelist)
    log(f"{len(train)} train sequences; whitelist {len(whitelist)} (sha {whitelist_sha256(whitelist)[:12]}); pool {len(pool)} eligible "
        f"({n_night} night), M {M} = {night_q} night + {day_q} day; R / R2 recomputed and equal to the pins; "
        f"{sum(len(v) for v in controls.values())} control frames")

    # ---- 2. environment, student pin, config
    free_gb = h.get("free_gb", lambda p: shutil.disk_usage(p).free / 1e9)(LOCAL)
    if free_gb < P.min_free_gb:
        raise RuntimeError(f"only {free_gb:.0f} GB free on {LOCAL}; need >= {P.min_free_gb:.0f}")
    gpu = h["gpu_preflight"](P)
    rev, rev_src = resolve_revision(R, P.sam3_revision)
    if pins.get("teacher_revision") and rev != pins["teacher_revision"]:
        raise RuntimeError(f"teacher revision {rev} != pinned {pins['teacher_revision']}")
    versions = h["versions"]()
    code = h["code_shas"]()
    student_src = student_ckpt_path(R)
    student_local, student_sha = core.stage_file(student_src, osp.join(LOCAL, "student", CKPT_NAME), dbu, pins["student_sha256"])
    scorer = h["load_scorer"](student_local, student_sha)
    sets_canon = {"R": pins["R_canon_sha256"], "R2": pins["R2_canon_sha256"]}
    cfg = build_config(P, pins=pins, whitelist=whitelist, val_seqs=val_seqs, pool=pool, caps=caps, sets_canon=sets_canon,
                       infos=infos, rev=rev, rev_src=rev_src, versions=versions, gpu_class=gpu["class"],
                       student_meta=scorer.info, src_sha=src_sha, base_sha=base_sha, code={
                           k: code[k] for k in ("tol_bench_core", "tol_distill_core", "tol_paths") if k in code})
    cfg = check_or_publish_config(OUT, cfg, LOCAL, dbu, log)

    rec_dir = f"{OUT}/records"
    run_seqs = list(P.smoke_seqs) if P.smoke_seqs else list(whitelist)
    done = {s for s in run_seqs if osp.isfile(f"{rec_dir}/{s}.json")}
    todo = [s for s in run_seqs if s not in done]
    log(f"records present for {len(done)} sequences; {len(todo)} to extract")
    run = {"nb": NB_VERSION, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu, "code": code,
           "params": {k: v for k, v in vars(P).items() if k != "whitelist"}, "todo": todo, "skipped_done": sorted(done),
           "setup_seconds": round(time.time() - t_run, 1), "free_gb_start": round(free_gb, 1)}
    elig_fis = collections.defaultdict(list)
    for s, fi, _nt in pool:
        elig_fis[s].append(fi)
    base_row = {(r["seq"], r["fi"]): r for r in base_rows}

    pi = None
    if todo:
        # ---- 3. stage the JPEGs of the sequences still to do: eligible + control frames, whitelisted sequences ONLY
        def src_of(s, fi):
            if s not in whitelist:
                raise WhitelistError(f"{s}: not a whitelisted sequence; its frames are never opened")
            return osp.join(seq_root, s, infos[s]["imDir"], dc.src_image_name(fi, infos[s]["imExt"]))
        jobs = []
        for s in todo:
            for fi in sorted(elig_fis[s] + controls[s]):
                jobs.append((src_of(s, fi), osp.join(LOCAL, "src", s, dc.src_image_name(fi, infos[s]["imExt"])), "src", None))
        st = dc.copy_files_parallel(jobs, threads=P.copy_threads, log=log)
        run["staging"] = st
        log(f"staged {st['files']} frames ({st['copied']} copied, {st['bytes_copied'] / 1e9:.2f} GB, {st['seconds']} s, {st['mb_per_s']} MB/s)")
        # ---- 4. teacher
        hf_token = h["hf_token"](dbu)
        teacher = h["load_teacher"](rev, hf_token)
        run["teacher"] = teacher.info
        if rev and teacher.info.get("commit_hash") != rev:
            raise RuntimeError(f"loaded commit {teacher.info.get('commit_hash')} != pinned {rev}")
        if any(v == "flash_attention_2" for v in (teacher.info.get("attn_resolved") or {}).values()):
            raise RuntimeError(f"flash_attention_2 resolved: {teacher.info.get('attn_resolved')}")
        s0 = todo[0]
        f0 = sorted(elig_fis[s0])[0]
        pi = teacher.prompt_independence(osp.join(LOCAL, "src", s0, dc.src_image_name(f0, infos[s0]["imExt"])),
                                         infos[s0]["width"], infos[s0]["height"])
        run["prompt_independence"] = pi
        log(f"prompt independence on {s0} frame {f0}: {pi}")
        if not pi["cos_global"] >= PROMPT_INDEP_MIN_COS:
            raise core.BenchAssertError(f"backbone output depends on the prompt? {pi}")

        # ---- 4b. positive-control PRE-PASS (DESIGN s.3 D2: "the job stops (nothing is published) if any control has cos_global <
        # 0.9999"). The control frames (2 base train frames per sequence still to do, 62 in a full run) are forwarded FIRST, one
        # short session per sequence, and compared with the stored Tol_66 targets BEFORE the first byte of anything is published. A
        # systematic environment mismatch therefore stops the job with nothing published at all. The per-sequence control inside
        # extract_one (the same frames forwarded in the production session, in ascending order among the eligible frames) stays as the
        # second gate: it is the one that measures the production path, and it can still stop the job after the sequences before
        # it were published (each of them has passed its own control; no manifest / summary exists until every sequence is done).
        t_pre = time.time()
        pre_rows = []
        for s in todo:
            ctrl_items = [(fi, osp.join(LOCAL, "src", s, dc.src_image_name(fi, infos[s]["imExt"]))) for fi in controls[s]]
            for fi, feat, stats in teacher.extract_seq(ctrl_items, infos[s]["width"], infos[s]["height"]):
                if tuple(stats["shape"]) != target_shape or stats["dtype"] != dc.TARGET_DTYPE:
                    raise core.BenchAssertError(f"{s} f{fi}: control target {stats['shape']} {stats['dtype']} (pre-pass)")
                if not (stats["finite"] and stats["std"] > core.MIN_STD):
                    raise core.BenchAssertError(f"{s} f{fi}: non-finite or collapsed control features (pre-pass) {stats}")
                cmp_ = h["compare_target"](feat, f"{BASE}/{base_row[(s, fi)]['feature']}")
                pre_rows.append({"seq": s, "fi": int(fi), **cmp_})
                if not cmp_["cos_global"] >= CONTROL_COS_GLOBAL_MIN:
                    raise core.BenchAssertError(f"positive control FAILED (pre-pass) on {s} f{fi}: {cmp_} (nothing is published)")
        if sorted((r["seq"], r["fi"]) for r in pre_rows) != sorted((s, fi) for s in todo for fi in controls[s]):
            raise core.BenchAssertError("the control pre-pass did not cover every control frame")
        run["control_prepass"] = {"n": len(pre_rows), "n_sequences": len(todo),
                                  "min_cos_global": min((r["cos_global"] for r in pre_rows), default=None),
                                  "max_max_abs": max((r["max_abs"] for r in pre_rows), default=None),
                                  "bit_identical": sum(1 for r in pre_rows if r["bit_identical"]), "gate": CONTROL_COS_GLOBAL_MIN,
                                  "seconds": round(time.time() - t_pre, 1)}
        log(f"control pre-pass: {run['control_prepass']}")

        # ---- 5. per sequence: extract (GPU, this thread; controls + cosine gates) -> publish (one background thread) -> record
        pub = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tol75_publish")
        futs = {}

        def publish_seq(s, seq_local, files, control, local_sha, t_extract):
            res = dc.publish_tree_attempt(seq_local, f"{OUT}/seqs", s, local_sha=local_sha, dbutils=dbu, log=log)
            if h.get("crash_after_dir_publish") == s:                          # selftest: crash before the record
                raise RuntimeError(f"simulated crash after publishing {s}")
            rec = {"seq": s, "dir": f"seqs/{res['name']}", "publish": {k: res[k] for k in ("mode", "files", "bytes", "seconds")},
                   "n": len(files), "files": files, "control": control, "extract_seconds": round(t_extract, 2),
                   "info": {k: infos[s][k] for k in ("n", "fps", "width", "height", "pen", "night")},
                   "role": "train", "nb": NB_VERSION,
                   "feat_mean_of_means": float(np.mean([f["feat_mean"] for f in files])),
                   "feat_mean_of_stds": float(np.mean([f["feat_std"] for f in files]))}
            lp = core.write_text_atomic(osp.join(LOCAL, "records", f"{s}.json"), dc.json_dumps(rec))
            core.publish_no_overwrite(lp, f"{rec_dir}/{s}.json", dbu)
            return res

        def extract_one(s):
            seq_local = osp.join(LOCAL, "out", "seqs", s)
            shutil.rmtree(seq_local, ignore_errors=True)
            os.makedirs(osp.join(seq_local, "frames"))
            os.makedirs(osp.join(seq_local, "features"))
            ctrl_set = set(controls[s])
            local_of = {fi: osp.join(LOCAL, "src", s, dc.src_image_name(fi, infos[s]["imExt"])) for fi in elig_fis[s] + controls[s]}
            items = [(fi, local_of[fi]) for fi in sorted(local_of)]
            t0 = time.time()
            files, control, local_sha = [], [], {}
            for fi, feat, stats in teacher.extract_seq(items, infos[s]["width"], infos[s]["height"]):
                if tuple(stats["shape"]) != target_shape or stats["dtype"] != dc.TARGET_DTYPE:
                    raise core.BenchAssertError(f"{s} f{fi}: target {stats['shape']} {stats['dtype']}")
                if not (stats["finite"] and stats["std"] > core.MIN_STD):
                    raise core.BenchAssertError(f"{s} f{fi}: non-finite or collapsed teacher features {stats}")
                cos = scorer.cos(local_of[fi], feat)
                if not math.isfinite(cos):
                    raise core.BenchAssertError(f"{s} f{fi}: non-finite student cosine {cos}")
                if fi in ctrl_set:                                               # positive control: NOT published here
                    cmp_ = h["compare_target"](feat, f"{BASE}/{base_row[(s, fi)]['feature']}")
                    control.append({"fi": int(fi), "cos": round(cos, 5), "night": int(infos[s]["night"]), **cmp_})
                    if not cmp_["cos_global"] >= CONTROL_COS_GLOBAL_MIN:
                        raise core.BenchAssertError(f"positive control FAILED on {s} f{fi}: {cmp_} (nothing of {s} is published)")
                    continue
                base = dc.target_basename(s, fi)
                fp = osp.join(seq_local, "features", base + ".pt")
                h["save_feature"](feat, fp)
                ip = osp.join(seq_local, "frames", base + ".jpg")
                shutil.copyfile(local_of[fi], ip)                       # byte-for-byte copy of the PigTrack JPEG
                fsha, isha = core.sha256_file(fp), core.sha256_file(ip)
                local_sha[f"features/{base}.pt"] = fsha
                local_sha[f"frames/{base}.jpg"] = isha
                files.append({"fi": int(fi), "src_image": dc.src_image_name(fi, infos[s]["imExt"]), "image": base + ".jpg",
                              "feature": base + ".pt", "image_bytes": osp.getsize(ip), "image_sha256": isha,
                              "feature_bytes": osp.getsize(fp), "feature_sha256": fsha, "feat_mean": float(stats["mean"]),
                              "feat_std": float(stats["std"]), "cos": round(cos, 5)})
            if [f["fi"] for f in files] != sorted(elig_fis[s]) or sorted(c["fi"] for c in control) != controls[s]:
                raise core.BenchAssertError(f"{s}: extracted {len(files)} targets + {len(control)} controls, "
                                            f"planned {len(elig_fis[s])} + {len(controls[s])}")
            bad = [c for c in scorer.checks if abs(c[0] - c[1]) >= CROSSCHECK_TOL]
            if bad:
                raise core.BenchAssertError(f"GPU cosine differs from fidelity_metrics cos_patch (float64, host): {bad[:3]}")
            return seq_local, files, control, local_sha, time.time() - t0

        t_gpu = time.time()
        try:
            for k, s in enumerate(todo):
                for fs, f in list(futs.items()):                                # fail fast on a publish error
                    if f.done() and f.exception() is not None:
                        raise RuntimeError(f"publishing failed for {fs}") from f.exception()
                seq_local, files, control, local_sha, t_ex = extract_one(s)
                futs[s] = pub.submit(publish_seq, s, seq_local, files, control, local_sha, t_ex)
                if P.progress_every and ((k + 1) % P.progress_every == 0 or k + 1 == len(todo)):
                    n_done = sum(len(elig_fis[x]) + len(controls[x]) for x in todo[:k + 1])
                    el = time.time() - t_gpu
                    log(f"  [{k + 1}/{len(todo)}] {s}: {len(files)} targets + {len(control)} controls in {t_ex:.1f} s; "
                        f"{n_done} forwards this run, {n_done / max(el, 1e-9):.2f} fwd/s, {el / 60:.1f} min")
            run["extract_seconds"] = round(time.time() - t_gpu, 1)
            errors = []
            for s, f in futs.items():
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    errors.append((s, e))
            run["publish_wait_seconds"] = round(time.time() - t_gpu - run["extract_seconds"], 1)
            if errors:
                raise RuntimeError(f"publishing failed for {[s for s, _ in errors]}") from errors[0][1]
        finally:
            pub.shutdown(wait=True, cancel_futures=True)        # never leave a publisher running behind an error
            teacher.close()
        run["crosscheck"] = {"n": len(scorer.checks), "max_abs_diff": max((abs(a - b) for a, b in scorer.checks), default=None),
                             "tol": CROSSCHECK_TOL}

    # ---- 6. records of this run's sequences
    records = {}
    for s in run_seqs:
        p = f"{rec_dir}/{s}.json"
        if not osp.isfile(p):
            raise RuntimeError(f"record of {s} missing after the run")
        records[s] = json.loads(_read(p))

    def publish_text(rel, text):
        lp = core.write_text_atomic(osp.join(LOCAL, "pub", rel), text)
        return core.publish_no_overwrite(lp, f"{OUT}/{rel}", dbu)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:6]     # one log per run
    ctrl_all = [dict(c, seq=s) for s in run_seqs for c in records[s]["control"]]
    if P.smoke_seqs:
        smoke = {"smoke_seqs": P.smoke_seqs, "n_targets": sum(r["n"] for r in records.values()), "controls": ctrl_all,
                 "control_bit_identical": sum(1 for c in ctrl_all if c["bit_identical"]), "n_controls": len(ctrl_all),
                 "crosscheck": run.get("crosscheck"), "pins_verified": True, "whitelist_sha256": whitelist_sha256(whitelist),
                 "R_R2_pins_verified": True, "nb": NB_VERSION}
        publish_text("SMOKE_SUMMARY.json" if not osp.isfile(f"{OUT}/SMOKE_SUMMARY.json") else f"SMOKE_SUMMARY_{stamp}.json",
                     dc.json_dumps(smoke))
        run.update({"seconds_total": round(time.time() - t_run, 1), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        publish_text(f"runs/{stamp}.json", dc.json_dumps(run))
        log(f"smoke: {smoke['n_targets']} targets, {smoke['n_controls']} controls ({smoke['control_bit_identical']} bit-identical)")
        return {"out": OUT, "smoke": True, "n_targets": smoke["n_targets"], "controls": smoke["n_controls"],
                "bit_identical": smoke["control_bit_identical"], "seconds_total": run["seconds_total"], "extracted_now": len(todo)}

    # ---- 7. selection from the stored scores (a pure function of candidates.csv), manifests, diagnostics, T1
    cand = []
    for s in whitelist:
        for f in records[s]["files"]:
            cand.append((s, int(f["fi"]), int(infos[s]["night"]), float(f["cos"])))
    if sorted((s, fi) for s, fi, _n, _c in cand) != sorted((s, fi) for s, fi, _ in pool):
        raise RuntimeError("the scored frames are not the eligible pool")
    sel_M, cnt_M = select_mined(cand, caps, night_q, day_q)
    sel_C = draw_comp(pool, cnt_M, SEED_COMP)
    sets = {"mined": sel_M, "random": sel_R, "random2": sel_R2, "comp": sel_C}
    counts = {"mined": cnt_M, "random": cnt_R, "random2": cnt_R2,
              "comp": collections.Counter(s for s, _ in sel_C)}
    night_of = {s: int(infos[s]["night"]) for s in whitelist}
    cos_of = {(s, fi): c for s, fi, _n, c in cand}
    base_pairs = set(base_train_pairs)
    for k, v in sets.items():
        if not (len(v) == M == len(set(v)) and sum(night_of[s] for s, _ in v) == night_q):
            raise SelectionError(f"set {k}: {len(v)} frames, {sum(night_of[s] for s, _ in v)} night (want {M} / {night_q})")
        if set(v) & base_pairs or not all(s in whitelist for s, _ in v):
            raise SelectionError(f"set {k}: contains a base train frame or a non-whitelisted sequence")
    if {s: c for s, c in counts["comp"].items()} != {s: c for s, c in cnt_M.items() if c}:
        raise SelectionError("comp counts != mined counts")
    t1 = t1_decision(cnt_M)
    rank = {p: i + 1 for i, p in enumerate(sorted(cos_of, key=lambda p: (cos_of[p], p[0], p[1])))}
    ctrl_rows = sorted(ctrl_all, key=lambda c: (c["seq"], c["fi"]))
    diag = build_diagnostics(pool=pool, caps=caps, cos_of=cos_of, sets=sets, counts=counts, control_rows=ctrl_rows, night_of=night_of)
    texts = {}
    texts["candidates.csv"] = csv_text(("seq", "fi", "night", "cos", "rank"),
                                       [(s, fi, nt, fmt_cos(c), rank[(s, fi)]) for s, fi, nt, c in sorted(cand)])
    texts["control_base.csv"] = csv_text(("seq", "fi", "night", "cos", "cos_global", "max_abs", "bit_identical"),
                                         [(c["seq"], c["fi"], c["night"], fmt_cos(c["cos"]), repr(float(c["cos_global"])),
                                           repr(float(c["max_abs"])), int(bool(c["bit_identical"]))) for c in ctrl_rows])
    for k, v in sets.items():
        texts[f"SELECTION_{k}.csv"] = csv_text(("seq", "fi", "night", "cos", "source", "rank"),
                                               [(s, fi, night_of[s], fmt_cos(cos_of[(s, fi)]), SET_SOURCE[k], rank[(s, fi)]) for s, fi in v])
        rows = set_manifest_rows(records, infos, v)
        dc.validate_manifest(rows)
        texts[f"manifest_{k}.csv"] = dc.manifest_csv(rows)
        ctrl_cos = {(c["seq"], c["fi"]): fmt_cos(c["cos"]) for c in ctrl_rows}
        ur = [(r["seq"], r["fi"], r["role"], "base", ctrl_cos.get((r["seq"], r["fi"]), ""), r["night"]) for r in base_rows]
        ur += [(s, fi, "train", SET_SOURCE[k], fmt_cos(cos_of[(s, fi)]), night_of[s]) for s, fi in v]
        texts[f"union_listing_{k}.csv"] = csv_text(("seq", "fi", "role", "source", "cos", "night"), sorted(ur, key=lambda t: (t[0], t[1])))
    summary = {
        "nb": NB_VERSION, "M": M, "night_q": night_q, "day_q": day_q, "n_eligible": len(pool), "n_base_train": len(base_pairs),
        "n_sequences": len(whitelist), "T1": t1,
        "set_canon_sha256": {k: canon_sha256(v) for k, v in sets.items()},
        "file_sha256": {n: hashlib.sha256(t.encode("utf-8")).hexdigest() for n, t in sorted(texts.items())},
        "positive_control": {"n": len(ctrl_rows), "bit_identical": sum(1 for c in ctrl_rows if c["bit_identical"]),
                             "min_cos_global": min(c["cos_global"] for c in ctrl_rows),
                             "max_max_abs": max(c["max_abs"] for c in ctrl_rows), "gate": CONTROL_COS_GLOBAL_MIN},
        "crosscheck": run.get("crosscheck"), "cos_selected_max": {k: max(cos_of[p] for p in v) for k, v in sets.items()},
        "run_completed": run["started_utc"],
        "note": "T1 decides whether the conditional composition-matched student (S-PT-comp) is trained; read it from here only",
    }
    texts["MINING_REPORT.json"] = dc.json_dumps(diag)
    texts["MINING_SUMMARY.json"] = dc.json_dumps(summary)
    scan_texts = dict(texts)
    scan_texts["MINING_CONFIG.json"] = _read(f"{OUT}/MINING_CONFIG.json")
    for s_ in whitelist:
        scan_texts[f"records/{s_}.json"] = _read(f"{rec_dir}/{s_}.json")
    problems = scan_outputs(scan_texts, fb, val_listing_ok={f"union_listing_{k}.csv" for k in SET_NAMES})
    if problems:
        raise RuntimeError("forbidden sequence names in the outputs (nothing of the post-processing is published): " + "; ".join(problems[:6]))
    for name in ("candidates.csv", "control_base.csv", *(f"SELECTION_{k}.csv" for k in SET_NAMES),
                 *(f"manifest_{k}.csv" for k in SET_NAMES), *(f"union_listing_{k}.csv" for k in SET_NAMES), "MINING_REPORT.json"):
        publish_text(name, texts[name])
    mrows = dc.read_manifest_csv(texts["manifest_mined.csv"])
    for r in mrows[:: max(1, len(mrows) // 3)][:3]:                    # end-to-end sha spot check of the Volume copies
        if core.sha256_file(f"{OUT}/{r['feature']}") != r["feature_sha256"]:
            raise RuntimeError(f"Volume copy of {r['feature']} differs from its record")
    if osp.isfile(f"{OUT}/MINING_SUMMARY.json"):                        # a resumed post-processing: must re-derive identically
        stored = json.loads(_read(f"{OUT}/MINING_SUMMARY.json"))
        if stored["file_sha256"] != summary["file_sha256"] or stored["set_canon_sha256"] != summary["set_canon_sha256"]:
            raise RuntimeError("MINING_SUMMARY.json on the Volume differs from the re-derived selection (records changed?)")
    else:
        publish_text("MINING_SUMMARY.json", texts["MINING_SUMMARY.json"])           # last = the post-processing done marker
    run.update({"manifest_sha256": {k: summary["file_sha256"][f"manifest_{k}.csv"] for k in SET_NAMES}, "T1": t1,
                "seconds_total": round(time.time() - t_run, 1), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    publish_text(f"runs/{stamp}.json", dc.json_dumps(run))
    log(f"selected {M} per set ({night_q} night); T1 top-{T1_TOPK} share {t1['top_share']:.3f} -> fired={t1['fired']}; "
        f"M-R overlap {diag['overlaps']['mined-random']}, M-R2 {diag['overlaps']['mined-random2']}; run {run['seconds_total'] / 60:.1f} min")
    return {"out": OUT, "smoke": False, "M": M, "T1": t1, "manifest_sha256": run["manifest_sha256"],
            "set_canon_sha256": summary["set_canon_sha256"], "extracted_now": len(todo), "seconds_total": run["seconds_total"],
            "positive_control": summary["positive_control"], "prompt_independence": pi, "staging": run.get("staging"),
            "extract_seconds": run.get("extract_seconds")}


# COMMAND ----------

# MAGIC %md
# MAGIC ## Selftest (local only): fake PigTrack tree + registry, stub teacher / student, controls, resume, whitelist refusal, leakage guards

# COMMAND ----------

def _selftest():
    import builtins
    import contextlib
    from PIL import Image
    results = []

    def run(name, fn):
        try:
            fn()
            results.append((name, "PASS", ""))
        except Exception:  # noqa: BLE001
            results.append((name, "FAIL", traceback.format_exc()))

    def raises(fn, exc, text=""):
        try:
            fn()
        except exc as e:
            assert text in str(e), (text, str(e))
            return
        raise AssertionError(f"no {exc.__name__} ({text})")

    reg_src = None
    for d in (CODE_DIR, osp.join(osp.dirname(CODE_DIR), "catalog", "build"), osp.join(CODE_DIR, "_vol")):
        if osp.isfile(osp.join(d, "paths_v1.yaml")):
            reg_src = osp.join(d, "paths_v1.yaml")
            break
    if reg_src is None:
        raise RuntimeError("paths_v1.yaml not found next to the code (bench/_vol or catalog/build)")
    td = tempfile.mkdtemp(prefix="tol75_st_")
    SHAPE = (1, 8, 4, 4)
    N_BASE = 40
    STUB_REV = "3c879f39826c281e95690f02c7821c4de09afae7"
    # name -> (images, width, height, pen, night). Val rule (scaled cells): (day,64x40) -> 0003, (day,96x54) -> 0041,
    # (night,64x40) -> 0050, then 0052; whitelist = the other six.
    SEQS = {"pigtrack0003": (24, 1280, 800, "1", 0), "pigtrack0009": (30, 1280, 800, "2", 0),
            "pigtrack0012": (36, 1280, 800, "3", 0), "pigtrack0013": (20, 1280, 800, "4", 0),
            "pigtrack0041": (26, 1920, 1080, "5", 0), "pigtrack0044": (22, 1920, 1080, "6", 0),
            "pigtrack0050": (28, 1280, 800, "7", 1), "pigtrack0052": (34, 1280, 800, "8", 1),
            "pigtrack0053": (22, 1280, 800, "9", 1), "pigtrack0057": (40, 1280, 800, "1", 1)}
    dc_cells = dc.VAL_CELLS
    dc.VAL_CELLS = tuple((nt, (wh[0] // 20, wh[1] // 20)) for nt, wh in dc_cells)
    choose_orig = dc.choose_val_sequences
    dc.choose_val_sequences = lambda infos, cells=None: choose_orig(infos, dc.VAL_CELLS)

    def _content_seed(seed, path):
        with open(path, "rb") as f:
            b = f.read()
        return int(hashlib.sha256(str(seed).encode() + b).hexdigest()[:8], 16)

    class StubTeacher:
        def __init__(self, seed):
            self.seed = seed
            self.info = {"commit_hash": STUB_REV, "attn_resolved": {"config": "sdpa"}, "stub": True}

        def prompt_independence(self, path, w, h):
            assert osp.isfile(path)
            return {"max_abs": 0.0, "cos_global": 1.0}

        def extract_seq(self, items, w, h):
            fis = [fi for fi, _ in items]
            assert fis == sorted(fis), "frames must arrive in ascending order"
            for fi, path in items:
                with Image.open(path) as im:
                    assert im.size == (w, h), (im.size, w, h)
                a = np.random.default_rng(_content_seed(self.seed, path)).normal(0, 1.4, size=SHAPE).astype(np.float16)
                yield fi, a, {"shape": a.shape, "dtype": "float16", "finite": bool(np.isfinite(a).all()),
                              "mean": float(a.astype(np.float32).mean()), "std": float(a.astype(np.float32).std())}

        def close(self):
            pass

    class LateBadTeacher(StubTeacher):
        """Correct everywhere except for the frames of `bad_seq` (found through the staged path). mode 'always': wrong in every session
        (the control pre-pass sees it); mode 'session': wrong only in a production session, i.e. one with more than the control frames
        (only the per-sequence control inside extract_one sees it)."""
        def __init__(self, bad_seq, mode):
            super().__init__(0)
            self.bad_seq, self.mode = bad_seq, mode

        def extract_seq(self, items, w, h):
            bad = (f"/{self.bad_seq}/" in items[0][1].replace("\\", "/")
                   and (self.mode == "always" or len(items) > N_CONTROL_PER_SEQ))
            for fi, feat, stats in super().extract_seq(items, w, h):
                if bad:
                    feat = (feat.astype(np.float32) + np.random.default_rng(fi).normal(0, 1.0, size=feat.shape)).astype(np.float16)
                    f32 = feat.astype(np.float32)
                    stats = dict(stats, mean=float(f32.mean()), std=float(f32.std()), finite=bool(np.isfinite(f32).all()))
                yield fi, feat, stats

    class StubScorer:
        info = {"stub": True}

        def __init__(self):
            self.checks = []

        def cos(self, path, feat):
            c = 0.3 + 0.6 * (_content_seed("cos", path) / 2 ** 32)
            if len(self.checks) < CROSSCHECK_N:
                self.checks.append((c, c))
            return c

    def stub_cos(path):
        return round(0.3 + 0.6 * (_content_seed("cos", path) / 2 ** 32), 5)

    def save_np(a, p):
        with open(p, "wb") as f:
            np.save(f, a)

    def compare_np(feat, stored_path):
        stored = np.load(stored_path)
        a, b = feat.astype(np.float64).reshape(-1), stored.astype(np.float64).reshape(-1)
        return {"cos_global": float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))),
                "max_abs": float(np.abs(a - b).max()), "bit_identical": bool(np.array_equal(feat, stored))}

    def make_world(tag):
        vol = osp.join(td, tag, "Volumes", "cat", "sch", "vol")
        text = re.sub(r'(?m)^  volume: ".*"$', f'  volume: "{vol.replace(chr(92), "/")}"', _read(reg_src))
        reg = osp.join(td, tag + "_paths_v1.yaml")
        with open(reg, "w", encoding="utf-8") as f:
            f.write(text)
        R = tol_paths.load(path=reg)
        root = R.path("datasets.pigtrack.sequences", None, "train")
        split_lines = []
        for k_s, (s, (n, w, hgt, pen, night)) in enumerate(SEQS.items()):
            os.makedirs(osp.join(root, s, "img1"))
            os.makedirs(osp.join(root, s, "gt"))
            with open(osp.join(root, s, "gt", "gt.txt"), "w") as f:
                f.write("POISON: the mining must never read GT\n")
            with open(osp.join(root, s, "seqinfo.ini"), "w") as f:
                f.write(f"[Sequence]\nname={s}\nimDir=img1\nframeRate=10\nseqLength={n}\nimWidth={w // 20}\n"
                        f"imHeight={hgt // 20}\nimExt=.jpg\npenID={pen}\nnight={night}\n")
            for k in range(n):
                Image.new("RGB", (w // 20, hgt // 20), ((k * 9) % 255, 40 + k_s, 90)).save(osp.join(root, s, "img1", f"{k + 1:08d}.jpg"))
            split_lines.append(f"{s} train")
        for s, sp in (("pigtrack0001", "val"), ("pigtrack0002", "test"), ("pigtrack0006", "val"), ("pigtrack0004", "test")):
            d = R.path("datasets.pigtrack.sequences", None, sp)
            os.makedirs(osp.join(d, s, "img1"))
            os.makedirs(osp.join(d, s, "gt"))
            Image.new("RGB", (64, 40), (1, 2, 3)).save(osp.join(d, s, "img1", "00000001.jpg"))
            split_lines.append(f"{s} {sp}")
        os.makedirs(R.path("datasets.pigtrack.seqmaps"), exist_ok=True)
        split_text = "\n".join(split_lines) + "\n"
        with open(R.path("datasets.pigtrack.split_file"), "w", newline="\n") as f:
            f.write(split_text)
        with open(R.path("datasets.pigtrack.seqmaps", None, "pigtrack-train.txt"), "w", newline="\n") as f:
            f.write("name\n" + "\n".join(sorted(SEQS)) + "\n")
        os.makedirs(R.path("paper1.root", None, "protocol"), exist_ok=True)
        with open(R.path("paper1.root", None, "protocol", "hf_revisions_v1.json"), "w") as f:
            json.dump({"revisions": {HF_SAM3: STUB_REV}}, f)
        train, infos, seq_root, _src = discover(R, len(SEQS))
        val4, _ = dc.choose_val_sequences(infos)
        assert val4 == ["pigtrack0003", "pigtrack0041", "pigtrack0050", "pigtrack0052"], val4
        plan = dc.selection_plan(infos, N_BASE)
        BASE = R.path(BASE_TARGETS_KEY)
        t0 = StubTeacher(0)
        records = {}
        for s in train:
            os.makedirs(f"{BASE}/seqs/{s}/frames")
            os.makedirs(f"{BASE}/seqs/{s}/features")
            items = [(fi, osp.join(seq_root, s, "img1", dc.src_image_name(fi))) for fi in plan[s]]
            files = []
            for fi, feat, stats in t0.extract_seq(items, infos[s]["width"], infos[s]["height"]):
                b = dc.target_basename(s, fi)
                fp, ip = f"{BASE}/seqs/{s}/features/{b}.pt", f"{BASE}/seqs/{s}/frames/{b}.jpg"
                save_np(feat, fp)
                shutil.copyfile(dict(items)[fi], ip)
                files.append({"fi": fi, "src_image": dc.src_image_name(fi), "image": b + ".jpg", "feature": b + ".pt",
                              "image_bytes": osp.getsize(ip), "image_sha256": core.sha256_file(ip),
                              "feature_bytes": osp.getsize(fp), "feature_sha256": core.sha256_file(fp),
                              "feat_mean": float(stats["mean"]), "feat_std": float(stats["std"])})
            records[s] = {"dir": f"seqs/{s}", "files": files}
        man_text = dc.manifest_csv(dc.manifest_rows(records, infos, val4))
        tcfg_text = json.dumps({"val": {"seqs": val4}, "selection": {"n_targets": N_BASE, "plan_sha256": dc.plan_digest(plan)}})
        with open(f"{BASE}/manifest.csv", "w", encoding="utf-8", newline="\n") as f:
            f.write(man_text)
        with open(f"{BASE}/TARGETS_CONFIG.json", "w", encoding="utf-8", newline="\n") as f:
            f.write(tcfg_text)
        whitelist = sorted(set(train) - set(val4))
        pool = eligible_pool(infos, plan, whitelist)
        M = derive_M(len(pool))
        nq, dq = derive_quotas(M, len(pool), sum(nt for *_x, nt in pool))
        caps = caps_for(pool)
        selR, _ = draw_random(pool, caps, SEED_R, nq, dq)
        selR2, _ = draw_random(pool, caps, SEED_R2, nq, dq)
        ckpt = student_ckpt_path(R)
        os.makedirs(osp.dirname(ckpt), exist_ok=True)
        with open(ckpt, "wb") as f:
            f.write(b"stub student")
        sh = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()
        pins = {"whitelist_sha256": whitelist_sha256(whitelist), "n_pool": sum(infos[s]["n"] for s in whitelist),
                "n_base_train": sum(len(plan[s]) for s in whitelist), "n_eligible": len(pool),
                "n_eligible_night": sum(nt for *_x, nt in pool), "eligible_canon_sha256": canon_sha256([(s, fi) for s, fi, _ in pool]),
                "M": M, "night_q": nq, "day_q": dq, "R_canon_sha256": canon_sha256(selR), "R2_canon_sha256": canon_sha256(selR2),
                "base_manifest_sha256": sh(man_text), "base_targets_config_sha256": sh(tcfg_text),
                "base_plan_sha256": dc.plan_digest(plan), "split_sha256": sh(split_text),
                "student_sha256": core.sha256_file(ckpt), "teacher_revision": STUB_REV}
        return {"R": R, "tag": tag, "infos": infos, "plan": plan, "val4": val4, "whitelist": whitelist, "pool": pool, "pins": pins,
                "root": root, "BASE": BASE, "man_text": man_text, "M": M, "nq": nq, "dq": dq, "caps": caps}

    def hooks(w, seed=0, **kw):
        h = {"registry": w["R"], "check_deployed_paths": False, "check_typed_forbidden": False, "expect_train_seqs": len(SEQS),
             "local_root": osp.join(td, "local_" + w["tag"]), "dbutils": None, "log": lambda *a: None, "pins": w["pins"],
             "target_shape": SHAPE, "gpu_preflight": lambda P: {"class": "A100", "name": "stub"},
             "versions": lambda: {"torch": "stub"},
             "code_shas": lambda: {"nb_version": "stub", "tol_bench_core": "x", "tol_distill_core": "y", "tol_paths": "z"},
             "hf_token": lambda d: "x", "load_teacher": lambda rev, tok: StubTeacher(seed), "load_scorer": lambda p, sha: StubScorer(),
             "compare_target": compare_np, "save_feature": save_np, "free_gb": lambda p: 1e9}
        h.update(kw)
        return h

    @contextlib.contextmanager
    def leak_guard(val4):
        """No file of a val / test sequence, no GT file, no image of a distillation-val train sequence may be opened."""
        rx = re.compile(r"/sequences/(val|test)/|/gt/|/train/(%s)/img1/\d+\.jpg" % "|".join(val4))
        real_open, real_listdir = builtins.open, os.listdir
        seen = []

        def g_open(p, *a, **k):
            sp = str(p).replace("\\", "/")
            if rx.search(sp):
                raise AssertionError(f"forbidden file opened: {sp}")
            seen.append(sp)
            return real_open(p, *a, **k)

        def g_listdir(p="."):
            sp = str(p).replace("\\", "/")
            if re.search(r"/sequences/(val|test)(/|$)", sp):
                raise AssertionError(f"forbidden directory listed: {sp}")
            return real_listdir(p)
        builtins.open, os.listdir = g_open, g_listdir
        try:
            yield seen
        finally:
            builtins.open, os.listdir = real_open, real_listdir

    def out_of(w):
        return out_dir_for(w["R"], OUT_NAME_DEFAULT)

    # ------------------------------------------------------------------------------------------------ pure functions
    def t_params():
        P = parse_params({})
        assert P.out_name == OUT_NAME_DEFAULT and P.whitelist == list(WHITELIST31) and not P.smoke_seqs
        P = parse_params({"smoke_seqs": "pigtrack0028", "out_name": "smoke_j14"})
        assert P.smoke_seqs == ["pigtrack0028"]
        for bad in ({"nope": "1"}, {"out_name": "other"}, {"smoke_seqs": "pigtrack0028"}, {"smoke_seqs": "pigtrack0028", "out_name": "x"},
                    {"require_gpu": "T4"}, {"sam3_revision": "main"}, {"copy_threads": "x"}):
            raises(lambda bad=bad: parse_params(bad), ValueError)

    def t_typed_constants():
        assert len(WHITELIST31) == 31 == len(set(WHITELIST31)) and list(WHITELIST31) == sorted(WHITELIST31)
        assert whitelist_sha256(WHITELIST31) == PINS["whitelist_sha256"]
        assert len(VAL4) == 4 and len(OFFICIAL_VAL20) == 20 and len(TEST25) == 25
        allv = set(WHITELIST31) | set(VAL4) | set(OFFICIAL_VAL20) | set(TEST25)
        assert len(allv) == 80 and not (set(WHITELIST31) & (set(VAL4) | set(OFFICIAL_VAL20) | set(TEST25)))
        assert sorted(set(WHITELIST31) | set(VAL4)) == sorted(set(allv) - set(OFFICIAL_VAL20) - set(TEST25))

    def t_numbers():
        assert derive_M(6094) == PINS["M"] == 2125
        assert derive_quotas(2125, 6094, 2142) == (PINS["night_q"], PINS["day_q"]) == (747, 1378)
        assert derive_M(100) == 34 and derive_quotas(3, 100, 50) == (2, 1)      # round() = banker's: 1.5 -> 2

    def synth_pool():
        pool, infos = [], {}
        spec = {"s1": (60, 1), "s2": (60, 1), "s3": (40, 1), "d1": (80, 0), "d2": (50, 0), "d3": (30, 0)}
        for s, (n, nt) in spec.items():
            infos[s] = {"n": n, "night": nt}
            pool += [(s, fi, nt) for fi in range(n) if fi % 3 != 1]
        return pool

    def t_selection():
        pool = synth_pool()
        caps = caps_for(pool)
        assert caps == {s: int(math.floor(0.79 * sum(1 for p in pool if p[0] == s))) for s in caps}
        rng = np.random.default_rng(7)
        cands = [(s, fi, nt, round(float(rng.uniform(0.4, 0.9)), 5)) for s, fi, nt in pool]
        nq, dq = 25, 40
        sel, cnt = select_mined(cands, caps, nq, dq)
        assert len(sel) == len(set(sel)) == 65 and sel == sorted(sel)
        night_of = {s: nt for s, _f, nt in pool}
        assert sum(night_of[s] for s, _ in sel) == nq and all(cnt[s] <= caps[s] for s in cnt)
        # hand-computed expectation for the night stratum: hardest first with the cap
        exp, c2 = [], collections.Counter()
        for s, fi, nt, cos in sorted((c for c in cands if c[2]), key=lambda c: (c[3], c[0], c[1])):
            if len(exp) >= nq:
                break
            if c2[s] < caps[s]:
                exp.append((s, fi))
                c2[s] += 1
        assert sorted(p for p in sel if night_of[p[0]]) == sorted(exp)
        # a cap that binds: every s1 frame is the hardest, quota larger than its cap
        adv = [(s, fi, nt, (0.1 if s == "s1" else 0.5 + fi / 1000)) for s, fi, nt in pool]
        sel2, cnt2 = select_mined(adv, caps, 50, 40)
        assert cnt2["s1"] == caps["s1"] and sum(1 for p in sel2 if night_of[p[0]]) == 50
        # ties are broken by (seq, fi)
        tie = [(s, fi, nt, 0.5) for s, fi, nt in pool]
        sel3, _ = select_mined(tie, caps, 10, 10)
        first_night = [(s, fi) for s, fi, nt in sorted(pool) if nt][:10]
        assert [p for p in sel3 if night_of[p[0]]] == first_night
        # infeasible quota under the caps raises
        raises(lambda: select_mined(cands, caps, 10 ** 4, 40), SelectionError)
        # deterministic, and independent of the input order
        assert select_mined(list(reversed(cands)), caps, nq, dq)[0] == sel
        # random draws: cosine-free, deterministic, seed-dependent, quota- and cap-respecting
        r1, c1 = draw_random(pool, caps, SEED_R, nq, dq)
        r1b, _ = draw_random(list(reversed(pool)), caps, SEED_R, nq, dq)
        r2, _ = draw_random(pool, caps, SEED_R2, nq, dq)
        assert r1 == r1b and r1 != r2 and len(r1) == 65 and all(c1[s] <= caps[s] for s in c1)
        assert sum(night_of[s] for s, _ in r1) == nq
        first = min(((s, fi) for s, fi, nt in pool if nt), key=lambda t: hash_key(SEED_R, t[0], t[1]))
        assert first in r1
        # composition-matched draw
        comp = draw_comp(pool, cnt, SEED_COMP)
        assert collections.Counter(s for s, _ in comp) == collections.Counter({s: c for s, c in cnt.items() if c}) and comp == sorted(comp)
        assert draw_comp(pool, cnt, SEED_COMP) == comp and draw_comp(pool, cnt, SEED_COMP + 1) != comp
        raises(lambda: draw_comp(pool, {"s3": 10 ** 4}, SEED_COMP), SelectionError)

    def t_t1_and_diag():
        c = collections.Counter({f"q{i}": v for i, v in enumerate([20, 20, 20, 20, 10, 10, 10, 10])})   # total 120, top6 = 100
        assert t1_decision(c)["fired"] and abs(t1_decision(c)["top_share"] - 100 / 120) < 1e-12
        half = collections.Counter({f"q{i}": v for i, v in enumerate([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])})   # 60 / 120 = 50 % exactly
        assert t1_decision(half)["top_share"] == 0.5 and not t1_decision(half)["fired"]
        half["q0"] += 1                                  # top6 = 61 of 121 > 50 %
        assert t1_decision(half)["fired"] and t1_decision(half)["top_share"] > 0.5
        assert n_for_share(collections.Counter({"a": 5, "b": 3, "c": 2}), 0.5) == 1 and n_for_share(collections.Counter({"a": 4, "b": 3, "c": 3}), 0.5) == 2
        assert adjacent_share([("a", 1), ("a", 2), ("a", 5), ("b", 5)]) == 0.5 and overlap([(1, 1), (2, 2)], [(2, 2), (3, 3)]) == 1
        d = describe([1, 2, 3, 4, 5])
        assert d["n"] == 5 and d["mean"] == 3.0 and d["p50"] == 3.0 and describe([])["n"] == 0

    def t_controls_and_whitelist(w=None):
        plan = {s: list(range(0, 30, 3)) for s in ("a", "b")}
        c = control_frames(plan, ["a", "b"])
        assert all(len(v) == 2 and set(v) <= set(plan[s]) and v == sorted(v) for s, v in c.items()) and control_frames(plan, ["b", "a"]) == c
        raises(lambda: control_frames({"a": [1]}, ["a"]), SelectionError)
        # whitelist refusal: every kind of violation
        train = ["pigtrack0003", "pigtrack0009", "pigtrack0041", "pigtrack0050", "pigtrack0052"]
        split = {"pigtrack0003": "train", "pigtrack0009": "train", "pigtrack0041": "train", "pigtrack0050": "train",
                 "pigtrack0052": "train", "pigtrack0001": "val", "pigtrack0002": "test"}
        val4 = ["pigtrack0003", "pigtrack0041", "pigtrack0050", "pigtrack0052"]
        assert check_whitelist(["pigtrack0009"], train=train, val_seqs=val4, split=split) == ["pigtrack0009"]
        for bad, text in ((["pigtrack0009", "pigtrack0041"], "val4"), (["pigtrack0009", "pigtrack0001"], "official_val"),
                          (["pigtrack0009", "pigtrack0002"], "test"), (["pigtrack0009", "pigtrack0009"], "duplicate"),
                          (["pigtrack0099"], "not PigTrack TRAIN"), ([], "empty"), (["pigtrack9"], "malformed")):
            raises(lambda bad=bad: check_whitelist(bad, train=train, val_seqs=val4, split=split), WhitelistError, text)
        raises(lambda: check_whitelist(["pigtrack0009"], train=train + ["pigtrack0013"], val_seqs=val4,
                                       split=dict(split, pigtrack0013="train")), WhitelistError, "missing")
        raises(lambda: check_whitelist(["pigtrack0009"], train=train, val_seqs=val4, split=split, expect_sha256="0" * 64),
               WhitelistError, "sha256")

    def t_scan():
        fb = {"val4": ["pigtrack0041"], "official_val": ["pigtrack0001"], "test": ["pigtrack0002"]}
        ok = {"a.csv": "seq,fi\npigtrack0009,1\n", "union_listing_mined.csv": "seq,fi,role,source,cos,night\npigtrack0041,1,val,base,,0\n"}
        assert scan_outputs(ok, fb, val_listing_ok={"union_listing_mined.csv"}) == []
        for name, text, n in (("a.csv", "pigtrack0002", 1), ("b.csv", "x pigtrack0001 y", 1), ("c.csv", "pigtrack0041", 1)):
            assert len(scan_outputs({name: text}, fb, val_listing_ok={"union_listing_mined.csv"})) == n
        bad_row = {"union_listing_mined.csv": "seq,fi,role,source,cos,night\npigtrack0041,1,train,mined,0.5,0\n"}
        assert len(scan_outputs(bad_row, fb, val_listing_ok={"union_listing_mined.csv"})) == 1

    # ------------------------------------------------------------------------------------------------ end to end (stubs)
    W1 = make_world("w1")
    Wp = {"whitelist": ",".join(W1["whitelist"]), "min_free_gb": "0"}

    def t_guard_works():
        with leak_guard(W1["val4"]):
            raises(lambda: open(osp.join(W1["root"], "pigtrack0041", "img1", "00000001.jpg"), "rb"), AssertionError, "forbidden")
            raises(lambda: open(osp.join(W1["root"], "pigtrack0009", "gt", "gt.txt")), AssertionError, "forbidden")
            raises(lambda: os.listdir(osp.join(osp.dirname(W1["root"]), "test")), AssertionError, "forbidden")
            open(osp.join(W1["root"], "pigtrack0009", "img1", "00000001.jpg"), "rb").close()

    def t_full_run():
        with leak_guard(W1["val4"]) as seen:
            out = main(Wp, hooks=hooks(W1))
        OUT = out_of(W1)
        assert not out["smoke"] and out["extracted_now"] == 6 and out["M"] == W1["M"], out
        assert any("/train/pigtrack0009/img1/" in p for p in seen) and not any("/test/" in p or "/val/" in p for p in seen)
        assert sorted(os.listdir(f"{OUT}/records")) == [f"{s}.json" for s in W1["whitelist"]]
        names = set(os.listdir(OUT))
        for need in ("MINING_CONFIG.json", "candidates.csv", "control_base.csv", "MINING_SUMMARY.json", "MINING_REPORT.json", "runs", "seqs",
                     *(f"SELECTION_{k}.csv" for k in SET_NAMES), *(f"manifest_{k}.csv" for k in SET_NAMES),
                     *(f"union_listing_{k}.csv" for k in SET_NAMES)):
            assert need in names, need
        # all eligible frames published, in the Tol_66 layout, with the recorded sha256
        pool = W1["pool"]
        for s in W1["whitelist"]:
            rec = json.loads(_read(f"{OUT}/records/{s}.json"))
            assert rec["n"] == sum(1 for p in pool if p[0] == s) == len(os.listdir(f"{OUT}/seqs/{s}/features")) and rec["role"] == "train"
            assert len(rec["control"]) == 2 and all(c["bit_identical"] and c["cos_global"] >= 0.9999 for c in rec["control"])
        # candidates.csv re-derives M offline exactly (selection is a pure function of it)
        lines = _read(f"{OUT}/candidates.csv").splitlines()
        assert lines[0] == "seq,fi,night,cos,rank" and len(lines) == 1 + len(pool)
        cands = [(c[0], int(c[1]), int(c[2]), float(c[3])) for c in (ln.split(",") for ln in lines[1:])]
        sel, cnt = select_mined(cands, W1["caps"], W1["nq"], W1["dq"])
        sel_file = [(c[0], int(c[1])) for c in (ln.split(",") for ln in _read(f"{OUT}/SELECTION_mined.csv").splitlines()[1:])]
        assert sel == sel_file and len(sel) == W1["M"]
        # the stored cosines are what the stub scorer returns on the byte-identical staged frame
        for s, fi, _nt, cos in cands[:20]:
            assert cos == stub_cos(osp.join(W1["root"], s, "img1", dc.src_image_name(fi)))
        # the other sets are the pre-registered draws; comp has M's per-sequence counts
        def sel_of(k):
            return [(c[0], int(c[1])) for c in (ln.split(",") for ln in _read(f"{OUT}/SELECTION_{k}.csv").splitlines()[1:])]
        assert canon_sha256(sel_of("random")) == W1["pins"]["R_canon_sha256"] and canon_sha256(sel_of("random2")) == W1["pins"]["R2_canon_sha256"]
        assert collections.Counter(s for s, _ in sel_of("comp")) == collections.Counter(s for s, _ in sel)
        # manifests: Tol_66 columns, role=train, relative paths that exist, sha256 equal, no overlap with the base, whitelist only
        base_rows = dc.read_manifest_csv(W1["man_text"])
        base_pairs = {(r["seq"], r["fi"]) for r in base_rows}
        for k in SET_NAMES:
            rows = dc.read_manifest_csv(_read(f"{OUT}/manifest_{k}.csv"))
            assert len(rows) == W1["M"] and {r["role"] for r in rows} == {"train"}
            assert not ({(r["seq"], r["fi"]) for r in rows} & base_pairs) and {r["seq"] for r in rows} <= set(W1["whitelist"])
            assert sorted((r["seq"], r["fi"]) for r in rows) == sorted(sel_of(k))
            for r in rows[::7]:
                assert core.sha256_file(f"{OUT}/{r['feature']}") == r["feature_sha256"] and osp.isfile(f"{OUT}/{r['image']}")
            ul = _read(f"{OUT}/union_listing_{k}.csv").splitlines()
            assert ul[0] == "seq,fi,role,source,cos,night" and len(ul) == 1 + len(base_rows) + W1["M"]
            assert sum(1 for ln in ul[1:] if ln.split(",")[3] == SET_SOURCE[k]) == W1["M"]
            assert sum(1 for ln in ul[1:] if ln.split(",")[2] == "val") == sum(1 for r in base_rows if r["role"] == "val")
        # forbidden names: none of the test / official-val names anywhere, val names only in the union listings
        fb = forbidden_from_split(dc.parse_split_text(_read(W1["R"].path("datasets.pigtrack.split_file"))), W1["val4"])
        files = {}
        for d, _ds, fs in os.walk(OUT):
            for f in fs:
                if f.endswith((".csv", ".json")):
                    files[osp.relpath(osp.join(d, f), OUT).replace("\\", "/")] = _read(osp.join(d, f))
        assert scan_outputs(files, fb, val_listing_ok={f"union_listing_{k}.csv" for k in SET_NAMES}) == [], "forbidden names in outputs"
        # summary: T1 equals the rule applied to M's counts; positive control all bit-identical
        sm = json.loads(_read(f"{OUT}/MINING_SUMMARY.json"))
        assert sm["T1"] == t1_decision(cnt) and sm["positive_control"]["n"] == 12 and sm["positive_control"]["bit_identical"] == 12
        assert sm["set_canon_sha256"]["mined"] == canon_sha256(sel) and sm["M"] == W1["M"]
        # the control pre-pass ran before the first publish, covered every control frame, and is in the run record
        runs_ = [json.loads(_read(f"{OUT}/runs/{n}")) for n in os.listdir(f"{OUT}/runs")]
        assert len(runs_) == 1, runs_
        cp = runs_[0]["control_prepass"]
        assert cp["n"] == 12 and cp["n_sequences"] == 6 and cp["bit_identical"] == 12 and cp["min_cos_global"] >= 0.9999, cp
        rep = json.loads(_read(f"{OUT}/MINING_REPORT.json"))
        assert set(rep) >= {"cos_eligible", "cos_selected", "cos_control_base", "per_sequence", "selected_vs_pool", "overlaps"}
        assert set(rep["per_sequence"]) == set(W1["whitelist"]) and rep["selected_vs_pool"]["mined"]["n"] == W1["M"]
        # hardest-first: the mined set is harder than the pool
        assert rep["selected_vs_pool"]["mined"]["cos_mean"] < rep["selected_vs_pool"]["mined"]["pool_cos_mean"]

    def t_resume_noop():
        before = {n: os.listdir(f"{out_of(W1)}/{n}") for n in ("records", "runs")}
        out = main(Wp, hooks=hooks(W1))
        assert out["extracted_now"] == 0, out
        assert len(os.listdir(f"{out_of(W1)}/runs")) == len(before["runs"]) + 1

    def t_config_mismatch():
        raises(lambda: main(Wp, hooks=hooks(W1, versions=lambda: {"torch": "other"})), RuntimeError, "mismatch")

    def t_crash_resume():
        w = make_world("w2")
        raises(lambda: main(Wp, hooks=hooks(w, crash_after_dir_publish="pigtrack0013")), RuntimeError, "publishing failed")
        OUT = out_of(w)
        assert osp.isdir(f"{OUT}/seqs/pigtrack0013") and not osp.isfile(f"{OUT}/records/pigtrack0013.json")
        assert not osp.isfile(f"{OUT}/MINING_SUMMARY.json")
        out = main(Wp, hooks=hooks(w))
        assert out["extracted_now"] >= 1 and osp.isfile(f"{OUT}/MINING_SUMMARY.json")
        rec = json.loads(_read(f"{OUT}/records/pigtrack0013.json"))
        assert rec["dir"] == "seqs/pigtrack0013" and rec["publish"]["mode"] in ("identical", "resume"), rec["publish"]

    def t_control_failure():
        w = make_world("w3")
        raises(lambda: main(Wp, hooks=hooks(w, seed=1)), core.BenchAssertError, "positive control FAILED")
        OUT = out_of(w)
        assert not osp.exists(f"{OUT}/seqs") and not osp.exists(f"{OUT}/records"), "a failed control must publish nothing"
        assert not osp.isfile(f"{OUT}/MINING_SUMMARY.json") and not any(n.startswith(("manifest_", "SELECTION_")) for n in os.listdir(OUT))

    def t_control_prepass_nothing_published():
        # the failing sequence is the LAST one: without the pre-pass the sequences before it would already have been published
        w = make_world("w8")
        last = sorted(w["whitelist"])[-1]
        raises(lambda: main(Wp, hooks=hooks(w, load_teacher=lambda rev, tok: LateBadTeacher(last, "always"))),
               core.BenchAssertError, "positive control FAILED (pre-pass)")
        OUT = out_of(w)
        assert not osp.exists(f"{OUT}/seqs") and not osp.exists(f"{OUT}/records"), "a pre-pass failure must publish nothing at all"
        assert not osp.isfile(f"{OUT}/MINING_SUMMARY.json") and not any(n.startswith(("manifest_", "SELECTION_")) for n in os.listdir(OUT))

    def t_control_session_gate():
        # a control that fails only inside the production session (second gate): the pre-pass passes, the failing sequence is never
        # recorded or published, and there is no summary / manifest (earlier sequences may be published: each passed its own control)
        w = make_world("w9")
        last = sorted(w["whitelist"])[-1]
        raises(lambda: main(Wp, hooks=hooks(w, load_teacher=lambda rev, tok: LateBadTeacher(last, "session"))),
               core.BenchAssertError, f"positive control FAILED on {last}")
        OUT = out_of(w)
        assert not osp.isfile(f"{OUT}/records/{last}.json") and not osp.exists(f"{OUT}/seqs/{last}")
        assert not osp.isfile(f"{OUT}/MINING_SUMMARY.json") and not any(n.startswith(("manifest_", "SELECTION_")) for n in os.listdir(OUT))

    def t_crosscheck_failure():
        w = make_world("w4")

        class BadScorer(StubScorer):
            def cos(self, path, feat):
                c = super().cos(path, feat)
                self.checks[-1] = (c, c + 1e-3)
                return c
        raises(lambda: main(Wp, hooks=hooks(w, load_scorer=lambda p, sha: BadScorer())), core.BenchAssertError, "differs from fidelity_metrics")
        assert not osp.exists(f"{out_of(w)}/seqs")

    def t_smoke():
        w = make_world("w5")
        S = dict(Wp, smoke_seqs="pigtrack0013", out_name="smoke_j14")
        with leak_guard(w["val4"]):
            out = main(S, hooks=hooks(w))
        OUT = out_dir_for(w["R"], "smoke_j14")
        assert out["smoke"] and out["n_targets"] > 0 and out["controls"] == 2 and out["bit_identical"] == 2, out
        assert os.listdir(f"{OUT}/records") == ["pigtrack0013.json"] and osp.isfile(f"{OUT}/SMOKE_SUMMARY.json")
        assert not any(n.startswith(("manifest_", "SELECTION_", "candidates")) for n in os.listdir(OUT))
        assert not osp.exists(out_of(w)), "a smoke run must not touch the real output folder"
        raises(lambda: main(dict(Wp, smoke_seqs="pigtrack0041", out_name="smoke_j14"), hooks=hooks(w)), WhitelistError)

    def t_whitelist_in_run():
        w = make_world("w6")
        bad = list(w["whitelist"])
        for inj in (w["val4"][0], "pigtrack0002", "pigtrack0001"):
            raises(lambda inj=inj: main(dict(Wp, whitelist=",".join(bad + [inj])), hooks=hooks(w)), WhitelistError)
        raises(lambda: main(dict(Wp, whitelist=",".join(bad[:-1])), hooks=hooks(w)), WhitelistError, "missing")
        # a wrong typed-constant set is refused when the check is on (production)
        raises(lambda: main(Wp, hooks=hooks(w, check_typed_forbidden=True)), RuntimeError, "typed")
        assert not osp.exists(out_of(w)), "a refused whitelist must write nothing"

    def t_pins_refuse():
        w = make_world("w7")
        for key, val, text in (("R_canon_sha256", "0" * 64, "random sets"), ("R2_canon_sha256", "0" * 64, "random sets"),
                               ("eligible_canon_sha256", "0" * 64, "eligible pool"), ("base_manifest_sha256", "0" * 64, "base_manifest_sha256"),
                               ("base_targets_config_sha256", "0" * 64, "base_targets_config_sha256"), ("base_plan_sha256", "0" * 64, "base_plan_sha256"),
                               ("split_sha256", "0" * 64, "split.txt"), ("whitelist_sha256", "0" * 64, "whitelist sha256"),
                               ("student_sha256", "0" * 64, "pinned"), ("teacher_revision", "f" * 40, "teacher revision"),
                               ("n_eligible", 1, "pool counts"), ("M", 2124, "derived")):
            raises(lambda key=key, val=val: main(Wp, hooks=hooks(w, pins=dict(w["pins"], **{key: val}))), (RuntimeError, WhitelistError), text)
        assert not osp.exists(f"{out_of(w)}/seqs")

    try:
        for name, fn in (("params", t_params), ("typed constants", t_typed_constants), ("M / quotas formulas", t_numbers),
                         ("selection: quotas, caps, ties, determinism, random draws, comp", t_selection),
                         ("T1 boundary + diagnostics helpers", t_t1_and_diag), ("controls + whitelist refusals", t_controls_and_whitelist),
                         ("forbidden-name scan", t_scan), ("leak guard self-check", t_guard_works),
                         ("full run (stubs): outputs, manifests, union listings, offline re-derivation, leakage", t_full_run),
                         ("resume no-op", t_resume_noop), ("config mismatch refused", t_config_mismatch),
                         ("crash before the record, resume", t_crash_resume),
                         ("positive control failure publishes nothing", t_control_failure),
                         ("control pre-pass: a failure on the LAST sequence publishes nothing at all", t_control_prepass_nothing_published),
                         ("control second gate (production session) still stops the job; no summary", t_control_session_gate),
                         ("GPU/host cosine cross-check failure publishes nothing", t_crosscheck_failure),
                         ("smoke mode", t_smoke), ("whitelist violations inside a run", t_whitelist_in_run),
                         ("pin mismatches refuse", t_pins_refuse)):
            run(name, fn)
    finally:
        dc.VAL_CELLS = dc_cells
        dc.choose_val_sequences = choose_orig
        shutil.rmtree(td, ignore_errors=True)
    for name, status, detail in results:
        print(f"{status}  {name}")
        if status == "FAIL":
            print(detail)
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"Tol_75 SELFTEST: {len(results) - n_fail}/{len(results)} passed")
    return n_fail == 0


# COMMAND ----------

if SELFTEST and IS_NOTEBOOK:
    raise RuntimeError("the selftest is local-only: TOL75_SELFTEST=1 python Tol_75_pigtrack_mine_targets.py")
if SELFTEST or not IS_NOTEBOOK:
    ok = _selftest() if SELFTEST else True
    if not SELFTEST:
        print("not a notebook and TOL75_SELFTEST != 1: nothing to run")
    raise SystemExit(0 if ok else 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production hooks (GPU): SAM 3 teacher (VERBATIM Tol_66), the current student as scorer, positive-control comparison

# COMMAND ----------

import torch                      # noqa: E402
import torch.nn.functional as F   # noqa: E402
from PIL import Image             # noqa: E402


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


class BackboneFeatureHook:
    """VERBATIM Tol_01: capture backbone output, reshape [1,5184,1024] -> [1,1024,72,72], fp16 on CPU."""
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
    """VERBATIM Tol_01."""
    for oi in range(session.get_obj_num()):
        nc = session.output_dict_per_obj[oi].get("non_cond_frame_outputs", {})
        if len(nc) > keep_last_n:
            for k in sorted(nc.keys())[:-keep_last_n]:
                del nc[k]


class Sam3Teacher:
    def __init__(self, revision, token, device="cuda"):
        import transformers
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        if transformers.__version__ != core.TRANSFORMERS_PIN:
            raise RuntimeError(f"transformers {transformers.__version__} != {core.TRANSFORMERS_PIN}")
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        # Tol_01 logs in first. Without it, the pinned-revision model load reached the Hub unauthenticated (J5 run 1 and
        # the J3e run of 2026-10-14: 401 on resolve/<commit>/config.json although the processor load, same kwargs, worked)
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
        """Tol_01's loop on the selected frames of one sequence: ONE session, dummy boxes on its first frame."""
        session = self.proc.init_video_session(inference_device=self.device, dtype=self.dtype)
        try:
            for si, (fi, path) in enumerate(items):
                with Image.open(path) as im:
                    pil = im.convert("RGB")
                if pil.size != (w, h):
                    raise core.BenchAssertError(f"{path}: {pil.size} != seqinfo {w}x{h}")
                inputs = self.proc(images=pil, device=self.device, return_tensors="pt")
                if si == 0:
                    # fresh lists per session -- the processor MUTATES obj_ids (CLAUDE.md §4)
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


class StudentScorer:
    """The CURRENT student (S-PT, EMA weights, fp32, eval) -> per-position cosine against a teacher feature (fp16 CPU tensor).
    The first CROSSCHECK_N frames are also scored on the host in float64 with tol_distill_core.fidelity_metrics (cos_patch)."""
    def __init__(self, ckpt_path, sha256, device="cuda"):
        self.device = torch.device(device)
        self.model, meta = core.load_student(ckpt_path, device=device, dtype=torch.float32, check_gn=True)
        self.model.eval()
        self.tf = core.make_student_transform()
        self.info = dict(meta, sha256=sha256, run=SPT_RUN)
        self.checks = []            # [(gpu cosine, float64 host cos_patch)] for the first CROSSCHECK_N frames

    def cos(self, path, t_feat):
        with Image.open(path) as im:
            x = self.tf(im.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            s_feat = self.model(x).float()
        t = F.normalize(t_feat.float().to(self.device).flatten(2).squeeze(0), dim=0)
        s = F.normalize(s_feat.flatten(2).squeeze(0), dim=0)
        c = float((t * s).sum(0).mean())
        if len(self.checks) < CROSSCHECK_N:
            host = dc.fidelity_metrics(s_feat.cpu().numpy(), t_feat.float().numpy())["cos_patch"]
            self.checks.append((c, float(host)))
        return c


def compare_target(feat, stored_path):
    """Positive control: the freshly extracted fp16 feature vs the stored Tol_66 target."""
    stored = torch.load(stored_path, map_location="cpu")
    a, b = feat.float().reshape(1, -1), stored.float().reshape(1, -1)
    return {"cos_global": float(F.cosine_similarity(a, b).item()), "max_abs": float((a - b).abs().max()),
            "bit_identical": bool(torch.equal(feat, stored))}


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


def _save_feature(feat, path):
    torch.save(feat, path)


PROD_HOOKS = {
    "local_root": "/local_disk0/tol75/pigtrack_mined_v1",
    "dbutils": dbutils,
    "log": _log,
    "gpu_preflight": gpu_preflight,
    "versions": core.versions,
    "code_shas": code_shas,
    "hf_token": lambda dbu: core.hf_token_from_dbutils(dbu),
    "load_teacher": lambda rev, tok: Sam3Teacher(rev, tok),
    "load_scorer": lambda path, sha: StudentScorer(path, sha),
    "compare_target": compare_target,
    "save_feature": _save_feature,
}

# COMMAND ----------

for _k, _v in WIDGETS.items():
    dbutils.widgets.text(_k, _v)
RAW = {k: dbutils.widgets.get(k) for k in WIDGETS}
print("widgets:", {k: (v if k != "whitelist" else f"<{len(v.split(','))} names>") for k, v in RAW.items()})
RESULT = main(RAW, hooks=PROD_HOOKS)
print(json.dumps(RESULT, indent=1, default=str))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({k: RESULT.get(k) for k in ("smoke", "M", "T1", "manifest_sha256", "set_canon_sha256", "positive_control",
                                                              "n_targets", "controls", "bit_identical", "seconds_total", "extracted_now")},
                                 default=str))
