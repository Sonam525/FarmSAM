# Databricks notebook source
# MAGIC %md
# MAGIC # Mine extra targets for the PigTrack student
# MAGIC
# MAGIC Runs SAM 3 and the PigTrack student on every eligible frame of the 31 training sequences and ranks frames by per-position
# MAGIC feature cosine, hardest first. The `mined` set takes 2,125 frames under a night quota and a per-sequence cap. Random sets
# MAGIC with the same quotas are available as controls. Next step: `distill_pigtrack_mined`.

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

NB_NAME = "mine_targets_pigtrack"
NB_VERSION = "mine_targets_pigtrack-2026-10-24.2"
CONFIG_VERSION = "mine_targets_pigtrack-mining-v1"
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
_PATHS_DIRS = [CODE_DIR, os.environ.get("FARMSAM_PATHS_DIR")]
if not IS_NOTEBOOK:
    _PATHS_DIRS.append(osp.join(osp.dirname(CODE_DIR), "catalog", "build"))
farm_paths = _import_exact("farm_paths", _PATHS_DIRS)
print(f"{NB_NAME} {NB_VERSION}: bench_core {core.BENCH_VERSION} sha256 {core.sha256_file(core.__file__)[:16]}, "
      f"distill_core {dc.DISTILL_CORE_VERSION} sha256 {core.sha256_file(dc.__file__)[:16]}, "
      f"farm_paths sha256 {core.sha256_file(farm_paths.__file__)[:16]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pinned design constants, parameters and the pure-Python functions

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
    "base_manifest_sha256": "6eff15e8350e2b155a2f3a16a7edede7883c95b8a105f340fb1faf38089d8a27",      # extract_teacher_targets_pigtrack manifest.csv
    "base_targets_config_sha256": "1d8bbed385bb1b37512157f0ce45673fa5bbf566c35465c771a1cad811f78335",
    "base_plan_sha256": "f33cb9c50963d14e1d45c03f249bd24d157f1e4b307fdabce0e854a60747a140",
    "split_sha256": "3f75399d240fc79f9371c9edf598b7a9428914d2e266a3cbd4dc41227f4948e5",
    "student_sha256": "c07140f2de80a5e9faf175bdb4fc071bb1064e4757b91c3eefe55e271e5da69c",           # S-PT = pigtrack_v2_dp01 (EMA ep 32)
    "teacher_revision": "3c879f39826c281e95690f02c7821c4de09afae7",
}
SEED_R, SEED_R2, SEED_COMP = 20261024, 20261025, 20261026
CAP_FRAC = 0.79                         # mine_targets: per-clip cap 400 of 288 - 514 candidates per clip = 0.78 - 0.80
RATIO_NUM, RATIO_DEN = 8000, 22934      # calf mining: 8,000 selected of 22,934 candidates
T1_TOPK, T1_THRESHOLD = 6, 0.5          # trigger T1: the six largest per-sequence counts of M sum to MORE than 50 % of M
N_CONTROL_PER_SEQ = 2
CONTROL_COS_GLOBAL_MIN = 0.9999
CROSSCHECK_N, CROSSCHECK_TOL = 50, 1e-4
SET_NAMES = ("mined", "random", "random2", "comp")
SET_SOURCE = {"mined": "mined", "random": "random", "random2": "random2", "comp": "comp"}
MODEL_KEY = "models.student_pigtrack"
SPT_RUN = "pigtrack_v2_dp01"
CKPT_NAME = "best_backbone_student_pigtrack.pt"
BASE_TARGETS_KEY = "paper1.teacher_targets.pigtrack_train"
OUT_NAME_DEFAULT = "pigtrack_mined_v1"
REGISTRY_VERSION = 1
HF_SAM3 = "facebook/sam3"
PRUNE_EVERY_T01, KEEP_LAST_N_T01 = 25, 8        # extract_teacher_targets's session hygiene (the backbone output does not depend on it)
PROMPT_INDEP_MIN_COS = 0.9999                    # same frame, two sessions, two prompt sets -> the same backbone output
SEQ_TOKEN_RX = re.compile(r"pigtrack\d{4}")

WIDGETS = collections.OrderedDict([
    ("whitelist", ",".join(WHITELIST31)), ("out_name", OUT_NAME_DEFAULT), ("smoke_seqs", ""),
    ("sam3_revision", "protocol:hf_revisions.json"), ("require_gpu", "A100"), ("copy_threads", "16"),
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
    """{'val4', 'official_val', 'test'} derived from split.txt and the fixed val rule."""
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
    """The fixed cosine-free draw: per stratum hash order, same caps and quotas as the mined selection."""
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
    """extract_teacher_targets_pigtrack-format manifest rows (role=train, paths relative to the mining folder) for the selected (seq, fi) pairs."""
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
    """Diagnostics D1 and D2 of a selection."""
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
# inputs: PigTrack train discovery, teacher revision (VERBATIM extract_teacher_targets_pigtrack)
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
    if osp.basename(v1_dir) != SPT_RUN:
        raise RuntimeError(f"registry {MODEL_KEY} -> {v1_dir}: expected a '{SPT_RUN}' directory")
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
                  "crosscheck": {"n": CROSSCHECK_N, "tol": CROSSCHECK_TOL, "vs": "distill_core.fidelity_metrics cos_patch (float64, host)"}},
        "control": {"per_seq": N_CONTROL_PER_SEQ, "rule": "base train frames with the smallest sha256('control|seq|fi')",
                    "gate": f"cos_global >= {CONTROL_COS_GLOBAL_MIN} vs the stored extract_teacher_targets_pigtrack target; bit identity expected"},
        "teacher": {"hf_id": HF_SAM3, "revision": rev, "revision_source": rev_src, "dtype": "bfloat16",
                    "transformers": core.TRANSFORMERS_PIN, "attention": "model default (resolved recorded; FA2 refused)",
                    "capture": "forward hook on vision_encoder.backbone; last_hidden_state [1,5184,1024] -> "
                               "permute/reshape [1,1024,72,72] -> .cpu().half() (extract_teacher_targets BackboneFeatureHook, = extract_teacher_targets_pigtrack)",
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
    R = h.get("registry") or farm_paths.load(version=REGISTRY_VERSION)
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
        raise RuntimeError("the recomputed random sets R / R2 differ from the fixed sha256 values")
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
                           k: code[k] for k in ("bench_core", "distill_core", "farm_paths") if k in code})
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
        # short session per sequence, and compared with the stored extract_teacher_targets_pigtrack targets BEFORE the first byte of anything is published. A
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
            if h.get("crash_after_dir_publish") == s:
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
        "note": "T1: is the selection concentrated in few sequences?",
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

if not IS_NOTEBOOK:
    print("This notebook needs a GPU cluster and the path registry: see docs/REPRODUCE.md")
    raise SystemExit(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production hooks (GPU): SAM 3 teacher (VERBATIM extract_teacher_targets_pigtrack), the current student as scorer, positive-control comparison

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
    """VERBATIM extract_teacher_targets: capture backbone output, reshape [1,5184,1024] -> [1,1024,72,72], fp16 on CPU."""
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
    def __init__(self, revision, token, device="cuda"):
        import transformers
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        if transformers.__version__ != core.TRANSFORMERS_PIN:
            raise RuntimeError(f"transformers {transformers.__version__} != {core.TRANSFORMERS_PIN}")
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        # extract_teacher_targets logs in first. Without it, the pinned-revision model load reached the Hub unauthenticated (J5 run 1 and
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


class StudentScorer:
    """The CURRENT student (S-PT, EMA weights, fp32, eval) -> per-position cosine against a teacher feature (fp16 CPU tensor).
    The first CROSSCHECK_N frames are also scored on the host in float64 with distill_core.fidelity_metrics (cos_patch)."""
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
    """Positive control: the freshly extracted fp16 feature vs the stored extract_teacher_targets_pigtrack target."""
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
    return {"notebook": nb, "distill_core": core.sha256_file(dc.__file__),
            "bench_core": core.sha256_file(core.__file__), "farm_paths": core.sha256_file(farm_paths.__file__),
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
