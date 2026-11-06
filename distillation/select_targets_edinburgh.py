# Databricks notebook source
# MAGIC %md
# MAGIC # Select the extra Edinburgh frames
# MAGIC
# MAGIC Picks the 209 frames with the lowest student cosine inside pig cells in each of `2019_11_05_000009`, `2019_11_11_000016`
# MAGIC and `2019_11_15_000033`, and writes `manifest_base.csv` and `manifest_mined.csv` for `distill_edinburgh_mined`.
# MAGIC Runs on CPU. Never reads a test clip.

# COMMAND ----------

import csv
import hashlib
import io
import json
import math
import os
import os.path as osp
import re
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

NB_NAME, NB_VERSION = "select_targets_edinburgh", "select_targets_edinburgh-2026-10-30.1"
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None
if not IS_NOTEBOOK and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- clips, roots ----------------------------------------------------------------------------------------------------------
VOL = "/Volumes/your_catalog/your_schema/your_volume/farmsam"
EM_REL = "paper1_distillation/edi_mining"
UNUSED_REL = f"{EM_REL}/teacher_targets_unused_v1"
STORED_REL = "paper1_distillation/teacher_targets/edinburgh"
FRAMES_REL = "datasets/edinburgh_pigs/decoded_frames"
MEASURE_REL = f"{EM_REL}/measure_v1"
NNT_REL = f"{EM_REL}/nn_teacher_v1"
EM_ROOT = f"{VOL}/{EM_REL}"
EDI_TRAIN = ("2019_11_05_000002", "2019_11_11_000028", "2019_11_11_000036", "2019_11_22_000010")
EDI_VAL = ("2019_11_28_000113",)
P3 = ("2019_11_05_000009", "2019_11_11_000016", "2019_11_15_000033")
_TEST_IDS = ("2019_12_02_000005", "2019_12_02_000208", "2019_12_10_000060", "2019_12_10_000078")
NF = 600
Q = math.floor(8000 / 22934 * NF)                    # 209 per clip (the calf-v2 / b1.6 ratio, applied per clip)
M_TOTAL = 3 * Q                                      # 627
CRIT, FALLBACK, ROUND_DP = "f_txt_fg_cos", "f_cos_mean", 5
HIGH_MOTION = 11.43                                  # training P90 of frame motion, as in the design
STORED_TARGET_BYTES = 10618052                       # E09: 57,337,480,800 B / 5,400 files, all equal
UNUSED_TARGET_BYTES = 10618663
FRAME_WH = (1280, 720)

# ---- values REGISTERED in b1.11 (item 3) or recorded by the analysis stage (item 2) -------------------------------------------
FID_SHA = "515feeda9ebbecfa48268ad317a71db0794c9b18587c93029709ade6d014f3a4"      # measure_edinburgh_hard_frames tables/per_frame_fidelity.csv
MAN_SHA = "3b9f6f59db4bcf6899f9f96968c31efe8925e383ec6aae7bfa8708d9c1ea7f6f"      # unused-target manifest.csv (1,800 rows)
CANON_SHA = "53668699a31324cf009cc8c8aef50309a9413cc0316509ecbb15d1cc552b2702"    # "{clip},{fi}\n", rows by clip then int fi
LEX_WRONG_SHA = "6eac1185df849bdd9995d6ee75f8079b3ae32202d88839f4fb8c65594d4e19c5"  # same lines sorted as STRINGS: known wrong
SEL_FILE_SHA = "11e629fef34a85ad35d5c237887c43f4b8b1531907dd7cecdb2bff7d80e4fa75"  # the 627-row selection file
SAMPLE_REL, SAMPLE_SHA = "2019_11_05_000002/backbone_000000.pt", \
    "9706704cd9fd853b13d775a1f72f7c2e8a2c64dc7057f3ff68d2d8793673c6bc"
MEASURE_RR_SHA = "85c75fb1e9591fcbdbd1981c6d48a4dafe2b372dd2bcc889620a6ee0a7d4f36c"  # measure_edinburgh_hard_frames measure_v1/RUN_RECORD.json
NNT_RR_SHA = "0020829703805bce4873d222d3f1d54246779ed53e39cc9cb36216137ca66282"       # edinburgh_frame_similarity nn_teacher_v1/RUN_RECORD.json
CUT_VALUES = {"2019_11_05_000009": 0.75974, "2019_11_11_000016": 0.74059, "2019_11_15_000033": 0.76228}
# D1 as recorded by the design step (design/v2/P3_M_D1.json, key "D1"); compared, not enforced
REG_D1 = json.loads('{"M_overall":{"adjacent_share":0.8102,"high_motion_frames_ge_11p43":178,"label":"M (627)","mean_crit":0.74583,"mean_f_cos_mean":0.78052,"mean_motion":9.483,"mean_nn_g18_even_frames":0.79406,"mean_txt_max_box_iou":0.42355,"n":627,"n_even_frames_with_nn":317,"stockperson_proxy_frames":72},"canonical_sha256":"53668699a31324cf009cc8c8aef50309a9413cc0316509ecbb15d1cc552b2702","n":627,"per_clip":{"000009":{"M":{"adjacent_share":0.8804,"high_motion_frames_ge_11p43":105,"label":"M","mean_crit":0.75256,"mean_f_cos_mean":0.78275,"mean_motion":11.47,"mean_nn_g18_even_frames":0.79448,"mean_txt_max_box_iou":0.35173,"n":209,"n_even_frames_with_nn":104,"stockperson_proxy_frames":5},"frames_in_three_longest_runs":59,"longest_run_share_of_clip_M":0.148,"n_runs":57,"pool":{"adjacent_share":1.0,"high_motion_frames_ge_11p43":260,"label":"pool","mean_crit":0.76512,"mean_f_cos_mean":0.78648,"mean_motion":11.383,"mean_nn_g18_even_frames":0.80267,"mean_txt_max_box_iou":0.29556,"n":600,"n_even_frames_with_nn":300,"stockperson_proxy_frames":15},"shift_of_M_below_pool_clip_mean_in_sds":1.06,"three_longest_runs":[{"first":285,"last":315,"length":31},{"first":332,"last":346,"length":15},{"first":164,"last":176,"length":13}],"within_clip_sd_of_criterion_over_600_frames":0.01188},"000016":{"M":{"adjacent_share":0.6842,"high_motion_frames_ge_11p43":0,"label":"M","mean_crit":0.73821,"mean_f_cos_mean":0.79591,"mean_motion":6.049,"mean_nn_g18_even_frames":0.81707,"mean_txt_max_box_iou":0.55635,"n":209,"n_even_frames_with_nn":105,"stockperson_proxy_frames":0},"frames_in_three_longest_runs":30,"longest_run_share_of_clip_M":0.062,"n_runs":111,"pool":{"adjacent_share":1.0,"high_motion_frames_ge_11p43":0,"label":"pool","mean_crit":0.74202,"mean_f_cos_mean":0.79679,"mean_motion":6.008,"mean_nn_g18_even_frames":0.81764,"mean_txt_max_box_iou":0.55353,"n":600,"n_even_frames_with_nn":300,"stockperson_proxy_frames":0},"shift_of_M_below_pool_clip_mean_in_sds":1.08,"three_longest_runs":[{"first":422,"last":434,"length":13},{"first":517,"last":525,"length":9},{"first":567,"last":574,"length":8}],"within_clip_sd_of_criterion_over_600_frames":0.00352},"000033":{"M":{"adjacent_share":0.866,"high_motion_frames_ge_11p43":73,"label":"M","mean_crit":0.74672,"mean_f_cos_mean":0.76291,"mean_motion":10.921,"mean_nn_g18_even_frames":0.77129,"mean_txt_max_box_iou":0.36257,"n":209,"n_even_frames_with_nn":108,"stockperson_proxy_frames":67},"frames_in_three_longest_runs":147,"longest_run_share_of_clip_M":0.359,"n_runs":44,"pool":{"adjacent_share":1.0,"high_motion_frames_ge_11p43":127,"label":"pool","mean_crit":0.76193,"mean_f_cos_mean":0.76816,"mean_motion":10.106,"mean_nn_g18_even_frames":0.77681,"mean_txt_max_box_iou":0.28516,"n":600,"n_even_frames_with_nn":300,"stockperson_proxy_frames":69},"shift_of_M_below_pool_clip_mean_in_sds":1.1,"three_longest_runs":[{"first":188,"last":262,"length":75},{"first":140,"last":186,"length":47},{"first":0,"last":24,"length":25}],"within_clip_sd_of_criterion_over_600_frames":0.01382}},"pool_overall":{"adjacent_share":1.0,"high_motion_frames_ge_11p43":387,"label":"pool (1,800)","mean_crit":0.75636,"mean_f_cos_mean":0.78381,"mean_motion":9.166,"mean_nn_g18_even_frames":0.79904,"mean_txt_max_box_iou":0.37808,"n":1800,"n_even_frames_with_nn":900,"stockperson_proxy_frames":84},"pooled_ranking_of_same_size":{"000009":27,"000016":512,"000033":88},"q_per_clip":209}')

OUT_NAMES = ("SELECTION_mined.csv", "manifest_base.csv", "manifest_mined.csv", "SELECTION_SUMMARY.json")
SEL_COLS = ["clip", "fi", "image", "image_sha256", "feature", "feature_sha256"]
MAN_COLS = ["clip", "role", "fi", "source", "image", "image_bytes", "image_sha256", "feature", "feature_bytes", "feature_sha256"]
EXISTING_EM_DIRS = ("measure_v1", "nn_teacher_v1", "smoke_measure_v1", "smoke_teacher_targets_unused_v1",
                    "teacher_targets_unused_v1")


class LeakError(RuntimeError):
    pass


def _is_test(s):
    s = str(s)
    return any(t in s for t in _TEST_IDS) or "2019_12_" in s


def guard_path(p):
    if _is_test(p):
        raise LeakError("refused: a test-clip path")
    return p


def no_test(names, what="input"):
    bad = sorted({str(n) for n in names if _is_test(n)})
    if bad:
        raise LeakError(f"refused: a test clip id in the {what}")
    return names


def assert_nov_clips(names, what):
    """Every training row's clip id must start with 2019_11_ (the test clips start 2019_12_)."""
    bad = sorted({str(n) for n in names if not str(n).startswith("2019_11_")})
    if bad:
        raise LeakError(f"{what}: clip id(s) not starting with 2019_11_: {bad}")


def scan_bytes(blob, label):
    if any(t.encode() in blob for t in _TEST_IDS) or b"2019_12_" in blob:
        raise LeakError(f"{label} contains a test id")


if (set(EDI_TRAIN) | set(EDI_VAL) | set(P3)) & set(_TEST_IDS):
    raise LeakError("a test clip is among the clips")
assert_nov_clips(EDI_TRAIN + EDI_VAL + P3, "constants")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# COMMAND ----------

# ---- selection (pure; numpy + pandas) ---------------------------------------------------------------------------------------
def select_m(f, clips, q):
    """The q lowest criterion values per clip. Key = CRIT (fallback FALLBACK when missing) rounded to 5 decimals, ascending,
    ties by frame index. Returns (rows [(clip, fi)] ordered by clip then INTEGER fi, per-clip info)."""
    rows, info = [], {}
    for c in clips:
        g = f[f["clip"] == c]
        fis = g["fi"].to_numpy().astype(np.int64)
        if len(g) != NF or sorted(fis.tolist()) != list(range(NF)):
            raise RuntimeError(f"{c}: {len(g)} rows / frame indices are not 0..{NF - 1}")
        crit = g[CRIT].astype(float)
        n_nan = int(crit.isna().sum())
        key = crit.where(crit.notna(), g[FALLBACK].astype(float)).round(ROUND_DP).to_numpy()
        if not np.isfinite(key).all():
            raise RuntimeError(f"{c}: a frame has neither {CRIT} nor {FALLBACK}")
        order = np.lexsort((fis, key))                       # primary: key, secondary: frame index
        take = [int(fis[i]) for i in order[:q]]
        ref = sorted(range(len(g)), key=lambda i: (float(key[i]), int(fis[i])))[:q]     # independent implementation
        if sorted(int(fis[i]) for i in ref) != sorted(take):
            raise RuntimeError(f"{c}: the two selection implementations disagree")
        cut = float(key[order[q - 1]])
        tied = sorted(int(x) for x in fis[key == cut])
        info[c] = {"n_frames": int(len(g)), "criterion_nan_frames": n_nan, "n_selected": len(take), "cut_value_209th": cut,
                   "frames_sharing_cut_value": tied, "tied_taken": sorted(set(tied) & set(take)),
                   "tied_left_out": sorted(set(tied) - set(take)),
                   "min_key": float(key.min()), "max_selected_key": float(key[order[q - 1]]),
                   "min_unselected_key": float(key[order[q]]) if len(g) > q else None}
        rows += [(c, fi) for fi in sorted(take)]
    rows.sort(key=lambda r: (r[0], int(r[1])))
    return rows, info


def canonical_text(rows):
    """'{clip},{fi}\\n' per row; the ORDER is part of the definition: by clip, then by integer fi."""
    rs = sorted(rows, key=lambda r: (r[0], int(r[1])))
    return "".join(f"{c},{int(fi)}\n" for c, fi in rs)


def canonical_lex_text(rows):
    """The known-WRONG variant: the same lines sorted as strings (',10' before ',2')."""
    return "".join(sorted(f"{c},{int(fi)}\n" for c, fi in rows))


def check_digest(d, expected, wrong):
    if d == wrong:
        raise AssertionError("canonical digest equals the known-wrong lexicographic digest")
    if d != expected:
        raise AssertionError(f"canonical-list sha256 {d} != registered {expected}")
    return d


def check_canonical(rows, expected, wrong):
    return check_digest(sha_text(canonical_text(rows)), expected, wrong)


def write_csv_lf(path, cols, records):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in records:
            w.writerow([r[c] for c in cols])


def selection_records(rows, man):
    """rows -> records with the manifest columns (clip, fi, image, image_sha256, feature, feature_sha256), validated 1:1."""
    idx = {}
    for r in man.itertuples(index=False):
        k = (r.seq, int(r.fi))
        if k in idx:
            raise RuntimeError(f"duplicate manifest row {k}")
        idx[k] = r
    recs = []
    for c, fi in rows:
        r = idx.get((c, int(fi)))
        if r is None or not isinstance(r.feature_sha256, str) or len(r.feature_sha256) != 64:
            raise RuntimeError(f"no manifest row / feature sha for {(c, fi)}")
        recs.append({"clip": c, "fi": int(fi), "image": r.image, "image_sha256": r.image_sha256, "feature": r.feature,
                     "feature_sha256": r.feature_sha256, "_image_bytes": int(r.image_bytes), "_feature_bytes": int(r.feature_bytes)})
    return recs


# ---- diagnostics D1 (port of design/v2_p3_rederive_d1.py; read only) ---------------------------------------------------------
def runs_of(fis):
    fis = sorted(fis)
    out, st, pr = [], fis[0], fis[0]
    for x in fis[1:]:
        if x != pr + 1:
            out.append((st, pr))
            st = x
        pr = x
    out.append((st, pr))
    return out


def stock_flags(left_by_clip):
    """HARDCASES 3.3(a) proxy: mean cosine of the left grid strip (x < 13 cells) more than 0.025 below the clip median."""
    return {c: (v < float(np.median(v)) - 0.025) for c, v in left_by_clip.items()}


def nn_g18_pool(index, sim, pool, train):
    """Nearest-training-frame similarity (g18) of every pool frame in the edinburgh_frame_similarity index (every 2nd frame), as hc_nn.py."""
    clip = index["clip"].astype(str).to_numpy()
    fi = index["fi"].to_numpy()
    tr_rows = np.concatenate([np.nonzero(clip == c)[0] for c in train])
    out = []
    for c in pool:
        r = np.nonzero(clip == c)[0]
        nn = sim[np.ix_(r, tr_rows)].max(1)
        out += [{"clip": c, "fi": int(fi[r[j]]), "nn_g18": float(nn[j])} for j in range(len(r))]
    return pd.DataFrame(out)


def d1_report(f, rows, flags, nn, q):
    sel = pd.DataFrame(rows, columns=["clip", "fi"])
    keys = set(zip(sel["clip"], sel["fi"]))
    j = sel.merge(f, on=["clip", "fi"], how="left")
    pool = f[f["clip"].isin(sorted({c for c, _ in rows}))]
    pool_keys = set(zip(pool["clip"], pool["fi"]))

    def block(sub_keys, jj, label):
        adj = sum(((c, fi - 1) in sub_keys) or ((c, fi + 1) in sub_keys) for c, fi in sub_keys)
        hm = jj.merge(nn[["clip", "fi", "nn_g18"]], on=["clip", "fi"], how="inner")
        return {
            "n": len(sub_keys), "mean_crit": round(float(jj[CRIT].mean()), 5),
            "mean_f_cos_mean": round(float(jj["f_cos_mean"].mean()), 5),
            "mean_motion": round(float(jj["motion"].mean()), 3),
            "stockperson_proxy_frames": int(sum(bool(flags[c][fi]) for c, fi in sub_keys)),
            "high_motion_frames_ge_11p43": int((jj["motion"] >= HIGH_MOTION).sum()),
            "mean_txt_max_box_iou": round(float(jj["txt_max_box_iou"].mean()), 5),
            "mean_nn_g18_even_frames": round(float(hm["nn_g18"].mean()), 5), "n_even_frames_with_nn": int(len(hm)),
            "adjacent_share": round(adj / len(sub_keys), 4), "label": label}

    d1 = {"n": len(rows), "canonical_sha256": sha_text(canonical_text(rows)), "q_per_clip": q,
          "M_overall": block(keys, j, f"M ({len(rows)})"), "pool_overall": block(pool_keys, pool, f"pool ({len(pool):,})"), "per_clip": {}}
    for c in sorted({c for c, _ in rows}):
        k_c = {k for k in keys if k[0] == c}
        pk_c = {k for k in pool_keys if k[0] == c}
        sd = float(f[f["clip"] == c][CRIT].std())
        mc = block(k_c, j[j["clip"] == c], "M")
        pc = block(pk_c, pool[pool["clip"] == c], "pool")
        rs = runs_of([fi for _, fi in k_c])
        top3 = sorted(rs, key=lambda r: (r[0] - r[1], r[0]))[:3]
        d1["per_clip"][c[-6:]] = {
            "M": mc, "pool": pc, "within_clip_sd_of_criterion_over_600_frames": round(sd, 5),
            "shift_of_M_below_pool_clip_mean_in_sds": round((pc["mean_crit"] - mc["mean_crit"]) / sd, 2),
            "n_runs": len(rs),
            "three_longest_runs": [{"first": a, "last": b, "length": b - a + 1} for a, b in top3],
            "frames_in_three_longest_runs": int(sum(b - a + 1 for a, b in top3)),
            "longest_run_share_of_clip_M": round(max(b - a + 1 for a, b in rs) / len(k_c), 3)}
    pick = pool.assign(_k=pool[CRIT].round(ROUND_DP)).sort_values(["_k", "clip", "fi"], kind="mergesort").head(len(rows))
    d1["pooled_ranking_of_same_size"] = {c[-6:]: int(n) for c, n in pick.groupby("clip").size().items()}
    return d1


def diff_tree(a, b, path=""):
    """Paths where two JSON-like trees differ (floats compared to 1e-9)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing")
            else:
                out += diff_tree(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff_tree(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (abs(float(a) - float(b)) <= 1e-9):
            out.append(f"{path}: {a} != {b}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


# COMMAND ----------

# ---- file layer: hashing, manifests, publishing ---------------------------------------------------------------------------
def read_retry(path, tries=4):
    guard_path(path)
    for attempt in range(tries):
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(5)


def hash_stream(path, tries=4):
    """(bytes, sha256, first 4 bytes) of a big file without holding it in memory."""
    guard_path(path)
    for attempt in range(tries):
        try:
            h, n, head = hashlib.sha256(), 0, b""
            with open(path, "rb") as fh:
                while True:
                    b = fh.read(1 << 23)
                    if not b:
                        break
                    if n == 0:
                        head = b[:4]
                    h.update(b)
                    n += len(b)
            return n, h.hexdigest(), head
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(5)


def hash_job(job):
    """job = (kind, abs path, expected bytes or None, expected sha or None). Returns (path, bytes, sha, extra)."""
    kind, p, eb, es = job
    if kind == "feature":
        n, sha, head = hash_stream(p)
        if head != b"PK\x03\x04":
            raise IOError(f"{osp.basename(p)}: not a torch zip archive")
        extra = None
    else:
        raw = read_retry(p)
        n, sha = len(raw), sha_bytes(raw)
        if raw[:2] != b"\xff\xd8":
            raise IOError(f"{osp.basename(p)}: not a JPEG")
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as im:
            extra = tuple(im.size)
        if extra != FRAME_WH:
            raise IOError(f"{osp.basename(p)}: image size {extra}, expected {FRAME_WH}")
    if eb is not None and n != eb:
        raise IOError(f"{osp.basename(p)}: {n} B, expected {eb}")
    if es is not None and sha != es:
        raise IOError(f"{osp.basename(p)}: sha256 {sha[:12]} differs from the manifest value {es[:12]}")
    return p, n, sha, extra


def hash_all(jobs, threads, label, progress=500):
    out, t0, done = {}, time.time(), 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for p, n, sha, extra in ex.map(hash_job, jobs):
            out[p] = (n, sha, extra)
            done += 1
            if done % progress == 0:
                print(f"{label}: {done}/{len(jobs)} files, {time.time() - t0:.1f} s", flush=True)
    return out, time.time() - t0


def base_specs():
    """(clip, role, fi, image rel, feature rel) for the stored 2026-02 targets: 2,400 train + 600 val rows."""
    specs = []
    for role, clips in (("train", EDI_TRAIN), ("val", EDI_VAL)):
        for c in clips:
            for fi in range(NF):
                specs.append((c, role, fi, f"{FRAMES_REL}/{c}/video1/{fi + 1:07d}.jpg", f"{STORED_REL}/{c}/backbone_{fi:06d}.pt"))
    return specs


def assert_disjoint_counts(base, extra):
    """(e): counts 2,400 / 600 / 627, every clip id 2019_11_*, no (clip, fi) overlap between train, val and extra rows."""
    tr = {(r["clip"], int(r["fi"])) for r in base if r["role"] == "train"}
    va = {(r["clip"], int(r["fi"])) for r in base if r["role"] == "val"}
    ex = {(r["clip"], int(r["fi"])) for r in extra}
    counts = (len(tr), len(va), len(ex))
    if counts != (4 * NF, NF, M_TOTAL) or len(base) != 5 * NF or len(extra) != M_TOTAL:
        raise AssertionError(f"counts {counts} / rows {len(base)} {len(extra)}, expected ({4 * NF}, {NF}, {M_TOTAL})")
    if tr & va or tr & ex or va & ex:
        raise AssertionError("a (clip, fi) pair occurs in two of base / val / extra")
    for names, what in (({k[0] for k in tr}, "base"), ({k[0] for k in va}, "val"), ({k[0] for k in ex}, "extra")):
        assert_nov_clips(names, what)
        no_test(names, what)
    if {k[0] for k in tr} != set(EDI_TRAIN) or {k[0] for k in va} != set(EDI_VAL) or {k[0] for k in ex} != set(P3):
        raise AssertionError("clip sets differ from EDI_TRAIN / EDI_VAL / P3")
    return counts


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_out_tag(tag):
    if not re.fullmatch(r"[A-Za-z0-9_]+", tag or "") or tag in EXISTING_EM_DIRS:
        raise ValueError(f"bad out_tag {tag!r}")
    return tag


# COMMAND ----------



def _refuse_existing(p):
    if osp.exists(p):
        raise RuntimeError(f"{p} exists: this notebook writes only NEW folders")


if not IS_NOTEBOOK:
    print("This notebook needs a GPU cluster and the path registry: see docs/REPRODUCE.md")
    raise SystemExit(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run: pinned inputs, selection, manifests, diagnostics, publish

# COMMAND ----------

T0 = time.time()
ERRORS = []


def _w(name, default):
    try:
        dbutils.widgets.text(name, default)  # noqa: F821
        return dbutils.widgets.get(name).strip()  # noqa: F821
    except NameError:
        return os.environ.get("TOL81_" + name.upper(), default)


OUT_TAG = check_out_tag(_w("out_tag", "selection_v1"))
THREADS = int(_w("threads", "32"))
if not (1 <= THREADS <= 64):
    raise ValueError("threads")
OUT = guard_path(f"{EM_ROOT}/{OUT_TAG}")
if osp.exists(OUT):
    raise RuntimeError(f"{OUT} exists: this notebook writes only NEW folders")
LOCAL = f"/local_disk0/tol81/{OUT_TAG}"
shutil.rmtree(LOCAL, ignore_errors=True)
os.makedirs(LOCAL, exist_ok=True)
print(NB_NAME, NB_VERSION, "out", OUT, "threads", THREADS, "cpus", os.cpu_count(), "pandas", pd.__version__, "numpy", np.__version__,
      "runtime", os.environ.get("DATABRICKS_RUNTIME_VERSION"))

# COMMAND ----------

# ---- (a) the two pinned inputs and the selection ------------------------------------------------------------------------------
def pinned(rel, want, label):
    raw = read_retry(guard_path(f"{VOL}/{rel}"))
    got = sha_bytes(raw)
    if got != want:
        raise AssertionError(f"{label}: sha256 {got} != pinned {want}")
    print(f"pinned {label}: {len(raw):,} B sha256 {got[:12]}...")
    return raw


fid_raw = pinned(f"{MEASURE_REL}/tables/per_frame_fidelity.csv", FID_SHA, "per_frame_fidelity.csv")
man_raw = pinned(f"{UNUSED_REL}/manifest.csv", MAN_SHA, "unused-target manifest.csv")
mrr_raw = pinned(f"{MEASURE_REL}/RUN_RECORD.json", MEASURE_RR_SHA, "measure_edinburgh_hard_frames RUN_RECORD.json")
nrr_raw = pinned(f"{NNT_REL}/RUN_RECORD.json", NNT_RR_SHA, "edinburgh_frame_similarity RUN_RECORD.json")
MRR, NRR = json.loads(mrr_raw), json.loads(nrr_raw)

F = pd.read_csv(io.BytesIO(fid_raw)).copy()          # .copy(): defragment the 140-column frame
MAN = pd.read_csv(io.BytesIO(man_raw))
no_test(F["clip"].unique(), "per-frame table")
no_test(MAN["seq"].unique(), "unused-target manifest")
assert_nov_clips(F["clip"].unique(), "per-frame table")
assert_nov_clips(MAN["seq"].unique(), "unused-target manifest")
if set(F["clip"].unique()) != set(EDI_TRAIN + EDI_VAL + P3) or len(F) != 8 * NF:
    raise RuntimeError("per-frame table does not hold exactly the 8 non-test clips x 600 frames")
if set(MAN["seq"].unique()) != set(P3) or len(MAN) != 3 * NF:
    raise RuntimeError("manifest does not hold exactly the 3 pool clips x 600 frames")

M_ROWS, SEL_INFO = select_m(F, P3, Q)
if len(M_ROWS) != M_TOTAL:
    raise AssertionError(f"{len(M_ROWS)} selected, expected {M_TOTAL}")
CANON = check_canonical(M_ROWS, CANON_SHA, LEX_WRONG_SHA)
LEX_DIGEST = sha_text(canonical_lex_text(M_ROWS))
if LEX_DIGEST != LEX_WRONG_SHA:
    raise AssertionError("the lexicographic variant no longer gives the documented known-wrong digest")
for c in P3:
    if abs(SEL_INFO[c]["cut_value_209th"] - CUT_VALUES[c]) > 1e-12 or SEL_INFO[c]["criterion_nan_frames"] != 0:
        raise AssertionError(f"{c}: cut value / missing criterion differ from the registered values: {SEL_INFO[c]}")
SEL_RECS = selection_records(M_ROWS, MAN)
write_csv_lf(osp.join(LOCAL, "SELECTION_mined.csv"), SEL_COLS, SEL_RECS)
SEL_FILE = sha_file(osp.join(LOCAL, "SELECTION_mined.csv"))
if SEL_FILE != SEL_FILE_SHA:
    raise AssertionError(f"selection file sha256 {SEL_FILE} != registered {SEL_FILE_SHA}")
print("selection OK: 627 rows, canonical", CANON[:12], "file", SEL_FILE[:12], "cuts",
      {c[-6:]: SEL_INFO[c]["cut_value_209th"] for c in P3})

# COMMAND ----------

# ---- (b) manifest_base.csv: every stored target and decoded frame, read once --------------------------------------------------
BASE_SPECS = base_specs()
jobs = []
for c, role, fi, img, ftr in BASE_SPECS:
    jobs.append(("feature", guard_path(f"{VOL}/{ftr}"), STORED_TARGET_BYTES, None))
    jobs.append(("image", guard_path(f"{VOL}/{img}"), None, None))
print("base files to read:", len(jobs))
H_BASE, T_HASH_BASE = hash_all(jobs, THREADS, "base")
BASE_BYTES = sum(v[0] for v in H_BASE.values())
print(f"base: {len(H_BASE)} files, {BASE_BYTES / 1e9:.2f} GB in {T_HASH_BASE:.1f} s = {BASE_BYTES / 1e6 / max(T_HASH_BASE, 1e-9):.0f} MB/s")
BASE_RECS = []
for c, role, fi, img, ftr in BASE_SPECS:
    hi, hf = H_BASE[f"{VOL}/{img}"], H_BASE[f"{VOL}/{ftr}"]
    BASE_RECS.append({"clip": c, "role": role, "fi": fi, "source": "stored_2026_02", "image": img, "image_bytes": hi[0],
                      "image_sha256": hi[1], "feature": ftr, "feature_bytes": hf[0], "feature_sha256": hf[1]})
sample_rec = next(r for r in BASE_RECS if r["clip"] + "/backbone_%06d.pt" % r["fi"] == SAMPLE_REL)
if sample_rec["feature_sha256"] != SAMPLE_SHA:
    raise AssertionError(f"sample {SAMPLE_REL}: sha256 {sample_rec['feature_sha256']} != recorded {SAMPLE_SHA}")
print("sample", SAMPLE_REL, "= 9706704c... OK")
if len({r["feature_sha256"] for r in BASE_RECS}) < 0.99 * len(BASE_RECS):
    raise RuntimeError("target files are largely identical to each other")

# COMMAND ----------

# ---- (c) manifest_mined.csv: the 627 rows of M, re-hashed against the measure_edinburgh_hard_frames manifest -------------------------------------------
jobs = []
for r in SEL_RECS:
    jobs.append(("feature", guard_path(f"{VOL}/{UNUSED_REL}/{r['feature']}"), r["_feature_bytes"], r["feature_sha256"]))
    jobs.append(("image", guard_path(f"{VOL}/{UNUSED_REL}/{r['image']}"), r["_image_bytes"], r["image_sha256"]))
H_MINED, T_HASH_MINED = hash_all(jobs, THREADS, "mined")
MINED_BYTES = sum(v[0] for v in H_MINED.values())
print(f"mined: {len(H_MINED)} files, {MINED_BYTES / 1e9:.2f} GB in {T_HASH_MINED:.1f} s; all equal the measure_edinburgh_hard_frames manifest values")
MINED_RECS = []
for r in SEL_RECS:
    MINED_RECS.append({"clip": r["clip"], "role": "extra", "fi": r["fi"], "source": "tol79_2026_10",
                       "image": f"{UNUSED_REL}/{r['image']}", "image_bytes": r["_image_bytes"], "image_sha256": r["image_sha256"],
                       "feature": f"{UNUSED_REL}/{r['feature']}", "feature_bytes": r["_feature_bytes"],
                       "feature_sha256": r["feature_sha256"]})
COUNTS = assert_disjoint_counts(BASE_RECS, MINED_RECS)
print("counts (train, val, extra):", COUNTS, "- disjoint, all clip ids 2019_11_*")
write_csv_lf(osp.join(LOCAL, "manifest_base.csv"), MAN_COLS, BASE_RECS)
write_csv_lf(osp.join(LOCAL, "manifest_mined.csv"), MAN_COLS, MINED_RECS)

# COMMAND ----------

# ---- (d) diagnostics D1 and SELECTION_SUMMARY.json -----------------------------------------------------------------------------
left_by_clip, maps_sha = {}, {}
for c in P3:
    rec = MRR["published"][f"clips/{c}/maps.npz"]
    raw = pinned(f"{MEASURE_REL}/clips/{c}/maps.npz", rec["sha256"], f"maps.npz {c}")
    m = np.load(io.BytesIO(raw))["cos_fresh"].astype(np.float32)
    if m.shape != (NF, 72, 72):
        raise RuntimeError(f"{c}: maps {m.shape}")
    left_by_clip[c] = m[:, :, :13].mean((1, 2))
    maps_sha[c] = rec["sha256"]
idx_raw = pinned(f"{NNT_REL}/index.csv", NRR["files"]["index.csv"]["sha256"], "nn index.csv")
sim_raw = pinned(f"{NNT_REL}/sim_g18.npy", NRR["files"]["sim_g18.npy"]["sha256"], "nn sim_g18.npy")
NN_INDEX = pd.read_csv(io.BytesIO(idx_raw))
no_test(NN_INDEX["clip"].unique(), "nn index")
NN_DF = nn_g18_pool(NN_INDEX, np.load(io.BytesIO(sim_raw)), P3, EDI_TRAIN)
D1 = d1_report(F, M_ROWS, stock_flags(left_by_clip), NN_DF, Q)
D1_DIFF = diff_tree(D1, REG_D1)
print("D1 vs the registered D1:", "identical" if not D1_DIFF else f"{len(D1_DIFF)} difference(s): {D1_DIFF[:8]}")
SUMMARY = {
    "notebook": NB_NAME, "version": NB_VERSION, "pool": "P3", "pool_clips": list(P3), "q_per_clip": Q, "M": M_TOTAL,
    "criterion": CRIT, "criterion_fallback": FALLBACK, "rounding_decimals": ROUND_DP, "rule": "lowest first within each clip, ties by frame index",
    "canonical_list": {"rule": "'{clip},{fi}\\n' per row, rows ordered by clip ascending then INTEGER fi ascending", "sha256": CANON,
                       "registered_sha256": CANON_SHA, "lexicographic_variant_sha256_known_wrong": LEX_DIGEST,
                       "lexicographic_variant_refused": LEX_DIGEST != CANON},
    "selection_file": {"name": "SELECTION_mined.csv", "sha256": SEL_FILE, "registered_sha256": SEL_FILE_SHA, "rows": M_TOTAL,
                       "columns": SEL_COLS, "paths_relative_to": "teacher_targets_unused_v1/"},
    "per_clip": SEL_INFO, "counts": {"train": COUNTS[0], "val": COUNTS[1], "extra": COUNTS[2]},
    "training_total": COUNTS[0] + COUNTS[2], "D1": D1, "D1_vs_registered": {"identical": not D1_DIFF, "differences": D1_DIFF},
    "inputs": {"per_frame_fidelity.csv": FID_SHA, "unused_manifest.csv": MAN_SHA, "tol79_run_record": MEASURE_RR_SHA,
               "tol80_run_record": NNT_RR_SHA, "maps_npz": maps_sha,
               "nn_index_csv": NRR["files"]["index.csv"]["sha256"], "nn_sim_g18_npy": NRR["files"]["sim_g18.npy"]["sha256"]},
    "diagnostic_notes": "The stockperson proxy shares its signal with the criterion.",
}
with open(osp.join(LOCAL, "SELECTION_SUMMARY.json"), "w", encoding="utf-8", newline="\n") as fh:
    json.dump(SUMMARY, fh, indent=1, sort_keys=False)

# COMMAND ----------

# ---- scan, publish (never overwrite), RUN_RECORD last --------------------------------------------------------------------------
FILES = {}
for name in OUT_NAMES:
    p = osp.join(LOCAL, name)
    with open(p, "rb") as fh:
        blob = fh.read()
    scan_bytes(blob, name)
    FILES[name] = {"sha256": sha_bytes(blob), "bytes": len(blob)}
    if name.endswith(".csv"):
        n_lines = blob.count(b"\n")
        FILES[name]["rows"] = n_lines - 1
        if b"\r" in blob:
            raise RuntimeError(f"{name}: CR in a CSV")
for name, rows_expected in (("SELECTION_mined.csv", M_TOTAL), ("manifest_base.csv", 5 * NF), ("manifest_mined.csv", M_TOTAL)):
    if FILES[name]["rows"] != rows_expected:
        raise AssertionError(f"{name}: {FILES[name]['rows']} rows, expected {rows_expected}")
if FILES["SELECTION_mined.csv"]["sha256"] != SEL_FILE_SHA:
    raise AssertionError("selection file changed between check and publish")


def publish(name):
    src, dst = osp.join(LOCAL, name), guard_path(f"{OUT}/{name}")
    if osp.exists(dst):
        raise FileExistsError(dst)
    want = sha_file(src)
    for attempt in range(4):
        try:
            dbutils.fs.cp("file:" + src, "dbfs:" + dst)  # noqa: F821
            if osp.getsize(dst) == osp.getsize(src) and sha_file(dst) == want:
                return want
        except Exception as e:  # noqa: BLE001
            print("publish retry", name, repr(e)[:200])
        time.sleep(5)
    raise IOError(f"publish {name} failed")


os.makedirs(OUT, exist_ok=False)
t_pub = time.time()
for name in OUT_NAMES:
    got = publish(name)
    if got != FILES[name]["sha256"]:
        raise AssertionError(f"{name}: published sha256 differs")
T_PUB = time.time() - t_pub

REC = {
    "notebook": NB_NAME, "version": NB_VERSION, "out": OUT, "volume_root": VOL,
    "path_note": "paths in manifest_base.csv and manifest_mined.csv are relative to volume_root; SELECTION_mined.csv paths are relative to teacher_targets_unused_v1/",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "files": FILES, "canonical_sha256": CANON, "selection_file_sha256": SEL_FILE, "counts": SUMMARY["counts"],
    "training_total": SUMMARY["training_total"], "D1_identical_to_registered": not D1_DIFF, "D1_differences": D1_DIFF,
    "hashing": {"base_files": len(H_BASE), "base_bytes": BASE_BYTES, "base_seconds": round(T_HASH_BASE, 1),
                "mined_files": len(H_MINED), "mined_bytes": MINED_BYTES, "mined_seconds": round(T_HASH_MINED, 1),
                "threads": THREADS, "sample": {SAMPLE_REL: SAMPLE_SHA}},
    "controls": {"stored_target_bytes_all_equal": STORED_TARGET_BYTES, "frame_size": list(FRAME_WH),
                 "mined_files_equal_tol79_manifest": True, "sample_sha256_equal_recorded": True,
                 "cut_values": {c: SEL_INFO[c]["cut_value_209th"] for c in P3}},
    "timings_s": {"publish": round(T_PUB, 1), "total": round(time.time() - T0, 1)},
    "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                 "runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION")},
    "test_ids": {"count": len(_TEST_IDS), "rule": "every path refused; every input name checked; every output scanned; every training clip id starts 2019_11_",
                 "sha256_of_sorted_ids": sha_text("\n".join(sorted(_TEST_IDS)))},
}
rr_blob = json.dumps(REC, indent=1).encode("utf-8")
scan_bytes(rr_blob, "RUN_RECORD.json")
with open(osp.join(LOCAL, "RUN_RECORD.json"), "wb") as fh:
    fh.write(rr_blob)
publish("RUN_RECORD.json")
print("done", REC["timings_s"], "RUN_RECORD sha256", sha_bytes(rr_blob))
dbutils.notebook.exit(json.dumps({"ok": True, "out": OUT, "canonical_sha256": CANON, "selection_file_sha256": SEL_FILE,  # noqa: F821
                                  "manifest_base_sha256": FILES["manifest_base.csv"]["sha256"],
                                  "manifest_mined_sha256": FILES["manifest_mined.csv"]["sha256"],
                                  "summary_sha256": FILES["SELECTION_SUMMARY.json"]["sha256"],
                                  "run_record_sha256": sha_bytes(rr_blob), "D1_identical": not D1_DIFF,
                                  "timings_s": REC["timings_s"]}))
