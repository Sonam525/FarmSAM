#!/usr/bin/env python
"""distill_core: torch-free helpers shared by the distillation notebooks.

Holds PigTrack split and seqinfo parsing, the frame selection, the rule that holds out 4 whole train sequences for checkpoint
selection, the target manifest, no-overwrite publishing and the feature fidelity metrics. It needs only Python and numpy, so
`python test_distill_core.py` runs on a laptop.
"""
from __future__ import annotations

import configparser
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import bench_core as core

DISTILL_CORE_VERSION = "dc1.0 2026-10-14"
N_TARGETS_DEFAULT = 3000                  # PigTrack train split, about 3,000 unlabeled frames
EXPECT_TRAIN_SEQS = 35                    # official split.txt: train 35 / val 20 / test 25
N_VAL_SEQS = 4                            # task J5: 4 whole train sequences = the distillation val set
SEQ_RX = re.compile(r"^pigtrack\d{4}$")
SPLITS = ("train", "val", "test")
TARGET_SHAPE = (1, 1024, 72, 72)          # SAM 3 vision_encoder.backbone at 1008^2 input (72x72 tokens), fp16 on disk
TARGET_DTYPE = "float16"
# extract_teacher_targets prompt formality, scaled to the frame: backbone features do not depend on prompts (checked at run time)
DUMMY_BOX_FRACS = ((0.25, 0.15, 0.45, 0.45), (0.55, 0.15, 0.75, 0.45),
                   (0.25, 0.55, 0.45, 0.85), (0.55, 0.55, 0.75, 0.85))
# Val rule cells, in pick order: (night flag, (width, height)). Night 1920x1080 holds only 2 train sequences (560 and 80
# frames) and is not held out (holding one out would remove most of that cell from training).
VAL_CELLS = ((0, (1280, 800)), (0, (1920, 1080)), (1, (1280, 800)), (1, (1280, 800)))
VAL_RULE = ("4 whole PigTrack train sequences, one per cell in this order: (day, 1280x800), (day, 1920x1080), "
            "(night, 1280x800), (night, 1280x800). Within a cell: the sequence whose IMAGE COUNT is closest to the median "
            "image count of the cell (statistics.median; ties -> lower sequence id), skipping any sequence whose pen is "
            "already held out (pens of the 4 val sequences are distinct). Night 1920x1080 (2 sequences) is not held out.")
SELECTION_RULE = ("n_targets frames over ALL train sequences (val included): per-sequence counts by largest remainder "
                  "(Hamilton) on the image counts, ties -> sequence order; inside a sequence of N images the n selected "
                  "0-based frame indices are floor((k + 0.5) * N / n), k = 0..n-1 (evenly spaced, deterministic).")
RUN_RX = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
RID_RX = re.compile(r"^[0-9a-f]{6}$")                       # one id per notebook run: names never collide across attempts
STATE_RX = re.compile(r"^state_ep(\d{3})_([0-9a-f]{6})\.pt$")
MANIFEST_COLUMNS = ("seq", "role", "fi", "mot_frame", "src_image", "image", "feature", "image_bytes", "image_sha256",
                    "feature_bytes", "feature_sha256", "feat_mean", "feat_std", "night", "pen", "fps", "width", "height",
                    "n_images_seq")
_INT_COLS = ("fi", "mot_frame", "image_bytes", "feature_bytes", "night", "width", "height", "n_images_seq")
_FLOAT_COLS = ("feat_mean", "feat_std", "fps")


# =====================================================================================================================
# PigTrack inputs
# =====================================================================================================================
def parse_split_text(text):
    """split.txt: one '<seq> <split>' per line -> {seq: split}. Malformed lines, unknown splits and duplicates raise."""
    out = {}
    for ln, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 2 or not SEQ_RX.match(parts[0]) or parts[1] not in SPLITS:
            raise ValueError(f"split.txt line {ln}: {raw!r}")
        if parts[0] in out:
            raise ValueError(f"split.txt: {parts[0]} listed twice")
        out[parts[0]] = parts[1]
    return out


def parse_seqmap_text(text):
    """TrackEval seqmap: a 'name' header, then one sequence per line."""
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    if not lines or lines[0] != "name":
        raise ValueError("seqmap must start with a 'name' header")
    seqs = lines[1:]
    bad = [s for s in seqs if not SEQ_RX.match(s)]
    if bad or len(set(seqs)) != len(seqs):
        raise ValueError(f"seqmap: bad or duplicate sequence names {bad[:5]}")
    return seqs


def parse_seqinfo_text(text):
    ini = configparser.ConfigParser()
    ini.optionxform = str
    ini.read_string(text)
    s = ini["Sequence"]
    return {"name": s["name"], "imDir": s.get("imDir", "img1"), "imExt": s.get("imExt", ".jpg"),
            "fps": float(s["frameRate"]), "seq_length_ini": int(s["seqLength"]), "width": int(s["imWidth"]),
            "height": int(s["imHeight"]), "pen": str(s.get("penID", "")), "night": int(s.get("night", "0"))}


def count_images(names, ext=".jpg"):
    """Image count of an img1 listing; the files must be 00000001<ext>..<n:08d><ext> contiguous (others are ignored and
    returned). Returns (n, ignored_names)."""
    imgs = sorted(n for n in names if n.endswith(ext))
    want = [f"{k + 1:08d}{ext}" for k in range(len(imgs))]
    if imgs != want:
        raise ValueError(f"images are not 00000001{ext}..{len(imgs):08d}{ext} contiguous (first diff "
                         f"{next((a, b) for a, b in zip(imgs, want) if a != b) if len(imgs) else None})")
    return len(imgs), sorted(n for n in names if not n.endswith(ext))


def seq_info(seqinfo_text, img_names, expect_name):
    """Merge seqinfo.ini and the img1 listing into one record; n = the IMAGE COUNT (seqLength may differ by 1-2)."""
    info = parse_seqinfo_text(seqinfo_text)
    if info["name"] != expect_name:
        raise ValueError(f"seqinfo name {info['name']!r} != directory {expect_name!r}")
    n, ignored = count_images(img_names, info["imExt"])
    if n == 0:
        raise ValueError(f"{expect_name}: no images")
    info.update({"n": n, "ignored_files": ignored,
                 "warning": (f"seqinfo seqLength {info['seq_length_ini']} != {n} images: using the image count"
                             if info["seq_length_ini"] != n else None)})
    return info


# =====================================================================================================================
# frame selection and the val rule
# =====================================================================================================================
def apportion(counts, total):
    """Largest-remainder apportionment of `total` over {key: N_k} (keys in sorted order); ties -> sorted order.
    Every share is <= its N_k; the shares sum to `total`."""
    keys = sorted(counts)
    N = sum(int(counts[k]) for k in keys)
    if total < 0 or total > N:
        raise ValueError(f"cannot pick {total} of {N} frames")
    if N == 0:
        return {k: 0 for k in keys}
    quota = {k: total * int(counts[k]) / N for k in keys}
    out = {k: int(math.floor(quota[k] + 1e-12)) for k in keys}
    left = total - sum(out.values())
    order = sorted(keys, key=lambda k: (-(quota[k] - out[k]), keys.index(k)))
    for k in order[:left]:
        out[k] += 1
    assert sum(out.values()) == total and all(out[k] <= int(counts[k]) for k in keys)
    return out


def even_indices(N, n):
    """n evenly spaced 0-based frame indices in a sequence of N images: floor((k + 0.5) * N / n)."""
    if not 0 <= n <= N:
        raise ValueError(f"cannot pick {n} of {N}")
    idx = [int(math.floor((k + 0.5) * N / n)) for k in range(n)]
    if any(b <= a for a, b in zip(idx, idx[1:])) or (idx and not 0 <= idx[0] <= idx[-1] < N):
        raise AssertionError(f"even_indices({N}, {n}) not strictly increasing in range")
    return idx


def selection_plan(infos, n_targets=N_TARGETS_DEFAULT):
    """{seq: [0-based frame indices]} over every sequence in `infos` ({seq: {'n': image count, ...}})."""
    shares = apportion({s: infos[s]["n"] for s in infos}, n_targets)
    return {s: even_indices(infos[s]["n"], shares[s]) for s in sorted(infos)}


def plan_digest(plan):
    blob = json.dumps({s: list(map(int, v)) for s, v in sorted(plan.items())}, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def choose_val_sequences(infos, cells=VAL_CELLS):
    """The fixed val rule (VAL_RULE). Returns (sorted list of seqs, working record)."""
    chosen, pens, working = [], set(), []
    for night, wh in cells:
        cell = sorted(s for s, i in infos.items() if int(i["night"]) == night and (i["width"], i["height"]) == tuple(wh))
        if not cell:
            raise ValueError(f"val rule: empty cell night={night} {wh}")
        med = statistics.median([infos[s]["n"] for s in cell])
        cands = sorted((abs(infos[s]["n"] - med), s) for s in cell if s not in chosen)
        pick = next((s for _d, s in cands if str(infos[s]["pen"]) not in pens), None)
        if pick is None:
            raise ValueError(f"val rule: no admissible sequence in cell night={night} {wh} (pens {sorted(pens)})")
        skipped = [s for _d, s in cands[:[c[1] for c in cands].index(pick)]]
        chosen.append(pick)
        pens.add(str(infos[pick]["pen"]))
        working.append({"cell": {"night": night, "width": wh[0], "height": wh[1]}, "n_cell": len(cell),
                        "median_images": med, "pick": pick, "pick_images": infos[pick]["n"],
                        "pick_pen": str(infos[pick]["pen"]), "skipped_pen_taken": skipped})
    if len(set(chosen)) != len(chosen):
        raise AssertionError("val rule chose a sequence twice")
    return sorted(chosen), {"rule": VAL_RULE, "picks": working}


def check_val_override(val_seqs, train_seqs, n=N_VAL_SEQS):
    vs = sorted(set(val_seqs))
    if len(vs) != len(val_seqs) or len(vs) != n:
        raise ValueError(f"val_seqs must name {n} distinct sequences, got {val_seqs}")
    bad = [s for s in vs if s not in train_seqs]
    if bad:
        raise ValueError(f"val_seqs {bad} are not PigTrack TRAIN sequences (val/test frames never enter distillation)")
    return vs


def target_basename(seq, fi):
    if not SEQ_RX.match(seq) or int(fi) < 0:
        raise ValueError(f"target_basename({seq!r}, {fi!r})")
    return f"{seq}_f{int(fi):06d}"


def src_image_name(fi, ext=".jpg"):
    return f"{int(fi) + 1:08d}{ext}"              # MOT frame number fi + 1, 8-digit, 1-indexed


def dummy_boxes(width, height):
    """Four [x1, y1, x2, y2] prompt boxes inside the frame (fresh lists every call: the processor mutates its inputs)."""
    out = []
    for a, b, c, d in DUMMY_BOX_FRACS:
        box = [int(round(a * width)), int(round(b * height)), int(round(c * width)), int(round(d * height))]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise AssertionError(f"dummy box {box} outside {width}x{height}")
        out.append(box)
    return out


# =====================================================================================================================
# manifest
# =====================================================================================================================
def manifest_rows(records, infos, val_seqs):
    """Per-sequence records (extract_teacher_targets_pigtrack) -> manifest rows, sorted by (seq, fi). Paths stay RELATIVE to the targets root."""
    rows = []
    for seq in sorted(records):
        rec = records[seq]
        i = infos[seq]
        for f in rec["files"]:
            rows.append({"seq": seq, "role": "val" if seq in val_seqs else "train", "fi": int(f["fi"]),
                         "mot_frame": int(f["fi"]) + 1, "src_image": f["src_image"],
                         "image": f"{rec['dir']}/frames/{f['image']}", "feature": f"{rec['dir']}/features/{f['feature']}",
                         "image_bytes": int(f["image_bytes"]), "image_sha256": f["image_sha256"],
                         "feature_bytes": int(f["feature_bytes"]), "feature_sha256": f["feature_sha256"],
                         "feat_mean": float(f["feat_mean"]), "feat_std": float(f["feat_std"]),
                         "night": int(i["night"]), "pen": str(i["pen"]), "fps": float(i["fps"]),
                         "width": int(i["width"]), "height": int(i["height"]), "n_images_seq": int(i["n"])})
    return sorted(rows, key=lambda r: (r["seq"], r["fi"]))


def manifest_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (repr(float(r[k])) if k in ("feat_mean", "feat_std") else r[k]) for k in MANIFEST_COLUMNS})
    return buf.getvalue()


def read_manifest_csv(text):
    rd = csv.DictReader(io.StringIO(text))
    if tuple(rd.fieldnames or ()) != MANIFEST_COLUMNS:
        raise ValueError(f"manifest columns {rd.fieldnames} != {list(MANIFEST_COLUMNS)}")
    rows = []
    for r in rd:
        r = dict(r)
        for k in _INT_COLS:
            r[k] = int(r[k])
        for k in _FLOAT_COLS:
            r[k] = float(r[k])
        rows.append(r)
    return rows


def validate_manifest(rows, plan=None, val_seqs=None, n_targets=None):
    """Structural checks; with plan / val_seqs / n_targets also the exact content. Returns a summary dict."""
    keys = [(r["seq"], r["fi"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("manifest: duplicate (seq, fi)")
    for r in rows:
        if r["role"] not in ("train", "val") or r["mot_frame"] != r["fi"] + 1:
            raise ValueError(f"manifest row {r['seq']} {r['fi']}: role / mot_frame")
        for k in ("image", "feature"):
            p = r[k]
            if p.startswith("/") or ".." in p.split("/") or ":" in p:
                raise ValueError(f"manifest {k} path must be relative: {p!r}")
        if not (re.match(r"^[0-9a-f]{64}$", r["image_sha256"]) and re.match(r"^[0-9a-f]{64}$", r["feature_sha256"])):
            raise ValueError(f"manifest row {r['seq']} {r['fi']}: sha256")
        if not (math.isfinite(r["feat_mean"]) and math.isfinite(r["feat_std"]) and r["feat_std"] > core.MIN_STD):
            raise ValueError(f"manifest row {r['seq']} {r['fi']}: feature stats {r['feat_mean']} {r['feat_std']}")
    roles = {}
    for r in rows:
        roles.setdefault(r["seq"], set()).add(r["role"])
    if any(len(v) != 1 for v in roles.values()):
        raise ValueError("manifest: a sequence carries both roles")
    if plan is not None:
        got = {}
        for r in rows:
            got.setdefault(r["seq"], []).append(r["fi"])
        want = {s: list(v) for s, v in plan.items() if v}
        if {s: sorted(v) for s, v in got.items()} != want:
            diff = sorted(set(want) ^ set(got)) or [s for s in want if sorted(got.get(s, [])) != want[s]]
            raise ValueError(f"manifest frames differ from the selection plan (sequences {diff[:5]})")
    if val_seqs is not None:
        vs = sorted(s for s, v in roles.items() if "val" in v)
        if vs != sorted(val_seqs):
            raise ValueError(f"manifest val sequences {vs} != {sorted(val_seqs)}")
    if n_targets is not None and len(rows) != n_targets:
        raise ValueError(f"manifest has {len(rows)} rows, expected {n_targets}")
    n_val = sum(1 for r in rows if r["role"] == "val")
    return {"rows": len(rows), "train": len(rows) - n_val, "val": n_val, "sequences": len(roles),
            "val_seqs": sorted(s for s, v in roles.items() if "val" in v),
            "night_rows": sum(1 for r in rows if r["night"]), "bytes": sum(r["image_bytes"] + r["feature_bytes"]
                                                                            for r in rows)}


# =====================================================================================================================
# files: hashing, parallel local staging, no-overwrite publishing
# =====================================================================================================================
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def list_tree(root):
    """{relative posix path: size} of every file under root ('' if root is missing -> {})."""
    out = {}
    if not os.path.isdir(root):
        return out
    for d, _ds, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root).replace("\\", "/")] = os.path.getsize(p)
    return out


def sha256_many(paths, threads=8):
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        return dict(zip(paths, ex.map(core.sha256_file, paths)))


def copy_files_parallel(jobs, threads=16, log=print):
    """jobs: [(src, dst_local, expected_size or None or 'src', expected_sha256 or None)]. Copies to dst.part then
    os.replace (a dst that already exists with the expected size and sha is kept). size 'src' = the source file's size.
    Destinations must be LOCAL. Returns stats."""
    for _s, d, _n, _h in jobs:
        core.assert_local_target(d)
    t0 = time.time()

    def one(job):
        src, dst, size, sha = job
        if size == "src":
            size = os.path.getsize(src)
        if os.path.isfile(dst) and (size is None or os.path.getsize(dst) == size) and \
                (sha is None or core.sha256_file(dst) == sha):
            return 0, 0
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        part = dst + ".part"
        core._retry(lambda: shutil.copyfile(src, part), f"copy {os.path.basename(src)}", sleep_s=5.0)
        got = os.path.getsize(part)
        if size is not None and got != size:
            raise IOError(f"{src}: copied {got} B, expected {size}")
        if sha is not None and core.sha256_file(part) != sha:
            raise IOError(f"{src}: sha256 differs from the manifest")
        os.replace(part, dst)
        return 1, got

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        res = list(ex.map(one, jobs))
    n = sum(r[0] for r in res)
    b = sum(r[1] for r in res)
    dt = time.time() - t0
    return {"files": len(jobs), "copied": n, "bytes_copied": b, "seconds": round(dt, 1),
            "mb_per_s": round(b / 1e6 / dt, 1) if dt > 0 and b else None}


def _copy_tree_new(local_dir, dst, dbutils):
    if dbutils is not None:
        core._retry(lambda: dbutils.fs.cp("file:" + os.path.abspath(local_dir), core.dbfs_uri(dst), recurse=True),
                    f"publish dir {os.path.basename(dst)}")
    else:
        shutil.copytree(local_dir, dst)


def publish_tree_attempt(local_dir, dst_parent, name, local_sha=None, dbutils=None, max_attempts=9, log=print):
    """Publish every file under local_dir as <dst_parent>/<name>/... without ever overwriting or deleting Volume content.
    If <name> exists: it is reused iff every file it already holds is one of ours with the same size AND sha256 (only the
    missing files are then copied); otherwise the next attempt directory <name>__a2, __a3, ... is used (a crash during
    an earlier copy of a re-extracted sequence whose bf16 features are not bit-identical lands here).
    local_sha: {relpath: sha256} if already known. Returns {'name', 'dst', 'mode', 'files', 'bytes', 'seconds'}."""
    t0 = time.time()
    local = list_tree(local_dir)
    if not local:
        raise ValueError(f"{local_dir}: nothing to publish")
    sha = dict(local_sha or {})
    missing_sha = [r for r in local if r not in sha]
    if missing_sha:
        sha.update({r: core.sha256_file(os.path.join(local_dir, r)) for r in missing_sha})
    for k in range(1, max_attempts + 1):
        nm = name if k == 1 else f"{name}__a{k}"
        dst = f"{dst_parent.rstrip('/')}/{nm}"
        core.assert_publish_target(dst + "/x")
        if not os.path.exists(dst):
            _copy_tree_new(local_dir, dst, dbutils)
            mode = "new"
        else:
            remote = list_tree(dst)
            extra = sorted(set(remote) - set(local))
            if extra:
                log(f"publish {nm}: holds {len(extra)} foreign file(s) (e.g. {extra[0]}); trying the next attempt dir")
                continue
            bad = [r for r in remote if remote[r] != local[r] or core.sha256_file(os.path.join(dst, r)) != sha[r]]
            if bad:
                log(f"publish {nm}: {len(bad)} existing file(s) differ (e.g. {bad[0]}); trying the next attempt dir")
                continue
            mode = "identical" if len(remote) == len(local) else "resume"
        remote = list_tree(dst)
        for rel in sorted(set(local) - set(remote)):         # a partial recursive copy, or the resume case
            core.publish_no_overwrite(os.path.join(local_dir, rel), f"{dst}/{rel}", dbutils)
        remote = list_tree(dst)
        extra = sorted(set(remote) - set(local))
        wrong = [r for r in local if remote.get(r) != local[r]]
        if wrong:
            raise IOError(f"publish {dst}: {len(wrong)} file(s) missing or with the wrong size (e.g. {wrong[0]})")
        if extra:
            log(f"publish {dst}: WARNING {len(extra)} unexpected extra file(s) on the Volume (e.g. {extra[0]}); kept")
        return {"name": nm, "dst": dst, "mode": mode, "files": len(local), "bytes": int(sum(local.values())),
                "seconds": round(time.time() - t0, 1), "extra_remote": extra}
    raise RuntimeError(f"publish {name}: no usable attempt directory among {max_attempts}")


def publish_file_guarded(src, dst, allowed_dir, dbutils=None):
    """No-overwrite publish of one finished LOCAL file to dst, which must lie directly inside `allowed_dir` (a path from
    R.out(); used for models/sam3_student_tinyvit/pigtrack_v1/, which the bench core's paper1-only guard refuses).
    Same semantics as bench_core.publish_no_overwrite: identical content -> 'identical', different -> raises."""
    a = str(allowed_dir).replace("\\", "/").rstrip("/")
    d = str(dst).replace("\\", "/")
    if os.path.dirname(d) != a or ".." in d.split("/") or "/Tol/" in d + "/":
        raise PermissionError(f"{dst}: may only publish directly inside {allowed_dir}")
    core.assert_no_secrets(src)
    if os.path.exists(dst):
        if core.sha256_file(dst) == core.sha256_file(src):
            return "identical"
        raise FileExistsError(f"{dst} exists with different content; refusing to overwrite")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    want = os.path.getsize(src)
    if dbutils is not None:
        def _cp():
            if os.path.exists(dst) and core.sha256_file(dst) == core.sha256_file(src):
                return
            if os.path.exists(dst) and os.path.getsize(dst) >= want:
                raise FileExistsError(f"{dst} appeared during the publish with different content")
            dbutils.fs.cp("file:" + os.path.abspath(src), core.dbfs_uri(dst))
            if not os.path.exists(dst) or os.path.getsize(dst) != want:
                raise IOError(f"published {dst}: size mismatch")
        core._retry(_cp, f"publish {os.path.basename(dst)}")
    else:
        part = dst + ".part"
        shutil.copyfile(src, part)
        os.replace(part, dst)
    if core.sha256_file(dst) != core.sha256_file(src):
        raise IOError(f"published {dst}: sha256 differs from the local file")
    return "copied"


# =====================================================================================================================
# training-run bookkeeping
# =====================================================================================================================
def check_run_tag(run):
    if not RUN_RX.match(run or "") or "__" in run:
        raise ValueError(f"run={run!r}: one path segment of letters, digits, '_', '.', '-'")
    return run


def is_smoke(run):
    return run.startswith("smoke")


def state_name(epoch_1based, rid):
    if not RID_RX.match(rid):
        raise ValueError(f"run id {rid!r}")
    return f"state_ep{int(epoch_1based):03d}_{rid}.pt"


def state_marker(name):
    return name[:-3] + ".json"                  # published AFTER the state: its presence = the state is complete


def latest_state(names, sizes=None, markers=None):
    """The highest-epoch COMPLETE resume state among `names` (a resume/ listing): 'state_epNNN_<rid>.pt' whose marker
    'state_epNNN_<rid>.json' is listed too and, when sizes / markers ({marker name: parsed json}) are given, whose size
    equals the marker's 'bytes'. Ties on the epoch -> the lexically last name. Returns (epoch, name) or (0, None)."""
    have = set(names)
    best = (0, None)
    for n in sorted(names):
        m = STATE_RX.match(n)
        if not m or state_marker(n) not in have:
            continue
        if sizes is not None and markers is not None:
            mk = markers.get(state_marker(n)) or {}
            if sizes.get(n) != mk.get("bytes"):
                continue
        if int(m.group(1)) >= best[0]:
            best = (int(m.group(1)), n)
    return best


# >>> VERBATIM the fidelity audit.py (stage d fidelity; definitions shared with the S-Edi audit) -------------------
FID_KEYS = ("cos_patch", "cos_patch_p05", "cos_patch_p50", "cos_global", "scale_ratio", "mse", "mse_dir",
            "ch_scale", "loss_total", "student_std", "teacher_std")
COS_LOSS_EPS = 1e-12                  # F.cosine_embedding_loss EPSILON
COS_SIM_EPS = 1e-8                    # F.cosine_similarity eps
NORM_EPS = 1e-12                      # F.normalize eps


def _ustd(a):
    m = a.mean()
    return float(np.sqrt(((a - m) ** 2).sum() / (a.size - 1)))


def fidelity_metrics(s, t):
    """Student vs teacher backbone features ([1,]C,H,W), float64 on the host. Definitions = the training loss
    (Fixed cell 17) and the paper's verification (cell 21):
      cos_patch   mean over the H*W positions of the C-dim cosine (1 - F.cosine_embedding_loss, eps 1e-12)
      cos_global  cosine of the flattened tensors (F.cosine_similarity, cell 21 'Cosine')
      scale_ratio std(student) / std(teacher), unbiased (torch .std(), cell 21 'Scale Ratio')
      mse         F.mse_loss (cell 21 'MSE');  mse_dir  MSE of the channel-L2-normalised maps (F.normalize dim=1)
      ch_scale    MSE of per-channel std (unbiased) + MSE of per-channel mean over H*W
      loss_total  1.0 mse_dir + 0.5 (1 - cos_patch) + 0.3 ch_scale + 0.1 mse (BackboneDistillationLoss, batch 1)"""
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    while s.ndim > 3 and s.shape[0] == 1:
        s = s[0]
    while t.ndim > 3 and t.shape[0] == 1:
        t = t[0]
    if s.shape != t.shape or s.ndim != 3:
        raise ValueError(f"feature shapes {s.shape} vs {t.shape}")
    C = s.shape[0]
    S, T = s.reshape(C, -1), t.reshape(C, -1)
    dot = (S * T).sum(0)
    ns, nt = (S * S).sum(0), (T * T).sum(0)
    cos_p = dot / np.sqrt((ns + COS_LOSS_EPS) * (nt + COS_LOSS_EPS))
    cos_g = float((S * T).sum() / max(math.sqrt(float(ns.sum())) * math.sqrt(float(nt.sum())), COS_SIM_EPS))
    Sn = S / np.maximum(np.sqrt(ns), NORM_EPS)
    Tn = T / np.maximum(np.sqrt(nt), NORM_EPS)
    ch_scale = float(((S.std(axis=1, ddof=1) - T.std(axis=1, ddof=1)) ** 2).mean()
                     + ((S.mean(axis=1) - T.mean(axis=1)) ** 2).mean())
    mse = float(((S - T) ** 2).mean())
    mse_dir = float(((Sn - Tn) ** 2).mean())
    cp = float(cos_p.mean())
    ss, ts = _ustd(S), _ustd(T)
    return {"cos_patch": cp, "cos_patch_p05": float(np.percentile(cos_p, 5)),
            "cos_patch_p50": float(np.percentile(cos_p, 50)), "cos_global": cos_g,
            "scale_ratio": ss / ts if ts > 0 else float("nan"), "mse": mse, "mse_dir": mse_dir, "ch_scale": ch_scale,
            "loss_total": 1.0 * mse_dir + 0.5 * (1.0 - cp) + 0.3 * ch_scale + 0.1 * mse,
            "student_std": ss, "teacher_std": ts}


def summarize(rows, keys=FID_KEYS):
    out = {"n": len(rows)}
    for k in keys:
        v = np.asarray([r[k] for r in rows], dtype=np.float64)
        if not v.size:
            continue
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max()), "median": float(np.median(v))}
    return out
# <<< VERBATIM the fidelity audit fidelity -------------------------------------------------------------------------------------------


def fidelity_groups(frame_rows, students=("S-PT",), precisions=("fp32", "bf16")):
    """frame_rows: {'seq', 'night', 'fi', '<student>': {'fp32': metrics, 'bf16': metrics}} -> summaries overall, per
    night flag and per sequence, for every student and precision."""
    out = {}
    for st in students:
        rs = [r for r in frame_rows if st in r]
        if not rs:
            continue
        d = {}
        for pr in precisions:
            d[pr] = {"overall": summarize([r[st][pr] for r in rs]),
                     "day": summarize([r[st][pr] for r in rs if not r["night"]]),
                     "night": summarize([r[st][pr] for r in rs if r["night"]]),
                     "per_seq": {s: summarize([r[st][pr] for r in rs if r["seq"] == s])
                                 for s in sorted({r["seq"] for r in rs})}}
        out[st] = d
    return out


def json_dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True, default=core._json_default) + "\n"
