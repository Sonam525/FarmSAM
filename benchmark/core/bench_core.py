#!/usr/bin/env python
"""bench_core: everything the tracking models share in the FarmSAM benchmark.

Frame sources, model loaders, the streaming loop (explicit frame_idx, frame-cache release, prune keep_last_n=16 every 25
frames, fresh obj_ids lists, one add_inputs call per frame), box and detector prompts, the output rule, MOT and RLE writers,
telemetry and the paired bootstrap.

Quick use:

    import bench_core as core
    student, meta = core.load_student(path, device="cuda")
    x = core.make_student_transform()(pil_image)
    arm = core.load_arm("S-PT", registry=farm_paths.load())

The module imports with only numpy and the standard library. torch, transformers, timm, PIL, cv2 and rfdetr load inside the
calls that need them. Paths come from the registry in farm_paths.py. `python bench_core.py --api` lists the public API.
"""
from __future__ import annotations

import collections
import configparser
import contextlib
import gc
import glob
import gzip
import hashlib
import json
import math
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:                                   # torch is OPTIONAL at import time 
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:                      # noqa: BLE001
    torch = None
    F = None
    TORCH_OK = False

    class nn:                          # noqa: N801 -- import-time stand-in so the VERBATIM class statements below
        Module = object                # still define; they are never instantiated without torch (_require_torch)

timm = None                            # set by _require_torch() (the verbatim student calls timm.create_model)
BaseModelOutput = None                 # set by _require_torch(need_transformers=True) (the verbatim DummyBackbone)

BENCH_VERSION = "b1.0 2026-10-14"     # label written into BENCH_CONFIG
CORE_REVISION = "core-2026-10-15.b13"  # harness revision: + ES-L wiring; BENCH_CONFIG holds the code sha256

# =====================================================================================================================
# Frozen protocol constants. Changing ANY of them is a protocol amendment, not a tweak.
# =====================================================================================================================
TRANSFORMERS_PIN = "5.2.0"            # every arm
DTYPE_NAME = "bfloat16"               # every module, every arm (A3); fp16 forbidden
ATTN_IMPL = "sdpa"                    # requested for every arm; a resolved flash_attention_2 fails the load
MATMUL_PRECISION = "highest"          # fp32 matmuls (none in the bf16 graph; restored after every detector call)
KEEP_LAST_N = 16                      # >= max_object_pointers_in_encoder - 1 = 15 -> output-identical
PRUNE_EVERY = 25                      # prune after the forward of frame fi when (fi + 1) % 25 == 0
POP_FRAMES = True                     # pop processed_frames[fi] after its own forward (explicit frame_idx always)
CHECK_EVERY = 25                      # feature / mask-logit collapse checks (finite checks run on EVERY frame)
FINITE_EVERY = 1                      # finite asserts on every frame (efficiency latency passes may use 25, §8)
FRAME_HASH_EVERY = 500                # sha256 of the RGB frame every 500 frames: identical pixels across arms
DIAG_EVERY = 5                        # pairwise mask overlap diagnostic cadence (frames)
MIN_STD = 1e-6                        # "collapsed" = std <= MIN_STD
MIN_AREA_PX = 0                       # output rule: NO minimum-area rule (review R1.7)
MOT_XY_OFFSET_DEFAULT = 0             # box origin written to MOT; fixed ONCE at J0 and recorded in BENCH_CONFIG
WARMUP_FRAMES = 32                    # telemetry: frames < 32 excluded from latency/steady-memory statistics
TIMED_FRAMES = 500                    # efficiency microbenchmark window (efficiency_bench)
STUDENT_N_PARAMS = 40_655_408         # StudentEncoderTinyViT(out_channels=1024)
GN_TRAIN_EVAL_MAX = 1e-5              # GroupNorm determinism gate (drop_path_rate=0.0) at student load
# ---- detector runs ----
D_PASS_PERIOD_S = 1.0                 # one detector pass per 1 s of video: frames fi % round(fps * 1.0) == 0
D_MATCH_IOU = 0.5                     # detection <-> live track: Hungarian on 1 - IoU, matched iff IoU >= 0.5
D_BIRTH_IOS = 0.3                     # a birth needs IoS < 0.3 with EVERY live track (and every birth of the pass)
D_BIRTH_PASSES = 2                    # ... and to be unmatched on 2 CONSECUTIVE passes
D_DEATH_S = 3.0                       # absent (score <= 0 or empty mask) for > 3 s -> dead (never output again)
D_TAU_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)   # tau (seed = birth score threshold) frozen on PigTrack val with T
DET_PROBE_THRESHOLD = 0.10            # detector called once at 0.10 (strict >); tau is applied offline from the cache
# ---- statistics ----
BOOT_B = 10_000
BOOT_SEED = 20261014
BOOT_ALPHA = 0.05
EQUIV_MARGIN_HOTA = 1.0               # delta for equivalence / non-inferiority statements (HOTA points)

MODULE_GROUPS = ("vision_encoder.backbone", "vision_encoder.neck", "memory_attention", "spatial_perceiver",
                 "memory_encoder", "mask_decoder", "prompt_encoder")   # baselines.md §6.0 (identical names in all 3)

DATASETS = ("edinburgh", "pigtrack", "calves_tol")
PROTOCOLS = ("A", "D")
EDI_TRAIN = ("2019_11_05_000002", "2019_11_11_000028", "2019_11_11_000036", "2019_11_22_000010")
EDI_VAL = ("2019_11_28_000113",)
EDI_TEST = ("2019_12_02_000005", "2019_12_02_000208", "2019_12_10_000060", "2019_12_10_000078")
EDI_FPS = 2.0                         # decoded frame k = raw frame 3k of the 6 fps color.mp4
EDI_N_FRAMES = 600
PIGTRACK_TEST = tuple(f"pigtrack{n}" for n in (
    "0002", "0004", "0005", "0010", "0011", "0017", "0018", "0021", "0030", "0038", "0040", "0042", "0049", "0054",
    "0055", "0058", "0060", "0065", "0067", "0068", "0071", "0072", "0074", "0077", "0078"))
CALF_TRACK_FPS = 5.0

# OpenCV enum values (stable across 4.x) -- used without importing cv2 in the calf source when a cap is injected
CV_CAP_PROP_POS_FRAMES, CV_CAP_PROP_FPS, CV_CAP_PROP_FRAME_COUNT = 1, 5, 7


class BenchAssertError(RuntimeError):
    """A validity assert failed: the run is INVALID -- never report its numbers."""


def _require_torch(need_transformers=False):
    global timm, BaseModelOutput
    if not TORCH_OK:
        raise RuntimeError("this call needs torch (GPU notebooks have it)")
    if timm is None:
        import timm as _timm
        timm = _timm
    if need_transformers and BaseModelOutput is None:
        from transformers.modeling_outputs import BaseModelOutput as _BMO
        BaseModelOutput = _BMO


# >>> VERBATIM student_fixed.txt cell 13 ("CELL 7: STUDENT ENCODER", Fixed notebook = the notebook that produced the
# published student; identical in _cleaned). ONE change, marked [bench-C1]: drop_path_rate=0.0 (current
# timm defaults to nonzero stochastic depth for this model; eval() is unaffected, but the GroupNorm train/eval identity
# is only true with it off).
def replace_bn_with_gn(module):
    """Recursively replace all BatchNorm layers with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nc = child.num_features
            # Find largest divisor of nc that is <= 32
            ng = min(32, nc)
            while nc % ng != 0:
                ng -= 1
            gn = nn.GroupNorm(ng, nc, affine=child.affine)
            if child.affine and child.weight is not None:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child)


class StudentEncoderTinyViT(nn.Module):
    """
    TinyViT-21M-512 with multi-scale feature fusion.
    ALL normalization is GroupNorm - zero train/eval mismatch.
    """
    def __init__(self, out_channels=1024, pretrained=True):
        super().__init__()
        self.out_channels = out_channels

        self.backbone = timm.create_model(
            'tiny_vit_21m_512',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
            drop_path_rate=0.0,  # [bench-C1] the only change vs the export
        )
        # Replace ALL BatchNorm in backbone with GroupNorm
        replace_bn_with_gn(self.backbone)

        # Per-stage lateral convs (already GroupNorm)
        self.lateral_s1 = nn.Sequential(
            nn.Conv2d(192, 256, 1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )
        self.lateral_s2 = nn.Sequential(
            nn.Conv2d(384, 256, 1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )
        self.lateral_s3 = nn.Sequential(
            nn.Conv2d(576, 256, 1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )

        # Project fused 768ch -> 1024
        self.channel_proj = nn.Sequential(
            nn.Conv2d(768, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

        # Spatial refinement at 72x72 - residual path
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
        )

        # Learnable per-channel affine
        self.output_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.output_bias  = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        self._init_weights()

    def _init_weights(self):
        for module_group in [self.lateral_s1, self.lateral_s2, self.lateral_s3,
                             self.channel_proj, self.spatial_refine]:
            for layer in module_group.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.GroupNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        # Zero-init last GroupNorm so residual starts as identity
        nn.init.zeros_(self.spatial_refine[-1].weight)

    def forward(self, x):
        stages = self.backbone(x)

        TARGET_H, TARGET_W = 72, 72

        f1 = self.lateral_s1(stages[0])
        f2 = self.lateral_s2(stages[1])
        f3 = self.lateral_s3(stages[2])

        f1 = F.interpolate(f1, size=(TARGET_H, TARGET_W), mode='area')
        f2 = F.interpolate(f2, size=(TARGET_H, TARGET_W), mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=(TARGET_H, TARGET_W), mode='bilinear', align_corners=False)

        fused = torch.cat([f1, f2, f3], dim=1)

        out = self.channel_proj(fused)
        out = out + self.spatial_refine(out)
        out = out * self.output_scale + self.output_bias

        return out

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
# <<< VERBATIM student


# >>> DummyBackbone
class DummyBackbone(nn.Module):
    def __init__(self): super().__init__(); self.student_features = None
    def forward(self, *a, **k):
        f = self.student_features; B, C, H, W = f.shape
        return BaseModelOutput(last_hidden_state=f.permute(0, 2, 3, 1).reshape(B, H*W, C))
# <<< VERBATIM DummyBackbone


def make_student_transform():
    """VERBATIM semantics of the export's STUDENT_TRANSFORM (cell 15): Resize((1024, 1024)) -> ToTensor -> Normalize
    with tiny_vit_21m_512's pretrained_cfg mean/std (timm resolve_data_config) (default, non-GPU)."""
    _require_torch()
    from torchvision import transforms
    from timm.data import resolve_data_config
    _tmp = timm.create_model('tiny_vit_21m_512', pretrained=False)
    _data_cfg = resolve_data_config(_tmp.pretrained_cfg)
    del _tmp
    return transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_data_cfg['mean'], std=_data_cfg['std']),
    ])


# =====================================================================================================================
# small utilities: hashing, provenance, atomic local writes, no-overwrite publish, staging
# =====================================================================================================================
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_shas(data_or_path):
    """{'source': sha256 of the raw bytes, 'source_norm': transport-invariant sha256} -- the same two digests
    bench_core.source_shas() defines (BOM dropped, CRLF->LF, trailing whitespace and blank lines dropped)."""
    if isinstance(data_or_path, (bytes, bytearray)):
        b = bytes(data_or_path)
    elif isinstance(data_or_path, str) and os.path.isfile(data_or_path):
        with open(data_or_path, "rb") as f:
            b = f.read()
    else:
        b = str(data_or_path).encode("utf-8")
    t = b.decode("utf-8", "replace").lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    norm = "\n".join(l.rstrip() for l in t.split("\n") if l.strip()).encode("utf-8")
    return {"source": hashlib.sha256(b).hexdigest(), "source_norm": hashlib.sha256(norm).hexdigest()}


def code_sha256():
    """The digests of THIS module file (recorded in every BENCH_CONFIG)."""
    return source_shas(os.path.abspath(__file__))


def write_text_atomic(path, text):
    """LOCAL disk only (UC Volumes refuse seek-writes and appends): temp file + os.replace."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp_{os.getpid()}_{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def write_json_atomic(path, obj):
    return write_text_atomic(path, json.dumps(obj, indent=1, sort_keys=True, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dbfs_uri(path):
    """'/Volumes/..' -> 'dbfs:/Volumes/..'; '/dbfs/mnt/..' -> 'dbfs:/mnt/..'; anything else is returned unchanged."""
    if path.startswith("/dbfs/"):
        return "dbfs:" + path[len("/dbfs"):]
    if path.startswith("/Volumes/"):
        return "dbfs:" + path
    return path


REMOTE_PREFIXES = ("/Volumes/", "/dbfs/", "dbfs:", "/mnt/", "/Workspace/")   # never a staging / scratch target
PAPER1_SEGMENT = "paper1_distillation"     # the ONLY Volume subtree new Paper 1 code may write (registry paper1.*)
CP_ATTEMPTS, CP_SLEEP_S = 3, 10.0          # dbutils.fs.cp: transient FUSE / storage errors are retried
_SECRET_VALUES = set()                     # in-memory only: secrets fetched through hf_token_from_dbutils
_SECRET_RX = re.compile(rb"hf_[A-Za-z0-9]{30,}")              # HF access-token shape
_SECRET_SCAN_EXT = (".json", ".jsonl", ".txt", ".csv", ".md", ".tex", ".log", ".yaml", ".yml", ".ini", ".py")


def _segments(path):
    return [s for s in str(path).replace("\\", "/").split("/") if s]


def assert_publish_target(dst):
    """Defence in depth behind R.out(): a remote (Volume / dbfs / mount) publish target must lie under
    paper1_distillation/ and never under Tol/ or the blob mount. Local paths pass unchanged."""
    p = str(dst).replace("\\", "/")
    if not p.startswith(REMOTE_PREFIXES):
        return dst
    seg = _segments(p)
    if p.startswith(("/dbfs/mnt/", "dbfs:/mnt/", "/mnt/")):
        raise PermissionError(f"{dst}: publishing to the blob mount is forbidden")
    if PAPER1_SEGMENT not in seg or "Tol" in seg[seg.index(PAPER1_SEGMENT):] or ".." in seg:
        raise PermissionError(f"{dst}: Paper 1 code writes only under {PAPER1_SEGMENT}/ (never Tol/); use R.out('paper1.*')")
    return dst


def assert_local_target(dst):
    """Staging / scratch targets are LOCAL disk only (stage_file removes a stale dst before copying)."""
    p = str(dst).replace("\\", "/")
    if p.startswith(REMOTE_PREFIXES):
        raise PermissionError(f"{dst}: staging target must be local disk (/local_disk0/...), not the Volume / mount")
    return dst


def register_secret(value):
    """Remember a secret (in memory only) so publish_no_overwrite refuses any file that contains it."""
    if value and len(value) >= 8:
        _SECRET_VALUES.add(value.encode("utf-8"))
    return value


def assert_no_secrets(path):
    """Refuse to publish a file that contains a registered secret, or (text files) an HF-token-shaped string."""
    pats = [s for s in _SECRET_VALUES]
    text = str(path).lower().endswith(_SECRET_SCAN_EXT)
    if not pats and not text:
        return path
    keep = max([len(s) for s in pats] + [64])
    tail = b""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            buf = tail + chunk
            if any(s in buf for s in pats) or (text and _SECRET_RX.search(buf)):
                raise PermissionError(f"{path}: contains a secret-shaped string; refusing to publish it")
            tail = buf[-keep:]
    return path


def _retry(fn, what, attempts=CP_ATTEMPTS, sleep_s=CP_SLEEP_S, log=print):
    """Run fn(); retry transient errors (never PermissionError / FileExistsError, which are decisions, not glitches)."""
    for k in range(1, attempts + 1):
        try:
            return fn()
        except (PermissionError, FileExistsError):
            raise
        except Exception as e:  # noqa: BLE001
            if k == attempts:
                raise
            log(f"{what}: attempt {k}/{attempts} failed ({type(e).__name__}: {str(e)[:200]}); retrying in {sleep_s:g} s")
            time.sleep(sleep_s)


def publish_no_overwrite(src, dst, dbutils=None, attempts=CP_ATTEMPTS, sleep_s=CP_SLEEP_S):
    """Copy a finished LOCAL file to dst (a Volume path from R.out). Never overwrites: an existing dst with the same
    sha256 is 'identical' (resume), a different one raises (existing Volume content is never
    replaced). Copies via dbutils.fs.cp when given (fast path; retried, size-verified), else shutil to dst.part +
    os.replace. A failed copy may leave a TRUNCATED dst that this call created moments ago; only such a file (shorter
    than src and modified after this call started) is replaced on the retry -- anything else still raises."""
    import shutil
    assert_publish_target(dst)
    assert_no_secrets(src)
    if os.path.exists(dst):
        if sha256_file(dst) == sha256_file(src):
            return "identical"
        raise FileExistsError(f"{dst} exists with different content; refusing to overwrite (write a new variant dir)")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    want = os.path.getsize(src)
    if dbutils is not None:
        t_start = time.time() - 2.0                          # FUSE mtime granularity

        def _cp():
            if os.path.exists(dst):                          # left by a failed attempt of THIS call?
                if sha256_file(dst) == sha256_file(src):
                    return
                if not (os.path.getsize(dst) < want and os.path.getmtime(dst) >= t_start):
                    raise FileExistsError(f"{dst} appeared during the publish with different content; refusing to "
                                          "overwrite (a concurrent publisher?)")
            dbutils.fs.cp("file:" + os.path.abspath(src), dbfs_uri(dst))
            got = os.path.getsize(dst) if os.path.exists(dst) else -1
            if got != want:
                raise IOError(f"published {dst}: {got} B on the Volume vs {want} B local")
        _retry(_cp, f"publish {os.path.basename(dst)}", attempts, sleep_s)
        return "copied"
    part = dst + ".part"
    shutil.copyfile(src, part)
    os.replace(part, dst)
    return "copied"


def _tree_stats(root):
    n, b = 0, 0
    for d, _ds, fs in os.walk(root):
        for f in fs:
            if f.startswith(".staged_ok"):
                continue
            n += 1
            b += os.path.getsize(os.path.join(d, f))
    return n, b


def _crc_sidecars(root):
    """Relative paths of Hadoop checksum sidecars ('.<name>.crc' beside '<name>') under root."""
    out = set()
    for d, _ds, fs in os.walk(root):
        names = set(fs)
        for f in fs:
            if f.startswith(".") and f.endswith(".crc") and f[1:-4] in names:
                out.add(os.path.relpath(os.path.join(d, f), root))
    return out


def drop_hadoop_crc(dst, src=None):
    """dbutils.fs.cp to a 'file:' target goes through Hadoop's ChecksumFileSystem, which writes a '.<name>.crc' sidecar
    beside every file it creates (J0 2026-10-14: 600 Edinburgh frames staged as 1,200 files, so the count check of
    stage_tree failed). Removes every such sidecar under dst (a directory, or the sidecar of one file) that the source
    does not itself contain. Returns the number removed."""
    if os.path.isfile(dst):
        side = os.path.join(os.path.dirname(dst), "." + os.path.basename(dst) + ".crc")
        keep = src is not None and os.path.isfile(os.path.join(os.path.dirname(str(src)), os.path.basename(side)))
        if os.path.isfile(side) and not keep:
            os.remove(side)
            return 1
        return 0
    keep = _crc_sidecars(src) if src is not None and os.path.isdir(src) else set()
    n = 0
    for rel in sorted(_crc_sidecars(dst) - keep):
        os.remove(os.path.join(dst, rel))
        n += 1
    return n


def stage_tree(src, dst, dbutils=None):
    """Stage a directory (Volume -> /local_disk0) ONCE: dbutils.fs.cp(recurse=True) when given (never read frames
    through /dbfs inside the loop, A3), else shutil.copytree. Verified by file count + bytes; a .staged_ok.json marker
    makes a re-stage a no-op. Returns the marker record."""
    import shutil
    assert_local_target(dst)
    marker = os.path.join(dst, ".staged_ok.json")
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8") as f:
            return json.load(f)
    t0 = time.time()
    os.makedirs(dst, exist_ok=True)
    if dbutils is not None:
        _retry(lambda: dbutils.fs.cp(dbfs_uri(src), "file:" + os.path.abspath(dst), recurse=True),
               f"stage {os.path.basename(str(src).rstrip('/'))}")
        how = "dbutils.fs.cp"
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        how = "shutil.copytree"
    crc = drop_hadoop_crc(dst, src)
    ns, bs = _tree_stats(src)
    nd, bd = _tree_stats(dst)
    if (ns, bs) != (nd, bd):
        raise RuntimeError(f"staging {src} -> {dst}: {nd} files / {bd} B staged vs {ns} / {bs} at the source")
    rec = {"src": src, "dst": dst, "files": nd, "bytes": bd, "seconds": round(time.time() - t0, 1), "how": how,
           "hadoop_crc_removed": crc}
    write_json_atomic(marker, rec)
    return rec


def stage_file(src, dst, dbutils=None, sha256=None):
    """Stage one file (model checkpoint / clip) to local disk; optional sha256 pin (raises on mismatch)."""
    import shutil
    assert_local_target(dst)                       # the os.remove below must never reach the Volume / mount
    if not (os.path.isfile(dst) and sha256 and sha256_file(dst) == sha256):
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        if os.path.exists(dst):
            os.remove(dst)
        if dbutils is not None:
            _retry(lambda: dbutils.fs.cp(dbfs_uri(src), "file:" + os.path.abspath(dst)), f"stage {os.path.basename(dst)}")
            drop_hadoop_crc(dst, src)
            if os.path.isfile(src) and os.path.getsize(src) != os.path.getsize(dst):
                raise RuntimeError(f"staging {src}: {os.path.getsize(dst)} B local vs {os.path.getsize(src)} B source")
        else:
            shutil.copyfile(src, dst)
    sha = sha256_file(dst)
    if sha256 and sha != sha256:
        raise RuntimeError(f"{src}: staged sha256 {sha[:16]} != pinned {sha256[:16]}")
    return dst, sha


def hf_token_from_dbutils(dbutils):
    """HF token ONLY via the secret scope, whitespace-stripped (pasted secrets carry a newline -> HF 400). Never print it.
    The value is registered (in memory) so that publish_no_overwrite refuses any output file that contains it."""
    return register_secret(dbutils.secrets.get("hf", "token").strip())


# =====================================================================================================================
# frame sources: every source yields (fi, PIL RGB image, meta) with fi = 0, 1, 2, ... contiguous
# =====================================================================================================================
_ITEM, _END, _ERR = object(), object(), object()


def prefetch(iterable, qsize=8, poll_s=5.0, name="bench_prefetch"):
    """Run `iterable` on ONE producer thread (image decode), yield its items IN ORDER through a bounded queue.
    Same contract as bench_core.iter_frames(seek_thread): a producer exception is re-raised in the consumer; on early
    exit / close() the consumer stops the producer, drains the queue and JOINS the thread."""
    q = queue.Queue(maxsize=qsize)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def producer():
        try:
            for item in iterable:
                if stop.is_set():
                    return
                if not put((_ITEM, item)):
                    return
            put((_END, None))
        except BaseException as e:  # noqa: BLE001 -- re-raised in the consumer
            put((_ERR, e))

    th = threading.Thread(target=producer, name=name, daemon=True)
    th.start()
    try:
        while True:
            try:
                tag, item = q.get(timeout=poll_s)
            except queue.Empty:
                if not th.is_alive() and q.empty():
                    raise RuntimeError("prefetch producer exited without a sentinel")
                continue
            if tag is _END:
                return
            if tag is _ERR:
                raise item
            yield item
    finally:
        stop.set()
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        th.join()


def _load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


class _JpegSource:
    kind = "jpeg"
    fps = None

    def __init__(self, files, name):
        self.files = list(files)
        self.n = len(self.files)
        self.name = name
        self.warnings = []

    def __len__(self):
        return self.n

    def _gen(self):
        for k, p in enumerate(self.files):
            yield k, _load_rgb(p), {"file": os.path.basename(p)}

    def __iter__(self):
        return prefetch(self._gen())

    @staticmethod
    def mot_frame(fi):
        return fi + 1


class EdinburghFrames(_JpegSource):
    """datasets/edinburgh_pigs/decoded_frames/<clip>/video1/0000001.jpg .. 0000600.jpg, staged locally.
    fi = k  <->  file {k+1:07d}.jpg  <->  GT frameNumber k  <->  raw color.mp4 frame 3k (6 fps): effective 2 fps."""

    def __init__(self, clip_dir, expect_n=EDI_N_FRAMES):
        files = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
        want = [f"{k + 1:07d}.jpg" for k in range(len(files))]
        got = [os.path.basename(p) for p in files]
        if got != want:
            raise ValueError(f"{clip_dir}: frame files are not 0000001.jpg..{len(files):07d}.jpg contiguous")
        if expect_n is not None and len(files) != expect_n:
            raise ValueError(f"{clip_dir}: {len(files)} frames, expected {expect_n}")
        super().__init__(files, os.path.basename(os.path.dirname(os.path.abspath(clip_dir))))
        self.fps = EDI_FPS


class PigTrackFrames(_JpegSource):
    """datasets/pigtrack/sequences/<split>/<seq>/{img1/00000001.jpg.., seqinfo.ini, gt/gt.txt}, staged locally.
    n = the IMAGE COUNT."""

    def __init__(self, seq_dir):
        ini = configparser.ConfigParser()
        ini.optionxform = str
        if not ini.read(os.path.join(seq_dir, "seqinfo.ini")):
            raise FileNotFoundError(f"{seq_dir}/seqinfo.ini")
        s = ini["Sequence"]
        files = sorted(glob.glob(os.path.join(seq_dir, s.get("imDir", "img1"), "*" + s.get("imExt", ".jpg"))))
        ext = s.get("imExt", ".jpg")
        want = [f"{k + 1:08d}{ext}" for k in range(len(files))]
        if [os.path.basename(p) for p in files] != want:
            raise ValueError(f"{seq_dir}: images are not 00000001{ext}..{len(files):08d}{ext} contiguous")
        super().__init__(files, s.get("name", os.path.basename(seq_dir)))
        self.fps = float(s.get("frameRate"))
        self.width, self.height = int(s.get("imWidth")), int(s.get("imHeight"))
        self.night = int(s.get("night", "0"))
        self.pen = s.get("penID")
        self.seq_length_ini = int(s.get("seqLength"))
        if self.seq_length_ini != self.n:
            self.warnings.append(f"seqinfo seqLength {self.seq_length_ini} != {self.n} images: using the image count")
        self.gt_path = os.path.join(seq_dir, "gt", "gt.txt")


def run_with_precision(set_prec, get_prec, fn, inner="high", outer="highest"):
    """TF32 only inside fn (RF-DETR predict); 'highest' is restored on every path, then asserted."""
    set_prec(inner)
    try:
        return fn()
    finally:
        set_prec(outer)
        if get_prec() != outer:
            raise RuntimeError(f"float32 matmul precision is {get_prec()!r} after the detector, expected {outer!r}")


_DECODE_END, _DECODE_ERR = object(), object()


def iter_frames(cap, fidxs, mode="seek_thread", pos_prop=1, qsize=8, poll_s=5.0):
    """Yields (fidx, ok, frame) in fidx order.
    seek        -- cap.set(POS_FRAMES, f); cap.read() inline.
    seek_thread -- the same calls on one producer thread with a bounded queue; a producer exception is re-raised in the
                   consumer; on close() the consumer sets stop, drains the queue and joins the producer.
    sequential  -- grab() every native frame, retrieve() the wanted ones."""
    fidxs = list(fidxs)
    if mode == "seek":
        for f in fidxs:
            cap.set(pos_prop, f)
            ok, fr = cap.read()
            yield f, ok, fr
        return
    if mode == "sequential":
        want = set(fidxs)
        last = fidxs[-1] if fidxs else -1
        i = 0
        while i <= last:
            if not cap.grab():
                for f in fidxs:
                    if f >= i:
                        yield f, False, None
                return
            if i in want:
                ok, fr = cap.retrieve()
                yield i, ok, fr
            i += 1
        return
    if mode != "seek_thread":
        raise ValueError(f"decode mode {mode!r}")
    q = queue.Queue(maxsize=qsize)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def producer():
        try:
            for f in fidxs:
                if stop.is_set():
                    return
                cap.set(pos_prop, f)
                ok, fr = cap.read()
                if not put((f, ok, fr)):
                    return
            put((_DECODE_END, None, None))
        except BaseException as e:            # noqa: BLE001 -- re-raised in the consumer
            put((_DECODE_ERR, e, None))
    th = threading.Thread(target=producer, name="bench_decode", daemon=True)
    th.start()
    try:
        while True:
            try:
                item = q.get(timeout=poll_s)
            except queue.Empty:
                if not th.is_alive() and q.empty():
                    raise RuntimeError("decode producer thread exited without a sentinel")
                continue
            if item[0] is _DECODE_END:
                return
            if item[0] is _DECODE_ERR:
                raise item[1]
            yield item
    finally:
        stop.set()
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        th.join()


class CalfWindowFrames:
    """A calf window decoded with cv2 from the locally staged
    clip, native fps from the container (25.0 if unreported), stride = max(1, round(native / 5)), frame indices on the
    clip's own 0, stride, 2*stride, ... grid (the grid the calf masks are keyed on), iter_frames
    (seek_thread), BGR -> RGB. An unreadable frame RAISES (the bench must keep fi contiguous)."""
    kind = "video"

    def __init__(self, mp4_local, start_fidx, n_frames, *, track_fps=CALF_TRACK_FPS, decode="seek_thread",
                 cap=None, name=None):
        if cap is None:
            import cv2
            cap = cv2.VideoCapture(mp4_local)
            if not cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture cannot open {mp4_local}")
        self.cap = cap
        native = cap.get(CV_CAP_PROP_FPS) or 25.0
        total = int(cap.get(CV_CAP_PROP_FRAME_COUNT) or 0)
        self.native_fps = float(native)
        self.stride = max(1, round(native / track_fps))
        if start_fidx % self.stride:
            raise ValueError(f"window start {start_fidx} is not on the clip's {self.stride}-frame grid")
        end = start_fidx + n_frames * self.stride
        if total and end > total:
            raise ValueError(f"window [{start_fidx}, {end}) crosses the clip end ({total} frames)")
        self.fidxs = list(range(start_fidx, end, self.stride))
        self.n = len(self.fidxs)
        self.fps = float(track_fps)
        self.decode = decode
        self.name = name or os.path.basename(mp4_local or "window")
        self.warnings = []

    def __len__(self):
        return self.n

    def __iter__(self):
        from PIL import Image
        it = iter_frames(self.cap, self.fidxs, mode=self.decode, pos_prop=CV_CAP_PROP_POS_FRAMES)
        try:
            for k, (f, ok, fr) in enumerate(it):
                if not ok or fr is None:
                    raise BenchAssertError(f"{self.name}: native frame {f} unreadable (window k={k})")
                yield k, Image.fromarray(np.ascontiguousarray(fr[:, :, ::-1])), {"native_fidx": int(f)}
        finally:
            it.close()

    def close(self):
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def mot_frame(fi):
        return fi + 1


# =====================================================================================================================
# ground truth readers + seeded run prompt plans
# =====================================================================================================================
@dataclass
class Prompt:
    """One prompt: frame (0-based bench index), obj_id (int >= 0), EITHER box [x1,y1,x2,y2] in original pixels OR a
    2-D bool mask. score / source are bookkeeping (detector run detections carry the detector score)."""
    frame: int
    obj_id: int
    box: Optional[tuple] = None
    mask: Optional[np.ndarray] = None
    score: Optional[float] = None
    source: str = "gt"

    @property
    def kind(self):
        return "mask" if self.mask is not None else "box"


def read_mot_txt(path):
    """MOT17 rows -> float64 array [R, 10] (frame, id, x, y, w, h, conf, class, vis, z); missing columns = -1."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            v = [float(x) for x in ln.replace(" ", "").split(",")]
            if len(v) < 6:
                raise ValueError(f"{path}: MOT row with {len(v)} fields: {ln!r}")
            rows.append((v + [-1.0] * 10)[:10])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 10)


def mot_row_valid(row):
    """Rows TrackEval's MOT17 preprocessing evaluates: int(conf) != 0 (TrackEval casts the conf column to int, so a
    fractional conf < 1 is zero-marked and ignored, exactly like conf == 0). Class/visibility are recorded by the J2
    census; any non-default value found there is an amendment, not a silent filter."""
    return int(row[6]) != 0


def protocol_a_from_mot(gt, valid=mot_row_valid):
    """seeded run: each GT id prompted EXACTLY ONCE, with its GT box, at its FIRST evaluated frame.
    Returns (prompts sorted by (frame, id), census). Bench frame = MOT frame - 1; box = [x, y, x+w, y+h]."""
    first = {}
    for r in gt:
        if not valid(r):
            continue
        oid, fr = int(r[1]), int(r[0])
        if oid < 0:
            raise ValueError(f"negative GT id {oid}")
        if oid not in first or fr < first[oid][0]:
            first[oid] = (fr, r)
    prompts = [Prompt(frame=fr - 1, obj_id=oid, box=(float(r[2]), float(r[3]), float(r[2] + r[4]), float(r[3] + r[5])))
               for oid, (fr, r) in first.items()]
    prompts.sort(key=lambda p: (p.frame, p.obj_id))
    f_min = min((p.frame for p in prompts), default=0)
    late = {p.obj_id for p in prompts if p.frame > f_min}
    n_valid = int(sum(1 for r in gt if valid(r)))
    n_late_boxes = int(sum(1 for r in gt if valid(r) and int(r[1]) in late))
    census = {"n_ids": len(prompts), "first_frame": f_min, "n_late_ids": len(late),
              "late_box_share": (n_late_boxes / n_valid) if n_valid else 0.0, "n_valid_rows": n_valid,
              "n_ignored_rows": int(len(gt) - n_valid),
              "conf_values": sorted({float(x) for x in gt[:, 6]}) if len(gt) else [],
              "class_values": sorted({float(x) for x in gt[:, 7]}) if len(gt) else []}
    return prompts, census


def read_edinburgh_keyframes(path_or_obj):
    """annotated/<YYYY_MM_DD>/<NNNNNN>/output.json -> {int id: [(frameNumber, x, y, w, h), ...]} (sorted)."""
    d = path_or_obj
    if not isinstance(d, dict):
        with open(path_or_obj, encoding="utf-8") as f:
            d = json.load(f)
    out = {}
    for o in d["objects"]:
        kf = []
        for fr in o["frames"]:
            if fr.get("visible", True) is not True:
                raise NotImplementedError("an Edinburgh keyframe with visible != true: the protocol has no rule for it")
            b = fr["bbox"]
            kf.append((int(fr["frameNumber"]), float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"])))
        kf.sort()
        if len({k[0] for k in kf}) != len(kf):
            raise ValueError(f"object {o['id']}: duplicate frameNumber")
        out[int(o["id"])] = kf
    return out


def edinburgh_gt_rows(keyframes, n_frames=EDI_N_FRAMES, interpolate=True):
    """Per-frame GT: boxes LINEARLY INTERPOLATED between keyframes (PIGDATA: 'Bounding boxes are interpolated'),
    held after the last keyframe to n_frames-1, absent before the first. interpolate=False = the old hold parser (the
    J3(c) bridge row only). Returns float64 [R, 6] (frame0, id, x, y, w, h) sorted by (frame, id)."""
    rows = []
    for oid, kf in sorted(keyframes.items()):
        for (fa, *ba), (fb, *bb) in zip(kf, kf[1:]):
            for f in range(fa, min(fb, n_frames)):
                if interpolate:
                    t = (f - fa) / (fb - fa)
                    box = [a + t * (b - a) for a, b in zip(ba, bb)]
                else:
                    box = list(ba)
                rows.append([f, oid] + box)
        last = kf[-1]
        for f in range(last[0], n_frames):
            rows.append([f, oid] + list(last[1:]))
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    if len(a):
        a = a[np.lexsort((a[:, 1], a[:, 0]))]
    return a


def protocol_a_edinburgh(keyframes):
    """seeded run on Edinburgh = the old protocol: every pig's FIRST keyframe box (all at frame 0, 80/80 equal to the
    BoundingBoxes_*.txt seeds) as [int(x), int(y), int(x+w), int(y+h)] (= the old load_bounding_box_prompts)."""
    ps = []
    for oid, kf in sorted(keyframes.items()):
        f, x, y, w, h = kf[0]
        ps.append(Prompt(frame=f, obj_id=oid, box=(int(x), int(y), int(x + w), int(y + h))))
    ps.sort(key=lambda p: (p.frame, p.obj_id))
    return ps


def read_bbox_prompts_txt(path):
    """VERBATIM parse of the old initial_prompts/BoundingBoxes_<clip>.txt ('id k: x,y,w,h') -> {int k: [x1,y1,x2,y2]}
   ."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            name, coords = line.split(':', 1)
            x, y, w, h = map(float, coords.strip().split(','))
            out[int(name.strip().split()[-1])] = [int(x), int(y), int(x + w), int(y + h)]
    return out


def prompts_from_masks(masks_by_id, frame=0):
    """Calf windows: GT chain masks at window start -> mask prompts (a python LIST of 2-D masks at add time)."""
    ps = []
    for oid in sorted(masks_by_id):
        m = np.asarray(masks_by_id[oid]).astype(bool)
        ps.append(Prompt(frame=int(frame), obj_id=int(oid), mask=m))
    return ps


def schedule_prompts(prompts):
    """Validate a prompt plan and group it by frame -> OrderedDict {frame: [Prompt]}. Rules: every obj_id is prompted
    ONCE (no re-prompts); all prompts of one frame share one kind (ONE add_inputs call per frame: a second call on the
    same frame REPLACES session.obj_with_new_inputs and the first group would be tracked unprompted); boxes positive."""
    seen, by_f = set(), collections.OrderedDict()
    for p in sorted(prompts, key=lambda p: (p.frame, p.obj_id)):
        if p.obj_id in seen:
            raise ValueError(f"obj_id {p.obj_id} prompted twice (seeded and detector runs never re-prompt)")
        seen.add(p.obj_id)
        if (p.box is None) == (p.mask is None):
            raise ValueError(f"obj_id {p.obj_id}: exactly one of box / mask")
        if p.box is not None:
            x1, y1, x2, y2 = p.box
            if not (x2 > x1 and y2 > y1):
                raise ValueError(f"obj_id {p.obj_id}: degenerate box {p.box}")
        else:
            if p.mask.ndim != 2 or not p.mask.any():
                raise ValueError(f"obj_id {p.obj_id}: mask prompt must be a non-empty 2-D mask")
        by_f.setdefault(int(p.frame), []).append(p)
    for f, ps in by_f.items():
        if len({p.kind for p in ps}) != 1:
            raise ValueError(f"frame {f}: box and mask prompts mixed (one add_inputs call per frame)")
    return by_f


# =====================================================================================================================
# assignment + box geometry (deterministic; no scipy so every environment gives the same tie-breaking)
# =====================================================================================================================
def linear_assignment(cost):
    """Minimum-cost assignment of a rectangular cost matrix (Hungarian / Jonker-Volgenant potentials, O(n^2 m)).
    Returns (rows, cols) sorted by row; min(n, m) pairs. Deterministic tie-breaking (first index)."""
    C = np.asarray(cost, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("cost must be 2-D")
    n0, m0 = C.shape
    if n0 == 0 or m0 == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if not np.isfinite(C).all():
        raise ValueError("cost must be finite")
    tr = n0 > m0
    if tr:
        C = C.T
    n, m = C.shape
    INF = np.inf
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = C[i0 - 1] - u[i0] - v[1:]
            upd = free & (cur < minv[1:])
            mv, wy = minv[1:], way[1:]
            mv[upd] = cur[upd]
            wy[upd] = j0
            cand = np.where(free, mv, INF)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            uj = np.flatnonzero(used)
            u[p[uj]] += delta
            v[uj] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    cols = np.flatnonzero(p[1:] != 0)
    rows = p[1:][cols] - 1
    if tr:
        rows, cols = cols, rows
    o = np.argsort(rows, kind="stable")
    return rows[o].astype(np.int64), cols[o].astype(np.int64)


def _box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def box_iou(a, b):
    """Continuous [x1,y1,x2,y2] IoU (mask boxes enter as their pixel extent [x0, y0, x1+1, y1+1])."""
    i = _box_inter(a, b)
    u = _box_area(a) + _box_area(b) - i
    return i / u if u > 0 else 0.0


def box_ios(a, b):
    """Intersection over the SMALLER box area (0 when either is empty)."""
    m = min(_box_area(a), _box_area(b))
    return _box_inter(a, b) / m if m > 0 else 0.0


def iou_matrix(A, B):
    M = np.zeros((len(A), len(B)), dtype=np.float64)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            M[i, j] = box_iou(a, b)
    return M


def match_by_iou(A, B, thr):
    """Hungarian on 1 - IoU, keep pairs with IoU >= thr (box IoU).
    Returns [(i, j, iou)] sorted by i."""
    if not len(A) or not len(B):
        return []
    M = iou_matrix(A, B)
    r, c = linear_assignment(1.0 - M)
    return [(int(i), int(j), float(M[i, j])) for i, j in zip(r, c) if M[i, j] >= thr]


# =====================================================================================================================
# detector run state machine (pure; the loop applies its births and its emit filter)
# =====================================================================================================================
class ProtocolD:
    """Detector-driven tracking rules (A11; ). Pass frames: fi % period == 0, period = max(1,
    round(fps * 1.0)). At a pass on frame fi the detector runs on frame fi BEFORE the tracker's forward on fi, and is
    compared with the track boxes emitted at the latest processed frame (fi - 1).
      first pass (frame 0): every detection with score >= tau is a seed, greedily by score, dropping one whose IoS
                            with an already-kept seed is >= 0.3;
      later passes:         detections (score >= tau) <-> tracks present at fi-1: Hungarian on 1 - IoU, matched iff
                            IoU >= 0.5 (matched detections do NOTHING: no re-prompts); unmatched detections link to the
                            previous pass's pending candidates by the same rule (count + 1) or start a new candidate
                            (count 1); candidates not seen again are dropped (CONSECUTIVE passes);
      birth:                candidate with count >= 2 and IoS < 0.3 with the reference box of EVERY alive track
                            (current box, else its last box) and with every birth already made in this pass; greedy
                            by score; prompt = the current detection box at frame fi;
      death:                absent (score <= 0 or empty mask) on more than round(3.0 * fps) consecutive frames -> dead:
                            never output again, never matched, never a reference (it stays in the session: no removal).
    Births get new ids next_id, next_id + 1, ... (seeds first)."""

    def __init__(self, fps, tau, *, pass_period_s=D_PASS_PERIOD_S, match_iou=D_MATCH_IOU, birth_ios=D_BIRTH_IOS,
                 birth_passes=D_BIRTH_PASSES, death_s=D_DEATH_S, first_id=0):
        if not (fps and fps > 0):
            raise ValueError("fps must be > 0")
        self.fps, self.tau = float(fps), float(tau)
        self.period = max(1, int(round(self.fps * pass_period_s)))
        self.death_frames = int(round(death_s * self.fps))
        self.match_iou, self.birth_ios, self.birth_passes = float(match_iou), float(birth_ios), int(birth_passes)
        self.params = {"fps": self.fps, "tau": self.tau, "pass_period_s": pass_period_s, "period_frames": self.period,
                       "match_iou": self.match_iou, "birth_ios": self.birth_ios, "birth_passes": self.birth_passes,
                       "death_s": death_s, "death_frames": self.death_frames, "first_id": int(first_id)}
        self.tracks = collections.OrderedDict()   # id -> {"cur": box|None, "last": box, "absent": int, "born": fi}
        self.dead = {}                            # id -> frame of death
        self.pending = []                         # [{"box", "score", "count", "first"}]
        self.next_id = int(first_id)
        self.seeded = False
        self.events = []
        self.n_passes = 0

    def is_pass(self, fi):
        return fi % self.period == 0

    def _birth(self, fi, box, score, kind):
        oid = self.next_id
        self.next_id += 1
        self.tracks[oid] = {"cur": None, "last": tuple(box), "absent": 0, "born": fi}
        self.events.append({"type": kind, "fi": fi, "id": oid, "score": round(score, 4),
                            "box": [round(v, 2) for v in box]})
        return Prompt(frame=fi, obj_id=oid, box=tuple(box), score=score, source=f"protD_{kind}")

    def on_pass(self, fi, dets):
        """dets: [(box_xyxy, score)] from the detector on frame fi. Returns the birth Prompts for frame fi."""
        self.n_passes += 1
        D = []
        for b, s in dets:
            b = tuple(float(v) for v in b)
            if not (b[2] > b[0] and b[3] > b[1]) or not all(math.isfinite(v) for v in b) or not math.isfinite(s):
                raise BenchAssertError(f"detector output contract: box {b} score {s}")
            if float(s) >= self.tau:
                D.append((b, float(s)))
        D.sort(key=lambda z: -z[1])          # stable: equal scores keep detector order
        if not self.seeded:
            self.seeded = True
            out, kept = [], []
            for b, s in D:
                if all(box_ios(b, k) < self.birth_ios for k in kept):
                    kept.append(b)
                    out.append(self._birth(fi, b, s, "seed"))
                else:
                    self.events.append({"type": "seed_dup", "fi": fi, "score": round(s, 4)})
            return out
        live = [(tid, t["cur"]) for tid, t in self.tracks.items() if t["cur"] is not None]
        matched = {i for i, _j, _v in match_by_iou([b for b, _ in D], [b for _, b in live], self.match_iou)}
        U = [D[i] for i in range(len(D)) if i not in matched]
        new_pending, linked = [], set()
        for i, j, _v in match_by_iou([b for b, _ in U], [p["box"] for p in self.pending], self.match_iou):
            p = self.pending[j]
            new_pending.append({"box": U[i][0], "score": U[i][1], "count": p["count"] + 1, "first": p["first"]})
            linked.add(i)
        for i, (b, s) in enumerate(U):
            if i not in linked:
                new_pending.append({"box": b, "score": s, "count": 1, "first": fi})
        refs = [t["cur"] if t["cur"] is not None else t["last"] for t in self.tracks.values()]
        births, born = [], []
        for p in sorted([q for q in new_pending if q["count"] >= self.birth_passes], key=lambda q: -q["score"]):
            worst = max([box_ios(p["box"], r) for r in refs + born], default=0.0)
            if worst < self.birth_ios:
                births.append(self._birth(fi, p["box"], p["score"], "birth"))
                born.append(p["box"])
                p["born"] = True
            else:
                self.events.append({"type": "birth_blocked", "fi": fi, "ios": round(worst, 4), "count": p["count"]})
        self.pending = [q for q in new_pending if not q.get("born")]
        return births

    def on_output(self, fi, present):
        """present: {obj_id: box_xyxy} of the objects the output rule emitted on frame fi (any session object).
        Updates presence / deaths; returns the ids allowed in the output (alive AND present)."""
        unknown = [k for k in present if k not in self.tracks and k not in self.dead]
        if unknown:
            raise BenchAssertError(f"detector run: output for ids never born: {unknown[:5]}")
        for tid in list(self.tracks):
            t = self.tracks[tid]
            b = present.get(tid)
            if b is not None:
                t["cur"], t["last"], t["absent"] = tuple(b), tuple(b), 0
            else:
                t["cur"] = None
                t["absent"] += 1
                if t["absent"] > self.death_frames:
                    self.dead[tid] = fi
                    del self.tracks[tid]
                    self.events.append({"type": "death", "fi": fi, "id": tid})
        return {tid for tid in self.tracks if tid in present}

    def summary(self):
        c = collections.Counter(e["type"] for e in self.events)
        return {"params": self.params, "passes": self.n_passes, "seeds": c["seed"], "births": c["birth"],
                "deaths": c["death"], "birth_blocked": c["birth_blocked"], "seed_dup": c["seed_dup"],
                "alive_end": len(self.tracks), "pending_end": len(self.pending)}


class DetectionCache:
    """Detector outputs at the pass frames of ONE sequence, computed ONCE at DET_PROBE_THRESHOLD and shared by every
    arm and every tau (byte-identical detections for all arms). JSON: {"meta": {...}, "dets": {"fi": [[x1,y1,x2,y2,s]]}}."""

    def __init__(self, dets=None, meta=None):
        self.dets = {int(k): [(tuple(float(x) for x in d[:4]), float(d[4])) for d in v] for k, v in (dets or {}).items()}
        self.meta = dict(meta or {})

    @classmethod
    def build(cls, detector, frames, period, meta=None):
        dets = {}
        for fi, pil, _m in frames:
            if fi % period == 0:
                dets[fi] = [(tuple(float(x) for x in b), float(s)) for b, s in detector(fi, pil)]
        c = cls(meta=dict(meta or {}, period=period, probe_threshold=DET_PROBE_THRESHOLD))
        c.dets = dets
        return c

    def to_json(self):
        return {"meta": self.meta, "dets": {str(k): [list(b) + [s] for b, s in v] for k, v in sorted(self.dets.items())}}

    @classmethod
    def from_json(cls, obj):
        return cls(dets={k: v for k, v in obj["dets"].items()}, meta=obj.get("meta"))

    def detector(self):
        def det(fi, _pil):
            if fi not in self.dets:
                raise BenchAssertError(f"detection cache has no pass frame {fi}")
            return list(self.dets[fi])
        return det


def load_rfdetr_detector(ckpt_local, *, resolution, class_names=None, threshold=DET_PROBE_THRESHOLD):
    """RF-DETR (rfdetr==1.10.1) as a detector run detector: callable (fi, pil) -> [(xyxy, score)] keeping score >
    threshold (strict, predict()) and class_id < num_classes. TF32 only inside predict, 'highest'
    restored and asserted on every path (bench_core.run_with_precision). resolution is PINNED (strip_checkpoint drops
    model_config). Boxes = the detector's own xyxy (also for the calf seg model)."""
    _require_torch()
    os.environ.setdefault("RF_HOME", "/local_disk0/rf_home")
    from rfdetr import from_checkpoint
    torch.set_float32_matmul_precision(MATMUL_PRECISION)        # import side effect 'high' -> restored
    rf = from_checkpoint(ckpt_local, resolution=int(resolution))
    nc = int(rf.model_config.num_classes)
    if class_names is not None and list(rf.class_names) != list(class_names):
        raise RuntimeError(f"class_names {rf.class_names} != {class_names}")

    def det(_fi, pil):
        arr = np.asarray(pil)

        def _pred():
            with torch.inference_mode():
                return rf.predict(arr, threshold=threshold, include_source_image=False)
        d = run_with_precision(torch.set_float32_matmul_precision, torch.get_float32_matmul_precision, _pred,
                                    outer=MATMUL_PRECISION)
        if not len(d):
            return []
        k = d.class_id < nc
        return [(tuple(float(v) for v in b), float(s)) for b, s in zip(d.xyxy[k], d.confidence[k])]
    det.info = {"ckpt_sha256": sha256_file(ckpt_local), "resolution": int(resolution), "num_classes": nc,
                "class_names": list(rf.class_names), "threshold": threshold}
    return det


# =====================================================================================================================
# array adapters (torch on the GPU in production, numpy) -- the ONLY backend-specific code
# =====================================================================================================================
def _is_t(x):
    return TORCH_OK and isinstance(x, torch.Tensor)


def _any_ax(x, ax):
    return x.any(dim=ax) if _is_t(x) else x.any(axis=ax)


def _as_i32(x):
    return x.to(torch.int32) if _is_t(x) else x.astype(np.int32)


def _argmax1(x):
    return x.argmax(dim=1) if _is_t(x) else x.argmax(axis=1)


def _flip1(x):
    return torch.flip(x, dims=[1]) if _is_t(x) else x[:, ::-1]


def _area2(b):
    return b.reshape(b.shape[0], -1).sum(dim=1) if _is_t(b) else b.reshape(b.shape[0], -1).sum(axis=1)


def _stack_i64(cols):
    if _is_t(cols[0]):
        return torch.stack([c.to(torch.int64) for c in cols], dim=1)
    return np.stack([np.asarray(c).astype(np.int64) for c in cols], axis=1)


def to_numpy(x):
    if _is_t(x):
        return x.detach().float().cpu().numpy() if x.is_floating_point() else x.detach().cpu().numpy()
    return np.asarray(x)


def _flat_scores(s, n):
    return s.reshape(n, -1)[:, 0]


def all_finite(x):
    if _is_t(x):
        return bool(torch.isfinite(x).all())
    return bool(np.isfinite(np.asarray(x, dtype=np.float64)).all())


def std_of(x):
    if _is_t(x):
        return float(x.detach().float().std()) if x.numel() > 1 else 0.0
    a = np.asarray(x, dtype=np.float64)
    return float(a.std()) if a.size > 1 else 0.0


def assert_finite(x, what):
    if not all_finite(x):
        raise BenchAssertError(f"non-finite values in {what} (dtype unsafe? fp16 trap) -- run INVALID")


def assert_spread(x, what, min_std=MIN_STD):
    s = std_of(x)
    if not (s > min_std):
        raise BenchAssertError(f"collapsed {what}: std {s:.3g} <= {min_std} -- run INVALID")


def _first_tensor(o):
    if o is None:
        return None
    if _is_t(o) or isinstance(o, np.ndarray):
        return o
    lhs = getattr(o, "last_hidden_state", None)
    if lhs is not None:                            # EdgeTAM's timm backbone puts a LIST of feature maps here
        t = _first_tensor(lhs)
        if t is not None:
            return t
    if isinstance(o, (tuple, list)):
        for e in o:
            t = _first_tensor(e)
            if t is not None:
                return t
    if hasattr(o, "to_tuple"):
        return _first_tensor(o.to_tuple())
    return None


# =====================================================================================================================
# output rule (A3): emit iff object_score_logits > 0 AND the binarised mask (logit > 0) is non-empty; tight box of the
# mask computed on the device; inclusive pixel indices x0..x1, y0..y1; MOT w = x1 - x0 + 1, h = y1 - y0 + 1.
# =====================================================================================================================
def mask_logits_2d(m):
    """post_process_masks output [N, 1, H, W] or [N, H, W] -> [N, H, W] (one mask per object; >1 channel raises)."""
    if m.ndim == 4:
        if m.shape[1] != 1:
            raise BenchAssertError(f"{m.shape[1]} masks per object (multimask output) -- the protocol needs one")
        return m[:, 0]
    if m.ndim != 3:
        raise BenchAssertError(f"mask logits of shape {tuple(m.shape)}")
    return m


def output_rule(mask_logits, score_logits):
    """mask_logits [N,(1,)H,W] (original resolution, logits), score_logits [N, ...]. Returns (packed int64 [N, 6] =
    (emit, x0, y0, x1, y1, area) on the HOST, scores float64 [N], binary masks [N, H, W] on the device).
    Two host transfers per frame (packed, scores); boxes/areas never leave the device as masks."""
    m = mask_logits_2d(mask_logits)
    n = m.shape[0]
    b = m > 0
    rows = _any_ax(b, 2)                                   # [N, H]
    cols = _any_ax(b, 1)                                   # [N, W]
    nonempty = _any_ax(rows, 1)
    H, W = rows.shape[1], cols.shape[1]
    y0 = _argmax1(_as_i32(rows))
    y1 = (H - 1) - _argmax1(_as_i32(_flip1(rows)))
    x0 = _argmax1(_as_i32(cols))
    x1 = (W - 1) - _argmax1(_as_i32(_flip1(cols)))
    area = _area2(b)
    s = _flat_scores(score_logits, n)
    emit = nonempty & (s > 0)
    if MIN_AREA_PX > 0:
        emit = emit & (area >= MIN_AREA_PX)
    packed = to_numpy(_stack_i64([emit, x0, y0, x1, y1, area])).astype(np.int64)
    scores = to_numpy(s).astype(np.float64).reshape(n)
    return packed, scores, b


@dataclass
class Row:
    k: int            # object index in the session (= insertion order)
    obj_id: int
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    score: float

    @property
    def box_cont(self):
        """Continuous pixel-extent box for IoU against detector boxes."""
        return (float(self.x0), float(self.y0), float(self.x1 + 1), float(self.y1 + 1))


def rows_from_packed(packed, scores, obj_ids):
    if len(obj_ids) != packed.shape[0]:
        raise BenchAssertError(f"{packed.shape[0]} outputs for {len(obj_ids)} session objects")
    out = []
    for k, oid in enumerate(obj_ids):
        e, x0, y0, x1, y1, a = (int(v) for v in packed[k])
        if e:
            out.append(Row(k, int(oid), x0, y0, x1, y1, a, float(scores[k])))
    return out


def pairwise_overlap_max(b, idx):
    """Diagnostic (not a metric): max pairwise mask IoS and IoU among the emitted objects idx (fp32 matmul: exact
    integer counts below 2^24 px). Returns (max_ios, max_iou) or (0, 0) with < 2 objects."""
    if len(idx) < 2:
        return 0.0, 0.0
    if _is_t(b):
        f = b[list(idx)].reshape(len(idx), -1).to(torch.float32)
        inter = to_numpy(f @ f.T).astype(np.float64)
    else:
        f = b[list(idx)].reshape(len(idx), -1).astype(np.float32)
        inter = (f @ f.T).astype(np.float64)
    a = np.diag(inter).copy()
    iu = np.triu_indices(len(idx), 1)
    it = inter[iu]
    mn = np.minimum(a[iu[0]], a[iu[1]])
    un = a[iu[0]] + a[iu[1]] - it
    ios = np.where(mn > 0, it / np.maximum(mn, 1), 0.0)
    iou = np.where(un > 0, it / np.maximum(un, 1), 0.0)
    return float(ios.max()), float(iou.max())


# =====================================================================================================================
# writers
# =====================================================================================================================
class MotWriter:
    """MOTChallenge tracker txt: 'frame,id,x,y,w,h,conf,-1,-1,-1', frame = fi + 1, id = obj_id + 1 (TrackEval matches
    ids globally, so the offset is metric-neutral), conf = 1, boxes = integers (xy + xy_offset). Rows sorted by
    (frame, id). Written ONCE, atomically, to LOCAL disk (publish with publish_no_overwrite)."""

    def __init__(self, xy_offset=MOT_XY_OFFSET_DEFAULT):
        if xy_offset not in (0, 1):
            raise ValueError("xy_offset must be 0 or 1")
        self.xy_offset = int(xy_offset)
        self.rows = []

    def add(self, fi, rows):
        for r in rows:
            self.rows.append((int(fi) + 1, int(r.obj_id) + 1, r.x0 + self.xy_offset, r.y0 + self.xy_offset,
                              r.x1 - r.x0 + 1, r.y1 - r.y0 + 1))

    def __len__(self):
        return len(self.rows)

    def lines(self):
        rows = sorted(self.rows)
        for a, b in zip(rows, rows[1:]):
            if a[:2] == b[:2]:
                raise BenchAssertError(f"duplicate MOT row frame {a[0]} id {a[1]}")
        return [f"{f},{i},{x},{y},{w},{h},1,-1,-1,-1" for f, i, x, y, w, h in rows]

    def write(self, path):
        ls = self.lines()
        return write_text_atomic(path, "\n".join(ls) + ("\n" if ls else ""))


# VERBATIM bench_core.rle_encode / rle_decode: column-major COCO-style counts
def rle_encode(mask):
    h, w = mask.shape
    flat = np.asfortranarray(mask).reshape(-1, order="F")
    idx = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate(([0], idx, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0]: counts = [0] + counts
    return {"size": [h, w], "counts": counts}


def rle_decode(r):
    h, w = r["size"]
    c = np.asarray(r["counts"], dtype=np.int64)
    vals = np.zeros(len(c), bool); vals[1::2] = True
    return np.repeat(vals, c).reshape((h, w), order="F")


class RleWriter:
    """Per-frame masks of the EMITTED objects as gzip JSON lines {"fi", "fidx", "objs": {"<obj_id>": rle}} (calves:
    always -- identity scoring needs masks; pigs: optional). Local file; publish with publish_no_overwrite."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.tmp = f"{path}.tmp_{os.getpid()}"
        self.f = gzip.open(self.tmp, "wt", encoding="utf-8")
        self.n = 0

    def add(self, fi, fidx, masks_by_id):
        rec = {"fi": int(fi), "fidx": int(fidx),
               "objs": {str(k): rle_encode(np.asarray(m, dtype=bool)) for k, m in sorted(masks_by_id.items())}}
        self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.n += 1

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None
            os.replace(self.tmp, self.path)
        return self.path


def read_rle_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


# =====================================================================================================================
# telemetry: CUDA-event module timings through forward hooks, CPU spans, memory trace, growth slope
# =====================================================================================================================
class _CpuEvent:
    """Stand-in for torch.cuda.Event(enable_timing=True) without CUDA (CPU runs)."""

    def __init__(self):
        self.t = None

    def record(self):
        self.t = time.perf_counter()

    def elapsed_time(self, end):
        return (end.t - self.t) * 1000.0


def mem_snapshot():
    """MiB: allocated / reserved / peak allocated / peak reserved (None without CUDA)."""
    if not (TORCH_OK and torch.cuda.is_available()):
        return None
    torch.cuda.synchronize()
    m = 2 ** 20
    return {"alloc": torch.cuda.memory_allocated() / m, "reserved": torch.cuda.memory_reserved() / m,
            "peak_alloc": torch.cuda.max_memory_allocated() / m, "peak_reserved": torch.cuda.max_memory_reserved() / m}


def growth_slope(fis, mib, warmup=WARMUP_FRAMES):
    """Memory growth in MiB/frame from post-prune samples: OLS slope and the median of step slopes (robust to a
    delayed free), on samples with fi >= warmup. None when fewer than 3 samples."""
    f = np.asarray(fis, dtype=np.float64)
    y = np.asarray(mib, dtype=np.float64)
    k = f >= warmup
    f, y = f[k], y[k]
    if f.size < 3:
        return {"ols_mib_per_frame": None, "median_step_mib_per_frame": None, "n": int(f.size)}
    ols = float(np.polyfit(f, y, 1)[0])
    steps = np.diff(y) / np.diff(f)
    return {"ols_mib_per_frame": ols, "median_step_mib_per_frame": float(np.median(steps)), "n": int(f.size),
            "frames": [int(f[0]), int(f[-1])]}


class Telemetry:
    """Per-frame timings: GPU spans (CUDA events; resolved at prune boundaries with ONE synchronize per 25 frames),
    module timings via forward pre/post hooks on MODULE_GROUPS (sum over calls within a frame: SAM2-family memory
    attention and mask decoder run once PER OBJECT), CPU spans (perf_counter), and the memory trace sampled after
    every prune. enabled=False makes every call a no-op."""

    def __init__(self, model=None, modules=MODULE_GROUPS, *, enabled=True, use_cuda=None, warmup=WARMUP_FRAMES,
                 event_factory=None):
        self.enabled = enabled
        self.use_cuda = (TORCH_OK and torch.cuda.is_available()) if use_cuda is None else use_cuda
        self._ev = event_factory or ((lambda: torch.cuda.Event(enable_timing=True)) if self.use_cuda else _CpuEvent)
        self.warmup = warmup
        self.fi = -1
        self.pending = []
        self.gpu_ms = collections.defaultdict(dict)
        self.cpu_ms = collections.defaultdict(dict)
        self.mem = []
        self._open = collections.defaultdict(list)
        self.hooks = []
        self.attached = []
        if model is not None and enabled:
            self.attach(model, modules)

    def attach(self, model, names=MODULE_GROUPS):
        mods = dict(model.named_modules()) if hasattr(model, "named_modules") else {}
        for n in names:
            m = mods.get(n)
            if m is None:
                continue
            self.hooks.append(m.register_forward_pre_hook(lambda _m, _a, n=n: self._start(n)))
            self.hooks.append(m.register_forward_hook(lambda _m, _a, _o, n=n: self._stop(n)))
            self.attached.append(n)
        return self.attached

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def begin_frame(self, fi):
        self.fi = int(fi)

    def _start(self, n):
        if not self.enabled:
            return
        e = self._ev()
        e.record()
        self._open[n].append(e)

    def _stop(self, n):
        if not self.enabled or not self._open[n]:
            return
        s = self._open[n].pop()
        e = self._ev()
        e.record()
        self.pending.append((self.fi, n, s, e))

    @contextlib.contextmanager
    def span(self, n):
        self._start(n)
        try:
            yield
        finally:
            self._stop(n)

    @contextlib.contextmanager
    def cpu_span(self, n):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.enabled:
                d = self.cpu_ms[n]
                d[self.fi] = d.get(self.fi, 0.0) + (time.perf_counter() - t0) * 1000.0

    def flush(self):
        if not self.enabled or not self.pending:
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        for fi, n, s, e in self.pending:
            d = self.gpu_ms[n]
            d[fi] = d.get(fi, 0.0) + s.elapsed_time(e)
        self.pending = []

    def mem_sample(self, fi, sess=None, n_obj=None, extra=None):
        if not self.enabled:
            return
        snap = mem_snapshot() or {}
        rec = {"fi": int(fi), "n_obj": n_obj, "n_pf": n_processed_frames(sess), "n_noncond": n_noncond(sess)}
        rec.update({k: round(v, 3) for k, v in snap.items()})
        if extra:
            rec.update(extra)
        self.mem.append(rec)

    @staticmethod
    def _stats(d, warmup):
        v = np.asarray([ms for fi, ms in sorted(d.items()) if fi >= warmup], dtype=np.float64)
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
                "p75": float(np.percentile(v, 75)), "mean": float(v.mean()), "total": float(v.sum())}

    def summary(self):
        self.flush()
        out = {"warmup_frames": self.warmup, "gpu_ms": {}, "cpu_ms": {}, "modules_attached": list(self.attached),
               "clock": "cuda_events" if self.use_cuda else "perf_counter"}
        for n, d in sorted(self.gpu_ms.items()):
            out["gpu_ms"][n] = self._stats(d, self.warmup)
        for n, d in sorted(self.cpu_ms.items()):
            out["cpu_ms"][n] = self._stats(d, self.warmup)
        if "model" in self.gpu_ms:                      # 'other' = model total - sum of the hooked modules
            fr = [fi for fi in self.gpu_ms["model"] if fi >= self.warmup]
            mods = [n for n in self.attached if n in self.gpu_ms]
            oth = {fi: self.gpu_ms["model"][fi] - sum(self.gpu_ms[n].get(fi, 0.0) for n in mods) for fi in fr}
            out["gpu_ms"]["model_other"] = self._stats(oth, self.warmup)
        if self.mem:
            out["mem_samples"] = self.mem
            if self.mem[0].get("alloc") is not None:
                out["growth_alloc"] = growth_slope([m["fi"] for m in self.mem], [m["alloc"] for m in self.mem],
                                                   self.warmup)
        return out


# =====================================================================================================================
# session bookkeeping (identical session layout in Sam2Video / EdgeTamVideo / Sam3TrackerVideo, baselines §2)
# =====================================================================================================================
def n_processed_frames(sess):
    """processed_frames is None until the first forward; sess may be None (= bench_core.n_pf)."""
    return len(getattr(sess, "processed_frames", None) or {}) if sess is not None else 0


def n_noncond(sess):
    if sess is None:
        return 0
    d = getattr(sess, "output_dict_per_obj", None) or {}
    return int(sum(len(v.get("non_cond_frame_outputs", {})) for v in d.values()))


def prune_session(sess, keep_last_n=KEEP_LAST_N):
    """Prune (keep the newest keep_last_n non-conditioning outputs per object). Returns the
    number of entries deleted. Conditioning outputs (the prompt frames) are never touched."""
    deleted = 0
    for oi in range(sess.get_obj_num()):
        nc = sess.output_dict_per_obj[oi].get("non_cond_frame_outputs", {})
        if len(nc) > keep_last_n:
            for key in sorted(nc.keys())[:-keep_last_n]:
                del nc[key]
                deleted += 1
    return deleted


def pop_processed_frame(sess, fi):
    """bench_core.sam3_forward(pop=True): the frame's pixels are read only inside its own forward (TV:1910)."""
    (sess.processed_frames or {}).pop(fi, None)


# =====================================================================================================================
# session teardown. J0 saw the previous SAM 3 session stay allocated into the next sequence (about
# 1.2 GB on PigTrack T, 16.9 GB of the teacher session into the S-Edi run of the fidelity audit) although the loop had dropped its
# own reference and called gc.collect(): something outside the loop still referenced the session object. Dropping a
# reference therefore cannot be trusted to free the device memory. The session is RELEASED IN PLACE instead: every
# container it owns is emptied (per-object outputs, prompts, tracked-frame records, the vision-feature cache, pixels),
# so whatever still points at the session object keeps an empty shell alive, not the tensors. track_stream records
# whether the object outlived its teardown and, if so, who held it (describe_referrers), and the callers assert that
# allocated memory returns to the weights-only baseline between sequences (check_memory_returned).
# =====================================================================================================================
SESSION_CONTAINERS = ("output_dict_per_obj", "point_inputs_per_obj", "mask_inputs_per_obj", "frames_tracked_per_obj")
FRAME_CLEAR_EVERY = 250        # clear_finished_frames_under(the loop's frame) at the prune of every 250th frame
MEM_RETURN_TOL_MIB = 512.0     # first sequence: allocated after teardown may exceed the weights-only baseline by at most
#                                this much -- caches built lazily by the first forward and kept for the process (J0b A10:
#                                T +67 MiB = lru-cached position encodings + cuBLAS workspaces; S-Edi +209 MiB = the same +
#                                timm TinyViT's per-device attention-bias cache); one retained SAM 3 session is 0.5-17 GB
MEM_DRIFT_TOL_MIB = 64.0       # later sequences: at most this much above the level the first teardown left (the lazy
#                                caches do not grow; the tracker's lru position encoding varies with the object count by
#                                < 20 MiB). Validity checks of the harness, not frozen protocol values.


def _storage_key(x):
    """Identity of the memory a tensor / array keeps alive: (device, storage data_ptr) for torch (a VIEW keeps its whole
    base storage alive, so views of one base share a key), id of the base buffer for numpy."""
    if _is_t(x):
        st = x.untyped_storage()
        return ("t", str(x.device), int(st.data_ptr())), int(st.nbytes())
    if isinstance(x, np.ndarray):
        base = x
        while isinstance(base.base, np.ndarray):
            base = base.base
        return ("n", id(base)), int(base.nbytes)
    return None, 0


def _walk_tensors(obj, path, out, depth=0, seen=None):
    seen = set() if seen is None else seen
    if depth > 8 or id(obj) in seen:
        return
    if _is_t(obj) or isinstance(obj, np.ndarray):
        out.append((path, obj))
        return
    if isinstance(obj, dict):
        seen.add(id(obj))
        for k, v in obj.items():
            _walk_tensors(v, path + (k,), out, depth + 1, seen)
    elif isinstance(obj, (list, tuple)):
        seen.add(id(obj))
        for k, v in enumerate(obj):
            _walk_tensors(v, path + (k,), out, depth + 1, seen)


def _session_groups(sess):
    """(group name, container) pairs of everything a SAM-family session owns that can hold tensors."""
    groups = []
    od = getattr(sess, "output_dict_per_obj", None)
    if isinstance(od, dict):
        for oi, d in od.items():
            for kind in ("cond_frame_outputs", "non_cond_frame_outputs"):
                for fr, entry in (d.get(kind) or {}).items():
                    tag = "cond" if kind.startswith("cond") else "noncond"
                    if isinstance(entry, dict):
                        for k, v in entry.items():
                            groups.append((f"{tag}:{k}", v))
                    else:
                        groups.append((f"{tag}:?", entry))
    for name in ("point_inputs_per_obj", "mask_inputs_per_obj", "frames_tracked_per_obj"):
        d = getattr(sess, name, None)
        if d:
            groups.append((name, d))
    cache = getattr(sess, "cache", None)
    vf = getattr(cache, "_vision_features", None) if cache is not None else None
    if vf:
        groups.append(("cache:vision_features", vf))
    pf = getattr(sess, "processed_frames", None)
    if pf:
        groups.append(("processed_frames", pf))
    return groups


def session_tensor_bytes(sess):
    """Bytes of the UNIQUE storages a session keeps alive, attributed to the first group that holds them (a view keeps
    its whole base: e.g. every object's maskmem_features is a slice of ONE batched [HW, N_obj, C] tensor per frame).
    -> {"total_mib", "n_storages", "by_group_mib": {group: MiB}} -- the G1-4 growth accounting."""
    seen, by = set(), collections.OrderedDict()
    total = 0
    if sess is None:
        return {"total_mib": 0.0, "n_storages": 0, "by_group_mib": {}}
    for name, cont in _session_groups(sess):
        found = []
        _walk_tensors(cont, (), found)
        for _p, t in found:
            key, nb = _storage_key(t)
            if key is None or key in seen:
                continue
            seen.add(key)
            by[name] = by.get(name, 0) + nb
            total += nb
    m = float(2 ** 20)
    return {"total_mib": round(total / m, 3), "n_storages": len(seen),
            "by_group_mib": {k: round(v / m, 3) for k, v in by.items()}}


def _clear_in_place(obj, depth=0):
    n = 0
    if depth > 6:
        return n
    if isinstance(obj, dict):
        for v in list(obj.values()):
            if isinstance(v, (dict, list)):
                n += _clear_in_place(v, depth + 1)
        n += len(obj)
        obj.clear()
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                n += _clear_in_place(v, depth + 1)
        n += len(obj)
        obj.clear()
    return n


def release_session(sess):
    """Empty every container a SAM-family inference session owns, IN PLACE (nested dicts first, so a dict that someone
    else still references is emptied too): per-object cond / non-cond outputs, point / mask prompts, tracked-frame
    records, the vision-feature cache and the pixel store. The object itself is left an empty shell. Returns
    {"released_mib", "n_storages", "entries_cleared", "by_group_mib"} measured just before the release."""
    if sess is None:
        return None
    acct = session_tensor_bytes(sess)
    n = 0
    for name in SESSION_CONTAINERS:
        d = getattr(sess, name, None)
        if isinstance(d, (dict, list)):
            n += _clear_in_place(d)
    cache = getattr(sess, "cache", None)
    if cache is not None:
        vf = getattr(cache, "_vision_features", None)
        if isinstance(vf, dict):
            n += _clear_in_place(vf)
        clear_all = getattr(cache, "clear_all", None)
        if callable(clear_all):
            clear_all()
    pf = getattr(sess, "processed_frames", None)
    if isinstance(pf, dict):
        n += len(pf)
        pf.clear()
    lst = getattr(sess, "obj_with_new_inputs", None)
    if isinstance(lst, list):
        lst.clear()
    return {"released_mib": acct["total_mib"], "n_storages": acct["n_storages"], "entries_cleared": int(n),
            "by_group_mib": acct["by_group_mib"]}


def mem_allocated_mib():
    """torch.cuda.memory_allocated in MiB after a synchronize (None without CUDA)."""
    if not (TORCH_OK and torch.cuda.is_available()):
        return None
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 2 ** 20


def settle_memory():
    """gc.collect() + empty_cache() + synchronize -> allocated MiB (None without CUDA). Called between sequences only
   ."""
    gc.collect()
    if TORCH_OK and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mem_allocated_mib()


def check_memory_returned(baseline_mib, now_mib=None, *, tol_mib=MEM_RETURN_TOL_MIB, what="", raise_on_fail=True):
    """Allocated memory must be back at the weights-only baseline (+ tol_mib) between sequences; otherwise the previous
    session (or a piece of it) is still allocated and every peak of the next sequence is contaminated. now_mib=None
    measures (settle_memory). Without CUDA (pass both values) the check is skipped. Raises BenchAssertError
    on failure unless raise_on_fail=False. -> {"baseline_mib", "alloc_mib", "excess_mib", "tol_mib", "ok", "what"}."""
    now = settle_memory() if now_mib is None else float(now_mib)
    rec = {"what": what, "baseline_mib": None if baseline_mib is None else round(float(baseline_mib), 3),
           "alloc_mib": None if now is None else round(float(now), 3), "tol_mib": float(tol_mib)}
    if now is None or baseline_mib is None:
        rec.update(excess_mib=None, ok=None, note="no CUDA / no baseline: not checked")
        return rec
    rec["excess_mib"] = round(float(now) - float(baseline_mib), 3)
    rec["ok"] = bool(rec["excess_mib"] <= tol_mib)
    if not rec["ok"] and raise_on_fail:
        raise BenchAssertError(f"{what}: {rec['alloc_mib']:.0f} MiB allocated, {rec['excess_mib']:.0f} MiB above the "
                               f"weights-only baseline {rec['baseline_mib']:.0f} MiB (tolerance {tol_mib:.0f}): a previous "
                               "session is still allocated -- see telemetry teardown.holders")
    return rec


_CO_GEN = 0x20 | 0x80 | 0x200   # CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR: never clear a generator's frame


def _clear_frames(frames):
    out = collections.Counter()
    for f in frames:
        if f.f_code.co_flags & _CO_GEN:
            continue
        try:
            f.clear()
            out[f"{f.f_code.co_name} ({os.path.basename(f.f_code.co_filename)})"] += 1
        except RuntimeError:                      # still executing (e.g. our own frame): leave it
            pass
    return dict(out)


def clear_frames_referencing(obj):
    """Clear every FINISHED frame whose locals reference `obj` (frame.clear(); an executing frame raises and is
    skipped; generator / coroutine frames are never touched). One heap scan. -> {'func (file)': count}."""
    import types as _types
    return _clear_frames([r for r in gc.get_referrers(obj) if type(r) is _types.FrameType])


def clear_finished_frames_under(anchor):
    """J0b root cause. On Databricks a runtime thread (not this code) keeps {thread id: frame} snapshots -- the shape of
    sys._current_frames() -- and with them FINISHED frames of earlier forwards: Sam3TrackerVideoModel.forward,
    _run_single_frame_inference, _prepare_memory_conditioned_features, the memory-attention / RoPE frames below them,
    Arm.forward. Their locals keep the session and that frame's tensors alive (fp32 memory keys [1,1,36288,256], high-res
    masks, the fp32 pixel frame, ...), so allocated memory steps up whenever a snapshot lands inside the tracker, and
    the session outlives the loop. Clear every finished frame whose f_back chain reaches `anchor` (the executing
    track_stream frame): everything the loop called and that has returned. Executing frames, generator / coroutine
    frames and frames of other call stacks are untouched. Output-neutral. One heap scan. -> {'func (file)': count}."""
    import types as _types
    aid = id(anchor)
    targets = []
    for o in gc.get_objects():
        if type(o) is not _types.FrameType or o is anchor:     # type(): no __class__ lookup on arbitrary objects
            continue
        g, depth = o.f_back, 0
        while g is not None and depth < 256:
            if id(g) == aid:
                targets.append(o)
                break
            g, depth = g.f_back, depth + 1
    out = _clear_frames(targets)
    targets = None
    return out


TEARDOWN_DIAG = {"left": 3}   # describe_referrers is slow on a large heap (one gc scan per node): at most 3 per process


def describe_referrers(obj, *, max_depth=4, max_nodes=40, max_scans=12, skip_ids=()):
    """Who keeps `obj` alive: a breadth-first walk over gc.get_referrers up to max_depth, one line per referrer
    ('<depth> <what> -> <child>'), frames named by function / file / line, dicts by their owner type when the dict is an
    instance __dict__. Diagnostic only: at most max_scans heap scans (each gc.get_referrers call walks the whole heap)."""
    import types as _types
    me = sys._getframe()
    skip = set(skip_ids) | {id(me), id(me.f_back)}
    lines, seen = [], {id(obj)}
    labels = {id(obj): type(obj).__name__}             # id -> label (strings only: no extra references)
    frontier = [obj]
    scans = 0
    for depth in range(max_depth):
        nxt = []
        for o in frontier:
            if scans >= max_scans:
                lines.append(f"(stopped after {scans} heap scans)")
                return lines
            scans += 1
            label = labels[id(o)]
            refs = gc.get_referrers(o)
            skip.add(id(refs))
            for r in refs:
                if id(r) in seen or id(r) in skip or r is frontier or r is nxt or r is lines or r is labels:
                    continue
                seen.add(id(r))
                if isinstance(r, _types.FrameType):
                    what = f"frame {r.f_code.co_name} ({os.path.basename(r.f_code.co_filename)}:{r.f_lineno})"
                elif isinstance(r, dict):
                    owners = [type(x).__name__ for x in gc.get_referrers(r)
                              if getattr(x, "__dict__", None) is r and not isinstance(x, _types.FrameType)]
                    what = (f"__dict__ of {owners[0]}" if owners else f"dict({len(r)} keys: {list(map(str, r))[:5]})")
                elif isinstance(r, (list, tuple)):
                    what = f"{type(r).__name__}(len {len(r)})"
                elif isinstance(r, _types.CellType):
                    what = "closure cell"
                elif isinstance(r, (_types.FunctionType, _types.MethodType)):
                    what = f"{type(r).__name__} {getattr(r, '__qualname__', '?')}"
                else:
                    what = type(r).__name__
                lines.append(f"{depth} {what} -> {label}")
                labels[id(r)] = what
                nxt.append(r)
                if len(lines) >= max_nodes:
                    return lines
            del refs
        frontier = nxt
        if not frontier:
            break
    return lines


def _no_grad():
    return torch.no_grad() if TORCH_OK else contextlib.nullcontext()


def _span(tel, name):
    return tel.span(name) if tel is not None else contextlib.nullcontext()


def _cpu_span(tel, name):
    return tel.cpu_span(name) if tel is not None else contextlib.nullcontext()


# =====================================================================================================================
# arms
# =====================================================================================================================
ARMS = {
    "T":     {"family": "sam3", "hf_id": "facebook/sam3", "scope": "S", "domain": "published weights (teacher)"},
    "S-Edi": {"family": "sam3", "hf_id": "facebook/sam3", "student_key": "models.student_edinburgh", "scope": "S",
              "domain": "Edinburgh clips 0:4 (2,400 unlabeled frames)"},
    "S-Edi-mined": {"family": "sam3", "hf_id": "facebook/sam3", "student_key": "models.student_edinburgh_mined",
                    "student_sha256": "8df6a0f86354f4113a5713e69b972f5cef5c73f3d584f0ba74462d5721550f2f", "scope": "S",
                    "domain": "Edinburgh clips 0:4 (2,400 base + 627 mined frames)"},
    "S-PT":  {"family": "sam3", "hf_id": "facebook/sam3", "student_key": "models.student_pigtrack",
              "student_sha256": "c07140f2de80a5e9faf175bdb4fc071bb1064e4757b91c3eefe55e271e5da69c", "scope": "S",
              "domain": "PigTrack train split (3,000 unlabeled frames)"},
    "S-PT-mined": {"family": "sam3", "hf_id": "facebook/sam3", "student_key": "models.student_pigtrack_mined",
                   "student_sha256": "b47ab0fa5801219946b88de4b2ad14f2b6ff71caf9d4a1ef6b14222177f9e920", "scope": "S",
                   "domain": "PigTrack train split (2,626 base + 2,125 mined targets)"},
    "S-v2":  {"family": "sam3", "hf_id": "facebook/sam3", "student_key": "models.student_calf", "scope": "S",
              "domain": "calf pen, 18-19 April, with night-mined targets"},
    "T21":   {"family": "sam2", "hf_id": "facebook/sam2.1-hiera-tiny", "scope": "S", "domain": "published weights"},
    "S21":   {"family": "sam2", "hf_id": "facebook/sam2.1-hiera-small", "scope": "S", "domain": "published weights"},
    "ET":    {"family": "edgetam", "hf_id": "yonigozlan/EdgeTAM-hf", "scope": "S", "domain": "published weights"},
    # EfficientSAM3 Stage-1 encoder inside SAM 3's tracker; see es3_encoder.py and benchmark/config/efficientsam3_encoder.json
    "ES-L":  {"family": "es3", "hf_id": "facebook/sam3", "es3_key": "models.es3_trunk",
              "scope": "S", "domain": "published weights (SA-1B distilled)"},
}


class Arm:
    """A loaded tracker arm. Everything the loop needs: init_session, preprocess, add_prompts (ONE call per frame,
    fresh obj_ids list), forward (explicit frame_idx; student features injected through DummyBackbone), postprocess."""

    def __init__(self, name, family, model, proc, *, device="cpu", dtype=None, student=None, student_input=None,
                 dummy=None, mask_input=None, info=None):
        self.name, self.family, self.model, self.proc = name, family, model, proc
        self.device, self.dtype = device, dtype
        self.student, self.student_input, self.dummy = student, student_input, dummy
        self.mask_input = mask_input or (lambda m: np.asarray(m, dtype=bool))
        self.info = dict(info or {})
        self._bb_hook, self._bb_want, self._bb_out = None, False, None
        self.n_add_calls = 0

    def attach_backbone_capture(self):
        """Non-student arms: capture vision_encoder.backbone's output on check frames (finite + collapse asserts)."""
        if self.student is not None or not hasattr(self.model, "named_modules"):
            return False
        bb = dict(self.model.named_modules()).get("vision_encoder.backbone")
        if bb is None:
            return False

        def hook(_m, _a, o):
            if self._bb_want:
                self._bb_out = _first_tensor(o)
        self._bb_hook = bb.register_forward_hook(hook)
        return True

    def init_session(self):
        return self.proc.init_video_session(inference_device=self.device, dtype=self.dtype)

    def reset_transient(self):
        """Drop every per-frame tensor the arm itself may still hold (captured backbone output, injected student
        features) -- part of the session teardown."""
        self._bb_out, self._bb_want = None, False
        if self.dummy is not None:
            self.dummy.student_features = None

    def preprocess(self, pil):
        return self.proc(images=pil, device=self.device, return_tensors="pt")

    def add_prompts(self, sess, fi, prompts, original_size):
        kinds = {p.kind for p in prompts}
        if len(kinds) != 1:
            raise ValueError("one prompt kind per add_inputs call")
        ids = [int(p.obj_id) for p in prompts]
        kw = dict(inference_session=sess, frame_idx=int(fi), obj_ids=list(ids), original_size=original_size)
        if kinds == {"box"}:
            kw["input_boxes"] = [[[float(v) for v in p.box] for p in prompts]]       # [image][object][x1,y1,x2,y2]
        else:
            kw["input_masks"] = [self.mask_input(p.mask) for p in prompts]          # python LIST of 2-D masks
        self.proc.add_inputs_to_inference_session(**kw)
        self.n_add_calls += 1

    def forward(self, sess, fi, inputs, pil, tel=None, check=False):
        with _no_grad():
            if self.student is not None:
                with _span(tel, "student"):
                    feat = self.student(self.student_input(pil))
                if check:
                    assert_finite(feat, f"{self.name} student features")
                    assert_spread(feat, f"{self.name} student features")
                self.dummy.student_features = feat
            self._bb_want = bool(check and self._bb_hook is not None)
            try:
                with _span(tel, "model"):
                    out = self.model(inference_session=sess, frame_idx=int(fi), frame=inputs.pixel_values[0])
            finally:
                if self.dummy is not None:
                    self.dummy.student_features = None
                self._bb_want = False
        if check and self._bb_hook is not None:
            if self._bb_out is None:                 # fail loud: an unknown output structure must not skip the check
                raise BenchAssertError(f"{self.name}: no tensor captured from vision_encoder.backbone on check frame "
                                       f"{fi} (extend _first_tensor for this output type)")
            assert_finite(self._bb_out, f"{self.name} backbone output")
            assert_spread(self._bb_out, f"{self.name} backbone output")
            self._bb_out = None
        return out

    def postprocess(self, out, inputs):
        m = self.proc.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes, binarize=False)[0]
        return mask_logits_2d(m)


def _config_scalars(cfg):
    try:
        d = cfg.to_dict()
    except Exception:  # noqa: BLE001
        d = dict(vars(cfg))
    out = {}

    def rec(prefix, v, depth=0):
        if isinstance(v, dict) and depth < 3:
            for k, x in v.items():
                if _SECRET_KEY_RX.match(str(k)):          # a from_pretrained(token=...) kwarg must never be recorded
                    continue
                rec(f"{prefix}.{k}" if prefix else str(k), x, depth + 1)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[prefix] = v
    rec("", d)
    return out


_SECRET_KEY_RX = re.compile(r"^(token|use_auth_token|hf_token|auth_token|access_token|api_key|password|secret)$", re.I)


def _attn_impls(cfg, path="config", depth=0, out=None):
    out = {} if out is None else out
    out[path] = getattr(cfg, "_attn_implementation", None)
    if depth < 3:
        for k, v in vars(cfg).items():
            if hasattr(v, "to_dict") and hasattr(v, "_attn_implementation") and not k.startswith("__"):
                _attn_impls(v, f"{path}.{k}", depth + 1, out)
    return out


def param_table(model, student=None):
    """baselines.md §6.1: params per module group + total (+ the student, which lives outside the HF graph)."""
    mods = dict(model.named_modules())
    t = {g: int(sum(p.numel() for p in mods[g].parameters())) for g in MODULE_GROUPS if g in mods}
    t["total_hf_graph"] = int(sum(p.numel() for p in model.parameters()))
    if student is not None:
        t["student"] = int(sum(p.numel() for p in student.parameters()))
        t["total_system"] = t["total_hf_graph"] + t["student"]     # DummyBackbone holds no parameters
    else:
        t["total_system"] = t["total_hf_graph"]
    return t


def gpu_info():
    if not (TORCH_OK and torch.cuda.is_available()):
        return {"name": None, "class": "CPU"}
    p = torch.cuda.get_device_properties(0)
    n = p.name
    cls = "A100" if "A100" in n.upper() else ("A10" if re.search(r"A10(?!\d)", n.upper()) else n)
    return {"name": n, "class": cls, "total_gib": round(p.total_memory / 2 ** 30, 2),
            "capability": f"{p.major}.{p.minor}"}


def versions():
    import importlib.metadata as md
    import platform
    out = {"python": platform.python_version()}
    for pkg in ("torch", "torchvision", "transformers", "timm", "numpy", "Pillow", "rfdetr", "opencv-python-headless",
                "scipy", "accelerate", "huggingface_hub", "supervision"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = None
    if TORCH_OK:
        out["cuda"] = torch.version.cuda
        try:
            out["cudnn"] = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001
            out["cudnn"] = None
    return out


def load_student(path, *, device="cuda", dtype=None, check_gn=True):
    """Published student architecture (VERBATIM export class, drop_path_rate=0.0) + checkpoint. Accepted formats:
    {'ema_shadow', 'model_state_dict', 'epoch', ...} (EMA weights win;) or the bare fp16 export
    {'model_state_dict', 'out_channels', 'num_parameters'}. Asserts: every student tensor comes from the checkpoint and
    no checkpoint key is unused, 0 BatchNorm, 40,655,408 params, GroupNorm train/eval max diff < 1e-5 (fp32)."""
    _require_torch()
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        wo = True
    except Exception:  # noqa: BLE001 -- our own sha-pinned files only
        ck = torch.load(path, map_location="cpu", weights_only=False)
        wo = False
    student = StudentEncoderTinyViT(out_channels=1024, pretrained=False)
    sd = student.state_dict()
    msd = dict(ck.get("model_state_dict") or {})
    ema = dict(ck.get("ema_shadow") or {})
    have = set(msd) | set(ema)
    missing = [k for k in sd if k not in have]
    extra = [k for k in have if k not in sd]
    if missing or extra:
        raise RuntimeError(f"student checkpoint/architecture mismatch: missing {missing[:6]} unused {extra[:6]}")
    sd.update({k: v.float() for k, v in msd.items()})
    sd.update({k: v.float() for k, v in ema.items()})
    student.load_state_dict(sd, strict=True)
    nbn = sum(1 for m in student.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))
    if nbn:
        raise RuntimeError(f"{nbn} BatchNorm layers survive")
    npar = student.get_num_parameters()
    if npar != STUDENT_N_PARAMS:
        raise RuntimeError(f"student has {npar} params, expected {STUDENT_N_PARAMS}")
    student = student.to(device=device, dtype=torch.float32)
    gn_diff = None
    if check_gn:
        g = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn(1, 3, 1024, 1024, generator=g).to(device)
        with torch.no_grad():
            student.train()
            a = student(x)
            student.eval()
            b = student(x)
        gn_diff = float((a - b).abs().max())
        del a, b, x
        if not gn_diff < GN_TRAIN_EVAL_MAX:
            raise RuntimeError(f"student train/eval max diff {gn_diff:.3g} >= {GN_TRAIN_EVAL_MAX} (drop_path?)")
    student = student.to(dtype=dtype or torch.bfloat16).eval()
    student.requires_grad_(False)
    meta = {"format": "ema" if ema else "model_state_dict", "epoch": (int(ck["epoch"]) + 1) if "epoch" in ck else None,
            "n_params": int(npar), "gn_train_eval_max_diff_fp32": gn_diff, "weights_only_load": wo,
            "dtype": str(dtype or torch.bfloat16).replace("torch.", "")}
    return student, meta


def hf_commit(model, hf_id, rev=None, token=None):
    """The HF commit the weights were loaded from. transformers 5.2.0 leaves
    config._commit_hash None, so: config._commit_hash, else the local hub cache ref the download
    resolved (refs/<rev or main>), else the Hub API. Returns (sha | None, source)."""
    c = getattr(getattr(model, "config", None), "_commit_hash", None)
    if c:
        return str(c), "config._commit_hash"
    if rev and re.match(r"^[0-9a-f]{40}$", str(rev)):
        return str(rev), "pinned_revision"
    try:
        from huggingface_hub import constants
        ref = os.path.join(constants.HF_HUB_CACHE, "models--" + hf_id.replace("/", "--"), "refs", rev or "main")
        if os.path.isfile(ref):
            with open(ref, encoding="utf-8") as f:
                v = f.read().strip()
            if re.match(r"^[0-9a-f]{40}$", v):
                return v, f"hub_cache_ref:{rev or 'main'}"
    except Exception:  # noqa: BLE001
        pass
    try:
        from huggingface_hub import HfApi
        return str(HfApi().model_info(hf_id, revision=rev, token=token).sha), "hub_api"
    except Exception as e:  # noqa: BLE001
        return None, f"unresolved: {type(e).__name__}"


STUDENT_RUN_RX = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")       # one directory name: no '/', no '.', no '..'
SHA256_RX = re.compile(r"^[0-9a-f]{64}$")


def student_source(name, registry):
    """(source path, sha256 pin | None, provenance) of student arm `name`, resolved ONLY through the registry.
    ARMS[name]["student_key"] names the registry entry. ARMS[name]["student_run"] (optional) selects the SIBLING run
    directory of that entry's model directory, same file name: models/<family>/<student_run>/<basename of the key's
    file>. Pin: ARMS[name]["student_sha256"]; a registry pin on the key must agree with it, except
    for a student_run arm, whose file is not the key's own file (the key's pin then belongs to another file and is NOT
    used). A student_run arm without its own sha256 pin is refused (an unpinned derived path would be unverified)."""
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    spec = ARMS[name]
    key = spec.get("student_key")
    if not key:
        raise ValueError(f"{name} is not a student arm")
    spin = spec.get("student_sha256")
    if spin is not None and not SHA256_RX.match(str(spin)):
        raise ValueError(f"{name}: student_sha256 {spin!r} is not a lowercase hex sha256")
    base = registry.path(key)
    rpin = registry.entry(key).get("sha256")
    run = spec.get("student_run")
    if run is None:
        if spin and rpin and spin != rpin:
            raise RuntimeError(f"{name}: arm-table pin {spin[:16]} != registry pin {rpin[:16]} for {key}")
        pin = spin or rpin
        src = "arm_table" if spin else ("registry" if rpin else None)
        return base, pin, {"student_run": None, "source": f"registry:{key}", "pin_source": src}
    if not STUDENT_RUN_RX.match(str(run)):
        raise ValueError(f"{name}: student_run {run!r} must be one directory name")
    if not spin:
        raise ValueError(f"{name}: student_run {run!r} needs its own student_sha256 pin in ARMS")
    model_dir = base.replace("\\", "/").rsplit("/", 1)[0]
    family_dir = model_dir.rsplit("/", 1)[0]
    if not family_dir or family_dir == model_dir or not family_dir.startswith(registry.volume + "/models/"):
        raise ValueError(f"{name}: {key} -> {base} is not a models/<family>/<run>/<file> path")
    src = f"{family_dir}/{run}/{base.replace(chr(92), '/').rsplit('/', 1)[1]}"
    return src, spin, {"student_run": run, "source": f"registry:{key} sibling run {run}", "pin_source": "arm_table"}


def _es3_module():
    """es3_encoder from this module's directory. An es3_encoder imported from another file is refused."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "es3_encoder.py")
    if not os.path.isfile(path):
        raise ImportError(f"es3_encoder.py not found beside bench_core.py ({here})")
    mod = sys.modules.get("es3_encoder")
    if mod is not None:
        mf = getattr(mod, "__file__", None)
        if not mf or os.path.normcase(os.path.realpath(mf)) != os.path.normcase(os.path.realpath(path)):
            raise ImportError(f"es3_encoder is already imported from {mf}, not from {path}")
        return mod
    import importlib.util
    spec = importlib.util.spec_from_file_location("es3_encoder", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["es3_encoder"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("es3_encoder", None)
        raise
    return mod


def es3_config(registry):
    """-> (config, path) of protocol/efficientsam3_encoder.json: the checkpoint and upstream TinyViT file to load."""
    m = _es3_module()
    p = protocol_file_path(registry, m.CONFIG_FILE)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} not found: copy benchmark/config/{m.CONFIG_FILE} there")
    with open(p, "rb") as f:
        rec = json.loads(f.read().decode("utf-8"))
    probs = m.validate_config(rec)
    if probs:
        raise ValueError(f"malformed {p}: {probs}")
    return rec, p


def es3_arm_gates(arm, name="ES-L"):
    """Post-load gates of an ES-L arm (independent of es3_encoder's own): ES3Backbone in the graph, ES3Processor in front
    (the authors' frame), every floating encoder tensor bf16, no encoder module in training mode, and every TinyViT
    attention-bias cache computed from the LOADED weights in their final dtype / device (a cache made by an eval()
    before the load -- the authors' video builder -- is stale). -> (n_attention, n_stale = 0); RuntimeError otherwise."""
    m = _es3_module()
    bb = arm.model.vision_encoder.backbone
    if not isinstance(bb, m.ES3Backbone):
        raise RuntimeError(f"{name}: vision_encoder.backbone is {type(bb).__name__}, not ES3Backbone")
    if not isinstance(arm.proc, m.ES3Processor):
        raise RuntimeError(f"{name}: the processor is {type(arm.proc).__name__}, not ES3Processor (authors' frame)")
    enc = bb.encoder
    dts = sorted({str(t.dtype).replace("torch.", "") for t in list(enc.parameters()) + list(enc.buffers())
                  if t.is_floating_point()})
    if dts != [DTYPE_NAME]:
        raise RuntimeError(f"{name}: encoder floating tensors are {dts}, expected [{DTYPE_NAME}]")
    if any(mm.training for mm in enc.modules()):
        raise RuntimeError(f"{name}: an encoder module is in training mode after the load")
    n_att, n_stale, worst = m.stale_attention_bias_caches(enc)
    if n_att == 0 or n_stale:
        raise RuntimeError(f"{name}: {n_stale}/{n_att} stale TinyViT attention-bias caches (max diff {worst}); eval() "
                           "must run after the checkpoint is loaded")
    return int(n_att), int(n_stale)


def _load_es3(name, spec, *, device, hf_token, revisions, registry, dbutils, local_dir, log):
    """ES-L = load_arm('T') + the EfficientSAM3 encoder swap, through es3_encoder.load_es3_arm:
    vision_encoder.backbone = ES3Backbone(ES-TV-L encoder), pixel_values = the authors' frame (ES3Processor:
    make_es3_transform, PIL default-filter (bicubic) resize to 1008^2, /255, ImageNet mean / std), encoder weights loaded
    strictly in fp32 -> cast to bf16 -> eval() (TinyViT's attention-bias caches are computed from the loaded weights;
    the authors' video builder calls eval() before loading), bf16 like every model. The config, the encoder payload
    (models.es3_trunk) and the upstream tiny_vit.py are resolved through the registry."""
    if registry is None:
        raise ValueError(f"{name}: pass registry=farm_paths.load() (the config, models.es3_trunk and the upstream "
                         "tiny_vit.py are resolved through it)")
    m = _es3_module()
    rec, _ = es3_config(registry)
    trunk_src = registry.path(spec["es3_key"])
    tv_src = os.path.dirname(trunk_src).replace("\\", "/") + "/" + rec["tinyvit_source"]["relpath"]
    d = os.path.join(local_dir, name)
    trunk_local, trunk_sha = stage_file(trunk_src, os.path.join(d, os.path.basename(trunk_src)), dbutils)
    tv_local, tv_sha = stage_file(tv_src, os.path.join(d, "tiny_vit.py"), dbutils, rec["tinyvit_source"]["sha256"])
    arm = m.load_es3_arm(sys.modules[__name__], trunk_local=trunk_local, tinyvit_local=tv_local,
                         trunk_sha256=trunk_sha, tinyvit_sha256=tv_sha, device=device,
                         hf_token=hf_token, revisions=revisions, registry=registry, dbutils=dbutils,
                         local_dir=local_dir, log=log)
    n_att, n_stale = es3_arm_gates(arm, name)
    rec_ck = dict(rec.get("checkpoint") or {})
    frame_route = {
        "route": ("the authors' SAM2-task video loader (efficientsam3 sam2_utils.load_video_frames, as "
                  "Sam3TrackerPredictor.init_state calls it); part of the model"),
        "transform": "es3_encoder.make_es3_transform (via es3_encoder.ES3Processor: replaces the HF processor's pixel_values)",
        "resize": f"PIL Image.resize(({m.IMAGE_SIZE}, {m.IMAGE_SIZE})) with PIL's default filter (bicubic)",
        "scale": "/ 255", "mean": list(m.IMG_MEAN), "std": list(m.IMG_STD),
        "hf_processor_kept_for": "original sizes, prompt handling and post_process_masks",
        "hf_processor_pixel_values_not_used": "bilinear resize, mean / std 0.5 (recorded in info.processor)"}
    arm.info["es3"].update({
        "frame_route": frame_route,
        "weights": {"encoder_payload": {"registry_key": spec["es3_key"], "sha256": trunk_sha,
                                        "format": rec["trunk"].get("format")},
                    "tinyvit_source": {"relpath": rec["tinyvit_source"]["relpath"], "sha256": tv_sha},
                    "checkpoint": {k: rec_ck.get(k) for k in ("hf_repo", "file", "revision", "sha256", "size")},
                    "non_encoder": (f"{spec['hf_id']} at commit_hash (every non-trunk tensor of the "
                                    "EfficientSAM3 checkpoint is bit-identical to sam3.pt)"),
                    "redistribution": "never (third-party; models/baselines/third_party/efficientsam3/)"},
        "load_order": ("encoder state_dict loaded strictly in fp32 -> cast to bf16 on the device -> eval(): TinyViT "
                       "attention-bias caches computed from the loaded weights (the authors' video builder calls eval() "
                       "before loading the checkpoint, leaving them at zero)"),
        "attention_bias_caches": {"n": int(n_att), "stale": int(n_stale)}})
    arm.info.update({"scope": spec["scope"], "domain_training": spec["domain"],
                     "wiring": CORE_REVISION})
    return arm


def load_arm(name, *, device="cuda", hf_token=None, revisions=None, registry=None, dbutils=None, student_ckpt=None,
             student_sha256=None, student_dtype=None, local_dir="/local_disk0/bench_models", log=print):
    """Load one arm under the frozen settings: transformers 5.2.0, bf16 weights, attn_implementation='sdpa' requested
    (rejection recorded; a resolved flash_attention_2 raises), eval + no grad, fp32 matmul precision 'highest'.
    S-* arms: the student checkpoint is resolved through the registry key of ARMS[name] (student_source: optionally a
    sibling run directory; or student_ckpt), staged to local disk, sha256-checked against
    the arm-table / registry pin when present; the teacher's vision_encoder.backbone is DELETED
    and replaced by the verbatim DummyBackbone. revisions: {hf_id: commit} pins.
    ES-L: _load_es3 = es3_encoder.load_es3_arm (this function's T arm + the EfficientSAM3 encoder
    in the graph, the authors' frame as pixel_values, bf16, eval() after the load) (needs protocol/efficientsam3_encoder.json)."""
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    spec = ARMS[name]
    if spec["family"] == "es3":
        if student_ckpt is not None or student_sha256 is not None or student_dtype is not None:
            raise ValueError(f"{name} is not a student arm (student_* arguments given)")
        return _load_es3(name, spec, device=device, hf_token=hf_token, revisions=revisions, registry=registry,
                         dbutils=dbutils, local_dir=local_dir, log=log)
    _require_torch(need_transformers=True)
    import transformers
    if transformers.__version__ != TRANSFORMERS_PIN:
        raise RuntimeError(f"transformers {transformers.__version__} != {TRANSFORMERS_PIN}")
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major < 8:
        raise RuntimeError("bf16 needs Ampere+ (A10 / A100); fp16 is forbidden by the protocol")
    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    dtype = torch.bfloat16
    from transformers import (Sam2VideoModel, Sam2VideoProcessor, EdgeTamVideoModel, Sam3TrackerVideoModel,
                              Sam3TrackerVideoProcessor)
    M, P = {"sam3": (Sam3TrackerVideoModel, Sam3TrackerVideoProcessor), "sam2": (Sam2VideoModel, Sam2VideoProcessor),
            "edgetam": (EdgeTamVideoModel, Sam2VideoProcessor)}[spec["family"]]
    hf_id = spec["hf_id"]
    rev = (revisions or {}).get(hf_id)
    kw = {k: v for k, v in (("token", hf_token), ("revision", rev)) if v}
    proc = P.from_pretrained(hf_id, **kw)
    try:
        model = M.from_pretrained(hf_id, torch_dtype=dtype, attn_implementation=ATTN_IMPL, **kw)
        attn_req = "accepted"
    except (TypeError, ValueError) as e:
        model = M.from_pretrained(hf_id, torch_dtype=dtype, **kw)
        attn_req = f"rejected: {type(e).__name__}: {str(e)[:200]}"
    model = model.to(device).eval()
    model.requires_grad_(False)
    attn = _attn_impls(model.config)
    if any(v == "flash_attention_2" for v in attn.values()):
        raise RuntimeError(f"{name}: flash_attention_2 resolved somewhere ({attn}); the protocol requires SDPA")
    commit, commit_src = hf_commit(model, hf_id, rev, hf_token)
    info = {"arm": name, "family": spec["family"], "hf_id": hf_id, "revision_requested": rev,
            "commit_hash": commit, "commit_source": commit_src, "attn_request": attn_req,
            "attn_resolved": attn, "domain_training": spec["domain"], "scope": spec["scope"],
            "model_config": _config_scalars(model.config),
            "processor": {k: v for k, v in _config_scalars(getattr(proc, "image_processor", proc)).items()
                          if not k.startswith("_")},
            "dtype": DTYPE_NAME, "matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}
    student = sin = dummy = None
    if spec.get("student_key"):
        pin = student_sha256
        if student_ckpt is None:
            if registry is None:
                raise ValueError(f"{name}: pass registry=farm_paths.load() (or student_ckpt)")
            src, spin, sprov = student_source(name, registry)
            if pin and spin and pin != spin:
                raise RuntimeError(f"{name}: student_sha256 argument {pin[:16]} != the arm's pin {spin[:16]}")
            pin = pin or spin
            student_ckpt, sha = stage_file(src, os.path.join(local_dir, name, os.path.basename(src)), dbutils, pin)
        else:
            sprov = {"student_run": spec.get("student_run"), "source": "student_ckpt argument",
                     "pin_source": "argument" if pin else ("arm_table" if spec.get("student_sha256") else None)}
            pin = pin or spec.get("student_sha256")          # an arm name always means its pinned file
            sha = sha256_file(student_ckpt)
            if pin and sha != pin:
                raise RuntimeError(f"{name}: student sha256 {sha[:16]} != pinned {pin[:16]}")
        student, smeta = load_student(student_ckpt, device=device, dtype=student_dtype or dtype)
        smeta.update({"registry_key": spec["student_key"], "sha256": sha, "file": os.path.basename(student_ckpt),
                      "student_run": sprov.get("student_run"), "source": sprov.get("source"),
                      "sha256_pinned": bool(pin), "pin_source": sprov.get("pin_source") if pin else None})
        info["student"] = smeta
        tf = make_student_transform()
        sdt = student_dtype or dtype
        sin = lambda pil: tf(pil).unsqueeze(0).to(device=device, dtype=sdt)      # noqa: E731
        dummy = DummyBackbone().to(device)
        model.vision_encoder.backbone = dummy          # the teacher backbone is DELETED, not
        gc.collect()                                     # parked on the CPU: weights-only VRAM = the student system
        torch.cuda.empty_cache()
    info["params"] = param_table(model, student)
    info["weights_mem_mib"] = mem_snapshot()
    arm = Arm(name, spec["family"], model, proc, device=device, dtype=dtype, student=student, student_input=sin,
              dummy=dummy, info=info,
              mask_input=lambda m: torch.from_numpy(np.ascontiguousarray(np.asarray(m, dtype=bool))).to(device))
    arm.info["backbone_capture"] = arm.attach_backbone_capture()
    log(f"arm {name}: {hf_id} commit {info['commit_hash']} attn {attn_req} params {info['params']}")
    return arm


# =====================================================================================================================
# THE streaming loop (one implementation for every arm, dataset and protocol)
# =====================================================================================================================
def track_stream(arm, frames, *, protocol="A", prompts=(), detector=None, protd=None, keep_last_n=KEEP_LAST_N,
                 prune_every=PRUNE_EVERY, pop=POP_FRAMES, mot=None, rle=None, tel=None, check_every=CHECK_EVERY,
                 diag_every=DIAG_EVERY, finite_every=FINITE_EVERY, frame_hash_every=FRAME_HASH_EVERY, max_frames=None,
                 progress_every=0, log=print, on_frame=None, release=True, diagnose_teardown=True,
                 frame_clear_every=FRAME_CLEAR_EVERY):
    """Track one sequence in ONE session. frames yields (fi, pil, meta) with fi = 0, 1, 2, ... (asserted).
    Per frame: preprocess -> prompts due at fi (seeded run plan; detector run: detector pass on frame fi + births) in ONE
    add_inputs call with a fresh obj_ids list -> forward(frame_idx=fi) -> pop processed_frames[fi] -> post_process_masks
    -> finite asserts -> output rule -> (detector run emit filter) -> writers -> prune every prune_every frames
    -> on_frame(fi=, sess=, out=, mask_logits=, binm=, rows=, live=, pruned=) (G1 checks; observe only, never mutate).
    keep_last_n=None disables pruning (factorial arms only). Frames before the first prompt are not forwarded.
    Every frame_clear_every frames (at a prune) and at teardown, finished frames the loop called are cleared
    (clear_finished_frames_under: a Databricks runtime thread keeps frame snapshots that would otherwise hold the
    session and earlier frames' tensors; output-neutral; 0 disables the in-loop clearing).
    Teardown (also on error): the session is RELEASED IN PLACE (release_session; release=False -- white-box tests and
    the G1 reproduction of the J0 behaviour -- skips it and the frame clearing), the arm's transient tensors are
    dropped, gc.collect() + empty_cache(); stats["teardown"] records what was released, the frames cleared, whether
    the session object outlived the teardown and, if it did, who held it (describe_referrers).
    Returns the run statistics (births, deaths, prunes, processed-frame / non-cond bookkeeping, overlap diagnostic)."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol {protocol!r}")
    if protocol == "D" and (detector is None or protd is None or prompts):
        raise ValueError("detector run needs detector + protd and takes no external prompts")
    if protocol == "A" and (detector is not None or protd is not None):
        raise ValueError("seeded run runs no detector")
    sched = schedule_prompts(prompts)
    live = []
    st = {"protocol": protocol, "frames": 0, "forwards": 0, "skipped_no_object": 0, "rows": 0, "prunes": 0,
          "pruned_entries": 0, "add_calls": 0, "births": [], "max_processed_frames": 0, "max_noncond_per_obj": 0,
          "noncond_per_obj_after_prune": [], "overlap": [], "checks": 0, "keep_last_n": keep_last_n,
          "prune_every": prune_every, "pop": pop, "finite_every": finite_every, "frame_hashes": [],
          "frame_clear_every": frame_clear_every, "frames_cleared_in_loop": 0, "frame_clear_s": 0.0}
    me = sys._getframe()                          # anchor for clear_finished_frames_under (dropped at teardown)
    sess = arm.init_session()
    t_start = time.perf_counter()
    expected = 0
    out = inputs = ml = packed = scores = binm = rows = pil = None
    teardown = None
    it = iter(frames)
    try:
        for fi, pil, meta in it:
            if fi != expected:
                raise BenchAssertError(f"frame index {fi} != expected {expected} (sources must be contiguous)")
            expected += 1
            if max_frames is not None and fi >= max_frames:
                break
            st["frames"] += 1
            if tel is not None:
                tel.begin_frame(fi)
            if frame_hash_every and fi % frame_hash_every == 0:
                st["frame_hashes"].append([fi, frame_sha256(pil)])
            with _cpu_span(tel, "preprocess"):
                inputs = arm.preprocess(pil)
            new = list(sched.pop(fi, []))
            if protocol == "D" and protd.is_pass(fi):
                with _span(tel, "detector"), _cpu_span(tel, "detector_wall"):
                    dets = detector(fi, pil)
                new += protd.on_pass(fi, dets)
            if new:
                dup = [p.obj_id for p in new if p.obj_id in live]
                if dup:
                    raise BenchAssertError(f"frame {fi}: re-prompt of live ids {dup}")
                arm.add_prompts(sess, fi, new, inputs.original_sizes[0])
                live.extend(int(p.obj_id) for p in new)
                st["add_calls"] += 1
                st["births"].append([fi, [int(p.obj_id) for p in new]])
            if not live:
                st["skipped_no_object"] += 1
                if protd is not None:
                    protd.on_output(fi, {})
                continue
            check = (fi % check_every == 0) or st["forwards"] < 3
            out = arm.forward(sess, fi, inputs, pil, tel=tel, check=check)
            st["forwards"] += 1
            if pop:
                pop_processed_frame(sess, fi)
            sess_ids = getattr(sess, "obj_ids", None)
            if isinstance(sess_ids, (list, tuple)) and [int(i) for i in sess_ids] != live:
                raise BenchAssertError(f"session object order {list(sess_ids)[:8]} != prompt order {live[:8]}")
            if check or (finite_every and fi % finite_every == 0):
                assert_finite(out.object_score_logits, f"{arm.name} object_score_logits @ {fi}")
                assert_finite(out.pred_masks, f"{arm.name} pred_masks @ {fi}")
            with _span(tel, "post"):
                ml = arm.postprocess(out, inputs)
                if ml.shape[0] != len(live):
                    raise BenchAssertError(f"{ml.shape[0]} masks for {len(live)} live objects @ {fi}")
                packed, scores, binm = output_rule(ml, out.object_score_logits)
            if check:
                st["checks"] += 1
                present = [k for k in range(len(live)) if scores[k] > 0]
                if present:
                    assert_spread(ml[present] if not _is_t(ml) else ml[torch.as_tensor(present, device=ml.device)],
                                  f"{arm.name} mask logits of present objects @ {fi}")
            rows = rows_from_packed(packed, scores, live)
            if protd is not None:
                allowed = protd.on_output(fi, {r.obj_id: r.box_cont for r in rows})
                rows = [r for r in rows if r.obj_id in allowed]
            if mot is not None:
                mot.add(fi, rows)
            if rle is not None:
                rle.add(fi, (meta or {}).get("native_fidx", fi),
                        {r.obj_id: to_numpy(binm[r.k]).astype(bool) for r in rows})
            st["rows"] += len(rows)
            if diag_every and fi % diag_every == 0 and len(rows) > 1:
                ios, iou = pairwise_overlap_max(binm, [r.k for r in rows])
                st["overlap"].append([fi, round(ios, 4), round(iou, 4), len(rows)])
            st["max_processed_frames"] = max(st["max_processed_frames"], n_processed_frames(sess))
            per = [len(sess.output_dict_per_obj[oi].get("non_cond_frame_outputs", {})) for oi in range(sess.get_obj_num())]
            st["max_noncond_per_obj"] = max([st["max_noncond_per_obj"]] + per)
            pruned = False
            if prune_every and (fi + 1) % prune_every == 0:
                pruned = True
                if keep_last_n is not None:
                    st["pruned_entries"] += prune_session(sess, keep_last_n)
                    st["noncond_per_obj_after_prune"].append(
                        max(len(sess.output_dict_per_obj[oi].get("non_cond_frame_outputs", {}))
                            for oi in range(sess.get_obj_num())))
                st["prunes"] += 1
                if release and frame_clear_every and (fi + 1) % frame_clear_every == 0:
                    t_fc = time.perf_counter()
                    st["frames_cleared_in_loop"] += sum(clear_finished_frames_under(me).values())
                    st["frame_clear_s"] = round(st["frame_clear_s"] + time.perf_counter() - t_fc, 4)
                if tel is not None:
                    tel.flush()
                    tel.mem_sample(fi, sess, len(live))
                if progress_every and st["prunes"] % max(1, progress_every // prune_every) == 0:
                    el = time.perf_counter() - t_start
                    log(f"  [{arm.name}] frame {fi + 1}: {st['frames'] / max(el, 1e-9):.2f} fps, {len(live)} objects")
            if on_frame is not None:
                on_frame(fi=fi, sess=sess, out=out, mask_logits=ml, binm=binm, rows=rows, live=list(live),
                         pruned=pruned)
    finally:
        close = getattr(it, "close", None)          # joins a prefetch / decode thread on early exit or error;
        if callable(close):                         # the SOURCE (e.g. a calf cv2 cap) stays owned by the caller
            close()
        it = None
        if tel is not None:
            tel.flush()
        # teardown: release IN PLACE, then drop every local that could reach the session or a frame's tensors
        teardown = {"release": None, "session_alive_after_teardown": None, "holders": None}
        wr = None
        try:
            import weakref
            wr = weakref.ref(sess)
        except TypeError:
            pass
        if release:
            try:
                teardown["release"] = release_session(sess)
            except Exception as e:  # noqa: BLE001 -- never mask the loop's own exception; the memory check reports it
                teardown["release"] = {"error": f"{type(e).__name__}: {e}"}
        arm.reset_transient()
        sess = out = inputs = ml = packed = scores = binm = rows = pil = None
        if release:
            try:
                teardown["frames_cleared"] = clear_finished_frames_under(me)
            except Exception as e:  # noqa: BLE001
                teardown["frames_cleared"] = {f"failed: {type(e).__name__}: {e}": 1}
        me = None
        gc.collect()
        if wr is not None:
            alive = wr() is not None
            teardown["session_alive_after_teardown"] = alive
            if alive and diagnose_teardown and TEARDOWN_DIAG["left"] > 0:
                TEARDOWN_DIAG["left"] -= 1
                try:
                    teardown["holders"] = describe_referrers(wr())
                except Exception as e:  # noqa: BLE001 -- a diagnostic never fails the run
                    teardown["holders"] = [f"describe_referrers failed: {type(e).__name__}: {e}"]
            if alive and release:                 # what is left after the loop's own frames were cleared
                try:
                    teardown["frames_cleared_referencing"] = clear_frames_referencing(wr())
                except Exception as e:  # noqa: BLE001
                    teardown["frames_cleared_referencing"] = {f"failed: {type(e).__name__}: {e}": 1}
                gc.collect()
                teardown["session_alive_after_frame_clear"] = wr() is not None
        wr = None
        if TORCH_OK and torch.cuda.is_available():
            torch.cuda.empty_cache()
            teardown["alloc_after_mib"] = round(mem_allocated_mib(), 3)
    st["teardown"] = teardown
    st["objects"] = len(live)
    st["wall_s"] = round(time.perf_counter() - t_start, 3)
    if protd is not None:
        st["protocol_D"] = protd.summary()
    st["noncond_after_prune_max"] = max(st["noncond_per_obj_after_prune"], default=None)
    st.pop("noncond_per_obj_after_prune")
    return st


# =====================================================================================================================
# BENCH_CONFIG (one per run folder) + resume check
# =====================================================================================================================
CONFIG_VOLATILE = ("created_utc", "extra")


def bench_config(arm_info, *, dataset, split, protocol, variant=None, protd_params=None, keep_last_n=KEEP_LAST_N,
                 prune_every=PRUNE_EVERY, pop=POP_FRAMES, xy_offset=MOT_XY_OFFSET_DEFAULT, detector_info=None,
                 registry=None, registry_keys=(), notebook_source=None, extra=None):
    """The record every run writes ONCE (BENCH_CONFIG.json in its run dir) and every resume re-checks."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset {dataset!r}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol {protocol!r}")
    rec = {
        "bench_version": BENCH_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": {"bench_core": code_sha256(),
                 "notebook": source_shas(notebook_source) if notebook_source is not None else None},
        "dataset": dataset, "split": split, "protocol": protocol, "variant": variant,
        "arm": arm_info,
        "stream": {"keep_last_n": keep_last_n, "prune_every": prune_every, "pop_processed_frames": pop,
                   "explicit_frame_idx": True, "dtype": DTYPE_NAME, "attn_requested": ATTN_IMPL,
                   "matmul_precision": MATMUL_PRECISION, "torch_compile": False, "batch": 1,
                   "one_add_inputs_call_per_frame": True, "fresh_obj_ids_list": True,
                   "teardown": "release_session in place + clear_finished_frames_under",
                   "frame_clear_every": FRAME_CLEAR_EVERY, "mem_return_tol_mib": MEM_RETURN_TOL_MIB,
                   "mem_drift_tol_mib": MEM_DRIFT_TOL_MIB},
        "output_rule": {"emit": "object_score_logits > 0 and (mask_logit > 0).any()", "min_area_px": MIN_AREA_PX,
                        "box": "tight box of the binarised mask on the device; MOT w = x1-x0+1, h = y1-y0+1",
                        "mot_xy_offset": xy_offset, "mot_id": "obj_id + 1", "conf": 1},
        "protocol_D": protd_params if protocol == "D" else None,
        "detector": detector_info if protocol == "D" else None,
        "versions": versions(), "gpu": gpu_info(),
        "registry": registry.provenance(list(registry_keys)) if registry is not None else None,
        "extra": extra,
    }
    return json.loads(json.dumps(rec, default=_json_default))


def config_diff(stored, current, ignore=CONFIG_VOLATILE, prefix=""):
    diffs = []
    if isinstance(stored, dict) and isinstance(current, dict):
        for k in sorted(set(stored) | set(current)):
            if not prefix and k in ignore:
                continue
            p = f"{prefix}.{k}" if prefix else k
            if k not in stored or k not in current:
                diffs.append((p, stored.get(k, "<absent>"), current.get(k, "<absent>")))
            else:
                diffs += config_diff(stored[k], current[k], ignore, p)
    elif stored != current:
        diffs.append((prefix, stored, current))
    return diffs


def config_check(stored, current):
    """Resume guard: a run dir is resumed only under an IDENTICAL config (code, versions, GPU class, arm, protocol)."""
    d = config_diff(stored, current)
    if d:
        raise RuntimeError("BENCH_CONFIG mismatch -- refusing to resume into this run dir: "
                           + "; ".join(f"{p}: {a!r} -> {b!r}" for p, a, b in d[:12]))
    return True


def frame_sha256(img):
    """First 16 hex of sha256 over the frame's RGB bytes (+ shape): the identical-pixels check across arms."""
    a = np.ascontiguousarray(np.asarray(img))
    h = hashlib.sha256(str(a.shape).encode() + a.tobytes())
    return h.hexdigest()[:16]


PATH_SEG_RX = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")    # one output path segment: no '/', no '.', no '..'


def detection_cache_path(registry, dataset, detector, split, seq):
    """paper1_distillation/bench/<dataset>/_detections/<detector>/<split>/<seq>.json (shared by all arms and taus)."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset {dataset!r}")
    for what, v in (("detector tag", detector), ("split", split), ("seq", seq)):
        if not PATH_SEG_RX.match(str(v)):
            raise ValueError(f"{what} {v!r}: one path segment (letters, digits, '_', '.', '-'; not starting with '.')")
    return registry.out("paper1.bench", dataset, "_detections", detector, split, f"{seq}.json")


def protocol_file_path(registry, name):
    """paper1_distillation/protocol/<name> -- the frozen, val-derived values (detector_threshold.json, hf_revisions.json)
    written ONCE before the first test job (publish_no_overwrite)."""
    if not re.match(r"^[A-Za-z0-9_.-]+\.json$", name):
        raise ValueError(f"protocol file name {name!r}")
    return registry.out("paper1.root", "protocol", name)


def bench_run_dir(registry, dataset, arm, protocol, split, variant=None):
    """paper1_distillation/bench/<dataset>/<arm>__prot<A|D>/<split>[/<variant>] via R.out() (refuses Tol/, the mount
    and read-only keys). Inside: BENCH_CONFIG.json, mot/<seq>.txt (= the TrackEval tracker folder), telemetry/<seq>.json,
    rle/<seq>.jsonl.gz, .done_<seq>."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset {dataset!r}")
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol {protocol!r}")
    for what, v in (("split", split), ("variant", variant)):
        if v is not None and (what == "split" or v) and not PATH_SEG_RX.match(str(v)):
            raise ValueError(f"{what} {v!r}: one path segment (letters, digits, '_', '.', '-'; not starting with '.')")
    extra = [dataset, f"{arm}__prot{protocol}", split] + ([variant] if variant else [])
    return registry.out("paper1.bench", *extra)


# =====================================================================================================================
# statistics: paired bootstrap that RE-COMBINES the pooled metric on every resample (TrackEval HOTA is computed from
# pooled per-alpha counts, not as a mean of per-sequence HOTA -- review R1.6)
# =====================================================================================================================
def paired_bootstrap(arm_res, ref_res, combine, stat, *, strata=None, B=BOOT_B, seed=BOOT_SEED, alpha=BOOT_ALPHA):
    """arm_res / ref_res: {seq: per-sequence raw result}. combine(list_of_results) -> combined result (TrackEval:
    lambda rs: metric.combine_sequences({f"r{i}": r for i, r in enumerate(rs)})); stat(combined) -> float. Sequences
    are resampled WITH replacement, the SAME indices for both arms (paired), within each stratum (strata: {seq: label},
    e.g. the night flag, so every resample keeps the 15 day / 10 night composition). Percentile CI."""
    seqs = sorted(arm_res)
    if set(seqs) != set(ref_res):
        raise ValueError("arm and reference must cover the same sequences")
    groups = collections.OrderedDict()
    for s in seqs:
        groups.setdefault((strata or {}).get(s, "all"), []).append(s)

    def val(res, idx):
        return float(stat(combine([res[s] for s in idx])))
    a0, r0 = val(arm_res, seqs), val(ref_res, seqs)
    rng = np.random.default_rng(seed)
    d = np.empty(B, dtype=np.float64)
    keys = sorted(groups, key=str)
    for b in range(B):
        idx = []
        for g in keys:
            mem = groups[g]
            idx += [mem[i] for i in rng.integers(0, len(mem), len(mem))]
        d[b] = val(arm_res, idx) - val(ref_res, idx)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"arm": a0, "ref": r0, "delta": a0 - r0, "ci": [float(lo), float(hi)], "B": B, "seed": seed,
            "alpha": alpha, "strata": {str(g): len(groups[g]) for g in keys}, "deltas_sd": float(d.std())}




def trackeval_combiner(metric):
    """combine() for paired_bootstrap from a TrackEval metric object (HOTA / CLEAR / Identity): unique keys per resample
    draw, TrackEval's own combine_sequences (pooled per-alpha counts)."""
    return lambda rs: metric.combine_sequences({f"r{i}": r for i, r in enumerate(rs)})


# =====================================================================================================================
def api_summary():
    return """bench_core {v} ({r}) -- public API
constants   KEEP_LAST_N=16 PRUNE_EVERY=25 POP_FRAMES=True DTYPE_NAME=bfloat16 ATTN_IMPL=sdpa MIN_AREA_PX=0
            D_PASS_PERIOD_S=1.0 D_MATCH_IOU=0.5 D_BIRTH_IOS=0.3 D_BIRTH_PASSES=2 D_DEATH_S=3.0 D_TAU_GRID
            DET_PROBE_THRESHOLD=0.10 WARMUP_FRAMES=32 TIMED_FRAMES=500 BOOT_B=10000 BOOT_SEED EQUIV_MARGIN_HOTA=1.0
            EDI_TRAIN/EDI_VAL/EDI_TEST EDI_FPS=2.0 PIGTRACK_TEST CALF_TRACK_FPS=5.0 ARMS MODULE_GROUPS
staging     stage_tree(src, dst, dbutils) ; stage_file(src, dst, dbutils, sha256) ; hf_token_from_dbutils(dbutils)
            (dbutils.fs.cp retried CP_ATTEMPTS x; staging targets must be local: assert_local_target(dst))
guards      assert_publish_target(dst) (remote dst only under paper1_distillation/, never Tol/ or the mount) ;
            assert_no_secrets(path) ; register_secret(value) ; PATH_SEG_RX (one output path segment)
sources     EdinburghFrames(local_clip_dir) ; PigTrackFrames(local_seq_dir) [.fps .night .gt_path .warnings]
            CalfWindowFrames(local_mp4, start_fidx, n_frames[, decode='seek_thread']) -> meta['native_fidx']
            every source iterates (fi, PIL RGB, meta), fi = 0.. contiguous; MOT frame = fi + 1
GT/prompts  read_mot_txt(p) ; protocol_a_from_mot(gt) -> (prompts, census)
            read_edinburgh_keyframes(p) ; protocol_a_edinburgh(kf) ; edinburgh_gt_rows(kf, interpolate=True)
            read_bbox_prompts_txt(p) ; prompts_from_masks({{id: mask}}, frame=0) ; Prompt ; schedule_prompts(ps)
arms        load_arm(name, device, hf_token, revisions, registry, dbutils) -> Arm
            (ES-L: es3_encoder beside this module; config protocol/efficientsam3_encoder.json)
            es3_config(R) -> (config, path)
            es3_arm_gates(arm) (ES3Backbone, ES3Processor, bf16, eval, no stale TinyViT attention-bias cache)
            Arm.info -> commit hash, attn, params, student meta, model/processor config scalars, weights VRAM
            load_student(path, device, dtype) -> (student, meta) ; make_student_transform() ; versions() ; gpu_info()
            student_source(name, R) -> (path, sha256 pin, provenance)
geometry    box_iou(a, b) ; box_ios(a, b) ; to_numpy(x)
loop        track_stream(arm, frames, protocol='A'|'D', prompts=..., detector=..., protd=ProtocolD(fps, tau),
                         keep_last_n=16, prune_every=25, pop=True, mot=MotWriter(), rle=RleWriter(p), tel=Telemetry(model),
                         on_frame=None, release=True)
            -> stats (births, prunes, max_processed_frames, noncond_after_prune_max, overlap, protocol_D summary,
                      teardown: release_session record, session_alive_after_teardown, holders, alloc_after_mib)
teardown    release_session(sess) (in place) ; clear_finished_frames_under(frame) ; clear_frames_referencing(obj) ;
            session_tensor_bytes(sess) ; FRAME_CLEAR_EVERY=250 ;
            Arm.reset_transient() ;
            settle_memory() ; mem_allocated_mib() ; check_memory_returned(baseline_mib, now_mib, tol_mib=512, what=)
            MEM_RETURN_TOL_MIB=512 (vs weights-only, first sequence) ; MEM_DRIFT_TOL_MIB=64 (vs the first teardown)
            (between sequences: allocated back at the weights-only baseline) ; describe_referrers(obj)
protocol D  ProtocolD(fps, tau) ; DetectionCache.build(detector, frames, period) / .to_json() / .from_json() / .detector()
            load_rfdetr_detector(ckpt_local, resolution=..., class_names=[...])
output      output_rule(mask_logits, score_logits) ; rows_from_packed ; MotWriter(xy_offset).add/.write ;
            RleWriter(path).add/.close ; read_rle_jsonl ; rle_encode/rle_decode
telemetry   Telemetry(model).span/.cpu_span/.flush/.mem_sample/.summary ; mem_snapshot() ; growth_slope(fis, mib)
validity    BenchAssertError ; assert_finite ; assert_spread
provenance  bench_config(arm.info, dataset, split, protocol, ...) ; config_check(stored, current) ;
            bench_run_dir(R, dataset, arm, protocol, split[, variant]) ; publish_no_overwrite(src, dst, dbutils)
            detection_cache_path(R, dataset, detector, split, seq) ; protocol_file_path(R, name) ; frame_sha256(img)
statistics  paired_bootstrap(arm_res, ref_res, combine, stat, strata=night) ; classify_delta(ci, margin) ;
            trackeval_combiner(metric)
""".format(v=BENCH_VERSION, r=CORE_REVISION)


if __name__ == "__main__":
    if "--api" in sys.argv:
        print(api_summary())
    else:
        print(__doc__)
