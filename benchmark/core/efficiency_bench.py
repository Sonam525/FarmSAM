# Databricks notebook source
# MAGIC %md
# MAGIC # Efficiency benchmark
# MAGIC
# MAGIC Measures size, speed and memory on an A10 with the same streaming loop as the tracking runs
# MAGIC (`bench_core.track_stream`: transformers 5.2.0, bf16, SDPA, explicit `frame_idx`, frame-cache release, k16 pruning
# MAGIC every 25 frames, one `add_inputs` call per frame).
# MAGIC
# MAGIC | mode | what it measures | input |
# MAGIC |---|---|---|
# MAGIC | `micro` | one model at 4 objects: parameters, weights-only VRAM, CUDA context, per-module latency, memory curve | a calf window |
# MAGIC | `micro_decode` | decode-only throughput of the same window | a calf window |
# MAGIC | `e2e` | latency and memory with the RF-DETR detector live | PigTrack sequence or calf window |
# MAGIC | `factorial` | pruning levels on the Edinburgh test clips, from existing tracking runs | Edinburgh clips |
# MAGIC | `report` | CSV tables from the published JSON files of one tag | `efficiency/A10/**/<tag>` |
# MAGIC
# MAGIC Outputs go under the registry key `paper1.efficiency`. A finished result is reused on a re-run.

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
import contextlib
import csv
import gc
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

NB_NAME = "efficiency_bench"
NB_VERSION = "efficiency_bench-2026-10-20.1"
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None


def _this_dir():
    env = os.environ.get("FARMSAM_CODE_DIR")
    if env:
        return env
    if not IS_NOTEBOOK:
        return os.path.dirname(os.path.abspath(__file__))
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    return "/Workspace" + os.path.dirname(nbp)


def _import_exact(name, dirs):
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
os.environ.setdefault("FARMSAM_CODE_DIR", CODE_DIR)
bc = _import_exact("bench_core", [CODE_DIR])
_PATHS_DIRS = [CODE_DIR, os.environ.get("FARMSAM_PATHS_DIR")]
farm_paths = _import_exact("farm_paths", _PATHS_DIRS)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print(f"{NB_NAME} {NB_VERSION}: bench_core {bc.BENCH_VERSION} (sha256 {_sha256(bc.__file__)[:16]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Constants

# COMMAND ----------

MODES = ("micro", "micro_decode", "e2e", "factorial", "report")
S_ARMS = ("T", "S-Edi", "S-PT", "S-v2", "T21", "S21", "ET", "ES-L")   # §A4: the efficiency set = every S arm
STUDENT_ARMS = ("S-Edi", "S-PT", "S-v2")
MICRO_WINDOW = "0422_night"        # §8: growth slope over >= 2,000 frames of the 04-22 night window; 4 calves
WARMUP = bc.WARMUP_FRAMES          # 32
TIMED = bc.TIMED_FRAMES            # 500
MEM_FRAMES_MIN = 2000              # §8 / §11: the growth slope needs >= 2,000 frames
MEM_FRAMES_DEFAULT = 2500
TRANSIENT_FRAMES = 64              # per-module transient memory: frames WARMUP .. WARMUP + 63
FLOP_FRAME = 40                    # steady state: memory bank full (num_maskmem 7, object pointers 16) by frame 16
LAT_FINITE_EVERY = 25              # §8: finite_every = 25 in the latency pass (accuracy runs use 1)
POWER_PERIOD_MS = 100
IDLE_S = 10.0
SLOPE_TOL_MIB = 0.25               # §11 "fits an A10": growth within +-0.25 MiB / frame over >= 2,000 frames
A10_BUDGET_MIB = 24.0 * 1024       # §11 "fits an A10": weights + steady peak + context <= 24 GiB
FASTER_RATIO = 1.20                # §11 "faster end-to-end": median fps ratio >= 1.20 with non-overlapping IQRs
J13_PIG_SEQ = "pigtrack0002"       # the longest PigTrack test sequence (985 frames, 26 pigs): memory worst case
J13_PIG_ARMS = ("T", "S-PT")
J13_PIG_DETECTOR = "models.detector_pig_pt:checkpoint_best_total.pth"     # d_pt_v1
J13_PIG_RESOLUTION = 512
J13_CALF_WINDOW = "0422_night"
J13_CALF_ARMS = ("S-v2",)
J13_CALF_DETECTOR = "models.detector_segv1d"                               # Paper 2's RF-DETR-Seg-Small (sha-pinned)
J13_CALF_RESOLUTION = 624
J13_CALF_TAU = 0.35                # §5.4: segv1d's own operating threshold (train_summary.json op_threshold)
J13_CALF_FRAMES_DEFAULT = 2500
TAU_FILE = "detector_threshold.json"
FACTORIAL_LEVELS = (("none", "__none"), ("k8", "__k8"), ("k16", "__k16"), ("k16+pop", ""))
FACTORIAL_ARMS = ("T", "S-Edi")
TAG_RX = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
WIDGETS = collections.OrderedDict([
    ("mode", ""), ("arm", ""), ("tag", ""), ("e2e_target", ""), ("mem_frames", str(MEM_FRAMES_DEFAULT)),
    ("e2e_calf_frames", str(J13_CALF_FRAMES_DEFAULT)), ("factorial_prefix", "pf1"), ("report_arms", ",".join(S_ARMS)),
    ("revisions", "protocol:hf_revisions.json"), ("require_gpu", "A10"), ("publish", "true"), ("progress_every", "500"),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pure helpers (statistics, power integration, FLOP grouping, analytic budget)

# COMMAND ----------


def stats(v):
    v = np.asarray([x for x in v if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def per_frame_walls(t_done, lo, hi):
    """t_done {fi: perf_counter when frame fi finished}; per-frame wall (ms) of frames lo..hi-1 = t[fi] - t[fi-1]."""
    out = []
    for fi in range(lo, hi):
        if fi in t_done and fi - 1 in t_done:
            out.append((t_done[fi] - t_done[fi - 1]) * 1000.0)
    return out


def integrate_power(samples, t0, t1):
    """samples [(t_s, watts)] (host clock, sorted); trapezoid of P(t) over [t0, t1] with linear interpolation at the
    edges. -> (joules, n_samples_inside, mean_watts) or (None, n, None) when fewer than 2 samples bracket the window."""
    s = sorted((float(t), float(w)) for t, w in samples if w is not None and math.isfinite(w))
    if len(s) < 2 or t1 <= t0 or s[0][0] > t0 or s[-1][0] < t1:
        return None, sum(1 for t, _ in s if t0 <= t <= t1), None

    def at(t):
        for (ta, wa), (tb, wb) in zip(s, s[1:]):
            if ta <= t <= tb:
                return wa if tb == ta else wa + (wb - wa) * (t - ta) / (tb - ta)
        return s[-1][1]
    pts = [(t0, at(t0))] + [(t, w) for t, w in s if t0 < t < t1] + [(t1, at(t1))]
    j = sum((tb - ta) * (wa + wb) / 2.0 for (ta, wa), (tb, wb) in zip(pts, pts[1:]))
    return j, sum(1 for t, _ in s if t0 <= t <= t1), j / (t1 - t0)


def parse_smi_line(line):
    """'power.draw, utilization.gpu, clocks.sm' CSV (noheader, nounits) -> dict; '[N/A]' / 'N/A' -> None."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 3:
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None
    return {"power_w": num(parts[0]), "util_pct": num(parts[1]), "sm_mhz": num(parts[2])}


def group_flops(counts, groups=bc.MODULE_GROUPS):
    """FlopCounterMode.get_flop_counts() {module path: {op: flops}} -> {"total", per MODULE_GROUP, "student"}. Module
    paths are '<RootClass>.<path>'; a group matches the path whose suffix after the root class is exactly the group
    (the HF graph: SAM 3 / SAM 2 / EdgeTAM share the names); the student is its own root (StudentEncoderTinyViT)."""
    def tot(d):
        return int(sum(int(v) for v in d.values()))
    out = {"total": tot(counts.get("Global", {})), "groups": {}, "student": None}
    for path, d in counts.items():
        if path == "Global":
            continue
        root, _, rest = path.partition(".")
        if rest in groups:
            out["groups"][rest] = out["groups"].get(rest, 0) + tot(d)
        if root == "StudentEncoderTinyViT" and not rest:
            out["student"] = tot(d)
    out["groups_sum"] = int(sum(out["groups"].values()))
    out["other"] = out["total"] - out["groups_sum"] - (out["student"] or 0)
    return out


def ratio_statement(fps_s, fps_t, iqr_s, iqr_t, what):
    """§11: 'faster end-to-end' only with median ratio >= 1.20 AND non-overlapping IQRs; else 'about equal'."""
    if not fps_s or not fps_t:
        return None
    r = fps_s / fps_t
    disjoint = iqr_s is not None and iqr_t is not None and (iqr_s[0] > iqr_t[1] or iqr_s[1] < iqr_t[0])
    if r >= FASTER_RATIO and disjoint:
        return {"ratio": r, "wording": f"faster end-to-end ({what}: {r:.2f}x the teacher's median fps, IQRs disjoint)"}
    return {"ratio": r, "wording": f"end-to-end speed is about equal (ratio {r:.2f}; {what})",
            "why_not_faster": ("ratio < 1.20" if r < FASTER_RATIO else "IQRs overlap")}


def fits_a10(weights_mib, steady_peak_reserved_mib, context_mib, slope, n_frames):
    """§11 'fits an A10': measured (weights in the peak) + context <= 24 GiB AND growth within +-0.25 MiB / frame
    over >= 2,000 frames. steady_peak_reserved already contains the weights (allocator view of the whole process)."""
    total = (steady_peak_reserved_mib or 0.0) + (context_mib or 0.0)
    ok_mem = bool(steady_peak_reserved_mib is not None and total <= A10_BUDGET_MIB)
    ok_slope = bool(slope is not None and abs(slope) <= SLOPE_TOL_MIB and n_frames >= MEM_FRAMES_MIN)
    return {"total_mib": round(total, 1), "budget_mib": A10_BUDGET_MIB, "mem_ok": ok_mem, "slope_ok": ok_slope,
            "slope_mib_per_frame": slope, "frames": n_frames, "fits": ok_mem and ok_slope,
            "rule": "weights + steady peak (reserved) + CUDA context <= 24 GiB and |growth| <= 0.25 MiB/frame over "
                    ">= 2,000 frames"}


def analytic_streams(weights_mib, steady_alloc_mib, context_mib, budget_mib=A10_BUDGET_MIB):
    """Concurrent streams per A10 at a fixed budget, ANALYTIC (labelled so, §8): one copy of the weights + context,
    each stream adds its steady per-stream state (steady allocated - weights)."""
    per = (steady_alloc_mib or 0.0) - (weights_mib or 0.0)
    if per <= 0:
        return None
    free = budget_mib - (weights_mib or 0.0) - (context_mib or 0.0)
    return {"per_stream_mib": round(per, 1), "streams": int(free // per), "budget_mib": budget_mib,
            "label": "analytic: one copy of the weights + CUDA context; each stream adds its steady allocated state"}


def growth(samples, key, warmup=WARMUP):
    fis = [m["fi"] for m in samples if m.get(key) is not None]
    ys = [m[key] for m in samples if m.get(key) is not None]
    return bc.growth_slope(fis, ys, warmup)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Instruments: power sampler, decode-wait wrapper, transient-memory hooks, FLOP frame

# COMMAND ----------


class PowerSampler:
    """nvidia-smi -lms sampler on a reader thread: [(host time, watts, util, sm MHz)]. On a vGPU the driver may not
    report power ('[N/A]'): available=False is recorded, never a fabricated number."""

    def __init__(self, period_ms=POWER_PERIOD_MS, cmd=None, clock=time.perf_counter):
        self.cmd = cmd or ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,clocks.sm", "--format=csv,noheader,nounits",
                           "-lms", str(int(period_ms))]
        self.clock, self.samples, self.raw_first, self.proc, self.th, self.err = clock, [], [], None, None, None

    def start(self):
        try:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        except Exception as e:  # noqa: BLE001
            self.err = f"{type(e).__name__}: {e}"
            return self

        def rd():
            for line in self.proc.stdout:
                t = self.clock()
                if len(self.raw_first) < 3:
                    self.raw_first.append(line.strip())
                d = parse_smi_line(line)
                if d is not None:
                    self.samples.append((t, d["power_w"], d["util_pct"], d["sm_mhz"]))
        self.th = threading.Thread(target=rd, daemon=True, name="tol65_power")
        self.th.start()
        return self

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    self.proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        if self.th is not None:
            self.th.join(timeout=10)
        return self

    def available(self):
        return any(w is not None for _t, w, _u, _c in self.samples)

    def energy(self, t0, t1):
        j, n, mean_w = integrate_power([(t, w) for t, w, _u, _c in self.samples], t0, t1)
        return {"joules": j, "n_samples": n, "mean_watts": mean_w}

    def mean_over(self, t0, t1):
        w = [x for t, x, _u, _c in self.samples if t0 <= t <= t1 and x is not None]
        u = [x for t, _w, x, _c in self.samples if t0 <= t <= t1 and x is not None]
        return {"mean_watts": float(np.mean(w)) if w else None, "mean_util_pct": float(np.mean(u)) if u else None,
                "n": len(w)}


class DecodeWait:
    """Wrap a frame source: per-frame time the loop WAITED for the next decoded frame (the exposed decode / IO cost;
    the calf decoder prefetches on a seek thread, PigTrack / Edinburgh JPEGs on a prefetch thread)."""

    def __init__(self, frames, n=None):
        self.frames, self.n, self.wait_ms = frames, n, {}
        for k in ("fps", "name", "kind", "cap", "native_fps", "stride", "fidxs", "gt_path"):
            if hasattr(frames, k):
                setattr(self, k, getattr(frames, k))

    def __len__(self):
        return min(len(self.frames), self.n) if self.n is not None else len(self.frames)

    def __iter__(self):
        it = iter(self.frames)
        try:
            k = 0
            while self.n is None or k < self.n:
                t0 = time.perf_counter()
                try:
                    item = next(it)
                except StopIteration:
                    return
                self.wait_ms[item[0]] = (time.perf_counter() - t0) * 1000.0
                yield item
                k += 1
        finally:
            c = getattr(it, "close", None)
            if callable(c):
                c()
            rel = getattr(self.frames, "close", None)       # the source owns a cv2 capture (calves): release it
            if callable(rel):
                rel()


class TransientHooks:
    """Per-module transient memory: on entry record allocated a0 and reset the peak; on exit peak - a0 (MiB) = the
    module's own transient. Resetting the peak inside the loop invalidates the global peak, hence its own pass."""

    def __init__(self, model, names=bc.MODULE_GROUPS, torch_mod=None):
        self.t = torch_mod
        self.fi, self.rec, self.hooks, self._open, self.attached = -1, collections.defaultdict(dict), [], {}, []
        mods = dict(model.named_modules()) if hasattr(model, "named_modules") else {}
        for n in names:
            m = mods.get(n)
            if m is None:
                continue
            self.hooks.append(m.register_forward_pre_hook(lambda _m, _a, n=n: self._start(n)))
            self.hooks.append(m.register_forward_hook(lambda _m, _a, _o, n=n: self._stop(n)))
            self.attached.append(n)

    def _alloc(self):
        return self.t.cuda.memory_allocated() / 2 ** 20

    def _start(self, n):
        if self.t is None:
            return
        self._open[n] = self._alloc()
        self.t.cuda.reset_peak_memory_stats()

    def _stop(self, n):
        if self.t is None or n not in self._open:
            return
        a0 = self._open.pop(n)
        pk = self.t.cuda.max_memory_allocated() / 2 ** 20 - a0
        d = self.rec[n]
        d[self.fi] = max(d.get(self.fi, 0.0), pk)        # per-object calls (SAM 2 family): the largest call

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def summary(self, lo, hi):
        return {n: stats([v for fi, v in d.items() if lo <= fi < hi]) for n, d in sorted(self.rec.items())}


def flop_counter_for_frame(arm, frame, sink, torch_mod, math=True):
    """Replace arm.forward for ONE frame by a FlopCounterMode-wrapped call under SDPBackend.MATH (§8 d: 2 FLOPs per
    MAC, FlopCounterMode's own convention). Restores the method afterwards via the returned undo(). math=False (only
    if MATH runs out of memory) keeps the default SDPA kernels, whose FlopCounterMode formulas count the same QK^T and AV
    matmuls."""
    orig = arm.forward
    had_own = "forward" in getattr(arm, "__dict__", {})

    def fwd(sess, fi, inputs, pil, tel=None, check=False):
        if fi != frame:
            return orig(sess, fi, inputs, pil, tel=tel, check=check)
        from torch.utils.flop_counter import FlopCounterMode
        from torch.nn.attention import SDPBackend, sdpa_kernel
        fc = FlopCounterMode(display=False)
        with (sdpa_kernel(SDPBackend.MATH) if math else contextlib.nullcontext()), fc:
            out = orig(sess, fi, inputs, pil, tel=tel, check=check)
        sink["counts"] = {str(k): {str(op): int(v) for op, v in d.items()} for k, d in fc.get_flop_counts().items()}
        sink["total"] = int(fc.get_total_flops())
        sink["frame"] = int(fi)
        return out
    arm.forward = fwd

    def undo():
        if had_own:
            arm.forward = orig
        else:
            del arm.forward                      # back to the class method
    return undo


# COMMAND ----------

# MAGIC %md
# MAGIC ## Inputs: registry, calf window, PigTrack sequence, detectors, arms

# COMMAND ----------


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json_local(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp_{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, default=_jd) + "\n")
    os.replace(tmp, path)
    return path


def calf_window(R, name):
    p = R.path("paper1.calf_windows_protocol")
    with open(p, "rb") as f:
        raw = f.read()
    J = json.loads(raw.decode("utf-8"))
    if J.get("version") != "calf_windows_v1" or name not in (J.get("windows") or {}):
        raise ValueError(f"{name} is not a window of calf_windows_v1.json")
    w = dict(J["windows"][name], name=name)
    return w, {"file": "calf_windows_v1.json", "sha256": hashlib.sha256(raw).hexdigest()}


def calf_prompts(w):
    masks = {int(k): bc.rle_decode(v).astype(bool) for k, v in w["seed_masks"].items()}
    return bc.prompts_from_masks(masks, frame=0), {str(k): int(m.sum()) for k, m in sorted(masks.items())}


def resolve_revisions(R, spec):
    if not spec:
        return {}
    if spec.startswith("protocol:"):
        with open(bc.protocol_file_path(R, spec[len("protocol:"):]), encoding="utf-8") as f:
            obj = json.load(f)
        obj = obj.get("revisions", obj)
    else:
        obj = json.loads(spec)
    return dict(sorted(obj.items()))


def stage_detector(ctx, spec, resolution):
    key, _, rel = spec.partition(":")
    src = ctx.R.path(key, None, rel) if rel else ctx.R.path(key)
    pin = None if rel else ctx.R.entry(key).get("sha256")
    local = os.path.join(ctx.local_root, "detectors", key.replace(".", "_"), os.path.basename(src))
    loc, sha = bc.stage_file(src, local, ctx.dbutils, pin)
    ts_path = os.path.join(os.path.dirname(src), "train_summary.json")
    ts = None
    if os.path.isfile(ts_path):
        with open(ts_path, encoding="utf-8") as f:
            ts = json.load(f)
        if ts.get("checkpoint_best_total_sha256") and ts["checkpoint_best_total_sha256"] != sha:
            raise RuntimeError(f"{spec}: train_summary.json names {ts['checkpoint_best_total_sha256'][:16]}, staged {sha[:16]}")
        if ts.get("resolution") is not None and int(ts["resolution"]) != int(resolution):
            raise ValueError(f"{spec}: resolution {resolution} != train_summary.json {ts['resolution']}")
    return {"spec": spec, "local": loc, "sha256": sha, "registry_pin": pin, "resolution": int(resolution),
            "op_threshold": None if ts is None else ts.get("op_threshold")}


def frozen_tau(R):
    p = bc.protocol_file_path(R, TAU_FILE)
    with open(p, "rb") as f:
        raw = f.read()
    return round(float(json.loads(raw.decode("utf-8"))["tau"]), 2), hashlib.sha256(raw).hexdigest()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Passes (one arm, one input)

# COMMAND ----------


def _cuda():
    if not bc.TORCH_OK:
        return None
    import torch
    return torch if torch.cuda.is_available() else None


def mem_now():
    t = _cuda()
    if t is None:
        return None
    t.cuda.synchronize()
    free, total = t.cuda.mem_get_info()
    m = 2 ** 20
    return {"alloc": t.cuda.memory_allocated() / m, "reserved": t.cuda.memory_reserved() / m,
            "device_used": (total - free) / m, "device_total": total / m,
            "context_est": (total - free) / m - t.cuda.memory_reserved() / m}


def reset_peak():
    t = _cuda()
    if t is not None:
        t.cuda.synchronize()
        t.cuda.reset_peak_memory_stats()


def peaks():
    t = _cuda()
    if t is None:
        return None
    t.cuda.synchronize()
    m = 2 ** 20
    return {"peak_alloc": t.cuda.max_memory_allocated() / m, "peak_reserved": t.cuda.max_memory_reserved() / m}


def latency_pass(ctx, arm, open_frames, prompts, n_frames, *, protocol="A", detector=None, protd=None, power=True):
    """§8 (a): module hooks + spans, finite_every = 25, per-frame wall, decode wait, energy (power x time)."""
    tel = bc.Telemetry(arm.model, enabled=True, use_cuda=ctx.use_cuda)
    t_done = {}

    def on_frame(fi, **_k):
        t_done[fi] = time.perf_counter()
    ps = ctx.hooks["power_sampler"]() if power else None
    idle = None
    if ps is not None:
        ps.start()
        t_i0 = time.perf_counter()
        time.sleep(ctx.idle_s)
        idle = ps.mean_over(t_i0, time.perf_counter())
    src = DecodeWait(open_frames(), n_frames)
    t_start = time.perf_counter()
    try:
        st = bc.track_stream(arm, src, protocol=protocol, prompts=prompts if protocol == "A" else (), detector=detector,
                             protd=protd, tel=tel, finite_every=LAT_FINITE_EVERY, on_frame=on_frame,
                             progress_every=ctx.progress_every, log=ctx.log)
    finally:
        tel.detach()
        if ps is not None:
            ps.stop()
    s = tel.summary()
    lo, hi = WARMUP, max(t_done) + 1 if t_done else 0
    walls = per_frame_walls(t_done, lo, hi)
    rec = {"frames": st["frames"], "forwards": st["forwards"], "objects": st["objects"], "warmup": WARMUP,
           "timed_frames": len(walls), "gpu_ms": s["gpu_ms"], "cpu_ms": s["cpu_ms"],
           "modules_attached": s["modules_attached"], "clock": s["clock"],
           "wall_ms_per_frame": stats(walls),
           "decode_wait_ms": stats([v for fi, v in src.wait_ms.items() if lo <= fi < hi]),
           "frame_clear_s": st.get("frame_clear_s"), "frames_cleared_in_loop": st.get("frames_cleared_in_loop"),
           "loop_wall_s": round(time.perf_counter() - t_start, 3), "track_wall_s": st.get("wall_s"),
           "finite_every": LAT_FINITE_EVERY, "teardown": st.get("teardown"), "protocol_D": st.get("protocol_D"),
           "births": st.get("births")[:50] if st.get("births") else []}
    w = rec["wall_ms_per_frame"]
    if w.get("n"):
        rec["e2e_fps"] = {"median": 1000.0 / w["median"], "iqr": [1000.0 / w["p75"], 1000.0 / w["p25"]],
                          "mean_over_timed": 1000.0 * len(walls) / sum(walls),
                          "note": "per-frame wall of the streaming loop INCLUDING the exposed decode wait, the "
                                  "finite assert every 25 frames and the 250-frame frame clearing"}
    if ps is not None:
        t0, t1 = t_done.get(lo - 1), t_done.get(hi - 1)
        e = ps.energy(t0, t1) if (t0 is not None and t1 is not None) else {"joules": None}
        n_t = (hi - lo)
        rec["energy"] = {"available": ps.available(), "idle": idle, "window_s": None if t0 is None else t1 - t0,
                         "joules_timed": e.get("joules"), "n_samples": e.get("n_samples"),
                         "mean_watts": e.get("mean_watts"),
                         "j_per_frame": (e["joules"] / n_t) if e.get("joules") is not None and n_t else None,
                         "j_per_frame_above_idle": ((e["joules"] - idle["mean_watts"] * (t1 - t0)) / n_t)
                         if e.get("joules") is not None and idle and idle.get("mean_watts") is not None and n_t else None,
                         "smi_first_lines": ps.raw_first, "error": ps.err,
                         "method": "nvidia-smi power.draw sampled every 100 ms, trapezoid over the timed frames; "
                                   "board power of the whole GPU (not per process)"}
    return rec


def memory_pass(ctx, arm, open_frames, prompts, n_frames, *, protocol="A", detector=None, protd=None):
    """§8 (b): no module hooks (spans only), peak reset after the warm-up, post-prune memory samples +
    session_tensor_bytes, growth slope (OLS and median step)."""
    tel = bc.Telemetry(None, enabled=True, use_cuda=ctx.use_cuda)
    sess_mib = {}
    base = {}

    def on_frame(fi, sess=None, pruned=False, **_k):
        if fi == WARMUP - 1:
            reset_peak()
            base.update(mem_now() or {})
        if pruned:
            sess_mib[fi] = bc.session_tensor_bytes(sess)["total_mib"]
    st = bc.track_stream(arm, DecodeWait(open_frames(), n_frames), protocol=protocol,
                         prompts=prompts if protocol == "A" else (), detector=detector, protd=protd, tel=tel,
                         on_frame=on_frame, progress_every=ctx.progress_every, log=ctx.log)
    pk = peaks()
    s = tel.summary()
    ms = s.get("mem_samples") or []
    for m in ms:
        m["session_mib"] = sess_mib.get(m["fi"])
    steady = [m for m in ms if m["fi"] >= WARMUP]
    rec = {"frames": st["frames"], "objects": st["objects"], "warmup": WARMUP, "after_warmup": base or None,
           "peak_after_warmup": pk, "mem_samples": ms,
           "steady_alloc_mib": stats([m.get("alloc") for m in steady]),
           "steady_reserved_mib": stats([m.get("reserved") for m in steady]),
           "session_mib": stats([m.get("session_mib") for m in steady]),
           "growth_alloc": growth(ms, "alloc"), "growth_session": growth(ms, "session_mib"),
           "growth_reserved": growth(ms, "reserved"),
           "frame_clear_s": st.get("frame_clear_s"), "teardown": st.get("teardown"), "protocol_D": st.get("protocol_D"),
           "noncond_after_prune_max": st.get("noncond_after_prune_max"), "max_processed_frames": st.get("max_processed_frames")}
    return rec


def transient_pass(ctx, arm, open_frames, prompts, n_frames):
    """§8 (c): per-module transient memory (TransientHooks), frames WARMUP .. WARMUP + TRANSIENT_FRAMES - 1."""
    th = TransientHooks(arm.model, torch_mod=_cuda())

    def on_frame(fi, **_k):
        th.fi = fi + 1                                  # the hooks of the NEXT frame's forward record under fi + 1
    th.fi = 0
    try:
        st = bc.track_stream(arm, DecodeWait(open_frames(), n_frames), prompts=prompts, on_frame=on_frame,
                             log=ctx.log)
    finally:
        th.detach()
    return {"frames": st["frames"], "attached": th.attached, "transient_mib": th.summary(WARMUP, WARMUP + TRANSIENT_FRAMES),
            "note": "peak allocated inside one module call minus allocated at its entry (largest call per frame)"}


def flops_pass(ctx, arm, open_frames, prompts, math=True):
    """§8 (d): FlopCounterMode (2 FLOPs per MAC) on frame FLOP_FRAME under SDPBackend.MATH, per module group."""
    sink = {}
    t = _cuda() or (__import__("torch") if bc.TORCH_OK else None)
    undo = flop_counter_for_frame(arm, FLOP_FRAME, sink, t, math=math) if t is not None else (lambda: None)
    try:
        st = bc.track_stream(arm, DecodeWait(open_frames(), FLOP_FRAME + 1), prompts=prompts, log=ctx.log)
    finally:
        undo()
    if not sink:
        return {"frames": st["frames"], "available": False}
    g = group_flops(sink["counts"])
    G = 1e9
    return {"frames": st["frames"], "frame": sink["frame"], "objects": st["objects"], "available": True,
            "sdpa_backend": "MATH" if math else "default (MATH ran out of memory)",
            "gflops_total": sink["total"] / G, "gmacs_total": sink["total"] / 2 / G,
            "gflops_by_group": {k: v / G for k, v in sorted(g["groups"].items())},
            "gflops_student": None if g["student"] is None else g["student"] / G,
            "gflops_other": g["other"] / G, "counts": sink["counts"],
            "convention": "torch.utils.flop_counter.FlopCounterMode: 2 FLOPs per multiply-accumulate (GMACs = GFLOPs / 2), "
                          "matmul / conv / attention ops only (elementwise ops are not counted), SDPBackend.MATH, "
                          "one steady-state frame (frame 40) at each model's native resolution"}


# COMMAND ----------

# MAGIC %md
# MAGIC ## Modes

# COMMAND ----------


class Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def out_paths(ctx, arm_dir, fname):
    return (os.path.join(ctx.local_root, "out", arm_dir, fname),
            ctx.R.out("paper1.efficiency", ctx.gpu_class, arm_dir, fname))


def publish(ctx, arm_dir, fname, obj=None, local=None):
    loc, dst = out_paths(ctx, arm_dir, fname)
    if obj is not None:
        write_json_local(loc, obj)
    elif local is not None:
        loc = local
    if not ctx.publish:
        return {"local": loc, "published": None}
    return {"local": loc, "published": dst, "status": bc.publish_no_overwrite(loc, dst, ctx.dbutils)}


def done(ctx, arm_dir, fname):
    _loc, dst = out_paths(ctx, arm_dir, fname)
    if ctx.publish and os.path.isfile(dst):
        with open(dst, encoding="utf-8") as f:
            return json.load(f)
    return None


def write_curve_csv(path, samples):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = ["fi", "n_obj", "alloc", "reserved", "peak_alloc", "peak_reserved", "session_mib", "n_pf", "n_noncond"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for m in samples:
            w.writerow([m.get(k) for k in keys])
    return path


def load_arm_ctx(ctx, name):
    m0 = mem_now()
    arm = ctx.hooks["arm_loader"](ctx, name)
    m1 = mem_now()
    info = json.loads(json.dumps(arm.info, default=str))
    return arm, {"before_load": m0, "after_load": m1, "weights_alloc_mib": None if m1 is None else m1["alloc"] - (m0 or {}).get("alloc", 0.0),
                 "context_mib": None if m1 is None else m1["context_est"], "info": info}


def _settle_check(ctx, what, baseline, first):
    """track_bench's rule between passes: before the first pass and after it, within MEM_RETURN_TOL_MIB of the weights-only
    baseline (the first pass builds lazy caches: cuBLAS workspaces, TinyViT attention-bias tables, J0b); before every
    later pass within MEM_DRIFT_TOL_MIB of the level the FIRST pass's teardown left. A failure raises (no contaminated
    peak is ever published)."""
    now = bc.settle_memory()
    n = first.get("n", 0)
    if first.get("ref") is None:
        rec = bc.check_memory_returned(baseline, now, tol_mib=bc.MEM_RETURN_TOL_MIB, what=what)
        if n >= 1 and now is not None:                 # this check follows the first pass: the drift reference
            first["ref"] = now
    else:
        rec = bc.check_memory_returned(first["ref"], now, tol_mib=bc.MEM_DRIFT_TOL_MIB, what=what)
    first["n"] = n + 1
    return rec


def mode_micro(ctx, name):
    fname = f"micro__{ctx.tag}.json"
    prev = done(ctx, name, fname)
    if prev is not None:
        ctx.log(f"{name}: micro__{ctx.tag} already published: reused")
        return prev
    w, wprov = calf_window(ctx.R, MICRO_WINDOW)
    mp4, sha = ctx.hooks["stage_clip"](ctx, w)
    prompts, areas = calf_prompts(w)
    opener = lambda: ctx.hooks["calf_frames"](mp4, w)           # noqa: E731
    arm, load = load_arm_ctx(ctx, name)
    baseline, first = (load["after_load"] or {}).get("alloc"), {}
    rec = {"notebook": f"{NB_NAME} {NB_VERSION}", "core_sha256": _sha256(bc.__file__), "mode": "micro", "arm": name,
           "tag": ctx.tag, "created_utc": _now(), "gpu": bc.gpu_info(), "versions": bc.versions(),
           "input": {"calf_window": MICRO_WINDOW, "windows_file": wprov, "clip": w["clip"], "clip_sha256": sha,
                     "start_fidx": w["start_fidx"], "objects": len(prompts), "seed_areas": areas},
           "stream": {"keep_last_n": bc.KEEP_LAST_N, "prune_every": bc.PRUNE_EVERY, "pop": bc.POP_FRAMES,
                      "frame_clear_every": bc.FRAME_CLEAR_EVERY, "dtype": bc.DTYPE_NAME, "attn": bc.ATTN_IMPL},
           "load": load, "params": load["info"].get("params")}
    passes = (("latency", lambda: latency_pass(ctx, arm, opener, prompts, WARMUP + TIMED)),
              ("memory", lambda: memory_pass(ctx, arm, opener, prompts, ctx.mem_frames)),
              ("transient", lambda: transient_pass(ctx, arm, opener, prompts, WARMUP + TRANSIENT_FRAMES)))
    rec["memory_checks"] = []
    for pname, fn in passes:
        rec["memory_checks"].append(_settle_check(ctx, f"{name} before {pname}", baseline, first))
        t0 = time.perf_counter()
        rec[pname] = fn()
        rec[pname]["pass_s"] = round(time.perf_counter() - t0, 1)
        ctx.log(f"{name}: {pname} pass {rec[pname]['pass_s']} s")
    rec["memory_checks"].append(_settle_check(ctx, f"{name} before flops", baseline, first))
    t0 = time.perf_counter()                       # LAST: FLOP counting needs no clean allocator afterwards
    try:
        rec["flops"] = flops_pass(ctx, arm, opener, prompts, math=True)
    except Exception as e:  # noqa: BLE001 -- MATH attention materialises the full attention matrices
        if "out of memory" not in str(e).lower():
            raise
        rec["flops_math_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        bc.settle_memory()
        rec["flops"] = flops_pass(ctx, arm, opener, prompts, math=False)
    rec["flops"]["pass_s"] = round(time.perf_counter() - t0, 1)
    try:
        rec["memory_checks"].append(_settle_check(ctx, f"{name} after all passes", baseline, first))
    except bc.BenchAssertError as e:                # every number is already measured: record, do not fail
        rec["memory_checks"].append({"what": f"{name} after all passes", "ok": False, "error": str(e)[:300]})
    mem = rec["memory"]
    rec["fits_a10"] = fits_a10(load["weights_alloc_mib"], (mem.get("peak_after_warmup") or {}).get("peak_reserved"),
                               load["context_mib"], (mem["growth_alloc"] or {}).get("ols_mib_per_frame"), mem["frames"])
    rec["streams_analytic"] = analytic_streams((load["after_load"] or {}).get("alloc"),
                                               (mem.get("steady_alloc_mib") or {}).get("median"), load["context_mib"])
    curve = write_curve_csv(os.path.join(ctx.local_root, "out", name, f"micro_memcurve__{ctx.tag}.csv"), mem["mem_samples"])
    rec["published"] = {"curve": publish(ctx, name, f"micro_memcurve__{ctx.tag}.csv", local=curve)}
    rec["published"]["json"] = publish(ctx, name, fname, rec)
    del arm
    bc.settle_memory()
    return rec


def mode_micro_decode(ctx):
    fname = f"micro_decode__{ctx.tag}.json"
    prev = done(ctx, "_decode", fname)
    if prev is not None:
        return prev
    w, wprov = calf_window(ctx.R, MICRO_WINDOW)
    mp4, sha = ctx.hooks["stage_clip"](ctx, w)
    out = {}
    for n in (WARMUP + TIMED,):
        src = DecodeWait(ctx.hooks["calf_frames"](mp4, w), n)
        t0 = time.perf_counter()
        k = 0
        for _fi, _pil, _m in src:
            k += 1
        dt = time.perf_counter() - t0
        out[str(n)] = {"frames": k, "wall_s": dt, "fps": k / dt if dt else None,
                       "per_frame_wait_ms": stats([v for fi, v in src.wait_ms.items() if fi >= WARMUP])}
    rec = {"notebook": f"{NB_NAME} {NB_VERSION}", "mode": "micro_decode", "tag": ctx.tag, "created_utc": _now(),
           "input": {"calf_window": MICRO_WINDOW, "windows_file": wprov, "clip": w["clip"], "clip_sha256": sha},
           "decode": out, "gpu": bc.gpu_info(),
           "note": "v5 decoder contract (cv2, seek thread, 2304x1296, every 5th native frame), PIL conversion; no model"}
    rec["published"] = publish(ctx, "_decode", fname, rec)
    return rec


def mode_e2e(ctx, name, target):
    fname = f"e2e_{target}__{ctx.tag}.json"
    prev = done(ctx, name, fname)
    if prev is not None:
        return prev
    if target == "pigtrack":
        if name not in J13_PIG_ARMS:
            raise ValueError(f"e2e pigtrack arms are {J13_PIG_ARMS}")
        seq_dir = ctx.hooks["stage_pig_seq"](ctx, J13_PIG_SEQ)
        opener = lambda: bc.PigTrackFrames(seq_dir)              # noqa: E731
        fps = opener().fps
        tau, tau_sha = frozen_tau(ctx.R)
        det = stage_detector(ctx, J13_PIG_DETECTOR, J13_PIG_RESOLUTION)
        inp = {"dataset": "pigtrack", "seq": J13_PIG_SEQ, "fps": fps, "tau": tau, "tau_file_sha256": tau_sha}
    elif target == "calf":
        if name not in J13_CALF_ARMS:
            raise ValueError(f"e2e calf arms are {J13_CALF_ARMS}")
        w, wprov = calf_window(ctx.R, J13_CALF_WINDOW)
        mp4, sha = ctx.hooks["stage_clip"](ctx, w)
        n_max = min(int(w["n_frames"]), ctx.e2e_calf_frames)
        opener = lambda: ctx.hooks["calf_frames"](mp4, w)       # noqa: E731
        fps = bc.CALF_TRACK_FPS
        det = stage_detector(ctx, J13_CALF_DETECTOR, J13_CALF_RESOLUTION)
        if det["op_threshold"] is not None and abs(float(det["op_threshold"]) - J13_CALF_TAU) > 1e-9:
            raise ValueError(f"segv1d op_threshold {det['op_threshold']} != the fixed {J13_CALF_TAU}")
        tau = J13_CALF_TAU
        inp = {"dataset": "calves_tol", "window": J13_CALF_WINDOW, "windows_file": wprov, "clip": w["clip"],
               "clip_sha256": sha, "frames": n_max, "tau": tau}
    else:
        raise ValueError(f"e2e_target {target!r}: pigtrack | calf")
    m0 = mem_now()
    live = ctx.hooks["detector_loader"](det["local"], det["resolution"])
    m_det = mem_now()
    arm, load = load_arm_ctx(ctx, name)
    baseline, first = (load["after_load"] or {}).get("alloc"), {}
    n_frames = None if target == "pigtrack" else n_max
    rec = {"notebook": f"{NB_NAME} {NB_VERSION}", "core_sha256": _sha256(bc.__file__), "mode": "e2e", "target": target,
           "arm": name, "tag": ctx.tag, "created_utc": _now(), "gpu": bc.gpu_info(), "versions": bc.versions(),
           "input": inp, "detector": dict(det, info=getattr(live, "info", None),
                                          weights_alloc_mib=None if m_det is None else m_det["alloc"] - (m0 or {}).get("alloc", 0.0)),
           "load": load, "params": load["info"].get("params"), "memory_checks": []}
    for pname, fn in (("latency", latency_pass), ("memory", memory_pass)):
        rec["memory_checks"].append(_settle_check(ctx, f"{name} e2e before {pname}", baseline, first))
        protd = bc.ProtocolD(fps, tau)
        t0 = time.perf_counter()
        kw = dict(protocol="D", detector=live, protd=protd)
        rec[pname] = fn(ctx, arm, opener, (), n_frames, **kw)
        rec[pname]["pass_s"] = round(time.perf_counter() - t0, 1)
        ctx.log(f"{name} e2e {target}: {pname} pass {rec[pname]['pass_s']} s")
    rec["memory_checks"].append(_settle_check(ctx, f"{name} e2e after all passes", baseline, first))
    mem = rec["memory"]
    rec["fits_a10"] = fits_a10(load["weights_alloc_mib"], (mem.get("peak_after_warmup") or {}).get("peak_reserved"),
                               load["context_mib"], (mem["growth_alloc"] or {}).get("ols_mib_per_frame"), mem["frames"])
    curve = write_curve_csv(os.path.join(ctx.local_root, "out", name, f"e2e_{target}_memcurve__{ctx.tag}.csv"), mem["mem_samples"])
    rec["published"] = {"curve": publish(ctx, name, f"e2e_{target}_memcurve__{ctx.tag}.csv", local=curve)}
    rec["published"]["json"] = publish(ctx, name, fname, rec)
    del arm, live
    bc.settle_memory()
    return rec


# ---- the pruning factorial (track_bench runs + the scorer scores -> one table) ------------------------------------------------
def _read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def latest_score(scores_root, dataset, protocol, split, variant):
    d = os.path.join(scores_root, dataset, f"prot{protocol}", split, *([variant] if variant else []))
    if not os.path.isdir(d):
        return None
    ids = sorted(x for x in os.listdir(d) if re.match(r"^\d{8}T\d{6}Z_[0-9a-f]{8}$", x)
                 and os.path.isfile(os.path.join(d, x, "score.json")))
    return os.path.join(d, ids[-1]) if ids else None


def run_dir_curves(rd, seqs):
    out = {}
    for s in seqs:
        tp = os.path.join(rd, "telemetry", f"{s}.json")
        if not os.path.isfile(tp):
            out[s] = None
            continue
        tel = _read_json(tp)
        ms = None
        stack = [tel]
        while stack and ms is None:
            o = stack.pop()
            if isinstance(o, dict):
                if "mem_samples" in o and isinstance(o["mem_samples"], list):
                    ms = o["mem_samples"]
                else:
                    stack.extend(o.values())
        ms = ms or []
        g = growth(ms, "alloc")
        out[s] = {"n_samples": len(ms), "peak_alloc_mib": max((m.get("peak_alloc") or m.get("alloc") or 0.0) for m in ms) if ms else None,
                  "max_alloc_mib": max((m.get("alloc") or 0.0) for m in ms) if ms else None,
                  "end_alloc_mib": ms[-1].get("alloc") if ms else None, "slope": g,
                  "curve": [[m["fi"], m.get("alloc"), m.get("n_pf"), m.get("n_noncond")] for m in ms]}
    return out


def files_identical(a, b, seqs, sub, ext):
    res = {}
    for s in seqs:
        pa, pb = os.path.join(a, sub, f"{s}{ext}"), os.path.join(b, sub, f"{s}{ext}")
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            res[s] = None
            continue
        res[s] = _sha256(pa) == _sha256(pb)
    vals = [v for v in res.values() if v is not None]
    return {"per_seq": res, "all_identical": bool(vals) and all(vals) and len(vals) == len(seqs)}


def rle_identical(a, b, seqs):
    """Decoded-content identity of two runs' per-frame masks (RLE writers are deterministic; this compares content)."""
    res = {}
    for s in seqs:
        pa, pb = os.path.join(a, "rle", f"{s}.jsonl.gz"), os.path.join(b, "rle", f"{s}.jsonl.gz")
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            res[s] = None
            continue
        same = True
        for ra, rb in zip(bc.read_rle_jsonl(pa), bc.read_rle_jsonl(pb)):
            if ra != rb:
                same = False
                break
        res[s] = same
    vals = [v for v in res.values() if v is not None]
    return {"per_seq": res, "all_identical": bool(vals) and all(vals) and len(vals) == len(seqs)}


def mode_factorial(ctx):
    fname = f"factorial__{ctx.tag}.json"
    prefix = ctx.factorial_prefix
    bench, scores = ctx.R.path("paper1.bench"), ctx.R.path("paper1.scores")
    seqs = list(bc.EDI_TEST)
    rows, dirs = {}, {}
    for arm in FACTORIAL_ARMS:
        for level, suf in FACTORIAL_LEVELS:
            var = prefix + suf
            rd = bc.bench_run_dir(ctx.R, "edinburgh", arm, "A", "test", var)
            dirs[(arm, level)] = rd
            cfg = _read_json(os.path.join(rd, "BENCH_CONFIG.json")) if os.path.isfile(os.path.join(rd, "BENCH_CONFIG.json")) else None
            done_n = sum(os.path.isfile(os.path.join(rd, f".done_{s}")) for s in seqs)
            sd = latest_score(scores, "edinburgh", "A", "test", var)
            sc = _read_json(os.path.join(sd, "score.json")) if sd else None
            res = ((sc or {}).get("results") or {}).get(arm) or {}
            comb = res.get("combined") or {}
            rows[f"{arm}|{level}"] = {
                "arm": arm, "level": level, "variant": var, "run_dir_complete": done_n == len(seqs),
                "stream": (cfg or {}).get("stream"), "core": ((cfg or {}).get("code") or {}).get("bench_core"),
                "score_dir": sd, "score_id": (sc or {}).get("score_id"),
                "HOTA": comb.get("HOTA"), "MOTA": comb.get("MOTA"), "IDF1": comb.get("IDF1"), "IDSW": comb.get("IDSW"),
                "per_clip": {s: {k: ((res.get("per_seq") or {}).get(s) or {}).get(k) for k in ("HOTA", "MOTA", "IDF1")}
                             for s in seqs},
                "vram": run_dir_curves(rd, seqs)}
            v = rows[f"{arm}|{level}"]["vram"]
            vals = [x for x in v.values() if x]
            rows[f"{arm}|{level}"]["vram_summary"] = {
                "peak_alloc_mib_max": max((x["peak_alloc_mib"] or 0) for x in vals) if vals else None,
                "end_alloc_mib_max": max((x["end_alloc_mib"] or 0) for x in vals) if vals else None,
                "slope_ols_mib_per_frame": {s: (x["slope"] or {}).get("ols_mib_per_frame") for s, x in v.items() if x}}
    flags = {}
    for arm in FACTORIAL_ARMS:
        a, b = dirs[(arm, "k16")], dirs[(arm, "none")]
        flags[f"{arm}: k16 == none"] = {"mot": files_identical(a, b, seqs, "mot", ".txt"), "rle": rle_identical(a, b, seqs)}
        flags[f"{arm}: k16+pop == k16 (pop)"] = {"mot": files_identical(dirs[(arm, "k16+pop")], a, seqs, "mot", ".txt"),
                                                 "rle": rle_identical(dirs[(arm, "k16+pop")], a, seqs)}
        flags[f"{arm}: k8 == k16"] = {"mot": files_identical(dirs[(arm, "k8")], a, seqs, "mot", ".txt")}
        j4 = bc.bench_run_dir(ctx.R, "edinburgh", arm, "A", "test")
        flags[f"{arm}: k16+pop ({prefix}) == J4 main run"] = {"mot": files_identical(dirs[(arm, "k16+pop")], j4, seqs, "mot", ".txt"),
                                                            "j4_core": ((_read_json(os.path.join(j4, "BENCH_CONFIG.json")).get("code") or {}).get("bench_core")
                                                                        if os.path.isfile(os.path.join(j4, "BENCH_CONFIG.json")) else None)}
    rec = {"notebook": f"{NB_NAME} {NB_VERSION}", "mode": "factorial", "tag": ctx.tag, "prefix": prefix,
           "created_utc": _now(), "levels": [lv for lv, _s in FACTORIAL_LEVELS], "arms": list(FACTORIAL_ARMS),
           "clips": seqs, "rows": rows, "identity_flags": flags,
           "definitions": {"none": "no pruning, no pop (processed_frames kept)", "k8": "keep 8 newest non-cond outputs, no pop",
                           "k16": "keep 16, no pop", "k16+pop": "keep 16 + pop (the protocol setting, §3)",
                           "vram": "allocated MiB sampled after every 25th frame (track_bench telemetry); peak = max of the "
                                   "running peak; end = the last sample of the clip; slope = OLS over samples fi >= 32"}}
    rec["published"] = publish(ctx, "_factorial", fname, rec)
    return rec


# ---- report: the paper tables -------------------------------------------------------------------------------------------
def _f(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def mode_report(ctx, arms):
    eff = ctx.R.path("paper1.efficiency", None, ctx.gpu_class)
    micro = {}
    for a in arms:
        p = os.path.join(eff, a, f"micro__{ctx.tag}.json")
        if os.path.isfile(p):
            micro[a] = _read_json(p)
    e2e = {}
    for a in set(J13_PIG_ARMS) | set(J13_CALF_ARMS):
        for tgt in ("pigtrack", "calf"):
            p = os.path.join(eff, a, f"e2e_{tgt}__{ctx.tag}.json")
            if os.path.isfile(p):
                e2e[f"{a}|{tgt}"] = _read_json(p)
    fac_p = os.path.join(eff, "_factorial", f"factorial__{ctx.tag}.json")
    dec_p = os.path.join(eff, "_decode", f"micro_decode__{ctx.tag}.json")
    fac = _read_json(fac_p) if os.path.isfile(fac_p) else None
    dec = _read_json(dec_p) if os.path.isfile(dec_p) else None
    tables = build_tables(micro, e2e, fac, dec)
    out_dir = os.path.join(ctx.local_root, "out", "_report", ctx.tag)
    os.makedirs(out_dir, exist_ok=True)
    for name, (header, rows) in tables["csv"].items():
        p = os.path.join(out_dir, f"{name}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    rep = {"notebook": f"{NB_NAME} {NB_VERSION}", "mode": "report", "tag": ctx.tag, "created_utc": _now(),
           "arms": sorted(micro), "e2e": sorted(e2e), "factorial": fac is not None, "decode": dec is not None,
           "tables": tables["json"], "statements": tables["statements"]}
    rp = write_json_local(os.path.join(out_dir, "report.json"), rep)
    if ctx.publish:
        dst_dir = ctx.R.out("paper1.efficiency", ctx.gpu_class, "_report", ctx.tag)
        for name in tables["csv"]:
            bc.publish_no_overwrite(os.path.join(out_dir, f"{name}.csv"), f"{dst_dir}/{name}.csv", ctx.dbutils)
        bc.publish_no_overwrite(rp, f"{dst_dir}/report.json", ctx.dbutils)
        rep["published"] = dst_dir
    return rep


def build_tables(micro, e2e, fac, dec):
    """Pure: the JSON records -> paper tables (CSV rows) + §11 sentences."""
    M = 2 ** 20
    out_json, csvs, sentences = {}, {}, []
    # tab:params -- params backbone / total, GFLOPs / GMACs per frame
    hdr = ["arm", "backbone_or_student_params", "total_params", "gflops_frame", "gmacs_frame", "gflops_backbone_or_student",
           "gflops_memory_attention", "gflops_mask_decoder"]
    rows = []
    for a, r in micro.items():
        p = r.get("params") or {}
        bb = p.get("student") if p.get("student") else p.get("vision_encoder.backbone")
        fl = r.get("flops") or {}
        gb = fl.get("gflops_by_group") or {}
        bbf = fl.get("gflops_student") if fl.get("gflops_student") else gb.get("vision_encoder.backbone")
        rows.append([a, bb, p.get("total_system"), fl.get("gflops_total"), fl.get("gmacs_total"), bbf,
                     gb.get("memory_attention"), gb.get("mask_decoder")])
    csvs["tab_params"] = (hdr, rows)
    # tab:efficiency -- per-module latency median [IQR], e2e fps, decode separate, weights, steady peak, slope, energy
    hdr = ["arm", "objects", "backbone_or_student_ms", "neck_ms", "memory_attention_ms", "memory_encoder_ms", "mask_decoder_ms",
           "prompt_encoder_ms", "spatial_perceiver_ms", "post_ms", "model_ms", "model_other_ms", "wall_ms_frame",
           "e2e_fps_median", "e2e_fps_iqr", "decode_wait_ms", "weights_mib", "context_mib", "steady_peak_alloc_mib",
           "steady_peak_reserved_mib", "session_mib_median", "slope_ols_mib_per_frame", "slope_median_step", "slope_frames",
           "fits_a10", "streams_analytic", "j_per_frame", "j_per_frame_above_idle"]
    rows = []

    def ms(g, k):
        d = (g or {}).get(k) or {}
        return None if not d.get("n") else f"{d['median']:.2f} [{d['p25']:.2f}, {d['p75']:.2f}]"
    fps = {}
    for a, r in micro.items():
        L, Mm = r.get("latency") or {}, r.get("memory") or {}
        g = L.get("gpu_ms") or {}
        bbk = "student" if a in STUDENT_ARMS else "vision_encoder.backbone"
        e = L.get("e2e_fps") or {}
        fps[a] = (e.get("median"), e.get("iqr"))
        en = L.get("energy") or {}
        pk = Mm.get("peak_after_warmup") or {}
        gr = Mm.get("growth_alloc") or {}
        rows.append([a, L.get("objects"), ms(g, bbk), ms(g, "vision_encoder.neck"), ms(g, "memory_attention"),
                     ms(g, "memory_encoder"), ms(g, "mask_decoder"), ms(g, "prompt_encoder"), ms(g, "spatial_perceiver"),
                     ms(g, "post"), ms(g, "model"), ms(g, "model_other"),
                     None if not (L.get("wall_ms_per_frame") or {}).get("n") else L["wall_ms_per_frame"]["median"],
                     e.get("median"), e.get("iqr"), (L.get("decode_wait_ms") or {}).get("median"),
                     ((r.get("load") or {}).get("after_load") or {}).get("alloc"), (r.get("load") or {}).get("context_mib"),
                     pk.get("peak_alloc"), pk.get("peak_reserved"), (Mm.get("session_mib") or {}).get("median"),
                     gr.get("ols_mib_per_frame"), gr.get("median_step_mib_per_frame"), Mm.get("frames"),
                     (r.get("fits_a10") or {}).get("fits"), (r.get("streams_analytic") or {}).get("streams"),
                     en.get("j_per_frame"), en.get("j_per_frame_above_idle")])
    csvs["tab_efficiency"] = (hdr, rows)
    for s in STUDENT_ARMS:
        if s in fps and "T" in fps:
            st = ratio_statement(fps[s][0], fps["T"][0], fps[s][1], fps["T"][1],
                                 f"A10, 4 calves, {s} vs T, decode in the loop")
            if st:
                sentences.append(dict(st, pair=f"{s}/T", setting="micro 4 objects"))
    # e2e
    hdr = ["arm", "target", "frames", "objects_end", "detector_ms", "detector_wall_ms", "backbone_or_student_ms",
           "memory_attention_ms", "mask_decoder_ms", "post_ms", "decode_wait_ms", "wall_ms_frame", "e2e_fps_median",
           "detector_weights_mib", "weights_mib", "steady_peak_alloc_mib", "steady_peak_reserved_mib", "context_mib",
           "slope_ols", "fits_a10", "j_per_frame"]
    rows, efps = [], {}
    for k, r in sorted(e2e.items()):
        a, tgt = k.split("|")
        L, Mm = r.get("latency") or {}, r.get("memory") or {}
        g, c = L.get("gpu_ms") or {}, L.get("cpu_ms") or {}
        bbk = "student" if a in STUDENT_ARMS else "vision_encoder.backbone"
        e = L.get("e2e_fps") or {}
        efps[(a, tgt)] = (e.get("median"), e.get("iqr"))
        pk = Mm.get("peak_after_warmup") or {}
        rows.append([a, tgt, L.get("frames"), L.get("objects"), ms(g, "detector"), ms(c, "detector_wall"), ms(g, bbk),
                     ms(g, "memory_attention"), ms(g, "mask_decoder"), ms(g, "post"), (L.get("decode_wait_ms") or {}).get("median"),
                     (L.get("wall_ms_per_frame") or {}).get("median"), e.get("median"), (r.get("detector") or {}).get("weights_alloc_mib"),
                     ((r.get("load") or {}).get("after_load") or {}).get("alloc"), pk.get("peak_alloc"), pk.get("peak_reserved"),
                     (r.get("load") or {}).get("context_mib"), (Mm.get("growth_alloc") or {}).get("ols_mib_per_frame"),
                     (r.get("fits_a10") or {}).get("fits"), (L.get("energy") or {}).get("j_per_frame")])
    csvs["tab_efficiency_protD"] = (hdr, rows)
    if ("S-PT", "pigtrack") in efps and ("T", "pigtrack") in efps:
        st = ratio_statement(efps[("S-PT", "pigtrack")][0], efps[("T", "pigtrack")][0], efps[("S-PT", "pigtrack")][1],
                             efps[("T", "pigtrack")][1], f"A10, d_pt_v1 + tracker live, {J13_PIG_SEQ}")
        if st:
            sentences.append(dict(st, pair="S-PT/T", setting="J13 PigTrack end to end"))
    # tab:pruning
    if fac:
        hdr = ["arm", "level", "HOTA", "MOTA", "IDF1", "IDSW", "peak_alloc_mib", "end_alloc_mib", "slope_ols_by_clip", "core"]
        rows = []
        for _k, r in fac["rows"].items():
            v = r.get("vram_summary") or {}
            rows.append([r["arm"], r["level"], r.get("HOTA"), r.get("MOTA"), r.get("IDF1"), r.get("IDSW"),
                         v.get("peak_alloc_mib_max"), v.get("end_alloc_mib_max"),
                         json.dumps({s: (None if x is None else round(x, 3)) for s, x in (v.get("slope_ols_mib_per_frame") or {}).items()}),
                         (r.get("core") or {}).get("source", "")[:12] if isinstance(r.get("core"), dict) else None])
        csvs["tab_pruning"] = (hdr, rows)
        out_json["identity_flags"] = fac.get("identity_flags")
    # tab:budget (analytic)
    hdr = ["arm", "weights_mib", "context_mib", "per_stream_steady_mib", "streams_per_a10_analytic", "budget_mib"]
    rows = []
    for a, r in micro.items():
        sa = r.get("streams_analytic") or {}
        rows.append([a, ((r.get("load") or {}).get("after_load") or {}).get("alloc"), (r.get("load") or {}).get("context_mib"),
                     sa.get("per_stream_mib"), sa.get("streams"), sa.get("budget_mib")])
    csvs["tab_budget"] = (hdr, rows)
    if dec:
        out_json["decode"] = dec.get("decode")
    out_json["fits_a10"] = {a: r.get("fits_a10") for a, r in micro.items()}
    out_json["fits_a10_e2e"] = {k: r.get("fits_a10") for k, r in e2e.items()}
    return {"csv": csvs, "json": out_json, "statements": sentences}


# COMMAND ----------

# MAGIC %md
# MAGIC ## Production hooks, parameters, main

# COMMAND ----------


def _prod_arm_loader(ctx, name):
    return bc.load_arm(name, device="cuda", hf_token=ctx.hf_token, revisions=ctx.revisions, registry=ctx.R,
                       dbutils=ctx.dbutils, log=ctx.log)


def _prod_stage_clip(ctx, w):
    src = ctx.R.path("datasets.calves.videos", None, w["clip"])
    return bc.stage_file(src, os.path.join(ctx.local_root, "calves", w["clip"]), ctx.dbutils, w.get("clip_sha256"))


def _prod_calf_frames(mp4, w, n=None):
    return bc.CalfWindowFrames(mp4, int(w["start_fidx"]), int(w["n_frames"]), name=w["name"])


def _prod_stage_pig_seq(ctx, seq):
    src = ctx.R.path("datasets.pigtrack.sequences", None, "test", seq)
    d = os.path.join(ctx.local_root, "pigtrack", "test", seq)
    bc.stage_tree(src, d, ctx.dbutils)
    return d


def _prod_gpu(require):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    cls = bc.gpu_info()["class"]
    if require != "any" and cls != require:
        raise RuntimeError(f"GPU class {cls} but require_gpu={require} (efficiency numbers come from the A10, §8)")
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError("PYTORCH_CUDA_ALLOC_CONF lacks expandable_segments:True (job cluster spec)")
    return cls


PROD_HOOKS = {"arm_loader": _prod_arm_loader, "stage_clip": _prod_stage_clip, "calf_frames": _prod_calf_frames,
              "stage_pig_seq": _prod_stage_pig_seq, "power_sampler": PowerSampler,
              "detector_loader": lambda ckpt, res: bc.load_rfdetr_detector(ckpt, resolution=res), "gpu": _prod_gpu,
              "hf_token": lambda dbu: bc.hf_token_from_dbutils(dbu)}


def parse_params(raw):
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}")
    g = {k: str(raw.get(k, d) if raw.get(k) is not None else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace(**g)
    if P.mode not in MODES:
        raise ValueError(f"mode={P.mode!r}: one of {MODES}")
    if not TAG_RX.match(P.tag or "") or "__" in P.tag:
        raise ValueError(f"tag={P.tag!r}: one path segment, no '__'")
    if P.mode in ("micro", "e2e") and P.arm not in S_ARMS:
        raise ValueError(f"arm={P.arm!r}: one of {S_ARMS}")
    if P.mode == "e2e" and P.e2e_target not in ("pigtrack", "calf"):
        raise ValueError("e2e needs e2e_target = pigtrack | calf")
    P.mem_frames = int(P.mem_frames)
    if P.mem_frames < MEM_FRAMES_MIN:
        raise ValueError(f"mem_frames {P.mem_frames} < {MEM_FRAMES_MIN}")
    P.e2e_calf_frames = int(P.e2e_calf_frames)
    if P.e2e_calf_frames < MEM_FRAMES_MIN:
        raise ValueError(f"e2e_calf_frames {P.e2e_calf_frames} < {MEM_FRAMES_MIN}")
    if not TAG_RX.match(P.factorial_prefix) or "__" in P.factorial_prefix:
        raise ValueError("factorial_prefix: one path segment, no '__'")
    P.publish = P.publish.lower() in ("1", "true", "yes")
    P.progress_every = int(P.progress_every or 0)
    P.report_arms = [a for a in P.report_arms.split(",") if a.strip()]
    return P


def main(raw, *, dbutils_=None, R=None, hooks=None):
    P = parse_params(raw)
    h = dict(PROD_HOOKS)
    h.update(hooks or {})
    R = R or farm_paths.load(version=1)
    gpu_class = "A10"
    use_cuda = False
    if P.mode in ("micro", "micro_decode", "e2e") and h.get("gpu"):
        gpu_class = h["gpu"](P.require_gpu)
        use_cuda = True
    elif P.mode in ("factorial", "report"):
        gpu_class = "A10" if P.require_gpu in ("A10", "any") else P.require_gpu
    local_root = h.get("local_root") or ("/local_disk0/tol65" if IS_NOTEBOOK else tempfile.mkdtemp(prefix="tol65_"))
    os.makedirs(local_root, exist_ok=True)
    ctx = Ctx(R=R, dbutils=dbutils_, hooks=h, tag=P.tag, publish=P.publish, local_root=local_root, gpu_class=gpu_class,
              use_cuda=use_cuda, mem_frames=P.mem_frames, e2e_calf_frames=P.e2e_calf_frames,
              factorial_prefix=P.factorial_prefix, progress_every=P.progress_every, log=h.get("log", print),
              idle_s=h.get("idle_s", IDLE_S), hf_token=None, revisions={})
    if P.mode in ("micro", "e2e"):
        ctx.hf_token = h["hf_token"](dbutils_) if h.get("hf_token") and dbutils_ is not None else None
        ctx.revisions = resolve_revisions(R, P.revisions)
    if P.mode == "micro":
        r = mode_micro(ctx, P.arm)
        return {"mode": "micro", "arm": P.arm, "published": (r.get("published") or {}).get("json"),
                "fps": (r.get("latency") or {}).get("e2e_fps"), "fits_a10": r.get("fits_a10"),
                "gflops": (r.get("flops") or {}).get("gflops_total")}
    if P.mode == "micro_decode":
        r = mode_micro_decode(ctx)
        return {"mode": "micro_decode", "decode": r.get("decode"), "published": r.get("published")}
    if P.mode == "e2e":
        r = mode_e2e(ctx, P.arm, P.e2e_target)
        return {"mode": "e2e", "arm": P.arm, "target": P.e2e_target, "fps": (r.get("latency") or {}).get("e2e_fps"),
                "fits_a10": r.get("fits_a10"), "published": (r.get("published") or {}).get("json")}
    if P.mode == "factorial":
        r = mode_factorial(ctx)
        return {"mode": "factorial", "flags": r.get("identity_flags"), "published": r.get("published")}
    r = mode_report(ctx, P.report_arms)
    return {"mode": "report", "statements": r.get("statements"), "published": r.get("published")}


# COMMAND ----------





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
            out[name] = os.environ.get("TOL65_" + name.upper(), default)
    return out


RAW = read_widgets()
print("widgets:", json.dumps(RAW, sort_keys=True))
SUMMARY = main(RAW, dbutils_=dbutils)
print(json.dumps(SUMMARY, indent=1, default=str)[:20000])

# COMMAND ----------

if IS_NOTEBOOK:
    dbutils.notebook.exit(json.dumps(SUMMARY, default=str)[:60000])
