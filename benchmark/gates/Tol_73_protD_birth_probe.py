# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N73 — Protocol D birth-frame memory-encoding probe (evidence only; BENCH_PROTOCOL §15 b1.1 births entry)
# MAGIC
# MAGIC transformers 5.2.0 `Sam2VideoModel` / `Sam3TrackerVideoModel` encode the memories of all objects of a frame in ONE
# MAGIC batch with `is_mask_from_pts = any(per-object flags)` (`_batch_encode_memories`). On a frame where a new object is
# MAGIC prompted (a Protocol D birth) every EXISTING object's memory is therefore encoded from its binarised mask instead of
# MAGIC its sigmoid. User decision (b1.1, 2026-10-15): keep the library code for every arm, disclose it, and measure its size
# MAGIC again in the Protocol D runs (J12), where births are frequent. This notebook makes that measurement. It produces
# MAGIC evidence, never a bench number, and never writes into a bench run directory.
# MAGIC
# MAGIC Per arm (widget `arms`), on the finished J12 run `bench/<dataset>/<arm>__protD/<split>/`:
# MAGIC 1. **Exposure** (no GPU), from every sequence's telemetry: seeds, births, birth frames, deaths, the objects alive at
# MAGIC    each birth frame, and the share of session object-frames that follow at least one birth-frame encoding.
# MAGIC 2. **Selection** (a rule fixed before any probe output): the `max_seqs` sequences (default 4) of at most `max_frames`
# MAGIC    frames (default 600) with the most birth frames in the FIRST arm's run (T); ties -> fewer frames, then the sequence
# MAGIC    name. The same sequences are replayed for every arm, so the arms are compared on the same sequences.
# MAGIC 3. **Two replays per selected sequence** of the J12 run's OWN prompts: seeds and births at the same frames with the
# MAGIC    exact cached detection boxes (matched to the telemetry events), deaths from each replay's own presence:
# MAGIC    - **U** = the library code unchanged. Its MOT file must be byte-identical to the J12 file (replay fidelity).
# MAGIC    - **P** = per-object memory flags (`per_object_memory_flags`, copied verbatim from Tol_60's G1-5 probe): each
# MAGIC      object's memory is encoded with its own flag.
# MAGIC    - **U vs P**: raw-output digests per frame; per object-frame mask IoU of the emitted masks (one-sided = 0, the G1-5
# MAGIC      aggregation), split into pre-exposure (must be identical), existing objects after a birth, and newborn objects
# MAGIC      before their own first exposure; MOT rows identical / changed / one-sided.
# MAGIC
# MAGIC An arm whose model has no batched memory encoder (EdgeTAM) is recorded as not affected and is not replayed.
# MAGIC Outputs: `paper1_distillation/evidence/protD_birth_probe/<tag>/` (`publish_no_overwrite`; resumable per sequence).
# MAGIC
# MAGIC **Local check** (no torch / GPU / Volume): `TOL_SELFTEST=1 TOL63_CODE_DIR=<bench dir> TOL_PATHS_DIR=<dir of tol_paths.py>
# MAGIC TOL63_NB=<Tol_63_track_bench.py> python Tol_73_protD_birth_probe.py` (stub tracker with a batched-encoder stand-in;
# MAGIC the J12 run fixture is produced by Tol_63's own `main`).

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
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

NB_NAME = "Tol_73_protD_birth_probe"
NB_VERSION = "tol73-2026-10-19.2"
SELFTEST = os.environ.get("TOL_SELFTEST") == "1"
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None


def _this_dir():
    """= Tol_63._this_dir: env TOL63_CODE_DIR (the pinned core folder of the job cluster), else this notebook's own
    workspace directory, else this file's directory."""
    env = os.environ.get("TOL63_CODE_DIR")
    if env:
        return env
    if not IS_NOTEBOOK:
        return os.path.dirname(os.path.abspath(__file__))
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    return "/Workspace" + os.path.dirname(nbp)


def _import_exact(name, dirs):
    """= Tol_63._import_exact: import `name` from the FIRST of `dirs` holding `<name>.py`, asserting that file runs."""
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
os.environ.setdefault("TOL_CORE_DIR", CODE_DIR)
bc = _import_exact("tol_bench_core", [CODE_DIR])
tol_paths = _import_exact("tol_paths", [CODE_DIR, os.environ.get("TOL_PATHS_DIR")])


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


print(f"{NB_NAME} {NB_VERSION}: tol_bench_core {bc.BENCH_VERSION} ({bc.__file__}, sha256 "
      f"{_sha256(bc.__file__)[:16]}), tol_paths {tol_paths.__file__} (sha256 {_sha256(tol_paths.__file__)[:16]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters, the G1-5 mechanism patch, the replay, the capture and the comparison (pure; exercised by the selftest)

# COMMAND ----------

WIDGETS = collections.OrderedDict([
    ("arms", "T,S-PT"), ("dataset", "pigtrack"), ("split", "test"), ("max_seqs", "4"), ("max_frames", "600"),
    ("tag", "j12_v1"), ("revisions", "protocol:hf_revisions_v1.json"), ("require_gpu", "A10"), ("publish", "true"),
    ("progress_every", "200"),
])
CORE_DATASET = {"pigtrack": "pigtrack", "edinburgh": "edinburgh"}
TAG_RX = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
BOX_TOL = 0.005 + 1e-6          # telemetry events round boxes to 2 decimals
SCORE_TOL = 5e-5 + 1e-9         # ... and scores to 4
TOL60_PROBE_SOURCE = "Tol_60_bench_g1.per_object_memory_flags (J0b G1-5 mechanism probe), copied verbatim"


def parse_params(raw):
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}; known: {list(WIDGETS)}")
    g = {k: str(raw.get(k) if raw.get(k) is not None else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace()
    P.arms = [a.strip() for a in g["arms"].split(",") if a.strip()]
    if not P.arms or len(set(P.arms)) != len(P.arms):
        raise ValueError(f"arms={g['arms']!r}: a non-empty comma list without repeats")
    bad = [a for a in P.arms if a not in bc.ARMS]
    if bad:
        raise ValueError(f"arms {bad} unknown; known: {sorted(bc.ARMS)}")
    if g["dataset"] not in CORE_DATASET:
        raise ValueError(f"dataset={g['dataset']!r}: expected {sorted(CORE_DATASET)}")
    P.dataset, P.core_dataset = g["dataset"], CORE_DATASET[g["dataset"]]
    P.split = g["split"]
    if P.split != "test":
        raise ValueError("split: the probe reads the J12 Protocol D TEST runs (val is the tau grid, one run per tau)")
    P.max_seqs, P.max_frames = int(g["max_seqs"]), int(g["max_frames"])
    if P.max_seqs < 1 or P.max_frames < 1:
        raise ValueError("max_seqs / max_frames must be >= 1")
    if not TAG_RX.match(g["tag"]):
        raise ValueError(f"tag={g['tag']!r}: one path segment")
    P.tag = g["tag"]
    P.revisions_spec = g["revisions"]
    P.require_gpu = g["require_gpu"] or "A10"
    if P.require_gpu not in ("A10", "A100", "any"):
        raise ValueError(f"require_gpu={P.require_gpu!r}")
    P.publish = g["publish"].lower() in ("1", "true", "yes")
    P.progress_every = int(g["progress_every"] or 0)
    return P


@contextlib.contextmanager
def per_object_memory_flags(model):
    """G1-5 mechanism probe (NOT a protocol setting). transformers 5.2.0 Sam2Video / Sam3TrackerVideo encode the memory of
    all objects of a frame in ONE batch with is_mask_from_pts = any(per-object flags): on a frame where one object is
    prompted (a birth), every other object's memory is encoded from its BINARISED mask instead of the sigmoid. This
    patch splits the batch by flag so each object keeps its own flag (EdgeTamVideo already encodes per object).
    Yields True when the model has the batched encoder."""
    orig = getattr(type(model), "_batch_encode_memories", None)
    if orig is None:
        yield False
        return
    bound = orig.__get__(model)

    def patched(**kw):
        groups = collections.OrderedDict()
        for i, f in enumerate(kw["is_mask_from_pts_per_obj"]):
            groups.setdefault(bool(f), []).append(i)
        for f, idx in groups.items():
            bound(inference_session=kw["inference_session"], frame_idx=kw["frame_idx"],
                  objects_needing_memory_encoding=[kw["objects_needing_memory_encoding"][i] for i in idx],
                  high_res_masks_for_memory=[kw["high_res_masks_for_memory"][i] for i in idx],
                  object_score_logits_for_memory=[kw["object_score_logits_for_memory"][i] for i in idx],
                  is_mask_from_pts_per_obj=[f] * len(idx))
    model.__dict__["_batch_encode_memories"] = patched
    try:
        yield True
    finally:
        model.__dict__.pop("_batch_encode_memories", None)


def has_batched_encoder(model):
    return getattr(type(model), "_batch_encode_memories", None) is not None


class ReplayProtocolD(bc.ProtocolD):
    """Protocol D whose seed / birth DECISIONS are replayed from a finished run: the planned prompts at the planned pass
    frames, whatever this run's outputs are. Presence, deaths and the emit filter are ProtocolD's own (on_output is not
    overridden), so with identical outputs the replay reproduces the original run row for row."""

    def __init__(self, fps, tau, plan):
        super().__init__(fps, tau)
        self.plan = collections.OrderedDict()
        for p in plan:
            self.plan.setdefault(int(p.frame), []).append(p)
        off = [fi for fi in self.plan if not self.is_pass(fi)]
        if off:
            raise ValueError(f"planned prompts on non-pass frames {off[:5]} (period {self.period})")
        self.replayed = 0

    def on_pass(self, fi, dets):
        self.n_passes += 1
        self.seeded = True
        out = []
        for p in self.plan.get(int(fi), []):
            if int(p.obj_id) != self.next_id:
                raise bc.BenchAssertError(f"replay: id {p.obj_id} out of creation order (next {self.next_id})")
            self.next_id += 1
            self.tracks[int(p.obj_id)] = {"cur": None, "last": tuple(p.box), "absent": 0, "born": int(fi)}
            kind = str(p.source).replace("protD_", "")
            self.events.append({"type": kind, "fi": int(fi), "id": int(p.obj_id), "score": round(float(p.score), 4),
                                "box": [round(v, 2) for v in p.box], "replayed": True})
            out.append(bc.Prompt(frame=int(fi), obj_id=int(p.obj_id), box=tuple(p.box), score=float(p.score),
                                 source=p.source))
            self.replayed += 1
        return out


def plan_from_run(tel, cache):
    """The seeds and births of a finished Protocol D run (telemetry events; boxes rounded to 2 decimals, scores to 4) ->
    Prompts with the EXACT cached detection boxes / scores. Each event must match exactly one distinct cached detection
    of its pass frame; ids must be 0..n-1 in creation order and agree with stats.births."""
    plan = []
    for e in (tel.get("protocol_D") or {}).get("events") or []:
        if e.get("type") not in ("seed", "birth"):
            continue
        fi = int(e["fi"])
        if fi not in cache.dets:
            raise bc.BenchAssertError(f"{tel.get('seq')}: event at frame {fi} but the cache has no such pass frame")
        hits = sorted({(tuple(b), float(s)) for b, s in cache.dets[fi]
                       if abs(float(s) - float(e["score"])) <= SCORE_TOL
                       and all(abs(float(b[i]) - float(e["box"][i])) <= BOX_TOL for i in range(4))})
        if len(hits) != 1:
            raise bc.BenchAssertError(f"{tel.get('seq')}: {e['type']} id {e['id']} at {fi} matches {len(hits)} cached "
                                      f"detections (need exactly 1)")
        b, s = hits[0]
        plan.append(bc.Prompt(frame=fi, obj_id=int(e["id"]), box=tuple(float(v) for v in b), score=s,
                              source=f"protD_{e['type']}"))
    ids = [p.obj_id for p in plan]
    if ids != list(range(len(ids))):
        raise bc.BenchAssertError(f"{tel.get('seq')}: seed / birth ids {ids[:12]} are not 0..n-1 in creation order")
    grouped = collections.OrderedDict()
    for p in plan:
        grouped.setdefault(p.frame, []).append(p.obj_id)
    want = [[int(f), [int(i) for i in ids_]] for f, ids_ in (tel.get("stats") or {}).get("births") or []]
    if [[f, v] for f, v in grouped.items()] != want:
        raise bc.BenchAssertError(f"{tel.get('seq')}: plan {list(grouped.items())[:4]} != stats.births {want[:4]}")
    return plan


def exposure(tel):
    """Births and their reach for one finished Protocol D sequence (telemetry only)."""
    st = tel.get("stats") or {}
    ev = (tel.get("protocol_D") or {}).get("events") or []
    n = int((tel.get("source") or {}).get("n_frames") or st.get("frames") or 0)
    created = {int(e["id"]): int(e["fi"]) for e in ev if e.get("type") in ("seed", "birth")}
    died = {int(e["id"]): int(e["fi"]) for e in ev if e.get("type") == "death"}
    bframes = sorted({int(e["fi"]) for e in ev if e.get("type") == "birth"})
    alive_at = [sum(1 for o, c in created.items() if c < fb and died.get(o, n) >= fb) for fb in bframes]
    tot = expo = 0
    for o, c in created.items():
        end = min(n, died.get(o, n) + 1)
        tot += max(0, end - c)
        e = next((fb for fb in bframes if fb > c), None)
        if e is not None and e < end:
            expo += end - e
    c = collections.Counter(e.get("type") for e in ev)
    return {"seq": tel.get("seq"), "frames": n, "forwards": st.get("forwards"), "night": (tel.get("source") or {}).get("night"),
            "fps": (tel.get("source") or {}).get("fps"), "seeds": c["seed"], "births": c["birth"],
            "birth_frames": len(bframes), "deaths": c["death"], "birth_blocked": c["birth_blocked"],
            "first_birth_frame": bframes[0] if bframes else None, "objects_alive_at_birth_frames": alive_at,
            "object_frames": tot, "object_frames_after_a_birth_encoding": expo}


def summarize_exposure(rows):
    n = len(rows)
    tot = sum(r["object_frames"] for r in rows)
    ex = sum(r["object_frames_after_a_birth_encoding"] for r in rows)
    alive = [a for r in rows for a in r["objects_alive_at_birth_frames"]]
    fr = sum(r["frames"] for r in rows)
    return {"sequences": n, "frames": fr, "seeds": sum(r["seeds"] for r in rows), "births": sum(r["births"] for r in rows),
            "birth_frames": sum(r["birth_frames"] for r in rows), "deaths": sum(r["deaths"] for r in rows),
            "birth_blocked": sum(r["birth_blocked"] for r in rows),
            "sequences_with_a_birth": sum(1 for r in rows if r["births"]),
            "births_per_sequence": {"min": min((r["births"] for r in rows), default=None),
                                    "median": float(np.median([r["births"] for r in rows])) if rows else None,
                                    "max": max((r["births"] for r in rows), default=None)},
            "birth_frames_per_1000_frames": round(1000.0 * sum(r["birth_frames"] for r in rows) / max(fr, 1), 3),
            "objects_alive_at_a_birth_frame": {"n": len(alive), "mean": round(float(np.mean(alive)), 3) if alive else None,
                                               "min": min(alive, default=None), "max": max(alive, default=None)},
            "object_frames": tot, "object_frames_after_a_birth_encoding": ex,
            "share_after_a_birth_encoding": round(ex / tot, 6) if tot else None}


def select_sequences(rows, max_seqs, max_frames):
    """Fixed rule (header): most birth frames among sequences of <= max_frames frames with >= 1 birth; ties -> fewer
    frames, then the name."""
    ok = [r for r in rows if r["birth_frames"] > 0 and r["frames"] <= max_frames]
    ok.sort(key=lambda r: (-r["birth_frames"], r["frames"], r["seq"]))
    return [r["seq"] for r in ok[:max_seqs]]


def _arr_bytes(x):
    """= Tol_60._arr_bytes: exact bytes of an output tensor for a digest (bf16 -> fp32 is lossless)."""
    if bc._is_t(x):
        return x.detach().float().cpu().numpy().tobytes()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32)).tobytes()


class Capture:
    """on_frame recorder (observe only): per forwarded frame the sha256 of the raw outputs (pred_masks logits + object
    scores; = Tol_60.Capture's digest) and the RLE of every EMITTED mask."""

    def __init__(self):
        self.digest, self.masks = {}, {}

    def __call__(self, fi, sess, out, mask_logits, binm, rows, live, pruned):
        h = hashlib.sha256()
        h.update(_arr_bytes(out.pred_masks))
        h.update(_arr_bytes(out.object_score_logits))
        self.digest[int(fi)] = h.hexdigest()[:24]
        self.masks[int(fi)] = {int(r.obj_id): bc.rle_encode(np.asarray(bc.to_numpy(binm[r.k]), dtype=bool))
                               for r in rows}


def rle_iou(a, b):
    if a is None and b is None:
        return None
    if a is None or b is None:
        return 0.0
    if a["size"] == b["size"] and a["counts"] == b["counts"]:
        return 1.0
    ma, mb = bc.rle_decode(a), bc.rle_decode(b)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return inter / union if union else 1.0


def _stats(v):
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": round(float(a.mean()), 6), "median": round(float(np.median(a)), 6),
            "p05": round(float(np.quantile(a, 0.05)), 6), "p01": round(float(np.quantile(a, 0.01)), 6),
            "min": round(float(a.min()), 6), "identical": int((a == 1.0).sum()), "ge_0.99": int((a >= 0.99).sum()),
            "lt_0.95": int((a < 0.95).sum()), "lt_0.9": int((a < 0.9).sum()), "lt_0.5": int((a < 0.5).sum())}


def compare(U, P, plan):
    """U vs P captures of the same prompt plan. Object o is created at c_o; its first exposure e_o is the first birth
    frame strictly after c_o. Groups: pre_exposure (frames < e_o of seed objects: must be identical), existing
    (frames >= e_o: the binarised-memory effect), newborn (c_o <= frames < e_o of born objects: batch composition only)."""
    if set(U.digest) != set(P.digest):
        raise bc.BenchAssertError(f"U and P forwarded different frames: {sorted(set(U.digest) ^ set(P.digest))[:5]}")
    created = {int(p.obj_id): int(p.frame) for p in plan}
    seeds = {o for o, c in created.items() if any(p.obj_id == o and p.source == "protD_seed" for p in plan)}
    bframes = sorted({int(p.frame) for p in plan if p.source == "protD_birth"})
    expo = {o: next((fb for fb in bframes if fb > c), None) for o, c in created.items()}
    groups = {"pre_exposure": [], "existing": [], "newborn": []}
    one_sided = collections.Counter()
    per_obj = collections.defaultdict(list)
    frames = sorted(U.digest)
    same = [f for f in frames if U.digest[f] == P.digest[f]]
    first_diff = next((f for f in frames if U.digest[f] != P.digest[f]), None)
    for fi in frames:
        mu, mp = U.masks.get(fi, {}), P.masks.get(fi, {})
        for o in sorted(set(mu) | set(mp)):
            if o not in created:
                raise bc.BenchAssertError(f"frame {fi}: emitted id {o} is not in the plan")
            v = rle_iou(mu.get(o), mp.get(o))
            e = expo.get(o)
            g = "existing" if (e is not None and fi >= e) else ("pre_exposure" if o in seeds else "newborn")
            groups[g].append(v)
            if (o in mu) != (o in mp):
                one_sided[g] += 1
            if g == "existing":
                per_obj[o].append(v)
    pobj = {str(k): round(float(np.mean(v)), 6) for k, v in sorted(per_obj.items())}
    return {"frames": len(frames), "first_birth_frame": bframes[0] if bframes else None, "birth_frames": bframes,
            "digest_identical_frames": len(same), "first_digest_diff": first_diff,
            "digests_identical_before_first_birth": all(U.digest[f] == P.digest[f] for f in frames
                                                        if bframes and f < bframes[0]),
            "groups": {g: dict(_stats(v), one_sided=int(one_sided[g])) for g, v in groups.items()},
            "existing_per_object_mean_iou": pobj,
            "existing_per_object_mean_iou_min": min(pobj.values()) if pobj else None,
            "existing_objects": len(pobj)}


def _mot_rows(lines):
    out = {}
    for ln in lines:
        v = ln.split(",")
        out[(int(v[0]), int(v[1]))] = tuple(int(x) for x in v[2:6])
    return out


def compare_mot(lines_u, lines_p):
    a, b = _mot_rows(lines_u), _mot_rows(lines_p)
    both = set(a) & set(b)
    changed = [k for k in both if a[k] != b[k]]
    ious = []
    for k in changed:
        x, y, w, h = a[k]
        x2, y2, w2, h2 = b[k]
        ious.append(bc.box_iou((x, y, x + w, y + h), (x2, y2, x2 + w2, y2 + h2)))
    return {"rows_U": len(a), "rows_P": len(b), "rows_identical": len(both) - len(changed), "rows_changed": len(changed),
            "rows_only_U": len(set(a) - set(b)), "rows_only_P": len(set(b) - set(a)),
            "changed_box_iou": _stats(ious) if ious else {"n": 0}}


def summarize_arm(per_seq):
    """Pool the per-sequence comparisons of one arm (counts summed; means weighted by object-frames)."""
    out = {"sequences": sorted(per_seq), "replay_fidelity": {s: r["replay_fidelity"]["mot_identical_to_j12"]
                                                             for s, r in sorted(per_seq.items())}}
    for g in ("pre_exposure", "existing", "newborn"):
        n = sum(r["compare"]["groups"][g]["n"] for r in per_seq.values())
        if not n:
            out[g] = {"n": 0}
            continue
        mean = sum(r["compare"]["groups"][g]["mean"] * r["compare"]["groups"][g]["n"]
                   for r in per_seq.values() if r["compare"]["groups"][g]["n"]) / n
        d = {"n": n, "mean": round(mean, 6),
             "min": min(r["compare"]["groups"][g]["min"] for r in per_seq.values() if r["compare"]["groups"][g]["n"])}
        for k in ("identical", "ge_0.99", "lt_0.95", "lt_0.9", "lt_0.5", "one_sided"):
            d[k] = sum(r["compare"]["groups"][g].get(k, 0) for r in per_seq.values())
        out[g] = d
    pobj = [v for r in per_seq.values() for v in r["compare"]["existing_per_object_mean_iou"].values()]
    out["existing_per_object_mean_iou"] = {"n": len(pobj), "min": min(pobj, default=None),
                                           "median": float(np.median(pobj)) if pobj else None,
                                           "lt_0.99": sum(1 for v in pobj if v < 0.99)}
    out["pre_birth_digests_identical"] = all(r["compare"]["digests_identical_before_first_birth"] for r in per_seq.values())
    out["mot"] = {k: sum(r["mot"][k] for r in per_seq.values())
                  for k in ("rows_U", "rows_P", "rows_identical", "rows_changed", "rows_only_U", "rows_only_P")}
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Driver (production hooks; the selftest replaces them)

# COMMAND ----------


def cuda_probe(retries=3, sleep_s=60):
    """= Tol_63.cuda_probe."""
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
    """= Tol_63.gpu_preflight (the J12 run dirs are A10; masks are bit-identical only within a GPU class)."""
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError("PYTORCH_CUDA_ALLOC_CONF lacks expandable_segments:True (job cluster spec)")
    probe = cuda_probe()
    import torch
    name = torch.cuda.get_device_name(0)
    up = name.upper()
    cls = "A100" if "A100" in up else ("A10" if re.search(r"A10(?!\d)", up) else name)
    if P.require_gpu != "any" and cls != P.require_gpu:
        raise RuntimeError(f"GPU {name} (class {cls}) but require_gpu={P.require_gpu}")
    probe["class"] = cls
    return probe


def _notebook_source():
    if not IS_NOTEBOOK:
        with open(os.path.abspath(__file__), "rb") as f:
            return f.read()
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    for cand in ("/Workspace" + nbp, "/Workspace" + nbp + ".py"):
        if os.path.isfile(cand):
            with open(cand, "rb") as f:
                b = f.read()
            if b.lstrip(b"\xef\xbb\xbf").startswith(b"# Databricks notebook source"):
                return b
    import base64
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ExportFormat
    return base64.b64decode(WorkspaceClient().workspace.export(nbp, format=ExportFormat.SOURCE).content)


def _arm_loader(name, R, token, revisions, dbu, log):
    return bc.load_arm(name, device="cuda", hf_token=token, revisions=revisions, registry=R, dbutils=dbu, log=log)


PROD_HOOKS = {"local_root": "/local_disk0/tol73", "arm_loader": _arm_loader, "gpu_preflight": gpu_preflight,
              "hf_token": lambda dbu: bc.hf_token_from_dbutils(dbu), "notebook_source": None,
              "mem_settle": bc.settle_memory, "log": _log}


def _jdump(obj):
    return json.dumps(obj, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"


def _write_local(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp_{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def _jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def resolve_revisions(R, spec):
    """= Tol_63.resolve_revisions (protocol:<file>.json | JSON | empty)."""
    if not spec:
        return {}
    if spec.startswith("protocol:"):
        obj = _jload(bc.protocol_file_path(R, spec[len("protocol:"):]))
        obj = obj.get("revisions", obj)
    else:
        obj = json.loads(spec)
    if not isinstance(obj, dict) or not all(re.match(r"^[0-9a-f]{7,40}$", str(v)) for v in obj.values()):
        raise ValueError("revisions must map hf_id -> commit hash")
    return dict(sorted(obj.items()))


def read_j12_run(R, P, arm_name):
    """The finished J12 run dir of one arm: BENCH_CONFIG, the sequence list (.done markers) and every telemetry."""
    rd = bc.bench_run_dir(R, P.core_dataset, arm_name, "D", P.split)
    cfg_p = f"{rd}/BENCH_CONFIG.json"
    if not os.path.isfile(cfg_p):
        raise FileNotFoundError(f"{arm_name}: no J12 run ({cfg_p})")
    cfg = _jload(cfg_p)
    if cfg.get("protocol") != "D" or cfg.get("variant") not in (None, ""):
        raise ValueError(f"{rd}: not the main Protocol D run (protocol {cfg.get('protocol')}, variant {cfg.get('variant')})")
    done = sorted(f[len(".done_"):] for f in os.listdir(rd) if f.startswith(".done_"))
    tels = {s: _jload(f"{rd}/telemetry/{s}.json") for s in done}
    return rd, cfg, _sha256(cfg_p), tels


def run_replay(arm, frames, cache, plan, tau, *, patched, log, progress_every):
    protd = ReplayProtocolD(frames.fps, tau, plan)
    cap, mot = Capture(), bc.MotWriter()
    ctx = per_object_memory_flags(arm.model) if patched else contextlib.nullcontext(False)
    with ctx as active:
        if patched and not active:
            raise RuntimeError(f"{arm.name}: no batched memory encoder to patch")
        st = bc.track_stream(arm, frames, protocol="D", detector=cache.detector(), protd=protd, mot=mot, tel=None,
                             on_frame=cap, progress_every=progress_every, log=log)
    if protd.replayed != len(plan):
        raise bc.BenchAssertError(f"replayed {protd.replayed} of {len(plan)} planned prompts")
    keep = {k: st.get(k) for k in ("frames", "forwards", "rows", "objects", "wall_s", "births", "protocol_D")}
    keep["events"] = [[e["type"], int(e["fi"]), e.get("id")] for e in protd.events
                      if e["type"] in ("seed", "birth", "death")]
    return cap, mot.lines(), keep


def main(raw, *, dbutils_=None, R=None, hooks=None):
    h = dict(PROD_HOOKS)
    h.update(hooks or {})
    log = h["log"]
    t0 = time.time()
    P = parse_params(raw)
    R = R if R is not None else tol_paths.load(version=1)
    out_root = R.out("paper1.root", "evidence", "protD_birth_probe", P.tag)
    local = os.path.join(h["local_root"], "out", P.tag)
    src = h["notebook_source"] if h.get("notebook_source") is not None else _notebook_source()

    def publish(rel, text):
        loc = _write_local(os.path.join(local, rel), text)
        if P.publish:
            bc.publish_no_overwrite(loc, f"{out_root}/{rel}", dbutils_)
        return loc

    def published(rel):
        return P.publish and os.path.isfile(f"{out_root}/{rel}")

    with open(bc.protocol_file_path(R, "protD_tau_v1.json"), "rb") as f:
        tau_raw = f.read()
    tau = round(float(json.loads(tau_raw.decode("utf-8"))["tau"]), 2)
    runs = {a: read_j12_run(R, P, a) for a in P.arms}
    core = bc.code_sha256()
    for a, (rd, cfg, _s, tels) in runs.items():
        if (cfg.get("code") or {}).get("tol_bench_core", {}).get("source") != core["source"]:
            raise RuntimeError(f"{a}: the J12 run used tol_bench_core {(cfg.get('code') or {}).get('tol_bench_core')}, "
                               f"this probe imports {core['source'][:16]} (one core per comparison)")
        if float((cfg.get("run") or {}).get("tau")) != tau:
            raise RuntimeError(f"{a}: J12 tau {(cfg.get('run') or {}).get('tau')} != frozen {tau}")
        req = (cfg.get("extra") or {}).get("seqs_requested") or []
        n_want = len(req) if req else ((cfg.get("run") or {}).get("sequences") or {}).get("n_split")
        if n_want is None or len(tels) != int(n_want):
            raise RuntimeError(f"{a}: {len(tels)} .done sequences of {n_want}: the J12 run is incomplete")
    pcfg = {"notebook": NB_NAME, "notebook_version": NB_VERSION, "notebook_source": bc.source_shas(src),
            "tol_bench_core": core, "tol_paths": bc.source_shas(tol_paths.__file__), "patch": TOL60_PROBE_SOURCE,
            "dataset": P.core_dataset, "split": P.split, "arms": P.arms, "max_seqs": P.max_seqs,
            "max_frames": P.max_frames, "selection_rule": "most birth frames IN THE FIRST ARM'S RUN among sequences of "
            "<= max_frames frames with >= 1 birth; ties -> fewer frames, then the name; the same sequences for every arm "
            "(paired across arms)", "selection_arm": P.arms[0], "tau": tau,
            "tau_file_sha256": hashlib.sha256(tau_raw).hexdigest(),
            "j12_runs": {a: {"run_dir": rd, "bench_config_sha256": s} for a, (rd, _c, s, _t) in runs.items()},
            "groups": {"pre_exposure": "seed objects before their first exposure (must be identical)",
                       "existing": "objects at and after the first birth frame strictly after their creation",
                       "newborn": "born objects before their own first exposure (batch composition only)"},
            "iou": "per object-frame mask IoU of the EMITTED masks; one-sided = 0 (G1-5 aggregation)"}
    if published("PROBE_CONFIG.json"):
        old = _jload(f"{out_root}/PROBE_CONFIG.json")
        diff = [k for k in pcfg if k != "created_utc" and old.get(k) != json.loads(json.dumps(pcfg[k]))]
        if diff:
            raise RuntimeError(f"PROBE_CONFIG.json exists with different {diff}: use a new tag")
    else:
        publish("PROBE_CONFIG.json", _jdump(dict(pcfg, created_utc=_now())))
    probe = h["gpu_preflight"](P) if h.get("gpu_preflight") else {"skipped": True}
    revisions = resolve_revisions(R, P.revisions_spec)
    summary = {"arms": {}, "gpu": probe}
    ref = runs[P.arms[0]][3]
    sel = select_sequences([exposure(ref[s]) for s in sorted(ref)], P.max_seqs, P.max_frames)
    for a in P.arms:
        rd, cfg, _s, tels = runs[a]
        exp_rows = [exposure(tels[s]) for s in sorted(tels)]
        exp = {"per_sequence": exp_rows, "pooled": summarize_exposure(exp_rows)}
        if not published(f"{a}/exposure.json"):
            publish(f"{a}/exposure.json", _jdump(exp))
        missing = [q for q in sel if q not in tels]
        if missing:
            raise RuntimeError(f"{a}: selected sequences {missing} have no J12 output")
        log(f"{a}: exposure {json.dumps(exp['pooled'])[:300]}; selected {sel}")
        token = h["hf_token"](dbutils_) if h.get("hf_token") else None
        arm = h["arm_loader"](a, R, token, revisions, dbutils_, log)
        del token
        for k in ("hf_id", "commit_hash", "dtype"):
            if (cfg.get("arm") or {}).get(k) != arm.info.get(k):
                raise RuntimeError(f"{a}: loaded {k} {arm.info.get(k)!r} != J12 {(cfg.get('arm') or {}).get(k)!r}")
        js, ls = ((cfg.get("arm") or {}).get("student") or {}), (arm.info.get("student") or {})
        if js.get("sha256") != ls.get("sha256"):
            raise RuntimeError(f"{a}: student {ls.get('sha256')} != J12 {js.get('sha256')}")
        if not has_batched_encoder(arm.model):
            rec = {"arm": a, "batched_memory_encoder": False, "selected": sel,
                   "note": "memories are encoded per object: births do not change other objects' encoding"}
            if not published(f"{a}/summary.json"):
                publish(f"{a}/summary.json", _jdump(rec))
            summary["arms"][a] = rec
            del arm
            h["mem_settle"]()
            continue
        per_seq = {}
        for seq in sel:
            rel = f"{a}/{seq}/compare.json"
            if published(rel):
                per_seq[seq] = _jload(f"{out_root}/{rel}")
                log(f"{a} {seq}: published earlier -> reused")
                continue
            tel = tels[seq]
            cache_p = (tel.get("protocol_D") or {}).get("cache", {}).get("path") or bc.detection_cache_path(
                R, P.core_dataset, cfg["detector"]["tag"], P.split, seq)
            cache_sha = _sha256(cache_p)
            if cache_sha != (tel.get("protocol_D") or {}).get("cache", {}).get("sha256"):
                raise RuntimeError(f"{seq}: detection cache sha {cache_sha[:12]} != the J12 telemetry's")
            cache = bc.DetectionCache.from_json(_jload(cache_p))
            if cache.meta.get("ckpt_sha256") != cfg["detector"]["ckpt_sha256"]:
                raise RuntimeError(f"{seq}: cache detector {cache.meta.get('ckpt_sha256')} != J12 detector")
            plan = plan_from_run(tel, cache)
            sdst = os.path.join(h["local_root"], "stage", P.core_dataset, P.split, seq)
            if P.core_dataset == "pigtrack":
                bc.stage_tree(R.path("datasets.pigtrack.sequences", None, P.split, seq), sdst, dbutils_)
                mk = lambda: bc.PigTrackFrames(sdst)                                         # noqa: E731
            else:
                bc.stage_tree(R.path("datasets.edinburgh_pigs.decoded_frames", None, seq, "video1"), sdst, dbutils_)
                mk = lambda: bc.EdinburghFrames(sdst)                                        # noqa: E731
            if len(mk()) != int(tel["source"]["n_frames"]):
                raise RuntimeError(f"{seq}: {len(mk())} frames staged vs {tel['source']['n_frames']} in J12")
            m0 = h["mem_settle"]()
            capU, linesU, stU = run_replay(arm, mk(), cache, plan, tau, patched=False, log=log,
                                           progress_every=P.progress_every)
            m1 = h["mem_settle"]()
            capP, linesP, stP = run_replay(arm, mk(), cache, plan, tau, patched=True, log=log,
                                           progress_every=P.progress_every)
            m2 = h["mem_settle"]()
            with open(f"{rd}/mot/{seq}.txt", encoding="utf-8") as f:
                j12 = f.read().splitlines()
            fidelity = {"mot_identical_to_j12": linesU == j12, "rows_j12": len(j12), "rows_U": len(linesU),
                        "events_identical_to_j12": [[e["type"], int(e["fi"]), e.get("id")]
                                                    for e in tel["protocol_D"]["events"]
                                                    if e["type"] in ("seed", "birth", "death")] == stU["events"]}
            cmpr = compare(capU, capP, plan)
            rec = {"arm": a, "seq": seq, "frames": int(tel["source"]["n_frames"]), "night": tel["source"].get("night"),
                   "plan": {"prompts": len(plan), "seeds": sum(1 for p in plan if p.source == "protD_seed"),
                            "births": sum(1 for p in plan if p.source == "protD_birth")},
                   "replay_fidelity": fidelity, "compare": cmpr, "mot": compare_mot(linesU, linesP),
                   "U": stU, "P": stP, "memory_mib": {"before_U": m0, "before_P": m1, "after_P": m2},
                   "exposure": next(r for r in exp_rows if r["seq"] == seq)}
            publish(f"{a}/{seq}/U.mot.txt", "\n".join(linesU) + ("\n" if linesU else ""))
            publish(f"{a}/{seq}/P.mot.txt", "\n".join(linesP) + ("\n" if linesP else ""))
            publish(rel, _jdump(rec))
            per_seq[seq] = rec
            ex = cmpr["groups"]["existing"]
            log(f"{a} {seq}: fidelity {fidelity['mot_identical_to_j12']}, pre-birth identical "
                f"{cmpr['digests_identical_before_first_birth']}, existing n={ex.get('n')} mean={ex.get('mean')} "
                f"min={ex.get('min')} one-sided={ex.get('one_sided')}; MOT changed {rec['mot']['rows_changed']}")
        arm_sum = dict(summarize_arm(per_seq), arm=a, batched_memory_encoder=True, selected=sel,
                       exposure_pooled=exp["pooled"]) if per_seq else {"arm": a, "selected": sel, "note": "no sequence"}
        if not published(f"{a}/summary.json"):
            publish(f"{a}/summary.json", _jdump(arm_sum))
        summary["arms"][a] = arm_sum
        del arm
        h["mem_settle"]()
    summary["wall_s"] = round(time.time() - t0, 1)
    summary["out_root"] = out_root
    return summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local selftest (`TOL_SELFTEST=1`; stub tracker; the J12 fixture comes from Tol_63's own `main`)

# COMMAND ----------


def selftest():
    import test_tol_bench_core as tb
    from PIL import Image
    res = []

    def check(name, cond, detail=""):
        res.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + ("" if cond or not detail else f"  [{detail}]"), flush=True)

    def raises(fn, exc=Exception, match=None):
        try:
            fn()
        except exc as e:  # noqa: BLE001
            return match is None or re.search(match, str(e)) is not None
        return False

    # A stand-in for the HF batched memory encoder. Like transformers 5.2.0, ONE call per frame encodes every object with
    # is_mask_from_pts = any(flags); an object that was NOT prompted on this frame but is encoded with True (a birth
    # frame) "drifts" +1 px on every later frame. Under the patch the unprompted objects are encoded in their own
    # all-False group and never drift. The frame's own outputs are computed before its memory is encoded.
    class BatchedStub(tb.StubModel):
        def __call__(self, inference_session=None, frame_idx=None, frame=None):
            s = inference_session
            new = set(s.obj_with_new_inputs)                 # the parent clears the aliased list
            out = super().__call__(inference_session=s, frame_idx=frame_idx, frame=frame)
            drift = getattr(s, "drift", {})
            for oid, idx in s._id2idx.items():
                d = drift.get(oid, 0)
                if d and out.object_score_logits[idx, 0] > 0:
                    out.pred_masks[idx, 0] = np.roll(out.pred_masks[idx, 0], d, axis=1)
            objs = list(s._id2idx)
            self._batch_encode_memories(inference_session=s, frame_idx=frame_idx,
                                        objects_needing_memory_encoding=objs,
                                        high_res_masks_for_memory=[None] * len(objs),
                                        object_score_logits_for_memory=[None] * len(objs),
                                        is_mask_from_pts_per_obj=[o in new for o in objs])
            return out

        def _batch_encode_memories(self, *, inference_session, frame_idx, objects_needing_memory_encoding,
                                   high_res_masks_for_memory, object_score_logits_for_memory, is_mask_from_pts_per_obj):
            s = inference_session
            s.drift = getattr(s, "drift", {})
            if any(is_mask_from_pts_per_obj):                 # the library's batched flag
                for oid, f in zip(objects_needing_memory_encoding, is_mask_from_pts_per_obj):
                    if not f:
                        s.drift[oid] = s.drift.get(oid, 0) + 1

    # ---------------- 1. the patch groups the batch by each object's own flag ----------------
    # a seeded at 0; b seen from 10 -> born at pass 20; c seen from 15 -> born at pass 30 (fps 10: passes every 10)
    world = {"a": tb.static((10, 10, 40, 40)), "b": tb.static((60, 10, 90, 40), frames=set(range(10, 60))),
             "c": tb.static((100, 50, 130, 90), frames=set(range(15, 60)))}

    class Rec:
        def __init__(self):
            self.calls = []

        def _batch_encode_memories(self, *, inference_session, frame_idx, objects_needing_memory_encoding,
                                   high_res_masks_for_memory, object_score_logits_for_memory, is_mask_from_pts_per_obj):
            self.calls.append((list(objects_needing_memory_encoding), list(is_mask_from_pts_per_obj)))
    r = Rec()
    with per_object_memory_flags(r) as on:
        r._batch_encode_memories(inference_session=None, frame_idx=3, objects_needing_memory_encoding=[0, 1, 2],
                                 high_res_masks_for_memory=[0, 1, 2], object_score_logits_for_memory=[0, 1, 2],
                                 is_mask_from_pts_per_obj=[False, False, True])
    check("patch: one call per flag group, own flags, class method restored afterwards",
          on and r.calls == [([0, 1], [False, False]), ([2], [True])] and "_batch_encode_memories" not in r.__dict__)
    with per_object_memory_flags(tb.StubModel(world, 100, 160)) as on2:
        pass
    check("patch: a model without a batched encoder is left alone (EdgeTAM)", on2 is False
          and not has_batched_encoder(tb.StubModel(world, 100, 160)) and has_batched_encoder(BatchedStub(world, 100, 160)))

    # ---------------- 2. the J12 fixture: Tol_63's own main() on a temp registry ----------------
    nb63 = os.environ.get("TOL63_NB") or os.path.join(CODE_DIR, "Tol_63_track_bench.py")
    src63 = open(nb63, encoding="utf-8").read()
    cut = src63.index("# MAGIC ## Run")
    ns63 = {"__name__": "tol63_fixture", "__file__": nb63}
    os.environ["TOL_SELFTEST"] = "0"                      # Tol_63's own selftest block must not run inside the exec
    try:
        exec(compile(src63[:cut], nb63, "exec"), ns63)                                  # definitions only
    finally:
        os.environ["TOL_SELFTEST"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = tmp.replace("\\", "/")
        reg_yaml = os.path.join(os.path.dirname(tol_paths.__file__), "paths_v1.yaml")
        R0 = tol_paths.load(path=reg_yaml)
        data = json.loads(json.dumps(R0.data))
        data["roots"].update(volume=tmp + "/vol", mount=tmp + "/mnt", mount_uri="file:" + tmp + "/mnt")
        R = tol_paths.Registry(data, source=reg_yaml, sha256=R0.sha256)

        def jpeg(p, v=0):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray(np.full((12, 16, 3), v % 256, np.uint8)).save(p)
        N = 60
        seqs = ["pigtrack0055", "pigtrack0011", "pigtrack0002"]
        for q, seq in enumerate(seqs):
            d = R.path("datasets.pigtrack.sequences", None, "test", seq)
            n = N if seq != "pigtrack0002" else 30
            for k in range(n):
                jpeg(f"{d}/img1/{k + 1:08d}.jpg", v=3 * k + q)
            with open(f"{d}/seqinfo.ini", "w", encoding="utf-8") as f:
                f.write(f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=10\nseqLength={n}\nimWidth=16\nimHeight=12\n"
                        "imExt=.jpg\npenID=3\nnight=0\n")
            os.makedirs(f"{d}/gt", exist_ok=True)
            with open(f"{d}/gt/gt.txt", "w", encoding="utf-8") as f:
                f.write("1,1,10,10,30,30,1,1,1\n")
        sm = R.path("datasets.pigtrack.seqmaps", None, "pigtrack-test.txt")
        os.makedirs(os.path.dirname(sm), exist_ok=True)
        with open(sm, "w", encoding="utf-8") as f:
            f.write("name\n" + "".join(f"{q}\n" for q in sorted(bc.PIGTRACK_TEST)))
        ck = R.path("models.detector_pig_pt", None, "checkpoint_best_total.pth")
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        with open(ck, "wb") as f:
            f.write(b"fake rf-detr checkpoint")
        tf = bc.protocol_file_path(R, "protD_tau_v1.json")
        os.makedirs(os.path.dirname(tf), exist_ok=True)
        with open(tf, "w", encoding="utf-8") as f:
            json.dump({"tau": 0.5}, f)

        def make_arm(name, batched=True):
            model = (BatchedStub if batched else tb.StubModel)(world, 100, 160)
            arm = bc.Arm(name, "sam3", model, tb.StubProcessor(100, 160), device="cpu")
            arm.info = {"arm": name, "hf_id": f"stub/{name}", "commit_hash": "0" * 40, "attn_request": "accepted",
                        "attn_resolved": {"config": "sdpa"}, "dtype": "bfloat16", "model_config": {},
                        "processor": {}, "student": {"sha256": "s" * 64} if name.startswith("S-") else None}
            return arm

        def det_loader(ckpt_local, resolution):
            def det(fi, _pil):
                out = [(world[n_](fi), sc) for n_, sc in (("a", 0.9), ("b", 0.85), ("c", 0.8)) if world[n_](fi)]
                return [(tuple(float(v) + 0.123456 for v in b), s) for b, s in out]
            det.info = {"ckpt_sha256": _sha256(ckpt_local), "resolution": resolution, "num_classes": 1,
                        "class_names": ["pig"], "threshold": bc.DET_PROBE_THRESHOLD}
            return det

        def hooks63(batched):
            return {"local_root": tmp + "/local63", "detector_loader": det_loader, "gpu_preflight": None,
                    "hf_token": None, "use_cuda": False, "log": lambda *a: None, "notebook_source": src63.encode(),
                    "arm_loader": lambda P_, R_, t, rv, dbu, lg: make_arm(P_.arm, batched)}
        raw63 = dict(ns63["WIDGETS"], require_gpu="any", dataset="pigtrack", protocol="D", detector_resolution="512",
                     seqs=",".join(seqs))
        s63 = ns63["main"](dict(raw63, arm="T"), R=R, hooks=hooks63(True))
        s63b = ns63["main"](dict(raw63, arm="ET"), R=R, hooks=hooks63(False))
        rd = s63["run_dir"]
        tel = _jload(f"{rd}/telemetry/pigtrack0055.json")
        check("fixture: Tol_63 Protocol D test run: seed a at 0, births b at pass 20 and c at pass 30",
              [e["type"] for e in tel["protocol_D"]["events"] if e["type"] in ("seed", "birth")] == ["seed", "birth",
                                                                                                    "birth"]
              and tel["stats"]["births"] == [[0, [0]], [20, [1]], [30, [2]]], json.dumps(tel["stats"]["births"]))
        cache = bc.DetectionCache.from_json(_jload(tel["protocol_D"]["cache"]["path"]))

        # ---------------- 3. the plan: exact cached boxes from rounded events ----------------
        plan = plan_from_run(tel, cache)
        check("plan: exact cached boxes (not the rounded event boxes), ids 0..n-1, = stats.births",
              [p.obj_id for p in plan] == [0, 1, 2] and plan[1].box == cache.dets[20][1][0]
              and plan[1].box != tuple(tel["protocol_D"]["events"][1]["box"]))
        bad = json.loads(json.dumps(tel))
        bad["protocol_D"]["events"][1]["box"][0] += 0.02
        check("plan: an event that matches no cached detection is refused",
              raises(lambda: plan_from_run(bad, cache), bc.BenchAssertError, "matches 0"))
        bad2 = json.loads(json.dumps(tel))
        bad2["stats"]["births"] = [[0, [0]], [20, [1, 2]]]
        check("plan: a plan that disagrees with stats.births is refused",
              raises(lambda: plan_from_run(bad2, cache), bc.BenchAssertError, "stats.births"))

        # ---------------- 4. replay fidelity and the effect ----------------
        frames = lambda: bc.PigTrackFrames(R.path("datasets.pigtrack.sequences", None, "test", "pigtrack0055"))  # noqa
        armU = make_arm("T")
        capU, linesU, stU = run_replay(armU, frames(), cache, plan, 0.5, patched=False, log=lambda *a: None,
                                       progress_every=0)
        with open(f"{rd}/mot/pigtrack0055.txt", encoding="utf-8") as f:
            j12 = f.read().splitlines()
        check("replay U: MOT byte-identical to the Tol_63 run (the drift of existing objects after births included)",
              linesU == j12 and len(j12) > 0, f"{len(linesU)} vs {len(j12)}")
        capP, linesP, stP = run_replay(make_arm("T"), frames(), cache, plan, 0.5, patched=True, log=lambda *a: None,
                                       progress_every=0)
        cm = compare(capU, capP, plan)
        g = cm["groups"]
        check("compare: frames up to the first birth frame are digest-identical, pre-exposure masks identical",
              cm["digests_identical_before_first_birth"] and cm["first_digest_diff"] == 21
              and g["pre_exposure"]["n"] == 20 and g["pre_exposure"]["identical"] == 20, json.dumps(cm)[:400])
        check("compare: existing objects after a birth differ (stub drift: a from 21, b from 31); newborns identical "
              "until their own exposure (b 20-29, c 30-59)",
              g["existing"]["n"] == 40 + 30 and g["existing"]["identical"] == 1 + 1 and g["existing"]["mean"] < 1.0
              and g["newborn"]["n"] == 10 + 30 and g["newborn"]["identical"] == 40 and g["existing"]["one_sided"] == 0
              and set(cm["existing_per_object_mean_iou"]) == {"0", "1"}, json.dumps(g))
        check("replay P: the same prompts at the same frames as U (only the encoding differs)",
              stP["births"] == stU["births"] and stP["events"] == stU["events"])
        mm = compare_mot(linesU, linesP)
        check("compare_mot: changed rows only for exposed objects, counts add up",
              mm["rows_changed"] > 0 and mm["rows_identical"] + mm["rows_changed"] + mm["rows_only_U"] == mm["rows_U"])
        check("rle_iou: identical -> 1, one-sided -> 0, none -> None",
              rle_iou(bc.rle_encode(np.ones((2, 2), bool)), bc.rle_encode(np.ones((2, 2), bool))) == 1.0
              and rle_iou(None, bc.rle_encode(np.ones((2, 2), bool))) == 0.0 and rle_iou(None, None) is None)

        # ---------------- 5. exposure and selection ----------------
        ex = exposure(tel)
        check("exposure: seeds / births / birth frames / alive objects at each birth / exposed object-frames",
              (ex["seeds"], ex["births"], ex["birth_frames"], ex["objects_alive_at_birth_frames"]) == (1, 2, 2, [1, 2])
              and ex["object_frames"] == 60 + 40 + 30 and ex["object_frames_after_a_birth_encoding"] == 40 + 30,
              json.dumps(ex))
        rows = [dict(ex, seq="s1", birth_frames=3, frames=500), dict(ex, seq="s2", birth_frames=3, frames=300),
                dict(ex, seq="s3", birth_frames=5, frames=900), dict(ex, seq="s4", birth_frames=0, frames=10),
                dict(ex, seq="s5", birth_frames=1, frames=100)]
        check("selection: most birth frames among <= max_frames, ties -> fewer frames, then name; zero-birth excluded",
              select_sequences(rows, 2, 600) == ["s2", "s1"] and select_sequences(rows, 9, 1000) == ["s3", "s2", "s1",
                                                                                                       "s5"])

        # ---------------- 6. main(): end to end on the fixture (T batched; ET without a batched encoder) ----------------
        def hooks73(batched_by_arm):
            return {"local_root": tmp + "/local73", "gpu_preflight": None, "hf_token": None,
                    "notebook_source": open(os.path.abspath(__file__), "rb").read(), "mem_settle": lambda: None,
                    "log": lambda *a: None,
                    "arm_loader": lambda name, R_, t, rv, dbu, lg: make_arm(name, batched_by_arm[name])}
        raw73 = dict(WIDGETS, arms="T,ET", require_gpu="any", max_seqs="2", max_frames="100", revisions="")
        s73 = main(raw73, R=R, hooks=hooks73({"T": True, "ET": False}))
        out = s73["out_root"]
        a_t = s73["arms"]["T"]
        check("main: T selects the two 60-frame sequences with 2 birth frames (0002 has 1), by name",
              a_t["selected"] == ["pigtrack0011", "pigtrack0055"], json.dumps(a_t["selected"]))
        check("main: replay fidelity (MOT and events) on every selected sequence; pre-birth digests identical; effect",
              all(a_t["replay_fidelity"].values()) and a_t["pre_birth_digests_identical"]
              and a_t["existing"]["n"] == 2 * 70 and a_t["existing"]["mean"] < 1.0 and a_t["mot"]["rows_changed"] > 0,
              json.dumps(a_t)[:600])
        c1 = _jload(f"{out}/T/pigtrack0011/compare.json")
        check("main: per-sequence record holds both fidelity checks, the plan and the exposure",
              c1["replay_fidelity"]["mot_identical_to_j12"] and c1["replay_fidelity"]["events_identical_to_j12"]
              and c1["plan"] == {"prompts": 3, "seeds": 1, "births": 2} and c1["exposure"]["birth_frames"] == 2)
        check("main: ET (no batched encoder) recorded as not affected, never replayed",
              s73["arms"]["ET"]["batched_memory_encoder"] is False
              and not os.path.isdir(f"{out}/ET/pigtrack0011"))
        ex_t = _jload(f"{out}/T/exposure.json")["pooled"]
        check("main: pooled exposure over the 3 fixture sequences (seeds 3, births 5, 5 birth frames)",
              (ex_t["sequences"], ex_t["seeds"], ex_t["births"], ex_t["birth_frames"]) == (3, 3, 5, 5), json.dumps(ex_t))
        check("main: outputs published under evidence/protD_birth_probe/<tag>/ only",
              os.path.isfile(f"{out}/PROBE_CONFIG.json") and os.path.isfile(f"{out}/T/summary.json")
              and os.path.isfile(f"{out}/T/exposure.json") and out.endswith("/evidence/protD_birth_probe/j12_v1"))
        n_files = sum(len(fs) for _r, _d, fs in os.walk(out))
        s73b = main(raw73, R=R, hooks=hooks73({"T": True, "ET": False}))
        check("main: resume reuses every published sequence and republishes nothing different",
              sum(len(fs) for _r, _d, fs in os.walk(out)) == n_files and s73b["arms"]["T"] == a_t)
        check("main: a different configuration under the same tag is refused",
              raises(lambda: main(dict(raw73, max_seqs="3"), R=R, hooks=hooks73({"T": True, "ET": False})),
                     RuntimeError, "new tag"))
        ns63["main"](dict(raw63, arm="S-PT"), R=R, hooks=hooks63(True))
        s73p = main(dict(raw73, arms="T,S-PT", tag="j12_pair"), R=R,
                    hooks=hooks73({"T": True, "S-PT": True}))
        check("main: the selection comes from the first arm (T) and is replayed for every arm (paired across arms)",
              s73p["arms"]["T"]["selected"] == s73p["arms"]["S-PT"]["selected"] == ["pigtrack0011", "pigtrack0055"]
              and all(s73p["arms"]["S-PT"]["replay_fidelity"].values())
              and _jload(f"{s73p['out_root']}/PROBE_CONFIG.json")["selection_arm"] == "T")
        _tamper_core(rd)
        check("main: a J12 run made with another core is refused",
              raises(lambda: main(dict(raw73, tag="j12_other"), R=R, hooks=hooks73({"T": True, "ET": False})),
                     RuntimeError, "one core"))
        del s63b

    # ---------------- 7. static audit ----------------
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    vol_lit, dbfs_lit = "/Vol" + "umes", "/db" + "fs"
    check("audit: no literal Volume / dbfs path; no secret-shaped strings; SOURCE format; same %pip line as Tol_63",
          vol_lit not in src and dbfs_lit not in src and not re.search(r"hf_[A-Za-z0-9]{20,}", src)
          and "secrets." + "get(" not in src and src.startswith("# Databricks notebook source")
          and '"transformers==5.2.0"' in src and '"rfdetr==1.10.1"' in src)
    t60 = os.path.join(CODE_DIR, "Tol_60_bench_g1.py")
    if os.path.isfile(t60):
        def fn_src(text, name):
            i = text.index(f"def {name}(")
            j = text.index("\n\n\n", i)
            return text[i:j]
        check("audit: per_object_memory_flags is Tol_60's verbatim",
              fn_src(open(t60, encoding="utf-8").read(), "per_object_memory_flags") == fn_src(src, "per_object_memory_flags"))
    n_fail = sum(1 for r in res if not r)
    print(f"\n{NB_NAME} selftest: {len(res) - n_fail}/{len(res)} passed")
    return n_fail


def _tamper_core(rd):
    p = f"{rd}/BENCH_CONFIG.json"
    cfg = _jload(p)
    cfg["code"]["tol_bench_core"]["source"] = "0" * 64
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return True


if SELFTEST:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _nf = selftest()
    raise SystemExit(1 if _nf else 0)

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
            out[name] = os.environ.get("TOL73_" + name.upper(), default)
    return out


RAW = read_widgets()
print("widgets:", json.dumps(RAW, sort_keys=True))
SUMMARY = main(RAW, dbutils_=dbutils)
print(json.dumps(SUMMARY, indent=1, default=str)[:20000])

# COMMAND ----------

if IS_NOTEBOOK:
    dbutils.notebook.exit(json.dumps({"arms": {a: {k: v for k, v in s.items() if k in (
        "selected", "replay_fidelity", "pre_birth_digests_identical", "existing", "newborn", "pre_exposure",
        "existing_per_object_mean_iou", "mot", "batched_memory_encoder")} for a, s in SUMMARY["arms"].items()},
        "wall_s": SUMMARY.get("wall_s"), "out_root": SUMMARY.get("out_root")}, default=str)[:60000])
