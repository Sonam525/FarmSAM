# Databricks notebook source
# MAGIC %md
# MAGIC # Extract SAM 3 targets for the PigTrack student
# MAGIC
# MAGIC Stores `vision_encoder.backbone` targets `[1,1024,72,72]` (fp16) for 3,000 frames of the 35 PigTrack train sequences,
# MAGIC apportioned by image count and evenly spaced inside each sequence. No ground truth is read. Four whole train sequences
# MAGIC are held out for checkpoint selection (`distill_core.choose_val_sequences`). Next step: `distill_student_pigtrack`.

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

NB_NAME = "extract_teacher_targets_pigtrack"
NB_VERSION = "extract_teacher_targets_pigtrack-2026-10-14.2"
CONFIG_VERSION = "extract_teacher_targets_pigtrack-targets-v1"
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
# MAGIC ## Parameters and the pure-Python driver

# COMMAND ----------

WIDGETS = collections.OrderedDict([
    ("n_targets", str(dc.N_TARGETS_DEFAULT)), ("val_seqs", ""), ("sam3_revision", "protocol:hf_revisions.json"),
    ("require_gpu", "A100"), ("copy_threads", "16"), ("progress_every", "5"),
])
HF_SAM3 = "facebook/sam3"
REGISTRY_VERSION = 1
TARGETS_KEY = "paper1.teacher_targets.pigtrack_train"
PRUNE_EVERY_T01, KEEP_LAST_N_T01 = 25, 8        # extract_teacher_targets's session hygiene (the backbone output does not depend on it)
PROMPT_INDEP_MIN_COS = 0.9999                    # same frame, two sessions, two prompt sets -> the same backbone output


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse_params(raw):
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}; known: {list(WIDGETS)}")
    g = {k: str(raw.get(k) if raw.get(k) not in (None, "") else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace()
    P.n_targets = int(g["n_targets"])
    if not 1 <= P.n_targets <= 9959:
        raise ValueError(f"n_targets={P.n_targets}")
    P.val_seqs = [s for s in re.split(r"[,\s]+", g["val_seqs"]) if s]
    P.sam3_revision = g["sam3_revision"]
    if P.sam3_revision and not (P.sam3_revision.startswith("protocol:") or re.match(r"^[0-9a-f]{40}$", P.sam3_revision)):
        raise ValueError(f"sam3_revision={P.sam3_revision!r}: '' | protocol:<file>.json | a 40-hex commit")
    P.require_gpu = g["require_gpu"]
    if P.require_gpu not in ("A100", "A10", "any"):
        raise ValueError(f"require_gpu={P.require_gpu!r}")
    P.copy_threads = int(g["copy_threads"])
    P.progress_every = int(g["progress_every"])
    return P


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _sha_text(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


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


def build_config(P, infos, plan, val_seqs, val_work, rev, rev_src, versions, gpu_class, target_shape):
    cfg = {
        "version": CONFIG_VERSION,
        "dataset": {"name": "PigTrack (CC BY 4.0, DOI 10.25625/P7VQTP)", "split": "train",
                    "infos": {s: {k: infos[s][k] for k in ("n", "fps", "width", "height", "pen", "night",
                                                           "seq_length_ini")} for s in sorted(infos)},
                    "gt_read": False},
        "selection": {"rule": dc.SELECTION_RULE, "n_targets": P.n_targets, "plan_sha256": dc.plan_digest(plan),
                      "per_seq": {s: len(v) for s, v in sorted(plan.items())}},
        "val": {"seqs": val_seqs, "rule": dc.VAL_RULE, "picks": val_work.get("picks"),
                "override": bool(P.val_seqs), "n_val_targets": sum(len(plan[s]) for s in val_seqs)},
        "teacher": {"hf_id": HF_SAM3, "revision": rev, "revision_source": rev_src, "dtype": "bfloat16",
                    "transformers": core.TRANSFORMERS_PIN, "attention": "model default (resolved recorded; FA2 refused)",
                    "capture": "forward hook on vision_encoder.backbone; last_hidden_state [1,5184,1024] -> "
                               "permute/reshape [1,1024,72,72] -> .cpu().half() (extract_teacher_targets BackboneFeatureHook)",
                    "input": "Sam3TrackerVideoProcessor (1008x1008), one session per sequence, implicit frame index",
                    "prompt": {"dummy_box_fracs": dc.DUMMY_BOX_FRACS, "note": "backbone output is prompt-independent"},
                    "prune": {"every": PRUNE_EVERY_T01, "keep_last_n": KEEP_LAST_N_T01}},
        "target": {"shape": list(target_shape), "dtype": dc.TARGET_DTYPE, "format": "torch.save(tensor)",
                   "frames": "img1 JPEG copied byte for byte"},
        "versions": {k: versions.get(k) for k in ("python", "torch", "transformers", "numpy", "Pillow", "cuda",
                                                    "cudnn")},
        "gpu_class": gpu_class, "created_utc": None, "extra": {"versions_all": versions},
    }
    return json.loads(dc.json_dumps(cfg))


def check_or_publish_config(OUT, cfg, local, dbutils_, log):
    p = f"{OUT}/TARGETS_CONFIG.json"
    if osp.isfile(p):
        stored = json.loads(_read(p))
        core.config_check(stored, cfg)
        log("TARGETS_CONFIG.json matches: resuming")
        return stored
    cfg = dict(cfg, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    lp = core.write_text_atomic(osp.join(local, "TARGETS_CONFIG.json"), dc.json_dumps(cfg))
    core.publish_no_overwrite(lp, p, dbutils_)
    return cfg


def main(raw, *, hooks):
    """One run: discover -> plan -> config -> stage -> teacher -> per-sequence extract + publish -> manifest."""
    h = dict(hooks)
    log = h.get("log", _log)
    dbu = h.get("dbutils")
    t_run = time.time()
    P = parse_params(raw)
    R = h.get("registry") or farm_paths.load(version=REGISTRY_VERSION)
    OUT = R.out(TARGETS_KEY)
    LOCAL = h["local_root"]
    os.makedirs(LOCAL, exist_ok=True)
    target_shape = tuple(h.get("target_shape", dc.TARGET_SHAPE))

    train, infos, seq_root, src_sha = discover(R, h.get("expect_train_seqs", dc.EXPECT_TRAIN_SEQS))
    for s in train:
        if infos[s]["warning"]:
            log(f"{s}: {infos[s]['warning']}")
    if P.val_seqs:
        val_seqs = dc.check_val_override(P.val_seqs, set(train))
        val_work = {"rule": "widget override", "picks": None}
    else:
        val_seqs, val_work = dc.choose_val_sequences(infos)
    plan = dc.selection_plan(infos, P.n_targets)
    log(f"{len(train)} train sequences, {sum(i['n'] for i in infos.values())} images; {P.n_targets} targets "
        f"(plan sha {dc.plan_digest(plan)[:12]}); val {val_seqs} = {sum(len(plan[s]) for s in val_seqs)} targets")

    gpu = h["gpu_preflight"](P)
    rev, rev_src = resolve_revision(R, P.sam3_revision)
    versions = h["versions"]()
    cfg = build_config(P, infos, plan, val_seqs, val_work, rev, rev_src, versions, gpu["class"], target_shape)
    cfg["dataset"].update(src_sha)
    cfg = check_or_publish_config(OUT, cfg, LOCAL, dbu, log)

    rec_dir = f"{OUT}/records"
    done = {s for s in train if osp.isfile(f"{rec_dir}/{s}.json")}
    todo = [s for s in train if plan[s] and s not in done]
    log(f"records present for {len(done)} sequences; {len(todo)} to extract")
    run = {"nb": NB_VERSION, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu,
           "code": h["code_shas"](), "todo": todo, "skipped_done": sorted(done)}

    pi = None
    if todo:
        # ---- stage the selected frames of the sequences still to do (Volume -> local NVMe, parallel, size-verified)
        jobs = [(osp.join(seq_root, s, infos[s]["imDir"], dc.src_image_name(fi, infos[s]["imExt"])),
                 osp.join(LOCAL, "src", s, dc.src_image_name(fi, infos[s]["imExt"])), "src", None)
                for s in todo for fi in plan[s]]
        st = dc.copy_files_parallel(jobs, threads=P.copy_threads, log=log)
        run["staging"] = st
        log(f"staged {st['files']} frames ({st['copied']} copied, {st['bytes_copied'] / 1e9:.2f} GB, "
            f"{st['seconds']} s, {st['mb_per_s']} MB/s)")
        # ---- teacher
        hf_token = h["hf_token"](dbu)
        teacher = h["load_teacher"](rev, hf_token)
        run["teacher"] = teacher.info
        if rev and teacher.info.get("commit_hash") != rev:
            raise RuntimeError(f"loaded commit {teacher.info.get('commit_hash')} != pinned {rev}")
        if any(v == "flash_attention_2" for v in (teacher.info.get("attn_resolved") or {}).values()):
            raise RuntimeError(f"flash_attention_2 resolved: {teacher.info.get('attn_resolved')}")
        s0 = todo[0]
        pi = teacher.prompt_independence(osp.join(LOCAL, "src", s0, dc.src_image_name(plan[s0][0], infos[s0]["imExt"])),
                                         infos[s0]["width"], infos[s0]["height"])
        run["prompt_independence"] = pi
        log(f"prompt independence on {s0} frame {plan[s0][0]}: {pi}")
        if not pi["cos_global"] >= PROMPT_INDEP_MIN_COS:
            raise core.BenchAssertError(f"backbone output depends on the prompt? {pi}")

        # ---- per sequence: extract (GPU, this thread) -> publish (one background thread) -> record
        pub = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tol66_publish")
        futs = {}

        def publish_seq(s, seq_local, files, local_sha, t_extract):
            res = dc.publish_tree_attempt(seq_local, f"{OUT}/seqs", s, local_sha=local_sha, dbutils=dbu, log=log)
            if h.get("crash_after_dir_publish") == s:
                raise RuntimeError(f"simulated crash after publishing {s}")
            rec = {"seq": s, "dir": f"seqs/{res['name']}", "publish": {k: res[k] for k in ("mode", "files", "bytes",
                                                                                            "seconds")},
                   "n": len(files), "files": files, "extract_seconds": round(t_extract, 2),
                   "info": {k: infos[s][k] for k in ("n", "fps", "width", "height", "pen", "night")},
                   "role": "val" if s in val_seqs else "train", "nb": NB_VERSION,
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
            src_of = {fi: osp.join(LOCAL, "src", s, dc.src_image_name(fi, infos[s]["imExt"])) for fi in plan[s]}
            t0 = time.time()
            files, local_sha = [], {}
            for fi, feat, stats in teacher.extract_seq([(fi, src_of[fi]) for fi in plan[s]], infos[s]["width"],
                                                       infos[s]["height"]):
                if tuple(stats["shape"]) != target_shape or stats["dtype"] != dc.TARGET_DTYPE:
                    raise core.BenchAssertError(f"{s} f{fi}: target {stats['shape']} {stats['dtype']}")
                if not (stats["finite"] and stats["std"] > core.MIN_STD):
                    raise core.BenchAssertError(f"{s} f{fi}: non-finite or collapsed teacher features {stats}")
                base = dc.target_basename(s, fi)
                fp = osp.join(seq_local, "features", base + ".pt")
                h["save_feature"](feat, fp)
                ip = osp.join(seq_local, "frames", base + ".jpg")
                shutil.copyfile(src_of[fi], ip)                       # byte-for-byte copy of the PigTrack JPEG
                fsha, isha = core.sha256_file(fp), core.sha256_file(ip)
                local_sha[f"features/{base}.pt"] = fsha
                local_sha[f"frames/{base}.jpg"] = isha
                files.append({"fi": int(fi), "src_image": dc.src_image_name(fi, infos[s]["imExt"]),
                              "image": base + ".jpg", "feature": base + ".pt", "image_bytes": osp.getsize(ip),
                              "image_sha256": isha, "feature_bytes": osp.getsize(fp), "feature_sha256": fsha,
                              "feat_mean": float(stats["mean"]), "feat_std": float(stats["std"])})
            if [f["fi"] for f in files] != list(plan[s]):
                raise core.BenchAssertError(f"{s}: extracted {len(files)} targets, planned {len(plan[s])}")
            return seq_local, files, local_sha, time.time() - t0

        t_gpu = time.time()
        try:
            for k, s in enumerate(todo):
                for fs, f in list(futs.items()):                                # fail fast on a publish error
                    if f.done() and f.exception() is not None:
                        raise RuntimeError(f"publishing failed for {fs}") from f.exception()
                seq_local, files, local_sha, t_ex = extract_one(s)
                futs[s] = pub.submit(publish_seq, s, seq_local, files, local_sha, t_ex)
                if P.progress_every and ((k + 1) % P.progress_every == 0 or k + 1 == len(todo)):
                    n_done = sum(len(plan[x]) for x in todo[:k + 1])
                    el = time.time() - t_gpu
                    log(f"  [{k + 1}/{len(todo)}] {s}: {len(files)} targets in {t_ex:.1f} s; {n_done} this run, "
                        f"{n_done / max(el, 1e-9):.2f} tgt/s, {el / 60:.1f} min")
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

    # ---- manifest from ALL records (this run's and earlier runs'), validated against the plan
    records = {}
    for s in train:
        if not plan[s]:
            continue
        p = f"{rec_dir}/{s}.json"
        if not osp.isfile(p):
            raise RuntimeError(f"record of {s} missing after the run")
        records[s] = json.loads(_read(p))
    rows = dc.manifest_rows(records, infos, val_seqs)
    summ = dc.validate_manifest(rows, plan, val_seqs, P.n_targets)

    def _vol_ok(r):
        out = []
        for k, n in (("image", r["image_bytes"]), ("feature", r["feature_bytes"])):
            p = f"{OUT}/{r[k]}"
            if not osp.isfile(p) or osp.getsize(p) != n:
                out.append(p)
        return out
    with ThreadPoolExecutor(max_workers=16) as ex:
        bad = [p for lst in ex.map(_vol_ok, rows) for p in lst]
    if bad:
        raise RuntimeError(f"{len(bad)} manifest files missing / wrong size on the Volume, e.g. {bad[0]}")
    for r in rows[:: max(1, len(rows) // 3)][:3]:                    # end-to-end sha spot check of Volume copies
        if core.sha256_file(f"{OUT}/{r['feature']}") != r["feature_sha256"]:
            raise RuntimeError(f"Volume copy of {r['feature']} differs from its record")
    lp = core.write_text_atomic(osp.join(LOCAL, "manifest.csv"), dc.manifest_csv(rows))
    man_state = core.publish_no_overwrite(lp, f"{OUT}/manifest.csv", dbu)
    run.update({"manifest": summ, "manifest_sha256": core.sha256_file(lp), "manifest_publish": man_state,
                "seconds_total": round(time.time() - t_run, 1),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if not osp.isfile(f"{OUT}/TARGETS_SUMMARY.json"):
        summary = {"manifest": summ, "manifest_sha256": run["manifest_sha256"], "val_seqs": val_seqs,
                   "per_seq": {s: {"n": records[s]["n"], "dir": records[s]["dir"], "role": records[s]["role"],
                                   "extract_seconds": records[s]["extract_seconds"],
                                   "feat_mean_of_stds": records[s]["feat_mean_of_stds"]} for s in sorted(records)},
                   "prompt_independence": pi, "run_completed": run["started_utc"], "nb": NB_VERSION,
                   "teacher_feature_std_mean": float(np.mean([r["feat_std"] for r in rows]))}
        lp2 = core.write_text_atomic(osp.join(LOCAL, "TARGETS_SUMMARY.json"), dc.json_dumps(summary))
        core.publish_no_overwrite(lp2, f"{OUT}/TARGETS_SUMMARY.json", dbu)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:6]     # one log per run
    lp3 = core.write_text_atomic(osp.join(LOCAL, "runs", f"{stamp}.json"), dc.json_dumps(run))
    core.publish_no_overwrite(lp3, f"{OUT}/runs/{stamp}.json", dbu)
    log(f"manifest: {summ['rows']} targets ({summ['train']} train / {summ['val']} val, {summ['sequences']} sequences, "
        f"{summ['bytes'] / 1e9:.1f} GB); run {run['seconds_total'] / 60:.1f} min")
    return {"out": OUT, "manifest": summ, "manifest_sha256": run["manifest_sha256"], "val_seqs": val_seqs,
            "extracted_now": len(todo), "seconds_total": run["seconds_total"], "prompt_independence": pi,
            "staging": run.get("staging"), "extract_seconds": run.get("extract_seconds")}


# COMMAND ----------

if not IS_NOTEBOOK:
    print("This notebook needs a GPU cluster and the path registry: see docs/REPRODUCE.md")
    raise SystemExit(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production hooks (GPU): SAM 3 teacher, backbone hook, CUDA preflight

# COMMAND ----------

import torch                      # noqa: E402
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
    "local_root": "/local_disk0/tol66/pigtrack_train",
    "dbutils": dbutils,
    "log": _log,
    "gpu_preflight": gpu_preflight,
    "versions": core.versions,
    "code_shas": code_shas,
    "hf_token": lambda dbu: core.hf_token_from_dbutils(dbu),
    "load_teacher": lambda rev, tok: Sam3Teacher(rev, tok),
    "save_feature": _save_feature,
}

# COMMAND ----------

for _k, _v in WIDGETS.items():
    dbutils.widgets.text(_k, _v)
RAW = {k: dbutils.widgets.get(k) for k in WIDGETS}
print("widgets:", RAW)
RESULT = main(RAW, hooks=PROD_HOOKS)
print(json.dumps(RESULT, indent=1, default=str))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"ok": True, "rows": RESULT["manifest"]["rows"], "train": RESULT["manifest"]["train"],
                                  "val": RESULT["manifest"]["val"], "val_seqs": RESULT["val_seqs"],
                                  "manifest_sha256": RESULT["manifest_sha256"],
                                  "seconds_total": RESULT["seconds_total"]}))
