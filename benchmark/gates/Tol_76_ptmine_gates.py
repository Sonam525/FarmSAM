# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N76 — b1.6 / b1.7 CPU gates of the PigTrack hard-mining experiment (Paper 1, jobs **J16** and **J17**)
# MAGIC
# MAGIC Two read-only checks that fail the task (and therefore stop the job, every later task depends on them with `ALL_SUCCESS`):
# MAGIC
# MAGIC | `mode` | what it checks | used by |
# MAGIC |---|---|---|
# MAGIC | `pin_check` | the two new students' checkpoint files `distillation_runs/<run>/final_<rid>/best_backbone_student_pigtrack.pt` exist, hash to the pinned sha256 values, and agree with the run's own `train_done.json` (`final_dir`, `ckpt`, `ckpt_sha256`) and `DONE.json` (if present) | first task of J16 and J17 |
# MAGIC | `replay_compare` | **the S-PT replay gate** (BENCH_PROTOCOL b1.6 item 7 / DESIGN s.6.3 t1): the three MOT files of the S-PT replay (variant `ptmine_repro`, sequences pigtrack0038, pigtrack0055, pigtrack0011 = the three shortest test sequences, 90 / 98 / 123 frames) must be BYTE-identical (sha256) to the J7 files of the S-PT main run directory AND to the sha256 values pinned in this notebook (taken from the Volume 2026-10-24); the replay's `BENCH_CONFIG.json` must show that it was made by the NEW core and the NEW Tol_63 (their sha256 values are widgets), as S-PT, Protocol A, split test, full length, on an A10 | J17, after the replay task and before any new-arm tracking |
# MAGIC
# MAGIC No frame, no model, no GPU, no write: it reads `mot/<seq>.txt`, `.done_<seq>`, `BENCH_CONFIG.json`, `train_done.json`, `DONE.json` and the
# MAGIC checkpoint bytes, resolves every location through the registry (`tol_paths`, pinned sha256), and returns a JSON report as the notebook's
# MAGIC exit value (`databricks jobs get-run-output`). A failed gate raises `RuntimeError("... GATE FAILED ...")` after printing the report.
# MAGIC
# MAGIC | widget | meaning |
# MAGIC |---|---|
# MAGIC | `mode` | `pin_check` \| `replay_compare` |
# MAGIC | `pins` | `pin_check`: JSON list `[{"arm": "S-PT-mined", "run": "pigtrack_mined_v1", "final_dir": "final_<6 hex>", "sha256": "<64 hex>"}, {...S-PT-rand...}]` (both arms, nothing else) |
# MAGIC | `core_sha256`, `notebook_sha256` | `replay_compare`: the sha256 of the `tol_bench_core.py` and of the `Tol_63_track_bench` source that must have produced the replay |
# MAGIC | `variant` | `replay_compare`: the replay's variant directory (default `ptmine_repro`) |
# MAGIC
# MAGIC **Local check** (no Databricks): `TOL76_SELFTEST=1 TOL76_PATHS_DIR=<dir of tol_paths.py> TOL76_REGISTRY_YAML=<paths_v1.yaml> python Tol_76_ptmine_gates.py`.

# COMMAND ----------

import hashlib
import json
import os
import os.path as osp
import re
import sys
import tempfile
import types

NB_NAME = "Tol_76_ptmine_gates"
NB_VERSION = "tol76-2026-10-24.1"
SELFTEST = os.environ.get("TOL76_SELFTEST") == "1"
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None
if not IS_NOTEBOOK and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _code_dir():
    env = os.environ.get("TOL76_PATHS_DIR")
    if env:
        return env
    if not IS_NOTEBOOK:
        return osp.dirname(osp.abspath(__file__))
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    return "/Workspace" + osp.dirname(nbp)


CODE_DIR = _code_dir()
TOL_PATHS_SHA256 = "f6603e639ec0e010640da401afc3837ae2e3923bf9c9dd62ff3e55bc9d9d3d46"   # registry loader (as Tol_02c / Tol_63)


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_tol_paths():
    sys.path.insert(0, CODE_DIR)
    sys.modules.pop("tol_paths", None)
    import tol_paths as tp
    got = _sha256_file(tp.__file__)
    if got != TOL_PATHS_SHA256:
        raise RuntimeError(f"tol_paths at {tp.__file__} is {got}, not the pinned {TOL_PATHS_SHA256}")
    return tp


# COMMAND ----------

# MAGIC %md
# MAGIC ## Constants (pinned) and the pure functions (exercised by the selftest)

# COMMAND ----------

REGISTRY_VERSION = 1
SHA_RX = re.compile(r"^[0-9a-f]{64}$")
FINAL_RX = re.compile(r"^final_[0-9a-f]{6}$")
CKPT_NAME = "best_backbone_student_pigtrack.pt"
ARM_RUN = {"S-PT-mined": "pigtrack_mined_v1", "S-PT-rand": "pigtrack_random_v1"}     # b1.7: exactly these two students
# The S-PT replay gate. Sequences = the three shortest PigTrack test sequences (frames: 90, 98, 123; J7 telemetry source.n_frames).
REPLAY = {"dataset": "pigtrack", "arm": "S-PT", "protocol": "A", "split": "test", "variant": "ptmine_repro",
          "seqs": ("pigtrack0038", "pigtrack0055", "pigtrack0011")}
# sha256 of the J7 MOT files <bench>/pigtrack/S-PT__protA/test/mot/<seq>.txt, read from the Volume on 2026-10-24 (two independent
# reads: `fs cp` and `fs cat`, equal; 67,184 / 36,744 / 50,316 bytes).
J7_SPT_MOT_SHA256 = {"pigtrack0038": "5b4d8479097abb69b0442d1dc26ce1234c6e755dfb7e0577fee874419cb6fd68",
                     "pigtrack0055": "fe13c2c738e0ab2aace8fbde737297824e1aea2f2a04cf65148188c702fb4b04",
                     "pigtrack0011": "f2148509d8bb8b9fff855ea1d5269fb65b587673239a90d3e2c912aebe4288ca"}
J7_SPT_MOT_BYTES = {"pigtrack0038": 67184, "pigtrack0055": 36744, "pigtrack0011": 50316}
WIDGETS = {"mode": "", "pins": "", "core_sha256": "", "notebook_sha256": "", "variant": REPLAY["variant"]}


def _read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def parse_params(raw):
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}; known: {list(WIDGETS)}")
    g = {k: str(raw.get(k) if raw.get(k) is not None else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace(mode=g["mode"], variant=g["variant"] or REPLAY["variant"])
    if P.mode not in ("pin_check", "replay_compare"):
        raise ValueError(f"mode={g['mode']!r}: expected pin_check | replay_compare")
    if P.mode == "pin_check":
        if not g["pins"]:
            raise ValueError("pin_check needs the pins widget (a JSON list)")
        try:
            P.pins = json.loads(g["pins"])
        except ValueError as e:
            raise ValueError(f"pins is not valid JSON: {e}") from None
        if g["core_sha256"] or g["notebook_sha256"]:
            raise ValueError("core_sha256 / notebook_sha256 are replay_compare parameters")
    else:
        for k in ("core_sha256", "notebook_sha256"):
            if not SHA_RX.match(g[k]):
                raise ValueError(f"replay_compare needs {k} (64 lowercase hex): a gate that does not check what produced the replay "
                                 f"proves nothing about the new code")
        if g["pins"]:
            raise ValueError("pins is a pin_check parameter")
        P.core_sha256, P.notebook_sha256 = g["core_sha256"], g["notebook_sha256"]
        if P.variant != REPLAY["variant"]:
            raise ValueError(f"variant={P.variant!r}: the replay gate registered the variant {REPLAY['variant']!r}")
    return P


def _validate_pins(pins):
    if not isinstance(pins, list) or len(pins) != len(ARM_RUN):
        raise ValueError(f"pins must be a list of exactly {len(ARM_RUN)} objects (one per arm {sorted(ARM_RUN)})")
    seen = set()
    for p in pins:
        if not isinstance(p, dict) or set(p) != {"arm", "run", "final_dir", "sha256"}:
            raise ValueError(f"pin {p!r}: keys must be exactly arm, run, final_dir, sha256")
        if p["arm"] not in ARM_RUN or ARM_RUN[p["arm"]] != p["run"]:
            raise ValueError(f"pin {p!r}: arm / run must be one of {ARM_RUN}")
        if p["arm"] in seen:
            raise ValueError(f"pin for {p['arm']} given twice")
        seen.add(p["arm"])
        if not FINAL_RX.match(str(p["final_dir"])):
            raise ValueError(f"pin {p['arm']}: final_dir {p['final_dir']!r} must be final_<6 hex>")
        if not SHA_RX.match(str(p["sha256"])):
            raise ValueError(f"pin {p['arm']}: sha256 must be 64 lowercase hex")
    shas = [p["sha256"] for p in pins]
    if len(set(shas)) != len(shas):
        raise ValueError("the two pinned sha256 values are equal (two different students cannot share a checkpoint)")
    return pins


def pin_check(runs_root, pins, log=print):
    """runs_root = the registry path of paper1.distillation_runs. Returns the report; raises RuntimeError if any check fails."""
    pins = _validate_pins(pins)
    rep = {"gate": "pin_check", "runs_root": runs_root, "arms": {}, "problems": []}
    for p in pins:
        arm, run, fdir, want = p["arm"], p["run"], p["final_dir"], p["sha256"]
        rd = f"{runs_root}/{run}"
        a = rep["arms"][arm] = {"run": run, "final_dir": fdir, "pinned_sha256": want}

        def bad(msg, arm=arm):
            rep["problems"].append(f"{arm}: {msg}")
        tdp = f"{rd}/train_done.json"
        if not osp.isfile(tdp):
            bad(f"{tdp} is missing (the run did not finish training)")
        else:
            td = _read_json(tdp)
            a["train_done"] = {k: td.get(k) for k in ("final_dir", "ckpt", "ckpt_sha256", "rid", "best_epoch", "epochs_run", "stopped")}
            if td.get("final_dir") != fdir:
                bad(f"train_done.json final_dir {td.get('final_dir')!r} != pinned {fdir!r}")
            if td.get("ckpt") != CKPT_NAME:
                bad(f"train_done.json ckpt {td.get('ckpt')!r} != {CKPT_NAME!r}")
            if td.get("ckpt_sha256") != want:
                bad(f"train_done.json ckpt_sha256 {str(td.get('ckpt_sha256'))[:16]} != pinned {want[:16]}")
        dp = f"{rd}/DONE.json"
        if osp.isfile(dp):
            dn = _read_json(dp)
            a["done_ckpt_sha256"] = dn.get("ckpt_sha256")
            if dn.get("ckpt_sha256") != want:
                bad(f"DONE.json ckpt_sha256 {str(dn.get('ckpt_sha256'))[:16]} != pinned {want[:16]}")
        f = f"{rd}/{fdir}/{CKPT_NAME}"
        if not osp.isfile(f):
            bad(f"{f} is missing")
        else:
            a["bytes"] = osp.getsize(f)
            a["sha256"] = _sha256_file(f)
            if a["sha256"] != want:
                bad(f"the checkpoint file hashes to {a['sha256'][:16]}, not the pinned {want[:16]}")
        if osp.isdir(rd):
            a["final_dirs_present"] = sorted(x for x in os.listdir(rd) if x.startswith("final_"))
        log(f"pin_check {arm}: {run}/{fdir} sha256 {str(a.get('sha256'))[:16]} pinned {want[:16]} "
            f"{'OK' if not [x for x in rep['problems'] if x.startswith(arm + ':')] else 'PROBLEM'}")
    rep["ok"] = not rep["problems"]
    if not rep["ok"]:
        print(json.dumps(rep, indent=1, sort_keys=True))
        raise RuntimeError("PIN CHECK GATE FAILED: " + "; ".join(rep["problems"]))
    return rep


def replay_compare(main_dir, replay_dir, core_sha256, notebook_sha256, log=print):
    """main_dir = the S-PT main run dir (J7); replay_dir = its ptmine_repro variant dir. Returns the report; raises RuntimeError on any failure."""
    rep = {"gate": "replay_compare", "main_dir": main_dir, "replay_dir": replay_dir, "seqs": {}, "problems": [], "code": {}}

    def bad(msg):
        rep["problems"].append(msg)
    cp = f"{replay_dir}/BENCH_CONFIG.json"
    if not osp.isfile(cp):
        bad(f"{cp} is missing (the replay task did not run)")
    else:
        cfg = _read_json(cp)
        arm = (cfg.get("arm") or {}).get("arm") if isinstance(cfg.get("arm"), dict) else None
        want = {"dataset": REPLAY["dataset"], "split": REPLAY["split"], "protocol": REPLAY["protocol"], "variant": REPLAY["variant"]}
        for k, v in want.items():
            if cfg.get(k) != v:
                bad(f"replay BENCH_CONFIG {k}={cfg.get(k)!r}, expected {v!r}")
        if arm != REPLAY["arm"]:
            bad(f"replay BENCH_CONFIG arm={arm!r}, expected {REPLAY['arm']!r}")
        if (cfg.get("run") or {}).get("smoke_max_frames") is not None:
            bad(f"replay BENCH_CONFIG smoke_max_frames={(cfg.get('run') or {}).get('smoke_max_frames')!r}: the replay must be full length")
        gpu = (cfg.get("gpu") or {}).get("class")
        if gpu != "A10":
            bad(f"replay GPU class {gpu!r}, expected 'A10' (bit identity is a same-GPU-class property)")
        code = cfg.get("code") or {}
        got_core = ((code.get("tol_bench_core") or {}).get("source"))
        got_nb = ((code.get("notebook") or {}).get("source"))
        rep["code"] = {"replay_tol_bench_core": got_core, "replay_notebook": got_nb, "expected_core": core_sha256,
                       "expected_notebook": notebook_sha256}
        if got_core != core_sha256:
            bad(f"the replay ran tol_bench_core {str(got_core)[:16]}, not the new core {core_sha256[:16]}")
        if got_nb != notebook_sha256:
            bad(f"the replay ran notebook {str(got_nb)[:16]}, not the new Tol_63 {notebook_sha256[:16]}")
        j7p = f"{main_dir}/BENCH_CONFIG.json"
        if osp.isfile(j7p):
            j7 = _read_json(j7p)
            rep["code"]["j7_tol_bench_core"] = ((j7.get("code") or {}).get("tol_bench_core") or {}).get("source")
            rep["code"]["j7_notebook"] = ((j7.get("code") or {}).get("notebook") or {}).get("source")
            vj, vr = j7.get("versions") or {}, cfg.get("versions") or {}
            rep["version_differences_info_only"] = {k: [vj.get(k), vr.get(k)] for k in sorted(set(vj) | set(vr)) if vj.get(k) != vr.get(k)}
        else:
            bad(f"{j7p} is missing")
    for seq in REPLAY["seqs"]:
        r = rep["seqs"][seq] = {"pinned_j7_sha256": J7_SPT_MOT_SHA256[seq]}
        mp, rp = f"{main_dir}/mot/{seq}.txt", f"{replay_dir}/mot/{seq}.txt"
        if not osp.isfile(f"{replay_dir}/.done_{seq}"):
            bad(f"{seq}: the replay has no .done_{seq} marker")
        for what, path in (("j7", mp), ("replay", rp)):
            if not osp.isfile(path):
                bad(f"{seq}: {path} is missing")
            else:
                r[f"{what}_bytes"] = osp.getsize(path)
                r[f"{what}_sha256"] = _sha256_file(path)
        if r.get("j7_sha256") is not None and r["j7_sha256"] != J7_SPT_MOT_SHA256[seq]:
            bad(f"{seq}: the J7 file on the Volume hashes to {r['j7_sha256'][:16]}, not the pinned {J7_SPT_MOT_SHA256[seq][:16]} "
                f"(the reference itself changed)")
        if r.get("j7_bytes") is not None and r["j7_bytes"] != J7_SPT_MOT_BYTES[seq]:
            bad(f"{seq}: the J7 file has {r['j7_bytes']} bytes, expected {J7_SPT_MOT_BYTES[seq]}")
        r["identical"] = bool(r.get("replay_sha256") and r.get("replay_sha256") == r.get("j7_sha256") == J7_SPT_MOT_SHA256[seq])
        if r.get("replay_sha256") is not None and r["replay_sha256"] != J7_SPT_MOT_SHA256[seq]:
            bad(f"{seq}: the replay MOT file hashes to {r['replay_sha256'][:16]}, J7's is {J7_SPT_MOT_SHA256[seq][:16]}: NOT byte-identical")
        log(f"replay {seq}: replay {str(r.get('replay_sha256'))[:16]} J7 {str(r.get('j7_sha256'))[:16]} "
            f"{'IDENTICAL' if r['identical'] else 'DIFFERENT'}")
    rep["ok"] = not rep["problems"] and all(r["identical"] for r in rep["seqs"].values())
    if not rep["ok"]:
        print(json.dumps(rep, indent=1, sort_keys=True))
        raise RuntimeError("REPLAY GATE FAILED (STOP before any new-arm tracking is scored; report to the controller): "
                           + "; ".join(rep["problems"]))
    return rep


def main(raw, *, R=None, tp=None):
    P = parse_params(raw)
    if R is None:
        tp = tp or _import_tol_paths()
        R = tp.load(version=REGISTRY_VERSION)
    if P.mode == "pin_check":
        rep = pin_check(R.path("paper1.distillation_runs"), P.pins)
    else:
        bench = R.path("paper1.bench")
        main_dir = f"{bench}/{REPLAY['dataset']}/{REPLAY['arm']}__prot{REPLAY['protocol']}/{REPLAY['split']}"
        rep = replay_compare(main_dir, f"{main_dir}/{P.variant}", P.core_sha256, P.notebook_sha256)
    rep["notebook"] = f"{NB_NAME} {NB_VERSION}"
    return rep


# COMMAND ----------

# MAGIC %md
# MAGIC ## Local selftest (`TOL76_SELFTEST=1`; temporary registry root, synthetic files; never on Databricks)

# COMMAND ----------


def selftest():
    res = []

    def check(name, cond, detail=""):
        res.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))

    def raises(fn, exc=Exception, text=""):
        try:
            fn()
        except exc as e:
            return text in str(e)
        return False
    quiet = lambda *a: None  # noqa: E731
    CORE, NB = "c" * 64, "d" * 64

    def w(path, data, mode="wb"):
        os.makedirs(osp.dirname(path), exist_ok=True)
        with open(path, mode) as f:
            f.write(data if isinstance(data, bytes) else data.encode("utf-8"))

    def sha(b):
        return hashlib.sha256(b).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = tmp.replace("\\", "/")
        # ---------------- parameters
        check("params: unknown widget / bad mode / missing pins / replay without code shas / mixed parameters refused",
              raises(lambda: parse_params({"nope": "1"}), ValueError, "unknown")
              and raises(lambda: parse_params({"mode": "x"}), ValueError, "mode")
              and raises(lambda: parse_params({"mode": "pin_check"}), ValueError, "pins")
              and raises(lambda: parse_params({"mode": "pin_check", "pins": "[", }), ValueError, "JSON")
              and raises(lambda: parse_params({"mode": "replay_compare"}), ValueError, "core_sha256")
              and raises(lambda: parse_params({"mode": "replay_compare", "core_sha256": CORE}), ValueError, "notebook_sha256")
              and raises(lambda: parse_params({"mode": "replay_compare", "core_sha256": CORE, "notebook_sha256": "D" * 64}), ValueError, "notebook_sha256")
              and raises(lambda: parse_params({"mode": "replay_compare", "core_sha256": CORE, "notebook_sha256": NB, "pins": "[]"}), ValueError, "pin_check parameter")
              and raises(lambda: parse_params({"mode": "replay_compare", "core_sha256": CORE, "notebook_sha256": NB, "variant": "other"}), ValueError, "registered")
              and raises(lambda: parse_params({"mode": "pin_check", "pins": "[]", "core_sha256": CORE}), ValueError, "replay_compare parameters"))
        Pr = parse_params({"mode": "replay_compare", "core_sha256": CORE, "notebook_sha256": NB})
        check("params: replay_compare parses; variant defaults to ptmine_repro", Pr.variant == "ptmine_repro" and Pr.core_sha256 == CORE)
        # ---------------- pin_check
        runs = f"{tmp}/runs"
        payload = {"S-PT-mined": b"mined-checkpoint-bytes", "S-PT-rand": b"random-checkpoint-bytes"}
        fdirs = {"S-PT-mined": "final_0a1b2c", "S-PT-rand": "final_3d4e5f"}
        pins = []
        for arm, run in ARM_RUN.items():
            w(f"{runs}/{run}/{fdirs[arm]}/{CKPT_NAME}", payload[arm])
            td = {"final_dir": fdirs[arm], "ckpt": CKPT_NAME, "ckpt_sha256": sha(payload[arm]), "rid": fdirs[arm][6:], "best_epoch": 40,
                  "epochs_run": 55, "stopped": "early_stop"}
            w(f"{runs}/{run}/train_done.json", json.dumps(td))
            w(f"{runs}/{run}/DONE.json", json.dumps({"ckpt_sha256": sha(payload[arm])}))
            pins.append({"arm": arm, "run": run, "final_dir": fdirs[arm], "sha256": sha(payload[arm])})
        rep = pin_check(runs, [dict(p) for p in pins], quiet)
        check("pin_check: both checkpoints hash to their pins and agree with train_done.json / DONE.json", rep["ok"]
              and rep["arms"]["S-PT-mined"]["sha256"] == pins[0]["sha256"] and rep["arms"]["S-PT-rand"]["bytes"] == len(payload["S-PT-rand"]))

        def pc(mut=None, files=None):
            p = [dict(x) for x in pins]
            if mut:
                mut(p)
            return lambda: pin_check(runs, p, quiet)
        check("pin_check: a wrong pin / final_dir / arm-run pair / duplicate / missing arm / extra key / bad hex is refused",
              raises(pc(lambda p: p[0].update(sha256="0" * 64)), RuntimeError, "PIN CHECK GATE FAILED")
              and raises(pc(lambda p: p[0].update(final_dir="final_ffffff")), RuntimeError, "PIN CHECK GATE FAILED")
              and raises(pc(lambda p: p[0].update(run="pigtrack_random_v1")), ValueError, "arm / run")
              and raises(pc(lambda p: p[1].update(arm="S-PT-mined", run="pigtrack_mined_v1")), ValueError, "twice")
              and raises(pc(lambda p: p.pop()), ValueError, "exactly 2")
              and raises(pc(lambda p: p[0].update(extra=1)), ValueError, "keys must be")
              and raises(pc(lambda p: p[0].update(sha256="A" * 64)), ValueError, "64 lowercase hex")
              and raises(pc(lambda p: p[0].update(final_dir="final_abc")), ValueError, "final_<6 hex>")
              and raises(pc(lambda p: p[1].update(sha256=p[0]["sha256"])), ValueError, "equal"))
        w(f"{runs}/pigtrack_mined_v1/{fdirs['S-PT-mined']}/{CKPT_NAME}", b"tampered")
        check("pin_check: a checkpoint file whose bytes changed after training is refused", raises(pc(), RuntimeError, "hashes to"))
        w(f"{runs}/pigtrack_mined_v1/{fdirs['S-PT-mined']}/{CKPT_NAME}", payload["S-PT-mined"])
        td = json.loads(open(f"{runs}/pigtrack_random_v1/train_done.json").read())
        w(f"{runs}/pigtrack_random_v1/train_done.json", json.dumps(dict(td, ckpt_sha256="9" * 64)))
        check("pin_check: train_done.json recording another checkpoint is refused", raises(pc(), RuntimeError, "train_done.json ckpt_sha256"))
        w(f"{runs}/pigtrack_random_v1/train_done.json", json.dumps(td))
        w(f"{runs}/pigtrack_random_v1/DONE.json", json.dumps({"ckpt_sha256": "8" * 64}))
        check("pin_check: DONE.json recording another checkpoint is refused", raises(pc(), RuntimeError, "DONE.json ckpt_sha256"))
        w(f"{runs}/pigtrack_random_v1/DONE.json", json.dumps({"ckpt_sha256": sha(payload['S-PT-rand'])}))
        os.remove(f"{runs}/pigtrack_random_v1/train_done.json")
        check("pin_check: a run without train_done.json is refused", raises(pc(), RuntimeError, "train_done.json is missing"))
        w(f"{runs}/pigtrack_random_v1/train_done.json", json.dumps(td))
        os.remove(f"{runs}/pigtrack_random_v1/{fdirs['S-PT-rand']}/{CKPT_NAME}")
        check("pin_check: a missing checkpoint file is refused", raises(pc(), RuntimeError, "is missing"))
        w(f"{runs}/pigtrack_random_v1/{fdirs['S-PT-rand']}/{CKPT_NAME}", payload["S-PT-rand"])
        check("pin_check: after restoring everything it passes again", pc()() is not None)
        # ---------------- replay_compare
        main_dir = f"{tmp}/bench/pigtrack/S-PT__protA/test"
        rdir = f"{main_dir}/ptmine_repro"
        synth = {s: (f"1,1,10,10,20,20,1,-1,-1,-1\n{s}\n").encode() for s in REPLAY["seqs"]}
        saved_sha, saved_bytes = dict(J7_SPT_MOT_SHA256), dict(J7_SPT_MOT_BYTES)
        J7_SPT_MOT_SHA256.update({s: sha(b) for s, b in synth.items()})
        J7_SPT_MOT_BYTES.update({s: len(b) for s, b in synth.items()})
        try:
            cfgok = {"dataset": "pigtrack", "split": "test", "protocol": "A", "variant": "ptmine_repro", "arm": {"arm": "S-PT"},
                     "run": {"smoke_max_frames": None}, "gpu": {"class": "A10"}, "versions": {"torch": "2.7.0"},
                     "code": {"tol_bench_core": {"source": CORE}, "notebook": {"source": NB}}}
            w(f"{main_dir}/BENCH_CONFIG.json", json.dumps({"code": {"tol_bench_core": {"source": "b" * 64}, "notebook": {"source": "e" * 64}},
                                                           "versions": {"torch": "2.7.0"}}))
            for s, b in synth.items():
                w(f"{main_dir}/mot/{s}.txt", b)
                w(f"{rdir}/mot/{s}.txt", b)
                w(f"{rdir}/.done_{s}", b"")
            w(f"{rdir}/BENCH_CONFIG.json", json.dumps(cfgok))

            def rc(core=CORE, nb=NB):
                return lambda: replay_compare(main_dir, rdir, core, nb, quiet)
            rep = rc()()
            check("replay: three byte-identical MOT files + a BENCH_CONFIG made by the new core / Tol_63 -> pass",
                  rep["ok"] and all(r["identical"] for r in rep["seqs"].values()) and rep["code"]["j7_tol_bench_core"] == "b" * 64)
            w(f"{rdir}/mot/pigtrack0055.txt", synth["pigtrack0055"] + b"x")
            check("replay: ONE byte more in one replay file -> STOP (RuntimeError names the sequence)",
                  raises(rc(), RuntimeError, "pigtrack0055") and raises(rc(), RuntimeError, "REPLAY GATE FAILED"))
            w(f"{rdir}/mot/pigtrack0055.txt", synth["pigtrack0055"])
            w(f"{main_dir}/mot/pigtrack0011.txt", synth["pigtrack0011"] + b"\n")
            check("replay: the J7 reference itself changed (!= the pinned hash) -> STOP", raises(rc(), RuntimeError, "the reference itself changed"))
            w(f"{main_dir}/mot/pigtrack0011.txt", synth["pigtrack0011"])
            os.remove(f"{rdir}/mot/pigtrack0038.txt")
            check("replay: a missing replay MOT file -> STOP", raises(rc(), RuntimeError, "pigtrack0038"))
            w(f"{rdir}/mot/pigtrack0038.txt", synth["pigtrack0038"])
            os.remove(f"{rdir}/.done_pigtrack0011")
            check("replay: a missing .done marker -> STOP", raises(rc(), RuntimeError, ".done_pigtrack0011"))
            w(f"{rdir}/.done_pigtrack0011", b"")
            check("replay: the replay was NOT made by the new core / the new Tol_63 -> STOP",
                  raises(rc(core="e" * 64), RuntimeError, "not the new core") and raises(rc(nb="e" * 64), RuntimeError, "not the new Tol_63"))
            for label, mut, text in (("arm", {"arm": {"arm": "S-PT-mined"}}, "arm="), ("variant", {"variant": "x"}, "variant="),
                                     ("split", {"split": "val"}, "split="), ("smoke", {"run": {"smoke_max_frames": 30}}, "full length"),
                                     ("gpu", {"gpu": {"class": "A100"}}, "GPU class")):
                w(f"{rdir}/BENCH_CONFIG.json", json.dumps(dict(cfgok, **mut)))
                check(f"replay: BENCH_CONFIG with another {label} -> STOP", raises(rc(), RuntimeError, text))
            os.remove(f"{rdir}/BENCH_CONFIG.json")
            check("replay: no BENCH_CONFIG (the replay never ran) -> STOP", raises(rc(), RuntimeError, "BENCH_CONFIG.json is missing"))
            w(f"{rdir}/BENCH_CONFIG.json", json.dumps(cfgok))
            # main() end to end through a registry whose volume root is the temporary directory
            tp = _import_tol_paths()
            reg_yaml = os.environ.get("TOL76_REGISTRY_YAML") or osp.join(osp.dirname(tp.__file__), "paths_v1.yaml")
            R0 = tp.load(path=reg_yaml)
            data = json.loads(json.dumps(R0.data))
            data["roots"].update(volume=tmp + "/vol", mount=tmp + "/mnt", mount_uri="file:" + tmp + "/mnt")
            R = tp.Registry(data, source=reg_yaml, sha256=R0.sha256)
            bench_root = R.path("paper1.bench")
            runs_root = R.path("paper1.distillation_runs")
            import shutil
            shutil.copytree(f"{tmp}/bench/pigtrack", f"{bench_root}/pigtrack")
            shutil.copytree(runs, runs_root)
            out = main({"mode": "replay_compare", "core_sha256": CORE, "notebook_sha256": NB}, R=R)
            check("main: replay_compare resolves the S-PT main / variant directories through the registry", out["ok"]
                  and out["replay_dir"].endswith("/paper1_distillation/bench/pigtrack/S-PT__protA/test/ptmine_repro"))
            out = main({"mode": "pin_check", "pins": json.dumps(pins)}, R=R)
            check("main: pin_check resolves paper1.distillation_runs through the registry", out["ok"]
                  and out["runs_root"].endswith("/paper1_distillation/distillation_runs"))
        finally:
            J7_SPT_MOT_SHA256.update(saved_sha)
            J7_SPT_MOT_BYTES.update(saved_bytes)
    # ---------------- audits of this notebook's own source
    src = open(osp.abspath(__file__), encoding="utf-8").read()
    check("audit: the pinned J7 hashes are the Volume values of 2026-10-24",
          J7_SPT_MOT_SHA256 == {"pigtrack0038": "5b4d8479097abb69b0442d1dc26ce1234c6e755dfb7e0577fee874419cb6fd68",
                                "pigtrack0055": "fe13c2c738e0ab2aace8fbde737297824e1aea2f2a04cf65148188c702fb4b04",
                                "pigtrack0011": "f2148509d8bb8b9fff855ea1d5269fb65b587673239a90d3e2c912aebe4288ca"})
    check("audit: no literal Volume / dbfs / Workspace path (registry only)", not re.search(r"/Volumes/|dbfs:/|/Workspace/Users", src.split("def selftest")[0]))
    check("audit: no secret-shaped strings, no write calls on the Volume", not re.search(r"hf_[A-Za-z0-9]{20}|secrets\.get|dbutils\.fs\.(cp|rm|put|mv)|\bshutil\.(copy|move|rmtree)", src.split("def selftest")[0]))
    check("audit: Databricks SOURCE format", src.startswith("# Databricks notebook source"))
    n_fail = sum(1 for r in res if not r)
    print(f"\n{NB_NAME} selftest: {len(res) - n_fail}/{len(res)} passed")
    return n_fail


if SELFTEST:
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
            out[name] = os.environ.get("TOL76_" + name.upper(), default)
    return out


RAW = read_widgets()
print(NB_NAME, NB_VERSION, "widgets:", json.dumps({k: (v if k != "pins" else v[:200]) for k, v in RAW.items()}, sort_keys=True))
REPORT = main(RAW)
print(json.dumps(REPORT, indent=1, sort_keys=True))

# COMMAND ----------

if IS_NOTEBOOK:
    dbutils.notebook.exit(json.dumps(REPORT, sort_keys=True))
