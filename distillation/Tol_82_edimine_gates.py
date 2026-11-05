# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N82 — b1.11 CPU gates of the Edinburgh hard-case mining experiment (S-Edi-mined; job EM6 = DESIGN_EDI_v2 "EDI4")
# MAGIC
# MAGIC Successor of `Tol_77_ptmine_gates` (the PigTrack gates; tol77-2026-10-25.1, unchanged): ONE student instead of three, plus the registered gate **G-M**, plus the
# MAGIC **identity gate** against the J4e score `eb9f09ff`. Three read-only modes; a failed gate raises `RuntimeError("... GATE FAILED ...")` after printing the report, and
# MAGIC every later task of the job depends on the previous one with `ALL_SUCCESS`, so a failed gate STOPS the job.
# MAGIC
# MAGIC | `mode` | what it checks | task |
# MAGIC |---|---|---|
# MAGIC | `pin_check` | **before any test clip is opened:** the checkpoint file `distillation_runs/edinburgh_mined_v1/final_<rid>/best_backbone_student_edinburgh.pt` exists and hashes to the pinned sha256; the pin equals `train_done.json` (`final_dir`, `ckpt`, `ckpt_sha256`) AND `DONE.json` (the fidelity job's last write; required unless `require_done=false`; whenever present it must also say `arm` = S-Edi-mined, `ckpt_pin_checked` = true and a passing `positive_control_status`, tol82-2026-11-02.1) AND the `S-Edi-mined` ARMS entry of the `tol_bench_core.py` in `TOL63_CODE_DIR` (whose own sha256 must equal `core_sha256`); **gate G-M** on `train_done.json`'s `best_ema_val_loss` (finite and within ±10 % of S-Edi's recorded 0.25041554232438407; fail closed); `tol_paths.py` and the registry file by sha256 | `t00_pin_check_gm` |
# MAGIC | `replay_compare` | **the S-Edi replay gate** (BENCH_PROTOCOL b1.11 item 5): S-Edi re-tracked on ONE test clip (2019_12_02_000005) under the NEW core (variant `edimine_repro`): its MOT bytes must equal the MOT bytes of the published J4 run on the Volume; the replay's `BENCH_CONFIG.json` must name the new core and the new Tol_63 (sha256 widgets), arm S-Edi, Protocol A, split test, full length, A10 | `t02_replay_compare` |
# MAGIC | `identity_gate` | **after the one `Tol_64` call:** the `results` block (`combined`, `strata`, `per_seq`, `xcheck`) of each of the six existing arms T, S-Edi, T21, S21, ET, ES-L in the NEW score equals the J4e score `eb9f09ff` field by field (0 differences), plus the inputs that make the comparison meaningful (sequences, GT sha256 per clip, TrackEval lengths and settings, xy offset, the per-arm run configs). **It reads the new score only to find and compare the six existing arms; it never prints or returns a value of `S-Edi-mined`.** `edinburgh_clip_deltas` is NOT compared (it depends on `ref_arm`). Since tol82-2026-11-02.1 the WHOLE `results.<arm>` object is compared (a key added outside the four blocks is a difference) and the raw per-sequence results (`per_seq_raw.json`, `results.<arm>` of the six arms) are compared too and GATE: both files must exist and be identical | `t05_identity_gate` |
# MAGIC
# MAGIC No frame, no model, no GPU, no write: it reads JSON files, MOT files and checkpoint bytes, resolves every location through the registry (`tol_paths`, pinned sha256), and
# MAGIC returns a JSON report as the notebook's exit value (`databricks jobs get-run-output`).
# MAGIC
# MAGIC | widget | meaning |
# MAGIC |---|---|
# MAGIC | `mode` | `pin_check` \| `replay_compare` \| `identity_gate` |
# MAGIC | `pin` | `pin_check`: JSON object `{"arm": "S-Edi-mined", "run": "edinburgh_mined_v1", "final_dir": "final_<6 hex>", "sha256": "<64 hex>"}` |
# MAGIC | `core_sha256` | `pin_check` and `replay_compare`: sha256 of the final `tol_bench_core.py` |
# MAGIC | `notebook_sha256` | `replay_compare`: sha256 of the `Tol_63_track_bench` source that must have produced the replay |
# MAGIC | `variant` | `replay_compare`: the replay's variant directory (default `edimine_repro`) |
# MAGIC | `require_done` | `pin_check`: `true` (default) = `DONE.json` of the run must exist and agree (the fidelity job ran first) |
# MAGIC | `ref_score_sid8`, `new_score_dir` | `identity_gate`: the reference score id suffix (default `eb9f09ff`); optional name of the new score directory when more than one candidate exists |
# MAGIC
# MAGIC **Local check** (no Databricks): `TOL82_SELFTEST=1 TOL82_PATHS_DIR=<dir of tol_paths.py> TOL82_REGISTRY_YAML=<paths_v1.yaml> python Tol_82_edimine_gates.py`.

# COMMAND ----------

import hashlib
import json
import math
import os
import os.path as osp
import re
import sys
import tempfile
import types

NB_NAME = "Tol_82_edimine_gates"
NB_VERSION = "tol82-2026-11-02.1"
SELFTEST = os.environ.get("TOL82_SELFTEST") == "1"
try:
    dbutils  # noqa: F821,B018
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
    dbutils = None
if not IS_NOTEBOOK and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _code_dir():
    env = os.environ.get("TOL82_PATHS_DIR")
    if env:
        return env
    if not IS_NOTEBOOK:
        return osp.dirname(osp.abspath(__file__))
    nbp = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    return "/Workspace" + osp.dirname(nbp)


CODE_DIR = _code_dir()
TOL_PATHS_SHA256 = "f6603e639ec0e010640da401afc3837ae2e3923bf9c9dd62ff3e55bc9d9d3d46"   # registry loader (as Tol_02c / Tol_63 / Tol_77)


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
# MAGIC ## PINS (one block of plain literals; `submit_edi_guarded.py` compares its own G-M constants with this block before it submits) and the pure functions

# COMMAND ----------

PINS = {
    "SEDI_BEST_EMA_VAL_LOSS": 0.25041554232438407,      # S-Edi's recorded ema_val_loss (EMA epoch 75 of 90; the history file of the published student)
    "GM_REL_TOL": 0.1,                                  # gate G-M: abs(v - SEDI_BEST_EMA_VAL_LOSS) <= GM_REL_TOL * SEDI_BEST_EMA_VAL_LOSS, v finite
    "ARM": "S-Edi-mined",
    "RUN": "edinburgh_mined_v1",
    "CKPT_NAME": "best_backbone_student_edinburgh.pt",
    "SCOPE": "ablation (b1.11)",
    "REPLAY_CLIP": "2019_12_02_000005",
    "REPLAY_VARIANT": "edimine_repro",
    "REFERENCE_SCORE_SID8": "eb9f09ff",                 # the J4e score (interp GT, test, Protocol A, ref_arm T); canonical for the six existing arms
    "EXISTING_ARMS": ["T", "S-Edi", "T21", "S21", "ET", "ES-L"],
    "REGISTRY_SHA256": "f882e7ac08f6a2fb64063628955249bb000f507cd0c49f92e31b1223056d7dbb",    # _admin/registry/paths_v1.yaml (b1.9 item 4; immutable by design)
}
REGISTRY_VERSION = 1
SHA_RX = re.compile(r"^[0-9a-f]{64}$")
FINAL_RX = re.compile(r"^final_[0-9a-f]{6}$")
ARM, RUN, CKPT_NAME = PINS["ARM"], PINS["RUN"], PINS["CKPT_NAME"]
EXISTING = tuple(PINS["EXISTING_ARMS"])
REPLAY = {"dataset": "edinburgh", "arm": "S-Edi", "protocol": "A", "split": "test", "variant": PINS["REPLAY_VARIANT"], "clip": PINS["REPLAY_CLIP"]}
WIDGETS = {"mode": "", "pin": "", "core_sha256": "", "notebook_sha256": "", "variant": REPLAY["variant"], "require_done": "true",
           "ref_score_sid8": PINS["REFERENCE_SCORE_SID8"], "new_score_dir": ""}
# fields compared by the identity gate (BENCH_PROTOCOL b1.11 item 5 + the inputs that make the comparison meaningful)
TOP_EQUAL = ("bench_version", "protocol_doc", "score_version", "dataset", "split", "protocol", "variant", "gt_variant", "smoke", "statistics", "xy_offset",
             "trackeval_metrics")
INPUT_EQUAL = ("gt", "seq_meta", "sequence_source", "sequences", "skipped_arms", "strata", "trackeval_lengths")
RESULT_KEYS = ("combined", "strata", "per_seq", "xcheck")      # what Tol_64 s1.2 writes under results.<arm>; since tol82-2026-11-02.1 the WHOLE object is compared, not only these
DONE_POSITIVE_CONTROL_PASS = ("identical", "within_4dp")        # = Tol_02d FIDE_PASS (the selftest compares them when Tol_02d is next to this file)


def _read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def gm_check(v):
    """Gate G-M (BENCH_PROTOCOL b1.11 item 4): v finite, a real number (not a bool, not a string), and abs(v - S) <= 0.1 * S for S = S-Edi's 0.25041554232438407.
    -> (ok, reason). The test is exactly `math.isfinite(v) and abs(v - S) <= REL * S` (the rounded band [0.2254, 0.2755] is NOT the test)."""
    s, rel = PINS["SEDI_BEST_EMA_VAL_LOSS"], PINS["GM_REL_TOL"]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False, f"best_ema_val_loss {v!r} is not a number"
    if not math.isfinite(v):
        return False, f"best_ema_val_loss {v!r} is not finite"
    if abs(v - s) <= rel * s:
        return True, "within the band"
    return False, f"best_ema_val_loss {v!r} is outside [{s - rel * s!r}, {s + rel * s!r}]"


def gm_from_train_done(train_done_path):
    """-> (ok, reason, value|None); a missing file, a missing key, malformed JSON all fail closed."""
    if not osp.isfile(train_done_path):
        return False, f"{train_done_path} is missing", None
    try:
        td = _read_json(train_done_path)
    except (ValueError, OSError) as e:
        return False, f"{train_done_path} is not readable JSON ({type(e).__name__})", None
    if not isinstance(td, dict) or "best_ema_val_loss" not in td:
        return False, "train_done.json has no best_ema_val_loss", None
    ok, why = gm_check(td["best_ema_val_loss"])
    return ok, why, td["best_ema_val_loss"]


def parse_params(raw):
    unknown = sorted(set(raw) - set(WIDGETS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}; known: {list(WIDGETS)}")
    g = {k: str(raw.get(k) if raw.get(k) is not None else d).strip() for k, d in WIDGETS.items()}
    P = types.SimpleNamespace(mode=g["mode"])
    if P.mode not in ("pin_check", "replay_compare", "identity_gate"):
        raise ValueError(f"mode={g['mode']!r}: expected pin_check | replay_compare | identity_gate")

    def forbid(*names):
        for k in names:
            if g[k] and g[k] != WIDGETS[k]:
                raise ValueError(f"{k} is not a {P.mode} parameter")
    if P.mode == "pin_check":
        if not g["pin"]:
            raise ValueError("pin_check needs the pin widget (a JSON object)")
        try:
            P.pin = json.loads(g["pin"])
        except ValueError as e:
            raise ValueError(f"pin is not valid JSON: {e}") from None
        if not SHA_RX.match(g["core_sha256"]):
            raise ValueError("pin_check needs core_sha256 (64 lowercase hex): the ARMS pin is read from the core that will track")
        if g["require_done"].lower() not in ("true", "false"):
            raise ValueError(f"require_done={g['require_done']!r}: expected true | false")
        P.core_sha256, P.require_done = g["core_sha256"], g["require_done"].lower() == "true"
        forbid("notebook_sha256", "variant", "ref_score_sid8", "new_score_dir")
    elif P.mode == "replay_compare":
        for k in ("core_sha256", "notebook_sha256"):
            if not SHA_RX.match(g[k]):
                raise ValueError(f"replay_compare needs {k} (64 lowercase hex): a gate that does not check what produced the replay proves nothing about the new code")
        P.core_sha256, P.notebook_sha256, P.variant = g["core_sha256"], g["notebook_sha256"], g["variant"] or REPLAY["variant"]
        if P.variant != REPLAY["variant"]:
            raise ValueError(f"variant={P.variant!r}: the replay gate registered the variant {REPLAY['variant']!r}")
        forbid("pin", "ref_score_sid8", "new_score_dir")
        if g["require_done"].lower() != "true":
            raise ValueError("require_done is a pin_check parameter")
    else:
        P.ref_sid8, P.new_score_dir = g["ref_score_sid8"], g["new_score_dir"]
        if not re.fullmatch(r"[0-9a-f]{8}", P.ref_sid8):
            raise ValueError(f"ref_score_sid8={P.ref_sid8!r}: expected 8 lowercase hex")
        if P.new_score_dir and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", P.new_score_dir):
            raise ValueError(f"new_score_dir={P.new_score_dir!r}: one directory name")
        forbid("pin", "core_sha256", "notebook_sha256", "variant")
        if g["require_done"].lower() != "true":
            raise ValueError("require_done is a pin_check parameter")
    return P


def _validate_pin(pin):
    if not isinstance(pin, dict) or set(pin) != {"arm", "run", "final_dir", "sha256"}:
        raise ValueError(f"pin {pin!r}: keys must be exactly arm, run, final_dir, sha256")
    if pin["arm"] != ARM or pin["run"] != RUN:
        raise ValueError(f"pin {pin!r}: arm / run must be {ARM!r} / {RUN!r}")
    if not FINAL_RX.match(str(pin["final_dir"])):
        raise ValueError(f"pin: final_dir {pin['final_dir']!r} must be final_<6 hex>")
    if not SHA_RX.match(str(pin["sha256"])):
        raise ValueError("pin: sha256 must be 64 lowercase hex")
    return pin


def arms_pin_from_core(core_path):
    """The ARMS entry of S-Edi-mined read from a tol_bench_core.py by `ast` (ARMS is a pure literal): -> dict or None. Nothing is imported or executed."""
    import ast
    tree = ast.parse(open(core_path, encoding="utf-8").read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ARMS" for t in n.targets) and isinstance(n.value, ast.Dict):
            for k, v in zip(n.value.keys, n.value.values):
                if isinstance(k, ast.Constant) and k.value == ARM:
                    return ast.literal_eval(v)
    return None


def pin_check(runs_root, pin, code_dir, core_sha256, require_done=True, registry_sha256=None, log=print):
    """runs_root = the registry path of paper1.distillation_runs; code_dir = TOL63_CODE_DIR (the pinned core folder). Returns the report; raises RuntimeError if any
    check fails. G-M is evaluated whatever else fails, so the report always says whether the band was met."""
    pin = _validate_pin(pin)
    fdir, want = pin["final_dir"], pin["sha256"]
    rd = f"{runs_root}/{RUN}"
    rep = {"gate": "pin_check", "runs_root": runs_root, "run": RUN, "final_dir": fdir, "pinned_sha256": want, "problems": []}

    def bad(msg):
        rep["problems"].append(msg)
    # --- train_done.json + G-M
    tdp = f"{rd}/train_done.json"
    ok_gm, why_gm, v = gm_from_train_done(tdp)
    s = PINS["SEDI_BEST_EMA_VAL_LOSS"]
    rep["g_m"] = {"ok": ok_gm, "reason": why_gm, "best_ema_val_loss": v, "s_edi_ema_val_loss": s,
                  "relative_deviation": (abs(v - s) / s) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None,
                  "band": [s - PINS["GM_REL_TOL"] * s, s + PINS["GM_REL_TOL"] * s]}
    if not ok_gm:
        bad(f"G-M: {why_gm}")
    if osp.isfile(tdp):
        try:
            td = _read_json(tdp)
        except (ValueError, OSError):
            td = {}
        if not isinstance(td, dict):
            td = {}
        rep["train_done"] = {k: td.get(k) for k in ("final_dir", "ckpt", "ckpt_sha256", "rid", "best_epoch", "epochs_run", "stopped", "best_ema_val_loss")}
        if td.get("final_dir") != fdir:
            bad(f"train_done.json final_dir {td.get('final_dir')!r} != pinned {fdir!r}")
        if td.get("ckpt") != CKPT_NAME:
            bad(f"train_done.json ckpt {td.get('ckpt')!r} != {CKPT_NAME!r}")
        if td.get("ckpt_sha256") != want:
            bad(f"train_done.json ckpt_sha256 {str(td.get('ckpt_sha256'))[:16]} != pinned {want[:16]}")
    # --- DONE.json (written by the fidelity job, eval_stage=only)
    dp = f"{rd}/DONE.json"
    if osp.isfile(dp):
        try:
            dn = _read_json(dp)
        except (ValueError, OSError):
            dn = None
        if not isinstance(dn, dict):
            bad("DONE.json is not a readable JSON object")
            dn = {}
        rep["done_ckpt_sha256"] = dn.get("ckpt_sha256")
        if dn.get("ckpt_sha256") != want:
            bad(f"DONE.json ckpt_sha256 {str(dn.get('ckpt_sha256'))[:16]} != pinned {want[:16]}")
        # review m3 (2026-11-02): the PRESENCE of DONE.json must mean what D3 says it means - the fidelity job (eval_stage=only) re-read the pinned checkpoint and its S-Edi
        # stride-5 positive control reproduced FID-E (Tol_02d writes DONE.json only then, but a smoke or a hand-made file must not satisfy this check either)
        rep["done"] = {k: dn.get(k) for k in ("arm", "ckpt_pin_checked", "positive_control_status")}
        if dn.get("arm") != ARM:
            bad(f"DONE.json arm {dn.get('arm')!r} != {ARM!r}")
        if dn.get("ckpt_pin_checked") is not True:
            bad(f"DONE.json ckpt_pin_checked {dn.get('ckpt_pin_checked')!r} is not true (the fidelity job did not re-check the pinned checkpoint, eval_stage=only)")
        if dn.get("positive_control_status") not in DONE_POSITIVE_CONTROL_PASS:
            bad(f"DONE.json positive_control_status {dn.get('positive_control_status')!r} is not one of {list(DONE_POSITIVE_CONTROL_PASS)} (the S-Edi positive control did not reproduce FID-E)")
    elif require_done:
        bad(f"{dp} is missing (the fidelity job, eval_stage=only, writes it last; it must run before the test job)")
    # --- the checkpoint bytes
    f = f"{rd}/{fdir}/{CKPT_NAME}"
    if not osp.isfile(f):
        bad(f"{f} is missing")
    else:
        rep["bytes"] = osp.getsize(f)
        rep["sha256"] = _sha256_file(f)
        if rep["sha256"] != want:
            bad(f"the checkpoint file hashes to {rep['sha256'][:16]}, not the pinned {want[:16]}")
    if osp.isdir(rd):
        rep["final_dirs_present"] = sorted(x for x in os.listdir(rd) if x.startswith("final_"))
    # --- the core that will track: its sha256 and the ARMS pin inside it
    cp = osp.join(code_dir or "", "tol_bench_core.py")
    if not osp.isfile(cp):
        bad(f"{cp} is missing (TOL63_CODE_DIR is not the pinned core folder)")
    else:
        rep["core_sha256"] = _sha256_file(cp)
        if rep["core_sha256"] != core_sha256:
            bad(f"the tol_bench_core.py in {code_dir} hashes to {rep['core_sha256'][:16]}, not the final core {core_sha256[:16]}")
        try:
            ent = arms_pin_from_core(cp)
        except (SyntaxError, ValueError) as e:
            ent = None
            bad(f"cannot read ARMS from the core ({type(e).__name__})")
        if ent is None:
            bad(f"the core has no ARMS entry {ARM}")
        else:
            rep["arms_entry"] = {k: ent.get(k) for k in ("student_distill_file", "student_sha256", "scope", "student_key")}
            if ent.get("student_sha256") != want:
                bad(f"ARMS pin {str(ent.get('student_sha256'))[:16]} != pinned {want[:16]}")
            if ent.get("student_distill_file") != f"{RUN}/{fdir}/{CKPT_NAME}":
                bad(f"ARMS student_distill_file {ent.get('student_distill_file')!r} != {RUN}/{fdir}/{CKPT_NAME}")
            if ent.get("scope") != PINS["SCOPE"]:
                bad(f"ARMS scope {ent.get('scope')!r} != {PINS['SCOPE']!r}")
    # --- the registry file
    if registry_sha256 is not None:
        rep["registry_sha256"] = registry_sha256
        if registry_sha256 != PINS["REGISTRY_SHA256"]:
            bad(f"the registry file hashes to {registry_sha256[:16]}, not the pinned {PINS['REGISTRY_SHA256'][:16]}")
    log(f"pin_check {ARM}: {RUN}/{fdir} sha256 {str(rep.get('sha256'))[:16]} pinned {want[:16]} G-M {'PASS' if ok_gm else 'FAIL'} "
        f"{'OK' if not rep['problems'] else 'PROBLEM'}")
    rep["ok"] = not rep["problems"]
    if not rep["ok"]:
        print(json.dumps(rep, indent=1, sort_keys=True, default=str))
        raise RuntimeError("PIN CHECK GATE FAILED (before any test clip was opened): " + "; ".join(rep["problems"]))
    return rep


def replay_compare(main_dir, replay_dir, core_sha256, notebook_sha256, log=print):
    """main_dir = the published S-Edi main run dir (J4); replay_dir = its edimine_repro variant dir. ONE clip. The reference MOT file is read from the Volume (no
    local copy of any test track was read by the wiring, b1.11 rule 1, so no sha256 of it is pinned here). Raises RuntimeError on any failure."""
    clip = REPLAY["clip"]
    rep = {"gate": "replay_compare", "main_dir": main_dir, "replay_dir": replay_dir, "clip": clip, "problems": [], "code": {}}

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
        rep["code"] = {"replay_tol_bench_core": got_core, "replay_notebook": got_nb, "expected_core": core_sha256, "expected_notebook": notebook_sha256}
        if got_core != core_sha256:
            bad(f"the replay ran tol_bench_core {str(got_core)[:16]}, not the new core {core_sha256[:16]}")
        if got_nb != notebook_sha256:
            bad(f"the replay ran notebook {str(got_nb)[:16]}, not the new Tol_63 {notebook_sha256[:16]}")
        j4p = f"{main_dir}/BENCH_CONFIG.json"
        if osp.isfile(j4p):                    # information only: the published run's own config (older code layouts are read tolerantly)
            try:
                j4 = _read_json(j4p)
                rep["code"]["j4_tol_bench_core"] = ((j4.get("code") or {}).get("tol_bench_core") or {}).get("source")
                rep["code"]["j4_notebook"] = ((j4.get("code") or {}).get("notebook") or {}).get("source")
                vj, vr = j4.get("versions") or {}, cfg.get("versions") or {}
                rep["version_differences_info_only"] = {k: [vj.get(k), vr.get(k)] for k in sorted(set(vj) | set(vr)) if vj.get(k) != vr.get(k)}
            except (ValueError, AttributeError, TypeError):
                rep["j4_bench_config_note"] = "present but not readable (information only)"
        else:
            rep["j4_bench_config_note"] = "absent (information only; the byte comparison below is the gate)"
    # the replay is ONE clip: exactly one .done marker, exactly one MOT file
    done = sorted(x for x in (os.listdir(replay_dir) if osp.isdir(replay_dir) else []) if x.startswith(".done_"))
    if done != [f".done_{clip}"]:
        bad(f"the replay directory holds the markers {done}, expected exactly ['.done_{clip}']")
    mot = sorted(os.listdir(f"{replay_dir}/mot")) if osp.isdir(f"{replay_dir}/mot") else []
    if mot != [f"{clip}.txt"]:
        bad(f"the replay mot/ holds {mot}, expected exactly ['{clip}.txt']")
    r = rep["clip_result"] = {"reference": "J4 main run MOT file on the Volume (not pinned locally)"}
    mp, rp = f"{main_dir}/mot/{clip}.txt", f"{replay_dir}/mot/{clip}.txt"
    for what, path in (("j4", mp), ("replay", rp)):
        if not osp.isfile(path):
            bad(f"{clip}: {path} is missing")
        else:
            r[f"{what}_bytes"] = osp.getsize(path)
            r[f"{what}_sha256"] = _sha256_file(path)
    r["identical"] = bool(r.get("replay_sha256") and r.get("replay_sha256") == r.get("j4_sha256") and r.get("replay_bytes") == r.get("j4_bytes"))
    if r.get("replay_sha256") is not None and r.get("j4_sha256") is not None and not r["identical"]:
        bad(f"{clip}: the replay MOT file hashes to {r['replay_sha256'][:16]} ({r['replay_bytes']} bytes), the J4 file to {r['j4_sha256'][:16]} ({r['j4_bytes']} bytes): "
            "NOT byte-identical")
    log(f"replay {clip}: replay {str(r.get('replay_sha256'))[:16]} J4 {str(r.get('j4_sha256'))[:16]} {'IDENTICAL' if r['identical'] else 'DIFFERENT'}")
    rep["ok"] = not rep["problems"] and r["identical"]
    if not rep["ok"]:
        print(json.dumps(rep, indent=1, sort_keys=True))
        raise RuntimeError("REPLAY GATE FAILED (STOP before S-Edi-mined is tracked on the test clips; report to the controller): " + "; ".join(rep["problems"]))
    return rep


def deep_diff(a, b, path="$", out=None):
    """Exact structural comparison -> list of (path, a, b). A float that differs in the last digit is a difference."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                out.append((f"{path}.{k}", "<absent>", b[k]))
            elif k not in b:
                out.append((f"{path}.{k}", a[k], "<absent>"))
            else:
                deep_diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}[len]", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f"{path}[{i}]", out)
    else:
        same = (a == b) and (type(a) is type(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool)))
        if not same:
            out.append((path, a, b))
    return out


def count_leaves(x):
    if isinstance(x, dict):
        return sum(count_leaves(v) for v in x.values())
    if isinstance(x, list):
        return sum(count_leaves(v) for v in x)
    return 1


def _is_new_candidate(sc):
    try:
        return (sc.get("dataset") == "edinburgh" and sc.get("split") == "test" and sc.get("protocol") == "A" and sc.get("ref_arm") == "S-Edi"
                and sorted((sc.get("results") or {})) == sorted(EXISTING + (ARM,)))
    except AttributeError:
        return False


def find_scores(scores_root, ref_sid8, new_score_dir=""):
    """-> (ref dir name, new dir name). Exactly one reference (`*_<sid8>`) and exactly one new candidate (or the named one); anything else raises RuntimeError."""
    if not osp.isdir(scores_root):
        raise RuntimeError(f"IDENTITY GATE FAILED: {scores_root} does not exist")
    names = sorted(d for d in os.listdir(scores_root) if osp.isfile(osp.join(scores_root, d, "score.json")))
    refs = [d for d in names if d.endswith("_" + ref_sid8)]
    if len(refs) != 1:
        raise RuntimeError(f"IDENTITY GATE FAILED: {len(refs)} score directories end with _{ref_sid8} under {scores_root} (expected exactly 1)")
    if new_score_dir:
        if new_score_dir not in names:
            raise RuntimeError(f"IDENTITY GATE FAILED: the named new score directory {new_score_dir!r} has no score.json")
        return refs[0], new_score_dir
    cands = [d for d in names if d != refs[0] and _is_new_candidate(_read_json(osp.join(scores_root, d, "score.json")))]
    if len(cands) != 1:
        raise RuntimeError(f"IDENTITY GATE FAILED: {len(cands)} candidate new scores (ref_arm S-Edi, the six existing arms + {ARM}) under {scores_root}: {cands}; "
                           "expected exactly 1 (name it with new_score_dir if a retried scoring left a duplicate)")
    return refs[0], cands[0]


def compare_scores(ref, new, raw_ref=None, raw_new=None):
    """Pure comparison of the two score.json objects (and, if given, the two per_seq_raw.json objects) for the six existing arms. -> report dict. Never copies a value
    of the new arm into the report: only the six arms' blocks of BOTH scores and the shared inputs are read."""
    problems, diffs = [], []
    six = list(EXISTING)
    for tag, sc, want_ref, want_arms in (("reference", ref, "T", six), ("new", new, "S-Edi", six + [ARM])):
        for k, v in (("dataset", "edinburgh"), ("split", "test"), ("protocol", "A"), ("gt_variant", "interp"), ("ref_arm", want_ref)):
            if sc.get(k) != v:
                problems.append(f"{tag} score: {k}={sc.get(k)!r}, expected {v!r}")
        if sc.get("variant") not in (None, ""):
            problems.append(f"{tag} score: variant={sc.get('variant')!r}, expected none")
        if sorted((sc.get("results") or {})) != sorted(want_arms):
            problems.append(f"{tag} score: arms {sorted((sc.get('results') or {}))}, expected {sorted(want_arms)}")
        cfg = sc.get("trackeval_dataset_config") or {}
        if cfg.get("TRACKERS_TO_EVAL") != sorted(want_arms):
            problems.append(f"{tag} score: TRACKERS_TO_EVAL {cfg.get('TRACKERS_TO_EVAL')!r}, expected {sorted(want_arms)}")
    n_fields = 0
    if not problems:
        for k in TOP_EQUAL:
            diffs += deep_diff(ref.get(k), new.get(k), f"$.{k}")
            n_fields += count_leaves(ref.get(k))
        te_r = {k: v for k, v in (ref.get("trackeval") or {}).items() if k != "file"}
        te_n = {k: v for k, v in (new.get("trackeval") or {}).items() if k != "file"}
        diffs += deep_diff(te_r, te_n, "$.trackeval")
        n_fields += count_leaves(te_r)
        cr = {k: v for k, v in (ref.get("trackeval_dataset_config") or {}).items() if k != "TRACKERS_TO_EVAL"}
        cn = {k: v for k, v in (new.get("trackeval_dataset_config") or {}).items() if k != "TRACKERS_TO_EVAL"}
        diffs += deep_diff(cr, cn, "$.trackeval_dataset_config")
        n_fields += count_leaves(cr)
        for k in INPUT_EQUAL:
            diffs += deep_diff((ref.get("inputs") or {}).get(k), (new.get("inputs") or {}).get(k), f"$.inputs.{k}")
            n_fields += count_leaves((ref.get("inputs") or {}).get(k))
        for a in six:
            ir, inn = ((ref.get("inputs") or {}).get("runs") or {}).get(a), ((new.get("inputs") or {}).get("runs") or {}).get(a)
            diffs += deep_diff(ir, inn, f"$.inputs.runs.{a}")
            n_fields += count_leaves(ir)
            rr, rn = (ref.get("results") or {}).get(a), (new.get("results") or {}).get(a)
            # review m1(b): the WHOLE results.<arm> object is compared (every key present on either side), not only the four registered blocks: a key added or removed
            # outside combined / strata / per_seq / xcheck is a difference like any other (deep_diff reports it as <absent>)
            diffs += deep_diff(rr, rn, f"$.results.{a}")
            n_fields += count_leaves(rr)
    # review m1(a): the raw per-sequence results of the six arms GATE (they are functions of the same MOT and GT bytes as `results`, so any difference is a real difference)
    raw = {"checked": False, "gating": True}
    if raw_ref is not None and raw_new is not None and not problems:
        rprob = [f"{tag} per_seq_raw.json: no results.{a} object" for tag, rw in (("reference", raw_ref), ("new", raw_new)) for a in six
                 if not isinstance(((rw if isinstance(rw, dict) else {}).get("results") or {}).get(a), dict)]
        if rprob:
            problems += rprob
        else:
            rd = []
            for a in six:
                rd += deep_diff(raw_ref["results"][a], raw_new["results"][a], f"$.per_seq_raw.{a}")
                n_fields += count_leaves(raw_ref["results"][a])
            raw = {"checked": True, "differences": len(rd), "first": [(p, repr(x)[:60], repr(y)[:60]) for p, x, y in rd[:10]], "gating": True}
            diffs += rd
    # warnings: informational; a warning that names the new arm is dropped (it could carry a value of the new arm) and only counted
    wr = [str(w) for w in (ref.get("warnings") or [])]
    wn_all = [str(w) for w in (new.get("warnings") or [])]
    wn = [w for w in wn_all if ARM not in w]
    rep = {"gate": "identity_gate", "arms_compared": six, "fields_compared": n_fields, "problems": problems, "differences": len(diffs),
           "first_differences": [(p, repr(x)[:80], repr(y)[:80]) for p, x, y in diffs[:25]], "per_seq_raw": raw,
           "warnings_reference": wr, "warnings_new_without_new_arm": wn, "warnings_new_naming_the_new_arm_dropped": len(wn_all) - len(wn),
           "new_arm_values_read": False}
    rep["ok"] = not problems and not diffs
    return rep


def identity_gate(scores_root, ref_sid8, new_score_dir="", log=print):
    """Reads <scores_root>/<ref dir>/score.json and the new dir's score.json (+ per_seq_raw.json of both when present)."""
    ref_name, new_name = find_scores(scores_root, ref_sid8, new_score_dir)
    rp, npth = osp.join(scores_root, ref_name), osp.join(scores_root, new_name)
    ref, new = _read_json(osp.join(rp, "score.json")), _read_json(osp.join(npth, "score.json"))
    raw_ref = _read_json(osp.join(rp, "per_seq_raw.json")) if osp.isfile(osp.join(rp, "per_seq_raw.json")) else None
    raw_new = _read_json(osp.join(npth, "per_seq_raw.json")) if osp.isfile(osp.join(npth, "per_seq_raw.json")) else None
    rep = compare_scores(ref, new, raw_ref, raw_new)
    miss = [n for n, r in (("reference", raw_ref), ("new", raw_new)) if r is None]
    if miss:                                                    # review m1(a): an absent raw file means the raw check did not run, which is not a pass
        rep["problems"].append(f"per_seq_raw.json is missing in the {' and '.join(miss)} score directory: the raw per-sequence results of the six arms were not compared")
        rep["ok"] = False
    rep.update(scores_root=scores_root, reference_dir=ref_name, new_dir=new_name)
    del ref, new, raw_ref, raw_new
    log(f"identity_gate: reference {ref_name} vs new {new_name}: {rep['fields_compared']} fields of {len(EXISTING)} arms compared, {rep['differences']} differences, "
        f"problems {len(rep['problems'])}")
    if not rep["ok"]:
        print(json.dumps(rep, indent=1, sort_keys=True, default=str))
        raise RuntimeError(f"IDENTITY GATE FAILED: {rep['differences']} differences, problems {rep['problems']}")
    return rep


def main(raw, *, R=None, tp=None, code_dir=None):
    P = parse_params(raw)
    if R is None:
        tp = tp or _import_tol_paths()
        R = tp.load(version=REGISTRY_VERSION)
    if P.mode == "pin_check":
        rep = pin_check(R.path("paper1.distillation_runs"), P.pin, code_dir if code_dir is not None else os.environ.get("TOL63_CODE_DIR", ""), P.core_sha256,
                        P.require_done, getattr(R, "sha256", None))
    elif P.mode == "replay_compare":
        bench = R.path("paper1.bench")
        main_dir = f"{bench}/{REPLAY['dataset']}/{REPLAY['arm']}__prot{REPLAY['protocol']}/{REPLAY['split']}"
        rep = replay_compare(main_dir, f"{main_dir}/{P.variant}", P.core_sha256, P.notebook_sha256)
    else:
        rep = identity_gate(f"{R.path('paper1.scores')}/{REPLAY['dataset']}/prot{REPLAY['protocol']}/{REPLAY['split']}", P.ref_sid8, P.new_score_dir)
    rep["notebook"] = f"{NB_NAME} {NB_VERSION}"
    return rep


# COMMAND ----------

# MAGIC %md
# MAGIC ## Local selftest (`TOL82_SELFTEST=1`; temporary registry root, synthetic files with SENTINEL numbers; never on Databricks; reads no Edinburgh clip)

# COMMAND ----------


def _synthetic_score(arms, ref_arm, sentinel_for=None):
    """A synthetic score.json shaped like Tol_64 s1.2 output (edinburgh / test / A / interp). Values are made-up numbers; `sentinel_for` arms get 91.2345-style values."""
    seqs = ["2019_12_02_000005", "2019_12_02_000208", "2019_12_10_000060", "2019_12_10_000078"]

    def summ(i, sent):
        base = 91.0 if sent else 70.0
        return {"HOTA": base + i + 0.2345, "DetA": base + i + 1.5, "AssA": base + i - 0.5, "LocA": base + 5.0, "MOTA": base + i + 7.0, "IDF1": base + i + 3.0, "IDSW": i % 3}

    def res(a, k):
        sent = sentinel_for is not None and a in sentinel_for
        return {"combined": summ(k, sent), "strata": {}, "per_seq": {s: summ(k + j, sent) for j, s in enumerate(seqs)}, "xcheck": {"ok": True, "d_MOTA": 0.01 * k, "d_IDF1": 0.0}}
    arms = list(arms)
    return {"score_version": "s1.2", "bench_version": "b1.0", "protocol_doc": "BENCH_PROTOCOL.md b1.0", "dataset": "edinburgh", "split": "test", "protocol": "A",
            "variant": None, "gt_variant": "interp", "smoke": None, "ref_arm": ref_arm, "created_utc": "2026-10-30T00:00:00Z", "score_id": "x",
            "code": {"tol64": {"a": "1"}, "tol_bench_core": "c"},
            "trackeval": {"version": "1.3.0", "commit": "abc", "file": "/x/y.py"},
            "trackeval_dataset_config": {"TRACKERS_TO_EVAL": sorted(arms), "SEQ_INFO": {s: 600 for s in seqs}, "CLASSES_TO_EVAL": ["pedestrian"]},
            "trackeval_metrics": {"HOTA": "alphas 0.05..0.95", "CLEAR": {"THRESHOLD": 0.5}},
            "inputs": {"runs": {a: {"dirs": [f"/v/bench/edinburgh/{a}__protA/test"], "configs": {s: f"cfg-{a}-{s}" for s in seqs}, "stored_xy_offset": 0.0, "xy_add": 0.0} for a in arms},
                       "skipped_arms": {}, "sequences": seqs, "sequence_source": "EDI_TEST", "seq_meta": {s: {"n_frames": 600} for s in seqs},
                       "gt": {s: f"gtsha-{s}" for s in seqs}, "trackeval_lengths": {s: 600 for s in seqs}, "strata": {}},
            "xy_offset": {"frozen": True, "o": 0.0}, "frame_hash_check": {"status": "ok", "hashes_compared": 10},
            "results": {a: res(a, k) for k, a in enumerate(sorted(arms))},
            "edinburgh_clip_deltas": {a: {"pooled_descriptive": {"HOTA": 0.0}} for a in arms if a != ref_arm},
            "warnings": ["arms ran with different tol_bench_core code (source_norm sha256)"] if ref_arm == "S-Edi" else [],
            "statistics": {"B": 10000, "seed": 20261014, "alpha": 0.05, "margin_hota": 1.0, "edinburgh": "n = 4 clips"}}


def selftest():
    res = []

    def check(name, cond, detail=""):
        res.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))

    import contextlib
    import io

    def raises(fn, exc=Exception, text=""):
        try:
            with contextlib.redirect_stdout(io.StringIO()):         # a failed gate prints its report; the selftest does not need to see it
                fn()
        except exc as e:
            return text in str(e)
        return False
    quiet = lambda *a: None  # noqa: E731
    CORE, NB = "c" * 64, "d" * 64
    S = PINS["SEDI_BEST_EMA_VAL_LOSS"]

    def w(path, data, mode="wb"):
        os.makedirs(osp.dirname(path), exist_ok=True)
        with open(path, mode) as f:
            f.write(data if isinstance(data, bytes) else data.encode("utf-8"))

    def sha(b):
        return hashlib.sha256(b).hexdigest()
    # ---------------- G-M (the cases of DESIGN_EDI_v2 3.3)
    check("G-M constants: S-Edi's recorded loss, the band, and no rounded-band shortcut",
          PINS["SEDI_BEST_EMA_VAL_LOSS"] == 0.25041554232438407 and PINS["GM_REL_TOL"] == 0.1
          and abs((S - 0.1 * S) - 0.22537398809194564) < 1e-15 and abs((S + 0.1 * S) - 0.2754570965568225) < 1e-15)
    check("G-M passes: 0.2254, 0.2754, S itself, 0.23, an int-valued float", all(gm_check(v)[0] for v in (0.2254, 0.2754, S, 0.23, 0.26)))
    check("G-M refuses: 0.22537, 0.2755 (inside the old rounded band), 0.275458", not any(gm_check(v)[0] for v in (0.22537, 0.2755, 0.275458)))
    check("G-M refuses: NaN, +inf, -inf, a string, None, True, a list, a negative", not any(gm_check(v)[0] for v in (float("nan"), float("inf"), float("-inf"), "0.25", None, True, [0.25], -0.25)))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = tmp.replace("\\", "/")
        gp = f"{tmp}/train_done.json"
        check("G-M from file: a missing file refuses", not gm_from_train_done(gp)[0])
        w(gp, json.dumps({"best_ema_val_loss": 0.26}))
        check("G-M from file: 0.26 passes", gm_from_train_done(gp)[0] and gm_from_train_done(gp)[2] == 0.26)
        w(gp, json.dumps({"other": 1}))
        check("G-M from file: a missing key refuses", not gm_from_train_done(gp)[0])
        w(gp, "{ not json")
        check("G-M from file: malformed JSON refuses", not gm_from_train_done(gp)[0])
        w(gp, '{"best_ema_val_loss": NaN}')
        check("G-M from file: a NaN literal refuses", not gm_from_train_done(gp)[0])
        w(gp, json.dumps({"best_ema_val_loss": "0.26"}))
        check("G-M from file: a string value refuses", not gm_from_train_done(gp)[0])
        # ---------------- parameters
        check("params: unknown widget / bad mode / missing pin / bad core sha / cross-mode parameters refused",
              raises(lambda: parse_params({"nope": "1"}), ValueError, "unknown")
              and raises(lambda: parse_params({"mode": "x"}), ValueError, "mode")
              and raises(lambda: parse_params({"mode": "pin_check"}), ValueError, "pin")
              and raises(lambda: parse_params({"mode": "pin_check", "pin": "["}), ValueError, "JSON")
              and raises(lambda: parse_params({"mode": "pin_check", "pin": "{}"}), ValueError, "core_sha256")
              and raises(lambda: parse_params({"mode": "pin_check", "pin": "{}", "core_sha256": "C" * 64}), ValueError, "core_sha256")
              and raises(lambda: parse_params({"mode": "pin_check", "pin": "{}", "core_sha256": CORE, "require_done": "maybe"}), ValueError, "require_done")
              and raises(lambda: parse_params({"mode": "pin_check", "pin": "{}", "core_sha256": CORE, "notebook_sha256": NB}), ValueError, "not a pin_check parameter")
              and raises(lambda: parse_params({"mode": "replay_compare"}), ValueError, "core_sha256")
              and raises(lambda: parse_params({"mode": "replay_compare", "core_sha256": CORE}), ValueError, "notebook_sha256")
              and raises(lambda: parse_params({"mode": "replay_compare", "core_sha256": CORE, "notebook_sha256": NB, "pin": "{}"}), ValueError, "not a replay_compare parameter")
              and raises(lambda: parse_params({"mode": "replay_compare", "core_sha256": CORE, "notebook_sha256": NB, "variant": "other"}), ValueError, "registered")
              and raises(lambda: parse_params({"mode": "replay_compare", "core_sha256": CORE, "notebook_sha256": NB, "require_done": "false"}), ValueError, "pin_check parameter")
              and raises(lambda: parse_params({"mode": "identity_gate", "ref_score_sid8": "xyz"}), ValueError, "8 lowercase hex")
              and raises(lambda: parse_params({"mode": "identity_gate", "new_score_dir": "../x"}), ValueError, "one directory name")
              and raises(lambda: parse_params({"mode": "identity_gate", "core_sha256": CORE}), ValueError, "not a identity_gate parameter"))
        Pr = parse_params({"mode": "replay_compare", "core_sha256": CORE, "notebook_sha256": NB})
        Pi = parse_params({"mode": "identity_gate"})
        check("params: replay_compare variant defaults to edimine_repro; identity_gate reference defaults to eb9f09ff",
              Pr.variant == "edimine_repro" and Pr.core_sha256 == CORE and Pi.ref_sid8 == "eb9f09ff" and Pi.new_score_dir == "")
        # ---------------- pin_check
        runs = f"{tmp}/runs"
        payload = b"edinburgh-mined-checkpoint-bytes"
        fdir = "final_a3f19c"
        run = f"{runs}/{RUN}"
        code = f"{tmp}/core"
        core_src = ('ARMS = {\n    "T": {"family": "sam3"},\n    "S-Edi-mined": {"family": "sam3", "student_key": "models.student_edinburgh",\n'
                    f'                    "student_distill_file": "{RUN}/{fdir}/{CKPT_NAME}",\n'
                    f'                    "student_sha256": "{sha(payload)}", "scope": "ablation (b1.11)", "domain": "x"}},\n}}\n')
        w(f"{code}/tol_bench_core.py", core_src)
        core_sha = sha(core_src.encode())
        w(f"{run}/{fdir}/{CKPT_NAME}", payload)
        td = {"final_dir": fdir, "ckpt": CKPT_NAME, "ckpt_sha256": sha(payload), "rid": "a3f19c", "best_epoch": 70, "epochs_run": 85, "stopped": "early_stop", "best_ema_val_loss": 0.2501}
        w(f"{run}/train_done.json", json.dumps(td))
        def done_doc(**kw):                                     # what Tol_02d (eval_stage=only) writes last, reduced to the fields this notebook reads
            d = {"ckpt_sha256": sha(payload), "arm": ARM, "ckpt_pin_checked": True, "positive_control_status": "identical"}
            d.update(kw)
            return json.dumps({k: v for k, v in d.items() if v != "<absent>"})
        w(f"{run}/DONE.json", done_doc())
        pin = {"arm": ARM, "run": RUN, "final_dir": fdir, "sha256": sha(payload)}
        rep = pin_check(runs, dict(pin), code, core_sha, True, PINS["REGISTRY_SHA256"], quiet)
        check("pin_check: the checkpoint hashes to the pin and agrees with train_done.json, DONE.json and the core's ARMS entry; G-M passes",
              rep["ok"] and rep["g_m"]["ok"] and rep["sha256"] == pin["sha256"] and rep["bytes"] == len(payload) and rep["arms_entry"]["student_sha256"] == pin["sha256"]
              and abs(rep["g_m"]["relative_deviation"] - abs(0.2501 - S) / S) < 1e-12)

        def pc(mut_pin=None, core=core_sha, req=True, regsha=PINS["REGISTRY_SHA256"], cdir=code):
            p = dict(pin)
            if mut_pin:
                mut_pin(p)
            return lambda: pin_check(runs, p, cdir, core, req, regsha, quiet)
        check("pin_check: a wrong pin / final_dir / arm / run / extra key / bad hex is refused",
              raises(pc(lambda p: p.update(sha256="0" * 64)), RuntimeError, "PIN CHECK GATE FAILED")
              and raises(pc(lambda p: p.update(final_dir="final_ffffff")), RuntimeError, "PIN CHECK GATE FAILED")
              and raises(pc(lambda p: p.update(arm="S-PT-mined")), ValueError, "arm / run")
              and raises(pc(lambda p: p.update(run="pigtrack_mined_v1")), ValueError, "arm / run")
              and raises(pc(lambda p: p.update(extra=1)), ValueError, "keys must be")
              and raises(pc(lambda p: p.update(sha256="A" * 64)), ValueError, "64 lowercase hex")
              and raises(pc(lambda p: p.update(final_dir="final_abc")), ValueError, "final_<6 hex>"))
        w(f"{run}/{fdir}/{CKPT_NAME}", b"tampered")
        check("pin_check: a checkpoint whose bytes changed after training is refused", raises(pc(), RuntimeError, "hashes to"))
        w(f"{run}/{fdir}/{CKPT_NAME}", payload)
        for label, tdm, text in (("another ckpt_sha256", dict(td, ckpt_sha256="9" * 64), "train_done.json ckpt_sha256"),
                                 ("another final_dir", dict(td, final_dir="final_ffffff"), "train_done.json final_dir"),
                                 ("another ckpt name", dict(td, ckpt="x.pt"), "train_done.json ckpt"),
                                 ("G-M out of band (0.2755)", dict(td, best_ema_val_loss=0.2755), "G-M"),
                                 ("G-M NaN", dict(td, best_ema_val_loss=float("nan")), "G-M"),
                                 ("G-M missing key", {k: v for k, v in td.items() if k != "best_ema_val_loss"}, "G-M"),
                                 ("G-M string", dict(td, best_ema_val_loss="0.25"), "G-M")):
            w(f"{run}/train_done.json", json.dumps(tdm))
            check(f"pin_check: train_done.json with {label} is refused", raises(pc(), RuntimeError, text))
        w(f"{run}/train_done.json", json.dumps(td))
        w(f"{run}/DONE.json", done_doc(ckpt_sha256="8" * 64))
        check("pin_check: DONE.json recording another checkpoint is refused", raises(pc(), RuntimeError, "DONE.json ckpt_sha256"))
        # review m3: the presence of DONE.json must mean what D3 says (the fidelity job re-checked the pin and its positive control reproduced FID-E)
        for label, kw, text in (("another arm", {"arm": "S-PT-mined"}, "DONE.json arm"), ("no arm", {"arm": "<absent>"}, "DONE.json arm"),
                                ("ckpt_pin_checked false", {"ckpt_pin_checked": False}, "ckpt_pin_checked"), ("ckpt_pin_checked absent", {"ckpt_pin_checked": "<absent>"}, "ckpt_pin_checked"),
                                ("ckpt_pin_checked as the string true", {"ckpt_pin_checked": "true"}, "ckpt_pin_checked"),
                                ("positive control differs", {"positive_control_status": "differs"}, "positive_control_status"),
                                ("positive control incomplete", {"positive_control_status": "incomplete"}, "positive_control_status"),
                                ("positive control null", {"positive_control_status": None}, "positive_control_status"),
                                ("positive control absent", {"positive_control_status": "<absent>"}, "positive_control_status")):
            w(f"{run}/DONE.json", done_doc(**kw))
            check(f"pin_check: DONE.json with {label} is refused (also when require_done is false: a present file must say what it means)",
                  raises(pc(), RuntimeError, text) and raises(pc(req=False), RuntimeError, text))
        w(f"{run}/DONE.json", "[1, 2]")
        check("pin_check: a DONE.json that is not a JSON object is refused", raises(pc(), RuntimeError, "not a readable JSON object"))
        w(f"{run}/DONE.json", "{ not json")
        check("pin_check: an unreadable DONE.json is refused", raises(pc(), RuntimeError, "not a readable JSON object"))
        w(f"{run}/DONE.json", done_doc(positive_control_status="within_4dp"))
        check("pin_check: positive_control_status within_4dp (the second registered pass status) is accepted", pc()()["ok"])
        _td2 = osp.join(osp.dirname(osp.abspath(__file__)), "Tol_02d_distill_edinburgh_mined.py")
        if osp.isfile(_td2):
            import ast as _ast
            _m = re.search(r"^FIDE_PASS = (\(.*?\))$", open(_td2, encoding="utf-8").read(), re.M)
            check("pin_check: the accepted positive-control statuses equal Tol_02d's FIDE_PASS", bool(_m) and tuple(_ast.literal_eval(_m.group(1))) == DONE_POSITIVE_CONTROL_PASS)
        os.remove(f"{run}/DONE.json")
        check("pin_check: a missing DONE.json is refused while require_done is true, and accepted (G-M still evaluated) when false",
              raises(pc(), RuntimeError, "DONE.json is missing") and pc(req=False)()["ok"])
        w(f"{run}/DONE.json", done_doc())
        os.remove(f"{run}/train_done.json")
        check("pin_check: a run without train_done.json is refused (G-M fails closed)", raises(pc(), RuntimeError, "G-M: "))
        w(f"{run}/train_done.json", json.dumps(td))
        os.remove(f"{run}/{fdir}/{CKPT_NAME}")
        check("pin_check: a missing checkpoint file is refused", raises(pc(), RuntimeError, "is missing"))
        w(f"{run}/{fdir}/{CKPT_NAME}", payload)
        check("pin_check: the core in TOL63_CODE_DIR hashing to another sha256 / lacking the ARMS entry / carrying another pin or scope is refused",
              raises(pc(core="e" * 64), RuntimeError, "not the final core")
              and raises(pc(cdir=f"{tmp}/nocore"), RuntimeError, "is missing"))
        for label, new_src, text in (("another pin", core_src.replace(sha(payload), "7" * 64), "ARMS pin"),
                                     ("another file", core_src.replace(fdir, "final_ffffff"), "ARMS student_distill_file"),
                                     ("another scope", core_src.replace("ablation (b1.11)", "S"), "ARMS scope"),
                                     ("no entry", core_src.replace('"S-Edi-mined"', '"S-Edi-other"'), "no ARMS entry")):
            w(f"{code}2/tol_bench_core.py", new_src)
            check(f"pin_check: a core with {label} is refused", raises(pc(core=sha(new_src.encode()), cdir=f"{code}2"), RuntimeError, text))
        check("pin_check: a registry file that is not the pinned one is refused", raises(pc(regsha="f" * 64), RuntimeError, "registry file hashes"))
        check("pin_check: after restoring everything it passes again", pc()()["ok"])
        # ---------------- replay_compare
        bench = f"{tmp}/bench/edinburgh/S-Edi__protA/test"
        rdir = f"{bench}/edimine_repro"
        clip = REPLAY["clip"]
        synth = b"1,1,10,10,20,20,1,-1,-1,-1\n2,1,11,10,20,20,1,-1,-1,-1\n"
        cfgok = {"dataset": "edinburgh", "split": "test", "protocol": "A", "variant": "edimine_repro", "arm": {"arm": "S-Edi"}, "run": {"smoke_max_frames": None},
                 "gpu": {"class": "A10"}, "versions": {"torch": "2.7.0"}, "code": {"tol_bench_core": {"source": CORE}, "notebook": {"source": NB}}}
        w(f"{bench}/BENCH_CONFIG.json", json.dumps({"code": {"tol_bench_core": {"source": "b" * 64}, "notebook": {"source": "e" * 64}}, "versions": {"torch": "2.7.0"}}))
        w(f"{bench}/mot/{clip}.txt", synth)
        w(f"{rdir}/mot/{clip}.txt", synth)
        w(f"{rdir}/.done_{clip}", b"")
        w(f"{rdir}/BENCH_CONFIG.json", json.dumps(cfgok))

        def rc(core=CORE, nb=NB):
            return lambda: replay_compare(bench, rdir, core, nb, quiet)
        rep = rc()()
        check("replay: a byte-identical MOT file + a BENCH_CONFIG made by the new core / Tol_63 -> pass", rep["ok"] and rep["clip_result"]["identical"]
              and rep["code"]["j4_tol_bench_core"] == "b" * 64 and rep["clip_result"]["replay_bytes"] == len(synth))
        w(f"{rdir}/mot/{clip}.txt", synth + b"x")
        check("replay: ONE byte more in the replay file -> STOP", raises(rc(), RuntimeError, "NOT byte-identical") and raises(rc(), RuntimeError, "REPLAY GATE FAILED"))
        w(f"{rdir}/mot/{clip}.txt", synth[:-1] + b" ")
        check("replay: one byte changed (same length) -> STOP", raises(rc(), RuntimeError, "NOT byte-identical"))
        w(f"{rdir}/mot/{clip}.txt", synth)
        os.remove(f"{rdir}/mot/{clip}.txt")
        check("replay: a missing replay MOT file -> STOP", raises(rc(), RuntimeError, clip))
        w(f"{rdir}/mot/{clip}.txt", synth)
        w(f"{rdir}/mot/other.txt", synth)
        check("replay: a second MOT file (not ONE clip) -> STOP", raises(rc(), RuntimeError, "mot/ holds"))
        os.remove(f"{rdir}/mot/other.txt")
        w(f"{rdir}/.done_other", b"")
        check("replay: a second .done marker -> STOP", raises(rc(), RuntimeError, "markers"))
        os.remove(f"{rdir}/.done_other")
        os.remove(f"{rdir}/.done_{clip}")
        check("replay: a missing .done marker -> STOP", raises(rc(), RuntimeError, "markers"))
        w(f"{rdir}/.done_{clip}", b"")
        os.remove(f"{bench}/mot/{clip}.txt")
        check("replay: the J4 reference file missing -> STOP", raises(rc(), RuntimeError, "is missing"))
        w(f"{bench}/mot/{clip}.txt", synth)
        check("replay: the replay was NOT made by the new core / the new Tol_63 -> STOP",
              raises(rc(core="e" * 64), RuntimeError, "not the new core") and raises(rc(nb="e" * 64), RuntimeError, "not the new Tol_63"))
        for label, mut, text in (("arm", {"arm": {"arm": "S-Edi-mined"}}, "arm="), ("variant", {"variant": "x"}, "variant="), ("split", {"split": "val"}, "split="),
                                 ("dataset", {"dataset": "pigtrack"}, "dataset="), ("smoke", {"run": {"smoke_max_frames": 30}}, "full length"),
                                 ("gpu", {"gpu": {"class": "A100"}}, "GPU class")):
            w(f"{rdir}/BENCH_CONFIG.json", json.dumps(dict(cfgok, **mut)))
            check(f"replay: BENCH_CONFIG with another {label} -> STOP", raises(rc(), RuntimeError, text))
        os.remove(f"{rdir}/BENCH_CONFIG.json")
        check("replay: no BENCH_CONFIG (the replay never ran) -> STOP", raises(rc(), RuntimeError, "BENCH_CONFIG.json is missing"))
        w(f"{rdir}/BENCH_CONFIG.json", json.dumps(cfgok))
        os.remove(f"{bench}/BENCH_CONFIG.json")
        check("replay: the J4 BENCH_CONFIG missing is information only (the byte comparison is the gate)", rc()()["ok"])
        w(f"{bench}/BENCH_CONFIG.json", "{ not json")
        check("replay: an unreadable J4 BENCH_CONFIG is information only", rc()()["ok"])
        # ---------------- identity gate (synthetic scores with SENTINEL values for the new arm)
        six = list(EXISTING)
        sroot = f"{tmp}/scores/edinburgh/protA/test"
        ref = _synthetic_score(six, "T")
        new = _synthetic_score(six + [ARM], "S-Edi", sentinel_for={ARM})
        for a in six:                                             # the six arms' blocks are the same objects in both scores (the registered expectation)
            new["results"][a] = json.loads(json.dumps(ref["results"][a]))
        new["warnings"] = ["arms ran with different tol_bench_core code (source_norm sha256)", f"{ARM}: py-motmetrics disagrees by 0.9183"]
        raw = {"note": "x", "results": {a: {"s1": {"HOTA": [0.5, 0.6]}} for a in six + [ARM]}}

        def put(name, sc, rw=None):
            w(f"{sroot}/{name}/score.json", json.dumps(sc))
            if rw is not None:
                w(f"{sroot}/{name}/per_seq_raw.json", json.dumps(rw))
        put("20261015T195520Z_eb9f09ff", ref, raw)
        put("20261030T230000Z_1a2b3c4d", new, raw)
        put("20261015T195520Z_fb93298d", _synthetic_score(["T", "S-Edi"], "T"), raw)          # an older unrelated score: ignored
        out = identity_gate(sroot, "eb9f09ff", "", quiet)
        blob = json.dumps(out, sort_keys=True, default=str)
        check("identity: 0 differences for the six existing arms -> pass; the report names no value of the new arm (sentinel 91.2345 / 92.2345 absent)",
              out["ok"] and out["differences"] == 0 and out["fields_compared"] > 200 and "91.2345" not in blob and "92.2345" not in blob and "0.9183" not in blob
              and out["new_arm_values_read"] is False and out["warnings_new_naming_the_new_arm_dropped"] == 1)
        check("identity: the raw per-sequence check ran, gates, and found 0 differences",
              out["per_seq_raw"]["checked"] and out["per_seq_raw"]["differences"] == 0 and out["per_seq_raw"]["gating"] is True)
        # the new arm's values may be anything: the gate does not compare them
        new2 = json.loads(json.dumps(new))
        new2["results"][ARM]["combined"]["HOTA"] = 12.5
        put("20261030T230000Z_1a2b3c4d", new2, raw)
        check("identity: a different value of the NEW arm does not matter", identity_gate(sroot, "eb9f09ff", "", quiet)["ok"])
        # single-field mutations of the six arms / inputs: each must fail
        muts = []

        def mut(label, fn):
            muts.append((label, fn))
        for a in six:
            mut(f"{a} combined HOTA last digit", lambda sc, a=a: sc["results"][a]["combined"].__setitem__("HOTA", sc["results"][a]["combined"]["HOTA"] + 1e-13))
            mut(f"{a} per_seq HOTA", lambda sc, a=a: sc["results"][a]["per_seq"]["2019_12_10_000060"].__setitem__("HOTA", 1.0))
            mut(f"{a} IDSW", lambda sc, a=a: sc["results"][a]["combined"].__setitem__("IDSW", 99))
            mut(f"{a} xcheck", lambda sc, a=a: sc["results"][a]["xcheck"].__setitem__("d_MOTA", 5.0))
            mut(f"{a} strata", lambda sc, a=a: sc["results"][a].__setitem__("strata", {"day": {"HOTA": 1}}))
            mut(f"{a} run config", lambda sc, a=a: sc["inputs"]["runs"][a]["configs"].__setitem__("2019_12_02_000005", "other"))
        for a in six:                                           # review m1(b): a key outside the four registered blocks of results.<arm>
            mut(f"{a} an extra key under results.<arm>", lambda sc, a=a: sc["results"][a].__setitem__("extra_block", {"x": 1}))
        mut("a result block removed", lambda sc: sc["results"]["T"].pop("xcheck"))
        mut("sequences", lambda sc: sc["inputs"]["sequences"].reverse())
        mut("GT sha256", lambda sc: sc["inputs"]["gt"].__setitem__("2019_12_02_000005", "other"))
        mut("trackeval lengths", lambda sc: sc["inputs"]["trackeval_lengths"].__setitem__("2019_12_02_000208", 599))
        mut("xy offset", lambda sc: sc["xy_offset"].__setitem__("o", 1.0))
        mut("bootstrap B", lambda sc: sc["statistics"].__setitem__("B", 1000))
        mut("score_version", lambda sc: sc.__setitem__("score_version", "s1.3"))
        mut("trackeval version", lambda sc: sc["trackeval"].__setitem__("version", "1.4"))
        mut("trackeval dataset config", lambda sc: sc["trackeval_dataset_config"].__setitem__("CLASSES_TO_EVAL", ["car"]))
        mut("gt_variant", lambda sc: sc.__setitem__("gt_variant", "hold"))
        mut("ref_arm of the new score", lambda sc: sc.__setitem__("ref_arm", "T"))
        mut("split", lambda sc: sc.__setitem__("split", "val"))
        mut("variant", lambda sc: sc.__setitem__("variant", "pf1"))
        mut("an arm missing", lambda sc: (sc["results"].pop("ET"), sc["trackeval_dataset_config"].__setitem__("TRACKERS_TO_EVAL", sorted(x for x in six + [ARM] if x != "ET"))))
        mut("TRACKERS_TO_EVAL", lambda sc: sc["trackeval_dataset_config"].__setitem__("TRACKERS_TO_EVAL", sorted(six)))
        n_mut = n_caught = 0
        for label, fn in muts:
            sc = json.loads(json.dumps(new2))
            fn(sc)
            put("20261030T230000Z_1a2b3c4d", sc, raw)
            n_mut += 1
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    identity_gate(sroot, "eb9f09ff", "", quiet)
            except RuntimeError as e:
                n_caught += "IDENTITY GATE FAILED" in str(e)
            else:
                print(f"  MISSED mutation: {label}")
        check(f"identity: {n_mut} single-field mutations of the six arms / inputs each FAIL the gate", n_mut == n_caught and n_mut >= 40, f"{n_caught}/{n_mut}")
        # reference-side mutation (a reference that is not the T-referenced interp test score)
        put("20261030T230000Z_1a2b3c4d", new2, raw)
        bad_ref = json.loads(json.dumps(ref))
        bad_ref["ref_arm"] = "S-Edi"
        put("20261015T195520Z_eb9f09ff", bad_ref, raw)
        check("identity: a reference that is not ref_arm T is refused", raises(lambda: identity_gate(sroot, "eb9f09ff", "", quiet), RuntimeError, "ref_arm"))
        put("20261015T195520Z_eb9f09ff", ref, raw)
        # review m1(a): a raw-only difference (results identical) GATES. Every one of the six arms, one leaf each; the new arm's raw values are never compared.
        n_rawm = n_rawc = 0
        for a in six:
            raw2 = json.loads(json.dumps(raw))
            raw2["results"][a]["s1"]["HOTA"][0] = 0.51
            put("20261030T230000Z_1a2b3c4d", new2, raw2)
            n_rawm += 1
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    identity_gate(sroot, "eb9f09ff", "", quiet)
            except RuntimeError as e:
                n_rawc += "IDENTITY GATE FAILED" in str(e)
            else:
                print(f"  MISSED raw mutation: {a}")
        check(f"identity: a raw-only per-sequence difference in any of the six arms FAILS the gate ({n_rawc}/{n_rawm})", n_rawm == n_rawc == 6)
        raw2 = json.loads(json.dumps(raw))
        raw2["results"]["T"]["s1"]["HOTA"][0] = 0.51
        d_ = compare_scores(json.loads(json.dumps(ref)), json.loads(json.dumps(new2)), raw, raw2)
        check("identity: the report counts the raw difference in `differences`, names it, and is not ok; results alone are identical",
              d_["differences"] == 1 and d_["ok"] is False and d_["per_seq_raw"]["checked"] and d_["per_seq_raw"]["differences"] == 1 and d_["per_seq_raw"]["gating"] is True
              and d_["first_differences"][0][0] == "$.per_seq_raw.T.s1.HOTA[0]")
        raw3 = json.loads(json.dumps(raw))
        raw3["results"][ARM]["s1"]["HOTA"][0] = 0.99
        put("20261030T230000Z_1a2b3c4d", new2, raw3)
        check("identity: a raw difference of the NEW arm only does not matter (the new arm is never compared)", identity_gate(sroot, "eb9f09ff", "", quiet)["ok"])
        raw4 = json.loads(json.dumps(raw))
        raw4["results"]["ET"].pop("s1")
        put("20261030T230000Z_1a2b3c4d", new2, raw4)
        check("identity: a clip missing from the new raw results of one arm FAILS the gate", raises(lambda: identity_gate(sroot, "eb9f09ff", "", quiet), RuntimeError, "IDENTITY GATE FAILED"))
        raw5 = json.loads(json.dumps(raw))
        raw5["results"].pop("S21")
        put("20261030T230000Z_1a2b3c4d", new2, raw5)
        check("identity: an arm missing from the new raw file is refused (no results.S21 object)", raises(lambda: identity_gate(sroot, "eb9f09ff", "", quiet), RuntimeError, "no results.S21 object"))
        put("20261030T230000Z_1a2b3c4d", new2, raw)
        os.remove(f"{sroot}/20261030T230000Z_1a2b3c4d/per_seq_raw.json")
        check("identity: no per_seq_raw.json in the NEW score directory is refused (the raw check did not run)",
              raises(lambda: identity_gate(sroot, "eb9f09ff", "", quiet), RuntimeError, "missing in the new score directory"))
        put("20261030T230000Z_1a2b3c4d", new2, raw)
        os.remove(f"{sroot}/20261015T195520Z_eb9f09ff/per_seq_raw.json")
        check("identity: no per_seq_raw.json in the REFERENCE score directory is refused too",
              raises(lambda: identity_gate(sroot, "eb9f09ff", "", quiet), RuntimeError, "missing in the reference score directory"))
        put("20261015T195520Z_eb9f09ff", ref, raw)
        put("20261030T230000Z_1a2b3c4d", new2, raw)
        check("identity: restored raw files -> passes again", identity_gate(sroot, "eb9f09ff", "", quiet)["ok"])
        # locating the scores
        put("20261030T231500Z_5e6f7a8b", new2, raw)
        check("identity: two candidate new scores -> STOP unless one is named",
              raises(lambda: identity_gate(sroot, "eb9f09ff", "", quiet), RuntimeError, "2 candidate")
              and identity_gate(sroot, "eb9f09ff", "20261030T231500Z_5e6f7a8b", quiet)["new_dir"] == "20261030T231500Z_5e6f7a8b"
              and raises(lambda: identity_gate(sroot, "eb9f09ff", "nope", quiet), RuntimeError, "no score.json"))
        shutil_rm = __import__("shutil").rmtree
        shutil_rm(f"{sroot}/20261030T231500Z_5e6f7a8b")
        shutil_rm(f"{sroot}/20261030T230000Z_1a2b3c4d")
        check("identity: no new score yet -> STOP", raises(lambda: identity_gate(sroot, "eb9f09ff", "", quiet), RuntimeError, "0 candidate"))
        put("20261030T230000Z_1a2b3c4d", new2, raw)
        shutil_rm(f"{sroot}/20261015T195520Z_eb9f09ff")
        check("identity: no reference score -> STOP", raises(lambda: identity_gate(sroot, "eb9f09ff", "", quiet), RuntimeError, "end with _eb9f09ff"))
        put("20261015T195520Z_eb9f09ff", ref, raw)
        check("identity: restored -> passes", identity_gate(sroot, "eb9f09ff", "", quiet)["ok"])
        # ---------------- main() end to end through a registry whose volume root is the temporary directory
        tp = _import_tol_paths()
        reg_yaml = os.environ.get("TOL82_REGISTRY_YAML") or osp.join(osp.dirname(tp.__file__), "paths_v1.yaml")
        R0 = tp.load(path=reg_yaml)
        check("registry: the registry file used by the selftest is the pinned one", R0.sha256 == PINS["REGISTRY_SHA256"])
        data = json.loads(json.dumps(R0.data))
        data["roots"].update(volume=tmp + "/vol", mount=tmp + "/mnt", mount_uri="file:" + tmp + "/mnt")
        R = tp.Registry(data, source=reg_yaml, sha256=R0.sha256)
        import shutil
        shutil.copytree(f"{tmp}/bench/edinburgh", f"{R.path('paper1.bench')}/edinburgh")
        shutil.copytree(runs, R.path("paper1.distillation_runs"))
        shutil.copytree(f"{tmp}/scores", R.path("paper1.scores"))
        out = main({"mode": "replay_compare", "core_sha256": CORE, "notebook_sha256": NB}, R=R)
        check("main: replay_compare resolves the S-Edi main / variant directories through the registry",
              out["ok"] and out["replay_dir"].endswith("/paper1_distillation/bench/edinburgh/S-Edi__protA/test/edimine_repro"))
        out = main({"mode": "pin_check", "pin": json.dumps(pin), "core_sha256": core_sha}, R=R, code_dir=code)
        check("main: pin_check resolves paper1.distillation_runs through the registry and checks the registry sha256",
              out["ok"] and out["runs_root"].endswith("/paper1_distillation/distillation_runs") and out["registry_sha256"] == PINS["REGISTRY_SHA256"])
        out = main({"mode": "identity_gate"}, R=R)
        check("main: identity_gate resolves paper1.scores through the registry", out["ok"] and out["scores_root"].endswith("/paper1_distillation/scores/edinburgh/protA/test"))
    # ---------------- audits of this notebook's own source
    src = open(osp.abspath(__file__), encoding="utf-8").read()
    head = src.split("def _synthetic_score")[0]
    check("audit: no literal Volume / dbfs / Workspace path (registry only)", not re.search(r"/Volumes/|dbfs:/|/Workspace/Users", head))
    check("audit: no secret-shaped strings, no write calls on the Volume", not re.search(r"hf_[A-Za-z0-9]{20}|secrets\.get|dbutils\.fs\.(cp|rm|put|mv)|\bshutil\.(copy|move|rmtree)", head))
    check("audit: Databricks SOURCE format", src.startswith("# Databricks notebook source"))
    check("audit: no Edinburgh clip is opened by a path built from a frame / video / annotation name",
          not re.search(r"decoded_frames|annotated|initial_prompts|\.mp4|\.jpg|output\.json", head))
    check("audit: the PINS block is a plain literal (ast.literal_eval) with the S-Edi value and the factor",
          __import__("ast").literal_eval(src[src.index("PINS = {"):src.index("}\nREGISTRY_VERSION") + 1].split("=", 1)[1].strip())["GM_REL_TOL"] == 0.1)
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
            out[name] = os.environ.get("TOL82_" + name.upper(), default)
    return out


RAW = read_widgets()
print(NB_NAME, NB_VERSION, "widgets:", json.dumps({k: (v if k != "pin" else v[:200]) for k, v in RAW.items()}, sort_keys=True))
REPORT = main(RAW)
print(json.dumps(REPORT, indent=1, sort_keys=True, default=str))

# COMMAND ----------

if IS_NOTEBOOK:
    dbutils.notebook.exit(json.dumps(REPORT, sort_keys=True, default=str))
