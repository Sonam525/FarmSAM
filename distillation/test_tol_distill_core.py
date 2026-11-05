#!/usr/bin/env python
"""Selftest of tol_distill_core (no torch, no GPU, no Volume): `python test_tol_distill_core.py`."""
import json
import os
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tol_bench_core as core      # noqa: E402
import tol_distill_core as dc      # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# the 35 PigTrack TRAIN sequences as listed on the Volume (seqinfo.ini, 2026-10-14): seq, fps, seqLength, w, h, pen, night
TRAIN_SEQINFO = """
0003 10 246 1280 800 2 1|0009 10 151 1280 800 1 0|0012 10 136 1280 800 6 1|0013 10 406 1280 800 7 0
0014 15 275 1280 800 4 0|0016 10 431 1280 800 6 1|0019 10 97 1280 800 2 1|0023 10 560 1920 1080 3 1
0024 15 450 1280 800 8 1|0025 15 276 1280 800 5 0|0026 10 334 1280 800 2 1|0027 10 393 1280 800 7 0
0028 10 60 1280 800 4 0|0035 10 189 1280 800 9 0|0036 15 195 1280 800 2 1|0039 10 140 1920 1080 3 0
0041 10 292 1920 1080 3 0|0043 10 110 1920 1080 3 0|0044 10 432 1280 800 2 0|0045 15 450 1280 800 8 0
0046 10 251 1280 800 7 0|0050 10 306 1280 800 2 1|0052 15 195 1280 800 8 1|0053 10 330 1920 1080 3 0
0057 10 300 1920 1080 3 0|0059 10 80 1920 1080 3 1|0061 10 343 1280 800 2 1|0063 10 224 1280 800 1 0
0064 10 352 1280 800 4 1|0070 10 291 1280 800 6 0|0073 15 450 1280 800 8 0|0075 10 167 1280 800 9 0
0076 15 361 1280 800 4 0|0079 10 298 1280 800 7 0|0080 10 390 1280 800 6 0
"""
SPLIT_TEST_SEQS = ("0002 0004 0005 0010 0011 0017 0018 0021 0030 0038 0040 0042 0049 0054 0055 0058 0060 0065 0067 "
                   "0068 0071 0072 0074 0077 0078").split()


def real_infos():
    out = {}
    for item in re.split(r"[|\n]", TRAIN_SEQINFO):
        p = item.split()
        if not p:
            continue
        s = f"pigtrack{p[0]}"
        out[s] = {"fps": float(p[1]), "n": int(p[2]), "width": int(p[3]), "height": int(p[4]), "pen": p[5],
                  "night": int(p[6])}
    return out


RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{detail}]"))


def raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True


def t_parsing():
    txt = "pigtrack0001 val\npigtrack0002 test\n\npigtrack0003 train\n"
    d = dc.parse_split_text(txt)
    check("split: parse", d == {"pigtrack0001": "val", "pigtrack0002": "test", "pigtrack0003": "train"})
    check("split: duplicate refused", raises(lambda: dc.parse_split_text("pigtrack0001 val\npigtrack0001 val\n")))
    check("split: bad split refused", raises(lambda: dc.parse_split_text("pigtrack0001 dev\n")))
    check("split: bad line refused", raises(lambda: dc.parse_split_text("pigtrack0001\n")))
    check("seqmap: parse", dc.parse_seqmap_text("name\npigtrack0003\npigtrack0009\n") == ["pigtrack0003", "pigtrack0009"])
    check("seqmap: header required", raises(lambda: dc.parse_seqmap_text("pigtrack0003\n")))
    ini = ("[Sequence]\nname=pigtrack0003\nimDir=img1\nframeRate=10\nseqLength=246\nimWidth=1280\nimHeight=800\n"
           "imExt=.jpg\npenID=2\nnight=1\n")
    i = dc.parse_seqinfo_text(ini)
    check("seqinfo: parse", i["fps"] == 10.0 and i["width"] == 1280 and i["night"] == 1 and i["pen"] == "2"
          and i["seq_length_ini"] == 246, str(i))
    names = [f"{k:08d}.jpg" for k in range(1, 245)] + [".00000001.jpg.crc"]
    si = dc.seq_info(ini, names, "pigtrack0003")
    check("seq_info: n = image count, seqLength mismatch is a warning", si["n"] == 244 and si["warning"]
          and si["ignored_files"] == [".00000001.jpg.crc"], str(si.get("warning")))
    check("seq_info: name mismatch refused", raises(lambda: dc.seq_info(ini, names, "pigtrack0004")))
    check("count_images: gap refused", raises(lambda: dc.count_images(["00000001.jpg", "00000003.jpg"])))
    check("count_images: empty -> seq_info refuses", raises(lambda: dc.seq_info(ini, [], "pigtrack0003")))


def t_selection():
    a = dc.apportion({"a": 10, "b": 10, "c": 10}, 10)
    check("apportion: ties -> sorted order", a == {"a": 4, "b": 3, "c": 3}, str(a))
    a = dc.apportion({"a": 1, "b": 99}, 50)
    check("apportion: proportional", a == {"a": 1, "b": 49} or a == {"a": 0, "b": 50}, str(a))
    check("apportion: sum and bounds", all(sum(dc.apportion({"x": n, "y": m, "z": 7}, t).values()) == t
                                           for n, m, t in ((5, 9, 3), (100, 1, 50), (3, 3, 13))))
    check("apportion: too many refused", raises(lambda: dc.apportion({"a": 3}, 4)))
    check("apportion: all", dc.apportion({"a": 3, "b": 2}, 5) == {"a": 3, "b": 2})
    ok = True
    for N in (1, 2, 7, 60, 97, 560):
        for n in sorted({0, 1, max(1, N // 3), N}):
            idx = dc.even_indices(N, n)
            ok &= len(idx) == n and len(set(idx)) == n and all(0 <= x < N for x in idx)
    check("even_indices: unique, in range, all sizes", ok)
    check("even_indices: N=10 n=3", dc.even_indices(10, 3) == [1, 5, 8], str(dc.even_indices(10, 3)))
    check("even_indices: all frames", dc.even_indices(5, 5) == [0, 1, 2, 3, 4])
    infos = real_infos()
    check("real train split: 35 sequences, 9,961 frames by seqLength (image counts: 9,959 per the paper)",
          len(infos) == 35 and sum(i["n"] for i in infos.values()) == 9961, str(sum(i["n"] for i in infos.values())))
    plan = dc.selection_plan(infos, 3000)
    tot = sum(len(v) for v in plan.values())
    worst = max(abs(len(plan[s]) - 3000 * infos[s]["n"] / 9961) for s in infos)
    check("plan: 3,000 frames, every sequence within 1 of its quota", tot == 3000 and worst < 1.0, f"{tot} {worst}")
    gaps = [b - a for s in plan for a, b in zip(plan[s], plan[s][1:])]
    check("plan: frames evenly spaced (stride 3-4 at 30 %)", set(gaps) <= {3, 4}, str(sorted(set(gaps))))
    check("plan: digest deterministic", dc.plan_digest(plan) == dc.plan_digest(dc.selection_plan(infos, 3000)))
    check("plan: digest sensitive", dc.plan_digest(plan) != dc.plan_digest(dc.selection_plan(infos, 2999)))
    print("     plan per seq:", {s[-4:]: len(v) for s, v in plan.items()})


def t_val_rule():
    infos = real_infos()
    vs, work = dc.choose_val_sequences(infos)
    check("val rule on the real train split -> 0041, 0050, 0064, 0070",
          vs == ["pigtrack0041", "pigtrack0050", "pigtrack0064", "pigtrack0070"], str(vs))
    picks = [p["pick"] for p in work["picks"]]
    check("val rule: pick order day-std, day-HD, night-std, night-std",
          picks == ["pigtrack0070", "pigtrack0041", "pigtrack0050", "pigtrack0064"], str(picks))
    check("val rule: pens distinct", len({infos[s]["pen"] for s in vs}) == 4)
    check("val rule: 2 day + 2 night", sum(infos[s]["night"] for s in vs) == 2)
    check("val rule: second night pick skipped pen-2 / pen-6 sequences closer to the median",
          work["picks"][3]["skipped_pen_taken"] == ["pigtrack0026", "pigtrack0061"],
          str(work["picks"][3]["skipped_pen_taken"]))
    # robustness: 0050's image count may differ from seqLength by 1-2 (census warning); the picks must not change
    for d in (-2, -1, 1, 2):
        inf2 = json.loads(json.dumps(infos))
        inf2["pigtrack0050"]["n"] += d
        check(f"val rule stable under 0050 n{d:+d}", dc.choose_val_sequences(inf2)[0] == vs)
    plan = dc.selection_plan(infos, 3000)
    nval = sum(len(plan[s]) for s in vs)
    check("val share of 3,000 targets is 12-13 %", 360 <= nval <= 390, str(nval))
    print(f"     val targets {nval}, train targets {3000 - nval}; val frames {sum(infos[s]['n'] for s in vs)}")
    check("val override: needs 4 train sequences",
          raises(lambda: dc.check_val_override(["pigtrack0003"], set(infos)))
          and raises(lambda: dc.check_val_override(["pigtrack0002", "pigtrack0003", "pigtrack0009", "pigtrack0012"],
                                                   set(infos)))
          and dc.check_val_override(["pigtrack0009", "pigtrack0003", "pigtrack0012", "pigtrack0013"], set(infos))
          == ["pigtrack0003", "pigtrack0009", "pigtrack0012", "pigtrack0013"])
    check("val rule: empty cell raises", raises(lambda: dc.choose_val_sequences(
        {s: i for s, i in infos.items() if i["width"] == 1280})))


def t_names_boxes():
    check("target_basename", dc.target_basename("pigtrack0003", 7) == "pigtrack0003_f000007")
    check("target_basename refuses junk", raises(lambda: dc.target_basename("../x", 1)))
    check("src_image_name = MOT frame fi + 1", dc.src_image_name(0) == "00000001.jpg" and dc.src_image_name(244) ==
          "00000245.jpg")
    b1 = dc.dummy_boxes(1280, 800)
    b2 = dc.dummy_boxes(1920, 1080)
    check("dummy boxes inside both geometries, fresh lists", len(b1) == 4 and b1 is not dc.dummy_boxes(1280, 800)
          and all(0 <= b[0] < b[2] <= 1920 and 0 <= b[1] < b[3] <= 1080 for b in b2), str(b1))


def _fake_records(plan, infos):
    recs = {}
    for s, fis in plan.items():
        if not fis:
            continue
        files = []
        for fi in fis:
            base = dc.target_basename(s, fi)
            files.append({"fi": fi, "src_image": dc.src_image_name(fi), "image": base + ".jpg", "feature": base + ".pt",
                          "image_bytes": 1000 + fi, "image_sha256": "a" * 64, "feature_bytes": 10617646,
                          "feature_sha256": "b" * 64, "feat_mean": 0.01, "feat_std": 1.4})
        recs[s] = {"seq": s, "dir": f"seqs/{s}", "files": files}
    return recs


def t_manifest():
    infos = real_infos()
    plan = dc.selection_plan(infos, 3000)
    vs, _ = dc.choose_val_sequences(infos)
    rows = dc.manifest_rows(_fake_records(plan, infos), infos, vs)
    txt = dc.manifest_csv(rows)
    back = dc.read_manifest_csv(txt)
    check("manifest: csv round trip", back == rows, str(back[0]) + " vs " + str(rows[0]))
    summ = dc.validate_manifest(back, plan, vs, 3000)
    check("manifest: validates against plan / val / count", summ["rows"] == 3000 and summ["val_seqs"] == vs
          and summ["train"] + summ["val"] == 3000, str(summ))
    check("manifest: relative paths only", all(not r["image"].startswith("/") and r["feature"].startswith("seqs/")
                                               for r in rows))
    bad = [dict(r) for r in rows]
    bad[5]["feature"] = "/Volumes/x.pt"
    check("manifest: absolute path refused", raises(lambda: dc.validate_manifest(bad)))
    bad = [dict(r) for r in rows] + [dict(rows[0])]
    check("manifest: duplicate refused", raises(lambda: dc.validate_manifest(bad)))
    bad = [dict(r) for r in rows[1:]]
    check("manifest: missing frame vs plan refused", raises(lambda: dc.validate_manifest(bad, plan)))
    check("manifest: wrong val set refused", raises(lambda: dc.validate_manifest(rows, plan, vs[:3] + ["pigtrack0003"])))
    bad = [dict(r) for r in rows]
    bad[0]["feat_std"] = 0.0
    check("manifest: collapsed feature refused", raises(lambda: dc.validate_manifest(bad)))
    check("manifest: wrong columns refused", raises(lambda: dc.read_manifest_csv("a,b\n1,2\n")))


def t_publish():
    with tempfile.TemporaryDirectory() as td:
        loc = os.path.join(td, "local", "pigtrack0003")
        os.makedirs(os.path.join(loc, "features"))
        os.makedirs(os.path.join(loc, "frames"))
        for k in range(3):
            with open(os.path.join(loc, "features", f"f{k}.pt"), "wb") as f:
                f.write(os.urandom(100 + k))
            with open(os.path.join(loc, "frames", f"f{k}.jpg"), "wb") as f:
                f.write(b"jpg" * (k + 1))
        vol = os.path.join(td, "vol", "paper1_distillation", "teacher_targets", "pigtrack_train", "seqs")
        r1 = dc.publish_tree_attempt(loc, vol, "pigtrack0003", log=lambda *a: None)
        check("publish tree: new", r1["mode"] == "new" and r1["files"] == 6 and r1["name"] == "pigtrack0003", str(r1))
        r2 = dc.publish_tree_attempt(loc, vol, "pigtrack0003", log=lambda *a: None)
        check("publish tree: rerun identical", r2["mode"] == "identical" and r2["name"] == "pigtrack0003", str(r2))
        os.remove(os.path.join(vol, "pigtrack0003", "frames", "f2.jpg"))
        r3 = dc.publish_tree_attempt(loc, vol, "pigtrack0003", log=lambda *a: None)
        check("publish tree: partial copy completed in place", r3["mode"] == "resume" and r3["name"] == "pigtrack0003")
        with open(os.path.join(loc, "features", "f1.pt"), "wb") as f:    # re-extracted, not bit-identical
            f.write(os.urandom(101))
        r4 = dc.publish_tree_attempt(loc, vol, "pigtrack0003", log=lambda *a: None)
        check("publish tree: differing content -> next attempt dir, nothing overwritten",
              r4["name"] == "pigtrack0003__a2" and r4["mode"] == "new"
              and dc.list_tree(os.path.join(vol, "pigtrack0003")) == dc.list_tree(loc), str(r4))
        os.makedirs(os.path.join(vol, "pigtrack0009"))
        with open(os.path.join(vol, "pigtrack0009", "foreign.txt"), "w") as f:
            f.write("x")
        loc9 = os.path.join(td, "local", "pigtrack0009")
        os.makedirs(loc9)
        with open(os.path.join(loc9, "a.pt"), "wb") as f:
            f.write(b"1")
        r5 = dc.publish_tree_attempt(loc9, vol, "pigtrack0009", log=lambda *a: None)
        check("publish tree: foreign file -> next attempt dir", r5["name"] == "pigtrack0009__a2"
              and os.path.isfile(os.path.join(vol, "pigtrack0009", "foreign.txt")))
        check("publish tree: refuses targets outside paper1_distillation on a Volume path",
              raises(lambda: dc.publish_tree_attempt(loc9, "/Volumes/x/models", "pigtrack0009"), PermissionError))
        check("publish tree: empty local dir refused", raises(lambda: dc.publish_tree_attempt(
            os.path.join(td, "nothing"), vol, "pigtrack0010")))
        # guarded model publish
        mdir = os.path.join(td, "vol", "models", "sam3_student_tinyvit", "pigtrack_v1")
        src = os.path.join(td, "best.pt")
        with open(src, "wb") as f:
            f.write(b"weights")
        dst = os.path.join(mdir, "best_backbone_student_pigtrack.pt")
        check("model publish: copied", dc.publish_file_guarded(src, dst, mdir) == "copied")
        check("model publish: identical rerun", dc.publish_file_guarded(src, dst, mdir) == "identical")
        with open(src, "wb") as f:
            f.write(b"other")
        check("model publish: never overwrites", raises(lambda: dc.publish_file_guarded(src, dst, mdir),
                                                        FileExistsError))
        check("model publish: refuses outside the allowed dir",
              raises(lambda: dc.publish_file_guarded(src, os.path.join(td, "vol", "x.pt"), mdir), PermissionError)
              and raises(lambda: dc.publish_file_guarded(src, os.path.join(mdir, "sub", "x.pt"), mdir), PermissionError))
        sec = os.path.join(td, "leak.json")
        with open(sec, "w") as f:
            f.write('{"t": "hf_' + "A" * 34 + '"}')
        check("model publish: secret-shaped text refused",
              raises(lambda: dc.publish_file_guarded(sec, os.path.join(mdir, "leak.json"), mdir), PermissionError))
        # parallel local copy with verification
        jobs = [(os.path.join(loc, "frames", f"f{k}.jpg"), os.path.join(td, "stage", f"f{k}.jpg"), 3 * (k + 1),
                 core.sha256_file(os.path.join(loc, "frames", f"f{k}.jpg"))) for k in range(3)]
        st = dc.copy_files_parallel(jobs, threads=3)
        st2 = dc.copy_files_parallel(jobs, threads=3)
        check("copy_files_parallel: copies, then skips verified files", st["copied"] == 3 and st2["copied"] == 0,
              f"{st} {st2}")
        st3 = dc.copy_files_parallel([(jobs[0][0], os.path.join(td, "stage2", "f0.jpg"), "src", None)], threads=1)
        check("copy_files_parallel: size 'src' = the source size", st3["copied"] == 1 and st3["bytes_copied"] == 3)
        badjob = [(jobs[0][0], os.path.join(td, "stage", "g.jpg"), 999, None)]
        check("copy_files_parallel: size mismatch raises", raises(lambda: dc.copy_files_parallel(badjob)))
        check("copy_files_parallel: refuses a Volume destination",
              raises(lambda: dc.copy_files_parallel([(jobs[0][0], "/Volumes/x/y.jpg", None, None)]), PermissionError))


def t_fidelity():
    rng = np.random.default_rng(0)
    t = rng.normal(0, 1.4, size=(1, 1024, 8, 8))
    m = dc.fidelity_metrics(t, t)
    check("fidelity: identical", abs(m["cos_patch"] - 1) < 1e-12 and abs(m["scale_ratio"] - 1) < 1e-12
          and m["mse"] == 0.0)
    m = dc.fidelity_metrics(2 * t, t)
    check("fidelity: scaled copy -> cos 1, scale 2", abs(m["cos_patch"] - 1) < 1e-12 and abs(m["scale_ratio"] - 2) < 1e-9)
    s = rng.normal(0, 1.0, size=t.shape)
    m = dc.fidelity_metrics(s, t)
    S, T = s[0].reshape(1024, -1), t[0].reshape(1024, -1)
    ref = np.mean((S * T).sum(0) / np.linalg.norm(S, axis=0) / np.linalg.norm(T, axis=0))
    check("fidelity: per-patch cosine = mean of column cosines", abs(m["cos_patch"] - ref) < 1e-9)
    check("fidelity: shape mismatch refused", raises(lambda: dc.fidelity_metrics(s[:, :10], t)))
    rows = [{"seq": "a", "night": 0, "fi": 1, "S-PT": {"fp32": m, "bf16": m}},
            {"seq": "b", "night": 1, "fi": 2, "S-PT": {"fp32": m, "bf16": m}}]
    g = dc.fidelity_groups(rows)
    check("fidelity groups: overall / day / night / per seq", g["S-PT"]["fp32"]["overall"]["n"] == 2
          and g["S-PT"]["bf16"]["night"]["n"] == 1 and set(g["S-PT"]["fp32"]["per_seq"]) == {"a", "b"})
    # VERBATIM check against Tol_68 (the S-Edi audit): the function bodies must be byte-identical
    t68 = os.path.join(HERE, "Tol_68_edinburgh_audit.py")
    if os.path.isfile(t68):
        with open(t68, encoding="utf-8") as f:
            a = f.read()
        with open(os.path.join(HERE, "tol_distill_core.py"), encoding="utf-8") as f:
            b = f.read()

        def block(text, start, end):
            i = text.index(start)
            return text[i:text.index(end, i)]
        same = all(block(a, s, e) == block(b, s, e) for s, e in (
            ("FID_KEYS = (", "COS_SIM_EPS"), ("COS_LOSS_EPS = ", "NORM_EPS = 1e-12"),
            ("def summarize(rows, keys=FID_KEYS):", "    return out\n")))
        same &= block(a, "def _ustd(a):", "\n\n\ndef cosine_global") == block(b, "def _ustd(a):", "\n\n\ndef summarize")
        check("fidelity_metrics / summarize / constants are VERBATIM Tol_68", same)
    else:
        print("SKIP verbatim check: Tol_68_edinburgh_audit.py not beside this test")


def t_runs():
    check("run tag", dc.check_run_tag("pigtrack_v1") == "pigtrack_v1" and raises(lambda: dc.check_run_tag("../x"))
          and raises(lambda: dc.check_run_tag("a__b")) and raises(lambda: dc.check_run_tag("")))
    check("smoke tag", dc.is_smoke("smoke_pt1") and not dc.is_smoke("pigtrack_v1"))
    check("state names", dc.state_name(7, "a1b2c3") == "state_ep007_a1b2c3.pt"
          and raises(lambda: dc.state_name(7, "xyz")) and dc.state_marker("state_ep007_a1b2c3.pt") ==
          "state_ep007_a1b2c3.json")
    names = ["state_ep010_aaaaaa.pt", "state_ep010_aaaaaa.json", "state_ep020_bbbbbb.pt",      # ep 20 has no marker
             "state_ep002_cccccc.pt", "state_ep002_cccccc.json", "x.pt", "state_ep10.pt", "state_ep030_dddddd.pt",
             "state_ep030_dddddd.json"]
    check("latest_state: only states with a marker", dc.latest_state(names) == (30, "state_ep030_dddddd.pt"))
    sizes = {"state_ep030_dddddd.pt": 5, "state_ep010_aaaaaa.pt": 9}
    marks = {"state_ep030_dddddd.json": {"bytes": 6}, "state_ep010_aaaaaa.json": {"bytes": 9},
             "state_ep002_cccccc.json": {"bytes": 1}}
    check("latest_state: a truncated state (size != marker) is skipped",
          dc.latest_state(names, sizes, marks) == (10, "state_ep010_aaaaaa.pt"))
    check("latest_state: none", dc.latest_state([]) == (0, None) and dc.latest_state(["state_ep020_bbbbbb.pt"]) ==
          (0, None))
    check("json_dumps handles numpy", json.loads(dc.json_dumps({"a": np.float32(1.5), "b": np.int64(3)})) ==
          {"a": 1.5, "b": 3})


if __name__ == "__main__":
    for fn in (t_parsing, t_selection, t_val_rule, t_names_boxes, t_manifest, t_publish, t_fidelity, t_runs):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} raised", False, repr(e))
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\ntol_distill_core selftest: {len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)
