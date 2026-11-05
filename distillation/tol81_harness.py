"""Local end-to-end rehearsal of Tol_81's MAIN flow against a STUB Volume (tiny files; real sha-pinned tables).

usage: python -I tol81_harness.py

The stub Volume holds: the real (sha-pinned) per_frame_fidelity.csv, the two RUN_RECORDs, the three pool maps.npz, the Tol_80
index + sim_g18, a REWRITTEN unused-target manifest (stub file hashes) and 3,000 + 1,800 tiny stub files (valid 1280x720 JPEGs,
4 KB zip-magic "targets"). The notebook source is patched in memory ONLY at: VOL, the local scratch root, the target byte
size, the sample sha, the manifest sha, the selection-file sha. Everything else (selection, canonical digest, D1, hashing,
manifests, scan, publish, RUN_RECORD) runs unchanged. No test clip name is created anywhere.
"""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "Tol_81_edinburgh_select_targets.py")
EDI_TRAIN = ("2019_11_05_000002", "2019_11_11_000028", "2019_11_11_000036", "2019_11_22_000010")
EDI_VAL = ("2019_11_28_000113",)
P3 = ("2019_11_05_000009", "2019_11_11_000016", "2019_11_15_000033")
TB = 4096


def sha(b):
    return hashlib.sha256(b).hexdigest()


def jpeg(seed):
    buf = io.BytesIO()
    Image.new("RGB", (1280, 720), (seed % 256, (seed // 256) % 256, 7)).save(buf, "JPEG", quality=50)
    return buf.getvalue()


def main():
    T = tempfile.mkdtemp(prefix="tol81_harness_")
    VOL = os.path.join(T, "vol").replace("\\", "/")
    LOC = os.path.join(T, "local").replace("\\", "/")
    EM = f"{VOL}/paper1_distillation/edi_mining"
    try:
        def put(rel, data):
            p = f"{VOL}/{rel}"
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(data)
            return p

        rng = np.random.default_rng(1)
        # real pinned inputs
        for rel_src, rel_dst in (("measure/measure_v1/tables/per_frame_fidelity.csv", "paper1_distillation/edi_mining/measure_v1/tables/per_frame_fidelity.csv"),
                                 ("measure/measure_v1/RUN_RECORD.json", "paper1_distillation/edi_mining/measure_v1/RUN_RECORD.json"),
                                 ("nn/nn_teacher_v1/RUN_RECORD.json", "paper1_distillation/edi_mining/nn_teacher_v1/RUN_RECORD.json"),
                                 ("nn/nn_teacher_v1/index.csv", "paper1_distillation/edi_mining/nn_teacher_v1/index.csv"),
                                 ("nn/nn_teacher_v1/sim_g18.npy", "paper1_distillation/edi_mining/nn_teacher_v1/sim_g18.npy")):
            put(rel_dst, open(os.path.join(HERE, rel_src), "rb").read())
        for c in P3:
            put(f"paper1_distillation/edi_mining/measure_v1/clips/{c}/maps.npz",
                open(os.path.join(HERE, "measure", "measure_v1", "clips", c, "maps.npz"), "rb").read())
        # stub base files
        sample_sha = None
        n = 0
        for c in EDI_TRAIN + EDI_VAL:
            for fi in range(600):
                put(f"datasets/edinburgh_pigs/decoded_frames/{c}/video1/{fi + 1:07d}.jpg", jpeg(n))
                body = b"PK\x03\x04" + rng.bytes(TB - 4)
                put(f"paper1_distillation/teacher_targets/edinburgh/{c}/backbone_{fi:06d}.pt", body)
                if c == "2019_11_05_000002" and fi == 0:
                    sample_sha = sha(body)
                n += 1
        # stub unused targets + rewritten manifest
        man = pd.read_csv(os.path.join(HERE, "measure", "measure_v1", "teacher_targets", "manifest.csv"))
        for i, r in man.iterrows():
            ib = jpeg(100000 + i)
            fb = b"PK\x03\x04" + rng.bytes(TB - 4)
            put(f"paper1_distillation/edi_mining/teacher_targets_unused_v1/{r['image']}", ib)
            put(f"paper1_distillation/edi_mining/teacher_targets_unused_v1/{r['feature']}", fb)
            man.loc[i, ["image_bytes", "image_sha256", "feature_bytes", "feature_sha256"]] = [len(ib), sha(ib), len(fb), sha(fb)]
        mp = f"{EM}/teacher_targets_unused_v1/manifest.csv"
        man.to_csv(mp, index=False, lineterminator="\n")
        man_sha = sha(open(mp, "rb").read())

        src = open(SRC, encoding="utf-8").read()

        def rep(old, new):
            nonlocal src
            assert src.count(old) == 1, old
            src = src.replace(old, new)

        rep('VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"', f'VOL = {VOL!r}')
        rep('LOCAL = f"/local_disk0/tol81/{OUT_TAG}"', f'LOCAL = f"{LOC}/{{OUT_TAG}}"')
        rep("STORED_TARGET_BYTES = 10618052 ", f"STORED_TARGET_BYTES = {TB} ")
        rep('MAN_SHA = "3b9f6f59db4bcf6899f9f96968c31efe8925e383ec6aae7bfa8708d9c1ea7f6f"', f'MAN_SHA = "{man_sha}"')
        rep('"9706704cd9fd853b13d775a1f72f7c2e8a2c64dc7057f3ff68d2d8793673c6bc"', f'"{sample_sha}"')
        # the selection file contains the (stub) feature hashes: compute its expected sha with the same writer
        sel_expected = {"v": None}

        class FS:
            @staticmethod
            def cp(a, b):
                shutil.copyfile(a.replace("file:", "", 1), b.replace("dbfs:", "", 1))

        class NB:
            @staticmethod
            def exit(s):
                sel_expected["exit"] = s
                print("notebook.exit:", s)

        class W:
            @staticmethod
            def text(*a, **k):
                pass

            @staticmethod
            def get(name):
                return {"out_tag": "selection_v1", "threads": "8"}[name]

        class DBU:
            fs = FS
            notebook = NB
            widgets = W

        # pass 1: with the registered SEL_FILE_SHA the stub run must STOP at the selection-file assertion (stub hashes differ)
        g = {"__name__": "__main__", "dbutils": DBU, "__file__": SRC}
        msg = None
        try:
            exec(compile(src, SRC, "exec"), g)
        except AssertionError as e:
            msg = str(e)
        assert msg and "selection file sha256" in msg, "HARNESS FAIL: the run did not stop at the registered selection-file sha"
        print("pass 1 OK: stopped at", msg[:100])
        got = msg.split()[3]
        assert len(got) == 64, got
        shutil.rmtree(LOC, ignore_errors=True)
        assert not os.path.exists(f"{EM}/selection_v1"), "pass 1 must not have created the output folder"
        rep('SEL_FILE_SHA = "11e629fef34a85ad35d5c237887c43f4b8b1531907dd7cecdb2bff7d80e4fa75"', f'SEL_FILE_SHA = "{got}"')
        g = {"__name__": "__main__", "dbutils": DBU, "__file__": SRC}
        exec(compile(src, SRC, "exec"), g)
        out = f"{EM}/selection_v1"
        files = sorted(os.listdir(out))
        print("published:", files)
        assert files == ["RUN_RECORD.json", "SELECTION_SUMMARY.json", "SELECTION_mined.csv", "manifest_base.csv", "manifest_mined.csv"], files
        rr = json.load(open(f"{out}/RUN_RECORD.json"))
        for name, rec in rr["files"].items():
            b = open(f"{out}/{name}", "rb").read()
            assert sha(b) == rec["sha256"] and len(b) == rec["bytes"], name
        assert rr["counts"] == {"train": 2400, "val": 600, "extra": 627}, rr["counts"]
        assert rr["D1_identical_to_registered"] is True, rr["D1_differences"]
        assert rr["canonical_sha256"] == "53668699a31324cf009cc8c8aef50309a9413cc0316509ecbb15d1cc552b2702"
        base = pd.read_csv(f"{out}/manifest_base.csv")
        mined = pd.read_csv(f"{out}/manifest_mined.csv")
        sel = pd.read_csv(f"{out}/SELECTION_mined.csv")
        assert list(base.columns) == list(mined.columns) and len(base) == 3000 and len(mined) == 627 and len(sel) == 627
        assert base["role"].value_counts().to_dict() == {"train": 2400, "val": 600}
        assert set(mined["clip"]) == set(P3) and (mined.groupby("clip").size() == 209).all()
        assert all(str(c).startswith("2019_11_") for c in list(base["clip"]) + list(mined["clip"]))
        # every manifest path resolves under the stub Volume and hashes to the recorded value
        bad = 0
        for df in (base, mined):
            for r in df.itertuples(index=False):
                for col in ("image", "feature"):
                    b = open(f"{VOL}/{getattr(r, col)}", "rb").read()
                    bad += sha(b) != getattr(r, col + "_sha256") or len(b) != getattr(r, col + "_bytes")
        assert bad == 0, bad
        blob = b"".join(open(f"{out}/{n}", "rb").read() for n in os.listdir(out))
        assert b"2019_12_" not in blob
        # a second run must refuse the existing folder
        g = {"__name__": "__main__", "dbutils": DBU, "__file__": SRC}
        try:
            exec(compile(src, SRC, "exec"), g)
            raise SystemExit("HARNESS FAIL: an existing output folder was not refused")
        except RuntimeError as e:
            assert "exists" in str(e)
            print("second run refused the existing folder OK")
        print("HARNESS PASS")
    finally:
        shutil.rmtree(T, ignore_errors=True)


if __name__ == "__main__":
    main()
