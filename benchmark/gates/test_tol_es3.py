#!/usr/bin/env python
"""Selftests for tol_es3 (the ES-L arm). Part 1 (stdlib + numpy) always runs; part 2 needs torch + transformers 5.2.0 +
timm (a laptop venv, CPU) and the read-only clone of the authors' repository (ES3_REPO_DIR, default
<scratchpad>/es3/repo) -- skipped, with a note, when either is missing.

Part 2 checks against the AUTHORS' code text at ES3_COMMIT (git blobs, LF):
  * load_upstream_tinyvit executes the pinned tiny_vit.py without touching timm's model registry;
  * ES3Encoder / TinyViTTrunkWrapper (re-implemented here) give the SAME output as the authors' ImageStudentEncoder /
    TinyViTTrunkWrapper class text (exec'd from model_builder.py) with identical weights, at 1008^2;
  * the stale attention-bias cache of the authors' video builder (eval() before load_state_dict) is detected, and
    load_es3_encoder never leaves one;
  * make_es3_transform is bit-identical to the authors' load_video_frames(<jpeg dir>) (sam2_utils.py exec'd with a stub
    sam3.device);
  * extract_trunk / trunk_payload round-trip through torch.save / torch.load(weights_only=True);
  * convert_arm_to_es3 on a tiny real Sam3TrackerVideoModel (ViT hidden 1024 so the neck takes the ES3 features)
    tracks through tol_bench_core.track_stream with the authors' pixels in the session.
Run: python test_tol_es3.py
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import tol_es3 as es3  # noqa: E402

REPO = os.environ.get("ES3_REPO_DIR", os.path.join(SCRATCH, "es3", "repo"))


def upstream_text(rel):
    """The git blob at ES3_COMMIT (LF), or None when the clone is absent."""
    if not os.path.isdir(os.path.join(REPO, ".git")):
        return None
    r = subprocess.run(["git", "-C", REPO, "show", f"{es3.ES3_COMMIT}:{rel}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


class PureTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(es3.TRUNK_PREFIX, "detector.backbone.vision_backbone.trunk.model.")
        self.assertEqual(len(es3.ES3_COMMIT), 40)
        self.assertEqual(es3.IMAGE_SIZE // 14, 72)
        self.assertEqual(es3.G2_FILE, "es3_g2_v1.json")

    def test_upstream_pins(self):
        for rel, sha in es3.ES3_UPSTREAM_SHA256.items():
            b = upstream_text(rel)
            if b is None:
                self.skipTest("no clone of the authors' repository")
            self.assertEqual(hashlib.sha256(b).hexdigest(), sha, rel)

    def test_extract_trunk_numpy(self):
        sd = {"detector.backbone.vision_backbone.trunk.model.student_trunk.head.0.weight": np.zeros((4, 2, 1, 1)),
              "detector.backbone.vision_backbone.trunk.model.backbone.model.patch_embed.seq.0.c.weight":
                  np.zeros((3, 3, 3, 3)),
              "detector.backbone.vision_backbone.sam2_convs.0.conv_1x1.weight": np.zeros((2, 2)),
              "tracker.maskmem_backbone.x": np.zeros(3)}
        enc, rep = es3.extract_trunk({"model": sd})
        self.assertEqual(sorted(enc), ["backbone.model.patch_embed.seq.0.c.weight", "head.0.weight"])
        self.assertEqual(rep["n_keys_with_student_trunk"], 1)
        self.assertTrue(rep["top_level_wrapped"])
        bad = dict(sd)
        bad["detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight"] = np.zeros(1)
        with self.assertRaises(es3.ES3Error):
            es3.extract_trunk(bad)
        with self.assertRaises(es3.ES3Error):
            es3.extract_trunk({"tracker.x": np.zeros(1)})
        # the authors' bare-detector rule: 'backbone.*' without detector./tracker. -> prefixed
        bare = {"backbone.vision_backbone.trunk.model.head.0.weight": np.zeros(1)}
        self.assertEqual(list(es3.extract_trunk(bare)[0]), ["head.0.weight"])

    def _rec(self, **kw):
        rec = {"version": "es3_g2_v1", "verdict": "PASS", "checks": {"1": "PASS"},
               "trunk": {"sha256": "a" * 64, "registry_key": "models.es3_trunk"},
               "tinyvit_source": {"sha256": es3.ES3_UPSTREAM_SHA256[es3.TINYVIT_UPSTREAM],
                                  "relpath": "third_party/efficientsam3/src_bd0936c788fe/tiny_vit.py"}}
        rec.update(kw)
        return rec

    def test_g2_gate(self):
        self.assertEqual(es3.validate_g2(self._rec()), [])
        self.assertIs(es3.g2_gate(self._rec())["verdict"], "PASS")
        with self.assertRaises(NotImplementedError) as cm:
            es3.g2_gate(None, "x.json")
        self.assertIn("R-scope", str(cm.exception))
        with self.assertRaises(NotImplementedError):
            es3.g2_gate(self._rec(verdict="FAIL"))
        with self.assertRaises(NotImplementedError):
            es3.g2_gate(self._rec(tinyvit_source={"sha256": "b" * 64, "relpath": "x/tiny_vit.py"}))
        with self.assertRaises(NotImplementedError):
            es3.g2_gate(self._rec(tinyvit_source={"sha256": es3.ES3_UPSTREAM_SHA256[es3.TINYVIT_UPSTREAM],
                                                  "relpath": "../x/tiny_vit.py"}))
        with self.assertRaises(NotImplementedError):
            es3.g2_gate({"verdict": "PASS"})


@unittest.skipUnless(es3.TORCH_OK, "torch not installed")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.tmp = tempfile.mkdtemp(prefix="tes3_")
        b = upstream_text(es3.TINYVIT_UPSTREAM)
        cls.have_repo = b is not None
        if cls.have_repo:
            cls.tv_path = os.path.join(cls.tmp, "tiny_vit.py")
            with open(cls.tv_path, "wb") as f:
                f.write(b)

    def need_repo(self):
        if not self.have_repo:
            self.skipTest("no clone of the authors' repository")

    def test_a_upstream_import_leaves_timm_registry(self):
        self.need_repo()
        import timm
        from timm.models import _registry as reg
        before = {k: reg._model_entrypoints.get(k) for k in ("tiny_vit_21m_224", "tiny_vit_21m_512")}
        tv = es3.load_upstream_tinyvit(self.tv_path)
        after = {k: reg._model_entrypoints.get(k) for k in before}
        self.assertEqual(before, after)
        self.assertTrue(hasattr(tv, "TinyViT"))
        self.assertEqual(timm.models.register_model.__module__, "timm.models._registry")
        with self.assertRaises(es3.ES3Error):
            es3.load_upstream_tinyvit(self.tv_path, expect_sha256="0" * 64)

    def _authors_encoder_classes(self, tv):
        """The authors' ImageStudentEncoder (model_builder.py:762-787) and TinyViTTrunkWrapper (887-902, dedented)
        class TEXT at ES3_COMMIT, exec'd in a namespace with torch / nn / F."""
        mb = upstream_text("sam3/sam3/model_builder.py").decode("utf-8").split("\n")
        ise = "\n".join(mb[761:787])
        tw = textwrap.dedent("\n".join(mb[886:902]))
        self.assertIn("class ImageStudentEncoder", ise)
        self.assertTrue(tw.startswith("class TinyViTTrunkWrapper"), tw[:60])
        ns = {"torch": self.torch, "nn": self.torch.nn}
        exec(ise, ns)   # noqa: S102 -- the authors' own class text, pinned by sha256 (test_upstream_pins)
        exec(tw, ns)    # noqa: S102
        return ns["ImageStudentEncoder"], ns["TinyViTTrunkWrapper"]

    def test_b_encoder_equals_authors(self):
        self.need_repo()
        torch = self.torch
        tv = es3.load_upstream_tinyvit(self.tv_path)
        ISE, TW = self._authors_encoder_classes(tv)
        torch.manual_seed(0)
        m = tv.TinyViT(img_size=1008, num_classes=0, **es3.TINYVIT_CFG["21m"])
        w = TW(m, m.layers[-1].dim)
        ref = ISE(backbone=w, in_channels=w.channel_list[0], embed_dim=1024, embed_size=72, img_size=1008)
        g = torch.Generator().manual_seed(1)
        sd = {k: (torch.randn(v.shape, generator=g) * 0.02 if v.is_floating_point() else v)
              for k, v in ref.state_dict().items()}
        for k in sd:
            if k.endswith("running_var"):
                sd[k] = sd[k].abs() + 1.0
        ref.load_state_dict(sd)
        ref.eval()
        mine, meta = es3.load_es3_encoder(sd, tv, device="cpu", dtype=torch.float32)
        self.assertEqual(sorted(mine.state_dict()), sorted(ref.state_dict()))
        x = torch.randn(1, 3, 1008, 1008, generator=g)
        with torch.no_grad():
            a, b = ref(x), mine(x)
        self.assertEqual(tuple(b.shape), (1, 1024, 72, 72))
        self.assertTrue(torch.equal(a, b), float((a - b).abs().max()))
        self.assertEqual(meta["n_params"], sum(p.numel() for p in ref.parameters()))
        self.assertGreater(meta["n_batchnorm"], 0)
        print(f"  ES-TV-L encoder params: {meta['n_params']:,} (TinyViT {meta['n_params_tinyvit']:,}, head "
              f"{meta['n_params_head']:,})", flush=True)
        # the stale-cache defect of the authors' video builder: eval() BEFORE the weights are loaded
        stale = ISE(backbone=TW(tv.TinyViT(img_size=1008, num_classes=0, **es3.TINYVIT_CFG["21m"]), 576),
                    in_channels=576, embed_dim=1024, embed_size=72, img_size=1008)
        stale.eval()
        stale.load_state_dict(sd)
        n, s, worst = es3.stale_attention_bias_caches(stale)
        self.assertEqual((n, s), (10, 10))
        self.assertGreater(worst, 0.0)
        with torch.no_grad():
            c = stale(x)
        self.assertFalse(torch.equal(a, c))
        self.assertEqual(es3.stale_attention_bias_caches(mine)[1], 0)
        # bf16: cast happens BEFORE eval() inside load_es3_encoder -> no dtype mismatch in the attention bias add
        m16, _ = es3.load_es3_encoder(sd, tv, device="cpu", dtype=torch.bfloat16)
        with torch.no_grad():
            d = m16(x.to(torch.bfloat16)).float()
        cos = torch.nn.functional.cosine_similarity(d.flatten(), a.flatten(), dim=0)
        self.assertGreater(float(cos), 0.99)
        # the payload round trip
        enc_sd = {es3.TRUNK_PREFIX + k: v for k, v in sd.items()}
        ck = {"model": dict(enc_sd, **{"tracker.foo": torch.zeros(2)})}
        got, rep = es3.extract_trunk(ck)
        p = os.path.join(self.tmp, "trunk.pt")
        torch.save(es3.trunk_payload(got, {"source": {"x": 1}}), p)
        h1 = es3.sha256_file(p)
        torch.save(es3.trunk_payload(got, {"source": {"x": 1}}), p + "2")
        self.assertEqual(h1, es3.sha256_file(p + "2"), "the payload file is deterministic")
        back = es3.read_trunk_payload(p)
        self.assertEqual(sorted(back["state_dict"]), sorted(sd))
        self.assertTrue(all(torch.equal(back["state_dict"][k], sd[k]) for k in sd))

    def test_c_transform_equals_authors_loader(self):
        self.need_repo()
        torch = self.torch
        from PIL import Image
        src = upstream_text("sam3/sam3/model/utils/sam2_utils.py").decode("utf-8")
        stub_pkg = types.ModuleType("sam3")
        stub_dev = types.ModuleType("sam3.device")
        stub_dev.get_device = lambda: torch.device("cpu")
        saved = {k: sys.modules.get(k) for k in ("sam3", "sam3.device")}
        sys.modules["sam3"], sys.modules["sam3.device"] = stub_pkg, stub_dev
        try:
            ns = {"__name__": "es3_sam2_utils"}
            exec(compile(src, "sam2_utils.py", "exec"), ns)   # noqa: S102 -- the authors' loader, pinned by sha256
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        d = os.path.join(self.tmp, "jpg")
        os.makedirs(d, exist_ok=True)
        rng = np.random.default_rng(3)
        sizes = [(240, 320), (300, 200), (1080, 1920)]
        for k, (h, w) in enumerate(sizes):
            Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(os.path.join(d, f"{k + 1:07d}.jpg"),
                                                                                    quality=90)
        imgs, vh, vw = ns["load_video_frames"](video_path=d, image_size=1008, offload_video_to_cpu=True,
                                               compute_device=torch.device("cpu"))
        tf = es3.make_es3_transform()
        for k in range(len(sizes)):
            with Image.open(os.path.join(d, f"{k + 1:07d}.jpg")) as im:
                mine = tf(im.convert("RGB"))
            self.assertEqual(mine.dtype, torch.float32)
            self.assertTrue(torch.equal(mine, imgs[k]), float((mine - imgs[k]).abs().max()))

    def test_d_arm_conversion_tracks(self):
        torch = self.torch
        sys.path.insert(0, HERE)
        import tol_bench_core as bc
        from transformers import Sam3TrackerVideoConfig, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        from transformers.models.sam3.image_processing_sam3_fast import Sam3ImageProcessorFast
        from PIL import Image
        torch.manual_seed(0)
        vit = dict(hidden_size=1024, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                   window_size=24, global_attn_indexes=[0], model_type="sam3_vit_model")
        cfg = Sam3TrackerVideoConfig(vision_config=dict(backbone_config=vit, fpn_hidden_size=256,
                                                        model_type="sam3_vision_model"),
                                     memory_attention_num_layers=1, memory_attention_feed_forward_hidden_size=64,
                                     memory_fuser_num_layers=1)
        model = Sam3TrackerVideoModel(cfg).eval()
        model.requires_grad_(False)
        try:
            from transformers.models.sam2_video.video_processing_sam2_video import Sam2VideoVideoProcessor
            vp = Sam2VideoVideoProcessor()
        except Exception:  # noqa: BLE001
            vp = None
        proc = Sam3TrackerVideoProcessor(image_processor=Sam3ImageProcessorFast(), video_processor=vp)
        arm = bc.Arm("T", "sam3", model, proc, device="cpu", dtype=torch.float32, info={"arm": "T"})
        arm.attach_backbone_capture()
        old_hook = arm._bb_hook
        if self.have_repo:
            tv = es3.load_upstream_tinyvit(self.tv_path)
            enc = es3.build_es3_encoder(tv).eval()
        else:
            self.skipTest("no clone of the authors' repository")
        seen = {}

        def spy(m, a, o):
            seen.setdefault("px", a[0].detach().clone())
        info = {"g2": "test", "trunk_sha256": "0" * 64}
        es3.convert_arm_to_es3(bc, arm, enc, es3.make_es3_transform(), info, log=lambda *a: None)
        arm.model.vision_encoder.backbone.register_forward_hook(spy)
        self.assertEqual(arm.name, "ES-L")
        self.assertEqual(arm.family, "es3")
        self.assertIsInstance(arm.model.vision_encoder.backbone, es3.ES3Backbone)
        self.assertIsNot(arm._bb_hook, old_hook)
        self.assertEqual(arm.info["es3"]["g2"], "test")
        self.assertEqual(arm.info["params"]["vision_encoder.backbone"], sum(p.numel() for p in enc.parameters()))
        rng = np.random.default_rng(0)
        d = os.path.join(self.tmp, "trk")
        os.makedirs(d, exist_ok=True)
        files = []
        for k in range(3):
            p = os.path.join(d, f"{k + 1:07d}.jpg")
            Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(p)
            files.append(p)
        src = bc._JpegSource(files, "rand")
        prompts = [bc.Prompt(0, 0, box=(20, 20, 120, 120)), bc.Prompt(0, 1, box=(150, 60, 300, 200))]
        st = bc.track_stream(arm, src, prompts=prompts, keep_last_n=16, mot=bc.MotWriter(), log=lambda *a: None)
        self.assertEqual(st["forwards"], 3)
        self.assertEqual(st["objects"], 2)
        with Image.open(files[0]) as im:
            want = es3.make_es3_transform()(im.convert("RGB"))
        self.assertTrue(torch.equal(seen["px"][0].float(), want), "the session received the authors' frame")
        self.assertTrue(st["teardown"]["session_alive_after_teardown"] in (False, None))
        with self.assertRaises(es3.ES3Error):
            es3.convert_arm_to_es3(bc, arm, enc, es3.make_es3_transform(), info)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    unittest.main(verbosity=2)
