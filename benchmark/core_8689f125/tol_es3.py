#!/usr/bin/env python
"""tol_es3 -- the EfficientSAM3 arm `ES-L` of the Paper 1 bench (BENCH_PROTOCOL §1; gate G2 = Tol_61_es3_port).

ES-L = SAM 3's HF tracker (transformers 5.2.0 `Sam3TrackerVideoModel`, `facebook/sam3` at the J0 pin) whose
`vision_encoder.backbone` is REPLACED by the EfficientSAM3 Stage-1 image encoder ES-TV-L (TinyViT-21M trunk at 1008^2 +
Conv1x1-BN-GELU-Conv3x3 head, bilinear to 72x72 -> [B, 1024, 72, 72]; Zeng, Jiang, Zhang 2025,
github.com/SimonZeng7108/efficientsam3). Everything after the backbone (neck = the SAM 2 branch `sam2_convs`, memory
attention / encoder, prompt encoder, mask decoder) is SAM 3's, unchanged. The encoder runs INSIDE the HF graph
(`ES3Backbone`, the DummyBackbone output contract [B, 72*72, 1024]), so the module-group timings of
`vision_encoder.backbone` cover it.

Input pixels: the frame exactly as the authors' own SAM2-style video loader prepares it (efficientsam3
sam3/sam3/model/utils/sam2_utils.py `_load_img_as_tensor` + `load_video_frames_from_jpg_images`, as called by
`Sam3TrackerPredictor.init_state`): PIL `.convert("RGB").resize((1008, 1008))` with PIL's default filter, / 255,
ImageNet mean / std (the loader defaults; NOT SAM 3's HF processor normalisation). `ES3Processor` replaces the HF
processor's `pixel_values` with that frame, so the tracker stores and the encoder consumes the authors' input; every
other processor output (original sizes, prompts, post-processing) is the HF processor's (BENCH_PROTOCOL §15, G2 entry).

Third-party material -- internal benchmarking only, NEVER redistributed:
  * weights: HF `Simon7108528/EfficientSAM3` `stage1_all_converted/efficient_sam3_tinyvit_l.pt` (model-card tag
    apache-2.0; the file is a full SAM 3 checkpoint with the ViT trunk replaced, so its other tensors are SAM 3
    parameters under the SAM License). Tol_61 keeps the file under models/baselines/third_party/efficientsam3/ and
    extracts the encoder state_dict (registry key `models.es3_trunk`, format TRUNK_FORMAT).
  * architecture: the authors' `sam3/sam3/backbones/tiny_vit.py` at ES3_COMMIT (TinyViT, (c) 2022 Microsoft, MIT
    header; it sits in the authors' fork of facebookresearch/sam3, whose LICENSE file is the SAM License). It is NOT
    copied into this module. Tol_61 stores the upstream file beside the checkpoint and `load_upstream_tinyvit` executes
    exactly that file (sha256-pinned) with timm's `register_model` neutralised, so importing it never replaces timm's
    own `tiny_vit_*` registry entries (the S-* students build `tiny_vit_21m_512` through `timm.create_model`).
  * the glue around TinyViT (trunk wrapper: patch_embed -> layers -> [B, C, s, s]; the projection head + resize) is
    re-implemented below with the authors' module names, so the state_dict keys are theirs; G2 check 1b shows the
    encoder's output equals the authors' model's on the same input.

Dependencies: stdlib + numpy at import; torch / transformers / timm / PIL lazily (selftests run without a GPU).
Public API: see `api()`; used by tol_bench_core.load_arm('ES-L') and Tol_61_es3_port only.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:  # noqa: BLE001
    torch = None
    F = None
    TORCH_OK = False

    class nn:  # noqa: N801 -- import-time stand-in; the classes below are never instantiated without torch
        Module = object

ES3_VERSION = "es3-2026-10-15.1"

# ---- provenance pins (G2 records them; load_arm('ES-L') checks them) ----
ES3_REPO_URL = "https://github.com/SimonZeng7108/efficientsam3"
ES3_COMMIT = "bd0936c788fed8d51fa799437f05abd97b401b06"            # cloned read-only 2026-10-15
# sha256 of the git blobs at ES3_COMMIT (LF line endings, as `git show <commit>:<path>` prints them)
ES3_UPSTREAM_SHA256 = {
    "sam3/sam3/backbones/tiny_vit.py": "6e1bdc938c48254268dc6ff7e0c4f8825c6479c33c5ac2af42dd9a097aff0995",
    "sam3/sam3/model/utils/sam2_utils.py": "5aafdf9865269df548dcc1615d1d1fab71b715d05d4071fdf8ede5d09405f05b",
    "sam3/sam3/model_builder.py": "3f67abb63509a41b75fdb84ab7bd9a209eee3dd02922117c8516e66b8333a63c",
}
TINYVIT_UPSTREAM = "sam3/sam3/backbones/tiny_vit.py"
ES3_HF_REPO = "Simon7108528/EfficientSAM3"
ES3_HF_FILE = "stage1_all_converted/efficient_sam3_tinyvit_l.pt"      # ES-TV-L = TinyViT-21M (README: 20.62 M)
TRUNK_PREFIX = "detector.backbone.vision_backbone.trunk.model."       # ListWrapper.model = ImageStudentEncoder
TRUNK_FORMAT = "es3_trunk_v1"
G2_FILE = "es3_g2_v1.json"                                            # paper1_distillation/protocol/<G2_FILE>
THIRD_PARTY_SUBDIR = ("third_party", "efficientsam3")                 # beside registry models.es3_trunk
REGISTRY_KEY = "models.es3_trunk"

IMAGE_SIZE = 1008
EMBED_DIM, EMBED_SIZE = 1024, 72
IMG_MEAN = (0.485, 0.456, 0.406)          # sam2_utils.load_video_frames defaults (init_state passes none)
IMG_STD = (0.229, 0.224, 0.225)
# tiny_vit.py tiny_vit_21m_224(img_size=1008, num_classes=0) (model_builder.py _create_student_vision_backbone, "21m")
TINYVIT_CFG = {"21m": dict(embed_dims=[96, 192, 384, 576], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 18],
                           window_sizes=[7, 7, 14, 7], drop_path_rate=0.2)}

DOMAIN = "zero-shot (SA-1B distilled)"
SCOPE = "S if G2"


class ES3Error(RuntimeError):
    pass


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_torch():
    if not TORCH_OK:
        raise RuntimeError("tol_es3: this call needs torch")


# =====================================================================================================================
# the upstream TinyViT (executed from the pinned file, never copied here)
# =====================================================================================================================
def load_upstream_tinyvit(path, expect_sha256=ES3_UPSTREAM_SHA256[TINYVIT_UPSTREAM]):
    """Execute the authors' tiny_vit.py (sha256-pinned) as a private module. Its `@register_model` factories would
    replace timm's own tiny_vit_* entries (the students' tiny_vit_21m_512 among them), so timm.models.register_model is
    replaced by the identity while the file executes and restored afterwards; ES-L never calls those factories
    (build_es3_encoder constructs TinyViT directly with the factory's arguments). Returns the module."""
    _require_torch()
    import timm.models as _tm
    with open(path, "rb") as f:
        b = f.read()
    sha = _sha256_bytes(b)
    if expect_sha256 and sha != expect_sha256:
        raise ES3Error(f"{path}: sha256 {sha[:16]} != pinned {expect_sha256[:16]} (efficientsam3 {ES3_COMMIT[:12]})")
    name = f"es3_upstream_tiny_vit_{sha[:12]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    orig = _tm.register_model
    _tm.register_model = lambda fn: fn
    try:
        spec.loader.exec_module(mod)
    finally:
        _tm.register_model = orig
    mod.__es3_sha256__ = sha
    sys.modules[name] = mod
    return mod


if TORCH_OK:
    class TinyViTTrunkWrapper(nn.Module):
        """Re-implementation of the authors' nested TinyViTTrunkWrapper (model_builder.py:887-902 at ES3_COMMIT):
        patch_embed -> layers -> (B, L, C) reshaped to (B, C, s, s) with s = int(sqrt(L)). Attribute names = theirs."""

        def __init__(self, model, channel_count):
            super().__init__()
            self.model = model
            self.channel_list = [channel_count]

        def forward(self, x):
            x = self.model.patch_embed(x)
            for layer in self.model.layers:
                x = layer(x)
            b, n, c = x.shape
            s = int(n ** 0.5)
            return x.view(b, s, s, c).permute(0, 3, 1, 2).contiguous()

    class ES3Encoder(nn.Module):
        """Re-implementation of the authors' ImageStudentEncoder (model_builder.py:764-787 at ES3_COMMIT): backbone ->
        head (Conv1x1 no bias, BatchNorm2d, GELU, Conv3x3) -> bilinear (align_corners=False) to embed_size if needed.
        Attribute names = theirs (state_dict keys backbone.model.* / head.{0,1,3}.*)."""

        def __init__(self, backbone, in_channels, embed_dim=EMBED_DIM, embed_size=EMBED_SIZE, img_size=IMAGE_SIZE):
            super().__init__()
            self.backbone = backbone
            self.embed_size = embed_size
            self.img_size = img_size
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )

        def forward(self, x):
            f = self.head(self.backbone(x))
            if f.shape[-1] != self.embed_size or f.shape[-2] != self.embed_size:
                f = F.interpolate(f, size=(self.embed_size, self.embed_size), mode="bilinear", align_corners=False)
            return f

    class ES3Backbone(nn.Module):
        """ES-L's vision_encoder.backbone: pixel_values [B, 3, 1008, 1008] (the authors' frame, via ES3Processor) ->
        BaseModelOutput(last_hidden_state=[B, 72*72, 1024]) -- the DummyBackbone contract (Tol_47:654-658):
        features permuted to (B, H, W, C) and flattened row-major, as Sam3VisionModel.forward views them back."""

        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, pixel_values, *args, **kwargs):
            from transformers.modeling_outputs import BaseModelOutput
            p = next(self.encoder.parameters())
            if pixel_values.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
                raise ES3Error(f"ES3Backbone: pixel_values {tuple(pixel_values.shape)} are not {IMAGE_SIZE}^2")
            f = self.encoder(pixel_values.to(device=p.device, dtype=p.dtype))
            b, c, h, w = f.shape
            return BaseModelOutput(last_hidden_state=f.permute(0, 2, 3, 1).reshape(b, h * w, c))
else:  # pragma: no cover -- import-time stand-ins
    TinyViTTrunkWrapper = ES3Encoder = ES3Backbone = None


def build_es3_encoder(tv, model_name="21m"):
    """ES-TV-L as the authors' video builder constructs it: tiny_vit_21m_224(img_size=1008, num_classes=0) wrapped by
    TinyViTTrunkWrapper and ImageStudentEncoder(in_channels=576, embed_dim=1024, embed_size=72, img_size=1008) with
    channel_list [1024]. Random weights (load_es3_encoder loads them). tv = load_upstream_tinyvit(...)."""
    _require_torch()
    m = tv.TinyViT(img_size=IMAGE_SIZE, num_classes=0, **TINYVIT_CFG[model_name])
    wrapped = TinyViTTrunkWrapper(m, m.layers[-1].dim)
    enc = ES3Encoder(backbone=wrapped, in_channels=wrapped.channel_list[0], embed_dim=EMBED_DIM,
                     embed_size=EMBED_SIZE, img_size=IMAGE_SIZE)
    enc.channel_list = [EMBED_DIM]
    return enc


def count_batchnorm(module):
    return sum(1 for m in module.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))


def stale_attention_bias_caches(module):
    """TinyViT's Attention caches `ab = attention_biases[:, attention_bias_idxs]` whenever train(False) runs; a cache
    computed BEFORE the weights were loaded (or before a dtype / device change) is stale. -> (n_attention, n_stale,
    max_abs_diff). Used by the G2 diagnosis of the authors' video builder and as a load gate here."""
    n = stale = 0
    worst = 0.0
    for m in module.modules():
        if hasattr(m, "attention_biases") and hasattr(m, "attention_bias_idxs"):
            n += 1
            ab = getattr(m, "ab", None)
            if ab is None:
                stale += 1
                worst = float("inf")
                continue
            ref = m.attention_biases[:, m.attention_bias_idxs]
            d = float((ab.detach().float().to(ref.device) - ref.detach().float()).abs().max())
            if d > 0 or ab.dtype != ref.dtype or ab.device != ref.device:
                stale += 1
                worst = max(worst, d)
    return n, stale, worst


def load_es3_encoder(state_dict, tv, *, device="cuda", dtype=None, model_name="21m"):
    """Build the encoder and load the extracted state_dict STRICTLY (every encoder tensor from the file, no unused
    key), cast to (device, dtype) and THEN eval(): TinyViT's attention-bias cache is computed by train(False) in the
    final dtype and device (the authors' video builder computes it before loading the weights -- G2 diagnosis). Gates:
    no stale attention-bias cache, BatchNorm in eval mode. -> (encoder, meta)."""
    _require_torch()
    enc = build_es3_encoder(tv, model_name)
    want = enc.state_dict()
    missing = [k for k in want if k not in state_dict]
    extra = [k for k in state_dict if k not in want]
    if missing or extra:
        raise ES3Error(f"ES3 encoder / state_dict mismatch: missing {missing[:6]} (n={len(missing)}), unused "
                       f"{extra[:6]} (n={len(extra)})")
    bad = [k for k in want if tuple(want[k].shape) != tuple(state_dict[k].shape)]
    if bad:
        raise ES3Error(f"ES3 encoder shape mismatch: {bad[:6]}")
    enc.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in state_dict.items()}, strict=True)
    dtype = dtype or torch.bfloat16
    enc = enc.to(device=device, dtype=dtype)
    enc.eval()
    enc.requires_grad_(False)
    n_att, n_stale, worst = stale_attention_bias_caches(enc)
    if n_att == 0 or n_stale:
        raise ES3Error(f"ES3 encoder: {n_stale}/{n_att} stale TinyViT attention-bias caches (max diff {worst})")
    if any(m.training for m in enc.modules()):
        raise ES3Error("ES3 encoder: a module is still in training mode")
    n_params = int(sum(p.numel() for p in enc.parameters()))
    n_trunk = int(sum(p.numel() for p in enc.backbone.parameters()))
    meta = {"n_params": n_params, "n_params_tinyvit": n_trunk, "n_params_head": n_params - n_trunk,
            "n_tensors": len(state_dict), "n_batchnorm": count_batchnorm(enc), "n_attention": n_att,
            "dtype": str(dtype).replace("torch.", ""), "model_name": model_name, "image_size": IMAGE_SIZE,
            "embed": [EMBED_DIM, EMBED_SIZE, EMBED_SIZE]}
    return enc, meta


# =====================================================================================================================
# the checkpoint -> the encoder state_dict (Tol_61), and the payload file models.es3_trunk
# =====================================================================================================================
def checkpoint_state_dict(ckpt):
    """The authors' loader (build_efficientsam3_video_model): ckpt['model'] if present, keys with 'student_trunk.'
    removed, and a bare detector state_dict ('backbone.*' without 'detector.' / 'tracker.') prefixed with 'detector.'."""
    sd = ckpt["model"] if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict) else ckpt
    cleaned = {k.replace("student_trunk.", ""): v for k, v in sd.items()}
    if not any(k.startswith(("detector.", "tracker.")) for k in cleaned) and any(k.startswith("backbone.")
                                                                                 for k in cleaned):
        cleaned = {f"detector.{k}": v for k, v in cleaned.items()}
    return cleaned


def extract_trunk(ckpt):
    """-> (encoder state_dict keyed relative to ImageStudentEncoder, report). Refuses a checkpoint that still holds a
    ViT trunk ('...trunk.' keys outside the student encoder) or has no encoder tensors."""
    raw = ckpt["model"] if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict) else ckpt
    cleaned = checkpoint_state_dict(ckpt)
    enc = {k[len(TRUNK_PREFIX):]: v for k, v in cleaned.items() if k.startswith(TRUNK_PREFIX)}
    foreign = sorted(k for k in cleaned if k.startswith("detector.backbone.vision_backbone.trunk.")
                     and not k.startswith(TRUNK_PREFIX))
    if not enc:
        raise ES3Error(f"no '{TRUNK_PREFIX}*' tensors in the checkpoint ({len(cleaned)} tensors)")
    if foreign:
        raise ES3Error(f"the checkpoint still holds non-student trunk tensors, e.g. {foreign[:4]}")
    groups = {}
    for k in cleaned:
        g = ".".join(k.split(".")[:3]) if k.startswith("detector.") else k.split(".")[0]
        groups[g] = groups.get(g, 0) + 1
    report = {"n_checkpoint_tensors": len(raw), "n_encoder_tensors": len(enc),
              "n_keys_with_student_trunk": sum(1 for k in raw if "student_trunk." in k),
              "top_level_wrapped": isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict),
              "key_groups": dict(sorted(groups.items())),
              "encoder_dtypes": sorted({str(v.dtype).replace("torch.", "") for v in enc.values()}),
              "n_encoder_elements": int(sum(int(np.prod(tuple(v.shape))) for v in enc.values()))}
    return enc, report


def trunk_payload(enc_sd, meta):
    """The models.es3_trunk file: deterministic (no timestamps), sorted keys, contiguous CPU tensors."""
    return {"format": TRUNK_FORMAT,
            "state_dict": {k: enc_sd[k].detach().cpu().contiguous().clone() for k in sorted(enc_sd)},
            "meta": json.loads(json.dumps(meta, sort_keys=True))}


def read_trunk_payload(path):
    _require_torch()
    p = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(p, dict) or p.get("format") != TRUNK_FORMAT or not isinstance(p.get("state_dict"), dict):
        raise ES3Error(f"{path}: not a {TRUNK_FORMAT} payload")
    return p


# =====================================================================================================================
# preprocessing: the authors' SAM2-task frame loader
# =====================================================================================================================
def make_es3_transform(image_size=IMAGE_SIZE, mean=IMG_MEAN, std=IMG_STD):
    """VERBATIM semantics of efficientsam3 sam2_utils._load_img_as_tensor + load_video_frames_from_jpg_images (the path
    Sam3TrackerPredictor.init_state(video_path=<jpeg dir>) takes): np.array(pil.convert('RGB').resize((S, S))) with PIL's
    default filter; uint8 required; / 255.0 (float64); HWC -> CHW; assigned into a float32 tensor; then -= mean,
    /= std in float32 (mean / std as float32 [3, 1, 1]). -> callable(pil) -> float32 [3, S, S] on the CPU."""
    _require_torch()
    m = torch.tensor(mean, dtype=torch.float32)[:, None, None]
    s = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def tf(pil):
        a = np.array(pil.convert("RGB").resize((image_size, image_size)))
        if a.dtype != np.uint8:
            raise ES3Error(f"ES3 transform: unexpected image dtype {a.dtype}")
        a = a / 255.0
        img = torch.zeros(3, image_size, image_size, dtype=torch.float32)
        img[...] = torch.from_numpy(a).permute(2, 0, 1)
        img -= m
        img /= s
        return img
    return tf


class ES3Processor:
    """SAM 3's HF processor with `pixel_values` replaced by the authors' frame (make_es3_transform): the tracker
    session stores, and ES3Backbone consumes, exactly the authors' input; original_sizes, prompt handling and
    post_process_masks are the HF processor's. Every other attribute is delegated."""

    def __init__(self, proc, transform):
        self._proc = proc
        self._tf = transform

    def __call__(self, images=None, device=None, return_tensors="pt", **kw):
        out = self._proc(images=images, device=device, return_tensors=return_tensors, **kw)
        ref = out["pixel_values"]
        px = self._tf(images).unsqueeze(0)
        if tuple(px.shape) != tuple(ref.shape):
            raise ES3Error(f"ES3Processor: authors' frame {tuple(px.shape)} vs processor {tuple(ref.shape)}")
        out["pixel_values"] = px.to(device=ref.device, dtype=ref.dtype)
        return out

    def __getattr__(self, name):
        if name.startswith("__") or name in ("_proc", "_tf"):
            raise AttributeError(name)
        return getattr(self._proc, name)


# =====================================================================================================================
# the arm: load_arm('T') -> ES-L (one function for Tol_61's route B and tol_bench_core.load_arm('ES-L'))
# =====================================================================================================================
def convert_arm_to_es3(core, arm, encoder, transform, es3_info, *, name="ES-L", log=print):
    """Turn a freshly loaded T arm (tol_bench_core.Arm, family sam3, no student) into ES-L IN PLACE: the teacher's
    vision_encoder.backbone is DELETED and replaced by ES3Backbone(encoder) (as the students' DummyBackbone replaces
    it), the processor is wrapped by ES3Processor, the backbone-capture hook moves to the new module, and arm.info
    records the ES-L facts (deterministic values only: Tol_63 resumes compare them)."""
    _require_torch()
    if arm.family != "sam3" or arm.student is not None or arm.dummy is not None:
        raise ES3Error(f"convert_arm_to_es3 needs a plain SAM 3 teacher arm, got {arm.name} ({arm.family})")
    hook = getattr(arm, "_bb_hook", None)
    if hook is not None:
        hook.remove()
        arm._bb_hook = None
    arm.model.vision_encoder.backbone = ES3Backbone(encoder)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    arm.proc = ES3Processor(arm.proc, transform)
    arm.name, arm.family = name, "es3"
    info = arm.info
    info.update({"arm": name, "family": "es3", "domain_training": DOMAIN, "scope": SCOPE,
                 "es3": json.loads(json.dumps(es3_info, sort_keys=True, default=str)),
                 "pixel_values": ("authors' SAM2-task loader: PIL resize to 1008x1008 (PIL default filter), /255, "
                                  "ImageNet mean/std (replaces the HF processor's pixel_values)"),
                 "params": core.param_table(arm.model), "weights_mem_mib": core.mem_snapshot()})
    info["backbone_capture"] = arm.attach_backbone_capture()
    log(f"arm {name}: ES-TV-L encoder in SAM 3's tracker; params {info['params']}")
    return arm


def load_es3_arm(core, *, trunk_local, tinyvit_local, trunk_sha256, tinyvit_sha256, g2_status, device="cuda",
                 hf_token=None, revisions=None, registry=None, dbutils=None, local_dir="/local_disk0/bench_models",
                 log=print):
    """The ES-L arm from LOCAL files (sha256-checked): the upstream tiny_vit.py and the models.es3_trunk payload.
    g2_status: 'PASS <protocol file> ...' (tol_bench_core.load_arm) or 'pending (Tol_61 G2 parity run)'."""
    _require_torch()
    if sha256_file(trunk_local) != trunk_sha256:
        raise ES3Error(f"{trunk_local}: sha256 != pinned {trunk_sha256[:16]}")
    tv = load_upstream_tinyvit(tinyvit_local, tinyvit_sha256)
    payload = read_trunk_payload(trunk_local)
    enc, emeta = load_es3_encoder(payload["state_dict"], tv, device=device, dtype=torch.bfloat16)
    info = {"g2": g2_status, "trunk_sha256": trunk_sha256, "trunk_format": payload["format"],
            "tinyvit_sha256": tinyvit_sha256, "efficientsam3_commit": ES3_COMMIT, "encoder": emeta,
            "source": payload["meta"].get("source"), "tol_es3": ES3_VERSION}
    arm = core.load_arm("T", device=device, hf_token=hf_token, revisions=revisions, registry=registry,
                        dbutils=dbutils, local_dir=local_dir, log=log)
    return convert_arm_to_es3(core, arm, enc, make_es3_transform(), info, log=log)


# =====================================================================================================================
# the G2 record (paper1_distillation/protocol/es3_g2_v1.json, written once by Tol_61)
# =====================================================================================================================
G2_REQUIRED = ("version", "verdict", "trunk", "tinyvit_source", "checks")


def validate_g2(rec):
    """-> list of problems with a G2 record (empty = well-formed). Does not judge the verdict."""
    probs = [f"missing '{k}'" for k in G2_REQUIRED if k not in rec]
    if probs:
        return probs
    if rec.get("version") != G2_FILE.rsplit(".", 1)[0]:
        probs.append(f"version {rec.get('version')!r}")
    for part, keys in (("trunk", ("sha256", "registry_key")), ("tinyvit_source", ("sha256", "relpath"))):
        for k in keys:
            if not rec[part].get(k):
                probs.append(f"{part}.{k} missing")
    if rec["tinyvit_source"].get("sha256") and rec["tinyvit_source"]["sha256"] != \
            ES3_UPSTREAM_SHA256[TINYVIT_UPSTREAM]:
        probs.append("tinyvit_source.sha256 != the pinned upstream file")
    rel = str(rec["tinyvit_source"].get("relpath", ""))
    if rel.startswith("/") or ".." in rel.replace("\\", "/").split("/"):
        probs.append(f"tinyvit_source.relpath {rel!r}")
    if rec["trunk"].get("registry_key") not in (None, REGISTRY_KEY):
        probs.append(f"trunk.registry_key {rec['trunk'].get('registry_key')!r}")
    if rec.get("verdict") not in ("PASS", "FAIL"):
        probs.append(f"verdict {rec.get('verdict')!r}")
    return probs


def g2_gate(rec, path=""):
    """NotImplementedError unless the G2 record is well-formed and PASS (ES-L stays R-scope otherwise, §13)."""
    if rec is None:
        raise NotImplementedError(f"ES-L: no G2 record ({path or G2_FILE}); Tol_61_es3_port has not passed G2, so "
                                  "ES-L is R-scope (BENCH_PROTOCOL §13)")
    probs = validate_g2(rec)
    if probs:
        raise NotImplementedError(f"ES-L: malformed G2 record {path}: {probs}")
    if rec["verdict"] != "PASS":
        raise NotImplementedError(f"ES-L: G2 verdict {rec['verdict']} in {path} (checks {rec.get('checks')}); "
                                  "ES-L is R-scope (BENCH_PROTOCOL §13 / §15)")
    return rec


def api():
    return ("ES3_* / TRUNK_* / IMAGE_SIZE / IMG_MEAN / IMG_STD constants; load_upstream_tinyvit(path, sha); "
            "build_es3_encoder(tv); load_es3_encoder(sd, tv, device, dtype); stale_attention_bias_caches(module); "
            "checkpoint_state_dict(ckpt); extract_trunk(ckpt); trunk_payload(sd, meta); read_trunk_payload(path); "
            "make_es3_transform(); ES3Processor(proc, tf); ES3Backbone(encoder); "
            "convert_arm_to_es3(core, arm, encoder, tf, info); load_es3_arm(core, ...); validate_g2(rec); "
            "g2_gate(rec, path)")


if __name__ == "__main__":
    print(api())
