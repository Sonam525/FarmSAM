# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N1 — Teacher backbone target extraction (train days 04-18 + 04-19)
# MAGIC
# MAGIC Extracts SAM3 `vision_encoder.backbone` targets `[1,1024,72,72]` (fp16) paired with their
# MAGIC RGB frames, subsampled across the **32.5 h** of the two training days. Feeds `Tol_02`.
# MAGIC
# MAGIC **Simpler than the Tol2 version:**
# MAGIC - No LocateAnything, no ROI polygon, no two-env split — `transformers==5.2.0` throughout.
# MAGIC - The session prompt is a **formality**: `vision_encoder.backbone` runs on the image alone,
# MAGIC   so backbone features are prompt-independent. We pass fixed dummy boxes purely to satisfy
# MAGIC   `add_inputs_to_inference_session`, and discard the tracking output. (Verified contract in
# MAGIC   CLAUDE.md §4; this is what the Tol2 run did too, via real boxes.)
# MAGIC
# MAGIC **Outputs** (under `<VOL>/Tol/`): `Frames/train/*.jpg`, `Backbone_Features/train/*.pt`,
# MAGIC `Metadata/train_manifest.csv`. Resume-safe via `.done_<clip>` markers.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "opencv-python-headless==4.11.0.86" "Pillow==11.1.0" "accelerate" "huggingface_hub" "timm"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, re, json, time, gc, glob, shutil
from datetime import datetime, timedelta
import numpy as np, cv2, torch, pandas as pd
from PIL import Image
from huggingface_hub import login

login(token=dbutils.secrets.get(scope="hf", key="token").strip().strip('"').strip("'").strip())
device = torch.device("cuda")
DTYPE = torch.bfloat16

VIDEO_DIR = "/dbfs/mnt/playbehavior/Tol/Videos"
VOL_ROOT  = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL       = osp.join(VOL_ROOT, "Tol")
FRAMES_DIR   = osp.join(TOL, "Frames", "train")
BACKBONE_DIR = osp.join(TOL, "Backbone_Features", "train")
META_DIR     = osp.join(TOL, "Metadata")
for d in (FRAMES_DIR, BACKBONE_DIR, META_DIR): os.makedirs(d, exist_ok=True)
LOCAL = "/local_disk0/tmp/tol_n1"; os.makedirs(LOCAL, exist_ok=True)

TRAIN_DAYS = ["20240418", "20240419"]
SECONDS_PER_TARGET = 30          # 32.5 h / 30 s ≈ 3.9k targets (Tol2 used 3,486)
JPEG_QUALITY = 92                # frames kept at native 2304x1296; the student resizes at load

print("device:", torch.cuda.get_device_name(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training clips
# MAGIC Three filename variants exist in this folder; also excludes the stray `Tol_ch04` 06-10 clips
# MAGIC and the duplicate `... (1).mp4`.

# COMMAND ----------

RX = re.compile(r'^(Tol_ch\d+)[-_](\d{8})[-_](\d{6})[-_](\d{6})(?: \((\d+)\))?\.mp4$')

def parse(n):
    m = RX.match(n)
    if not m: return None
    ch, d, s, e, dup = m.groups()
    ds = datetime.strptime(d+s, '%Y%m%d%H%M%S'); de = datetime.strptime(d+e, '%Y%m%d%H%M%S')
    if de < ds: de += timedelta(days=1)
    return {"ch": ch, "date": d, "start": ds, "end": de, "dup": dup is not None}

all_files = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4'))
CLIPS = []
for n in all_files:
    p = parse(n)
    if not p or p["ch"] != "Tol_ch01" or p["dup"]: continue
    if p["date"] in TRAIN_DAYS: CLIPS.append((n, p))
CLIPS.sort(key=lambda z: z[1]["start"])
tot_h = sum((p["end"]-p["start"]).total_seconds() for _, p in CLIPS)/3600
print(f"{len(CLIPS)} training clips over {TRAIN_DAYS}: {tot_h:.1f} h")
print(f"  first {CLIPS[0][0]}\n  last  {CLIPS[-1][0]}")
print(f"expected targets ≈ {tot_h*3600/SECONDS_PER_TARGET:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAM3 teacher + backbone hook

# COMMAND ----------

from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=DTYPE).to(device).eval()
print(f"SAM3 loaded | {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

class BackboneFeatureHook:
    """Capture backbone output, reshape [1,5184,1024] -> [1,1024,72,72], fp16 on CPU."""
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
    def remove(self): self.h.remove()

def prune_inference_session(session, keep_last_n=8):
    for oi in range(session.get_obj_num()):
        nc = session.output_dict_per_obj[oi].get("non_cond_frame_outputs", {})
        if len(nc) > keep_last_n:
            for k in sorted(nc.keys())[:-keep_last_n]: del nc[k]

# Dummy prompt — backbone features do NOT depend on it (see header). Four boxes inside the pen.
DUMMY_BOXES = [[600, 200, 1000, 600], [1100, 200, 1500, 600],
               [600, 700, 1000, 1100], [1100, 700, 1500, 1100]]

def sample_frames(local_path, seconds_per_frame):
    """Seek-sample a clip. Falls back to a sequential pass if the container reports no frame
    count — without that branch a bad header yields ZERO frames silently (this bit us on the
    truncated 04-18 11:33 clip, which is 0.7 MB instead of ~1.3 GB)."""
    cap = cv2.VideoCapture(local_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, round(seconds_per_frame*fps))
    if total > 0:
        for fidx in range(0, total, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, fr = cap.read()
            if not ok: continue
            yield fidx, fidx/fps, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    else:
        idx = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if idx % stride == 0:
                yield idx, idx/fps, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            idx += 1
    cap.release()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract

# COMMAND ----------

hook = BackboneFeatureHook(model)
t0 = time.time(); n_new = 0; SKIPPED = []
print(f"already on Volume: {len(glob.glob(osp.join(BACKBONE_DIR, '*.pt')))} targets")

for ci, (vname, meta) in enumerate(CLIPS):
    stem = vname[:-4].replace(" ", "_")
    done = osp.join(BACKBONE_DIR, f".done_{stem}")
    if osp.exists(done):
        print(f"  [{ci+1}/{len(CLIPS)}] {stem}: done — skip"); continue

    dst = osp.join(LOCAL, "clip.mp4")
    if osp.exists(dst): os.remove(dst)
    shutil.copy(osp.join(VIDEO_DIR, vname), dst)

    lf = osp.join(LOCAL, "feat"); li = osp.join(LOCAL, "img")
    shutil.rmtree(lf, ignore_errors=True); shutil.rmtree(li, ignore_errors=True)
    os.makedirs(lf); os.makedirs(li)

    session = processor.init_video_session(inference_device=device, dtype=DTYPE)
    n_clip = 0
    for si, (fidx, tsec, pil) in enumerate(sample_frames(dst, SECONDS_PER_TARGET)):
        inputs = processor(images=pil, device=device, return_tensors="pt")
        if si == 0:
            # fresh lists per session — the processor MUTATES obj_ids (CLAUDE.md §4)
            processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=0,
                obj_ids=list(range(len(DUMMY_BOXES))),
                input_boxes=[[list(b) for b in DUMMY_BOXES]],
                original_size=inputs.original_sizes[0])
        with torch.no_grad():
            model(inference_session=session, frame=inputs.pixel_values[0])
        feat = hook.pop()
        if feat is not None:
            base = f"{stem}_f{fidx:07d}"
            torch.save(feat, osp.join(lf, base + ".pt"))
            pil.save(osp.join(li, base + ".jpg"), quality=JPEG_QUALITY)
            n_clip += 1; n_new += 1
        del inputs
        if (si+1) % 25 == 0:
            prune_inference_session(session, 8); gc.collect(); torch.cuda.empty_cache()

    os.system(f"cp -r '{lf}/.' '{BACKBONE_DIR}/'")
    os.system(f"cp -r '{li}/.' '{FRAMES_DIR}/'")
    # Only mark done if the clip actually produced targets. A corrupt/truncated file must NOT be
    # silently recorded as complete — it would never be retried and never surface in any count.
    if n_clip > 0:
        open(done, "w").close()
    else:
        sz = osp.getsize(osp.join(VIDEO_DIR, vname)) if osp.exists(osp.join(VIDEO_DIR, vname)) else -1
        SKIPPED.append({"clip": vname, "bytes": sz})
        print(f"  ⚠ {stem}: ZERO targets — unreadable/truncated ({sz/1e6:.1f} MB). "
              f"NOT marked done; excluded from the manifest.")
    shutil.rmtree(lf, ignore_errors=True); shutil.rmtree(li, ignore_errors=True)
    os.remove(dst)
    del session; gc.collect(); torch.cuda.empty_cache()
    el = time.time()-t0
    print(f"  [{ci+1}/{len(CLIPS)}] {stem}: +{n_clip} (total {n_new}) "
          f"{n_new/max(el,1e-9):.2f} tgt/s  elapsed {el/60:.1f} min")

hook.remove()
print(f"\n✓ {n_new} new targets in {(time.time()-t0)/60:.1f} min")
if SKIPPED:
    print(f"\n⚠ {len(SKIPPED)} clip(s) produced no targets (logged to Metadata/skipped_clips.json):")
    for s in SKIPPED: print(f"    {s['clip']}  ({s['bytes']/1e6:.1f} MB)")
    with open(osp.join(META_DIR, "skipped_clips.json"), "w") as f:
        json.dump(SKIPPED, f, indent=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manifest + verification (rebuilt by scanning, so resumed runs are covered)

# COMMAND ----------

rows = []
for f in sorted(glob.glob(osp.join(BACKBONE_DIR, "*.pt"))):
    base = osp.basename(f)[:-3]
    stem, fpart = base.rsplit("_f", 1)
    img = osp.join(FRAMES_DIR, base + ".jpg")
    rows.append({"clip": stem, "frame_idx": int(fpart), "image": img, "feature": f})
man = pd.DataFrame(rows).sort_values(["clip", "frame_idx"]).reset_index(drop=True)
man_path = osp.join(META_DIR, "train_manifest.csv")
man.to_csv(man_path, index=False)
print(f"✓ {man_path}: {len(man)} pairs from {man['clip'].nunique()} clips")

missing = (~man["image"].map(osp.exists)).sum()
print(f"images missing for {missing} features (must be 0)")
assert missing == 0, "frame/feature pairing broken"

s = torch.load(man.iloc[0]["feature"], map_location="cpu")
print("target shape:", list(s.shape), s.dtype)
assert s.shape[1] == 1024 and list(s.shape)[-2:] == [72, 72], "unexpected target shape"
im = Image.open(man.iloc[0]["image"]); print("frame size:", im.size)
gb = sum(osp.getsize(f) for f in glob.glob(osp.join(BACKBONE_DIR, '*.pt')))/1e9
print(f"feature store: {gb:.1f} GB")
print("✓ ready for Tol_02_distill_student")
