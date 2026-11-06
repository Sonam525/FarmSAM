# Databricks notebook source
# MAGIC %md
# MAGIC # Mine night targets for the calf student
# MAGIC
# MAGIC Runs SAM 3 and the current student on candidate frames every 5 s across both training days, ranks them by per-position
# MAGIC feature cosine (hardest first) and selects `n_targets` under a night quota and a per-clip cap. SAM 3 features of the
# MAGIC selected frames are stored in the format of `extract_teacher_targets`, so `distill_student` can train on the union.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "transformers==5.2.0" "opencv-python-headless==4.11.0.86" "Pillow==11.1.0" "timm" "huggingface_hub" "accelerate" "pandas"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, re, json, time, gc, glob, shutil
from datetime import datetime, timedelta
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import timm
from timm.data import resolve_data_config
from torchvision import transforms
from huggingface_hub import login
import pandas as pd

dbutils.widgets.text("days", "20240418,20240419")
dbutils.widgets.text("cand_every_sec", "5.0")
dbutils.widgets.text("n_targets", "8000")
dbutils.widgets.text("night_quota_frac", "0.45")
dbutils.widgets.text("per_clip_cap", "400")
dbutils.widgets.text("max_clips", "0")

DAYS = [d.strip() for d in dbutils.widgets.get("days").split(",") if d.strip()]
CAND_SEC = float(dbutils.widgets.get("cand_every_sec") or 5.0)
N_TARGETS = int(dbutils.widgets.get("n_targets") or 8000)
NIGHT_FRAC = float(dbutils.widgets.get("night_quota_frac") or 0.45)
CLIP_CAP = int(dbutils.widgets.get("per_clip_cap") or 400)
MAX_CLIPS = int(dbutils.widgets.get("max_clips") or 0)

VIDEO_DIR = "/dbfs/mnt/your_videos/Tol/Videos"
DBFS_VIDEO = "dbfs:/mnt/your_videos/Tol/Videos"
VOL = "/Volumes/your_catalog/your_schema/your_volume/farmsam"
TOL = osp.join(VOL, "Tol")
BK_V2 = osp.join(TOL, "Backbone_Features", "train_v2")
FR_V2 = osp.join(TOL, "Frames", "train_v2")
META = osp.join(TOL, "Metadata")
LOCAL = "/local_disk0/tmp/mine"
LF, LI = osp.join(LOCAL, "feat"), osp.join(LOCAL, "img")
for d in (BK_V2, FR_V2, META, LOCAL, LF, LI): os.makedirs(d, exist_ok=True)

login(token=dbutils.secrets.get(scope="hf", key="token").strip().strip('"').strip("'").strip())
device = torch.device("cuda")
DTYPE = torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16
print(f"GPU {torch.cuda.get_device_name(0)} | {len(DAYS)} days | cand every {CAND_SEC}s | "
      f"select {N_TARGETS} (night>={NIGHT_FRAC:.0%}, <= {CLIP_CAP}/clip)")

RX = re.compile(r'^(Tol_ch\d+)[-_](\d{8})[-_](\d{6})[-_](\d{6})(?: \((\d+)\))?\.mp4$')
CORRUPT = {"Tol_ch01_20240418-113301-121506.mp4"}
CLIPS = []
for n in sorted(os.listdir(VIDEO_DIR)):
    m = RX.match(n)
    if not m or m.group(5) is not None: continue
    if m.group(1) != "Tol_ch01" or m.group(2) not in DAYS: continue
    if n in CORRUPT: continue
    CLIPS.append((n, datetime.strptime(m.group(2)+m.group(3), "%Y%m%d%H%M%S")))
CLIPS.sort(key=lambda z: z[1])
if MAX_CLIPS: CLIPS = CLIPS[:MAX_CLIPS]
print(f"{len(CLIPS)} training clips")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Models: teacher backbone (extract_teacher_targets's exact hook) + current student (fp32)

# COMMAND ----------

from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
teacher = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=DTYPE).to(device).eval()

class BackboneFeatureHook:
    def __init__(self, m):
        self.captured = None
        self.h = m.vision_encoder.backbone.register_forward_hook(self._hook)
    def _hook(self, mod, inp, out):
        if hasattr(out, "last_hidden_state"):
            lhs = out.last_hidden_state
            B, Nn, C = lhs.shape
            H = W = int(Nn ** 0.5)
            self.captured = lhs.permute(0, 2, 1).reshape(B, C, H, W).detach()
        elif isinstance(out, torch.Tensor):
            self.captured = out.detach()
    def pop(self):
        f, self.captured = self.captured, None
        return f
    def remove(self): self.h.remove()

def prune_session(session, keep=8):
    for oi in range(session.get_obj_num()):
        nc = session.output_dict_per_obj[oi].get("non_cond_frame_outputs", {})
        if len(nc) > keep:
            for k in sorted(nc.keys())[:-keep]: del nc[k]

DUMMY_BOXES = [[600, 200, 1000, 600], [1100, 200, 1500, 600],
               [600, 700, 1000, 1100], [1100, 700, 1500, 1100]]

def replace_bn_with_gn(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nc = child.num_features; ng = min(32, nc)
            while nc % ng != 0: ng -= 1
            gn = nn.GroupNorm(ng, nc, affine=child.affine)
            if child.affine and child.weight is not None:
                gn.weight.data.copy_(child.weight.data); gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else: replace_bn_with_gn(child)

class StudentEncoderTinyViT(nn.Module):
    def __init__(self, out_channels=1024):
        super().__init__()
        self.backbone = timm.create_model('tiny_vit_21m_512', pretrained=False,
                                          features_only=True, out_indices=(1, 2, 3),
                                          drop_path_rate=0.0)
        replace_bn_with_gn(self.backbone)
        def lat(c): return nn.Sequential(nn.Conv2d(c, 256, 1, bias=False),
                                         nn.GroupNorm(32, 256), nn.GELU())
        self.lateral_s1, self.lateral_s2, self.lateral_s3 = lat(192), lat(384), lat(576)
        self.channel_proj = nn.Sequential(nn.Conv2d(768, out_channels, 1, bias=False),
                                          nn.GroupNorm(32, out_channels), nn.GELU())
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels))
        self.output_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.output_bias  = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
    def forward(self, x):
        s = self.backbone(x)
        f1 = F.interpolate(self.lateral_s1(s[0]), size=(72, 72), mode='area')
        f2 = F.interpolate(self.lateral_s2(s[1]), size=(72, 72), mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.lateral_s3(s[2]), size=(72, 72), mode='bilinear', align_corners=False)
        o = self.channel_proj(torch.cat([f1, f2, f3], 1))
        o = o + self.spatial_refine(o)
        return o * self.output_scale + self.output_bias

ck = torch.load(osp.join(TOL, "Checkpoints", "best_backbone_student.pt"), map_location='cpu')
student = StudentEncoderTinyViT()
sd = student.state_dict(); sd.update(ck['model_state_dict'])
for k, v in ck['ema_shadow'].items(): sd[k] = v
student.load_state_dict(sd); student = student.to(device).eval()
_t = timm.create_model('tiny_vit_21m_512', pretrained=False)
_cfg = resolve_data_config(_t.pretrained_cfg); del _t
STUDENT_TF = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                 transforms.Normalize(mean=_cfg['mean'], std=_cfg['std'])])
hook = BackboneFeatureHook(teacher)
print("teacher + student ready")

def per_position_cosine(t_feat, s_feat):
    """mean over 72x72 positions of cosine between 1024-d vectors (the recipe's fidelity metric)."""
    t = t_feat.float().flatten(2).squeeze(0)      # [1024, 5184]
    s = s_feat.float().flatten(2).squeeze(0)
    t = F.normalize(t, dim=0); s = F.normalize(s, dim=0)
    return float((t * s).sum(0).mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single pass: stage clip -> per candidate: teacher feat (kept local) + student cosine

# COMMAND ----------

def is_night(w): return w.hour >= 21 or w.hour < 6

CAND = []                       # {"base", "clip", "fidx", "wall", "night", "cos"}
t0 = time.time()
for ci, (vname, wall0) in enumerate(CLIPS):
    stem = vname[:-4].replace(" ", "_")
    dst = osp.join(LOCAL, "cur.mp4")
    if osp.exists(dst): os.remove(dst)
    try:
        dbutils.fs.cp(f"{DBFS_VIDEO}/{vname}", "file:" + dst)
    except Exception:
        shutil.copy(osp.join(VIDEO_DIR, vname), dst)
    cap = cv2.VideoCapture(dst)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, round(CAND_SEC * fps))
    if total <= 0:
        print(f"  ⚠ {stem}: unreadable — skipped"); cap.release(); os.remove(dst); continue
    session = processor.init_video_session(inference_device=device, dtype=DTYPE)
    n_clip = 0
    for si, fidx in enumerate(range(0, total, stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, fr = cap.read()
        if not ok: continue
        pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil, device=device, return_tensors="pt")
        if si == 0:
            processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=0,
                obj_ids=list(range(len(DUMMY_BOXES))),
                input_boxes=[[list(b) for b in DUMMY_BOXES]],
                original_size=inputs.original_sizes[0])
        with torch.no_grad():
            teacher(inference_session=session, frame=inputs.pixel_values[0])
            t_feat = hook.pop()
            s_feat = student(STUDENT_TF(pil).unsqueeze(0).to(device))
        if t_feat is None: continue
        cos = per_position_cosine(t_feat, s_feat)
        wall = wall0 + timedelta(seconds=fidx/fps)
        base = f"{stem}_f{fidx:07d}"
        torch.save(t_feat.cpu().half(), osp.join(LF, base + ".pt"))
        pil.save(osp.join(LI, base + ".jpg"), quality=92)
        CAND.append({"base": base, "clip": vname, "frame_idx": int(fidx),
                     "wall": wall.isoformat(), "night": is_night(wall), "cos": round(cos, 5)})
        n_clip += 1
        del inputs
        if (si+1) % 25 == 0:
            prune_session(session); gc.collect(); torch.cuda.empty_cache()
    cap.release(); os.remove(dst); del session
    gc.collect(); torch.cuda.empty_cache()
    el = (time.time()-t0)/60
    print(f"[{ci+1}/{len(CLIPS)}] {stem}: {n_clip} candidates | total {len(CAND)} | {el:.0f} min")
    with open(osp.join(LOCAL, "cand.json"), "w") as f: json.dump(CAND, f)

print(f"\npass complete: {len(CAND)} candidates, "
      f"{sum(1 for c in CAND if c['night'])} night, {(time.time()-t0)/3600:.1f} h")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select (hardest-first, night quota, per-clip cap) and publish to the Volume

# COMMAND ----------

cands = sorted(CAND, key=lambda c: c["cos"])
per_clip = {}
sel, sel_set = [], set()
def try_add(c):
    if c["base"] in sel_set: return False
    if per_clip.get(c["clip"], 0) >= CLIP_CAP: return False
    sel.append(c); sel_set.add(c["base"])
    per_clip[c["clip"]] = per_clip.get(c["clip"], 0) + 1
    return True

need_night = int(N_TARGETS * NIGHT_FRAC)
for c in cands:                                   # hardest night first, up to the quota
    if len([s for s in sel if s["night"]]) >= need_night: break
    if c["night"]: try_add(c)
for c in cands:                                   # fill the rest hardest-first, any tod
    if len(sel) >= N_TARGETS: break
    try_add(c)
n_night = sum(1 for c in sel if c["night"])
print(f"selected {len(sel)} targets | night {n_night} ({n_night/len(sel):.0%}) | "
      f"cosine range [{sel[0]['cos'] if sel else '-'} .. {max(c['cos'] for c in sel) if sel else '-'}]")

t0 = time.time()
seld = osp.join(LOCAL, "sel_feat"); seli = osp.join(LOCAL, "sel_img")
shutil.rmtree(seld, ignore_errors=True); shutil.rmtree(seli, ignore_errors=True)
os.makedirs(seld); os.makedirs(seli)
for c in sel:
    os.link(osp.join(LF, c["base"] + ".pt"), osp.join(seld, c["base"] + ".pt"))
    os.link(osp.join(LI, c["base"] + ".jpg"), osp.join(seli, c["base"] + ".jpg"))
os.system(f"cp -r '{seld}/.' '{BK_V2}/'")
os.system(f"cp -r '{seli}/.' '{FR_V2}/'")
print(f"published {len(sel)} pairs in {(time.time()-t0)/60:.1f} min")

# ---- manifest v2 = v1 rows + mined rows ------------------------------------------------------
v1 = pd.read_csv(osp.join(META, "train_manifest.csv"))
rows = [{"clip": c["base"].rsplit("_f", 1)[0], "frame_idx": c["frame_idx"],
         "image": osp.join(FR_V2, c["base"] + ".jpg"),
         "feature": osp.join(BK_V2, c["base"] + ".pt"),
         "source": "mined_v2", "cos_at_mining": c["cos"], "night": c["night"]} for c in sel]
v1["source"] = "v1"; v1["cos_at_mining"] = np.nan; v1["night"] = np.nan
man = pd.concat([v1, pd.DataFrame(rows)], ignore_index=True)
man.to_csv(osp.join(META, "train_manifest_v2.csv"), index=False)
print(f"manifest v2: {len(man)} pairs ({len(v1)} v1 + {len(rows)} mined)")

# sanity: pairing + shape on one mined target
s = torch.load(rows[0]["feature"], map_location="cpu")
assert s.shape[1] == 1024 and list(s.shape)[-2:] == [72, 72]
assert osp.exists(rows[0]["image"])
print("target shape + pairing verified")

# ---- mining report: cosine by hour (the no-night-penalty claim at 23k-frame scale) ----------
by_hour = {}
for c in CAND:
    h = int(c["wall"][11:13])
    by_hour.setdefault(h, []).append(c["cos"])
rep = {"n_candidates": len(CAND), "n_selected": len(sel), "night_selected_frac": round(n_night/max(len(sel),1), 3),
       "cos_by_hour": {str(h): {"n": len(v), "mean": round(float(np.mean(v)), 4),
                                "p10": round(float(np.percentile(v, 10)), 4)}
                       for h, v in sorted(by_hour.items())},
       "selected_cos_max": max(c["cos"] for c in sel) if sel else None}
with open(osp.join(TOL, "Metrics_mining_report.json"), "w") as f:
    json.dump(rep, f, indent=1)
print("\ncosine by hour (mean / p10):")
for h in sorted(by_hour):
    v = by_hour[h]
    print(f"  {h:02d}h n={len(v):5d} mean={np.mean(v):.4f} p10={np.percentile(v,10):.4f}")
hook.remove()
print("\n✓ mining complete")
