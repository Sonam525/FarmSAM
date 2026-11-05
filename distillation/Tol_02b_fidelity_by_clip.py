# Databricks notebook source
# MAGIC %md
# MAGIC # Tol · N2b — Distillation fidelity **per validation clip** (day vs IR night)
# MAGIC
# MAGIC The aggregate held-out cosine is 0.801. This splits it by held-out clip — 08:03 morning /
# MAGIC 12:58 midday / 19:16 evening / **22:49 IR night** — to answer a question that matters for
# MAGIC the headline claim: *do the student's features degrade at night?*
# MAGIC
# MAGIC Why it matters: identity has to survive a **6 h night gap** every day on this dataset, and
# MAGIC SAM3's detector already over-segments on IR frames. If the student is also weak there, night
# MAGIC tracking is doubly handicapped and we would want to re-weight night frames in training.
# MAGIC If night is fine, any later night failure is the tracker/detector, not distillation.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "timm" "Pillow==11.1.0" "pandas"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, re, json, gc
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from timm.data import resolve_data_config

device = torch.device("cuda")
VOL = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id"
TOL = osp.join(VOL, "Tol")
CKPT = osp.join(TOL, "Checkpoints", "best_backbone_student.pt")
man = pd.read_csv(osp.join(TOL, "Metadata", "train_manifest.csv"))

ck = torch.load(CKPT, map_location="cpu")
VAL_CLIPS = ck["config"]["val_clips"]
print(f"checkpoint epoch {ck['epoch']+1}; val clips:")
for c in VAL_CLIPS: print("   ", c)

# COMMAND ----------

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

student = StudentEncoderTinyViT()
sd = student.state_dict(); sd.update(ck["model_state_dict"])
for k, v in ck["ema_shadow"].items(): sd[k] = v      # report EMA weights
student.load_state_dict(sd); student = student.to(device).eval()

_t = timm.create_model('tiny_vit_21m_512', pretrained=False)
CFG = resolve_data_config(_t.pretrained_cfg); del _t
NORM = transforms.Normalize(mean=CFG['mean'], std=CFG['std'])

class DS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image"]).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        f = torch.load(r["feature"], map_location="cpu").float()
        if f.dim() == 4: f = f[0]
        return NORM(transforms.functional.to_tensor(img)), f

# COMMAND ----------

RXT = re.compile(r'-(\d{8})-(\d{6})-')
def start_of(stem):
    m = RXT.search(stem.replace("_", "-"))
    return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S") if m else None

rows = []
for clip in VAL_CLIPS:
    sub = man[man["clip"] == clip]
    if not len(sub): print(f"⚠ no rows for {clip}"); continue
    dl = DataLoader(DS(sub), batch_size=4, shuffle=False, num_workers=4)
    cs, sr, ms = [], [], []
    with torch.no_grad():
        for img, feat in dl:
            img = img.to(device); feat = feat.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = student(img)
            out = out.float()
            cs.append(F.cosine_similarity(out, feat, dim=1).mean().item())
            sr.append((out.std()/feat.std()).item())
            ms.append(F.mse_loss(out, feat).item())
    st = start_of(clip); h = st.hour if st else -1
    tod = "night/IR" if (h >= 21 or h < 6) else ("dawn/dusk" if h < 8 or h >= 19 else "day")
    rows.append({"clip": clip, "start": st.strftime("%m-%d %H:%M") if st else "?", "tod": tod,
                 "n": len(sub), "cosine": float(np.mean(cs)), "cosine_std": float(np.std(cs)),
                 "scale_ratio": float(np.mean(sr)), "mse": float(np.mean(ms))})
    print(f"  {clip}  [{tod}]  cos {np.mean(cs):.4f}  scale {np.mean(sr):.3f}  mse {np.mean(ms):.3f}")
    gc.collect(); torch.cuda.empty_cache()

# COMMAND ----------

df = pd.DataFrame(rows).sort_values("cosine")
print("\n=== fidelity by held-out clip (worst first) ===")
print(f'{"start":<14}{"tod":<11}{"n":>5}{"cosine":>9}{"scale":>8}{"mse":>8}')
for _, r in df.iterrows():
    print(f'{r["start"]:<14}{r["tod"]:<11}{r["n"]:>5}{r["cosine"]:>9.4f}{r["scale_ratio"]:>8.3f}{r["mse"]:>8.3f}')

night = df[df["tod"] == "night/IR"]["cosine"]
day = df[df["tod"] != "night/IR"]["cosine"]
print()
if len(night) and len(day):
    gap = float(day.mean() - night.mean())
    print(f"day/dusk mean cosine {day.mean():.4f}   night/IR mean cosine {night.mean():.4f}")
    print(f"NIGHT PENALTY: {gap:+.4f}")
    if gap > 0.05:
        print("  ⇒ student IS materially weaker on IR. Night tracking is doubly handicapped")
        print("    (detector over-segments there too). Consider re-weighting night frames.")
    else:
        print("  ⇒ student holds up on IR. Any night tracking failure is the tracker/detector,")
        print("    NOT distillation.")
with open(osp.join(TOL, "Metrics_fidelity_by_clip.json"), "w") as f:
    json.dump(rows, f, indent=2)
print("\nsaved Tol/Metrics_fidelity_by_clip.json")
