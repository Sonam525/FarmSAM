# Databricks notebook source
# MAGIC %md
# MAGIC # Distil the calf student
# MAGIC
# MAGIC Trains the TinyViT-21M-512 student on the calf targets from `extract_teacher_targets` and, for the night-mined student,
# MAGIC `mine_targets`. Student: `features_only` stages 1 to 3, 1x1 lateral + GroupNorm + GELU, resize to 72x72, concat, 1x1 to
# MAGIC 1024, residual refine, learnable scale and bias. All BatchNorm is replaced by GroupNorm and `drop_path_rate=0.0`.
# MAGIC Loss: `1.0 dirMSE + 0.5 cosine + 0.3 (std + mean) + 0.1 rawMSE`. AdamW (backbone 1e-4, rest 3e-4), warm restarts,
# MAGIC bf16 autocast, EMA 0.9995, early stop after 15 epochs, batch 4 at 1024.

# COMMAND ----------

# MAGIC %pip install "numpy<2.0" "timm" "opencv-python-headless==4.11.0.86" "Pillow==11.1.0" "pandas"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, os.path as osp, re, json, time, gc, shutil, random
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from timm.data import resolve_data_config

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda")
VOL_ROOT = "/Volumes/your_catalog/your_schema/your_volume/farmsam"
TOL      = osp.join(VOL_ROOT, "Tol")
# v2 parameterization (defaults preserve the verified v1 run byte-for-byte): manifest may be the
# mined-augmented train_manifest_v2.csv; ckpt_name isolates the v2 student from the v1 checkpoint.
dbutils.widgets.text("manifest", "train_manifest.csv")
dbutils.widgets.text("ckpt_name", "best_backbone_student.pt")
MANIFEST_NAME = dbutils.widgets.get("manifest").strip() or "train_manifest.csv"
CKPT_NAME     = dbutils.widgets.get("ckpt_name").strip() or "best_backbone_student.pt"
META     = osp.join(TOL, "Metadata", MANIFEST_NAME)
CKPT_DIR = osp.join(TOL, "Checkpoints"); os.makedirs(CKPT_DIR, exist_ok=True)
print(f"manifest={MANIFEST_NAME}  ckpt={CKPT_NAME}")
LOCAL    = "/local_disk0/tmp/calf_student"; os.makedirs(LOCAL, exist_ok=True)

EPOCHS, BATCH, PATIENCE = 100, 4, 15
EMA_DECAY = 0.9995
print("GPU:", torch.cuda.get_device_name(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage data on local NVMe (Volume seek latency would dominate training)

# COMMAND ----------

man = pd.read_csv(META)
print(f"manifest: {len(man)} pairs, {man['clip'].nunique()} clips")

LF, LI = osp.join(LOCAL, "feat"), osp.join(LOCAL, "img")
os.makedirs(LF, exist_ok=True); os.makedirs(LI, exist_ok=True)
if len(os.listdir(LF)) < len(man):
    t0 = time.time()
    subs = ["train"]
    # mined-augmented manifests reference train_v2 dirs — stage them too, or the exists-filter
    # below silently drops every mined pair and "v2" trains on v1 data (caught in review)
    if "source" in man.columns and (man["source"] == "mined_v2").any():
        subs.append("train_v2")
    for sub in subs:
        os.system(f"cp -r '{osp.join(TOL,'Backbone_Features',sub)}/.' '{LF}/'")
        os.system(f"cp -r '{osp.join(TOL,'Frames',sub)}/.' '{LI}/'")
    print(f"staged {subs} in {(time.time()-t0)/60:.1f} min")
man["image"] = man["image"].map(lambda p: osp.join(LI, osp.basename(p)))
man["feature"] = man["feature"].map(lambda p: osp.join(LF, osp.basename(p)))
n_before = len(man)
man = man[man["image"].map(osp.exists) & man["feature"].map(osp.exists)].reset_index(drop=True)
print(f"usable pairs: {len(man)} of {n_before}")
assert len(man) >= 0.98 * n_before, \
    f"staging dropped {n_before-len(man)} pairs — a manifest source dir was not staged"

# ---- validation = whole clips nearest 08:00 / 13:00 / 19:00 / 23:00 (day+dusk+night) ----
RXT = re.compile(r'-(\d{8})-(\d{6})-')
def clip_start(stem):
    m = RXT.search(stem.replace("_", "-"))
    if not m: return None
    return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")

clips = sorted(man["clip"].unique())
starts = {c: clip_start(c) for c in clips}
val_clips = []
for target_h in (8, 13, 19, 23):
    cand = [c for c in clips if starts[c] and c not in val_clips]
    if not cand: continue
    val_clips.append(min(cand, key=lambda c: abs(starts[c].hour*60+starts[c].minute - target_h*60)))
print("validation clips (held out whole):")
for c in val_clips: print(f"   {c}  start {starts[c]:%m-%d %H:%M}")

# VAL COMPARABILITY: mined rows (source=mined_v2) are selected as the HARDEST frames — letting
# them into the val split would make v2's fidelity numbers incomparably harsher than v1's.
# Val = v1-source rows of the held-out clips only; train = everything outside the val clips.
_src = man["source"] if "source" in man.columns else pd.Series(["v1"] * len(man))
val_df = man[man["clip"].isin(val_clips) & (_src == "v1")].reset_index(drop=True)
trn_df = man[~man["clip"].isin(val_clips)].reset_index(drop=True)
print(f"train {len(trn_df)} / val {len(val_df)} (val restricted to v1-source rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset — synced h-flip, image-only colour jitter (no rotation/vflip: overhead view)

# COMMAND ----------

_t = timm.create_model('tiny_vit_21m_512', pretrained=False)
CFG = resolve_data_config(_t.pretrained_cfg); del _t
NORM = transforms.Normalize(mean=CFG['mean'], std=CFG['std'])
JITTER = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

class DistillDS(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True); self.train = train
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image"]).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        feat = torch.load(r["feature"], map_location="cpu").float()
        if feat.dim() == 4: feat = feat[0]                     # [1024,72,72]
        if self.train:
            if random.random() < 0.5:                          # synced horizontal flip
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                feat = torch.flip(feat, dims=[-1])
            img = JITTER(img)
        return NORM(transforms.functional.to_tensor(img)), feat

trn_dl = DataLoader(DistillDS(trn_df, True), batch_size=BATCH, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
val_dl = DataLoader(DistillDS(val_df, False), batch_size=BATCH, shuffle=False,
                    num_workers=4, pin_memory=True, persistent_workers=True)
print(f"batches: train {len(trn_dl)} | val {len(val_dl)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Student

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
    def __init__(self, out_channels=1024, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('tiny_vit_21m_512', pretrained=pretrained,
                                          features_only=True, out_indices=(1, 2, 3),
                                          drop_path_rate=0.0)   # MUST be 0.0 — see header
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

student = StudentEncoderTinyViT().to(device)
n_par = sum(p.numel() for p in student.parameters())
print(f"student params: {n_par/1e6:.2f}M")

# determinism check — GroupNorm + no stochastic depth => train/eval must be identical
bn = sum(1 for m in student.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))
d = torch.randn(1, 3, 1024, 1024, device=device)
with torch.no_grad():
    student.train(); a = student(d)
    student.eval();  b = student(d)
diff = (a-b).abs().max().item()
print(f"BatchNorm layers: {bn} | train/eval max diff: {diff:.2e} | out {list(b.shape)}")
assert bn == 0 and diff < 1e-5, "non-deterministic student (check drop_path_rate / BN)"
del d, a, b; gc.collect(); torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loss / optimiser / EMA

# COMMAND ----------

class BackboneDistillationLoss(nn.Module):
    def __init__(self, w_dir=1.0, w_cos=0.5, w_stat=0.3, w_raw=0.1):
        super().__init__(); self.wd, self.wc, self.ws, self.wr = w_dir, w_cos, w_stat, w_raw
    def forward(self, s, t):
        sn = F.normalize(s, dim=1); tn = F.normalize(t, dim=1)
        l_dir = F.mse_loss(sn, tn)
        l_cos = (1 - F.cosine_similarity(s, t, dim=1)).mean()
        l_stat = (F.l1_loss(s.mean((2, 3)), t.mean((2, 3))) +
                  F.l1_loss(s.std((2, 3)),  t.std((2, 3))))
        l_raw = F.mse_loss(s, t)
        total = self.wd*l_dir + self.wc*l_cos + self.ws*l_stat + self.wr*l_raw
        return total, {"dir": l_dir.item(), "cos": l_cos.item(),
                       "stat": l_stat.item(), "raw": l_raw.item()}

crit = BackboneDistillationLoss()
opt = torch.optim.AdamW([
    {"params": student.backbone.parameters(), "lr": 1e-4},
    {"params": [p for n, p in student.named_parameters() if not n.startswith("backbone.")],
     "lr": 3e-4}], weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)

ema = {k: v.detach().clone() for k, v in student.state_dict().items() if v.dtype.is_floating_point}
def ema_update():
    with torch.no_grad():
        for k, v in student.state_dict().items():
            if k in ema: ema[k].mul_(EMA_DECAY).add_(v.detach(), alpha=1-EMA_DECAY)

def swap_in_ema():
    backup = {k: v.detach().clone() for k, v in student.state_dict().items() if k in ema}
    student.load_state_dict({**student.state_dict(), **ema}, strict=False)
    return backup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train

# COMMAND ----------

@torch.no_grad()
def evaluate():
    student.eval(); tot = 0.0; cs = []; sr = []; mses = []; n = 0
    for img, feat in val_dl:
        img = img.to(device, non_blocking=True); feat = feat.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student(img)
        out = out.float()
        l, _ = crit(out, feat)
        tot += l.item()*img.size(0); n += img.size(0)
        cs.append(F.cosine_similarity(out, feat, dim=1).mean().item())
        sr.append((out.std()/feat.std()).item())
        mses.append(F.mse_loss(out, feat).item())
    return tot/max(n, 1), float(np.mean(cs)), float(np.mean(sr)), float(np.mean(mses))

hist = []; best = float("inf"); bad = 0
t_start = time.time()
for ep in range(EPOCHS):
    student.train(); run = 0.0; nb = 0; t_ep = time.time()
    for img, feat in trn_dl:
        img = img.to(device, non_blocking=True); feat = feat.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student(img)
        loss, parts = crit(out.float(), feat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); ema_update()
        run += loss.item(); nb += 1
    sched.step()

    backup = swap_in_ema()
    v_loss, v_cos, v_scale, v_mse = evaluate()
    student.load_state_dict({**student.state_dict(), **backup}, strict=False)

    rec = {"epoch": ep+1, "train_loss": run/max(nb, 1), "ema_val_loss": v_loss,
           "ema_cosine": v_cos, "ema_scale_ratio": v_scale, "ema_mse": v_mse,
           "lr": opt.param_groups[0]["lr"], "sec": time.time()-t_ep}
    hist.append(rec)
    flag = ""
    if v_loss < best - 1e-5:
        best = v_loss; bad = 0; flag = "  ← best"
        torch.save({"epoch": ep, "model_state_dict": student.state_dict(), "ema_shadow": ema,
                    "val_loss": v_loss, "cosine": v_cos, "scale_ratio": v_scale,
                    "config": {"arch": "tiny_vit_21m_512", "out": 1024, "dataset": "Tol ch01",
                               "train_days": ["20240418", "20240419"], "val_clips": val_clips}},
                   osp.join(CKPT_DIR, CKPT_NAME))
    else:
        bad += 1
    print(f"ep {ep+1:>3}/{EPOCHS} train {rec['train_loss']:.4f} | EMA val {v_loss:.4f} "
          f"cos {v_cos:.4f} scale {v_scale:.3f} | {rec['sec']/60:.1f} min{flag}")
    with open(osp.join(CKPT_DIR, "distill_history.json"), "w") as f: json.dump(hist, f, indent=2)
    if bad >= PATIENCE:
        print(f"early stop at epoch {ep+1} (no improvement for {PATIENCE})"); break

print(f"\n✓ training done in {(time.time()-t_start)/3600:.2f} h | best EMA val {best:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Held-out fidelity (EMA weights) — compare against Tol2: cos 0.852, scale 0.958, MSE 0.663

# COMMAND ----------

ck = torch.load(osp.join(CKPT_DIR, CKPT_NAME), map_location="cpu")
student.load_state_dict(ck["model_state_dict"])
student.load_state_dict({**student.state_dict(), **{k: v for k, v in ck["ema_shadow"].items()}},
                        strict=False)
student.to(device).eval()

cs, sr, ms = [], [], []
with torch.no_grad():
    for img, feat in val_dl:
        img = img.to(device); feat = feat.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student(img)
        out = out.float()
        cs.append(F.cosine_similarity(out, feat, dim=1).mean().item())
        sr.append((out.std()/feat.std()).item())
        ms.append(F.mse_loss(out, feat).item())
res = {"epoch": ck["epoch"]+1, "cosine_mean": float(np.mean(cs)), "cosine_std": float(np.std(cs)),
       "scale_ratio_mean": float(np.mean(sr)), "scale_ratio_std": float(np.std(sr)),
       "mse_mean": float(np.mean(ms)), "n_val": len(val_df), "val_clips": val_clips,
       "reference_tol2": {"cosine": 0.852, "scale_ratio": 0.958, "mse": 0.663},
       "reference_edinburgh_pigs": {"cosine": 0.808, "scale_ratio": 0.949, "mse": 0.763}}
print(json.dumps(res, indent=2))
ok_cos = res["cosine_mean"] >= 0.70
ok_scale = 0.8 <= res["scale_ratio_mean"] <= 1.2
print(f"\nbands: cosine>=0.7 {'OK' if ok_cos else 'FAIL'} | scale 0.8-1.2 {'OK' if ok_scale else 'FAIL'}")
with open(osp.join(TOL, f"Metrics_distill_fidelity_{CKPT_NAME.replace('.pt','')}.json"), "w") as f: json.dump(res, f, indent=2)
print("\nNext: adapt N3 v2 (mask-based tracking + re-ID) to Tol and prove the 6 h night-gap recall.")
