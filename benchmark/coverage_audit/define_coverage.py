"""
define_coverage.py  --  PigTrack scene-coverage definition (EXPLORATORY, post hoc).

Reads ONLY signature-level inputs (no performance file):
  signatures/signatures.json, signatures/pairs.csv, signatures/controls.json
  pt_mining/_ref/vol/manifest.csv  (columns seq, role only)
  pt_mining/_ref/vol/split.txt
  _pt_train_seqinfo.txt
Writes (next to this script):
  coverage_table.csv            primary table, 80 rows
  coverage_table_variants.csv   A5 sensitivity variants (strict U, loose U, view-only score)
  coverage_constants.json       every constant with its provenance + counts

Run:  python -I define_coverage.py
Rules (all numbers come from controls.json or are declared conventions; see AUDIT_PLAN.md):
  L  = chance ceiling = max RANSAC inliers over all different-pen pairs (controls strata)         -> 6
  U  = self-match floor = min split-half inliers over 80 sequences                                -> 124
  geometry valid = h_ok & scale in [0.8,1.25] & |rot|<=10 deg & hull >= negative-control p95 hull
  view category : none (n<=L or geometry invalid) / partial (L<n<U) / same (n>=U)
  V = 0 | log(n/L)/log(U/L) | 1
  zoom   : dz = |log2(box_q/box_ref)| / log2(1.25);   Z  = clip((3-dz)/2, 0, 1)
  light  : night flags equal AND d = max(|dBright|/tau_b, |dSat|/tau_s);  Lt = night_eq*clip((3-d)/2,0,1)
  pair coverage c(q,r) = 0.5 V + 0.25 Z + 0.25 Lt   (only for r with V>0; Z,Lt relative to r)
  R-overall typicality c_R(q) = 0.25 Z_R + 0.25 Lt_R (zoom/light vs R medians; light vs same-night-flag subset)
  score S(q) = max( max_r c(q,r), c_R(q) )
  tier: covered  = exists r: same view & dz<=1 & night equal & d<=1
        uncovered = no r with view category != none
        partly   = everything else
"""
import json, math, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIG = HERE / "signatures"
BENCH = HERE.parent
MAN = BENCH / "pt_mining" / "_ref" / "vol" / "manifest.csv"
SPLIT = BENCH / "pt_mining" / "_ref" / "vol" / "split.txt"
TRSI = BENCH / "_pt_train_seqinfo.txt"

W_V, W_Z, W_L = 0.5, 0.25, 0.25          # declared convention
SCALE_LO, SCALE_HI, ROT_MAX = 0.8, 1.25, 10.0   # from the task's own example
TAU_Z = math.log2(1.25)                  # zoom tolerance = same-view scale band
FULL, ZERO = 1.0, 3.0                    # full credit within 1 tolerance, zero credit at 3 tolerances

sigs = json.load(open(SIG / "signatures.json"))["sequences"]
controls = json.load(open(SIG / "controls.json"))
pairs = pd.read_csv(SIG / "pairs.csv")

# ---------------- constants from controls ----------------
L_neg_pair = controls["negative_control"]["n_inliers"]["max"]
L_neg_samres = controls["strata_all_pairs"]["diffpen_sameres"]["n_inliers"]["max"]
L = int(max(L_neg_pair, L_neg_samres))
U_prim = int(controls["split_half"]["n_inliers"]["min"])
U_strict = int(round(controls["split_half"]["n_inliers"]["p05"]))
U_loose = int(round(math.sqrt(L * U_prim)))
L_floor2 = 2 * L   # EXTRA variant (added after seeing one test sequence at n = L+1; before any outcome file was opened)
HULL_MIN = round(float(controls["negative_control"]["inlier_hull_frac"]["p95"]), 3)

# ---------------- sequence table ----------------
rows = []
for k, v in sigs.items():
    rows.append(dict(
        seq=k, split_official=v["split"], pen=v["meta"]["penID"], night=v["meta"]["night"],
        W=v["meta"]["imWidth"], H=v["meta"]["imHeight"], fps=v["meta"]["frameRate"],
        n_frames=v["meta"]["seqLength"], objects=v["gt"]["n_objects"],
        objs_per_frame=v["gt"]["objs_per_frame_mean"], crowding=v["gt"]["crowding_mean"],
        box=v["gt"]["box_size_median"], bright=v["image"]["bg_brightness_mean"],
        sat=v["image"]["bg_saturation_mean"], self_inl=v["split_half"]["n_inliers"]))
S = pd.DataFrame(rows).set_index("seq").sort_index()
S["size"] = S["W"].astype(str) + "x" + S["H"].astype(str)

# R, distillation-val, val, test
man = pd.read_csv(MAN, usecols=["seq", "role"])
R = sorted(man.loc[man.role == "train", "seq"].unique())
trs = []
for line in open(TRSI):
    if line.startswith("[Sequence]"):
        trs.append(line.split("name=")[1].split()[0])
DV = sorted(set(trs) - set(R))
split_txt = dict(l.split() for l in open(SPLIT) if l.strip())
assert len(R) == 31, len(R)
assert len(trs) == 35 and len(DV) == 4, (len(trs), DV)
assert all(split_txt[s] == "train" for s in R + DV)
assert set(split_txt) == set(S.index) and len(S) == 80
def role(s):
    if s in R: return "R_train"
    if s in DV: return "distill_val"
    return split_txt[s]
S["split"] = [role(s) for s in S.index]

# lighting tolerance = pooled within-night-flag SD over all 80 sequences
def pooled_sd(col):
    num = den = 0.0
    for g, d in S.groupby("night"):
        n = len(d); num += (n - 1) * d[col].var(ddof=1); den += n - 1
    return math.sqrt(num / den)
TAU_B = pooled_sd("bright")
TAU_S = pooled_sd("sat")

# ---------------- pair lookup (symmetric) ----------------
P = {}
for r in pairs.itertuples(index=False):
    P[(r.seq_a, r.seq_b)] = r
def prow(q, r):
    """return (n_inliers, hull, scale_q_to_r, rot_q_to_r, h_ok, dino_bg, dino_fr) oriented q->r"""
    if (q, r) in P:
        x = P[(q, r)]; sc, ro = x.scale_ab, x.rot_deg
    else:
        x = P[(r, q)]
        sc = (1.0 / x.scale_ab) if (x.scale_ab == x.scale_ab and x.scale_ab) else float("nan")
        ro = -x.rot_deg if x.rot_deg == x.rot_deg else float("nan")
    return dict(n=float(x.n_inliers), hull=x.inlier_hull_frac, scale=sc, rot=ro, h_ok=int(x.h_ok),
                dino_bg=float(x.dino_bg_cos), dino_fr=float(x.dino_frames_cos), shift=x.shift_px)

def geo_valid(p):
    if p["h_ok"] != 1: return False
    if not (p["scale"] == p["scale"]) or not (SCALE_LO <= p["scale"] <= SCALE_HI): return False
    if not (p["rot"] == p["rot"]) or abs(p["rot"]) > ROT_MAX: return False
    if not (p["hull"] == p["hull"]) or p["hull"] < HULL_MIN: return False
    return True
def view_cat(p, U, Lc=None):
    Lc = L if Lc is None else Lc
    if p["n"] <= Lc or not geo_valid(p): return "none"
    return "same" if p["n"] >= U else "partial"
def view_score(p, U, Lc=None):
    Lc = L if Lc is None else Lc
    c = view_cat(p, U, Lc)
    if c == "none": return 0.0
    if c == "same": return 1.0
    return math.log(p["n"] / Lc) / math.log(U / Lc)
def credit(d):  # d = distance in tolerance units
    return float(min(1.0, max(0.0, (ZERO - d) / (ZERO - FULL))))
def zoom_d(bq, br): return abs(math.log2(bq / br)) / TAU_Z
def light_d(q, r_bright, r_sat): return max(abs(S.bright[q] - r_bright) / TAU_B, abs(S.sat[q] - r_sat) / TAU_S)

def assess(q, refs, U, weights=(W_V, W_Z, W_L), Lc=None):
    wv, wz, wl = weights
    best = None; any_same_ok = False; any_nonnone = False; n_same = 0
    cand = []
    top_raw = (-1.0, None)
    for r in refs:
        p = prow(q, r)
        if p["n"] > top_raw[0]: top_raw = (p["n"], r)
        vc = view_cat(p, U, Lc); V = view_score(p, U, Lc)
        dz = zoom_d(S.box[q], S.box[r]); Z = credit(dz)
        dl = light_d(q, S.bright[r], S.sat[r]); night_eq = int(S.night[q] == S.night[r])
        Lt = night_eq * credit(dl)
        if vc != "none": any_nonnone = True
        if vc == "same":
            n_same += 1
            if dz <= 1.0 and night_eq == 1 and dl <= 1.0: any_same_ok = True
        if V > 0:
            c = wv * V + wz * Z + wl * Lt
            cand.append((c, V, p["n"], p["dino_bg"], r, p, vc, dz, dl, night_eq, Z, Lt))
    # R-overall typicality
    boxR = float(np.median([S.box[r] for r in refs]))
    sub = [r for r in refs if S.night[r] == S.night[q]]
    bR = float(np.median([S.bright[r] for r in sub])); sR = float(np.median([S.sat[r] for r in sub]))
    ZR = credit(zoom_d(S.box[q], boxR)); LR = credit(light_d(q, bR, sR))
    cR = wz * ZR + wl * LR
    if cand:
        cand.sort(key=lambda t: (-t[0], -t[2], -t[3], t[4]))
        # best-matching reference = argmax of VIEW score (task spec), ties -> pair coverage, raw inliers
        vbest = sorted(cand, key=lambda t: (-t[1], -t[0], -t[2], -t[3], t[4]))[0]
        cbest = cand[0]
        score = max(cbest[0], cR)
    else:
        vbest = cbest = None; score = cR
    tier = "covered" if any_same_ok else ("partly covered" if any_nonnone else "uncovered")
    out = dict(tier=tier, score=score, n_same_refs=n_same, view_score=(vbest[1] if vbest else 0.0),
               top_raw_inl=top_raw[0], top_raw_ref=top_raw[1], cR=cR, ZR=ZR, LR=LR, boxR=boxR)
    if vbest:
        p = vbest[5]; r = vbest[4]
        out.update(best_ref=r, inliers=p["n"], scale=p["scale"], rot=p["rot"], hull=p["hull"], shift=p["shift"],
                   view_cat=vbest[6], zoom_ratio=S.box[q] / S.box[r], zoom_ref=r,
                   dz=vbest[7], dl=vbest[8], night_eq=vbest[9], Z=vbest[10], Lt=vbest[11])
    else:
        out.update(best_ref="", inliers=float("nan"), scale=float("nan"), rot=float("nan"), hull=float("nan"),
                   shift=float("nan"), view_cat="none", zoom_ratio=S.box[q] / boxR, zoom_ref="R_overall",
                   dz=zoom_d(S.box[q], boxR), dl=light_d(q, bR, sR), night_eq=1, Z=ZR, Lt=LR)
    return out

out_rows, var_rows = [], []
for q in S.index:
    refs = [r for r in R if r != q]
    a = assess(q, refs, U_prim)
    a_s = assess(q, refs, U_strict)
    a_l = assess(q, refs, U_loose)
    a_v = assess(q, refs, U_prim, weights=(1.0, 0.0, 0.0))
    a_f = assess(q, refs, U_prim, Lc=L_floor2)
    dn = max(((prow(q, r)["dino_bg"], r) for r in refs))
    df = max(((prow(q, r)["dino_fr"], r) for r in refs))
    sp_same = sum(1 for r in refs if S.pen[r] == S.pen[q] and S.night[r] == S.night[q])
    sp_opp = sum(1 for r in refs if S.pen[r] == S.pen[q] and S.night[r] != S.night[q])
    note = ""
    if a["tier"] == "uncovered" and sp_same == 0 and sp_opp > 0:
        note = "same pen in R only with opposite night flag"
    elif a["tier"] == "uncovered" and sp_same + sp_opp == 0:
        note = "pen has no sequence in R"
    out_rows.append(dict(
        sequence=q, split=S.split[q], pen=int(S.pen[q]), night=int(S.night[q]), size=S["size"][q], fps=int(S.fps[q]),
        objects=int(S.objects[q]), best_ref=a["best_ref"], inliers=a["inliers"], scale=a["scale"], rotation_deg=a["rot"],
        zoom_ratio=a["zoom_ratio"], dino_nearest_ref_cos=dn[0], coverage_score=a["score"], tier=a["tier"],
        # extra columns
        view_category=a["view_cat"], V=a["view_score"], Z=a["Z"], Lt=a["Lt"], zoom_ref=a["zoom_ref"], dz_tol_units=a["dz"],
        light_d_tol_units=a["dl"], night_equal_best=a["night_eq"], hull_frac=a["hull"], shift_px=a["shift"],
        n_ref_same_view=a["n_same_refs"], top_raw_inliers_any_ref=a["top_raw_inl"], top_raw_ref=a["top_raw_ref"],
        c_R_typicality=a["cR"], dino_nearest_ref=dn[1], dino_frames_nearest_cos=df[0], dino_frames_nearest_ref=df[1],
        box_size_median=S.box[q], bg_brightness=S.bright[q], bg_saturation=S.sat[q], crowding_mean=S.crowding[q],
        objs_per_frame_mean=S.objs_per_frame[q], self_match_inliers=int(S.self_inl[q]),
        n_R_samepen_samenight=sp_same, n_R_samepen_oppnight=sp_opp, tier_note=note))
    var_rows.append(dict(
        sequence=q, split=S.split[q],
        score_strictU=a_s["score"], tier_strictU=a_s["tier"], score_looseU=a_l["score"], tier_looseU=a_l["tier"],
        score_floorL12=a_f["score"], tier_floorL12=a_f["tier"],
        score_viewonly=a_v["score"], tier_viewonly=a_v["tier"]))

T = pd.DataFrame(out_rows)
Vv = pd.DataFrame(var_rows)
# D2 variant: queries that are "uncovered" ONLY because every same-pen reference has the opposite night flag (SIFT cannot
# match IR against colour, so their view status is unmeasurable, not known to be new) are recoded "partly covered".
NOTE_OPP = "same pen in R only with opposite night flag"
Vv["tier_penaware"] = np.where((T.tier.values == "uncovered") & (T.tier_note.values == NOTE_OPP), "partly covered", T.tier.values)
T.to_csv(HERE / "coverage_table.csv", index=False, float_format="%.6g")
Vv.to_csv(HERE / "coverage_table_variants.csv", index=False, float_format="%.6g")

def counts(df, col="tier"):
    return {s: df[df.split == s][col].value_counts().reindex(["covered", "partly covered", "uncovered"]).fillna(0).astype(int).to_dict()
            for s in ["R_train", "distill_val", "val", "test"]}
consts = dict(
    L=L, L_provenance=dict(neg_control_diffpen_diffres_max=L_neg_pair, diffpen_sameres_max=L_neg_samres),
    U_primary=U_prim, U_primary_provenance="min of split-half n_inliers over 80 sequences",
    U_strict=U_strict, U_strict_provenance="p05 of split-half n_inliers (278.15 rounded)",
    U_loose=U_loose, U_loose_provenance="round(sqrt(L*U_primary)) = geometric mean of chance ceiling and self-match floor",
    hull_min=HULL_MIN, hull_min_provenance="p95 of inlier hull fraction in negative control (pairs with >=4 matches)",
    scale_band=[SCALE_LO, SCALE_HI], rot_max_deg=ROT_MAX, tau_zoom_log2=TAU_Z,
    tau_brightness=TAU_B, tau_saturation=TAU_S,
    tau_provenance="pooled within-night-flag SD over all 80 sequences (brightness 0-255, saturation 0-1)",
    weights=dict(view=W_V, zoom=W_Z, light=W_L), credit_full_at=FULL, credit_zero_at=ZERO,
    R=R, distill_val=DV,
    tier_counts_primary=counts(T), tier_counts_strictU=counts(Vv.assign(tier=Vv.tier_strictU), "tier"),
    tier_counts_looseU=counts(Vv.assign(tier=Vv.tier_looseU), "tier"),
    L_floor2=L_floor2, tier_counts_floorL12=counts(Vv.assign(tier=Vv.tier_floorL12), "tier"),
    tier_counts_penaware=counts(Vv.assign(tier=Vv.tier_penaware), "tier"),
    penaware_recoded_test=sorted(T[(T.split == "test") & (T.tier == "uncovered") & (T.tier_note == NOTE_OPP)].sequence.tolist()))
# within-view calibration check (pairs that are 'same' view under the primary rule and share the night flag)
cal = []
for (a, b), x in P.items():
    p = prow(a, b)
    if view_cat(p, U_prim) == "same" and S.night[a] == S.night[b]:
        cal.append((abs(S.bright[a] - S.bright[b]), abs(S.sat[a] - S.sat[b]), abs(math.log2(S.box[a] / S.box[b]))))
cal = np.array(cal)
consts["within_view_calibration_check"] = dict(
    n_pairs=int(len(cal)),
    brightness_abs_diff_p90_p95=[float(np.percentile(cal[:, 0], 90)), float(np.percentile(cal[:, 0], 95))],
    saturation_abs_diff_p90_p95=[float(np.percentile(cal[:, 1], 90)), float(np.percentile(cal[:, 1], 95))],
    log2_boxsize_abs_diff_p90_p95=[float(np.percentile(cal[:, 2], 90)), float(np.percentile(cal[:, 2], 95))],
    note="same-view, same-night-flag pairs under the frozen rule; tolerances are NOT tuned to these, this only checks they are not tighter than within-view variation")
# collinearity of the score with covariates among the 25 test sequences (signature-only)
from scipy.stats import spearmanr
tt = T[T.split == "test"].copy()
tt["is1080"] = (tt["size"] == "1920x1080").astype(int)
consts["test_collinearity_spearman"] = {c: float(spearmanr(tt.coverage_score, tt[c])[0]) for c in ["night", "objects", "is1080", "crowding_mean", "fps"]}
consts["test_score_ties"] = dict(n=int(len(tt)), n_unique=int(tt.coverage_score.round(9).nunique()),
                                 n_at_1=int((tt.coverage_score > 0.999999).sum()), n_at_half=int(((tt.coverage_score - 0.5).abs() < 1e-9).sum()))
vt = Vv[Vv.split == "test"]
consts["test_variant_score_spearman_vs_primary"] = {c: float(spearmanr(tt.coverage_score.values, vt[c].values)[0]) for c in ["score_strictU", "score_looseU", "score_floorL12", "score_viewonly"]}
consts["test_tier_crosstab_primary_vs_variants"] = {c: pd.crosstab(tt.tier.values, vt[c.replace("score", "tier")].values).to_dict() for c in ["score_strictU", "score_looseU", "score_floorL12"]}
consts["test_pen_sizes"] = tt.groupby("pen").size().to_dict()
json.dump(consts, open(HERE / "coverage_constants.json", "w"), indent=1)
print(json.dumps({k: v for k, v in consts.items() if k not in ("R",)}, indent=1))
