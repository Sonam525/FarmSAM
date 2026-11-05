"""
run_outcomes.py -- OUTCOMES step of the PigTrack coverage audit (EXPLORATORY, post hoc).

Applies the FROZEN plan (AUDIT_PLAN.md) with the FROZEN library (audit_analysis_lib.py) to the outcome files.
Everything that is not in the plan lives in the block EXPLORATORY_ADDED and is labelled so in the output.
Run:  python -I run_outcomes.py
Writes coverage_audit_results.json next to this file.
"""
import sys, os, json, hashlib, datetime, glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata

CA = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(CA, ".."))
sys.path.insert(0, CA)
import audit_analysis_lib as L

SEED, B = L.SEED, L.B_DEFAULT
OUT = {}

# ----------------------------------------------------------------------------------- 0. verify the freeze
fz = json.load(open(os.path.join(CA, "plan_frozen.json")))
def sha(p):
    return hashlib.sha256(open(os.path.join(CA, p), "rb").read()).hexdigest()
verif = {k: (sha(k) == v) for k, v in fz["sha256"].items()}
assert all(verif.values()), f"FROZEN FILE CHANGED: {verif}"
assert fz["seed"] == SEED and fz["bootstrap_B"] == B
OUT["freeze_check"] = dict(all_match=True, frozen_utc=fz["frozen_utc"], sha256=fz["sha256"],
                           outcome_run_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
print("freeze verified:", verif)

# ----------------------------------------------------------------------------------- 1. load
tab = pd.read_csv(os.path.join(CA, "coverage_table.csv"))
var = pd.read_csv(os.path.join(CA, "coverage_table_variants.csv")).set_index("sequence")
test = tab[tab.split == "test"].sort_values("sequence").reset_index(drop=True)
val = tab[tab.split == "val"].sort_values("sequence").reset_index(drop=True)
assert len(test) == 25 and len(val) == 20
seqs = list(test.sequence)

P_MAIN = os.path.join(BENCH, "_results", "j7v", "score_main", "per_sequence.csv")
P_XCHK = os.path.join(BENCH, "pt_mining", "scores", "20261027T011643Z_3efb9dd4", "per_sequence.csv")  # ref_arm = T
P_VAL = os.path.join(BENCH, "_results", "j5c", "score", "20261015T223533Z_8700f99b", "per_sequence.csv")
sc_x = json.load(open(os.path.join(os.path.dirname(P_XCHK), "score.json")))
assert sc_x["ref_arm"] == "T"

main = pd.read_csv(P_MAIN)
xchk = pd.read_csv(P_XCHK)
valo = pd.read_csv(P_VAL)
pv = main.pivot(index="seq", columns="arm", values="HOTA")
px = xchk.pivot(index="seq", columns="arm", values="HOTA")
pval = valo.pivot(index="seq", columns="arm", values="HOTA")

assert set(pv.index) == set(seqs), "j7v test sequences != coverage_table test rows"
assert set(px.index) == set(seqs)
assert set(pval.index) == set(val.sequence), "j5c val sequences != coverage_table val rows"
for a in ["T", "S21", "S-PT", "S-Edi"]:
    assert pv[a].notna().all(), f"missing {a} in primary"
pv = pv.loc[seqs]; px = px.loc[seqs]

# cross-check: primary vs pt_mining ref_arm=T file (report mismatch > 0.05 HOTA points; do not pick)
xc = {}
for a in ["T", "S21", "S-PT", "S-Edi"]:
    d = (pv[a] - px[a]).abs()
    xc[a] = dict(max_abs_diff=float(d.max()), n_over_0p05=int((d > 0.05).sum()),
                 seqs_over=[s for s in seqs if d[s] > 0.05])
# all 5 pt_mining files: shared arms identical?
allpm = {}
for f in sorted(glob.glob(os.path.join(BENCH, "pt_mining", "scores", "*", "per_sequence.csv"))):
    q = pd.read_csv(f).pivot(index="seq", columns="arm", values="HOTA").loc[seqs]
    allpm[os.path.basename(os.path.dirname(f))] = q
ref0 = list(allpm.values())[0]
pm_max = max(float((q[ref0.columns] - ref0).abs().max().max()) for q in allpm.values())
OUT["crosscheck"] = dict(primary=os.path.relpath(P_MAIN, BENCH), crosscheck=os.path.relpath(P_XCHK, BENCH), per_arm=xc,
                         five_pt_mining_files_max_abs_diff_any_arm=pm_max)
print("crosscheck:", json.dumps(xc), "| 5 pt_mining files max diff:", pm_max)

# ----------------------------------------------------------------------------------- 2. variables
HT, H21, HPT, HEDI = pv["T"].values, pv["S21"].values, pv["S-PT"].values, pv["S-Edi"].values
D_PT = HPT - HT          # main
D_21 = H21 - HT          # control
night = test.night.values.astype(float)
objects = test.objects.values.astype(float)
is1080 = (test["size"] == "1920x1080").values.astype(float)
pen = test.pen.values
OUT["context_numbers_check"] = dict(
    D_PT_mean=float(D_PT.mean()), D_PT_sd=float(D_PT.std(ddof=1)), D_PT_min=float(D_PT.min()), D_PT_min_seq=seqs[int(D_PT.argmin())],
    D_PT_max=float(D_PT.max()), D_PT_max_seq=seqs[int(D_PT.argmax())],
    D_21_mean=float(D_21.mean()), D_21_sd=float(D_21.std(ddof=1)), D_21_min=float(D_21.min()), D_21_max=float(D_21.max()),
    note="task text gave D_PT mean -12.9, SD 9.3, range -36.9 (0067) to -1.1 (0049); D_21 mean -4.6, SD 2.5")
print("D_PT mean %.2f sd %.2f min %.2f (%s) max %.2f (%s) | D_21 mean %.2f sd %.2f" % (
    D_PT.mean(), D_PT.std(ddof=1), D_PT.min(), seqs[int(D_PT.argmin())], D_PT.max(), seqs[int(D_PT.argmax())], D_21.mean(), D_21.std(ddof=1)))


def perm_test(x, y, nperm=B, seed=SEED):
    """Permutation test for Spearman rho. Added because the task text names it; not in the frozen plan."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    rng = np.random.default_rng(seed)
    r0 = L.spearman(x, y)
    rx = rankdata(x); ry = rankdata(y)
    idx = np.argsort(rng.random((nperm, n)), axis=1)
    r = L._pearson_rows(rx[None, :], ry[idx])
    tol = 1e-12
    return dict(n_perm=int(nperm), seed=int(seed), rho=float(r0),
                p_one_sided_pos=float((1 + np.sum(r >= r0 - tol)) / (nperm + 1)),
                p_two_sided=float((1 + np.sum(np.abs(r) >= abs(r0) - tol)) / (nperm + 1)))


def battery(label, score, tier, d_main, d_ctrl, with_a4=True, with_perm=True):
    """A1, A2, A3, A4(+A4b) for one definition of coverage (score, tier) and one (main, control) pair."""
    score = np.asarray(score, float); tier = np.asarray(tier)
    r = dict(label=label)
    r["A1"] = L.spearman_boot(score, d_main)
    r["A1_ctrl"] = L.spearman_boot(score, d_ctrl)
    r["A3_delta"] = L.paired_delta_rho_boot(score, d_main, d_ctrl)
    cov, unc, par = (tier == "covered"), (tier == "uncovered"), (tier == "partly covered")
    r["tier_n"] = dict(covered=int(cov.sum()), partly=int(par.sum()), uncovered=int(unc.sum()))
    for nm, d in (("A2", d_main), ("A2_ctrl", d_ctrl)):
        if cov.sum() >= 3 and unc.sum() >= 3:
            r[nm] = L.strat_diff_boot(d[cov], d[unc])
        else:
            r[nm] = dict(not_estimable=True, n_a=int(cov.sum()), n_b=int(unc.sum()))
    if with_perm:
        r["A1_perm"] = perm_test(score, d_main)
        r["A1_ctrl_perm"] = perm_test(score, d_ctrl)
    if with_a4:
        r["A4"] = L.partial_rank_corr_boot(score, d_main, np.column_stack([night, objects]))
        r["A4_ctrl"] = L.partial_rank_corr_boot(score, d_ctrl, np.column_stack([night, objects]))
        r["A4b"] = L.partial_rank_corr_boot(score, d_main, np.column_stack([night, objects, is1080]))
        r["A4b_ctrl"] = L.partial_rank_corr_boot(score, d_ctrl, np.column_stack([night, objects, is1080]))
    return r


# ----------------------------------------------------------------------------------- 3. PLAN: A1-A6, D1
print("\n=== frozen plan: primary definition ===")
x_prim = test.coverage_score.values
tier_prim = test.tier.values
prim = battery("primary", x_prim, tier_prim, D_PT, D_21)
OUT["A1_A4_primary"] = prim

# descriptive tier table that goes with A2 (no test)
def tier_desc(mask_dict):
    rows = {}
    for t, m in mask_dict.items():
        m = np.asarray(m)
        rows[t] = dict(n=int(m.sum()),
                       D_PT_mean=float(D_PT[m].mean()), D_PT_sd=float(D_PT[m].std(ddof=1)) if m.sum() > 1 else None,
                       D_21_mean=float(D_21[m].mean()), D_21_sd=float(D_21[m].std(ddof=1)) if m.sum() > 1 else None,
                       HOTA_T_mean=float(HT[m].mean()), HOTA_S21_mean=float(H21[m].mean()), HOTA_SPT_mean=float(HPT[m].mean()),
                       HOTA_SEdi_mean=float(HEDI[m].mean()),
                       HOTA_T_sd=float(HT[m].std(ddof=1)) if m.sum() > 1 else None,
                       HOTA_S21_sd=float(H21[m].std(ddof=1)) if m.sum() > 1 else None,
                       HOTA_SPT_sd=float(HPT[m].std(ddof=1)) if m.sum() > 1 else None)
    return rows
tiers_order = ["covered", "partly covered", "uncovered"]
OUT["tier_descriptive_primary"] = tier_desc({t: tier_prim == t for t in tiers_order} | {"all": np.ones(25, bool)})

# A5 variants (from coverage_table_variants.csv)
vt = var.loc[seqs]
variants = {
    "strictU": (vt.score_strictU.values, vt.tier_strictU.values),
    "looseU": (vt.score_looseU.values, vt.tier_looseU.values),
    "floorL12": (vt.score_floorL12.values, vt.tier_floorL12.values),
    "viewonly": (vt.score_viewonly.values, vt.tier_viewonly.values),
}
OUT["A5_variants"] = {}
for k, (s, t) in variants.items():
    OUT["A5_variants"][k] = battery(k, s, t, D_PT, D_21)
    OUT["A5_variants"][k]["tier_counts_test"] = {tt: int((t == tt).sum()) for tt in tiers_order}
# penaware: A2 only
t_pa = vt.tier_penaware.values
pa = dict(label="penaware", tier_counts_test={tt: int((t_pa == tt).sum()) for tt in tiers_order})
cov, unc = t_pa == "covered", t_pa == "uncovered"
for nm, d in (("A2", D_PT), ("A2_ctrl", D_21)):
    pa[nm] = L.strat_diff_boot(d[cov], d[unc]) if (cov.sum() >= 3 and unc.sum() >= 3) else dict(not_estimable=True, n_a=int(cov.sum()), n_b=int(unc.sum()))
OUT["A5_variants"]["penaware"] = pa

# A6 descriptive: S-PT minus S21 by tier
diff_pt_21 = HPT - H21
a6 = {"all_25": L.describe_boot(diff_pt_21)}
for t in tiers_order:
    m = tier_prim == t
    a6[t] = L.describe_boot(diff_pt_21[m])
OUT["A6"] = a6

# D1: dependence
OUT["D1"] = dict(cluster_boot=L.cluster_boot_rho(x_prim, D_PT, pen), leave_one_pen_out=L.leave_one_cluster_out_rho(x_prim, D_PT, pen),
                 pen_table={str(p): dict(n=int((pen == p).sum()), seqs=[s for s, pp in zip(seqs, pen) if pp == p]) for p in np.unique(pen)})
l1 = OUT["D1"]["leave_one_pen_out"]["per_cluster"]
OUT["D1"]["pen3_dropped_rho"] = l1["3"]["rho"]
OUT["D1"]["pen3_rule_triggered"] = bool((np.sign(l1["3"]["rho"]) != np.sign(prim["A1"]["rho"])) or l1["3"]["rho"] < 0.20)
OUT["D1"]["control_leave_one_pen_out"] = L.leave_one_cluster_out_rho(x_prim, D_21, pen)   # EXPLORATORY, added after seeing outcomes

# ----------------------------------------------------------------------------------- 4. decision rule, applied literally (plan section 7)
def decide(prim, v):
    a1 = prim["A1"]; a2 = prim["A2"]; a3 = prim["A3_delta"]; a4 = prim["A4"]
    c = {}
    c["A1_rho>=0.40_and_CIlo>0"] = bool(a1["rho"] >= 0.40 and a1["lo"] > 0)
    c["A2_diff>0_and_CIlo>0"] = bool(a2.get("diff", -9) > 0 and a2.get("lo", -9) > 0)
    c["A3_ctrl_rho<=0.5*rho_A1"] = bool(a3["rho_ctrl"] <= 0.5 * a3["rho_main"])
    c["A3_delta>=0.25"] = bool(a3["delta"] >= 0.25)
    c["A3_control_clearly_weaker"] = bool(c["A3_ctrl_rho<=0.5*rho_A1"] and c["A3_delta>=0.25"])
    c["A3_delta_CIlo>0 (else suggestive only)"] = bool(a3["lo"] > 0)
    c["A4_partial_rho>=0.30"] = bool(a4["partial_rho"] >= 0.30)
    c["A5_strict_rho>=0.30"] = bool(v["strictU"]["A1"]["rho"] >= 0.30)
    c["A5_loose_rho>=0.30"] = bool(v["looseU"]["A1"]["rho"] >= 0.30)
    supported = all(c[k] for k in ["A1_rho>=0.40_and_CIlo>0", "A2_diff>0_and_CIlo>0", "A3_control_clearly_weaker",
                                   "A4_partial_rho>=0.30", "A5_strict_rho>=0.30", "A5_loose_rho>=0.30"])
    not_supported = bool(a1["rho"] < 0.20 or a1["hi"] < 0.40)
    label = "SUPPORTED (exploratory)" if supported else ("NOT SUPPORTED" if not_supported else "INCONCLUSIVE")
    return dict(conditions=c, supported=supported, not_supported_rule_met=not_supported, decision=label,
                not_supported_detail=dict(rho_lt_0p20=bool(a1["rho"] < 0.20), ci_hi_lt_0p40=bool(a1["hi"] < 0.40)))
OUT["decision"] = decide(prim, OUT["A5_variants"])

# print plan results
def pr(lbl, d, keys=("rho", "lo", "hi", "n_invalid")):
    print(f"  {lbl}: " + ", ".join(f"{k}={d[k]:.3f}" if isinstance(d[k], float) else f"{k}={d[k]}" for k in keys if k in d))
print("A1"); pr("rho(S,D_PT)", prim["A1"]); pr("perm", prim["A1_perm"], ("rho", "p_one_sided_pos", "p_two_sided"))
print("A2"); pr("covered-uncovered", prim["A2"], ("n_a", "n_b", "mean_a", "mean_b", "diff", "lo", "hi", "n_invalid"))
print("A3"); pr("rho(S,D_21)", prim["A1_ctrl"]); pr("delta", prim["A3_delta"], ("rho_main", "rho_ctrl", "delta", "lo", "hi", "n_invalid"))
pr("A2 ctrl", prim["A2_ctrl"], ("mean_a", "mean_b", "diff", "lo", "hi"))
print("A4"); pr("partial(night,objects)", prim["A4"], ("partial_rho", "rank_ols_beta", "lo", "hi", "n_invalid")); pr("A4b(+is1080)", prim["A4b"], ("partial_rho", "rank_ols_beta", "lo", "hi", "n_invalid"))
print("A4 ctrl", prim["A4_ctrl"]["partial_rho"], prim["A4b_ctrl"]["partial_rho"])
for k, v in OUT["A5_variants"].items():
    if "A1" in v:
        print(f"A5 {k}: rho={v['A1']['rho']:.3f} [{v['A1']['lo']:.3f},{v['A1']['hi']:.3f}] ctrl={v['A1_ctrl']['rho']:.3f} delta={v['A3_delta']['delta']:.3f} "
              f"A2diff={v['A2'].get('diff', float('nan')):.2f} A4={v['A4']['partial_rho']:.3f} tiers={v['tier_counts_test']}")
    else:
        print(f"A5 {k}: A2 diff={v['A2'].get('diff')} tiers={v['tier_counts_test']}")
print("A6", {k: (round(v['mean'], 2), round(v['lo'], 2), round(v['hi'], 2)) for k, v in a6.items()})
print("D1 cluster boot", OUT["D1"]["cluster_boot"]); print("D1 LOPO", {k: round(v['rho'], 3) for k, v in l1.items()}, "pen3 rule:", OUT["D1"]["pen3_rule_triggered"])
print("DECISION:", OUT["decision"]["decision"]); print(json.dumps(OUT["decision"]["conditions"], indent=1))

# ----------------------------------------------------------------------------------- 5. EXPLORATORY, added after seeing outcomes
EXPLORATORY_ADDED = {}
# E1: the three post hoc students + base student, and pooled
students = {a: px[a].values - HT for a in ["S-PT", "S-PT-mined", "S-PT-rand", "S-PT-rand2"]}
students["pooled_mean_of_4"] = np.mean([students[a] for a in ["S-PT", "S-PT-mined", "S-PT-rand", "S-PT-rand2"]], axis=0)
E1 = {}
for a, d in students.items():
    E1[a] = dict(D_mean=float(d.mean()), D_sd=float(d.std(ddof=1)), D_min=float(d.min()), D_min_seq=seqs[int(d.argmin())],
                 **battery(a, x_prim, tier_prim, d, D_21, with_perm=True))
E1["_pairwise_spearman_between_students_D"] = {
    f"{a}|{b}": L.spearman(students[a], students[b]) for i, a in enumerate(list(students)[:4]) for b in list(students)[i + 1:4]}
E1["_note"] = ("EXPLORATORY, added after seeing outcomes. The four students share the base pool of 2,626 distillation targets (31 whitelisted train "
               "sequences); S-PT-mined / -rand / -rand2 are post hoc variants of S-PT. They are NOT independent evidence: the pairwise Spearman "
               "between their D vectors is reported, and 'pooled' only reduces seed noise, not the n = 25 limit.")
EXPLORATORY_ADDED["E1_post_hoc_students"] = E1
print("\n=== EXPLORATORY (added after seeing outcomes) ===\nE1")
for a in students:
    r = E1[a]; print(f"  {a:18s} D mean {r['D_mean']:.2f} | rho={r['A1']['rho']:.3f} [{r['A1']['lo']:.3f},{r['A1']['hi']:.3f}] p1={r['A1_perm']['p_one_sided_pos']:.3f} "
                     f"| A2 diff={r['A2']['diff']:.2f} [{r['A2']['lo']:.2f},{r['A2']['hi']:.2f}] | A4={r['A4']['partial_rho']:.3f} | delta={r['A3_delta']['delta']:.3f}")

# E2: gain of the PigTrack student over the Edinburgh student (which never saw PigTrack) vs coverage; test and val
G_test = HPT - HEDI
E2 = dict(test=dict(G_mean=float(G_test.mean()), G_sd=float(G_test.std(ddof=1)), spearman_S_vs_G=L.spearman_boot(x_prim, G_test),
                    perm=perm_test(x_prim, G_test),
                    by_tier={t: dict(n=int((tier_prim == t).sum()), G_mean=float(G_test[tier_prim == t].mean()),
                                     G_sd=float(G_test[tier_prim == t].std(ddof=1)) if (tier_prim == t).sum() > 1 else None) for t in tiers_order}))
vseq = list(val.sequence)
G_val = pval.loc[vseq, "S-PT"].values - pval.loc[vseq, "S-Edi"].values
x_val = val.coverage_score.values; tier_val = val.tier.values
E2["val"] = dict(G_mean=float(G_val.mean()), G_sd=float(G_val.std(ddof=1)), spearman_S_vs_G=L.spearman_boot(x_val, G_val), perm=perm_test(x_val, G_val),
                 by_tier={t: dict(n=int((tier_val == t).sum()), G_mean=float(G_val[tier_val == t].mean()),
                                  G_sd=float(G_val[tier_val == t].std(ddof=1)) if (tier_val == t).sum() > 1 else None) for t in tiers_order})
E2["_note"] = ("EXPLORATORY, added after seeing outcomes. S-Edi = the Edinburgh-pig student (no PigTrack targets in its distillation). "
               "A coverage effect would show as a larger S-PT minus S-Edi gain where PigTrack training covered the view. Descriptive; registered contrast C6 is S-PT - S-Edi on the whole test set.")
EXPLORATORY_ADDED["E2_gain_over_Edinburgh_student"] = E2
print("E2 test rho(S,G)=%.3f [%.3f,%.3f]; val rho=%.3f [%.3f,%.3f]" % (E2["test"]["spearman_S_vs_G"]["rho"], E2["test"]["spearman_S_vs_G"]["lo"], E2["test"]["spearman_S_vs_G"]["hi"],
      E2["val"]["spearman_S_vs_G"]["rho"], E2["val"]["spearman_S_vs_G"]["lo"], E2["val"]["spearman_S_vs_G"]["hi"]))
print("   by tier test:", {t: round(v["G_mean"], 2) for t, v in E2["test"]["by_tier"].items()}, "val:", {t: round(v["G_mean"], 2) for t, v in E2["val"]["by_tier"].items()})

# E3: shortfall vs general difficulty; co-variation of the two shortfalls
E3 = dict(rho_DPT_D21=L.spearman_boot(D_PT, D_21), rho_DPT_HOTA_T=L.spearman_boot(D_PT, HT), rho_S_HOTA_T=L.spearman_boot(x_prim, HT),
          rho_S_HOTA_SPT=L.spearman_boot(x_prim, HPT), rho_S_HOTA_S21=L.spearman_boot(x_prim, H21),
          note="EXPLORATORY, added after seeing outcomes. Does the shortfall just follow how hard the sequence is for SAM 3, and does coverage track the teacher's own HOTA?")
EXPLORATORY_ADDED["E3_difficulty_context"] = E3
print("E3", {k: round(v["rho"], 3) for k, v in E3.items() if isinstance(v, dict)})

# E4: official-val, descriptive only (S-PT, S-Edi, S-PT-dp0 exist; no T, no S21)
E4 = {"arms_on_val": sorted(pval.columns.tolist())}
E4["by_tier"] = {}
for t in tiers_order:
    m = tier_val == t
    E4["by_tier"][t] = dict(n=int(m.sum()), **{f"HOTA_{a}_mean": float(pval.loc[vseq, a].values[m].mean()) for a in pval.columns})
E4["spearman_S_vs_HOTA_SPT_val"] = L.spearman_boot(x_val, pval.loc[vseq, "S-PT"].values)
E4["spearman_S_vs_HOTA_SEdi_val"] = L.spearman_boot(x_val, pval.loc[vseq, "S-Edi"].values)
# test-side equivalents for comparison
E4["by_tier_test_for_comparison"] = {t: dict(n=int((tier_prim == t).sum()), HOTA_SPT_mean=float(HPT[tier_prim == t].mean()), HOTA_SEdi_mean=float(HEDI[tier_prim == t].mean()),
                                              HOTA_T_mean=float(HT[tier_prim == t].mean())) for t in tiers_order}
E4["spearman_S_vs_HOTA_SPT_test"] = L.spearman_boot(x_prim, HPT)
E4["spearman_S_vs_HOTA_SEdi_test"] = L.spearman_boot(x_prim, HEDI)
def order(d, key):
    return [t for t, _ in sorted(d.items(), key=lambda kv: -kv[1][key])]
E4["ordering_SPT_val"] = order(E4["by_tier"], "HOTA_S-PT_mean")
E4["ordering_SPT_test"] = order(E4["by_tier_test_for_comparison"], "HOTA_SPT_mean")
E4["ordering_SEdi_val"] = order(E4["by_tier"], "HOTA_S-Edi_mean")
E4["ordering_SEdi_test"] = order(E4["by_tier_test_for_comparison"], "HOTA_SEdi_mean")
E4["ordering_T_test"] = order(E4["by_tier_test_for_comparison"], "HOTA_T_mean")
E4["_note"] = ("EXPLORATORY, added after seeing outcomes; plan section 8.4 pre-specifies no val analysis. Val has no SAM 3 / SAM 2.1 arm, so no shortfall "
               "can be computed; HOTA by tier mixes scene difficulty with coverage.")
EXPLORATORY_ADDED["E4_official_val_descriptive"] = E4
print("E4 val by tier:", {t: round(v["HOTA_S-PT_mean"], 2) for t, v in E4["by_tier"].items()}, "order val", E4["ordering_SPT_val"], "order test", E4["ordering_SPT_test"],
      "T order test", E4["ordering_T_test"])
print("   rho(S, HOTA S-PT) val=%.3f test=%.3f" % (E4["spearman_S_vs_HOTA_SPT_val"]["rho"], E4["spearman_S_vs_HOTA_SPT_test"]["rho"]))

# E5: per-sequence table for the report + figure
per_seq = test[["sequence", "pen", "night", "size", "objects", "coverage_score", "tier", "inliers", "zoom_ratio"]].copy()
per_seq["HOTA_T"] = HT; per_seq["HOTA_S21"] = H21; per_seq["HOTA_SPT"] = HPT; per_seq["HOTA_SEdi"] = HEDI
per_seq["D_PT"] = D_PT; per_seq["D_21"] = D_21
for a in ["S-PT-mined", "S-PT-rand", "S-PT-rand2"]:
    per_seq["D_" + a] = px[a].values - HT
per_seq.to_csv(os.path.join(CA, "coverage_audit_per_sequence.csv"), index=False)

# E6: leave-one-sequence-out for the single most extreme sequence (is rho driven by 0067 / 0049?)
loo = {}
for i, s in enumerate(seqs):
    keep = np.arange(25) != i
    loo[s] = L.spearman(x_prim[keep], D_PT[keep])
E6 = dict(min=float(min(loo.values())), max=float(max(loo.values())), argmin=min(loo, key=loo.get), argmax=max(loo, key=loo.get),
          without_0067=float(loo["pigtrack0067"]), without_0049=float(loo["pigtrack0049"]),
          note="EXPLORATORY, added after seeing outcomes: leave-one-sequence-out rho(S, D_PT); 0067 and 0049 are the extremes the controller already knew.")
EXPLORATORY_ADDED["E6_leave_one_sequence_out"] = E6
print("E6 LOO rho range [%.3f, %.3f]; without 0067 %.3f; without 0049 %.3f" % (E6["min"], E6["max"], E6["without_0067"], E6["without_0049"]))

# E7: dependence / covariate checks for the E2 pattern (gain of S-PT over S-Edi vs coverage), val, test and pooled
night_val = val.night.values.astype(float); obj_val = val.objects.values.astype(float); pen_val = val.pen.values
is1080_val = (val["size"] == "1920x1080").values.astype(float)
sets = {"val_n20": (x_val, G_val, pen_val, night_val, obj_val, is1080_val, tier_val),
        "test_n25": (x_prim, G_test, pen, night, objects, is1080, tier_prim),
        "pooled_n45": (np.concatenate([x_prim, x_val]), np.concatenate([G_test, G_val]), np.concatenate([pen, pen_val]),
                       np.concatenate([night, night_val]), np.concatenate([objects, obj_val]), np.concatenate([is1080, is1080_val]),
                       np.concatenate([tier_prim, tier_val]))}
E7 = {}
for nm, (xx, gg, pp, nn, oo, ii, tt) in sets.items():
    E7[nm] = dict(n=int(len(xx)), spearman=L.spearman_boot(xx, gg), perm=perm_test(xx, gg), cluster_boot_pen=L.cluster_boot_rho(xx, gg, pp),
                  leave_one_pen_out=L.leave_one_cluster_out_rho(xx, gg, pp),
                  partial_night_objects=L.partial_rank_corr_boot(xx, gg, np.column_stack([nn, oo])),
                  partial_night_objects_is1080=L.partial_rank_corr_boot(xx, gg, np.column_stack([nn, oo, ii])),
                  by_tier={t: dict(n=int((tt == t).sum()), G_mean=float(gg[tt == t].mean()),
                                   G_sd=float(gg[tt == t].std(ddof=1)) if (tt == t).sum() > 1 else None) for t in tiers_order},
                  covered_minus_uncovered=L.strat_diff_boot(gg[tt == "covered"], gg[tt == "uncovered"]))
E7["_note"] = ("EXPLORATORY, added after seeing outcomes. Checks whether the E2 pattern survives pen clustering, leave-one-pen-out and the A4 covariates. "
               "Val and test come from the same 9 pens; 'pooled' treats the 45 as one sample. Post hoc, no multiplicity correction.")
EXPLORATORY_ADDED["E7_gain_pattern_dependence_checks"] = E7
for nm in sets:
    e = E7[nm]; print(f"E7 {nm}: rho={e['spearman']['rho']:.3f} [{e['spearman']['lo']:.3f},{e['spearman']['hi']:.3f}] p2={e['perm']['p_two_sided']:.4f} "
                      f"| pen-cluster CI [{e['cluster_boot_pen']['lo']:.3f},{e['cluster_boot_pen']['hi']:.3f}] | LOPO [{e['leave_one_pen_out']['min']:.3f},{e['leave_one_pen_out']['max']:.3f}] "
                      f"| partial(night,obj)={e['partial_night_objects']['partial_rho']:.3f} [{e['partial_night_objects']['lo']:.3f},{e['partial_night_objects']['hi']:.3f}] "
                      f"| +is1080={e['partial_night_objects_is1080']['partial_rho']:.3f} | cov-unc diff={e['covered_minus_uncovered']['diff']:.2f} [{e['covered_minus_uncovered']['lo']:.2f},{e['covered_minus_uncovered']['hi']:.2f}]")
    print("    LOPO:", {k: round(v['rho'], 3) for k, v in e["leave_one_pen_out"]["per_cluster"].items()})

OUT["EXPLORATORY_added_after_seeing_outcomes"] = EXPLORATORY_ADDED
OUT["settings"] = dict(seed=SEED, B=B, n_test=25, n_val=20, ci="percentile 95%", ties="mid-ranks", multiplicity="none")

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, float): return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
json.dump(clean(OUT), open(os.path.join(CA, "coverage_audit_results.json"), "w"), indent=1)
print("\nwrote coverage_audit_results.json")
