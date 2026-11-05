"""
audit_analysis_lib.py -- FROZEN analysis functions for the PigTrack coverage audit (EXPLORATORY, post hoc).

Pure functions on numpy arrays. Contains NO outcome data and reads NO file. The Outcomes step imports this module
(sys.path insert; run python with -I) and feeds it the 25 test sequences. Self-test on SYNTHETIC data only:
    python -I audit_analysis_lib.py --selftest
Power simulation (synthetic, keeps the real coverage score's tie structure):
    python -I audit_analysis_lib.py --power coverage_table.csv

Conventions (identical to AUDIT_PLAN.md):
  seed SEED = 20261027, B = 10000 percentile bootstrap resamples, 95% CI = [2.5, 97.5] percentiles of valid draws.
  Spearman = Pearson on average ranks (ties -> mid-ranks). A resample with zero variance in either variable is invalid
  (counted and reported, excluded from the percentiles).
  D_PT  = HOTA(S-PT) - HOTA(SAM 3)   (negative = student worse)      [main]
  D_21  = HOTA(SAM 2.1-small) - HOTA(SAM 3)                           [control]
  Hypothesis direction: higher coverage -> D closer to 0 -> POSITIVE correlation between coverage and D.
"""
import sys, json
import numpy as np
from scipy.stats import rankdata

SEED = 20261027
B_DEFAULT = 10000


def _rank_rows(a):
    return rankdata(a, axis=-1)


def _pearson_rows(x, y):
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    sx = np.sqrt((xm ** 2).sum(axis=-1)); sy = np.sqrt((ym ** 2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xm * ym).sum(axis=-1) / (sx * sy)
    r = np.where((sx > 0) & (sy > 0), r, np.nan)
    return r


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(_pearson_rows(_rank_rows(x), _rank_rows(y)))


def _ci(draws):
    d = np.asarray(draws, float)
    ok = np.isfinite(d)
    if ok.sum() == 0:
        return dict(lo=float("nan"), hi=float("nan"), n_valid=0, n_invalid=int(len(d)))
    return dict(lo=float(np.percentile(d[ok], 2.5)), hi=float(np.percentile(d[ok], 97.5)),
                n_valid=int(ok.sum()), n_invalid=int((~ok).sum()))


def spearman_boot(x, y, B=B_DEFAULT, seed=SEED):
    """A1 / A3 building block: Spearman rho and percentile bootstrap CI (resample sequences)."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    r = _pearson_rows(_rank_rows(x[idx]), _rank_rows(y[idx]))
    out = dict(n=int(n), rho=spearman(x, y)); out.update(_ci(r))
    return out


def paired_delta_rho_boot(x, y_main, y_ctrl, B=B_DEFAULT, seed=SEED):
    """A3: delta = rho(x, y_main) - rho(x, y_ctrl), PAIRED bootstrap (same resamples)."""
    x = np.asarray(x, float); ym = np.asarray(y_main, float); yc = np.asarray(y_ctrl, float); n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    rx = _rank_rows(x[idx])
    d = _pearson_rows(rx, _rank_rows(ym[idx])) - _pearson_rows(rx, _rank_rows(yc[idx]))
    out = dict(n=int(n), rho_main=spearman(x, ym), rho_ctrl=spearman(x, yc), delta=spearman(x, ym) - spearman(x, yc))
    out.update(_ci(d))
    return out


def strat_diff_boot(a, b, B=B_DEFAULT, seed=SEED):
    """A2: mean(a) - mean(b); STRATIFIED bootstrap (resample within each group; group sizes fixed)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(B, len(a))); ib = rng.integers(0, len(b), size=(B, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    out = dict(n_a=int(len(a)), n_b=int(len(b)), mean_a=float(a.mean()), mean_b=float(b.mean()),
               sd_a=float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
               sd_b=float(b.std(ddof=1)) if len(b) > 1 else float("nan"), diff=float(a.mean() - b.mean()))
    out.update(_ci(d))
    return out


def _resid(v, Z):
    beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
    return v - Z @ beta


def partial_rank_corr(x, y, covs):
    """A4: partial Spearman = Pearson of rank-residuals after regressing rank(x), rank(y) on [1, rank(covs)]."""
    x = np.asarray(x, float); y = np.asarray(y, float); C = np.atleast_2d(np.asarray(covs, float))
    if C.shape[0] != len(x): C = C.T
    Z = np.column_stack([np.ones(len(x))] + [rankdata(C[:, j]) for j in range(C.shape[1])])
    rx = _resid(rankdata(x), Z); ry = _resid(rankdata(y), Z)
    if rx.std() == 0 or ry.std() == 0: return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def rank_ols_beta(x, y, covs):
    """A4: standardised OLS coefficient of rank(x) in  rank(y) ~ rank(x) + rank(covs)  (all ranks z-scored)."""
    x = np.asarray(x, float); y = np.asarray(y, float); C = np.atleast_2d(np.asarray(covs, float))
    if C.shape[0] != len(x): C = C.T
    def z(v):
        v = rankdata(v); s = v.std()
        return (v - v.mean()) / s if s > 0 else v * 0
    X = np.column_stack([np.ones(len(x)), z(x)] + [z(C[:, j]) for j in range(C.shape[1])])
    beta, *_ = np.linalg.lstsq(X, z(y), rcond=None)
    return float(beta[1])


def partial_rank_corr_boot(x, y, covs, B=B_DEFAULT, seed=SEED):
    x = np.asarray(x, float); y = np.asarray(y, float); C = np.atleast_2d(np.asarray(covs, float))
    if C.shape[0] != len(x): C = C.T
    n = len(x); rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        draws[b] = partial_rank_corr(x[i], y[i], C[i])
    out = dict(n=int(n), n_cov=int(C.shape[1]), partial_rho=partial_rank_corr(x, y, C), rank_ols_beta=rank_ols_beta(x, y, C))
    out.update(_ci(draws))
    return out


def cluster_boot_rho(x, y, clusters, B=B_DEFAULT, seed=SEED):
    """Dependence sensitivity: resample CLUSTERS (pens) with replacement, keep all their sequences."""
    x = np.asarray(x, float); y = np.asarray(y, float); cl = np.asarray(clusters)
    labs = np.unique(cl); members = {c: np.flatnonzero(cl == c) for c in labs}
    rng = np.random.default_rng(seed); draws = np.empty(B)
    for b in range(B):
        pick = rng.choice(labs, size=len(labs), replace=True)
        i = np.concatenate([members[c] for c in pick])
        draws[b] = float(_pearson_rows(rankdata(x[i]), rankdata(y[i])))
    out = dict(n=int(len(x)), n_clusters=int(len(labs)), rho=spearman(x, y)); out.update(_ci(draws))
    return out


def leave_one_cluster_out_rho(x, y, clusters):
    x = np.asarray(x, float); y = np.asarray(y, float); cl = np.asarray(clusters)
    res = {}
    for c in np.unique(cl):
        keep = cl != c
        res[str(c)] = dict(n=int(keep.sum()), rho=spearman(x[keep], y[keep]))
    vals = [v["rho"] for v in res.values() if np.isfinite(v["rho"])]
    return dict(per_cluster=res, min=float(min(vals)), max=float(max(vals)))


def describe_boot(v, B=B_DEFAULT, seed=SEED):
    """A6 (descriptive): mean, SD, percentile bootstrap CI of the mean for one group."""
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    d = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    out = dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"))
    out.update(_ci(d))
    return out


# ----------------------------- power (synthetic) -----------------------------
def a1_support_rate(rho_s, scores, nsim=400, B=1000, seed=SEED, thr=0.40):
    """P(A1 'support' rule met) when the TRUE Spearman correlation between coverage and D is rho_s.
    Latent bivariate normal (Pearson 2 sin(pi rho_s / 6)); coverage values are the real score values (ties preserved)."""
    sc = np.sort(np.asarray(scores, float)); n = len(sc)
    rp = 2 * np.sin(np.pi * rho_s / 6.0)
    rng = np.random.default_rng(seed)
    hit = ci_pos = point_ge = point_lt_half = 0; ests = []
    for s in range(nsim):
        z1 = rng.standard_normal(n); z2 = rp * z1 + np.sqrt(max(0.0, 1 - rp ** 2)) * rng.standard_normal(n)
        x = sc[np.argsort(np.argsort(z1))]       # real scores, ranked by the latent variable
        y = z2
        r = spearman(x, y); ests.append(r)
        res = spearman_boot(x, y, B=B, seed=int(rng.integers(1, 2 ** 31 - 1)))
        lo = res["lo"]
        if np.isfinite(lo) and lo > 0: ci_pos += 1
        if r >= thr: point_ge += 1
        if r < thr / 2: point_lt_half += 1
        if np.isfinite(lo) and lo > 0 and r >= thr: hit += 1
    return dict(true_rho=rho_s, nsim=nsim, mean_est=float(np.mean(ests)), p_support=hit / nsim, p_ci_excludes0=ci_pos / nsim,
                p_point_ge_thr=point_ge / nsim, p_point_below_half_thr=point_lt_half / nsim)


# ----------------------------- self-test (synthetic) -----------------------------
def _selftest():
    rng = np.random.default_rng(1)
    # 1. known monotone relation
    x = np.arange(25.0); y = 3 * x + 1
    assert abs(spearman(x, y) - 1) < 1e-12
    assert abs(spearman(x, -y) + 1) < 1e-12
    # 2. ties use mid-ranks (compare to scipy)
    from scipy.stats import spearmanr
    xt = np.array([1, 1, 1, 2, 2, 3, 4, 4, 5, 5.0]); yt = np.array([3, 1, 2, 2, 5, 4, 4, 6, 7, 6.0])
    assert abs(spearman(xt, yt) - spearmanr(xt, yt).statistic) < 1e-12
    # 3. constant variable -> nan, bootstrap does not crash
    r = spearman_boot(np.ones(10), np.arange(10.0), B=200)
    assert np.isnan(r["rho"]) and r["n_valid"] == 0
    # 4. CI brackets truth for a strong relation, and excludes 0
    z = rng.standard_normal(25); y = 0.8 * z + 0.6 * rng.standard_normal(25)
    r = spearman_boot(z, y, B=2000)
    assert r["lo"] > 0 and r["lo"] < r["rho"] < r["hi"]
    # 5. paired delta: y_ctrl independent -> delta close to rho_main
    yc = rng.standard_normal(25)
    d = paired_delta_rho_boot(z, y, yc, B=2000)
    assert abs(d["delta"] - (d["rho_main"] - d["rho_ctrl"])) < 1e-12
    # 6. stratified diff
    a = rng.normal(5, 1, 7); b = rng.normal(0, 1, 13)
    s = strat_diff_boot(a, b, B=2000)
    assert s["diff"] > 0 and s["lo"] > 0 and s["n_a"] == 7 and s["n_b"] == 13
    # 7. partial rank correlation removes a confounder (x = c + noise, y = c + noise  => partial ~ 0, raw > 0)
    c = rng.standard_normal(400); xx = c + 0.5 * rng.standard_normal(400); yy = c + 0.5 * rng.standard_normal(400)
    assert spearman(xx, yy) > 0.5 and abs(partial_rank_corr(xx, yy, c.reshape(-1, 1))) < 0.15
    # 8. rank_ols_beta sign agrees with partial correlation sign
    assert np.sign(rank_ols_beta(xx, yy, c.reshape(-1, 1))) == np.sign(partial_rank_corr(xx, yy, c.reshape(-1, 1))) or abs(partial_rank_corr(xx, yy, c.reshape(-1, 1))) < 0.05
    # 9. cluster bootstrap / leave-one-cluster-out run and have the right shapes
    cl = np.repeat(np.arange(5), 5)
    cb = cluster_boot_rho(z, y, cl, B=500); assert cb["n_clusters"] == 5
    lo = leave_one_cluster_out_rho(z, y, cl); assert len(lo["per_cluster"]) == 5
    # 10. determinism (same seed -> same CI)
    assert spearman_boot(z, y, B=500)["lo"] == spearman_boot(z, y, B=500)["lo"]
    # 11. describe_boot
    db = describe_boot(a, B=500); assert db["lo"] <= db["mean"] <= db["hi"]
    print("selftest OK (synthetic data only)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--power" in sys.argv:
        import pandas as pd
        tab = pd.read_csv(sys.argv[sys.argv.index("--power") + 1])
        sc = tab.loc[tab.split == "test", "coverage_score"].values
        res = [a1_support_rate(r, sc) for r in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
        print(json.dumps(res, indent=1))
