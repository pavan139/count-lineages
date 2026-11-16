"""Pilot: effective panel size (N_eff) of an LLM response matrix.

Two estimators, both built on an item split D (discovery) / T (audit):

 spectral  N_eff = M^2 / (M + sum_{i!=k} r^D_ik * r^T_ik)
           r^D, r^T are model-model correlations of ability-adjusted residuals
           computed on disjoint item halves, so item noise does not inflate the
           squared-correlation sum. Threshold-free. Exactly invariant to
           uniform duplication of models.

 cluster   N_eff = M / deff, deff = cluster-robust / heteroskedasticity-robust
           sandwich variance of per-item logistic coefficients (intercept,
           slope on ability), groups discovered on D, items audited on T.
           Control: same statistic with group labels shuffled within score bins.
"""
import json
import sys
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def ability(X):
    s = X.mean(1)
    n = X.shape[1]
    s = (s * n + 0.5) / (n + 1.0)
    z = np.log(s / (1 - s))
    sd = z.std()
    return (z - z.mean()) / (sd if sd > 0 else 1.0)


def logistic_resid(X, z, iters=25, ridge=1e-3):
    """Per-item logistic regression of X[:, j] on (1, z). Returns residuals, P."""
    M, J = X.shape
    Xf = X.astype(np.float64)
    p = np.clip(Xf.mean(0), 1e-3, 1 - 1e-3)
    b0 = np.log(p / (1 - p))
    b1 = np.zeros(J)
    for _ in range(iters):
        eta = np.clip(b0[None, :] + z[:, None] * b1[None, :], -30, 30)
        P = 1 / (1 + np.exp(-eta))
        W = P * (1 - P)
        g0 = (Xf - P).sum(0) - ridge * b0
        g1 = (z[:, None] * (Xf - P)).sum(0) - ridge * b1
        h00 = W.sum(0) + ridge
        h01 = (z[:, None] * W).sum(0)
        h11 = (z[:, None] ** 2 * W).sum(0) + ridge
        det = h00 * h11 - h01 ** 2
        d0 = (h11 * g0 - h01 * g1) / det
        d1 = (-h01 * g0 + h00 * g1) / det
        step = np.maximum(np.abs(d0), np.abs(d1))
        scale = np.minimum(1.0, 1.0 / np.maximum(step, 1e-12))
        b0 += scale * d0
        b1 += scale * d1
        if step.max() < 1e-7:
            break
    eta = np.clip(b0[None, :] + z[:, None] * b1[None, :], -30, 30)
    P = 1 / (1 + np.exp(-eta))
    return Xf - P, P


def row_corr(E):
    E = E - E.mean(1, keepdims=True)
    nrm = np.sqrt((E ** 2).sum(1))
    nrm[nrm == 0] = 1.0
    E = E / nrm[:, None]
    return E @ E.T


def discover(R, threshold=0.35):
    D = 1 - R
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    D = np.clip(D, 0, 2)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=1 - threshold, criterion="distance")


def shuffle_within_bins(labels, z, rng, n_bins=10):
    out = labels.copy()
    edges = np.quantile(z, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.searchsorted(edges, z)
    for k in np.unique(b):
        idx = np.where(b == k)[0]
        out[idx] = labels[rng.permutation(idx)]
    return out


def sandwich_deff(E, P, z, labels):
    """Ratio of cluster-robust to HC0 sandwich variance per item, per coefficient."""
    W = P * (1 - P)
    h00 = W.sum(0)
    h01 = (z[:, None] * W).sum(0)
    h11 = (z[:, None] ** 2 * W).sum(0)
    det = h00 * h11 - h01 ** 2
    ok = det > 1e-8
    u0 = E
    u1 = z[:, None] * E
    # HC0 meat
    a00 = (u0 * u0).sum(0); a01 = (u0 * u1).sum(0); a11 = (u1 * u1).sum(0)
    # cluster meat
    _, inv = np.unique(labels, return_inverse=True)
    G = inv.max() + 1
    S = np.zeros((G, len(labels)))
    S[inv, np.arange(len(labels))] = 1.0
    c0 = S @ u0; c1 = S @ u1
    m00 = (c0 * c0).sum(0); m01 = (c0 * c1).sum(0); m11 = (c1 * c1).sum(0)

    def diag_var(q00, q01, q11):
        # B^-1 Q B^-1 diagonal, B = [[h00,h01],[h01,h11]]
        i00, i01, i11 = h11 / det, -h01 / det, h00 / det
        v0 = i00 * i00 * q00 + 2 * i00 * i01 * q01 + i01 * i01 * q11
        v1 = i01 * i01 * q00 + 2 * i01 * i11 * q01 + i11 * i11 * q11
        return v0, v1

    hc0, hc1 = diag_var(a00, a01, a11)
    cr0, cr1 = diag_var(m00, m01, m11)
    ok &= (hc0 > 0) & (hc1 > 0)
    return {
        "intercept": float(cr0[ok].sum() / hc0[ok].sum()),
        "slope": float(cr1[ok].sum() / hc1[ok].sum()),
        "intercept_median": float(np.median(cr0[ok] / hc0[ok])),
        "slope_median": float(np.median(cr1[ok] / hc1[ok])),
        "intercept_q": [float(q) for q in np.quantile(cr0[ok] / hc0[ok], [0.1, 0.25, 0.5, 0.75, 0.9])],
        "intercept_mean": float(np.mean(cr0[ok] / hc0[ok])),
        "meat_ratio": float(m00[ok].sum() / a00[ok].sum()),
    }


def neff(X, n_splits=4, seed=0, threshold=0.35, n_shuffles=4):
    """X: complete binary M x J. Returns averaged estimates."""
    M, J = X.shape
    keep = (X.mean(0) > 0) & (X.mean(0) < 1)
    X = X[:, keep]
    J = X.shape[1]
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ["spec", "deff_int", "deff_slope", "ctl_int", "ctl_slope",
                           "groups", "largest", "mean_r", "frac_r35"]}
    for _ in range(n_splits):
        perm = rng.permutation(J)
        Dc, Tc = perm[: J // 2], perm[J // 2:]
        for a, b in ((Dc, Tc), (Tc, Dc)):
            zD = ability(X[:, a]); zT = ability(X[:, b])
            ED, _ = logistic_resid(X[:, a], zD)
            ET_own, _ = logistic_resid(X[:, b], zT)
            RD = row_corr(ED); RT = row_corr(ET_own)
            off = ~np.eye(M, dtype=bool)
            s2 = float((RD[off] * RT[off]).sum())
            acc["spec"].append(M * M / (M + max(s2, -0.5 * M)))
            acc["mean_r"].append(float(RD[off].mean()))
            acc["frac_r35"].append(float((RD[off] > 0.35).mean()))
            labels = discover(RD, threshold)
            cnt = np.bincount(labels)
            acc["groups"].append(int((cnt > 0).sum()))
            acc["largest"].append(float(cnt.max() / M))
            # audit items regressed on discovery ability; grouping never saw them
            ET, PT = logistic_resid(X[:, b], zD)
            d = sandwich_deff(ET, PT, zD, labels)
            acc["deff_int"].append(d["intercept_median"]); acc["deff_slope"].append(d["slope_median"])
            ci, cs = [], []
            for _s in range(n_shuffles):
                dd = sandwich_deff(ET, PT, zD, shuffle_within_bins(labels, zD, rng))
                ci.append(dd["intercept_median"]); cs.append(dd["slope_median"])
            acc["ctl_int"].append(float(np.mean(ci))); acc["ctl_slope"].append(float(np.mean(cs)))
    r = {k: float(np.mean(v)) for k, v in acc.items()}
    r["spec_range"] = [float(np.min(acc["spec"])), float(np.max(acc["spec"]))]
    r.update(M=M, J=J,
             neff_spec=r["spec"],
             neff_int=M / r["deff_int"], neff_slope=M / r["deff_slope"],
             neff_ctl_int=M / r["ctl_int"], neff_ctl_slope=M / r["ctl_slope"])
    return r


def sim_2pl(M, J, rng, family=None, sigma=0.0):
    theta = rng.normal(size=M)
    a = rng.lognormal(0, 0.45, size=J)
    b = rng.normal(0, 1.2, size=J)
    eta = a[None, :] * (theta[:, None] - b[None, :])
    if family is not None and sigma > 0:
        G = family.max() + 1
        eta = eta + sigma * rng.normal(size=(G, J))[family]
    return (rng.random((M, J)) < 1 / (1 + np.exp(-eta))).astype(np.int8)


def fmt(name, r, extra=""):
    print(f"{name:26s} M={r['M']:5d} J={r['J']:6d} | spectral {r['neff_spec']:7.1f} "
          f"({r['neff_spec']/r['M']:.2f}M) | cluster int {r['neff_int']:7.1f} slope {r['neff_slope']:7.1f} "
          f"| shuffled-ctl int {r['neff_ctl_int']:7.1f} slope {r['neff_ctl_slope']:7.1f} "
          f"| groups {r['groups']:6.1f} largest {r['largest']:.2f} {extra}", flush=True)


def calibrate():
    rng = np.random.default_rng(7)
    print("== calibration ==")
    X = sim_2pl(300, 2000, rng)
    fmt("iid 2PL (truth 300)", neff(X, n_splits=2))
    X = sim_2pl(100, 600, rng)
    fmt("iid 2PL small (truth 100)", neff(X, n_splits=2))
    base = sim_2pl(100, 2000, rng)
    mult = rng.integers(1, 8, size=100)
    Xd = np.repeat(base, mult, axis=0)
    truth = mult.sum() ** 2 / (mult ** 2).sum()
    fmt(f"unequal copies (Kish {truth:.1f})", neff(Xd, n_splits=2), f"rows={len(Xd)} unique=100")
    fam = np.repeat(np.arange(30), 10)
    for sg in (0.5, 1.0, 2.0):
        X = sim_2pl(300, 2000, rng, family=fam, sigma=sg)
        fmt(f"30 families x10, sigma={sg}", neff(X, n_splits=2), "(bounds: 30..300)")


def real(root):
    out = {}
    print("== real archives ==")
    for name, f in (("core", "data/core_matrix_trimmed.npz"), ("benchmirt", "data/benchmirt_matrix.npz")):
        z = np.load(f"{root}/{f}", allow_pickle=True)
        X, tasks = z["X"], z["tasks"]
        for t in np.unique(tasks):
            sub = X[:, tasks == t]
            rows = (sub >= 0).all(1)
            sub = sub[rows]
            r = neff(sub, n_splits=2 if sub.shape[1] > 5000 else 4)
            out[f"{name}:{t}"] = r
            fmt(f"{name}:{t}", r)
    return out


def duplication(root):
    print("== duplication invariance on real data (ARC) ==")
    z = np.load(f"{root}/data/core_matrix_trimmed.npz", allow_pickle=True)
    X = z["X"][:, z["tasks"] == "arc"]
    X = X[(X >= 0).all(1)]
    rng = np.random.default_rng(3)
    sel = rng.choice(len(X), 300, replace=False)
    for c in (1, 2, 4):
        fmt(f"ARC 300-model subsample x{c}", neff(np.repeat(X[sel], c, axis=0), n_splits=2))


if __name__ == "__main__":
    root = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if which in ("all", "cal"):
        calibrate()
    if which in ("all", "dup"):
        duplication(root)
    if which in ("all", "real"):
        res = real(root)
        json.dump(res, open(sys.argv[3], "w"), indent=1)
