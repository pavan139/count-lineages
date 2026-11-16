"""Uncertainty with base-model families as the sampling unit."""
import numpy as np
from scipy.stats import norm, spearmanr
from .stats import ability, item_rest, logistic_fit, residual_similarity


def _codes(families):
    """Integer family codes. None / '' / NaN entries become singleton families."""
    lab = [f"\x00{i}" if (f is None or f == "" or f != f) else str(f) for i, f in enumerate(families)]
    return np.unique(lab, return_inverse=True)[1], np.array([not x.startswith("\x00") for x in lab])


def effective_clusters(families):
    """G* = M^2 / sum(m_g^2): the number of equal-sized families with the same concentration."""
    code, _ = _codes(families)
    m = np.bincount(code)
    return float(len(code) ** 2 / (m ** 2).sum())


def panel_card(families):
    """Lineage composition of a model panel."""
    code, known = _codes(families)
    m = np.bincount(code)
    labelled = np.bincount(code[known]) if known.any() else np.array([0])
    return {"models": int(len(code)), "families": int((labelled > 0).sum()), "unassigned": int((~known).sum()),
            "largest_family": int(m.max()), "largest_family_share": float(m.max() / len(code)),
            "kish_design_effect": float((m ** 2).sum() / len(code)), "effective_clusters": effective_clusters(families)}


def _agg(Mat, code):
    order = np.argsort(code, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(code[order])) + 1]
    return np.add.reduceat(Mat[order], starts, axis=0)


def variance_inflation(X, families, z=None):
    """Per-item ratio of the family-jackknife variance to the delete-one jackknife variance
    for logistic difficulty (intercept) and discrimination (slope on ability).

    Pass `z` computed on items other than those in X to keep the ability proxy
    independent of the audited items. Estimates run low when one family dominates."""
    code, _ = _codes(families)
    z = ability(X) if z is None else z
    _, _, E, P = logistic_fit(X, z)
    Mm = len(z); zc = z[:, None]
    W = P * (1 - P)
    h00 = W.sum(0); h01 = (zc * W).sum(0); h11 = (zc ** 2 * W).sum(0)
    u0, u1 = E, zc * E

    def jk(H00, H01, H11, s0, s1, G):
        A00, A01, A11 = h00 - H00, h01 - H01, h11 - H11
        d = A00 * A11 - A01 ** 2
        good = d > 1e-8
        d = np.where(good, d, 1.0)
        d0 = np.where(good, (A11 * s0 - A01 * s1) / d, 0.0)
        d1 = np.where(good, (-A01 * s0 + A00 * s1) / d, 0.0)
        return (G - 1) / G * (d0 ** 2).sum(0), (G - 1) / G * (d1 ** 2).sum(0)
    i0, i1 = jk(W, zc * W, zc ** 2 * W, u0, u1, Mm)
    G = int(code.max()) + 1
    c0, c1 = jk(_agg(W, code), _agg(zc * W, code), _agg(zc ** 2 * W, code), _agg(u0, code), _agg(u1, code), G)
    ok = (h00 * h11 - h01 ** 2 > 1e-8) & (i0 > 0) & (i1 > 0)
    return {"difficulty": np.where(ok, c0 / np.where(ok, i0, 1), np.nan),
            "discrimination": np.where(ok, c1 / np.where(ok, i1, 1), np.nan)}


def _difficulty(X):
    return logistic_fit(X, ability(X))[0]


def _discrimination(X):
    return logistic_fit(X, ability(X))[1]


STATISTICS = {"item_rest": item_rest, "difficulty": _difficulty, "discrimination": _discrimination,
              "accuracy": lambda X: X.mean(0)}


def family_bootstrap(X, families, statistic="item_rest", B=400, seed=0, level=0.95):
    """Resample whole families; recompute the statistic (and ability) in every resample.

    `statistic` is a name in STATISTICS or a function mapping a response matrix to one value per item.
    Returns the estimate, the bootstrap standard error and a percentile interval."""
    fn = STATISTICS[statistic] if isinstance(statistic, str) else statistic
    code, _ = _codes(families)
    members = [np.flatnonzero(code == g) for g in range(int(code.max()) + 1)]
    rng = np.random.default_rng(seed)
    est = fn(X)
    draws = np.empty((B, len(est)))
    for b in range(B):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(members), size=len(members))])
        draws[b] = fn(X[rows])
    a = (1 - level) / 2
    return {"estimate": est, "se": np.nanstd(draws, axis=0),
            "lo": np.nanquantile(draws, a, axis=0), "hi": np.nanquantile(draws, 1 - a, axis=0)}


def _bh(p, q):
    m = len(p); order = np.argsort(p)
    passed = np.flatnonzero(p[order] <= q * np.arange(1, m + 1) / m)
    out = np.zeros(m, bool)
    if len(passed):
        out[order[: passed.max() + 1]] = True
    return out


def screen_items(X, families, B=400, seed=0, fdr=0.05):
    """Items whose item-rest correlation is significantly negative (one-sided, BH), under three
    standard errors: the Fisher formula, a bootstrap over individual models, and a bootstrap
    over whole families. The model bootstrap is the no-lineage baseline; `se_ratio` is the
    family-level over the model-level standard error."""
    Mm = X.shape[0]
    fz = lambda A: np.arctanh(np.clip(item_rest(A), -0.999, 0.999))
    bm = family_bootstrap(X, np.arange(Mm), fz, B=B, seed=seed)
    bs = family_bootstrap(X, families, fz, B=B, seed=seed + 1)
    zstat = bs["estimate"]; se_fisher = 1 / np.sqrt(Mm - 3)
    ok = np.isfinite(zstat) & np.isfinite(bs["se"]) & (bs["se"] > 0) & np.isfinite(bm["se"]) & (bm["se"] > 0)
    out = {k: np.zeros(len(zstat), bool) for k in ("flag_fisher", "flag_model", "flag_family")}
    neg = zstat[ok] < 0
    out["flag_fisher"][ok] = _bh(norm.cdf(zstat[ok] / se_fisher), fdr) & neg
    out["flag_model"][ok] = _bh(norm.cdf(zstat[ok] / bm["se"][ok]), fdr) & neg
    out["flag_family"][ok] = _bh(norm.cdf(zstat[ok] / bs["se"][ok]), fdr) & neg
    out["item_rest"] = np.tanh(zstat)
    out["se_ratio"] = np.where(ok, bs["se"] / np.where(ok, bm["se"], 1), np.nan)
    return out


def similarity_by_family(X, families):
    """Mean ability-adjusted residual correlation for same-family and different-family pairs."""
    code, known = _codes(families)
    R = residual_similarity(X)
    iu = np.triu_indices(len(code), 1)
    same = (code[:, None] == code[None, :])[iu]
    kn = (known[:, None] & known[None, :])[iu]
    r = R[iu]
    return {"same_family": float(r[same].mean()) if same.any() else float("nan"),
            "different_family": float(r[~same & kn].mean()) if (~same & kn).any() else float("nan"),
            "same_family_pairs": int(same.sum())}


def leave_family_out(X, families, seeds=30, controls=8, seed=0):
    """Rank agreement of item-rest correlations between two halves of the panel when whole
    families are held out, against families shuffled within ability deciles."""
    code, _ = _codes(families)
    Mm = X.shape[0]
    z = ability(X)
    bins = np.searchsorted(np.quantile(z, np.linspace(0, 1, 11)[1:-1]), z)
    rng = np.random.default_rng(seed)

    def arms(labels, order_seed):
        fams = np.random.default_rng(order_seed).permutation(np.unique(labels))
        sizes = np.bincount(labels, minlength=labels.max() + 1)
        a, na, nb = [], 0, 0
        for f in fams:
            if na <= nb:
                a.append(f); na += sizes[f]
            else:
                nb += sizes[f]
        ina = np.isin(labels, a)
        return np.flatnonzero(ina), np.flatnonzero(~ina)

    def agree(A, B, n):
        ra = item_rest(X[rng.choice(A, n, replace=False)]); rb = item_rest(X[rng.choice(B, n, replace=False)])
        ok = np.isfinite(ra) & np.isfinite(rb)
        return spearmanr(ra[ok], rb[ok]).statistic
    real, shuf = [], []
    for s in range(seeds):
        A, Bm = arms(code, s)
        ctl = []
        for _ in range(controls):
            ls = code.copy()
            for b in np.unique(bins):
                idx = np.flatnonzero(bins == b)
                ls[idx] = code[rng.permutation(idx)]
            ctl.append(arms(ls, s))
        n = min([len(A), len(Bm)] + [min(len(a), len(b)) for a, b in ctl])
        real.append(agree(A, Bm, n)); shuf.append(np.mean([agree(a, b, n) for a, b in ctl]))
    real, shuf = np.array(real), np.array(shuf)
    return {"real_families": float(real.mean()), "shuffled_families": float(shuf.mean()),
            "gap": float((shuf - real).mean()), "seeds_with_positive_gap": int((shuf > real).sum()), "seeds": seeds}
