"""Re-audit of family differential item functioning (Mantel-Haenszel DIF).

Published family-DIF analyses compare two named model families with models as
the unit. We run the same procedure on the archive and ask how many items it
flags (a) when group labels are permuted across models, (b) when whole
base-model families that belong to neither named group are split at random
into two super-groups, and (c) for the named contrast. We also check whether
flags replicate across the base models inside one named family.
"""
import json
import sys
import numpy as np
from scipy.stats import chi2
import study

HERE = study.HERE
TASKS = ["arc", "truthfulqa", "winogrande", "gsm8k", "hellaswag"]
LLAMA = ("llama-7b", "llama-13b", "llama-30b", "llama-70b", "llama3-8b", "llama3-70b")
MISTRAL = ("mistral-7b",)
N_GROUP = 200
K = 10


def mh(X, focal, strata, cc=True):
    """Mantel-Haenszel chi-square (continuity corrected), p-value and ETS delta per item."""
    J = X.shape[1]
    sumA = np.zeros(J); sumE = np.zeros(J); sumV = np.zeros(J); num = np.zeros(J); den = np.zeros(J)
    for k in np.unique(strata):
        idx = strata == k
        f = idx & focal; r = idx & ~focal
        nf, nr = f.sum(), r.sum()
        N = nf + nr
        if nf == 0 or nr == 0 or N < 2:
            continue
        A = X[f].sum(0).astype(float); C = X[r].sum(0).astype(float)
        B = nf - A; D = nr - C
        m1 = A + C; m0 = N - m1
        sumA += A; sumE += nf * m1 / N; sumV += nf * nr * m1 * m0 / (N * N * (N - 1))
        num += A * D / N; den += B * C / N
    ok = sumV > 0
    stat = np.where(ok, (np.abs(sumA - sumE) - (0.5 if cc else 0.0)).clip(min=0) ** 2 / np.where(ok, sumV, 1), 0.0)
    p = chi2.sf(stat, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = -2.35 * np.log(num / den)
    return p, delta, ok


def flags(p, delta, ok):
    p = p[ok]; delta = delta[ok]
    m = len(p)
    order = np.argsort(p)
    thresh = 0.05 * np.arange(1, m + 1) / m
    passed = np.flatnonzero(p[order] <= thresh)
    bh = np.zeros(m, bool)
    if len(passed):
        bh[order[: passed.max() + 1]] = True
    big = np.isfinite(delta) & (np.abs(delta) >= 1.5)
    return {"p05": float((p < 0.05).mean()), "bh": float(bh.mean()), "ets_c": float(((p < 0.05) & big).mean())}


def strata_of(score, k=K):
    edges = np.quantile(score, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, score)


def run(task, reps=100):
    X, names = study.load_task(task)
    li, known, lab = study.load_labels(names)
    score = X.mean(1)
    rng = np.random.default_rng(99)
    isL = np.isin(lab, LLAMA); isM = np.isin(lab, MISTRAL)
    out = {"task": task, "n_llama": int(isL.sum()), "n_mistral": int(isM.sum()),
           "llama_base_models": {b: int((lab == b).sum()) for b in LLAMA if (lab == b).sum()}}

    def contrast(idx_f, idx_r, cc=True, k=K):
        idx = np.r_[idx_f, idx_r]
        focal = np.r_[np.ones(len(idx_f), bool), np.zeros(len(idx_r), bool)]
        Xs = X[idx]
        return mh(Xs, focal, strata_of(score[idx], k), cc=cc)

    def sub(pool, n):
        return rng.choice(pool, n, replace=False)
    L, Mi = np.flatnonzero(isL), np.flatnonzero(isM)
    named, perm, perm_nocc, named40 = [], [], [], []
    for _ in range(20):
        a, b = sub(L, N_GROUP), sub(Mi, N_GROUP)
        named.append(flags(*contrast(a, b)))
        named40.append(flags(*contrast(a, b, k=40))["p05"])
        both = rng.permutation(np.r_[a, b])
        perm.append(flags(*contrast(both[:N_GROUP], both[N_GROUP:])))
        perm_nocc.append(flags(*contrast(both[:N_GROUP], both[N_GROUP:], cc=False))["p05"])
    out["perm_no_continuity"] = float(np.mean(perm_nocc)); out["named_40_strata"] = float(np.mean(named40))
    out["ability_gap_sd"] = float(abs(score[L].mean() - score[Mi].mean()) / score.std())
    out["named"] = {k: float(np.mean([d[k] for d in named])) for k in named[0]}
    out["model_permutation"] = {k: float(np.mean([d[k] for d in perm])) for k in perm[0]}
    # placebo: families outside both named groups, split at random at the family level
    # exclude lineages known to descend from a named group (Mistral-7B merges and up-scalings, Code Llama)
    related = np.array([x.startswith("mistral-moe") or x in ("solar-10.7b", "codellama-34b") for x in lab])
    other = np.flatnonzero(~isL & ~isM & ~related)
    out["placebo_excluded_related"] = int(related.sum())
    fams = np.unique(li[other])
    out["placebo_pool"] = {"models": int(len(other)), "families": int(len(fams)),
                           "largest_family": int(np.bincount(li[other]).max())}
    plc = []
    for _ in range(reps):
        order = rng.permutation(fams)
        a, b, na, nb = [], [], 0, 0
        sizes = {f: int((li[other] == f).sum()) for f in fams}
        for f in order:
            if na <= nb:
                a.append(f); na += sizes[f]
            else:
                b.append(f); nb += sizes[f]
        ia = other[np.isin(li[other], a)]; ib = other[np.isin(li[other], b)]
        n = min(N_GROUP, len(ia), len(ib))
        plc.append(flags(*contrast(sub(ia, n), sub(ib, n))))
    out["family_placebo"] = {k: {"mean": float(np.mean([d[k] for d in plc])), "lo": float(np.quantile([d[k] for d in plc], .025)),
                                 "hi": float(np.quantile([d[k] for d in plc], .975))} for k in plc[0]}
    out["named_exceeds_placebo_share"] = float(np.mean([d["p05"] < out["named"]["p05"] for d in plc]))
    # replication across base models inside the named family
    bases = [b for b in LLAMA if (lab == b).sum() >= 45]
    p_all, d_all, ok_all = contrast(L, Mi)
    flagged = ok_all & (p_all < 0.05)
    same = np.ones(len(p_all), bool); every = np.ones(len(p_all), bool)
    for b in bases:
        pb, db, okb = contrast(np.flatnonzero(lab == b), Mi)
        same &= okb & (np.sign(db) == np.sign(d_all))
        every &= okb & (pb < 0.05) & (np.sign(db) == np.sign(d_all))
    # null for the replication statistic: random pseudo-bases of the same sizes drawn from all Llama models
    sizes_b = [int((lab == b).sum()) for b in bases]
    null = []
    for _ in range(20):
        pool = rng.permutation(L); ev = np.ones(len(p_all), bool); pos = 0
        for sz in sizes_b:
            pb, db, okb = contrast(pool[pos:pos + sz], Mi); pos += sz
            ev &= okb & (pb < 0.05) & (np.sign(db) == np.sign(d_all))
        null.append(float(ev[flagged].mean()) if flagged.any() else np.nan)
    out["replication_null"] = float(np.nanmean(null))
    out["replication"] = {"bases": bases, "flagged_pooled": float(flagged.mean()),
                          "share_flagged_same_sign_all_bases": float(same[flagged].mean()) if flagged.any() else None,
                          "share_flagged_significant_all_bases": float(every[flagged].mean()) if flagged.any() else None}
    print(f"{task}: Llama-named {out['n_llama']} models ({len(out['llama_base_models'])} base models) vs Mistral-7B {out['n_mistral']}"
          f" | placebo pool {len(other)} models in {len(fams)} families (largest {out['placebo_pool']['largest_family']})")
    print(f"  items flagged at p<0.05:  model-label permutation {out['model_permutation']['p05']:.3f} | random split of unrelated families "
          f"{out['family_placebo']['p05']['mean']:.3f} [{out['family_placebo']['p05']['lo']:.3f},{out['family_placebo']['p05']['hi']:.3f}] | named families {out['named']['p05']:.3f}")
    print(f"  after BH 5%: perm {out['model_permutation']['bh']:.3f} | placebo {out['family_placebo']['bh']['mean']:.3f} | named {out['named']['bh']:.3f}"
          f"   large (ETS C-like): perm {out['model_permutation']['ets_c']:.3f} | placebo {out['family_placebo']['ets_c']['mean']:.3f} | named {out['named']['ets_c']:.3f}")
    print(f"  replication inside Llama ({bases}): of pooled flags, same sign in every base model {out['replication']['share_flagged_same_sign_all_bases']:.2f},"
          f" significant in every base model {out['replication']['share_flagged_significant_all_bases']:.2f} (random pseudo-bases: {out['replication_null']:.2f})"
          f" | perm without continuity correction {out['perm_no_continuity']:.3f} | named with 40 strata {out['named_40_strata']:.3f} | placebo FDR {out['family_placebo']['bh']['mean']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    res = {t: run(t) for t in (sys.argv[1:] or TASKS)}
    json.dump(res, open(f"{HERE}/study_dif.json", "w"), indent=1)
