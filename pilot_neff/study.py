"""Main experiments for the lineage paper.

  inflate  <task>   variance inflation with lineage clusters: 20 item splits,
                    family bootstrap interval, jackknife (CR3) correction,
                    label sensitivities, link to item-rest correlation
  sim               known-truth simulation on the real family-size layout,
                    including interval coverage
  transfer <task>   leave-family-out transfer of an item screen vs entry
                    holdout and size/ability-matched shuffled families
"""
import collections
import json
import sys
import numpy as np
from scipy.stats import spearmanr, t as student_t
import neff_pilot as n

import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
ROOT = os.path.join(BASE, "evalens_release_package")
HERE = os.path.join(BASE, "pilot_neff")
LLAMA2_RELEASE = "2023-07-18"


def load_task(task, archive="core"):
    f = "core_matrix_trimmed.npz" if archive == "core" else "benchmirt_matrix.npz"
    z = np.load(f"{ROOT}/data/{f}", allow_pickle=True)
    X = z["X"][:, z["tasks"] == task]
    rows = (X >= 0).all(1)
    X = X[rows]
    names = [str(m) for m, k in zip(z["models"], rows) if k]
    keep = (X.mean(0) > 0) & (X.mean(0) < 1)
    return X[:, keep], names


def load_labels(names, variant="all", labels_file="lineage_labels.json", raw_file="lineage_raw.json"):
    d = json.load(open(f"{HERE}/{labels_file}"))
    fam, tier = d["family"], d["tier"]
    raw = {r["model"]: r for r in json.load(open(f"{HERE}/{raw_file}"))} if variant == "split_llama" else {}
    lab = []
    for i, m in enumerate(names):
        f, t = fam.get(m), tier.get(m)
        if f is None or (variant == "no_name" and t == "name"):
            lab.append(f"?{i}")
            continue
        if variant == "split_llama" and f in ("llama-7b", "llama-13b", "llama-70b"):
            s = m.lower()
            created = (raw.get(m, {}).get("created") or "")[:10]
            if any(k in s for k in ("llama-2", "llama2", "l2-", "-l2")):
                f += "-v2"
            elif created and created < LLAMA2_RELEASE:
                f += "-v1"
            elif created:
                f += "-v2"
            else:
                f += "-unk"
        lab.append(f)
    lab = np.array(lab)
    known = np.array([not x.startswith("?") for x in lab])
    _, li = np.unique(lab, return_inverse=True)
    return li, known, lab


def _agg(Mat, li):
    order = np.argsort(li, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(li[order])) + 1]
    return np.add.reduceat(Mat[order], starts, axis=0)


def variances(E, P, z, li, jackknife=True):
    """Per-item variance estimates for logistic (intercept, slope on z).
    naive: model-based; hc0/cr0: sandwich; hc3/cr3: delete-one / delete-cluster jackknife."""
    Mm = len(z)
    W = P * (1 - P)
    zc = z[:, None]
    h00 = W.sum(0); h01 = (zc * W).sum(0); h11 = (zc ** 2 * W).sum(0)
    det = h00 * h11 - h01 ** 2
    ok = det > 1e-8
    det = np.where(ok, det, 1.0)
    i00, i01, i11 = h11 / det, -h01 / det, h00 / det
    u0, u1 = E, zc * E

    def sand(q00, q01, q11):
        return (i00 * i00 * q00 + 2 * i00 * i01 * q01 + i01 * i01 * q11,
                i01 * i01 * q00 + 2 * i01 * i11 * q01 + i11 * i11 * q11)
    out = {"ok": ok, "naive": (i00, i11)}
    out["hc0"] = sand((u0 * u0).sum(0), (u0 * u1).sum(0), (u1 * u1).sum(0))
    c0, c1 = _agg(u0, li), _agg(u1, li)
    out["cr0"] = sand((c0 * c0).sum(0), (c0 * c1).sum(0), (c1 * c1).sum(0))
    out["meat_ratio"] = float((c0[:, ok] ** 2).sum() / (u0[:, ok] ** 2).sum())
    if jackknife:
        def jk(H00, H01, H11, s0, s1, G):
            A00, A01, A11 = h00 - H00, h01 - H01, h11 - H11
            d = A00 * A11 - A01 ** 2
            good = d > 1e-8
            d = np.where(good, d, 1.0)
            d0 = np.where(good, (A11 * s0 - A01 * s1) / d, 0.0)
            d1 = np.where(good, (-A01 * s0 + A00 * s1) / d, 0.0)
            return (G - 1) / G * (d0 ** 2).sum(0), (G - 1) / G * (d1 ** 2).sum(0)
        out["hc3"] = jk(W, zc * W, zc ** 2 * W, u0, u1, Mm)
        G = int(li.max()) + 1
        out["cr3"] = jk(_agg(W, li), _agg(zc * W, li), _agg(zc ** 2 * W, li), c0, c1, G)
    return out


def ratios(v):
    ok = v["ok"] & (v["hc0"][0] > 0) & (v["hc0"][1] > 0)
    r = {"cr0_int": v["cr0"][0][ok] / v["hc0"][0][ok], "cr0_slope": v["cr0"][1][ok] / v["hc0"][1][ok]}
    if "cr3" in v:
        r["cr3_int"] = v["cr3"][0][ok] / v["hc3"][0][ok]
        r["cr3_slope"] = v["cr3"][1][ok] / v["hc3"][1][ok]
    return r, ok


def item_rest(X):
    Xf = X.astype(np.float64)
    S = Xf.sum(1)
    rest = (S[:, None] - Xf) / (X.shape[1] - 1)
    xc = Xf - Xf.mean(0); rc = rest - rest.mean(0)
    den = np.sqrt((xc ** 2).sum(0) * (rc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xc * rc).sum(0) / den
    r[den == 0] = np.nan
    return r


def summarize(arr):
    q = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"p10": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p90": q[4], "mean": float(arr.mean())}


# --------------------------------------------------------------------------- inflate
def inflate(task, n_splits=20, B=0, max_items_boot=1500, archive="core", labels_file="lineage_labels.json"):
    X, names = load_task(task, archive)
    Mm, J = X.shape
    li, known, lab = load_labels(names, "all", labels_file)
    off = ~np.eye(Mm, dtype=bool)
    Wm = (li[:, None] == li[None, :]) & off
    rng = np.random.default_rng(20261115)
    per = collections.defaultdict(list)
    first = None
    for s in range(n_splits):
        perm = rng.permutation(J)
        D, T = perm[: J // 2], perm[J // 2:]
        zD = n.ability(X[:, D]); zT = n.ability(X[:, T])
        E, P = n.logistic_resid(X[:, T], zD)
        v = variances(E, P, zD, li)
        r, ok = ratios(v)
        for k, a in r.items():
            for kk, val in summarize(a).items():
                per[f"{k}:{kk}"].append(float(val))
        per["pooled_meat"].append(v["meat_ratio"])
        RD = n.row_corr(n.logistic_resid(X[:, D], zD)[0])
        RT = n.row_corr(n.logistic_resid(X[:, T], zT)[0])
        per["assoc"].append(float(1 + (RD[Wm] * RT[Wm]).sum() / Mm))
        per["block_difficulty"].append(float(1 + ((RD[Wm] + RT[Wm]) / 2).sum() / Mm))
        if first is None:
            first = (D, T, zD, r, ok)
    res = {"task": task, "M": Mm, "J": J, "families": int(li.max()) + 1, "known": int(known.sum()),
           "largest_family": int(np.bincount(li).max()),
           "splits": {k: {"mean": float(np.mean(a)), "sd": float(np.std(a)), "min": float(np.min(a)), "max": float(np.max(a))}
                      for k, a in per.items()}}

    D, T, zD, r0, ok0 = first
    # link to the screening statistic
    rir = item_rest(X)[T][ok0]
    fin = np.isfinite(rir)
    rho = spearmanr(r0["cr3_int"][fin], rir[fin]).statistic
    neg = rir < 0
    order = np.argsort(np.where(fin, rir, np.inf))[:100]
    res["screen_link"] = {"spearman_inflation_vs_itemrest": float(rho), "n_negative": int(neg.sum()),
                          "median_inflation_negative": float(np.median(r0["cr3_int"][neg])) if neg.any() else None,
                          "median_inflation_nonnegative": float(np.median(r0["cr3_int"][~neg & fin])),
                          "median_inflation_bottom100": float(np.median(r0["cr3_int"][order]))}
    # label sensitivities (first split)
    sens = {}
    for variant in ("all", "no_name", "split_llama"):
        lv, kv, labv = load_labels(names, variant, labels_file)
        E, P = n.logistic_resid(X[:, T], zD)
        rr, _ = ratios(variances(E, P, zD, lv))
        sens[variant] = {"families": int(lv.max()) + 1, "cr3_int_median": float(np.median(rr["cr3_int"])),
                         "cr3_slope_median": float(np.median(rr["cr3_slope"])), "cr3_int_p90": float(np.quantile(rr["cr3_int"], .9))}
    keepk = known
    Xk = X[keepk]; lk = np.unique(lab[keepk], return_inverse=True)[1]
    zk = n.ability(Xk[:, D]); E, P = n.logistic_resid(Xk[:, T], zk)
    rr, _ = ratios(variances(E, P, zk, lk))
    sens["labelled_only"] = {"M": int(keepk.sum()), "cr3_int_median": float(np.median(rr["cr3_int"])),
                             "cr3_slope_median": float(np.median(rr["cr3_slope"]))}
    big = np.bincount(li).argmax()
    kb = li != big
    Xb = X[kb]; lb = np.unique(li[kb], return_inverse=True)[1]
    zb = n.ability(Xb[:, D]); E, P = n.logistic_resid(Xb[:, T], zb)
    rr, _ = ratios(variances(E, P, zb, lb))
    sens["drop_largest_family"] = {"M": int(kb.sum()), "cr3_int_median": float(np.median(rr["cr3_int"])),
                                   "cr3_slope_median": float(np.median(rr["cr3_slope"]))}
    res["sensitivity"] = sens
    # family bootstrap (first split)
    Tb = T if len(T) <= max_items_boot else rng.choice(T, max_items_boot, replace=False)
    members = [np.flatnonzero(li == g) for g in range(int(li.max()) + 1)]
    G = len(members)
    bs = collections.defaultdict(list)
    for b in range(B):
        draw = rng.integers(0, G, size=G)
        rows = np.concatenate([members[g] for g in draw])
        lb = np.concatenate([np.full(len(members[g]), k) for k, g in enumerate(draw)])
        zb = zD[rows]
        zb = (zb - zb.mean()) / zb.std()
        E, P = n.logistic_resid(X[np.ix_(rows, Tb)], zb)
        rr, _ = ratios(variances(E, P, zb, lb))
        bs["cr3_int_median"].append(float(np.median(rr["cr3_int"])))
        bs["cr3_slope_median"].append(float(np.median(rr["cr3_slope"])))
        bs["cr3_int_p90"].append(float(np.quantile(rr["cr3_int"], .9)))
        bs["has_largest"].append(bool((draw == big).any()))
    res["bootstrap"] = {k: {"lo": float(np.quantile(a, .025)), "hi": float(np.quantile(a, .975)), "median": float(np.median(a))}
                        for k, a in bs.items() if k != "has_largest"} if B else {}
    json.dump(res, open(f"{HERE}/study_inflate_{archive}_{task}.json", "w"), indent=1)
    sp = res["splits"]
    print(f"{task}: M={Mm} J={J} families={res['families']} largest={res['largest_family']}")
    print(f"  difficulty inflation per item (CR3 jackknife): p25 {sp['cr3_int:p25']['mean']:.2f} median {sp['cr3_int:median']['mean']:.2f}"
          f" p90 {sp['cr3_int:p90']['mean']:.2f}"
          f" | CR0 median {sp['cr0_int:median']['mean']:.2f} | pooled {sp['pooled_meat']['mean']:.2f} | split sd of median {sp['cr3_int:median']['sd']:.3f}")
    print(f"  discrimination median (CR3) {sp['cr3_slope:median']['mean']:.2f}"
          f" | association {sp['assoc']['mean']:.2f} (sd {sp['assoc']['sd']:.2f}) | block difficulty {sp['block_difficulty']['mean']:.2f}")
    print("  sensitivity (difficulty median):", {k: round(v["cr3_int_median"], 2) for k, v in sens.items()})
    print("  screen link:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["screen_link"].items()}, flush=True)


# --------------------------------------------------------------------------- sim
def sim(reps=80, J=400, boot_reps=25, sigma=0.9, sizes=None, out_name="study_sim.json"):
    if sizes is None:
        d = json.load(open(f"{HERE}/lineage_labels.json"))
        cnt = collections.Counter(v for v in d["family"].values() if v)
        sizes = sorted(cnt.values(), reverse=True) + [1] * sum(v is None for v in d["family"].values())
    sizes = sorted((int(x) for x in sizes), reverse=True)
    FAM = np.repeat(np.arange(len(sizes)), sizes)
    Mm = len(FAM)
    single = np.arange(Mm)
    print(f"layout from the real panel: M={Mm}, {len(sizes)} families, largest {sizes[:5]}")
    scen = {"null: no families": dict(sigma=0.0, share=0.0, hetero=False),
            "families, unrelated ability": dict(sigma=sigma, share=0.0, hetero=False),
            "families, shared ability": dict(sigma=sigma, share=0.5, hetero=False),
            "families, shared ability, uneven items": dict(sigma=sigma, share=0.5, hetero=True)}
    out = {}
    for name, cfg in scen.items():
        rng = np.random.default_rng(42)
        a = rng.lognormal(0, 0.35, size=J); b = rng.normal(0, 1.0, size=J)
        sj = np.full(J, cfg["sigma"])
        if cfg["hetero"]:
            sj = np.where(rng.random(J) < 0.3, 2.0, 0.3) * (cfg["sigma"] > 0)

        def gen(fam):
            G = fam.max() + 1
            theta = np.sqrt(cfg["share"]) * rng.normal(size=G)[fam] + np.sqrt(1 - cfg["share"]) * rng.normal(size=Mm)
            eta = a[None, :] * (theta[:, None] - b[None, :]) + sj[None, :] * rng.normal(size=(G, J))[fam]
            return (rng.random((Mm, J)) < 1 / (1 + np.exp(-eta))).astype(np.int8)

        def stats(X, li):
            z = n.ability(X)
            E, P = n.logistic_resid(X, z)
            L = np.log(P / (1 - P)); zc = z - z.mean()
            b1 = (zc[:, None] * L).sum(0) / (zc ** 2).sum(); b0 = L.mean(0) - b1 * z.mean()
            v = variances(E, P, z, li)
            Z = E / np.sqrt(np.clip(P * (1 - P), 1e-6, None))
            return b0, b1, item_rest(X), (Z[:, 0::2] * Z[:, 1::2]).mean(0), v, n.row_corr(E)
        dep, ind, ses, est = [[] for _ in range(4)], [[] for _ in range(4)], collections.defaultdict(list), collections.defaultdict(list)
        # a fixed ability scale: centre and spread of logit accuracy in the independent population
        ref = np.concatenate([np.log((s_ := (gen(single).mean(1) * J + 0.5) / (J + 1)) / (1 - s_)) for _ in range(3)])
        mu0, sd0 = float(ref.mean()), float(ref.std())
        depf, indf = [[], []], [[], []]

        def fixed(X):
            s_ = (X.mean(1) * J + 0.5) / (J + 1)
            zf = (np.log(s_ / (1 - s_)) - mu0) / sd0
            _, P = n.logistic_resid(X, zf)
            L = np.log(P / (1 - P)); zc = zf - zf.mean()
            b1 = (zc[:, None] * L).sum(0) / (zc ** 2).sum()
            return L.mean(0) - b1 * zf.mean(), b1
        off = ~np.eye(Mm, dtype=bool); Wm = (FAM[:, None] == FAM[None, :]) & off
        geff = Mm ** 2 / float((np.array(sizes) ** 2).sum())
        tcrit = float(student_t.ppf(0.975, max(geff - 1, 1)))
        members = [np.flatnonzero(FAM == g) for g in range(len(sizes))]
        JB = 120
        boot_lo, boot_hi, boot_est = [[], []], [[], []], [[], []]

        def coef(X, cols):
            z = n.ability(X)
            _, P = n.logistic_resid(X[:, cols], z)
            L = np.log(P / (1 - P)); zc = z - z.mean()
            b1 = (zc[:, None] * L).sum(0) / (zc ** 2).sum()
            return L.mean(0) - b1 * z.mean(), b1
        for r in range(reps):
            Xr = gen(FAM)
            b0, b1, ir, pr, v, R = stats(Xr, FAM)
            est["within_family_r"].append(float(R[Wm].mean()) if Wm.any() else 0.0)
            if r < boot_reps:
                draws = [[], []]
                for _b in range(99):
                    pick = rng.integers(0, len(members), size=len(members))
                    rows = np.concatenate([members[g] for g in pick])
                    c0, c1 = coef(Xr[rows], np.arange(JB))
                    draws[0].append(c0); draws[1].append(c1)
                for kk in (0, 1):
                    a_ = np.array(draws[kk])
                    boot_lo[kk].append(np.quantile(a_, .025, axis=0)); boot_hi[kk].append(np.quantile(a_, .975, axis=0))
                    boot_est[kk].append((b0, b1)[kk][:JB])
            for k, val in enumerate((b0, b1, ir, pr)):
                dep[k].append(val)
            for key in ("naive", "hc0", "cr0", "cr3"):
                ses[key + "0"].append(np.sqrt(v[key][0])); ses[key + "1"].append(np.sqrt(v[key][1]))
            rr, _ = ratios(v)
            for k2 in rr:
                est[k2].append(float(np.median(rr[k2])))
            est["block_difficulty"].append(float(1 + R[Wm].sum() / Mm))
            est["block_assoc"].append(float(1 + (R[Wm] ** 2 - 1.0 / J).sum() / Mm))
            Xi = gen(single)
            b0, b1, ir, pr, _, _ = stats(Xi, single)
            for k, val in enumerate((b0, b1, ir, pr)):
                ind[k].append(val)
            for k, val in enumerate(fixed(Xr)):
                depf[k].append(val)
            for k, val in enumerate(fixed(Xi)):
                indf[k].append(val)
        truth = {}
        for k, nm in enumerate(["difficulty", "discrimination", "item_rest", "association"]):
            vd = np.nanvar(np.array(dep[k]), axis=0); vi = np.nanvar(np.array(ind[k]), axis=0)
            okk = (vi > 0) & np.isfinite(vd)
            truth[nm] = float(np.median(vd[okk] / vi[okk]))
        truth_fixed = {}
        for k, nm in ((0, "difficulty"), (1, "discrimination")):
            vd = np.var(np.array(depf[k]), axis=0); vi = np.var(np.array(indf[k]), axis=0)
            truth_fixed[nm] = float(np.median(vd[vi > 0] / vi[vi > 0]))
        halves = {}
        for k, nm in enumerate(["difficulty", "discrimination", "item_rest", "association"]):
            h_ = []
            for sl in (slice(0, reps // 2), slice(reps // 2, reps)):
                vd = np.nanvar(np.array(dep[k])[sl], axis=0); vi = np.nanvar(np.array(ind[k])[sl], axis=0)
                okk = (vi > 0) & np.isfinite(vd); h_.append(float(np.median(vd[okk] / vi[okk])))
            halves[nm] = h_
        cover, cover_se = {}, {}
        for k, nm in ((0, "difficulty"), (1, "discrimination")):
            arr = np.array(dep[k]); true = arr.mean(0)
            for key in ("naive", "hc0", "cr0", "cr3"):
                se = np.array(ses[key + str(k)])
                cover[f"{nm}:{key}"] = float((np.abs(arr - true) <= 1.96 * se).mean())
            cover[f"{nm}:cr3_t"] = float((np.abs(arr - true) <= tcrit * np.array(ses["cr3" + str(k)])).mean())
            for key, mult in (("naive", 1.96), ("cr3", 1.96), ("cr3", tcrit)):
                per_rep = (np.abs(arr - true) <= mult * np.array(ses[key + str(k)])).mean(1)
                cover_se[f"{nm}:{key}{'_t' if mult != 1.96 else ''}"] = float(per_rep.std() / np.sqrt(len(per_rep)))
            if boot_lo[k]:
                lo, hi = np.array(boot_lo[k]), np.array(boot_hi[k])
                hit = (true[:JB] >= lo) & (true[:JB] <= hi)
                cover[f"{nm}:family_bootstrap"] = float(hit.mean())
                cover_se[f"{nm}:family_bootstrap"] = float(hit.mean(1).std() / np.sqrt(len(hit)))
        e = {k: float(np.mean(v)) for k, v in est.items()}
        out[name] = {"truth": truth, "truth_fixed_scale": truth_fixed, "truth_half_replicates": halves, "estimates_true_labels": e,
                     "coverage95": cover, "coverage_se": cover_se, "models": Mm, "families": len(sizes), "largest": sizes[0],
                     "effective_clusters": geff, "t_crit": tcrit, "sigma": sigma, "reps": reps, "items": J}
        print("  fixed ability scale: difficulty %.2f | discrimination %.2f" % (truth_fixed["difficulty"], truth_fixed["discrimination"]))
        print("  within-family residual correlation %.3f | effective clusters %.1f (t crit %.2f) | CR3+t coverage: difficulty %.3f discrimination %.3f | family bootstrap: %s"
              % (e["within_family_r"], geff, tcrit, cover["difficulty:cr3_t"], cover["discrimination:cr3_t"],
                 {k2.split(":")[0]: round(v2, 3) for k2, v2 in cover.items() if "bootstrap" in k2}))
        print(f"\n-- {name} --")
        print("  TRUE inflation: difficulty %.2f | discrimination %.2f | item-rest %.2f | association %.2f" % tuple(truth.values()))
        print("  estimated (true labels): difficulty CR0 %.2f CR3 %.2f block %.2f | discrimination CR0 %.2f CR3 %.2f | association %.2f"
              % (e["cr0_int"], e["cr3_int"], e["block_difficulty"], e["cr0_slope"], e["cr3_slope"], e["block_assoc"]))
        print("  95%% interval coverage, difficulty: naive %.3f HC0 %.3f CR0 %.3f CR3 %.3f | discrimination: naive %.3f HC0 %.3f CR0 %.3f CR3 %.3f"
              % tuple(cover[f"{nm}:{key}"] for nm in ("difficulty", "discrimination") for key in ("naive", "hc0", "cr0", "cr3")), flush=True)
    json.dump(out, open(f"{HERE}/{out_name}", "w"), indent=1)


# --------------------------------------------------------------------------- transfer
def transfer(task, seeds=30, controls=8, k=100):
    X, names = load_task(task)
    Mm, J = X.shape
    li, known, lab = load_labels(names)
    z = n.ability(X)
    edges = np.quantile(z, np.linspace(0, 1, 11)[1:-1])
    bins = np.searchsorted(edges, z)
    rng = np.random.default_rng(777)

    def family_arms(labels, order_seed):
        r = np.random.default_rng(order_seed)
        fams = r.permutation(np.unique(labels))
        sizes = np.bincount(labels, minlength=labels.max() + 1)
        a, b = [], []
        na = nb = 0
        for f in fams:
            if na <= nb:
                a.append(f); na += sizes[f]
            else:
                b.append(f); nb += sizes[f]
        ina = np.isin(labels, a)
        return np.flatnonzero(ina), np.flatnonzero(~ina)

    def outcome(A, B, nsub, r):
        A = r.choice(A, nsub, replace=False); B = r.choice(B, nsub, replace=False)
        res = collections.defaultdict(list)
        for tr, te in ((A, B), (B, A)):
            ra, rb = item_rest(X[tr]), item_rest(X[te])
            fin = np.isfinite(ra)
            sel = np.argsort(np.where(fin, ra, np.inf))[:k]
            negtr = ra[sel] < 0
            rep = negtr & (rb[sel] < 0)
            res["yield"].append(rep.sum() / k)
            res["cond"].append(rep.sum() / max(negtr.sum(), 1))
            both = np.isfinite(ra) & np.isfinite(rb)
            res["rank"].append(spearmanr(ra[both], rb[both]).statistic)
            res["msd"].append(float(np.mean((X[tr].mean(0) - X[te].mean(0)) ** 2)))
        return {kk: float(np.mean(v)) for kk, v in res.items()}
    rows = []
    for s in range(seeds):
        A, B = family_arms(li, s)
        shuf = []
        for c in range(controls):
            ls = li.copy()
            for bb in np.unique(bins):
                idx = np.flatnonzero(bins == bb)
                ls[idx] = li[rng.permutation(idx)]
            shuf.append(family_arms(ls, s))
        nsub = min([len(A), len(B)] + [min(len(a), len(b)) for a, b in shuf] + [Mm // 2])
        fam_o = outcome(A, B, nsub, rng)
        ctl = [outcome(a, b, nsub, rng) for a, b in shuf]
        p = rng.permutation(Mm)
        ent = outcome(p[: Mm // 2], p[Mm // 2:], nsub, rng)
        rows.append({"family": fam_o, "shuffled": {kk: float(np.mean([c[kk] for c in ctl])) for kk in fam_o},
                     "entry": ent, "n_arm": int(nsub)})
    res = {"task": task, "M": Mm, "J": J, "seeds": seeds}
    print(f"{task}: M={Mm} J={J} arm size ~{int(np.mean([r['n_arm'] for r in rows]))}")
    for kk, nm in (("yield", "negative-repeat yield@100"), ("cond", "conditional repetition"), ("rank", "rank agreement of item-rest corr"), ("msd", "mean sq. difference of item accuracy")):
        f = np.array([r["family"][kk] for r in rows]); c = np.array([r["shuffled"][kk] for r in rows]); e = np.array([r["entry"][kk] for r in rows])
        gap = c - f
        res[kk] = {"family": float(f.mean()), "shuffled": float(c.mean()), "entry": float(e.mean()),
                   "gap_mean": float(gap.mean()), "gap_lo": float(np.quantile(gap, .025)), "gap_hi": float(np.quantile(gap, .975)),
                   "seeds_gap_positive": int((gap > 0).sum()), "ratio_family_over_shuffled": float(f.mean() / c.mean()) if c.mean() else None}
        print(f"  {nm:36s} entry {e.mean():.4f} | shuffled families {c.mean():.4f} | real families {f.mean():.4f} | gap {gap.mean():+.4f}"
              f" [{np.quantile(gap,.025):+.4f},{np.quantile(gap,.975):+.4f}] positive in {(gap>0).sum()}/{seeds} seeds", flush=True)
    json.dump(res, open(f"{HERE}/study_transfer_{task}.json", "w"), indent=1)


# --------------------------------------------------------------------------- second archive
def build_labels(raw_file, models, out_file):
    import lineage_analyze as la
    raw = {r["model"]: r for r in json.load(open(f"{HERE}/{raw_file}"))}

    def parents(m):
        r = raw.get(m, {})
        p = list(r.get("base_model") or [])
        if r.get("adapter_base"):
            p.append(r["adapter_base"])
        return [q for q in p if isinstance(q, str) and "/" in q and q != m]

    def root_of(m, depth=0):
        ps = parents(m)
        return m if (not ps or depth > 6) else root_of(ps[0], depth + 1)
    fam, tier = {}, {}
    for m in models:
        s = la.sig_label(raw.get(m, {}))
        if s:
            fam[m], tier[m] = s, "signature"; continue
        rt = root_of(m)
        lab = (la.sig_label(raw.get(rt, {})) or la.name_label(rt)) if rt != m else None
        if lab:
            fam[m], tier[m] = lab, "declared-root"; continue
        nm = la.name_label(m)
        fam[m], tier[m] = (nm, "name") if nm else (None, "unknown")
    json.dump({"family": fam, "tier": tier}, open(f"{HERE}/{out_file}", "w"), indent=1)
    stat = {"reachable": sum(raw.get(m, {}).get("api_status") == 200 for m in models),
            "config": sum("cfg" in raw.get(m, {}) for m in models),
            "declares_parent": sum(bool(parents(m)) for m in models)}
    return fam, tier, stat


def second_archive():
    z = np.load(f"{ROOT}/data/benchmirt_matrix.npz", allow_pickle=True)
    models = [str(m) for m in z["models"]]
    fam, tier, stat = build_labels("lineage_raw_bm.json", models, "lineage_labels_bm.json")
    tc = collections.Counter(tier.values()); fc = collections.Counter(v for v in fam.values() if v)
    sizes = np.array(list(fc.values()) + [1] * tc["unknown"])
    print(f"BenchMIRT panel {len(models)}: {stat} | label source {dict(tc)} | families {len(fc)}, largest {fc.most_common(6)}")
    print(f"  models in a family of size >=2: {sum(v for v in fc.values() if v > 1)} | Kish design effect {np.sum(sizes**2)/len(models):.2f}"
          f" | effective clusters {len(models)**2/np.sum(sizes**2):.1f}")
    res = {"stat": stat, "tiers": dict(tc), "families": len(fc), "largest": fc.most_common(6), "tasks": {}}
    for t in sorted(set(z["tasks"])):
        X, names = load_task(t, "benchmirt")
        if X.shape[1] < 200:
            continue
        const = (X.std(1) == 0)
        X = X[~const]; names = [m for m, c in zip(names, const) if not c]
        if len(names) < 60:
            print(f"  {t}: skipped, {const.sum()} models at ceiling/floor"); continue
        li, known, lab = load_labels(names, "all", "lineage_labels_bm.json")
        Mm, J = X.shape
        zA = n.ability(X); R = n.row_corr(n.logistic_resid(X, zA)[0])
        iu = np.triu_indices(Mm, 1)
        same = (li[:, None] == li[None, :])[iu]
        rng = np.random.default_rng(5); meds, slopes = [], []
        for s in range(10):
            perm = rng.permutation(J); D, T = perm[: J // 2], perm[J // 2:]
            zD = n.ability(X[:, D]); E, P = n.logistic_resid(X[:, T], zD)
            rr, _ = ratios(variances(E, P, zD, li))
            meds.append(np.median(rr["cr3_int"])); slopes.append(np.median(rr["cr3_slope"]))
        res["tasks"][t] = {"M": Mm, "J": J, "r_same": float(R[iu][same].mean()) if same.any() else None,
                           "r_diff": float(R[iu][~same].mean()), "n_same_pairs": int(same.sum()),
                           "cr3_int_median": float(np.mean(meds)), "cr3_slope_median": float(np.mean(slopes))}
        print(f"  {t:26s} M={Mm:3d} same-family r {res['tasks'][t]['r_same']:+.3f} (n={same.sum()}) vs different {R[iu][~same].mean():+.3f}"
              f" | difficulty inflation median {np.mean(meds):.2f} | discrimination {np.mean(slopes):.2f}", flush=True)
    json.dump(res, open(f"{HERE}/study_second_archive.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inflate":
        for t in sys.argv[2:]:
            inflate(t)
    elif cmd == "sim":
        sim()
    elif cmd == "bm":
        second_archive()
    elif cmd == "transfer":
        for t in sys.argv[2:]:
            transfer(t)
