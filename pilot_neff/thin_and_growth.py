"""(1) Does keeping one model per family fix naive intervals, and what does it cost?
(2) How did the archive's lineage diversity grow as entries were added?"""
import collections
import json
import numpy as np
import neff_pilot as n
import study

HERE = study.HERE


def wfit(X, z, w, iters=25):
    """Weighted per-item logistic fit; returns intercept at the weighted mean of z, and slope."""
    Xf = X.astype(float); wc = w[:, None]; zc = z[:, None]
    p = np.clip((wc * Xf).sum(0) / w.sum(), 1e-3, 1 - 1e-3)
    b0 = np.log(p / (1 - p)); b1 = np.zeros(X.shape[1])
    for _ in range(iters):
        P = 1 / (1 + np.exp(-np.clip(b0[None, :] + zc * b1[None, :], -30, 30)))
        W = wc * P * (1 - P)
        g0 = (wc * (Xf - P)).sum(0); g1 = (wc * zc * (Xf - P)).sum(0)
        h00 = W.sum(0) + 1e-3; h01 = (zc * W).sum(0); h11 = (zc ** 2 * W).sum(0) + 1e-3
        det = h00 * h11 - h01 ** 2
        d0 = (h11 * g0 - h01 * g1) / det; d1 = (-h01 * g0 + h00 * g1) / det
        step = np.maximum(np.abs(d0), np.abs(d1)); sc = np.minimum(1.0, 1.0 / np.maximum(step, 1e-12))
        b0 += sc * d0; b1 += sc * d1
        if step.max() < 1e-7:
            break
    return b0, b1


def thinning(reps=80, J=400, sigma=1.1):
    d = json.load(open(f"{HERE}/lineage_labels.json"))
    cnt = collections.Counter(v for v in d["family"].values() if v)
    sizes = sorted(cnt.values(), reverse=True) + [1] * sum(v is None for v in d["family"].values())
    FAM = np.repeat(np.arange(len(sizes)), sizes); Mm = len(FAM)
    first = np.r_[0, np.cumsum(sizes)[:-1]]
    out = {}
    for name, share in (("families, unrelated ability", 0.0), ("families, shared ability", 0.5)):
        rng = np.random.default_rng(42)
        a = rng.lognormal(0, 0.35, size=J); b = rng.normal(0, 1.0, size=J)
        est = {"full": [[], []], "thin": [[], []], "weighted": [[], []]}; se = [[], []]
        wts = 1.0 / np.array(sizes, float)[FAM]
        for r in range(reps):
            G = FAM.max() + 1
            theta = np.sqrt(share) * rng.normal(size=G)[FAM] + np.sqrt(1 - share) * rng.normal(size=Mm)
            eta = a[None, :] * (theta[:, None] - b[None, :]) + sigma * rng.normal(size=(G, J))[FAM]
            X = (rng.random((Mm, J)) < 1 / (1 + np.exp(-eta))).astype(np.int8)
            for key, Xs in (("full", X), ("thin", X[first])):
                z = n.ability(Xs)
                E, P = n.logistic_resid(Xs, z)
                L = np.log(P / (1 - P)); zc = z - z.mean()
                b1 = (zc[:, None] * L).sum(0) / (zc ** 2).sum(); b0 = L.mean(0) - b1 * z.mean()
                est[key][0].append(b0); est[key][1].append(b1)
                if key == "thin":
                    v = study.variances(E, P, z, np.arange(len(z)), jackknife=False)
                    se[0].append(np.sqrt(v["naive"][0])); se[1].append(np.sqrt(v["naive"][1]))
            s_ = (X.mean(1) * J + 0.5) / (J + 1); lz = np.log(s_ / (1 - s_))
            mu = (wts * lz).sum() / wts.sum(); sdw = np.sqrt((wts * (lz - mu) ** 2).sum() / wts.sum())
            wb0, wb1 = wfit(X, (lz - mu) / sdw, wts)
            est["weighted"][0].append(wb0); est["weighted"][1].append(wb1)
        res = {}
        for k, nm in ((0, "difficulty"), (1, "discrimination")):
            full = np.array(est["full"][k]); thin = np.array(est["thin"][k])
            wt = np.array(est["weighted"][k])
            res[nm] = {"naive_coverage_thinned": float((np.abs(thin - thin.mean(0)) <= 1.96 * np.array(se[k])).mean()),
                       "variance_thinned_over_full": float(np.median(thin.var(0) / full.var(0))),
                       "variance_weighted_over_full": float(np.median(wt.var(0) / full.var(0)))}
        out[name] = res
        print(f"{name}: one model per family ({len(sizes)} of {Mm} models)")
        for nm, v in res.items():
            print(f"   {nm}: naive 95% coverage on thinned panel {v['naive_coverage_thinned']:.3f} | true variance, thinned / full panel {v['variance_thinned_over_full']:.2f}"
                  f" | equal-family-weight full panel / full panel {v['variance_weighted_over_full']:.2f}")
    json.dump(out, open(f"{HERE}/study_thinning.json", "w"), indent=1)


def growth():
    raw = {r["model"]: r for r in json.load(open(f"{HERE}/lineage_raw.json"))}
    lab = json.load(open(f"{HERE}/lineage_labels.json"))["family"]
    rows = sorted((raw[m]["created"][:10], m) for m in lab if raw.get(m, {}).get("created"))
    seen = collections.Counter(); curve = []
    for i, (date, m) in enumerate(rows, 1):
        seen[lab[m] or f"?{m}"] += 1
        sz = np.array(list(seen.values()))
        curve.append({"date": date, "entries": i, "families": int(len(sz)), "effective_clusters": float(i ** 2 / (sz ** 2).sum())})
    json.dump(curve, open(f"{HERE}/study_growth.json", "w"))
    print(f"dated entries {len(rows)}: {rows[0][0]} to {rows[-1][0]}")
    for k in (50, 100, 200, 400, 600, 800, len(rows)):
        c = curve[k - 1]
        print(f"   after {c['entries']:4d} entries ({c['date']}): {c['families']:3d} families, effective clusters {c['effective_clusters']:.1f}")
    print("   maximum effective clusters:", round(max(c["effective_clusters"] for c in curve), 1))


if __name__ == "__main__":
    growth()
    thinning()
