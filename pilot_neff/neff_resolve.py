"""Resolve the spectral-vs-cluster disagreement with a known-truth experiment.

Truth: simulate panels where the population is defined, so the real variance
inflation of each item statistic (dependent panel / independent panel) is known
by Monte Carlo. Then ask which estimator predicts it.

Scenarios separate two things that look alike in one response matrix:
  families  - models that are relatives (shared item-level shocks)
  2D        - independent models with a second ability dimension
"""
import json
import sys
import numpy as np
import neff_pilot as n

SIZES = [40, 25, 15, 10, 10] + [5] * 8 + [2] * 20 + [1] * 120   # 300 models
FAM = np.repeat(np.arange(len(SIZES)), SIZES)
M = len(FAM)
KISH = (np.array(SIZES) ** 2).sum() / M        # design effect if relatives were exact copies


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def gen(rng, a, b, c, fam, sigma, tau):
    J = len(a)
    theta = rng.normal(size=M)
    eta = a[None, :] * (theta[:, None] - b[None, :])
    if sigma > 0:
        eta = eta + sigma * rng.normal(size=(fam.max() + 1, J))[fam]
    if tau > 0:
        eta = eta + tau * rng.normal(size=M)[:, None] * c[None, :]
    return (rng.random((M, J)) < sigmoid(eta)).astype(np.int8)


def item_stats(X):
    """Four statistics per item (or item pair) computed on the whole panel."""
    Xf = X.astype(float)
    z = n.ability(X)
    E, P = n.logistic_resid(X, z)
    W = P * (1 - P)
    # recover coefficients from the fitted probabilities: logit P = b0 + b1 z
    L = np.log(P / (1 - P))
    zc = z - z.mean()
    b1 = (zc[:, None] * L).sum(0) / (zc ** 2).sum()
    b0 = L.mean(0) - b1 * z.mean()
    S = Xf.sum(1)
    rest = (S[:, None] - Xf) / (X.shape[1] - 1)
    xc = Xf - Xf.mean(0); rc = rest - rest.mean(0)
    den = np.sqrt((xc ** 2).sum(0) * (rc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ir = (xc * rc).sum(0) / den
    Z = E / np.sqrt(np.clip(W, 1e-6, None))
    pair = (Z[:, 0::2] * Z[:, 1::2]).mean(0)
    return b0, b1, ir, pair


def estimators(X, rng, labels_true, thr=0.35):
    J = X.shape[1]
    perm = rng.permutation(J)
    out = {k: [] for k in ["spec_all", "block2", "block1", "sand_int", "sand_slope",
                           "or_block2", "or_block1", "or_sand_int", "or_sand_slope"]}
    off = ~np.eye(M, dtype=bool)
    for D, T in ((perm[: J // 2], perm[J // 2:]), (perm[J // 2:], perm[: J // 2])):
        zD = n.ability(X[:, D]); zT = n.ability(X[:, T])
        RD = n.row_corr(n.logistic_resid(X[:, D], zD)[0])
        RT = n.row_corr(n.logistic_resid(X[:, T], zT)[0])
        out["spec_all"].append(1 + (RD[off] * RT[off]).sum() / M)
        ET, PT = n.logistic_resid(X[:, T], zD)
        for tag, lab in (("", n.discover(RD, thr)), ("or_", labels_true)):
            Wm = (lab[:, None] == lab[None, :]) & off
            out[tag + "block2"].append(1 + (RT[Wm] ** 2 - 1.0 / len(T)).sum() / M)
            out[tag + "block1"].append(1 + RT[Wm].sum() / M)
            d = n.sandwich_deff(ET, PT, zD, lab)
            out[tag + "sand_int"].append(d["intercept_median"])
            out[tag + "sand_slope"].append(d["slope_median"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def scenario(name, sigma, tau, J=800, reps=100, est_reps=12, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.lognormal(0, 0.35, size=J); b = rng.normal(0, 1.0, size=J); c = rng.normal(size=J)
    single = np.arange(M)
    dep = [[] for _ in range(4)]; ind = [[] for _ in range(4)]; est = []
    for r in range(reps):
        X = gen(rng, a, b, c, FAM, sigma, tau)
        for k, v in enumerate(item_stats(X)):
            dep[k].append(v)
        if r < est_reps:
            est.append(estimators(X, rng, FAM))
        Xi = gen(rng, a, b, c, single, sigma, tau)
        for k, v in enumerate(item_stats(Xi)):
            ind[k].append(v)
    truth = {}
    for k, nm in enumerate(["intercept", "slope", "item_rest", "item_pair"]):
        vd = np.nanvar(np.array(dep[k]), axis=0); vi = np.nanvar(np.array(ind[k]), axis=0)
        ok = (vi > 0) & np.isfinite(vd)
        truth[nm] = float(np.median(vd[ok] / vi[ok]))
    e = {k: float(np.mean([x[k] for x in est])) for k in est[0]}
    print(f"\n-- {name} (families sigma={sigma}, second dimension tau={tau}) --")
    print("TRUE variance inflation:  difficulty %.2f | discrimination %.2f | item-rest corr %.2f | item-item corr %.2f"
          % (truth["intercept"], truth["slope"], truth["item_rest"], truth["item_pair"]))
    print("first-moment estimators:  sandwich int %.2f slope %.2f | block-Kish %.2f   (true labels: %.2f, %.2f | %.2f)"
          % (e["sand_int"], e["sand_slope"], e["block1"], e["or_sand_int"], e["or_sand_slope"], e["or_block1"]))
    print("second-moment estimators: block-spectral %.2f (true labels %.2f) | full spectral %.2f"
          % (e["block2"], e["or_block2"], e["spec_all"]), flush=True)
    return {"truth": truth, "est": e}


def simulate(out):
    print(f"300 models: family sizes {sorted(set(SIZES), reverse=True)}; exact-copy design effect would be {KISH:.2f}")
    res = {}
    res["A strong families"] = scenario("A strong families", 2.0, 0.0)
    res["B weak families"] = scenario("B weak families", 1.0, 0.0)
    res["C second dimension only"] = scenario("C second dimension only", 0.0, 0.7)
    res["D strong second dimension only"] = scenario("D strong second dimension only", 0.0, 1.5)
    res["E families + second dimension"] = scenario("E families + second dimension", 2.0, 0.7)
    json.dump(res, open(out, "w"), indent=1)


def resolved(X, thr=0.35, n_splits=3, seed=0):
    Mm = X.shape[0]
    keep = (X.mean(0) > 0) & (X.mean(0) < 1)
    X = X[:, keep]; J = X.shape[1]
    rng = np.random.default_rng(seed)
    off = ~np.eye(Mm, dtype=bool)
    acc = {k: [] for k in ["d1_sand_int", "d1_sand_slope", "d1_block", "d2_block", "d_spec", "groups"]}
    for _ in range(n_splits):
        perm = rng.permutation(J)
        for D, T in ((perm[: J // 2], perm[J // 2:]), (perm[J // 2:], perm[: J // 2])):
            zD = n.ability(X[:, D]); zT = n.ability(X[:, T])
            RD = n.row_corr(n.logistic_resid(X[:, D], zD)[0])
            RT = n.row_corr(n.logistic_resid(X[:, T], zT)[0])
            lab = n.discover(RD, thr)
            Wm = (lab[:, None] == lab[None, :]) & off
            ET, PT = n.logistic_resid(X[:, T], zD)
            d = n.sandwich_deff(ET, PT, zD, lab)
            acc["d1_sand_int"].append(d["intercept_median"]); acc["d1_sand_slope"].append(d["slope_median"])
            acc["d1_block"].append(1 + RT[Wm].sum() / Mm)
            acc["d2_block"].append(1 + (RT[Wm] ** 2 - 1.0 / len(T)).sum() / Mm)
            acc["d_spec"].append(1 + (RD[off] * RT[off]).sum() / Mm)
            acc["groups"].append(len(np.unique(lab)))
    r = {k: float(np.mean(v)) for k, v in acc.items()}
    r["M"] = Mm; r["J"] = J
    return r


def real(root, out):
    res = {}
    print("N_eff by estimand (M / design effect). first = difficulty & discrimination, second = item-item association")
    for name, f in (("core", "data/core_matrix_trimmed.npz"), ("benchmirt", "data/benchmirt_matrix.npz")):
        z = np.load(f"{root}/{f}", allow_pickle=True)
        X, tasks = z["X"], z["tasks"]
        for t in np.unique(tasks):
            sub = X[:, tasks == t]; sub = sub[(sub >= 0).all(1)]
            row = {}
            for thr in (0.2, 0.35, 0.5):
                row[thr] = resolved(sub, thr, n_splits=1 if sub.shape[1] > 5000 else 3)
            r = row[0.35]; Mm = r["M"]
            first = lambda q: Mm / max(q["d1_sand_int"], q["d1_sand_slope"])
            diffuse = 1 - (r["d2_block"] - 1) / max(r["d_spec"] - 1, 1e-9)
            print(f"{name}:{t:26s} M={Mm:5d} | first {first(r):6.1f}  (thr .2/.5: {first(row[0.2]):6.1f}/{first(row[0.5]):6.1f})"
                  f" | second {Mm / r['d2_block']:6.1f} (thr .2/.5: {Mm / row[0.2]['d2_block']:6.1f}/{Mm / row[0.5]['d2_block']:6.1f})"
                  f" | groups {r['groups']:5.0f} | diffuse share of spectral {diffuse:4.0%}", flush=True)
            res[f"{name}:{t}"] = {str(k): v for k, v in row.items()}
    json.dump(res, open(out, "w"), indent=1)


def ancestry(root):
    z = np.load(f"{root}/data/core_matrix_trimmed.npz", allow_pickle=True)
    X, tasks, models = z["X"], z["tasks"], [str(m) for m in z["models"]]
    anc = {d["model"]: d for d in json.load(open(f"{root}/data/ancestry.json"))}

    def base(m):
        b = anc.get(m, {}).get("base_model")
        if isinstance(b, list):
            b = b[0] if b else None
        return b
    print("\n== do response-defined groups match declared ancestry? (ARC, threshold 0.35) ==")
    sub = X[:, tasks == "arc"]; rows = (sub >= 0).all(1); sub = sub[rows]
    names = [m for m, k in zip(models, rows) if k]
    zz = n.ability(sub)
    R = n.row_corr(n.logistic_resid(sub[:, (sub.mean(0) > 0) & (sub.mean(0) < 1)], zz)[0])
    lab = n.discover(R, 0.35)
    bases = [base(m) for m in names]; arch = [anc.get(m, {}).get("arch") for m in names]
    have = np.array([b is not None for b in bases])
    print(f"models with a declared base model: {have.sum()} of {len(names)}")
    iu = np.triu_indices(len(names), 1)
    same_grp = (lab[:, None] == lab[None, :])[iu]
    both = (have[:, None] & have[None, :])[iu]
    same_base = np.array([[bases[i] == bases[j] for j in range(len(names))] for i in range(len(names))])[iu]
    same_arch = np.array([[arch[i] is not None and arch[i] == arch[j] for j in range(len(names))] for i in range(len(names))])[iu]
    for nm, mask in (("pairs in the same response group", same_grp), ("all other pairs", ~same_grp)):
        mm = mask & both
        print(f"{nm:34s}: share same declared base model {same_base[mm].mean():.3f} (n={mm.sum()}), share same architecture {same_arch[mask].mean():.3f}")
    zb = np.load(f"{root}/data/benchmirt_matrix.npz", allow_pickle=True)
    T = zb["X"][:, zb["tasks"] == "toxigen"]
    acc = T.mean(1)
    print("\n== ToxiGen diagnosis ==")
    print("model accuracy quantiles (0,25,50,75,100%):", np.round(np.quantile(acc, [0, .25, .5, .75, 1]), 3))
    print("item mean quantiles:", np.round(np.quantile(T.mean(0), [0, .25, .5, .75, 1]), 3))
    half = T.shape[1] // 2
    print("items 1st half mean %.3f, 2nd half mean %.3f; per-model corr(acc on low-mean items, acc on high-mean items): %.2f"
          % (T[:, :half].mean(), T[:, half:].mean(),
             np.corrcoef(T[:, T.mean(0) < np.median(T.mean(0))].mean(1), T[:, T.mean(0) >= np.median(T.mean(0))].mean(1))[0, 1]))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sim":
        simulate(sys.argv[2])
    elif mode == "real":
        real(sys.argv[2], sys.argv[3])
    elif mode == "anc":
        ancestry(sys.argv[2])
