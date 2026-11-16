"""Does lineage-aware screening find documented label errors better?

Gold labels: GSM8K-Platinum (items removed or revised by its authors count as flawed).
Screening scores (higher = more suspicious), all from the archive's responses:
  raw       minus the item-rest correlation, all models equally weighted (the usual screen)
  z_model   minus r / SE, SE from a bootstrap over individual models
  z_family  minus r / SE, SE from a bootstrap over whole base-model families
  weighted  minus the item-rest correlation with every family given equal total weight
  thinned   minus the item-rest correlation on one random model per family (mean of 20 draws)
"""
import gzip, json, re, unicodedata
import numpy as np
import study, screen_audit

HERE = study.HERE


def norm(t):
    t = unicodedata.normalize("NFKC", t).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip().lower()


def auroc(score, y):
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]; i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2; i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def avg_precision(score, y):
    o = np.argsort(-score); yy = y[o]; hits = np.cumsum(yy)
    return float((hits[yy == 1] / (np.flatnonzero(yy == 1) + 1)).mean())


def wcorr_item_rest(X, w):
    Xf = X.astype(float); w = w / w.sum()
    rest = (Xf.sum(1)[:, None] - Xf) / (X.shape[1] - 1)
    mx = (w[:, None] * Xf).sum(0); mr = (w[:, None] * rest).sum(0)
    xc = Xf - mx; rc = rest - mr
    den = np.sqrt((w[:, None] * xc ** 2).sum(0) * (w[:, None] * rc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (w[:, None] * xc * rc).sum(0) / den
    r[den == 0] = np.nan
    return r


def evaluate(X, li, y, eval_cols, tag, B=1000, seed=8):
    """X: models x items (all items used for the rest score). y: 0/1 gold for eval_cols."""
    Mm = X.shape[0]
    rng = np.random.default_rng(seed)
    r = study.item_rest(X); zr = np.arctanh(np.clip(r, -0.999, 0.999))
    se_m = screen_audit.boot_se(X, np.arange(Mm), B, rng); se_g = screen_audit.boot_se(X, li, B, rng)
    w = 1.0 / np.bincount(li)[li]
    members = [np.flatnonzero(li == g) for g in range(int(li.max()) + 1)]
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        thin = np.nanmean([study.item_rest(X[[rng.choice(m) for m in members]]) for _ in range(20)], axis=0)
    scores = {"raw": -r, "z_model": -zr / se_m, "z_family": -zr / se_g, "weighted": -wcorr_item_rest(X, w), "thinned": -thin}
    scores = {k: v[eval_cols] for k, v in scores.items()}
    ok = np.all([np.isfinite(v) for v in scores.values()], axis=0)
    yb = y[ok]
    out = {"models": Mm, "items": int(ok.sum()), "flawed": int(yb.sum()), "methods": {}}
    for k, v in scores.items():
        v = v[ok]; top = np.argsort(-v)
        out["methods"][k] = {"auroc": auroc(v, yb), "ap": avg_precision(v, yb), "p50": float(yb[top[:50]].mean()), "p100": float(yb[top[:100]].mean())}
    n_ = len(yb); diffs = {k: [] for k in scores if k != "raw"}; dap = {k: [] for k in scores if k != "raw"}
    for _ in range(2000):
        idx = rng.integers(0, n_, n_)
        if yb[idx].sum() in (0, n_):
            continue
        base = auroc(scores["raw"][ok][idx], yb[idx]); bap = avg_precision(scores["raw"][ok][idx], yb[idx])
        for k in diffs:
            diffs[k].append(auroc(scores[k][ok][idx], yb[idx]) - base); dap[k].append(avg_precision(scores[k][ok][idx], yb[idx]) - bap)
    for k in diffs:
        out["methods"][k]["auroc_diff_ci"] = [float(np.quantile(diffs[k], .025)), float(np.quantile(diffs[k], .975))]
        out["methods"][k]["ap_diff_ci"] = [float(np.quantile(dap[k], .025)), float(np.quantile(dap[k], .975))]
    # significance flags (one-sided, BH 5%) and how many of them are documented errors
    zz = zr[eval_cols][ok]; neg = zz < 0
    for nm, se in (("model", se_m), ("family", se_g)):
        fl = screen_audit.bh(__import__("scipy.stats").stats.norm.cdf(zz / se[eval_cols][ok])) & neg
        out[f"flags_{nm}"] = {"n": int(fl.sum()), "flawed": int(yb[fl].sum()), "precision": float(yb[fl].mean()) if fl.any() else None}
    Xe = X[:, eval_cols][:, ok]
    out["mean_accuracy"] = {"clean": float(Xe[:, yb == 0].mean()), "flawed": float(Xe[:, yb == 1].mean())}
    print(f"{tag}: {Mm} models, {out['items']} annotated items, {out['flawed']} flawed ({100*yb.mean():.1f}%) | mean accuracy clean {out['mean_accuracy']['clean']:.3f}, flawed {out['mean_accuracy']['flawed']:.3f}")
    for k, v in out["methods"].items():
        extra = f" | AUROC diff vs raw [{v['auroc_diff_ci'][0]:+.3f},{v['auroc_diff_ci'][1]:+.3f}], AP diff [{v['ap_diff_ci'][0]:+.3f},{v['ap_diff_ci'][1]:+.3f}]" if "auroc_diff_ci" in v else ""
        print(f"  {k:9s} AUROC {v['auroc']:.3f}  AP {v['ap']:.3f}  P@50 {v['p50']:.2f}  P@100 {v['p100']:.2f}{extra}")
    print(f"  significant flags: model bootstrap {out['flags_model']}, family bootstrap {out['flags_family']}", flush=True)
    return out


def gsm8k():
    z = np.load(f"{study.ROOT}/data/core_matrix_trimmed.npz", allow_pickle=True)
    meta = json.load(gzip.open(f"{study.ROOT}/data/cache/item_meta_core.json.gz"))
    cols = z["tasks"] == "gsm8k"
    X = z["X"][:, cols]; items = [str(i) for i in z["items"][cols]]
    rows = (X >= 0).all(1); X = X[rows]; names = [str(m) for m, k in zip(z["models"], rows) if k]
    keep = (X.mean(0) > 0) & (X.mean(0) < 1)
    X = X[:, keep]; items = [i for i, k in zip(items, keep) if k]
    gold = {norm(r["question"]): r["cleaning_status"] for r in json.load(open(f"{HERE}/gold_gsm8k_platinum.json"))}
    status = [gold.get(norm(meta[i]["text"]), "absent") for i in items]
    y = np.array([s in ("absent", "revised") for s in status]).astype(int)
    li, known, lab = study.load_labels(names)
    out = evaluate(X, li, y, np.arange(X.shape[1]), "GSM8K (Platinum annotations)")
    out["status"] = {k: status.count(k) for k in set(status)}
    return out


def mmlu():
    import safe_pickle
    d = safe_pickle.load(HERE.replace("pilot_neff", "external_routereval") + "/leaderboard_score/leaderboard_old.pkl")
    redux = json.load(open(f"{HERE}/gold_mmlu_redux.json")); orig = json.load(open(f"{HERE}/gold_mmlu_original_questions.json"))
    lab = json.load(open(f"{HERE}/lineage_labels.json"))["family"]
    names = [str(m).split("details_", 1)[-1].replace("__", "/", 1) for m in d["model"]]
    rows = np.array([i for i, m in enumerate(names) if m in lab])
    mats, gold_cols, gold_y, gold_type, off, miss, ambiguous = [], [], [], [], 0, 0, 0
    for subj in sorted(redux):
        arr = d["data"][f"harness_hendrycksTest_{subj}_5"]["correctness"]
        assert arr.shape[0] == len(orig[subj]), (subj, arr.shape, len(orig[subj]))
        mats.append(arr[:, rows].T)
        index = {}
        for i, q in enumerate(orig[subj]):
            index.setdefault(norm(q), []).append(i)
        for r in redux[subj]:
            hit = index.get(norm(r["question"]))
            if not hit:
                miss += 1; continue
            if len(hit) > 1:
                ambiguous += 1; continue
            gold_cols.append(off + hit[0]); gold_y.append(int(r["error_type"] != "ok")); gold_type.append(r["error_type"])
        off += arr.shape[0]
    X = np.concatenate(mats, axis=1)
    assert np.isin(X, [0, 1]).all()
    X = X.astype(np.int8)
    fam = [lab[names[i]] or f"?{i}" for i in rows]
    li = np.unique(fam, return_inverse=True)[1]
    print(f"MMLU: {len(rows)} archive models found among {len(names)}; {X.shape[1]} items; annotated items aligned {len(gold_cols)}, not found {miss}, duplicate text {ambiguous}")
    gold_cols = np.array(gold_cols); gold_y = np.array(gold_y); gt = np.array(gold_type)
    keep = (X.mean(0) > 0) & (X.mean(0) < 1)
    sel = keep[gold_cols]
    out = evaluate(X, li, gold_y[sel], gold_cols[sel], "MMLU (Redux annotations, any flaw)", B=500)
    strict = (gt[sel] == "wrong_groundtruth") | (gt[sel] == "ok")
    out["wrong_groundtruth_only"] = evaluate(X, li, (gt[sel][strict] == "wrong_groundtruth").astype(int), gold_cols[sel][strict], "MMLU (wrong ground truth vs ok)", B=500)
    out["alignment"] = {"aligned": int(len(gold_cols)), "not_found": miss, "duplicate_text": ambiguous}
    return out


if __name__ == "__main__":
    import sys
    path = f"{HERE}/study_gold.json"
    try:
        res = json.load(open(path))
    except FileNotFoundError:
        res = {}
    for which in sys.argv[1:] or ["gsm8k"]:
        res[which] = {"gsm8k": gsm8k, "mmlu": mmlu}[which]()
        json.dump(res, open(path, "w"), indent=1)
