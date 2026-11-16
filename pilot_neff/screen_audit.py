"""Re-audit of significance-based item screening.

Items whose item-rest correlation is significantly negative (one-sided, BH 5%), under three
standard errors: the Fisher formula, a bootstrap over individual models (no lineage), and a
bootstrap over whole families. The model bootstrap is the baseline that isolates lineage."""
import json
import numpy as np
from scipy.stats import norm
import study

HERE = study.HERE


def bh(p, q=0.05):
    m = len(p); order = np.argsort(p)
    passed = np.flatnonzero(p[order] <= q * np.arange(1, m + 1) / m)
    out = np.zeros(m, bool)
    if len(passed):
        out[order[: passed.max() + 1]] = True
    return out


def boot_se(X, code, B, rng):
    members = [np.flatnonzero(code == g) for g in range(int(code.max()) + 1)]
    draws = np.empty((B, X.shape[1]), dtype=np.float32)
    for b in range(B):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(members), size=len(members))])
        draws[b] = np.arctanh(np.clip(study.item_rest(X[rows]), -0.999, 0.999))
    return np.nanstd(draws, axis=0).astype(float)


def audit(X, code, B=1000, seed=31):
    Mm = X.shape[0]
    rng = np.random.default_rng(seed)
    z = np.arctanh(np.clip(study.item_rest(X), -0.999, 0.999))
    se_f = 1 / np.sqrt(Mm - 3)
    se_m = boot_se(X, np.arange(Mm), B, rng)
    se_g = boot_se(X, code, B, rng)
    ok = np.isfinite(z) & np.isfinite(se_m) & np.isfinite(se_g) & (se_m > 0) & (se_g > 0)
    neg = z[ok] < 0
    f_fisher = bh(norm.cdf(z[ok] / se_f)) & neg
    f_model = bh(norm.cdf(z[ok] / se_m[ok])) & neg
    f_family = bh(norm.cdf(z[ok] / se_g[ok])) & neg
    r = se_g[ok] / se_m[ok]
    return {"M": Mm, "items": int(ok.sum()), "negative": int(neg.sum()), "fisher": int(f_fisher.sum()), "model": int(f_model.sum()),
            "family": int(f_family.sum()), "survive_vs_model": float((f_model & f_family).sum() / max(f_model.sum(), 1)),
            "se_family_over_model": float(np.median(r)), "se_family_over_model_negative": float(np.median(r[neg])) if neg.any() else None,
            "se_model_over_fisher": float(np.median(se_m[ok] / se_f)), "B": B}


if __name__ == "__main__":
    res = {}
    for t in ["arc", "truthfulqa", "winogrande", "gsm8k", "hellaswag"]:
        X, names = study.load_task(t)
        li, known, lab = study.load_labels(names)
        res[t] = v = audit(X, li, B=1000 if t != "hellaswag" else 500)
        print(f"{t}: {v['negative']} negative | flagged: Fisher {v['fisher']}, model bootstrap {v['model']}, family bootstrap {v['family']}"
              f" ({100*v['survive_vs_model']:.0f}% of model-bootstrap flags survive) | SE family/model {v['se_family_over_model']:.2f}"
              f" (negative items {v['se_family_over_model_negative']}), model/Fisher {v['se_model_over_fisher']:.2f}", flush=True)
        json.dump(res, open(f"{HERE}/study_screen.json", "w"), indent=1)
