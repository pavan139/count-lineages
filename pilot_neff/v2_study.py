"""The same analyses on the newer leaderboard (mid 2024 to March 2025), including the
Llama-versus-Qwen family DIF contrast studied in the concurrent literature."""
import collections, json, sys
import numpy as np
from scipy.stats import norm, spearmanr
import neff_pilot as n
import study, dif_audit, screen_audit

EXT = study.HERE.replace("pilot_neff", "external_routereval")
NICE = {"mmlu_pro": "MMLU-Pro", "bbh": "BBH", "math": "MATH", "musr": "MuSR"}


def codes(fam):
    return np.unique(fam, return_inverse=True)[1]


def run(tag):
    z = np.load(f"{EXT}/v2_{tag}.npz", allow_pickle=False)
    X, fam = z["X"], z["family"]
    import os
    fine_path = f"{study.HERE}/v2_families.json"
    family_definition = "architecture and parameter count"
    if os.path.exists(fine_path) and not os.environ.get("V2_COARSE"):
        fine = json.load(open(fine_path))
        fam = np.array([fine.get(str(m), f"coarse:{f}") for m, f in zip(z["models"], fam)])
        family_definition = "configuration signature"
    meta = {}
    for r_ in json.load(open(f"{study.HERE}/v2_contents.json")):
        meta.setdefault(r_["fullname"], r_)
    arch = np.array([meta[str(m)]["Architecture"] or "" for m in z["models"]])
    text = np.array([f"{m} {meta[str(m)]['Base Model'] or ''}".lower() for m in z["models"]])
    # a named family needs both the name and the matching architecture class
    named = np.where(np.char.find(text, "llama") >= 0, "Llama", "")
    named = np.where((named == "Llama") & (arch == "LlamaForCausalLM") & (np.char.find(text, "qwen") < 0), "Llama", "")
    named = np.where((np.char.find(text, "qwen") >= 0) & np.char.startswith(arch, "Qwen2") & (np.char.find(text, "llama") < 0), "Qwen", named)
    Mm, J = X.shape
    li = codes(fam)
    rng = np.random.default_rng(2026)
    sz = np.bincount(li)
    out = {"M": Mm, "J": J, "family_definition": family_definition, "families": int(len(sz)), "largest_share": float(sz.max() / Mm), "effective_clusters": float(Mm ** 2 / (sz ** 2).sum())}
    # similarity on an item subsample
    cols = rng.choice(J, min(J, 1500), replace=False)
    zab = n.ability(X)
    R = n.row_corr(n.logistic_resid(X[:, cols], zab)[0])
    iu = np.triu_indices(Mm, 1)
    same = (li[:, None] == li[None, :])[iu]
    out["r_same"] = float(R[iu][same].mean()); out["r_diff"] = float(R[iu][~same].mean())
    del R
    # inflation, one split, item subsample
    perm = rng.permutation(J); D, T = perm[: J // 2], perm[J // 2:][:1500]
    zD = n.ability(X[:, D]); E, P = n.logistic_resid(X[:, T], zD)
    rr, _ = study.ratios(study.variances(E, P, zD, li))
    out["infl_difficulty_median"] = float(np.median(rr["cr3_int"])); out["infl_difficulty_p90"] = float(np.quantile(rr["cr3_int"], .9))
    out["infl_discrimination_median"] = float(np.median(rr["cr3_slope"]))
    del E, P
    # screening, with a model-level bootstrap as the no-lineage baseline
    out["screen"] = screen_audit.audit(X, li, B=200 if J > 3000 else 400, seed=5)
    # sensitivity of the standard-error ratio to a finer family definition (item subsample)
    Xs = X[:, cols]
    se_m = screen_audit.boot_se(Xs, np.arange(Mm), 100, rng)
    sens = {}
    for parts in (1, 2, 4):
        fine = np.unique(li * parts + rng.integers(0, parts, size=Mm), return_inverse=True)[1]
        se_g = screen_audit.boot_se(Xs, fine, 100, rng)
        o_ = np.isfinite(se_m) & np.isfinite(se_g) & (se_m > 0)
        fs = np.bincount(fine)
        sens[str(parts)] = {"se_ratio": float(np.median(se_g[o_] / se_m[o_])), "effective_clusters": float(Mm ** 2 / (fs ** 2).sum())}
    out["se_sensitivity"] = sens
    # family DIF: Llama vs Qwen
    score = X.mean(1)
    isL, isQ = named == "Llama", named == "Qwen"
    # placebo pool: architecture classes shared with neither named family
    other = np.flatnonzero((arch != "LlamaForCausalLM") & ~np.char.startswith(arch, "Qwen") & (arch != "") & (arch != "?"))

    def contrast(a, b):
        idx = np.r_[a, b]; focal = np.r_[np.ones(len(a), bool), np.zeros(len(b), bool)]
        return dif_audit.mh(X[idx], focal, dif_audit.strata_of(score[idx]))
    L, Q = np.flatnonzero(isL), np.flatnonzero(isQ)
    NG = min(200, len(L), len(Q))
    nm, pm = [], []
    for _ in range(10):
        a, b = rng.choice(L, NG, replace=False), rng.choice(Q, NG, replace=False)
        nm.append(dif_audit.flags(*contrast(a, b)))
        both = rng.permutation(np.r_[a, b]); pm.append(dif_audit.flags(*contrast(both[:NG], both[NG:])))
    plc, plc_bh = [], []
    fams_o = np.unique(li[other]); so = {f: int((li[other] == f).sum()) for f in fams_o}
    for _ in range(40):
        a, b, na, nb = [], [], 0, 0
        for f in rng.permutation(fams_o):
            if na <= nb:
                a.append(f); na += so[f]
            else:
                b.append(f); nb += so[f]
        ia = other[np.isin(li[other], a)]; ib = other[np.isin(li[other], b)]
        k = min(NG, len(ia), len(ib))
        fl_ = dif_audit.flags(*contrast(rng.choice(ia, k, replace=False), rng.choice(ib, k, replace=False)))
        plc.append(fl_["p05"]); plc_bh.append(fl_["bh"])
    p_all, d_all, ok_all = contrast(L, Q)
    flagged = ok_all & (p_all < 0.05)
    rep = {}
    for side, own, opp in (("Llama", L, Q), ("Qwen", Q, L)):
        bases = [f for f, c in collections.Counter(fam[own]).most_common(3) if c >= 40]
        every = np.ones(J, bool); samesign = np.ones(J, bool)
        for f in bases:
            sub = own[fam[own] == f]
            pb, db, okb = contrast(sub, opp) if side == "Llama" else contrast(opp, sub)
            samesign &= okb & (np.sign(db) == np.sign(d_all)); every &= okb & (pb < 0.05) & (np.sign(db) == np.sign(d_all))
        null = []
        for _ in range(10):
            pool = rng.permutation(own); ev = np.ones(J, bool); pos = 0
            for f in bases:
                k_ = int((fam[own] == f).sum()); sub = pool[pos:pos + k_]; pos += k_
                pb, db, okb = contrast(sub, opp) if side == "Llama" else contrast(opp, sub)
                ev &= okb & (pb < 0.05) & (np.sign(db) == np.sign(d_all))
            null.append(float(ev[flagged].mean()))
        rep[side] = {"bases": [str(f) for f in bases], "same_sign": float(samesign[flagged].mean()), "significant": float(every[flagged].mean()),
                     "null": float(np.mean(null))}
    out["dif"] = {"n_llama": int(len(L)), "n_qwen": int(len(Q)), "group": int(NG), "placebo_models": int(len(other)), "placebo_families": int(len(fams_o)),
                  "perm": float(np.mean([d["p05"] for d in pm])), "named": float(np.mean([d["p05"] for d in nm])), "named_bh": float(np.mean([d["bh"] for d in nm])),
                  "placebo_bh": float(np.mean(plc_bh)), "placebo_largest_share": float(max(so.values()) / len(other)),
                  "placebo_mean": float(np.mean(plc)), "placebo_lo": float(np.quantile(plc, .025)), "placebo_hi": float(np.quantile(plc, .975)), "replication": rep}
    # leave-family-out rank agreement
    bins = np.searchsorted(np.quantile(zab, np.linspace(0, 1, 11)[1:-1]), zab)

    def arms(labels, seed):
        fs = np.random.default_rng(seed).permutation(np.unique(labels)); s_ = np.bincount(labels, minlength=labels.max() + 1)
        a, na, nb = [], 0, 0
        for f in fs:
            if na <= nb:
                a.append(f); na += s_[f]
            else:
                nb += s_[f]
        ina = np.isin(labels, a); return np.flatnonzero(ina), np.flatnonzero(~ina)

    def agree(A, B_, k):
        ra = study.item_rest(X[rng.choice(A, k, replace=False)]); rb = study.item_rest(X[rng.choice(B_, k, replace=False)])
        o = np.isfinite(ra) & np.isfinite(rb); return spearmanr(ra[o], rb[o]).statistic
    real, shuf = [], []
    for s in range(10):
        A, B_ = arms(li, s); ctl = []
        for _ in range(3):
            ls = li.copy()
            for bb in np.unique(bins):
                idx = np.flatnonzero(bins == bb); ls[idx] = li[rng.permutation(idx)]
            ctl.append(arms(ls, s))
        k = min([len(A), len(B_)] + [min(len(a), len(b)) for a, b in ctl])
        real.append(agree(A, B_, k)); shuf.append(np.mean([agree(a, b, k) for a, b in ctl]))
    out["transfer"] = {"real": float(np.mean(real)), "shuffled": float(np.mean(shuf)), "positive_seeds": int((np.array(shuf) > np.array(real)).sum()), "seeds": 10}
    d = out["dif"]; s = out["screen"]
    print(f"{NICE[tag]}: {Mm} models x {J} items | {out['families']} families, largest {100*out['largest_share']:.0f}%, G* {out['effective_clusters']:.1f}")
    print(f"  residual similarity same family {out['r_same']:+.3f} vs different {out['r_diff']:+.3f} | inflation difficulty median {out['infl_difficulty_median']:.1f} (p90 {out['infl_difficulty_p90']:.1f}), discrimination {out['infl_discrimination_median']:.1f}")
    print(f"  screening: {s['negative']} negative; flagged Fisher {s['fisher']}, model bootstrap {s['model']}, family bootstrap {s['family']} ({100*s['survive_vs_model']:.0f}% survive), SE family/model {s['se_family_over_model']:.2f}")
    print(f"  SE ratio with each family split at random into 1/2/4: " + ", ".join(f"{v_['se_ratio']:.2f} (G* {v_['effective_clusters']:.0f})" for v_ in out["se_sensitivity"].values()))
    print(f"  DIF Llama ({d['n_llama']}) vs Qwen ({d['n_qwen']}), {d['group']} per group: permuted {d['perm']:.3f} | placebo {d['placebo_mean']:.3f} [{d['placebo_lo']:.3f},{d['placebo_hi']:.3f}]"
          f" ({d['placebo_models']} models, {d['placebo_families']} families) | named {d['named']:.3f} (FDR {d['named_bh']:.3f})")
    print(f"  replication: {d['replication']}")
    print(f"  leave-family-out rank agreement: shuffled {out['transfer']['shuffled']:.3f} vs real {out['transfer']['real']:.3f} (gap positive in {out['transfer']['positive_seeds']}/10)", flush=True)
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["musr", "math", "bbh", "mmlu_pro"]
    path = f"{study.HERE}/study_v2_responses.json"
    try:
        res = json.load(open(path))
    except FileNotFoundError:
        res = {}
    for t in tags:
        res[t] = run(t)
        json.dump(res, open(path, "w"), indent=1)
