"""How much lineage can be recovered, and do the labels explain response dependence?

Label sources, all independent of benchmark responses:
  declared  - base_model / adapter base chains from model cards
  signature - architecture + layer dimensions + vocabulary bucket from config.json
  name      - tokens in the repository name (lowest confidence)
"""
import collections
import json
import re
import sys
import numpy as np
import neff_pilot as n


table = {(4096, 32): "llama-7b", (5120, 40): "llama-13b", (6656, 60): "llama-30b", (8192, 80): "llama-70b",
         (4096, 48): "solar-10.7b", (7168, 60): "yi-34b", (8192, 48): "codellama-34b", (2048, 22): "tinyllama-1.1b"}


def sig_label(r):
    cfg = r.get("cfg") or {}
    arch = (cfg.get("architectures") or [r.get("arch")])[0] if (cfg.get("architectures") or r.get("arch")) else None
    h = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model")
    L = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    v = cfg.get("vocab_size")
    if not arch or not h or not L:
        return None
    inter = cfg.get("intermediate_size")
    if arch in ("MistralForCausalLM", "LlamaForCausalLM") and (h, L) == (4096, 32):
        # four base models share this shape; vocabulary and feed-forward width tell them apart
        if v and v >= 150000:
            return "qwen-7b"
        if v and v >= 120000:
            return "llama3-8b"
        if v and 60000 <= v < 70000 and inter != 14336 and arch == "LlamaForCausalLM":
            return "yi-6b"
        if inter == 14336:
            return "mistral-7b"
        if inter == 11008:
            return "llama-7b"
        return "mistral-7b" if arch == "MistralForCausalLM" else "llama-7b"
    if arch == "MixtralForCausalLM":
        return "mixtral-8x7b" if cfg.get("num_local_experts", 8) == 8 and (h, L) == (4096, 32) else f"mistral-moe-{cfg.get('num_local_experts')}x"
    if arch == "LlamaForCausalLM":
        lab = table.get((h, L))
        if lab == "llama-70b" and v and v >= 120000:
            lab = "llama3-70b"
        if lab and lab != "llama-7b":
            return lab
    return f"{arch}|h{h}|L{L}|v{(v or 0) // 1000}k"


def name_label(mid):
    s = mid.lower()
    if "mixtral" in s or "8x7b" in s: return "mixtral-8x7b"
    if "solar" in s or "10.7b" in s or re.search(r"\b11b\b", s): return "solar-10.7b"
    if "tinyllama" in s or "1.1b" in s: return "tinyllama-1.1b"
    if "codellama" in s and "34b" in s: return "codellama-34b"
    if "34b" in s and "yi" in s: return "yi-34b"
    if re.search(r"70b|65b", s): return "llama-70b"
    if re.search(r"30b|33b", s): return "llama-30b"
    if "13b" in s: return "llama-13b"
    if re.search(r"mistral|zephyr|openhermes-2|neural-chat-7b|openchat[-_]3|starling|dolphin-2\.\d-mistral", s): return "mistral-7b"
    if "7b" in s and re.search(r"llama|vicuna|alpaca|wizardlm|guanaco|orca|platypus|airoboros-l2|beluga", s): return "llama-7b"
    return None


def main(root, raw_path):
    raw = {r["model"]: r for r in json.load(open(raw_path))}
    z = np.load(f"{root}/data/core_matrix_trimmed.npz", allow_pickle=True)
    X, tasks, models = z["X"], z["tasks"], [str(m) for m in z["models"]]
    M = len(models)

    def parents(m):
        r = raw.get(m, {})
        p = list(r.get("base_model") or [])
        if r.get("adapter_base"):
            p.append(r["adapter_base"])
        return [q for q in p if isinstance(q, str) and "/" in q and q != m]

    reach = sum(raw.get(m, {}).get("api_status") == 200 for m in models)
    has_cfg = sum("cfg" in raw.get(m, {}) for m in models)
    declared = sum(bool(parents(m)) for m in models)
    print(f"panel {M} | still on the Hub {reach} ({reach/M:.0%}) | config readable {has_cfg} ({has_cfg/M:.0%}) | declares a parent {declared} ({declared/M:.0%})")

    # ---- label system 1: root family (signature > declared-root signature/name > own name)
    def root_of(m, depth=0):
        ps = parents(m)
        if not ps or depth > 6:
            return m
        return root_of(ps[0], depth + 1)
    fam, tier = {}, {}
    agree = [0, 0]
    for m in models:
        r = raw.get(m, {})
        s = sig_label(r) if r else None
        nm = name_label(m)
        if s and nm and "|" not in s:
            agree[1] += 1; agree[0] += (s == nm)
        if s:
            fam[m], tier[m] = s, "signature"; continue
        rt = root_of(m)
        if rt != m:
            lab = sig_label(raw.get(rt, {})) or name_label(rt)
            if lab:
                fam[m], tier[m] = lab, "declared-root"; continue
        if nm:
            fam[m], tier[m] = nm, "name"; continue
        fam[m], tier[m] = None, "unknown"
    tc = collections.Counter(tier.values())
    print("root-family label source:", dict(tc), f"| coverage {1 - tc['unknown']/M:.0%}")
    print(f"name-token labels agree with config signature in {agree[0]}/{agree[1]} cases ({agree[0]/max(agree[1],1):.0%})")
    fc = collections.Counter(v for v in fam.values() if v)
    print("largest root families:", fc.most_common(12))
    sizes = np.array(list(fc.values()) + [1] * tc["unknown"])
    print(f"root families: {len(fc)} labelled + {tc['unknown']} unknown singletons; exact-copy (Kish) design effect {np.sum(sizes**2)/M:.1f}")

    # ---- label system 2: declared-lineage components (union-find over model -> parent edges)
    par = {}
    def find(a):
        par.setdefault(a, a)
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for m in list(raw):
        for p in parents(m):
            par[find(m)] = find(p)
    comp = {m: find(m) for m in models}
    cc = collections.Counter(comp.values())
    multi = sum(c for c in cc.values() if c > 1)
    print(f"declared-lineage components: {multi} panel models ({multi/M:.0%}) sit in a multi-model component; largest {cc.most_common(4)}")
    sib = collections.defaultdict(list)
    for m in models:
        for p in parents(m)[:1]:
            sib[p].append(m)
    print(f"models sharing an immediate declared parent with another panel model: {sum(len(v) for v in sib.values() if len(v) > 1)}")

    # ---- validation against responses
    print("\n== do lineage labels explain response similarity? (ability-adjusted residual correlation) ==")
    out = {}
    for t in ["arc", "truthfulqa", "winogrande", "gsm8k", "hellaswag"]:
        sub = X[:, tasks == t]; rows = (sub >= 0).all(1); sub = sub[rows]
        names = [m for m, k in zip(models, rows) if k]
        Mm = len(names)
        keep = (sub.mean(0) > 0) & (sub.mean(0) < 1); sub = sub[:, keep]; J = sub.shape[1]
        rng = np.random.default_rng(0); perm = rng.permutation(J); D, T = perm[: J // 2], perm[J // 2:]
        zA = n.ability(sub); R = n.row_corr(n.logistic_resid(sub, zA)[0])
        zD = n.ability(sub[:, D]); zT = n.ability(sub[:, T])
        RD = n.row_corr(n.logistic_resid(sub[:, D], zD)[0]); RT = n.row_corr(n.logistic_resid(sub[:, T], zT)[0])
        ET, PT = n.logistic_resid(sub[:, T], zD)
        f = np.array([fam[m] or f"?{i}" for i, m in enumerate(names)])
        known = np.array([fam[m] is not None for m in names])
        c = np.array([comp[m] for m in names])
        iu = np.triu_indices(Mm, 1)
        same_f = (f[:, None] == f[None, :])[iu]; kn = (known[:, None] & known[None, :])[iu]
        same_c = (c[:, None] == c[None, :])[iu]
        r = R[iu]
        line = (f"{t:11s} mean r: same declared lineage {r[same_c].mean():+.3f} (n={same_c.sum()}) | "
                f"same root family, no declared link {r[same_f & kn & ~same_c].mean():+.3f} (n={(same_f & kn & ~same_c).sum()}) | "
                f"different root family {r[~same_f & kn].mean():+.3f}")
        print(line)
        res = {}
        off = ~np.eye(Mm, dtype=bool)
        for nm, lab in (("root family", f), ("declared lineage", c)):
            _, li = np.unique(lab, return_inverse=True)
            Wm = (li[:, None] == li[None, :]) & off
            d = n.sandwich_deff(ET, PT, zD, li)
            d1 = 1 + R[Wm].sum() / Mm
            d2 = 1 + (RD[Wm] * RT[Wm]).sum() / Mm
            res[nm] = dict(sand_int=d["intercept_median"], sand_slope=d["slope_median"], block1=d1, block2=d2,
                           int_q=d["intercept_q"], int_mean=d["intercept_mean"], meat_ratio=d["meat_ratio"])
            if nm == "root family":
                print("      per-item difficulty inflation, 10/25/50/75/90th pct: " + " ".join(f"{q:.2f}" for q in d["intercept_q"])
                      + f" | mean {d['intercept_mean']:.2f} | pooled {d['meat_ratio']:.2f}")
            print(f"   {nm:17s}: difficulty inflation {d['intercept_median']:5.2f} (sandwich) / {d1:5.2f} (block) | "
                  f"discrimination {d['slope_median']:5.2f} | item-item association {d2:5.2f}")
        out[t] = res
    json.dump({"family": fam, "tier": tier, "inflation": out}, open(raw_path.replace("lineage_raw", "lineage_labels"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
