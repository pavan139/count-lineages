"""Finer families for the newer leaderboard: the same configuration signature as the archive
(architecture class, hidden size, layer count, feed-forward width / vocabulary where shapes collide),
falling back to architecture and parameter count where the config file is unavailable."""
import collections, json
import numpy as np
import lineage_analyze as la

import os
EXT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_routereval")
cfgs = json.load(open("v2_configs.json"))
out, src = {}, collections.Counter()
for tag in ("bbh", "mmlu_pro"):
    z = np.load(f"{EXT}/v2_{tag}.npz", allow_pickle=False)
    for m, coarse in zip(z["models"], z["family"]):
        m = str(m)
        rec = cfgs.get(m) or {}
        s = la.sig_label({"cfg": rec.get("cfg")}) if rec.get("cfg") else None
        out[m] = s or f"coarse:{coarse}"
        src["signature" if s else "architecture+size"] += 1 if m not in out or True else 0
# models without a readable config: adopt the signature family that dominates their architecture+size group
by_coarse = collections.defaultdict(collections.Counter)
coarse_of = {}
for tag in ("bbh", "mmlu_pro"):
    z = np.load(f"{EXT}/v2_{tag}.npz", allow_pickle=False)
    for m, coarse in zip(z["models"], z["family"]):
        coarse_of[str(m)] = str(coarse)
for m, f in out.items():
    if not f.startswith("coarse:"):
        by_coarse[coarse_of[m]][f] += 1
imputed = 0
for m, f in list(out.items()):
    if f.startswith("coarse:"):
        c_ = by_coarse.get(coarse_of[m])
        if c_ and sum(c_.values()) >= 5 and c_.most_common(1)[0][1] / sum(c_.values()) >= 0.8:
            out[m] = c_.most_common(1)[0][0]; imputed += 1
print("models without config assigned to their group's dominant signature family:", imputed)
json.dump(out, open("v2_families.json", "w"))
c = collections.Counter(out.values()); sz = np.array(list(c.values()))
n_sig = sum(not v.startswith("coarse:") for v in out.values())
print(f"{len(out)} models | config signature for {n_sig} ({100*n_sig/len(out):.0f}%) | {len(c)} families, largest {c.most_common(6)}")
print(f"effective number of families {len(out)**2/(sz**2).sum():.1f}")
json.dump({"models": len(out), "signature": n_sig, "imputed": imputed, "coarse": sum(v.startswith("coarse:") for v in out.values()), "families": len(c), "largest": int(sz.max()), "effective": float(len(out) ** 2 / (sz ** 2).sum()),
           "top": c.most_common(8)}, open("study_v2_families.json", "w"), indent=1)
