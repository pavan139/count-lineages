"""Lineage composition of the Open LLM Leaderboard v2 table (June 2024 to March 2025)."""
import collections, json
import numpy as np
rows = json.load(open("v2_contents.json"))
seen, uniq = set(), []
for r in sorted(rows, key=lambda r: r["Submission Date"] or "9999"):
    if r["fullname"] not in seen:
        seen.add(r["fullname"]); uniq.append(r)
M = len(uniq)


def signature(r):
    p = r["#Params (B)"]
    return f"{r['Architecture']}|{round(p, 1) if isinstance(p, (int, float)) and p > 0 else '?'}"


def declared(r):
    b = r["Base Model"]
    return b if (b and b != "Removed" and b != r["fullname"]) else None


def card(labels):
    c = collections.Counter(labels); sz = np.array(list(c.values()))
    return {"families": len(c), "largest": int(sz.max()), "largest_share": float(sz.max() / len(labels)),
            "effective_clusters": float(len(labels) ** 2 / (sz ** 2).sum()), "top": c.most_common(5)}
sig = [signature(r) for r in uniq]
dec = [declared(r) or f"self:{r['fullname']}" for r in uniq]
# a model that is itself someone's declared base joins that family
bases = {declared(r) for r in uniq if declared(r)}
dec = [d if not d.startswith("self:") or d[5:] not in bases else d[5:] for d in dec]
out = {"models": M, "period": [min(r["Submission Date"] for r in uniq if r["Submission Date"]), max(r["Submission Date"] for r in uniq if r["Submission Date"])],
       "declares_other_base": sum(declared(r) is not None for r in uniq), "removed": sum(r["Base Model"] == "Removed" for r in uniq),
       "signature": card(sig), "declared": card(dec)}
curve, cs = [], collections.Counter()
for i, (r, s) in enumerate(zip(uniq, sig), 1):
    cs[s] += 1
    if i % 25 == 0 or i == M:
        sz = np.array(list(cs.values()))
        curve.append({"date": r["Submission Date"], "entries": i, "families": len(sz), "effective_clusters": float(i ** 2 / (sz ** 2).sum())})
out["growth_signature"] = curve
json.dump(out, open("study_v2_composition.json", "w"), indent=1)
print(f"{M} unique models, submitted {out['period'][0]} to {out['period'][1]}; {out['declares_other_base']} declare another model as base; {out['removed']} bases removed")
for k in ("signature", "declared"):
    c = out[k]; print(f"  {k:9s}: {c['families']} families, largest {c['largest']} ({100*c['largest_share']:.0f}%), effective clusters {c['effective_clusters']:.1f} | top {c['top']}")
for c in curve[::30] + [curve[-1]]:
    print(f"   {c['date']} entries {c['entries']:5d} families {c['families']:4d} effective clusters {c['effective_clusters']:.1f}")
