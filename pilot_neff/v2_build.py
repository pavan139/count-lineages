"""Build response matrices for the newer leaderboard from the RouterEval score files and
attach submission-time metadata. The pickles are read with a loader that only permits
plain numeric and string arrays."""
import collections, json, re
import numpy as np
import safe_pickle

import os
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_routereval")
SRC = os.path.join(OUT, "leaderboard_score")
meta = {}
for r in json.load(open("v2_contents.json")):
    meta.setdefault(r["fullname"], r)
lower = {k.lower(): k for k in meta}


def clean(n):
    n = str(n)
    n = n[:-len("-details")] if n.endswith("-details") else n
    return n.replace("__", "/", 1)


def named(full, base):
    s = f"{full} {base or ''}".lower()
    hits = [k for k, pat in (("Qwen", "qwen"), ("Llama", "llama"), ("Mistral", "mistral|mixtral"), ("Gemma", "gemma"), ("Phi", r"phi-?[1-4]")) if re.search(pat, s)]
    return hits[0] if len(hits) == 1 else None


def build(tag, arrays, models):
    names = [clean(m) for m in models]
    keys = [n if n in meta else lower.get(n.lower()) for n in names]
    have = np.array([k is not None for k in keys])
    X = np.concatenate(arrays, axis=0).T
    assert np.isin(X, [0, 1]).all()
    X = X.astype(np.int8)[have]
    rows = [meta[k] for k in keys if k is not None]
    fam = np.array([f"{r['Architecture']}|{round(r['#Params (B)'], 1) if isinstance(r['#Params (B)'], (int, float)) and r['#Params (B)'] > 0 else '?'}" for r in rows])
    nm = np.array([named(r["fullname"], r["Base Model"]) or "" for r in rows])
    keep = (X.mean(0) > 0) & (X.mean(0) < 1)
    np.savez_compressed(f"{OUT}/v2_{tag}.npz", X=X[:, keep], models=np.array([r["fullname"] for r in rows]), family=fam, named=nm,
                        date=np.array([r["Submission Date"] or "" for r in rows]))
    c = collections.Counter(fam)
    print(f"{tag}: {len(names)} models in file, {have.sum()} matched to leaderboard metadata; {X.shape[1]} items ({keep.sum()} informative)"
          f" | families {len(c)}, largest {c.most_common(3)} | named {collections.Counter(nm).most_common(6)}")


d = safe_pickle.load(f"{SRC}/leaderboard_mmlu_pro.pkl")
build("mmlu_pro", [d["data"]["mmlu_pro"]["correctness"]], d["model"])
d = safe_pickle.load(f"{SRC}/leaderboard_new.pkl")
print("subtasks:", sorted(d["data"]))
for tag, pref in (("bbh", "bbh_"), ("math", "math_"), ("musr", "musr_")):
    build(tag, [d["data"][k]["correctness"] for k in sorted(d["data"]) if k.startswith(pref)], d["model"])
