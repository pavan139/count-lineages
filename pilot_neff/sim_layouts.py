"""Matched simulation under several family layouts, with Monte Carlo standard errors."""
import collections, json, sys
import numpy as np
import study

HERE = study.HERE
EXT = HERE.replace("pilot_neff", "external_routereval")


def layout(tag):
    if tag == "archive":
        return None
    if tag == "newer":      # newer leaderboard's family sizes, scaled to the archive's panel size
        fam = np.load(f"{EXT}/v2_bbh.npz", allow_pickle=False)["family"]
        sz = np.array(sorted(collections.Counter(fam).values(), reverse=True), float)
        scaled = np.maximum(1, np.round(sz * 1151 / sz.sum())).astype(int)
        while scaled.sum() > 1151:
            scaled = scaled[:-1] if scaled[-1] == 1 else scaled - (np.arange(len(scaled)) == 0)
        return list(scaled) + [1] * (1151 - scaled.sum())
    if tag == "balanced":   # same panel size, 115 families of 10
        return [10] * 115 + [1]
    if tag == "curated":    # the 100-model curated panel
        d = json.load(open(f"{HERE}/lineage_labels_bm.json"))["family"]
        c = collections.Counter(v for v in d.values() if v)
        return sorted(c.values(), reverse=True) + [1] * sum(v is None for v in d.values())


if __name__ == "__main__":
    tag, reps, boot = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sz = layout(tag)
    if sz is not None:
        s_ = np.array(sz); print(tag, "M", s_.sum(), "families", len(s_), "largest", s_.max(), "G*", round(s_.sum() ** 2 / (s_ ** 2).sum(), 1))
    study.sim(reps=reps, J=400, boot_reps=boot, sigma=1.1, sizes=sz, out_name=f"study_sim_{tag}.json")
