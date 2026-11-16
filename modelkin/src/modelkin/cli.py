"""Command line: modelkin labels | card | audit."""
import argparse
import json
import numpy as np
from . import inference, lineage


def _load(path, task=None):
    z = np.load(path, allow_pickle=False) if path.endswith(".npy") else np.load(path, allow_pickle=True)
    X = z["X"]; models = [str(m) for m in z["models"]]
    if task is not None:
        X = X[:, z["tasks"] == task]
    rows = (X >= 0).all(1)
    X = X[rows]; models = [m for m, k in zip(models, rows) if k]
    keep = (X.mean(0) > 0) & (X.mean(0) < 1)
    return X[:, keep], models


def main(argv=None):
    ap = argparse.ArgumentParser(prog="modelkin", description="Lineage-aware uncertainty for item-level LLM evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("labels", help="fetch repository metadata and assign base-model families")
    a.add_argument("--matrix", required=True); a.add_argument("--out", required=True); a.add_argument("--cafile")
    for name, hlp in (("card", "lineage composition of a panel"), ("audit", "composition, similarity, inflation and screening")):
        a = sub.add_parser(name, help=hlp)
        a.add_argument("--matrix", required=True); a.add_argument("--labels", required=True); a.add_argument("--task")
        a.add_argument("--bootstrap", type=int, default=400); a.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "labels":
        models = [str(m) for m in np.load(args.matrix, allow_pickle=True)["models"]]
        meta = lineage.fetch_metadata(models, cafile=args.cafile,
                                      progress=lambda n, q: print(f"\r{n} fetched, {q} queued", end="", flush=True))
        lineage.write_labels(args.out, lineage.assign_families(models, meta))
        print(f"\nwrote {args.out}")
        return 0
    X, models = _load(args.matrix, args.task)
    fam = lineage.read_labels(args.labels, models)
    out = {"panel": inference.panel_card(fam)}
    if args.cmd == "audit":
        out["similarity"] = inference.similarity_by_family(X, fam)
        infl = inference.variance_inflation(X, fam)
        out["inflation"] = {k: {"median": float(np.nanmedian(v)), "p90": float(np.nanquantile(v, 0.9))} for k, v in infl.items()}
        sc = inference.screen_items(X, fam, B=args.bootstrap)
        out["screening"] = {"flagged_fisher": int(sc["flag_fisher"].sum()), "flagged_model_bootstrap": int(sc["flag_model"].sum()),
                            "flagged_family_bootstrap": int(sc["flag_family"].sum()), "se_ratio_median": float(np.nanmedian(sc["se_ratio"]))}
    if args.json:
        print(json.dumps(out, indent=1)); return 0
    p = out["panel"]
    print(f"Panel: {p['models']} models, {p['families']} labelled families, {p['unassigned']} unassigned")
    print(f"  largest family {p['largest_family']} ({100 * p['largest_family_share']:.0f}%), effective clusters {p['effective_clusters']:.1f}")
    if args.cmd == "audit":
        s = out["similarity"]; i = out["inflation"]; c = out["screening"]
        print(f"Residual similarity: same family {s['same_family']:+.3f}, different family {s['different_family']:+.3f}")
        print(f"Variance inflation (lower bound): difficulty median {i['difficulty']['median']:.1f}x, 90th pct {i['difficulty']['p90']:.1f}x;"
              f" discrimination median {i['discrimination']['median']:.1f}x")
        print(f"Significantly negative items: Fisher test {c['flagged_fisher']}, model bootstrap {c['flagged_model_bootstrap']},"
              f" family bootstrap {c['flagged_family_bootstrap']} (family/model standard errors x{c['se_ratio_median']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
