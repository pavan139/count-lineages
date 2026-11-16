"""Generate every table and every in-text number of the paper from result files.
Nothing in rewrite/tab_*.tex or rewrite/numbers.tex is typed by hand."""
import collections
import json
import numpy as np
from scipy.stats import norm
import neff_pilot as n
import study

HERE = study.HERE
OUT = HERE.replace("pilot_neff", "rewrite")
TASKS = [("arc", "ARC-Challenge"), ("hellaswag", "HellaSwag"), ("truthfulqa", "TruthfulQA"), ("winogrande", "WinoGrande"), ("gsm8k", "GSM8K")]
SHORT = {"arc": "ARC", "hellaswag": "HellaSwag", "truthfulqa": "TruthfulQA", "winogrande": "WinoGrande", "gsm8k": "GSM"}
num = {}
J = lambda name: json.load(open(f"{HERE}/{name}"))


def w(name, text):
    open(f"{OUT}/{name}", "w").write(text)


def table(caption, label, spec, header, rows, size="\\small", wide=False):
    if "footnotesize" in size:
        size += "\\setlength{\\tabcolsep}{3.5pt}"
    body = ("\\begin{tabular}{" + spec + "}\n\\toprule\n" + header + "\\\\\n\\midrule\n" + "\\\\\n".join(rows)
            + "\\\\\n\\bottomrule\n\\end{tabular}")
    if wide:
        body = "\\resizebox{\\linewidth}{!}{%\n" + body + "}"
    return "\\begin{table}[!htb]\\centering" + size + "\n\\caption{" + caption + "}\\label{" + label + "}\n" + body + "\n\\end{table}\n"


def rng_(vals, fmt):
    return fmt.format(min(vals)), fmt.format(max(vals))


# ================================================================ identifiability simulation (300 models)
d = J("resolve_sim.json")
names = {"A strong families": "A\; strong families", "B weak families": "B\; weak families",
         "C second dimension only": "C\; second dimension only", "D strong second dimension only": "D\; strong second dimension",
         "E families + second dimension": "E\; families + second dimension"}
rows = []
for k, lab in names.items():
    t, e = d[k]["truth"], d[k]["est"]
    rows.append(f"{lab} & {t['intercept']:.2f} & {t['item_pair']:.2f} & {e['or_block1']:.2f} & {e['or_block2']:.2f} & {e['sand_int']:.2f} & {e['spec_all']:.2f}")
w("tab_ident.tex", table(
    "Can dependence be read from responses? Simulated panels of 300 models and 800 items (100 replicates). True inflation is the median over items of the Monte Carlo variance ratio against an independent panel. With the true family labels the expressions of Corollary~\\ref{cor:families} are close to the truth whenever families are present (A, B, E). The response-only estimators miss weak families entirely (B), understate difficulty inflation for strong ones (A, E), and report inflation among independent models that differ on a second ability dimension (C, D).",
    "tab:ident", "lrr|rr|rr",
    " & \\multicolumn{2}{c|}{True inflation} & \\multicolumn{2}{c|}{True labels} & \\multicolumn{2}{c}{Response-only}\\\\\nScenario & difficulty & assoc. & difficulty & assoc. & difficulty & assoc.", rows))
num["IdentMissTruth"] = f"{d['B weak families']['truth']['intercept']:.1f}"
num["IdentMissEst"] = f"{d['B weak families']['est']['sand_int']:.1f}"
num["IdentFalseEst"] = f"{d['D strong second dimension only']['est']['sand_int']:.1f}"
num["IdentFalseTruth"] = f"{d['D strong second dimension only']['truth']['intercept']:.1f}"

# ================================================================ matched simulation (real layout)
s = J("study_sim_archive.json")
order = ["null: no families", "families, unrelated ability", "families, shared ability", "families, shared ability, uneven items"]
short = ["No families", "Families, unrelated ability", "Families, shared ability", "Shared ability, uneven items"]
fam_sc = order[1:]
meta = s[order[1]]
rows = []
for k, lab in zip(order, short):
    t, tf, e = s[k]["truth"], s[k]["truth_fixed_scale"], s[k]["estimates_true_labels"]
    rows.append(f"{lab} & {abs(e['within_family_r']):.2f} & {t['difficulty']:.1f} & {tf['difficulty']:.1f} & {t['discrimination']:.1f} & {tf['discrimination']:.1f}"
                f" & {t['item_rest']:.1f} & {t['association']:.1f} & {e['cr3_int']:.1f} & {e['cr3_slope']:.1f} & {e['block_assoc']:.1f}")
w("tab_simtruth.tex", table(
    f"True variance inflation for the median item on the archive's family layout ({len(J('lineage_labels.json')['family']):,} models; {meta['items']} items; {meta['reps']} replicates; family-effect spread {meta['sigma']} on the logit scale), and estimates that use the true family labels. $\\bar r$ is the mean within-family residual correlation, for comparison with Table~\\ref{{tab:validation}}. ``Fixed'' holds the ability scale fixed across replicates. The cluster-jackknife estimates of difficulty inflation run below the truth when one family dominates.",
    "tab:simtruth", "lr|rrrrrr|rrr",
    " & & \\multicolumn{6}{c|}{True inflation} & \\multicolumn{3}{c}{Estimated, true labels}\\\\\nScenario & $\\bar r$ & diff. & fixed & discr. & fixed & item--rest & assoc. & diff. & discr. & assoc.", rows, "\\footnotesize", wide=True))
BOOTREPS = 40
MAXSE = 100 * max(v for k in order for kk, v in s[k]["coverage_se"].items() if "bootstrap" not in kk)
MAXSEB = 100 * max(v for k in order for kk, v in s[k]["coverage_se"].items() if "bootstrap" in kk)
rows = []
for k, lab in zip(order, short):
    c = s[k]["coverage95"]
    f = lambda key: f"{100 * c[key]:.0f}" if key in c else "--"
    rows.append(f"{lab} & {f('difficulty:naive')} & {f('difficulty:cr0')} & {f('difficulty:cr3')} & {f('difficulty:cr3_t')} & {f('difficulty:family_bootstrap')}"
                f" & {f('discrimination:naive')} & {f('discrimination:cr3')} & {f('discrimination:cr3_t')} & {f('discrimination:family_bootstrap')}")
w("tab_simcover.tex", table(
    "Share (\\%) of nominal 95\\% intervals that contain the Monte Carlo mean of the estimator, in the same simulation; this measures the calibration of the standard error. Naive: model-based standard errors treating models as independent. CR0: cluster-robust with families as clusters. CR3: linearised delete-one-family jackknife with a normal critical value. CR3+$t$: CR3 with a $t$ critical value on $G^\\ast-1$ degrees of freedom. Bootstrap: percentile interval from resampling whole families (" + str(BOOTREPS) + " replicates, 99 resamples, 120 items). Monte Carlo standard errors are at most " + f"{MAXSE:.1f}" + " points for the first four methods and " + f"{MAXSEB:.1f}" + " for the bootstrap.",
    "tab:simcover", "l|rrrrr|rrrr",
    " & \\multicolumn{5}{c|}{Difficulty} & \\multicolumn{4}{c}{Discrimination}\\\\\nScenario & naive & CR0 & CR3 & CR3+$t$ & boot. & naive & CR3 & CR3+$t$ & boot.", rows, "\\footnotesize"))
# ---------------- other family layouts
LAY = [("archive", "Archive"), ("newer", "Newer leaderboard, scaled"), ("balanced", "Balanced, 115 families of 10"), ("curated", "Curated panel")]
rows, lay_naive = [], {}
for tag, nice in LAY:
    d_ = J(f"study_sim_{tag}.json")
    first = True
    for k, lab in zip(fam_sc, short[1:]):
        v = d_[k]; c = v["coverage95"]; e = v["estimates_true_labels"]
        head = (f"{nice} & {v['models']:,} & {v['effective_clusters']:.0f}" if first else " & & ")
        rows.append(("\\midrule\n" if first and rows else "") + f"{head} & {lab.replace('Families, ', '').replace('Shared ability, ', 'shared, ').capitalize()} & {abs(e['within_family_r']):.2f}"
                    f" & {v['truth']['difficulty']:.1f} & {v['truth_fixed_scale']['difficulty']:.1f} & {v['truth']['item_rest']:.1f}"
                    f" & {100 * c['difficulty:naive']:.0f} & {100 * c['difficulty:cr3_t']:.0f} & {100 * c['difficulty:family_bootstrap']:.0f}")
        lay_naive.setdefault(tag, []).append(100 * c["difficulty:naive"])
        first = False
w("tab_simlayouts.tex", table(
    "The matched simulation under four family layouts (400 items; 200 replicates for the archive layout, 300 for the curated panel, 120 otherwise). The newer leaderboard's family sizes are scaled to the archive's panel size. $G^\\ast$: effective number of families. True inflation of difficulty for the median item on the panel's own ability scale and on a fixed scale, and of the item--rest correlation. Coverage (\\%) of nominal 95\\% intervals for difficulty: naive, family jackknife with a $t$ critical value, and family bootstrap.",
    "tab:simlayouts", "lrrlr|rrr|rrr",
    " & & & & & \\multicolumn{3}{c|}{True inflation} & \\multicolumn{3}{c}{Coverage (\\%)}\\\\\nLayout & $M$ & $G^\\ast$ & Scenario & $\\bar r$ & diff. & fixed & item--rest & naive & CR3+$t$ & boot.", rows, "\\footnotesize"))
num["LayBalancedLo"], num["LayBalancedHi"] = rng_(lay_naive["balanced"], "{:.0f}")
num["LayCuratedLo"], num["LayCuratedHi"] = rng_(lay_naive["curated"], "{:.0f}")
num["LayNewerLo"], num["LayNewerHi"] = rng_(lay_naive["newer"], "{:.0f}")
allboot = [100 * J(f"study_sim_{tag}.json")[k]["coverage95"]["difficulty:family_bootstrap"] for tag, _ in LAY for k in fam_sc]
num["LayBootLo"], num["LayBootHi"] = rng_(allboot, "{:.0f}")
num["SimReps"] = str(s[order[1]]["reps"])
cv = lambda key: [100 * s[k]["coverage95"][key] for k in fam_sc]
num["SimNaiveCoverShared"], num["SimNaiveCoverUnrel"] = rng_(cv("difficulty:naive"), "{:.0f}")
num["SimTCoverLo"], num["SimTCoverHi"] = rng_(cv("difficulty:cr3_t"), "{:.0f}")
num["SimCRCoverLo"], num["SimCRCoverHi"] = rng_(cv("difficulty:cr3"), "{:.0f}")
num["SimBootCoverLo"], num["SimBootCoverHi"] = rng_(cv("difficulty:family_bootstrap"), "{:.0f}")
num["SimNullNaive"] = f"{100 * s[order[0]]['coverage95']['difficulty:naive']:.0f}"
un, sh, uv = s[order[1]], s[order[2]], s[order[3]]
num["SimTrueDiffUnrel"] = f"{un['truth']['difficulty']:.1f}"; num["SimTrueDiffShared"] = f"{sh['truth']['difficulty']:.1f}"
num["SimFixDiffUnrel"] = f"{un['truth_fixed_scale']['difficulty']:.1f}"; num["SimFixDiffShared"] = f"{sh['truth_fixed_scale']['difficulty']:.1f}"
num["SimTrueItemRestUnrel"] = f"{un['truth']['item_rest']:.1f}"; num["SimTrueItemRestShared"] = f"{sh['truth']['item_rest']:.1f}"
num["SimTrueDiscrShared"] = f"{sh['truth']['discrimination']:.1f}"
num["SimEffClusters"] = f"{sh['effective_clusters']:.0f}"
num["SimUnevenEst"] = f"{uv['estimates_true_labels']['cr3_int']:.1f}"; num["SimUnevenCover"] = f"{100 * uv['coverage95']['difficulty:naive']:.0f}"
num["SimUnevenFix"] = f"{uv['truth_fixed_scale']['difficulty']:.1f}"

# ================================================================ thinning and equal family weights
th = J("study_thinning.json")
rows = []
for k, labk in (("families, unrelated ability", "Families, unrelated ability"), ("families, shared ability", "Families, shared ability")):
    v = th[k]
    rows.append(f"{labk} & {100 * v['difficulty']['naive_coverage_thinned']:.0f} & {v['difficulty']['variance_thinned_over_full']:.2f} & {v['difficulty']['variance_weighted_over_full']:.2f}"
                f" & {100 * v['discrimination']['naive_coverage_thinned']:.0f} & {v['discrimination']['variance_thinned_over_full']:.2f} & {v['discrimination']['variance_weighted_over_full']:.2f}")
w("tab_thin.tex", table(
    "Giving families equal weight in the matched simulation (80 replicates). One per family keeps the first listed member of each family (members are exchangeable in the simulation). Coverage (\\%) of naive 95\\% intervals on the thinned panel, and the true variance of the estimate relative to the usual equal-weight analysis of all models, for the thinned panel and for the full panel with each model weighted by the inverse of its family size. A ratio below one means more precise than the usual analysis.",
    "tab:thin", "l|rrr|rrr",
    " & \\multicolumn{3}{c|}{Difficulty} & \\multicolumn{3}{c}{Discrimination}\\\\\nScenario & coverage & one per family & weighted & coverage & one per family & weighted", rows, "\\footnotesize"))
num["ThinCoverLo"], num["ThinCoverHi"] = rng_([100 * th[k]["difficulty"]["naive_coverage_thinned"] for k in th], "{:.0f}")
num["ThinGainLo"], num["ThinGainHi"] = rng_([1 / th[k]["difficulty"]["variance_thinned_over_full"] for k in th], "{:.1f}")
num["WeightGainLo"], num["WeightGainHi"] = rng_([1 / th[k]["difficulty"]["variance_weighted_over_full"] for k in th], "{:.1f}")

# ================================================================ lineage recovery
raw = {r["model"]: r for r in J("lineage_raw.json")}
lab = J("lineage_labels.json")
z = np.load(f"{study.ROOT}/data/core_matrix_trimmed.npz", allow_pickle=True)
models = [str(m) for m in z["models"]]
M = len(models)


def parents(m):
    r = raw.get(m, {})
    p = list(r.get("base_model") or [])
    if r.get("adapter_base"):
        p.append(r["adapter_base"])
    return [q for q in p if isinstance(q, str) and "/" in q and q != m]
reach = sum(raw.get(m, {}).get("api_status") == 200 for m in models)
cfg = sum("cfg" in raw.get(m, {}) for m in models)
dec = sum(bool(parents(m)) for m in models)
tc = collections.Counter(lab["tier"][m] for m in models)
assigned = M - tc["unknown"]
fc = collections.Counter(lab["family"][m] for m in models if lab["family"][m])
pct = lambda x: f"{100 * x / M:.0f}\\%"
rows = [f"Repository still public & {reach} & {pct(reach)}", f"Configuration readable & {cfg} & {pct(cfg)}", f"Declares a parent model & {dec} & {pct(dec)}",
        "\\midrule\nFamily assigned by configuration signature & %d & %s" % (tc["signature"], pct(tc["signature"])),
        f"Family assigned by declared root & {tc['declared-root']} & {pct(tc['declared-root'])}",
        f"Family assigned by repository name only & {tc['name']} & {pct(tc['name'])}",
        f"\\textbf{{Family assigned, any signal}} & \\textbf{{{assigned}}} & \\textbf{{{pct(assigned)}}}",
        f"Unassigned (treated as singletons) & {tc['unknown']} & {pct(tc['unknown'])}"]
w("tab_coverage.tex", table("Lineage recovery for the 1,151-entry archive from public repository metadata retrieved on 15 November 2026.",
                            "tab:coverage", "lrr", "Signal & Entries & Share", rows))
top = fc.most_common(4)
sizes = np.array(list(fc.values()) + [1] * tc["unknown"])
num.update(NModels=f"{M:,}", LineageCoverage=f"{100 * assigned / M:.0f}", DeclaredShare=f"{100 * dec / M:.0f}",
           LargestFamily=str(top[0][1]), LargestFamilyShare=f"{100 * top[0][1] / M:.0f}", NFamilies=str(len(fc)),
           SecondFamily=str(top[1][1]), ThirdFamily=str(top[2][1]), FourthFamily=str(top[3][1]),
           EffClusters=f"{M ** 2 / (sizes ** 2).sum():.0f}", UnassignedShare=f"{100 * tc['unknown'] / M:.0f}")
assert top[0][0] == "mistral-7b"

par = {}


def find(a):
    par.setdefault(a, a)
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a
for m in list(raw):
    for p in parents(m):
        par[find(m)] = find(p)
comp = {m: find(m) for m in models}

# ================================================================ validation
rows, val = [], {}
for t, nice in TASKS:
    X, names_t = study.load_task(t)
    li, known, labs = study.load_labels(names_t)
    R = n.row_corr(n.logistic_resid(X, n.ability(X))[0])
    iu = np.triu_indices(len(names_t), 1)
    c = np.array([comp[m] for m in names_t])
    same_c = (c[:, None] == c[None, :])[iu]; same_f = (li[:, None] == li[None, :])[iu]; kn = (known[:, None] & known[None, :])[iu]
    r = R[iu]
    val[t] = (r[same_c].mean(), r[same_f & kn & ~same_c].mean(), r[~same_f & kn & ~same_c].mean())
    rows.append(f"{nice} & ${val[t][0]:+.3f}$ & ${val[t][1]:+.3f}$ & ${val[t][2]:+.3f}$")
w("tab_validation.tex", table("Mean ability-adjusted residual correlation between pairs of models, by lineage relation. Labels were built without any benchmark response. Declared relatives: connected through declared parent links. Same family: same base-model family but no declared link. Different family: different families and no declared link.",
                              "tab:validation", "lrrr", "Benchmark & Declared relatives & Same family, no declared link & Different family", rows))
json.dump({t: {"declared": float(v[0]), "same": float(v[1]), "different": float(v[2])} for t, v in val.items()}, open(f"{HERE}/validation.json", "w"), indent=1)
num["ValSameFamily"] = f"{val['arc'][1]:.2f}"; num["ValDeclared"] = f"{val['arc'][0]:.2f}"; num["ValDifferent"] = f"{val['arc'][2]:.2f}"
arch_same_max = max(max(v[0], v[1]) for v in val.values())

# ================================================================ inflation on the archive
rows, rows2, meds = [], [], []
for t, nice in TASKS:
    r = J(f"study_inflate_core_{t}.json")
    sp, se, sl = r["splits"], r["sensitivity"], r["screen_link"]
    meds.append(sp["cr3_int:median"]["mean"])
    rows.append(f"{nice} & {sp['cr3_int:p25']['mean']:.1f} & {sp['cr3_int:median']['mean']:.1f} & {sp['cr3_int:p90']['mean']:.1f} & {sp['pooled_meat']['mean']:.1f}"
                f" & {sp['cr3_slope:median']['mean']:.1f} & {sp['assoc']['mean']:.1f}")
    rows2.append(f"{nice} & {se['all']['cr3_int_median']:.1f} & {se['no_name']['cr3_int_median']:.1f} & {se['split_llama']['cr3_int_median']:.1f}"
                 f" & {se['labelled_only']['cr3_int_median']:.1f} & {se['drop_largest_family']['cr3_int_median']:.1f} & ${sl['spearman_inflation_vs_itemrest']:+.2f}$")
    num["Infl" + SHORT[t] + "Top"] = f"{sp['cr3_int:p90']['mean']:.0f}"
    num["Link" + SHORT[t]] = f"{sl['spearman_inflation_vs_itemrest']:+.2f}"
w("tab_inflation.tex", table(
    "Estimated variance inflation on the archive with base-model families as clusters: delete-one-family jackknife variance over delete-one-model jackknife variance, mean over 20 random item splits. Difficulty is shown across items (25th percentile, median, 90th percentile) because it is strongly heterogeneous, and pooled over items (cluster-robust over heteroskedasticity-robust score variance), which is the quantity comparable with the association column. Discrimination is the item median. Simulation (Table~\\ref{tab:simtruth}) indicates the difficulty estimates understate the true inflation.",
    "tab:inflation", "lrrrrrr",
    " & \\multicolumn{4}{c}{Difficulty} & & \\\\\nBenchmark & 25th & median & 90th & pooled & Discrimination & Association", rows))
w("tab_sensitivity.tex", table(
    "Sensitivity of the median difficulty inflation (first item split) to labelling choices, and rank correlation between an item's inflation and its item--rest correlation. No name labels: families assigned from repository names are treated as unknown. Llama split: Llama-1 and Llama-2 derivatives separated by name and creation date. Labelled only: unassigned entries dropped. No largest: the largest family dropped.",
    "tab:sensitivity", "lrrrrrr", "Benchmark & All labels & No name labels & Llama split & Labelled only & No largest & $\\rho$(inflation, $r_{ir}$)", rows2))
num["InflMedMin"], num["InflMedMax"] = rng_(meds, "{:.1f}")
cons = [100 * (2 * norm.cdf(1.96 / np.sqrt(m_)) - 1) for m_ in meds]
num["ConsCoverLo"], num["ConsCoverHi"] = rng_(cons, "{:.0f}")

# ================================================================ growth
gr = J("study_growth.json")
gs = [c["effective_clusters"] for c in gr]
peak = int(np.argmax(gs))
ceiling = 12
settle = max(i for i, v in enumerate(gs) if v >= ceiling) + 2
num.update(GrowthDated=str(len(gr)), GrowthPeak=f"{gs[peak]:.0f}", GrowthPeakEntries=str(gr[peak]["entries"]), GrowthCeiling=str(ceiling),
           GrowthSettle=str(int(round(settle, -1))), GrowthFinalClusters=f"{gs[-1]:.0f}")

# ================================================================ three panels
v2 = J("study_v2_composition.json")
lbm = J("lineage_labels_bm.json")
fcb = collections.Counter(v for v in lbm["family"].values() if v); nunk = sum(v is None for v in lbm["family"].values())
szb = np.array(list(fcb.values()) + [1] * nunk)
rows = [f"Open LLM Leaderboard archive & 2023--mid 2024 & {M:,} & {len(fc) + tc['unknown']} & {100 * top[0][1] / M:.0f}\\% & {M ** 2 / (sizes ** 2).sum():.0f}",
        f"Newer Open LLM Leaderboard, all entries & June 2024--March 2025 & {v2['models']:,} & {v2['signature']['families']} & {100 * v2['signature']['largest_share']:.0f}\\% & {v2['signature']['effective_clusters']:.0f}",
        f"BenchMIRT (curated subset of the newer one) & to early 2025 & 100 & {len(fcb) + nunk} & {100 * szb.max() / 100:.0f}\\% & {100 ** 2 / (szb ** 2).sum():.0f}"]
w("tab_panels.tex", table(
    "Lineage composition of three panels. Families count unassigned entries as singletons. $G^\\ast$ is the effective number of families. For the newer leaderboard, families are defined from the architecture and parameter count recorded at submission, which is coarser than the configuration signature used for the other two panels.",
    "tab:panels", "llrrrr", "Panel & Period & Entries & Families & Largest family & $G^\\ast$", rows, wide=True))
g2 = v2["growth_signature"]
late = [c["effective_clusters"] for c in g2 if c["entries"] >= 1000]
num.update(VtwoModels=f"{v2['models']:,}", VtwoClusters=f"{v2['signature']['effective_clusters']:.0f}", VtwoLargestShare=f"{100 * v2['signature']['largest_share']:.0f}",
           VtwoFamilies=str(v2["signature"]["families"]), VtwoClustersMin=f"{min(late):.0f}", VtwoClustersMax=f"{max(late):.0f}")

# ================================================================ curated panel
b = J("study_second_archive.json")
rows = []
for t, v in b["tasks"].items():
    rs = f"${v['r_same']:+.3f}$" if v["r_same"] is not None else "--"
    rows.append(f"{t.replace('_', ' ')} & {v['M']} & {rs} & ${v['r_diff']:+.3f}$ & {v['cr3_int_median']:.2f} & {v['cr3_slope_median']:.2f}")
w("tab_second.tex", table(
    "Curated panel: the 100-model BenchMIRT release with lineage recovered by the same procedure. Residual correlation for same-family and different-family pairs, and median estimated inflation (cluster jackknife, 10 item splits). Models with constant responses on a task are removed for that task.",
    "tab:second", "lrrrrr", "Task & Models & Same family $r$ & Different $r$ & Difficulty & Discrimination", rows, "\\footnotesize"))
ds = [v["cr3_int_median"] for v in b["tasks"].values()]
num["SecondInflMin"], num["SecondInflMax"] = rng_(ds, "{:.1f}")
num.update(SecondFamilies=str(b["families"]), SecondAssigned=str(100 - nunk), SecondLargest=str(max(fcb.values())),
           SecondEffClusters=f"{100 ** 2 / (szb ** 2).sum():.0f}", SecondTasks=str(len(b["tasks"])),
           SecondClear=str(sum(v["r_same"] is not None and v["r_same"] - v["r_diff"] > 0.02 for v in b["tasks"].values())),
           SecondPairs=str(max(v["n_same_pairs"] for v in b["tasks"].values())))
assert not (b["tasks"]["math"]["r_same"] - b["tasks"]["math"]["r_diff"] > 0.02)

# ================================================================ screening re-audit (archive)
sc = J("study_screen.json")
rows, lost, ser = [], [], []
for t, nice in TASKS:
    v = sc[t]
    enough = v["model"] >= 10
    ser.append(v["se_family_over_model"])
    if enough:
        lost.append(100 * (1 - v["survive_vs_model"]))
    rows.append(f"{nice} & {v['items']:,} & {v['negative']} & {v['fisher']} & {v['model']} & {v['family']} & "
                + (f"{100 * v['survive_vs_model']:.0f}\\%" if enough else "--") + f" & {v['se_family_over_model']:.1f}")
w("tab_screen.tex", table(
    "Re-audit of the significance of screening flags on the archive. Items whose item--rest correlation is significantly negative (one-sided, 5\\% false discovery rate) under three standard errors: the Fisher formula, a bootstrap over individual models, and a bootstrap over whole base-model families (1,000 resamples each; 500 on HellaSwag). Survive: share of model-bootstrap flags still flagged under the family bootstrap, not reported where fewer than ten items are flagged. SE ratio: median family-level over model-level standard error, over all items.",
    "tab:screen", "lrrrrrrr", "Benchmark & Items & $r_{ir}<0$ & Fisher & Model & Family & Survive & SE ratio", rows))
num["ScreenLostMin"], num["ScreenLostMax"] = rng_(lost, "{:.0f}")
num["ScreenSEMin"], num["ScreenSEMax"] = rng_(ser, "{:.1f}")
assert len(lost) == 4 and sc["gsm8k"]["model"] < 10
num["ScreenVarMin"], num["ScreenVarMax"] = rng_([x ** 2 for x in ser], "{:.1f}")
num["SimItemRestMin"], num["SimItemRestMax"] = rng_([s[k]["truth"]["item_rest"] for k in fam_sc], "{:.1f}")

# ================================================================ transfer
rows, pos = [], 0
for t, nice in TASKS:
    r = J(f"study_transfer_{t}.json")
    a, y, m = r["rank"], r["yield"], r["msd"]
    pos += a["seeds_gap_positive"] == r["seeds"]
    rows.append(f"{nice} & {a['entry']:.3f} & {a['shuffled']:.3f} & {a['family']:.3f} & ${a['gap_mean']:+.3f}$ & {a['seeds_gap_positive']}/{r['seeds']}"
                f" & {y['shuffled']:.3f} & {y['family']:.3f} & ${y['gap_mean']:+.3f}$ & {y['seeds_gap_positive']}/{r['seeds']} & {m['family'] / m['shuffled']:.1f}")
w("tab_transfer.tex", table(
    "Transfer of item screening statistics between two halves of the panel (30 seeds, both directions). Random: halves drawn at random. Shuffled: whole families held out after family labels are permuted within ability deciles, preserving family sizes and ability composition. Real: whole base-model families held out. Rank agreement is the Spearman correlation of item--rest correlations between halves. Yield is the share of the 100 most negative items in one half that are negative in both halves. Gap is shuffled minus real; ``seeds'' counts seeds with a positive gap. MSD ratio: mean squared difference of item accuracy between halves, real over shuffled.",
    "tab:transfer", "l|rrrrr|rrrr|r",
    " & \\multicolumn{5}{c|}{Rank agreement} & \\multicolumn{4}{c|}{Yield at 100} & MSD\\\\\nBenchmark & random & shuf. & real & gap & seeds & shuf. & real & gap & seeds & ratio", rows, "\\footnotesize"))
num["TransferRankAllSeeds"] = str(pos)

# ================================================================ DIF re-audit (archive)
df = J("study_dif.json")
rows = []
for t, nice in TASKS:
    v = df[t]; fp = v["family_placebo"]
    rows.append(f"{nice} & {100 * v['model_permutation']['p05']:.0f} & {100 * fp['p05']['mean']:.0f} ({100 * fp['p05']['lo']:.0f}--{100 * fp['p05']['hi']:.0f}) & {100 * v['named']['p05']:.0f}"
                f" & {100 * fp['bh']['mean']:.1f} & {100 * v['named']['bh']:.0f} & {100 * v['replication']['share_flagged_significant_all_bases']:.0f} & {100 * v['replication_null']:.0f}")
w("tab_dif.tex", table(
    "Re-audit of family differential item functioning on the archive with the Mantel--Haenszel procedure (continuity-corrected, ten score strata, 200 models per group). Share of items (\\%) flagged at $p<0.05$ when group labels are permuted across models, when families with no known descent from either named group are split at random into two groups (mean and 2.5--97.5 percentiles over 100 splits), and for the Llama families against Mistral-7B derivatives; then the placebo and named shares after a 5\\% false discovery rate. Replication: of the items flagged in the pooled named contrast (all models), the share significant with the same sign in each of the three largest Llama base-model labels, and the same share when the Llama models are split at random into groups of those sizes.",
    "tab:dif", "lrrr|rr|rr",
    " & \\multicolumn{3}{c|}{Flagged, $p<0.05$ (\\%)} & \\multicolumn{2}{c|}{After FDR (\\%)} & \\multicolumn{2}{c}{Replication (\\%)}\\\\\nBenchmark & permuted & placebo & named & placebo & named & real & random", rows, "\\footnotesize"))
g = lambda f: [f(df[t]) for t, _ in TASKS]
num["DifNamedMin"], num["DifNamedMax"] = rng_(g(lambda v: 100 * v["named"]["p05"]), "{:.0f}")
num["DifNamedFdrMin"], num["DifNamedFdrMax"] = rng_(g(lambda v: 100 * v["named"]["bh"]), "{:.0f}")
num["DifPlaceboMin"], num["DifPlaceboMax"] = rng_(g(lambda v: 100 * v["family_placebo"]["p05"]["mean"]), "{:.0f}")
num["DifPermMin"], num["DifPermMax"] = rng_(g(lambda v: 100 * v["model_permutation"]["p05"]), "{:.0f}")
num["DifPermNoCCMin"], num["DifPermNoCCMax"] = rng_(g(lambda v: 100 * v["perm_no_continuity"]), "{:.0f}")
num["DifRepMin"], num["DifRepMax"] = rng_(g(lambda v: 100 * v["replication"]["share_flagged_significant_all_bases"]), "{:.0f}")
num["DifRepNullMin"], num["DifRepNullMax"] = rng_(g(lambda v: 100 * v["replication_null"]), "{:.0f}")
num["DifLlamaBases"] = str(len(df["arc"]["llama_base_models"]))

# ================================================================ newer leaderboard, responses
vr = J("study_v2_responses.json")
VT = [(k, v) for k, v in (("mmlu_pro", "MMLU-Pro"), ("bbh", "BBH"), ("math", "MATH"), ("musr", "MuSR")) if k in vr]
rows, rows2 = [], []
for k, nice in VT:
    v = vr[k]; s_, tr_, d_, sn = v["screen"], v["transfer"], v["dif"], v["se_sensitivity"]
    rows.append(f"{nice} & {v['M']:,} & {v['J']:,} & {v['effective_clusters']:.0f} & ${v['r_same']:+.3f}$ & ${v['r_diff']:+.3f}$"
                f" & {s_['se_family_over_model']:.1f} & {sn['2']['se_ratio']:.1f} & {sn['4']['se_ratio']:.1f} & {s_['model']} & {s_['family']} & {tr_['shuffled']:.2f} & {tr_['real']:.2f}")
    rp = d_["replication"]
    rs = lambda side, key: f"{100 * rp[side][key]:.0f}" if len(rp[side]["bases"]) >= 2 else "--"
    rows2.append(f"{nice} & {100 * d_['perm']:.0f} & {100 * d_['placebo_mean']:.0f} ({100 * d_['placebo_lo']:.0f}--{100 * d_['placebo_hi']:.0f}) & {100 * d_['named']:.0f}"
                 f" & {100 * d_['placebo_bh']:.0f} & {100 * d_['named_bh']:.0f} & {rs('Llama', 'significant')} & {rs('Llama', 'null')} & {rs('Qwen', 'significant')} & {rs('Qwen', 'null')}")
w("tab_vtwo.tex", table(
    "The newer leaderboard (entries submitted June 2024 to March 2025; responses from the RouterEval release), with families defined by configuration signature as on the archive. Residual correlation for same-family and different-family pairs (1,500 sampled items). SE ratio: family-level over model-level bootstrap standard error of item--rest correlations, with families as defined and with every family split at random into two and into four (100 resamples, 1,500 sampled items). Flags: items significantly negative at a 5\\% false discovery rate under the model and the family bootstrap (200--400 resamples, all items). Transfer: rank agreement of item--rest correlations between panel halves for shuffled and real families (10 seeds).",
    "tab:vtwo", "lrrr|rr|rrr|rr|rr",
    " & & & & \\multicolumn{2}{c|}{Residual $r$} & \\multicolumn{3}{c|}{SE ratio} & \\multicolumn{2}{c|}{Flags} & \\multicolumn{2}{c}{Transfer}\\\\\nBenchmark & Models & Items & $G^\\ast$ & same & diff. & as is & $\\div2$ & $\\div4$ & model & family & shuf. & real", rows, "\\footnotesize"))
w("tab_difvtwo.tex", table(
    "The same Mantel--Haenszel re-audit on the newer leaderboard, for the family pair compared by \\citet{dif2026llm}: Llama against Qwen models (up to 200 per group; 40 placebo splits of the remaining architecture classes). Columns as in Table~\\ref{tab:dif}. Replication is shown for each family's three largest groups by architecture and parameter count, with the random-split comparison beside it; it is omitted where a family has a single such group with at least 40 entries.",
    "tab:difvtwo", "lrrr|rr|rr|rr",
    " & \\multicolumn{3}{c|}{Flagged, $p<0.05$ (\\%)} & \\multicolumn{2}{c|}{After FDR (\\%)} & \\multicolumn{2}{c|}{Llama repl.} & \\multicolumn{2}{c}{Qwen repl.}\\\\\nBenchmark & permuted & placebo & named & placebo & named & real & random & real & random", rows2, "\\footnotesize", wide=True))
f_ = lambda fn: [fn(vr[k]) for k, _ in VT]
num["VrModelsMin"], num["VrModelsMax"] = rng_(f_(lambda v: v["M"]), "{:,}")
num["VrSameMin"], num["VrSameMax"] = rng_(f_(lambda v: v["r_same"]), "{:.2f}")
num["VrSEMin"], num["VrSEMax"] = rng_(f_(lambda v: v["screen"]["se_family_over_model"]), "{:.1f}")
num["VrSESplitMin"], num["VrSESplitMax"] = rng_(f_(lambda v: v["se_sensitivity"]["4"]["se_ratio"]), "{:.1f}")
num["VrInflMin"], num["VrInflMax"] = rng_(f_(lambda v: v["infl_difficulty_median"]), "{:.0f}")
num["VrPlaceboMin"], num["VrPlaceboMax"] = rng_(f_(lambda v: 100 * v["dif"]["placebo_mean"]), "{:.0f}")
num["VrPermMin"], num["VrPermMax"] = rng_(f_(lambda v: 100 * v["dif"]["perm"]), "{:.0f}")
num["VrNamedMin"], num["VrNamedMax"] = rng_(f_(lambda v: 100 * v["dif"]["named"]), "{:.0f}")
reps_ = [(100 * vr[k]["dif"]["replication"][sd]["significant"], 100 * vr[k]["dif"]["replication"][sd]["null"]) for k, _ in VT for sd in ("Llama", "Qwen")
         if len(vr[k]["dif"]["replication"][sd]["bases"]) >= 2]
num["VrRepMin"], num["VrRepMax"] = rng_([a for a, _ in reps_], "{:.0f}")
num["VrRepNullMin"], num["VrRepNullMax"] = rng_([b_ for _, b_ in reps_], "{:.0f}")
num["VrAbove"] = str(sum(vr[k]["dif"]["named"] > vr[k]["dif"]["placebo_hi"] for k, _ in VT)); num["VrTasks"] = str(len(VT))
num["VrTransferPos"] = str(sum(vr[k]["transfer"]["positive_seeds"] == vr[k]["transfer"]["seeds"] for k, _ in VT))
num["VrSameAbove"] = str(sum(vr[k]["r_same"] >= arch_same_max for k, _ in VT))
assert all(v["family_definition"] == "configuration signature" for v in vr.values())
assert all(vr[k]["dif"]["replication"][sd]["significant"] < vr[k]["dif"]["replication"][sd]["null"] for k, _ in VT for sd in ("Llama", "Qwen")
           if len(vr[k]["dif"]["replication"][sd]["bases"]) >= 2)
vl = [100 * (1 - vr[k]["screen"]["survive_vs_model"]) for k, _ in VT if vr[k]["screen"]["model"] >= 10]
num["VrLostMin"], num["VrLostMax"] = rng_(vl, "{:.0f}"); num["VrLostTasks"] = str(len(vl))
vf = J("study_v2_families.json")
num.update(VfModels=f"{vf['models']:,}", VfDirect=f"{100 * (vf['signature'] - vf['imputed']) / vf['models']:.0f}", VfImputed=f"{100 * vf['imputed'] / vf['models']:.0f}",
           VfCoarse=f"{100 * vf['coarse'] / vf['models']:.0f}", VfFamilies=str(vf["families"]), VfEff=f"{vf['effective']:.0f}",
           VfPerFamily=f"{vf['models'] / vf['effective']:.0f}", ArchPerFamily=f"{M / (M ** 2 / (sizes ** 2).sum()):.0f}")

# ================================================================ documented label errors
gd = J("study_gold.json")
GL = {"raw": "Item--rest correlation, all models (usual screen)", "z_model": "Divided by model-level standard error",
      "z_family": "Divided by family-level standard error", "weighted": "Families weighted equally", "thinned": "One model per family"}
GB = [(k, v) for k, v in (("gsm8k", "GSM8K"), ("mmlu", "MMLU")) if k in gd]
rows = []
for m, nice in GL.items():
    cells = []
    for k, _ in GB:
        v = gd[k]["methods"][m]
        ci = f"[{v['auroc_diff_ci'][0]:+.3f}, {v['auroc_diff_ci'][1]:+.3f}]" if "auroc_diff_ci" in v else "--"
        cells.append(f"{v['auroc']:.3f} & {ci} & {100 * v['p50']:.0f}")
    rows.append(f"{nice} & " + " & ".join(cells))
hdr = " & " + " & ".join(f"\\multicolumn{{3}}{{c{'|' if i < len(GB) - 1 else ''}}}{{{nice_}}}" for i, (_, nice_) in enumerate(GB))
w("tab_gold.tex", table(
    "Detecting documented flawed items from the archive's responses. " + "; ".join(
        f"{nice_}: {gd[k]['items']:,} annotated items, {gd[k]['flawed']} flawed, {gd[k]['models']:,} models" for k, nice_ in GB)
    + ". AUROC of each screening score for separating flawed from clean items; 95\\% interval for its difference from the usual screen (paired bootstrap over items); and precision (\\%) among the 50 most suspicious items.",
    "tab:gold", "l" + "|".join(["rrr"] * len(GB)), hdr + "\\\\\nScreening score" + " & AUROC & difference & P@50" * len(GB), rows, "\\footnotesize", wide=True))
for k, _ in GB:
    tag_ = "Gsm" if k == "gsm8k" else "Mmlu"
    v = gd[k]
    num[f"Gold{tag_}Items"] = f"{v['items']:,}"; num[f"Gold{tag_}Flawed"] = str(v["flawed"]); num[f"Gold{tag_}Base"] = f"{100 * v['flawed'] / v['items']:.0f}"
    num[f"Gold{tag_}Auroc"] = f"{v['methods']['raw']['auroc']:.2f}"; num[f"Gold{tag_}Top"] = f"{100 * v['methods']['raw']['p50']:.0f}"
    num[f"Gold{tag_}FamAuroc"] = f"{v['methods']['z_family']['auroc']:.2f}"
    num[f"Gold{tag_}WeightAuroc"] = f"{v['methods']['weighted']['auroc']:.2f}"
    num[f"Gold{tag_}FlagsModel"] = str(v["flags_model"]["n"]); num[f"Gold{tag_}FlagsFamily"] = str(v["flags_family"]["n"])
    num[f"Gold{tag_}PrecModel"] = "--" if v["flags_model"]["precision"] is None else f"{100 * v['flags_model']['precision']:.0f}"
    num[f"Gold{tag_}PrecFamily"] = "--" if v["flags_family"]["precision"] is None else f"{100 * v['flags_family']['precision']:.0f}"
num["GoldBenchmarks"] = {1: "one benchmark", 2: "two benchmarks"}[len(GB)]
num["GoldMmluText"] = ""
if "mmlu" in gd:
    v = gd["mmlu"]; mm = v["methods"]; wg = v["wrong_groundtruth_only"]
    worse = mm["z_family"]["auroc_diff_ci"][1] < 0
    better_w = mm["weighted"]["auroc_diff_ci"][0] > 0
    fm, ff = v["flags_model"], v["flags_family"]
    al = v["alignment"]
    num["GoldMmluText"] = (
        f"MMLU-Redux \\citep{{gema2024mmlu}} provides human annotations for 5,700 MMLU items; {v['items']:,} of them could be matched "
        f"unambiguously to the responses of the {v['models']:,} archive models that were also evaluated on MMLU, and {v['flawed']} of those are flawed. "
        f"There the usual screen is much weaker (AUROC {mm['raw']['auroc']:.2f}; {wg['methods']['raw']['auroc']:.2f} for the {wg['flawed']} items with a wrong answer key), "
        + ("and equal family weights are slightly better. " if better_w else "and again no lineage-aware variant is distinguishable from it. ")
        + f"Family-level testing reduces the significantly negative items from {fm['n']} to {ff['n']} without changing how many of them are "
        f"documented as flawed ({100 * fm['precision']:.0f}\\% and {100 * ff['precision']:.0f}\\%). ")
    num["GoldMmluAligned"] = f"{al['aligned']:,}"; num["GoldMmluDropped"] = str(al["not_found"] + al["duplicate_text"])

w("numbers.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(num.items())))
for k, v in sorted(num.items()):
    print(k, v)
