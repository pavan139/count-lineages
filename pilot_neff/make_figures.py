"""Figures for the paper, drawn from the same result files as the tables."""
import collections
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates
import study

HERE = study.HERE; OUT = HERE.replace("pilot_neff", "rewrite") + "/figures"
C = ["#2a78d6", "#eb6834", "#1baf7a"]            # validated categorical slots 1-3 (light surface)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

# ---- Figure 1: who is in the panel, and do relatives answer alike?
lab = json.load(open(f"{HERE}/lineage_labels.json"))
fc = collections.Counter(v for v in lab["family"].values() if v)
unk = sum(v is None for v in lab["family"].values())
top = fc.most_common(8)
names = {"mistral-7b": "Mistral-7B", "llama-13b": "Llama 13B", "llama-7b": "Llama 7B", "llama3-8b": "Llama-3 8B", "mixtral-8x7b": "Mixtral 8x7B",
         "solar-10.7b": "SOLAR 10.7B", "llama-70b": "Llama 70B", "mistral-moe-2x": "Mistral 2x MoE"}
rows = [(names.get(k, k), v) for k, v in top] + [(f"{len(fc) - 8} other families", sum(fc.values()) - sum(v for _, v in top)), ("Unassigned", unk)]
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.55})
y = range(len(rows))[::-1]
ax[0].barh(list(y), [v for _, v in rows], color=[C[0]] * 8 + ["#b9b8b0", "#b9b8b0"], height=0.62)
for yy, (nm, v) in zip(y, rows):
    ax[0].text(v + 6, yy, str(v), va="center", color=INK)
ax[0].set_yticks(list(y)); ax[0].set_yticklabels([nm for nm, _ in rows])
ax[0].set_xlabel("Entries in the 1,151-entry archive"); ax[0].set_xlim(0, 380)
ax[0].tick_params(length=0); ax[0].spines["left"].set_visible(False)
ax[0].set_title("(a) Base-model families", loc="left", color=INK, fontsize=8.5)

TASKS = [("arc", "ARC-Challenge"), ("hellaswag", "HellaSwag"), ("truthfulqa", "TruthfulQA"), ("winogrande", "WinoGrande"), ("gsm8k", "GSM8K")]
val = json.load(open(f"{HERE}/validation.json"))
series = [("declared", "Declared relatives", "o"), ("same", "Same family, no declared link", "s"), ("different", "Different family", "^")]
yy = list(range(len(TASKS)))[::-1]
ax[1].axvline(0, color=MUTED, lw=0.8)
for (key, name, mk), col in zip(series, C):
    ax[1].scatter([val[t][key] for t, _ in TASKS], yy, color=col, marker=mk, s=34, label=name, zorder=3, edgecolor="white", linewidth=0.6)
ax[1].set_yticks(yy); ax[1].set_yticklabels([n for _, n in TASKS])
ax[1].set_xlabel("Mean residual correlation between model pairs")
ax[1].grid(axis="x", color=GRID, lw=0.6); ax[1].set_axisbelow(True); ax[1].tick_params(length=0); ax[1].spines["left"].set_visible(False)
ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=1, handletextpad=0.3, labelcolor=INK)
ax[1].set_title("(b) Relatives answer alike", loc="left", color=INK, fontsize=8.5)
fig.savefig(f"{OUT}/f1_lineage.pdf", bbox_inches="tight", metadata={"CreationDate": None}); fig.savefig(f"{OUT}/f1_lineage.png", dpi=170, bbox_inches="tight")

# ---- Figure 2: do 95% intervals cover the truth?
s = json.load(open(f"{HERE}/study_sim_archive.json"))
order = [("null: no families", "No families"), ("families, unrelated ability", "Families,\nunrelated ability"),
         ("families, shared ability", "Families,\nshared ability"), ("families, shared ability, uneven items", "Families, shared\nability, uneven items")]
methods = [("difficulty:naive", "Models treated as independent", "o"), ("difficulty:cr3_t", "Family jackknife, $t$ critical value", "s"),
           ("difficulty:family_bootstrap", "Family bootstrap", "^")]
fig, ax = plt.subplots(figsize=(5.2, 2.5))
yy = list(range(len(order)))[::-1]
ax.axvline(95, color=MUTED, lw=0.9, ls=(0, (4, 3))); ax.text(94.3, -0.42, "nominal 95%", ha="right", va="center", color=MUTED, fontsize=7.5)
ax.set_ylim(-0.6, len(order) - 0.5)
for (key, name, mk), col in zip(methods, C):
    xs = [100 * s[k]["coverage95"][key] for k, _ in order]
    ax.scatter(xs, yy, color=col, marker=mk, s=40, label=name, zorder=3, edgecolor="white", linewidth=0.6)
    if key == "difficulty:naive":
        for x, y_ in zip(xs, yy):
            if x < 80:
                ax.text(x - 2.2, y_, f"{x:.0f}%", ha="right", va="center", color=INK)
ax.set_yticks(yy); ax.set_yticklabels([n for _, n in order]); ax.set_xlim(15, 100)
ax.set_xlabel("Share of nominal 95% intervals for item difficulty that contain their target (%)")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0); ax.spines["left"].set_visible(False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.26), ncol=3, handletextpad=0.2, columnspacing=1.0, labelcolor=INK)
fig.savefig(f"{OUT}/f2_coverage.pdf", bbox_inches="tight", metadata={"CreationDate": None}); fig.savefig(f"{OUT}/f2_coverage.png", dpi=170, bbox_inches="tight")
# ---- Figure 3: more entries, less lineage diversity
import datetime as dt
g = json.load(open(f"{HERE}/study_growth.json"))
g = [c for c in g if c["date"] >= "2023-03-01"]
x = [dt.date.fromisoformat(c["date"]) for c in g]
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={"wspace": 0.32})
ax[0].plot(x, [c["entries"] for c in g], color=C[0], lw=2, label="Entries")
ax[0].plot(x, [c["families"] for c in g], color=C[1], lw=2, label="Families (unassigned counted singly)")
ax[0].text(x[-1], g[-1]["entries"], f" {g[-1]['entries']}", va="center", color=INK); ax[0].text(x[-1], g[-1]["families"], f" {g[-1]['families']}", va="center", color=INK)
ax[0].set_ylabel("Count"); ax[0].set_title("(a) The archive grows", loc="left", color=INK, fontsize=8.5)
ax[0].legend(frameon=False, loc="upper left", labelcolor=INK)
ax[1].plot(x, [c["effective_clusters"] for c in g], color=C[2], lw=2)
ax[1].text(x[-1], g[-1]["effective_clusters"], f" {g[-1]['effective_clusters']:.0f}", va="center", color=INK)
ax[1].set_ylabel("Effective number of families $G^*$"); ax[1].set_ylim(0, None)
ax[1].set_title("(b) Its effective number of families does not", loc="left", color=INK, fontsize=8.5)
for a in ax:
    a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=0)
    a.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 7))); a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    a.set_xlim(x[0], x[-1] + dt.timedelta(days=45))
fig.savefig(f"{OUT}/f3_growth.pdf", bbox_inches="tight", metadata={"CreationDate": None}); fig.savefig(f"{OUT}/f3_growth.png", dpi=170, bbox_inches="tight")
print("figures written")
