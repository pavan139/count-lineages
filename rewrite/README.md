# Lineage rewrite of the evalens paper

Paper: `main.pdf` (source `main.tex`). `main_status.pdf` is the same paper plus working notes: one open TODO and an evidence-status appendix. Title: *Count Lineages, Not Checkpoints: LLMs Are Not Independent Test-Takers*.
The original manuscript in `../evalens_release_package/` is untouched.

Every table (`tab_*.tex`), in-text number (`numbers.tex`) and figure (`figures/`) is generated
from result files in `../pilot_neff/`. Do not edit them by hand.

## Rebuild

Any Python 3 with numpy, scipy and matplotlib works; set `PY` to it. The path below is the one used for the paper:

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
cd ../pilot_neff
$PY lineage_fetch.py ../evalens_release_package/data/core_matrix_trimmed.npz lineage_raw.json      # live metadata, ~15 min
$PY lineage_analyze.py ../evalens_release_package lineage_raw.json                                  # writes lineage_labels.json
$PY lineage_fetch.py ../evalens_release_package/data/benchmirt_matrix.npz lineage_raw_bm.json
$PY neff_resolve.py sim resolve_sim.json                                                            # Table 1
$PY study.py inflate arc truthfulqa winogrande gsm8k hellaswag                                      # Tables 6-7
$PY study.py transfer arc truthfulqa winogrande gsm8k hellaswag                                     # Table 9
$PY study.py bm                                                                                     # second-archive table
$PY screen_audit.py && $PY dif_audit.py                                                             # re-audit tables
$PY thin_and_growth.py                                                                              # one-per-family table, growth figure
$PY v2_fetch.py && $PY v2_analyze.py                                                                # newer leaderboard composition (live API)
$PY v2_build.py && $PY v2_config_fetch.py && $PY v2_families.py && $PY v2_study.py                  # newer leaderboard responses and families (needs ../external_routereval; live fetch)
for L in curated archive newer balanced; do $PY sim_layouts.py $L 200 40; done                      # matched simulation, four layouts (archive 200/40, curated 300/50, others 120/25 were used)
$PY gold_fetch.py madrylab/gsm8k-platinum main test gold_gsm8k_platinum.json                        # GSM8K error annotations
$PY gold_fetch_mmlu.py                                                                              # MMLU question order (MMLU-Redux annotations fetched the same way)
$PY gold_audit.py gsm8k mmlu                                                                        # ground-truth table
$PY make_tables.py && $PY make_figures.py
cd ../rewrite && tectonic main.tex && tectonic main_status.tex
```

Re-fetching metadata changes the lineage tables if repositories have been deleted or edited since 15 November 2026.

## External data

`../external_routereval/` holds `leaderboard_score.zip` (84 MB) from the Hugging Face dataset
`linggm/RouterEval` (MIT licence), downloaded 15 November 2026. Its pickles are read with
`pilot_neff/safe_pickle.py`, which only permits plain numeric and string arrays. The 1.1 GB
`leaderboard_old.pkl` inside the zip is read only by `pilot_neff/gold_audit.py`, for the
MMLU-Redux audit.

## Tool

`../modelkin/` is a standalone package with the estimators, the family labels for three panels,
a command line (`modelkin labels | card | audit`) and 14 tests. Its `audit` command reproduces
the archive figures in the paper. It has not been published anywhere.

## Review history

On 15 November 2026 the paper went through five AI-assisted review passes (mathematics, archive
numbers and code, newer panel and DIF, citations, adversarial referee). No independent human
reviewer has checked it yet. Confirmed findings from those passes were
applied: corrected family labels (6 models), a model-level bootstrap baseline for screening, a
fixed-ability-scale simulation, equal family weighting, a power-matched null for DIF replication,
a stricter name match and placebo pool on the newer leaderboard, restated propositions, a new
title, and bibliography corrections.

## Not done

See the "Evidence status" appendix in `main_status.pdf`. Main gaps: no
item-level responses for 2025-26 models (none found publicly at scale; the newest panel ends
March 2025); the re-audits re-run published procedures without reproducing the original
pipelines; Lee et al. (NeurIPS 2026)
read from the abstract only.

## Clean-clone check

On 16 November 2026 the repository was cloned fresh and `make_tables.py`, `make_figures.py` and
the family-labelling rule were re-run from the stored result files. Tables, in-text numbers and
labels reproduce exactly. The heavy steps (simulations, bootstraps, audits) were not re-run in
the clone; their stored results are used as they are.
