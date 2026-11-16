# Count Lineages, Not Checkpoints

Code, results and paper source for *Count Lineages, Not Checkpoints: LLMs Are Not Independent
Test-Takers*.

Item-level analyses of LLM benchmarks (item response models, label-error screens, differential
item functioning) treat every evaluated model as an independent respondent. Most leaderboard
entries share a base model with many others. This repository measures what that does to
uncertainty and provides a tool to account for it.

## Layout

| Folder | Contents |
|---|---|
| `rewrite/` | Paper source and PDF. Every table, figure and in-text number is generated; see `rewrite/README.md`. |
| `pilot_neff/` | Analysis code and stored result files. `make_tables.py` and `make_figures.py` build the paper's tables and figures from them. |
| `modelkin/` | Standalone Python tool: family labels for three panels, family bootstrap, panel composition, screening audit. |
| `evalens_release_package/data/` | Response matrices for the archive and the curated panel, with upstream licences in `evalens_release_package/licenses/`. |

## Quick start

```bash
cd modelkin
pip install -e .
modelkin audit --matrix ../evalens_release_package/data/core_matrix_trimmed.npz \
               --labels data/open_llm_leaderboard_v1_families.csv --task arc
```

## Reproducing the paper

`rewrite/README.md` lists the commands in order. Three inputs are fetched by the scripts and
are not stored here:

- item-level responses for the newer leaderboard, from the RouterEval release
  (`leaderboard_score.zip`, read only through `pilot_neff/safe_pickle.py`);
- GSM8K-Platinum and MMLU-Redux error annotations;
- live repository metadata from Hugging Face (snapshots of what was retrieved on
  15 November 2026 are stored in `pilot_neff/`).

Run the scripts from inside `pilot_neff/`. Several of them are plain scripts that execute on
import, so run them, do not import them.


## Licence

Code: MIT (see `LICENSE`). Data: see `evalens_release_package/DATA_SOURCES.md` and the
licence files next to it; the benchmark question text in `data/cache/` remains under its
upstream licences.
