# modelkin

Lineage-aware uncertainty for item-level evaluation of language models.

Models on a leaderboard are not independent test-takers. Fine-tunes and merges of one base
model answer alike, so item statistics computed from a large panel (item difficulty,
discrimination, item-rest correlations, label-error flags, differential item functioning) are
much less certain than the usual formulas say. `modelkin` assigns models to base-model families
from repository metadata, never from their answers, and reports uncertainty with families as
the sampling unit.

## Install

```bash
pip install -e .
```

Needs Python 3.10+, NumPy and SciPy.

## Use

A response matrix is an `.npz` file with `X` (models by items, 0/1, -1 for missing) and
`models` (repository ids); an optional `tasks` array labels the items.

```bash
modelkin labels --matrix panel.npz --out families.csv        # queries public Hugging Face metadata
modelkin card   --matrix panel.npz --labels families.csv     # who is in the panel
modelkin audit  --matrix panel.npz --labels families.csv --task arc
```

```python
import numpy as np, modelkin as mk

z = np.load("panel.npz", allow_pickle=True)
X, models = z["X"], z["models"]
fam = mk.read_labels("families.csv", models)

mk.panel_card(fam)                      # families, largest, unassigned, effective clusters
mk.similarity_by_family(X, fam)         # do relatives answer alike?
mk.family_bootstrap(X, fam, "item_rest")  # standard errors and intervals per item
mk.screen_items(X, fam)                 # significantly negative items: Fisher, model bootstrap, family bootstrap
mk.leave_family_out(X, fam)             # do item statistics transfer to unseen families?
mk.variance_inflation(X, fam)           # per-item inflation (a lower bound)
```

## Shipped labels

- `data/open_llm_leaderboard_v1_families.csv`: 1,151 archived Open LLM Leaderboard entries, 84% assigned.
- `data/benchmirt_families.csv`: the 100 models of the BenchMIRT panel, 92 assigned.
- `data/open_llm_leaderboard_v2_families.csv`: 3,795 entries of the newer Open LLM Leaderboard; 76% by their own
  configuration, 16% by the configuration that dominates their architecture-and-size group, 8% by architecture and size.

Columns: `model`, `family`, `source` (`signature`, `declared-root`, `name`, or `unknown`).
Metadata was retrieved on 15 November 2026.

## What to report

1. The panel's lineage composition, including the effective number of clusters
   `G* = M^2 / sum(m_g^2)`.
2. Item-level uncertainty from a bootstrap over whole families.
3. Validation on held-out families against a shuffled-family control.

## Limits

- Family is base-model family, a coarse notion of lineage. Models with identical
  configurations (for example Llama 1 and Llama 2 of the same size) share a label.
- Unassigned models are treated as unrelated, which understates dependence.
- `variance_inflation` runs low when one family dominates the panel; prefer `family_bootstrap`, which itself
  gives intervals that are somewhat too narrow in that situation.
- `effective_clusters` describes how concentrated a panel is. It is not a count of independent models.
- Name-based labels are heuristics tuned to 2023-24 open models.

## Tests

```bash
pip install -e ".[test]" && pytest
```
