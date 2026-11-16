# Data sources, attribution, and changes

The MIT license in `LICENSE` applies to the evalens software. Third-party
question text, evaluation records, and source notices retain the terms below.
The anonymous collective notice “evalens contributors” describes contributors
to this project; it makes no claim of ownership over upstream data.

The manuscript bibliography provides the corresponding research citations.
These source terms were checked on 5 September 2026. Copies of the applicable
notices and source declarations are preserved unchanged in `licenses/`;
`licenses/manifest.json` records the exact URLs, retrieval date, and checksums.
For sources other than the pinned BenchMIRT revision, those are current notice
snapshots, not a reconstruction of unrecorded historical harvest revisions.

## Included materials

| Material | Files and transformations | Source and applicable notices |
|---|---|---|
| ARC-Challenge question text | `arc::` entries in `data/cache/item_meta_core.json.gz`; excerpts in `artifacts/flagged_items.csv`. Archived prompts are normalized for item keys and excerpts are truncated to 600 characters. | Allen Institute for Artificial Intelligence and ARC contributors; *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge* (Clark et al.). The [official data card](https://huggingface.co/datasets/allenai/ai2_arc) declares [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). ARC text and adaptations are distributed under those terms, including attribution and share-alike; see `licenses/CC-BY-SA-4.0.txt`. |
| HellaSwag question text | `hellaswag::` metadata entries and flagged-item excerpts; the same normalization and truncation apply. | Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi; *HellaSwag: Can a Machine Really Finish Your Sentence?* The [source repository license](https://github.com/rowanz/hellaswag/blob/master/LICENSE) is MIT, copyright 2019 Rowan Zellers; preserved in `licenses/hellaswag-MIT.txt`. |
| GSM8K question text | `gsm8k::` metadata entries and flagged-item excerpts; the same normalization and truncation apply. | OpenAI and the GSM8K authors; *Training Verifiers to Solve Math Word Problems* (Cobbe et al.). The [source repository license](https://github.com/openai/grade-school-math/blob/master/LICENSE) is MIT, copyright 2021 OpenAI; preserved in `licenses/gsm8k-MIT.txt`. |
| TruthfulQA question text | `truthfulqa::` metadata entries and flagged-item excerpts; binary MC1 scoring is selected from the archived evaluation records. | Stephanie Lin, Jacob Hilton, and Owain Evans; *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. The [source repository](https://github.com/sylinrl/TruthfulQA) declares Apache 2.0 for the dataset; preserved in `licenses/truthfulqa-Apache-2.0.txt`. The copied license's template copyright line is retained as supplied, without inventing a holder. |
| WinoGrande question text | `winogrande::` metadata entries and flagged-item excerpts; the same normalization and truncation apply. | Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi / Allen Institute for Artificial Intelligence; *WinoGrande: An Adversarial Winograd Schema Challenge at Scale*. The [official README](https://github.com/allenai/winogrande#license) declares the **dataset** CC-BY and the **codebase** Apache 2.0. The README does not specify a CC-BY version; `licenses/winogrande-readme.md` preserves the exact declaration. This inventory does not silently substitute the code license or invent a data-license version. |
| Primary per-item correctness and model metadata | `data/core_matrix.npz`, its trimmed derivative, model/item IDs, `data/raw/`, and `data/ancestry.json`. Correctness is selected from archived metrics, assembled into binary arrays, filtered for coverage, and summarized by the supplied code. | Hugging Face's [archived Open LLM Leaderboard](https://huggingface.co/open-llm-leaderboard-old), the individual model/evaluation contributors identified in the repository index and model IDs, and the benchmark creators above. The original harvest did not retain upstream parquet revisions or a dataset-license snapshot for every details repository. A [representative official details card](https://huggingface.co/datasets/open-llm-leaderboard-old/details_meta-llama__Llama-2-7b-hf) documents evaluation provenance without a blanket dataset-license grant. We do not assert that evalens's MIT software license grants additional rights over those records. Local snapshot hashes and selection rules are in `artifacts/provenance.json`. |
| BenchMIRT correctness scores | `data/benchmirt_matrix.npz` and its study outputs; the upstream score table is transposed to models × questions, identifiers are converted to the documented representation, and binary scores are unchanged. Question text and model-generated answer text from the sixteen underlying benchmarks are **not** included in this matrix. | Ai2 / Malia Morgan, Vasudha Varadarajan, Faeze Brahman, and Maarten Sap; *BenchMIRT: Disentangling Safety and General Capabilities in LLM Evaluation*. The [score data card](https://huggingface.co/datasets/allenai/BenchMIRT-eval-data/tree/67edc41cba6795b596959ffb81f631f6bc285e3b) declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and references [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use). The frozen score revision, upstream hash, transformations, and matrix identity are recorded in `data/benchmirt_provenance.json`; notices are in `licenses/benchmirt-data-card.md` and `licenses/CC-BY-4.0.txt`. |

The metadata archive contains 14,592 prompt entries: 1,170 ARC, 10,019
HellaSwag, 1,319 GSM8K, 817 TruthfulQA, and 1,267 WinoGrande. These counts
describe the included metadata, which can exceed the retained matrix vocabulary.
Metadata and statistical flags provide source traceability; a flag is not a
claim that a source answer key is wrong.

Original evalens statistical summaries and generated figures identify the
included matrix snapshots. Reduced item-ID lists are selections, not newly
authored benchmark questions. Neither attribution nor a statistical
transformation implies endorsement by the upstream authors.

Downstream redistribution should retain this inventory and the source notices.
The known uncertainty about original per-repository archive terms and the
unspecified WinoGrande CC-BY version is disclosed rather than replaced by an
unsupported universal data-license claim.
