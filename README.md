# KnownRD – Biomedical Entity Normalization

KnownRD is a modular biomedical named entity normalization (NEN) pipeline that links biomedical mentions to standardized concepts in biomedical reference ontologies.

The pipeline combines exact matching, lexical and semantic candidate retrieval, and candidate reranking. We adopt a common workflow rather than combining separate ontology-specific systems, aiming to simplify integration and maintenance and facilitate extension to additional ontologies. The workflow does not require ontology-specific training or architectural modifications, although ontology-adapted models can be incorporated when available.

## Workflow

KnownRD consists of three main stages:

**1. Candidate retrieval**

Candidate concepts are retrieved through exact matching and complementary lexical and semantic retrieval methods:

- **BM25:** lexical retrieval based on term relevance.
- **Fuzzy matching:** approximate string matching.
- **N-gram:** character n-gram similarity.
- **SapBERT:** semantic retrieval using biomedical entity representations.

**2. Candidate selection (reranking)**

Retrieved candidates are ranked and selected using three complementary approaches:

- **LLM-based reranking:** large language models evaluate candidate concepts and select the most appropriate match.
- **Borda Count:** rank aggregation across retrieval methods.
- **Reciprocal Rank Fusion (RRF):** rank aggregation based on reciprocal candidate ranks.

**3. Evaluation**

The pipeline provides evaluation metrics to assess exact matching, candidate retrieval, candidate selection and overall normalization performance:

- **Exact precision:** precision of assignments obtained through exact matching.
- **Recall@k:** proportion of evaluated mentions for which at least one correct reference concept is retrieved within the top *k* candidates.
- **RerankingAccuracy@k:** proportion of mentions with at least one correct retrieved candidate for which reranking selects a correct concept.
- **OverallAccuracy@k:** proportion of all evaluated mentions for which reranking selects at least one correct reference concept.

Hierarchically related concepts (ancestors and descendants) can additionally be considered correct according to the evaluation criteria.

---

## Directory structure

The repository is organised into the following main directories:

```text
KnownRD/
│
├── src/
│   └── entity_normalization/
│       ├── normalize.py
│       ├── retrieval.py
│       ├── reranking.py
│       └── evaluation/
│           ├── recall.py
│           └── metrics.py
│
├── scripts/
│   ├── run_retrieval.py
│   ├── run_reranking.py
│   └── run_evaluation.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── examples/
│
├── models/
│
├── results/
│
├── notebooks/
│
├── tests/
│
├── pyproject.toml
├── uv.lock
├── environment.yml
├── README.md
└── .gitignore
```

**Note:** The directory structure may be adapted depending on the selected ontologies, retrieval methods and evaluation datasets.

Large ontology files, UMLS resources, model weights and generated results are not included in the GitHub repository.

---

## Download ontology versions

KnownRD supports multiple biomedical reference ontologies. Ontology files must be downloaded separately and placed in the appropriate directories.

The ontology versions used in the current implementation are:

| Ontology | Description | Version |
|---|---|---|
| UBERON | Uber-anatomy ontology | 2023-07-25 |
| CL | Cell Ontology | 2026-08-06 |
| HPO | Human Phenotype Ontology | 2026-06-23 |
| GO | Gene Ontology | Specify release |
| UMLS | Unified Medical Language System | 2026AA |

Ontology resources are available from their respective official sources:

- [UBERON](https://obofoundry.org/ontology/uberon.html)
- [Cell Ontology](https://obofoundry.org/ontology/cl.html)
- [Human Phenotype Ontology](https://hpo.jax.org/)
- [Gene Ontology](https://geneontology.org/)
- [UMLS](https://www.nlm.nih.gov/research/umls/)

UMLS requires registration and acceptance of the applicable licence agreement. Users must download the files independently.

Place the downloaded ontology files in:

```text
data/raw/
```

For example:

```text
data/raw/
├── uberon.owl
├── cl.owl
├── hp.owl
├── go-basic.owl
├── MRSTY.RRF
└── MRCONSO_ENG.RRF
```

The expected filenames and directory locations should match the corresponding configuration of the pipeline.

---

## Installation

### Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- CUDA-compatible GPU (optional, for GPU-accelerated semantic retrieval)
- Git

### Clone the repository

```bash
git clone https://github.com/USERNAME/KnownRD.git
cd KnownRD
```

### Install uv

```bash
pip install uv
```

### Install dependencies

```bash
uv sync
```

To install development dependencies:

```bash
uv sync --extra dev
```

### GPU configuration

Semantic retrieval using SapBERT can benefit from GPU acceleration.

PyTorch and CUDA compatibility depends on the GPU, operating system and CUDA environment. Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for the appropriate installation.

Verify GPU availability:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## Run candidate retrieval

Candidate retrieval identifies potential reference concepts for each biomedical mention.

The retrieval stage combines exact matching with lexical and semantic retrieval methods.

Available methods:

| Method | Description |
|---|---|
| Exact matching | Matches preferred ontology labels and synonyms |
| BM25 | Lexical ranking |
| Fuzzy | Approximate string similarity |
| N-gram | Character n-gram similarity |
| SapBERT | Semantic similarity using biomedical embeddings |

To run candidate retrieval:

```bash
uv run python scripts/run_retrieval.py
```

The retrieval configuration should specify:

- Input dataset and entity type.
- Reference ontology.
- Retrieval methods.
- Number of candidates to retrieve.
- Output directory.

Retrieval results include the query mention, candidate concept, retrieval score, retrieval method and candidate rank.

---

## Run candidate selection

Candidate selection reranks the concepts retrieved in the previous stage.

KnownRD implements three reranking approaches:

1. LLM-based reranking.
2. Borda Count.
3. Reciprocal Rank Fusion (RRF).

To run candidate selection:

```bash
uv run python scripts/run_reranking.py
```

The reranking stage takes the candidate retrieval outputs as input and produces ranked candidate concepts for each mention.

For LLM-based reranking, the configuration should specify the language model and candidate-set size.

For rank aggregation, Borda Count and RRF combine rankings from the selected retrieval methods.

---

## Run evaluation metrics

KnownRD provides evaluation modules for candidate retrieval, reranking and the complete normalization pipeline.

To run the evaluation:

```bash
uv run python scripts/run_evaluation.py
```

### Evaluation metrics

| Metric | Evaluation stage | Description |
|---|---|---|
| Exact precision | Exact matching | Precision of exact-match assignments |
| Recall@k | Candidate retrieval | Proportion of mentions with at least one correct candidate among the top *k* retrieved concepts |
| RerankingAccuracy@k | Candidate selection | Accuracy among mentions for which at least one correct candidate was retrieved |
| OverallAccuracy@k | Complete pipeline | Proportion of all mentions for which reranking selects at least one correct concept |

Evaluation results can be calculated for different candidate-set sizes and retrieval/reranking configurations.

---

## Results directory

Generated outputs are organised under the `results/` directory.

Results may include:

- Candidate retrieval outputs.
- Reranking predictions.
- Evaluation metrics.
- Performance comparisons across retrieval and reranking methods.
- Figures and statistical analyses.

Results are generated during pipeline execution and are not included in the repository by default.

---

## Citation

If you use KnownRD in your research, please cite the corresponding publication:

**Publication details to be added.**

---

## License

Specify the project licence here.

---

## Contact

For questions, suggestions or issues, please open a GitHub issue or contact the corresponding author.
