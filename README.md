# KnownRD – Biomedical Entity Normalization

KnownRD is a modular biomedical named entity normalization (NEN) pipeline that links biomedical mentions to standardized concepts in biomedical reference ontologies.

The pipeline combines exact matching, lexical and semantic candidate retrieval, and candidate reranking. The workflow does not require ontology-specific training or architectural modifications, although ontology-adapted models can be incorporated when available.

## Workflow

KnownRD consists of three main stages:

**1. Candidate retrieval**

Candidate concepts are retrieved through exact matching and complementary lexical and semantic retrieval methods:

- **BM25:** lexical retrieval based on term relevance.
- **Fuzzy matching:** approximate string matching.
- **N-gram:** character n-gram similarity.
- **SapBERT:** semantic retrieval.

**2. Candidate selection (reranking)**

Retrieved candidates are ranked and selected using three complementary approaches:

- **LLM-based reranking:** large language models evaluate candidate terms and select the most appropriate match.
- **Borda Count:** rank aggregation across retrieval methods.
- **Reciprocal Rank Fusion (RRF):** rank aggregation based on reciprocal candidate ranks.

**3. Evaluation**

- **Exact precision:** precision of candidates obtained through exact matching.
- **Recall\@k:** proportion of evaluated mentions for which at least one correct reference concept is retrieved within the top *k* candidates.
- **RerankingAccuracy\@k:** proportion of mentions with at least one correct retrieved candidate for which reranking selects a correct concept.
- **OverallAccuracy\@k:** proportion of all evaluated mentions for which reranking selects at least one correct reference concept.

Hierarchically related concepts (ancestors and descendants) can additionally be considered correct according to the evaluation criteria.

---

## Directory structure

The repository is organised into the following main directories:

```text
KnownRD/
│
├── src/
│   └── entity_normalization/
│       ├── build_lexical_index.py
│       ├── build_lexical_index_umls.py
│       ├── llm_reranker_prompt.py
│       ├── normalize.py
│       ├── parse_cl_ontology.py
│       ├── parse_hpo_ontology.py
│       ├── parse_go_ontology.py
│       ├── parse_uberon_ontology.py
│       ├── parse_umls_ontology.py
│       ├── retrieve_candidates.py
│       └── schema.py
│
├── scripts/
│   ├── build_cl_index.py
│   ├── build_go_index.py
│   ├── build_uberon_index.py
│   ├── build_hpo_index.py
│   ├── build_uberon_index.py
│   ├── evaluation_normalization.py
│   ├── reranking_candidates_pipeline.py
│   └── retrieval_concepts_pipeline.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── retrieval/
│   ├── plots/
│   └── reranker/
│
├── pyproject.toml
├── README.md
└── .gitignore
```

Large ontology files, UMLS resources, model weights and generated results are not included in the GitHub repository.

---

## Download ontology versions

KnownRD supports multiple biomedical reference ontologies. Ontology files must be downloaded separately.

The ontology versions used in the current implementation are:

| Ontology | Description                     | Version    |
| -------- | ------------------------------- | ---------- |
| UBERON   | Uber-anatomy ontology           | 2023-07-25 |
| CL       | Cell Ontology                   | 2026-08-06 |
| HPO      | Human Phenotype Ontology        | 2026-06-23 |
| GO       | Gene Ontology                   | 2026-06-15 |
| UMLS     | Unified Medical Language System | 2026AA     |

Ontology resources are available from their respective official sources:

- [UBERON](https://obofoundry.org/ontology/uberon.html)
- [Cell Ontology](https://obofoundry.org/ontology/cl.html)
- [Human Phenotype Ontology](https://hpo.jax.org/)
- [Gene Ontology](https://geneontology.org/)
- [UMLS](https://www.nlm.nih.gov/research/umls/)

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
---

## Installation

### Clone the repository

```bash
git clone https://github.com/KnownRD/entity-normalization.git
cd entity-normalization
```

### Install dependencies

```bash
pip install -e .
```

---

## Processed data

Before running candidate retrieval, the ontology resources must be processed to generate the ontology dictionaries required by the normalization pipeline.

For each ontology, execute the corresponding `build_x_index.py` script, where `x` identifies the ontology.

```text
python scripts/build_cl_index.py
```

The generated dictionaries are stored in:

```text
data/processed/
```

**Important:** This step must be completed before running candidate retrieval. The retrieval scripts use the generated ontology dictionaries rather than processing the original ontology files at runtime.

---

## Run candidate retrieval

Candidate retrieval identifies potential reference concepts for each biomedical mention.

To run candidate retrieval:

```bash
python scripts/retrieval_concepts_pipeline.py \
        --terms_file $"path" \
        --output_file $"name" \
        --output_exact $"name" \
        --lexical_jsonl_file $"path" \
        --embeddings_path $"path"

        
```
The retrieval input data should specify:

- Path to dataset with list of terms to analyze.
- Name of retrieval output (in /results/retrieval/).
- Name of exact-matching retrieval output (in /results/retrieval/) .
- Path to ontology dictionary (in /data/processed/).
- Path to embeddings dictionary (in /data/processed/).

---

## Run candidate selection

Candidate selection reranks the concepts retrieved in the previous stage.

To run candidate selection:

```bash
python scripts/reranking_candidates_pipeline.py \
        --results_normalization $"path" \
        --ontology_owl_file $"path" \
        --id $"id_ontology" \
        --k $"num_k" \
        --method $"combined/sapbert" \
        --entity $"name_entity"
```

The reranking input data should specify:

- Path to input retrieval file (in /results/retrieval/)
- Path to reference ontology (in /data/raw/)
- Id_term of the reference ontology (CL, UBERON, HPO, GO, ...)
- Number of candidates to retrieve.
- Type of candidates to rerank (sapbert - only sapbert candidates; combined - bm25, fuzzy, n-gram and sapbert candidates) (default="combined") 
- Entity name
---

## Run evaluation metrics

The pipeline provides evaluation modules for candidate retrieval, reranking and the complete normalization pipeline.

To run the evaluation:

```bash
python scripts/evaluation_normalization.py \
        --exact_normalization $"path" \
        --results_normalization $"path" \
        --ontology_owl_file $"path" \
        --id $"id_ontology" \
        --k $"num_k" \
        --method $"combined/sapbert" \
        --entity $"name_entity"
```
The evaluation input data should specify:

- Path to exact-matching retrieval file (in /results/retrieval/)
- Path to retrieval file (in /results/retrieval/)
- Path to reference ontology (in /data/raw/)
- Id_term of the reference ontology (CL, UBERON, HPO, GO, ...)
- Number of candidates to retrieve.
- Type of candidates to rerank (sapbert - only sapbert candidates; combined - bm25, fuzzy, n-gram and sapbert candidates) (default="combined") 
- Entity name

---

## Results directory

Generated outputs are organized under the `results/` directory.

Results may include:

- Candidate retrieval outputs in /results/retrieval/
- Reranking predictions in /results/reranker/
- Evaluation metrics in /results/
- Temporary files (log and llm_reranker files) in /tmp/
- Figures and statistical analyses in /results/plots/

Results are generated during pipeline execution and are not included in the repository by default.

---

## Citation

