from pathlib import Path

from entity_normalization.parse_hpo_ontology import parse_ontology_owl
from entity_normalization.build_lexical_index import build_lexical_docs, build_embeddings_doc


RAW = Path("data/raw")
PROCESSED = Path("data/processed")
INDICES = Path("data/indices")

PROCESSED.mkdir(parents=True, exist_ok=True)
(INDICES / "faiss").mkdir(parents=True, exist_ok=True)

df = parse_ontology_owl(RAW / "hp.owl")
df.to_parquet(PROCESSED / "hpo_concepts.parquet", index=False)

build_lexical_docs(df, PROCESSED / "hpo_lexical_docs.jsonl")

build_embeddings_doc(PROCESSED / "hpo_lexical_docs.jsonl", PROCESSED / "hpo_sapbert_embeddings.npy")

