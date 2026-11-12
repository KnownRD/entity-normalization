from pathlib import Path

from entity_normalization.parse_uberon_ontology import parse_ontology_owl
from entity_normalization.build_lexical_index import build_lexical_docs, build_embeddings_doc

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
INDICES = Path("data/indices")

PROCESSED.mkdir(parents=True, exist_ok=True)
(INDICES / "faiss").mkdir(parents=True, exist_ok=True)

df = parse_ontology_owl(RAW / "uberon.owl")
df.to_parquet(PROCESSED / "uberon_concepts.parquet", index=False)

build_lexical_docs(df, PROCESSED / "uberon_lexical_docs.jsonl")

build_embeddings_doc(PROCESSED / "uberon_lexical_docs.jsonl", PROCESSED / "uberon_sapbert_embeddings.npy")

