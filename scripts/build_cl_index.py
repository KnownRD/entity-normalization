from pathlib import Path

from entity_normalization.parse_cl_ontology import parse_ontology_owl
from entity_normalization.build_lexical_index import build_lexical_docs, build_embeddings_doc


RAW = Path("data/raw")
PROCESSED = Path("data/processed")
INDICES = Path("data/indices")

PROCESSED.mkdir(parents=True, exist_ok=True)
(INDICES / "faiss").mkdir(parents=True, exist_ok=True)

df = parse_ontology_owl(RAW / "cl.owl")
df.to_parquet(PROCESSED / "cl_concepts.parquet", index=False)

build_lexical_docs(df, PROCESSED / "cl_lexical_docs.jsonl")

build_embeddings_doc(PROCESSED / "cl_lexical_docs.jsonl", PROCESSED / "cl_sapbert_embeddings.npy")

