from pathlib import Path

from entity_normalization.parse_umls_ontology import parse_ontology_df
from entity_normalization.build_lexical_index_umls import build_lexical_docs, build_embeddings_doc


RAW = Path("data/raw")
PROCESSED = Path("data/processed")
INDICES = Path("data/indices")

PROCESSED.mkdir(parents=True, exist_ok=True)
(INDICES / "faiss").mkdir(parents=True, exist_ok=True)

df = parse_ontology_df(RAW / "MRCONSO_ENG.RRF", "cell")
df.to_parquet(PROCESSED / "umls_cell_concepts.parquet", index=False)

build_lexical_docs(df, PROCESSED / "umls_cell_lexical_docs.jsonl")

build_embeddings_doc(PROCESSED / "umls_cell_lexical_docs.jsonl", PROCESSED / "umls_cell_sapbert_embeddings.npy")

