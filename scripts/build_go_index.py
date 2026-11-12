from pathlib import Path

from entity_normalization.parse_go_ontology import parse_ontology_owl
from entity_normalization.build_lexical_index import build_lexical_docs, build_embeddings_doc


RAW = Path("data/raw")
PROCESSED = Path("data/processed")
INDICES = Path("data/indices")

PROCESSED.mkdir(parents=True, exist_ok=True)
(INDICES / "faiss").mkdir(parents=True, exist_ok=True)

df = parse_ontology_owl(RAW / "go-basic.owl")
df.to_parquet(PROCESSED / "go_concepts.parquet", index=False)
for entity in ["biological_process", "molecular_function", "cellular_component"]:
    df_entity = df[df["namespace"] == entity]
    build_lexical_docs(df_entity, PROCESSED / f"{entity}_lexical_docs.jsonl")

    build_embeddings_doc(PROCESSED / f"{entity}_lexical_docs.jsonl", PROCESSED / f"{entity}_sapbert_embeddings.npy")
