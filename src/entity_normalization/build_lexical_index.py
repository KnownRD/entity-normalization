from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModel  
import numpy as np
import torch
from tqdm import tqdm
from entity_normalization.normalize import normalize_text


def build_lexical_docs(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for row in df.to_dict(orient="records"):
            entries = [("label", row["label"])]
            entries += [("exact", s) for s in row["exact_synonyms"]]
            entries += [("narrow", s) for s in row["narrow_synonyms"]]
            entries += [("broad", s) for s in row["broad_synonyms"]]
            entries += [("related", s) for s in row["related_synonyms"]]

            for scope, text in entries:
                rec = {
                    "concept_id": row["concept_id"],
                    "label": row["label"],
                    "scope": scope,
                    "text": text,
                    "text_norm": normalize_text(text)
                }
                f.write(json.dumps(rec) + "\n")



def build_embeddings_doc(lexical_docs_path:str, out_path: str | Path) -> None:
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)
    docs = [json.loads(line) for line in Path(lexical_docs_path).read_text().splitlines() if line.strip()]
    texts = [d["text_norm"] for d in docs]
    
    n_docs = len(texts)
    hidden_size = model.config.hidden_size    
    bs=128
    embeddings = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_docs, hidden_size)
    )

    with torch.inference_mode():

        for i in tqdm(
            range(0, n_docs, bs),
            desc="Generating SapBERT embeddings"
        ):

            batch = texts[i:i + bs]

            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=25,
                return_tensors="pt"
            )

            toks = {
                k: v.to(device)
                for k, v in toks.items()
            }

            output = model(**toks)

            cls_rep = output.last_hidden_state[:, 0, :]

            # Save directly to disk
            embeddings[i:i + len(batch)] = (
                cls_rep.cpu().numpy()
            )

    embeddings.flush()

    del embeddings
    del model
    del tokenizer
    if "cl" in str(lexical_docs_path):
        tokenizer = AutoTokenizer.from_pretrained("almire/CellExLink-Sapbert")
        model = AutoModel.from_pretrained("almire/CellExLink-Sapbert").to(device)
        docs = [json.loads(line) for line in Path(lexical_docs_path).read_text().splitlines() if line.strip()]
        texts = [d["text_norm"] for d in docs]
        
        n_docs = len(texts)
        hidden_size = model.config.hidden_size    
        bs=128
        embeddings = np.lib.format.open_memmap(
            Path("data/processed/CellExLink_sapbert_embeddings.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(n_docs, hidden_size)
        )

        with torch.inference_mode():

            for i in tqdm(
                range(0, n_docs, bs),
                desc="Generating cell SapBERT embeddings"
            ):

                batch = texts[i:i + bs]

                toks = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=25,
                    return_tensors="pt"
                )

                toks = {
                    k: v.to(device)
                    for k, v in toks.items()
                }

                output = model(**toks)

                cls_rep = output.last_hidden_state[:, 0, :]

                # Save directly to disk
                embeddings[i:i + len(batch)] = (
                    cls_rep.cpu().numpy()
                )

        embeddings.flush()
        
        del embeddings
        del model
        del tokenizer
    print(f"Saved embeddings to {out_path}")