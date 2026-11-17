from __future__ import annotations
import json
from pathlib import Path
from statistics import mean
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
import pandas as pd
from entity_normalization.normalize import normalize_text, singularize
from transformers import AutoTokenizer, AutoModel  
import numpy as np
import torch
from pathlib import Path

top_k = 100

class LexicalRetriever:
    def __init__(self, lexical_docs_path: str, embeddings_path: str):
        self.device = torch.device("cuda:0")
        self.path = str(lexical_docs_path)
        self.tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        self.model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(self.device)
        self.tokenizer_cell = AutoTokenizer.from_pretrained("almire/CellExLink-Sapbert")
        self.model_cell = AutoModel.from_pretrained("almire/CellExLink-Sapbert").to(self.device)
        self.docs = [json.loads(line) for line in Path(lexical_docs_path).read_text().splitlines() if line.strip()]
        self.texts = [d["text_norm"] for d in self.docs]
        self.tokens = [d["text_norm"].split() for d in self.docs]
        self.bm25 = BM25Okapi(self.tokens)

        # Exact matching
        self.exact_map = {}

        for d in self.docs:
            self.exact_map.setdefault(
                d["text_norm"], []
            ).append(d)

        self.all_reps_emb = np.load(
            embeddings_path
            )
        self.all_reps_emb_cell = np.load(
            Path("data/processed/CellExLink_sapbert_embeddings.npy")
            )      
    def char_ngrams(self, text: str, n: int = 3) -> set[str]:
        text = normalize_text(text).replace(" ", "_")
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text)-n+1)}


    def jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
    
    def rank_and_threshold(self, results: list, top_k: int):
        # Sort by descending score
        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )
    
        # Keep the highest-scoring entry for each candidate
        unique_results = {}
        for item in results:
            candidate = item["candidate"]
    
            if candidate not in unique_results:
                unique_results[candidate] = item
    
        results = list(unique_results.values())
    
        # Assign ranks, giving tied scores the same rank
        rank = 1
        prev_score = None
    
        for i, item in enumerate(results):
            if item["score"] != prev_score:
                rank = i + 1
                prev_score = item["score"]
    
            item["rank"] = rank
    
        # Keep top-k candidates, including ties at the threshold
        if len(results) > top_k:
            threshold = results[top_k - 1]["score"]
            results = [
                item for item in results
                if item["score"] >= threshold
            ]
    
        return results
    

    def search(self, query: str):
        qn = normalize_text(query)
        qn_bm25=singularize(qn)
        exact_out = []
        bm25_out = []
        fuzzy_out = []
        ngram_out = []
        sapbert_output = []
        sapbert_output_cell = []
       # exact_candidates
        for d in self.exact_map.get(qn, []): 
            exact_out.append({
                "candidate": d["concept_id"],
                "label": d["text"],
                "score": 1000.0,
                "method": "exact",
                "synonym_type": d.get("scope"),
            })

        
        exact_out = sorted(exact_out,key=lambda c: c["score"],reverse=True,)
        rank = 1
        prev_score = None
        
        for i, item in enumerate(exact_out):
            if item["score"] != prev_score:
                rank = i + 1
                prev_score = item["score"]
            item["rank"] = rank
        if len(exact_out) > 0:
            return exact_out
        
        # bm25
        bm25_scores = self.bm25.get_scores(qn_bm25.split())

        
        # ONE pass over the whole ontology
        for i, d in enumerate(self.docs):
    # ---------------- BM25 ----------------
            bm25_score = bm25_scores[i]

            if bm25_score > 0:
                bm25_out.append({
                    "candidate": d["concept_id"],
                    "label": d["text"],
                    "score": float(bm25_score),
                    "method": "bm25",
                    "synonym_type": d.get("scope"),
                })

            # ---------------- FUZZY ----------------
            sim = mean((
                fuzz.ratio(qn, d["text_norm"]),
                fuzz.partial_ratio(qn, d["text_norm"]),
                fuzz.token_sort_ratio(qn, d["text_norm"]),
                fuzz.token_set_ratio(qn, d["text_norm"]),
            )) / 100.0
        
            if sim > 0:
                fuzzy_out.append({
                    "candidate": d["concept_id"],
                    "label": d["text"],
                    "score": sim,
                    "method": "fuzzy",
                    "synonym_type": d.get("scope"),
                })

    # ---------------- NGRAM ----------------
            query_ngrams = self.char_ngrams(qn)
            doc_ngrams = self.char_ngrams(d["text_norm"])
            sim_ngram = self.jaccard(query_ngrams, doc_ngrams)
            if sim_ngram > 0:
                ngram_out.append({
                    "candidate": d["concept_id"],
                    "label": d["text"],
                    "score": sim_ngram,
                    "method": "ngram",
                    "synonym_type": d.get("scope"),
                })        
    # ---------------- SAPBERT ---------------
        query_toks = self.tokenizer([qn], padding=True, max_length=25, truncation=True, return_tensors="pt")
        query_toks = {
            k: v.to(self.device)
            for k, v in query_toks.items()
        }
        with torch.no_grad():
            query_output = self.model(**query_toks)
            query_rep = query_output.last_hidden_state[:, 0, :]
            query_rep = torch.nn.functional.normalize(query_rep, p=2, dim=1)
  
        similarities = (query_rep.cpu().detach().numpy() @ self.all_reps_emb.T)[0]
    
        ranked_indices = np.argsort(similarities)[::-1]
        
        for rank, idx in enumerate(ranked_indices, start=1):
        
            doc = self.docs[idx]
        
            sapbert_output.append({
                "candidate": doc["concept_id"],
                "label": doc["text_norm"],
                "score": similarities[idx],
                "method": "sapbert",
                "synonym_type": doc["scope"],
                "rank": rank,
 
            })
    # ---------------- SAPBERT cellexlink ---------------
        if "cl" in str(self.path):
            query_toks_cell = self.tokenizer_cell([qn], padding=True, max_length=25, truncation=True, return_tensors="pt")
            query_toks_cell = {
                k: v.to(self.device)
                for k, v in query_toks_cell.items()
            }
            with torch.no_grad():
                query_output_cell = self.model_cell(**query_toks_cell)
                query_rep_cell = query_output_cell.last_hidden_state[:, 0, :]
                query_rep_cell = torch.nn.functional.normalize(
                    query_rep_cell, p=2, dim=1
                )
            similarities_cell = (query_rep_cell.cpu().detach().numpy() @ self.all_reps_emb_cell.T)[0]
        
            ranked_indices_cell = np.argsort(similarities_cell)[::-1]
    
            for rank, idx in enumerate(ranked_indices_cell, start=1):
            
                doc = self.docs[idx]
            
                sapbert_output_cell.append({
                    "candidate": doc["concept_id"],
                    "label": doc["text_norm"],
                    "score": similarities_cell[idx],
                    "method": "sapbert_cell",
                    "synonym_type": doc["scope"],
                    "rank": rank
                })


        bm25_out = self.rank_and_threshold(bm25_out, top_k)
        fuzzy_out = self.rank_and_threshold(fuzzy_out, top_k)
        ngram_out = self.rank_and_threshold(ngram_out, top_k)
        sapbert_output = self.rank_and_threshold(sapbert_output, top_k)
        sapbert_output_cell = self.rank_and_threshold(sapbert_output_cell, top_k)

        results = exact_out + bm25_out + fuzzy_out + ngram_out + sapbert_output + sapbert_output_cell            

        df = pd.DataFrame(results)
        
        return df