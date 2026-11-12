#==========================================================================================================================

# Reranking retrieved candidates from lexical and semantic retrieval methods
# Each queries candidates are reranked with 3 methods:
    # LLM-based reranker
    # Borda count reranker
    # RRF reranker
# A file from each reranker is generated in results/reranker

#==========================================================================================================================


import numpy as np
import argparse
import pandas as pd
from entity_normalization.llm_reranker_prompt import classify
import importlib
from pathlib import Path

def llm_reranker(results_noramlization, ontology_owl_file, k, method,ID, entity):
    module = importlib.import_module(
            f"entity_normalization.parse_{ID.lower()}_ontology"
        )
    ontology_parse = module.parse_ontology_owl(ontology_owl_file) 
    term_cols = [
        "label",
        "exact_synonyms",
        "related_synonyms",
        "broad_synonyms",
        "narrow_synonyms"
    ]

    def get_all_terms(group):
        terms = []

        for col in term_cols:
            for value in group[col]:
                if isinstance(value, list):
                    terms.extend(value)
                elif pd.notna(value):
                    terms.append(value)

        return list(dict.fromkeys(terms))


    grouped = (
        ontology_parse
        .groupby("concept_id")
        .apply(get_all_terms, include_groups=False)
        .reset_index(name="terms")
        .rename(columns={"concept_id": "candidate"})
    )
    
    if method == "combined":
        results_noramlization_filter = results_noramlization[results_noramlization["method"] != "sapbert_cell"]
    else:
        results_noramlization_filter = results_noramlization[results_noramlization["method"] != method]
    
                
    results_noramlization_topk = results_noramlization_filter[results_noramlization_filter["rank"] <= k][["candidate", "label", "query"]].drop_duplicates()
    results_noramlization_topk = results_noramlization_topk[["candidate", "query"]].drop_duplicates().merge(grouped, on = "candidate", how="left")
    
    log_file=Path(f"tmp/log/{ID.lower()}_{entity}_log_file_{k}_{method}.log")
    failed_cases_file=Path(f"tmp/log/{ID.lower()}_{entity}_failed_cases_{k}_{method}.log")
    output_file=Path(f"tmp/{ID.lower()}_{entity}_LLM_{k}_{method}.tsv")
    
    candidates_reranked = classify(results_noramlization_topk, output_file, log_file, failed_cases_file)
    topk_candidates = results_noramlization_topk.groupby("query")["candidate"].apply(set).apply(list).reset_index().rename(columns={"candidate": "candidate_list"})
    candidates_reranked = candidates_reranked.merge(topk_candidates, on= "query", how="left")
    return candidates_reranked
    

def rrf_reranker(results_noramlization, k, method):
        if method == "combined":
            results_noramlization_filter = results_noramlization[results_noramlization["method"] != "sapbert_cell"]
        else:
            results_noramlization_filter = results_noramlization[results_noramlization["method"] != method]
        topk_candidates_matrix= results_noramlization_filter.groupby(["query", "candidate", "method"])["rank"].min().reset_index().pivot(columns="method", values="rank", index=["query", "candidate"]).reset_index().fillna(np.inf)
        if method == "combined":
           topk_candidates = topk_candidates_matrix[topk_candidates_matrix[["bm25", "fuzzy", "ngram", "sapbert"]].le(k).any(axis=1)]
        else:
           topk_candidates = topk_candidates_matrix[topk_candidates_matrix[["sapbert"]].le(k).any(axis=1)]

        candidates_list = topk_candidates.groupby("query")["candidate"].apply(set).apply(list).reset_index().rename(columns={"candidate": "candidate_list"})
        
        for method in ["bm25", "fuzzy", "ngram", "sapbert"]:
            topk_candidates[f"{method}_score"] = 1 / (60 + topk_candidates[method])
        
        topk_candidates["rank_score"] = topk_candidates[[f"{method}_score" for method in ["bm25", "fuzzy", "ngram", "sapbert"]]].sum(axis=1)
        results_score = topk_candidates.merge(candidates_list, on="query", how="left")
        results_score["rank"] = (
                results_score
                .groupby("query")["rank_score"]
                .rank(method="min", ascending=False)
                .astype(int)
            )
        
        results_score = results_score.sort_values(
            ["query", "rank_score"],
            ascending=[True, False]
        )
        results_score_k = results_score[results_score["rank"] <= 1][["candidate", "query", "candidate_list"]]
        
        return results_score_k
    
def borda_reranker(results_noramlization, k, method):
        if method == "combined":
            results_noramlization_filter = results_noramlization[results_noramlization["method"] != "sapbert_cell"]
        else:
            results_noramlization_filter = results_noramlization[results_noramlization["method"] != method]
        topk_candidates_matrix= results_noramlization_filter.groupby(["query", "candidate", "method"])["rank"].min().reset_index().pivot(columns="method", values="rank", index=["query", "candidate"]).reset_index().fillna(101)
        if method == "combined":
            topk_candidates = topk_candidates_matrix[topk_candidates_matrix[["bm25", "fuzzy", "ngram", "sapbert"]].le(k).any(axis=1)]
        else:
            topk_candidates = topk_candidates_matrix[topk_candidates_matrix[[method]].le(k).any(axis=1)]

        candidates_list = topk_candidates.groupby("query")["candidate"].apply(set).apply(list).reset_index().rename(columns={"candidate": "candidate_list"})
        
        topk_candidates["rank_score"] = topk_candidates[["bm25", "fuzzy", "ngram", "sapbert"]].sum(axis=1)
        results_score = topk_candidates.merge(candidates_list, on= "query", how="left")
        results_score["rank"] = (
                results_score
                .groupby("query")["rank_score"]
                .rank(method="min", ascending=True)
                .astype(int)
            )
        
        results_score = results_score.sort_values(
            ["query", "rank_score"],
            ascending=[True, True]
        )
        results_score_k = results_score[results_score["rank"] <= 1][["candidate", "query", "candidate_list"]]
        
        return results_score_k



def main():

    parser = argparse.ArgumentParser(
        description="Rerank biomedical entity normalization candidates"
    )

    parser.add_argument("--results_normalization", required=True)
    parser.add_argument("--ontology_owl_file", required=True)
    parser.add_argumnent("--id", required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--method", choices=["combined", "sapbert"], default="combined")
    parser.add_argument("--entity", required=True)

    args = parser.parse_args()
    ID=args.id
    k=args.k
    method=args.method
    entity=args.entity
    
    # Load input data
    results_normalization = pd.read_csv(
        args.results_normalization, sep="\t"
    )
    output_llm_file=Path(f"results/reranker/{ID.lower()}_{entity}_LLM_reranker_{k}_{method}.tsv")
    output_borda_file=Path(f"results/reranker/{ID.lower()}_{entity}_borda_reranker_{k}_{method}.tsv")
    output_rrf_file=Path(f"results/reranker/{ID.lower()}_{entity}_RRF_reranker_{k}_{method}.tsv")

    # LLM reranking
    llm_results = llm_reranker(
        results_normalization,
        args.ontology_owl_file,
        k,
        method,
        ID,
        entity
    )
    llm_results.to_csv(output_llm_file, sep="\t", index=False)

    # Borda reranking
    borda_results = borda_reranker(
        results_normalization,
        k,
        method
    )

    borda_results.to_csv(output_borda_file, sep="\t", index=False)

    # RRF reranking
    rrf_results = rrf_reranker(
        results_normalization,
        k,
        method
    )

    rrf_results.to_csv(output_rrf_file, sep="\t", index=False)


if __name__ == "__main__":
    main()
