#==========================================================================================================================

# Evaluation of all steps from the normalization pipeline
    # Precision: metric to evaluate exact-matching retrieval candidates
    # Recall: metric to evaluate candidate retrievl with lexical and semantic methods
    # Reranking_accuracy: metric to evaluate candidate reranking (with all 3 methods)
    # Overall accuracy: metric to evaluate the whole non-exact matching normalization

# Recall, Reranking_accuracy and Overall_accuracy are representated in two different plots

#==========================================================================================================================



import pandas as pd
import ast
import argparse
import matplotlib.pyplot as plt
import importlib
from pathlib import Path

def exact_precision(df_val, exact_normalization, ontology_parse, entity, parents, children, ID):
    candidates_list_exact=exact_normalization.groupby("query")["candidate"].apply(set).apply(list).reset_index().rename(columns={"candidate": "candidate_list"})

    priority = {
        "label": 0,
        "exact": 1,
        "related": 3,
        "broad": 2,
        "narrow": 4
    }

    exact_normalization["priority"] = exact_normalization["synonym_type"].map(priority)

    exact_normalization = (
        exact_normalization.sort_values(["query", "priority"])
          .drop_duplicates("query", keep="first")
          .drop(columns="priority"))
          
                   
    df_normalization = exact_normalization.merge(df_val, on="query", how="left").merge(candidates_list_exact, on="query", how="left")
    df_normalization["rank"] = df_normalization["rank"].astype(int)
    df_normalization["correct"] = (df_normalization.apply(lambda row: (row["candidate"] in row[ID]),axis=1).astype(int))
    df_normalization["correct"] = df_normalization["correct"].fillna(0).astype(int)
    recall = (
        df_normalization.groupby("query")["correct"].max().mean()
    )
    error_num=df_normalization.groupby("query")["correct"].max().eq(0).sum()
    recall_parents=None
    recall_parents_children=None
    if parents==True:
        df_normalization.loc[(df_normalization["correct"] == 0) & df_normalization.apply(lambda row: (row["candidate"] in row["parents_id"]) ,axis=1),"correct"] = 1
        recall_parents = (
           df_normalization.groupby("query")["correct"].max().mean()
        )
    error_num_parents=df_normalization.groupby("query")["correct"].max().eq(0).sum()
    if children==True:
        df_normalization.loc[(df_normalization["correct"] == 0) & df_normalization.apply(lambda row: (row["candidate"] in row["children_id"]) ,axis=1),"correct"] = 1
        recall_parents_children = (
           df_normalization.groupby("query")["correct"].max().mean()
        )
    error_num_parents_children=df_normalization.groupby("query")["correct"].max().eq(0).sum()
    if isinstance(ontology_parse, pd.DataFrame):
        queries_incorrect = (df_normalization.groupby("query")["correct"].max().loc[lambda x: x == 0].index)
        queries_norm_incorrect=df_normalization[df_normalization["query"].isin(queries_incorrect)].rename(columns={"label":"label_search"})
        queries_norm_incorrect = queries_norm_incorrect.merge(ontology_parse[["concept_id", "label"]].drop_duplicates(), left_on=ID, right_on="concept_id", how="left" ).rename(columns={"label":"label_gold"})
        queries_norm_incorrect = queries_norm_incorrect.merge(ontology_parse[["concept_id", "label"]].drop_duplicates(), left_on="candidate", right_on="concept_id", how="left" ).rename(columns={"label":"label_candidate"})
        queries_norm_incorrect = queries_norm_incorrect[["candidate",ID,"synonym_type","query", "label_search","label_gold", "label_candidate"]]
    else:
        queries_norm_incorrect=None


    print(f"number_terms {entity}: {exact_normalization['query'].nunique()}")
    print(f"num_error_terms {entity}: {error_num}")
    print(f"recall {entity}: {recall}")
    print(f"num_error_terms_parents {entity}:  {error_num_parents}")
    print(f"recall_parents {entity}: {recall_parents}")
    print(f"num_error_terms_parents_children {entity}:  {error_num_parents_children}")
    print(f"recall_parents-children {entity}: {recall_parents_children}")
    return 

def recall_per_method(positive_list_file, results_noramlization, entity, ID):
    metrics_recall = {}
    results_noramlization = results_noramlization[results_noramlization["method"] != "sapbert_cell"]
    results_linker = results_noramlization.merge(positive_list_file, left_on="query", right_on="text", how="left").drop(columns={"text"})
    results_linker["rank"] = results_linker["rank"].astype(int)
    results_linker["correct"] = (results_linker.apply(lambda row: row["candidate"] in row[ID],axis=1).astype(int))
    results_linker["correct"] = results_linker["correct"].fillna(0).astype(int)
    results = []

    
    for method, df in results_linker.groupby("method"):
    
        for k in range(1, 51):
    
            topk = df[df["rank"] <= k][["candidate", ID,"rank", "method", "query", "correct"]].drop_duplicates()
            k_correct=topk.groupby("query")["candidate"].apply(lambda x: len(set(x))).median()

            recall = (
                topk.groupby(["query"])["correct"].max().mean()
            )
    
            results.append({
                "method": method,
                
                "k": k_correct,
                "recall": recall
            })
    
    metrics_df = pd.DataFrame(results)
    
    results_merge=[]
    df_method = results_linker
    for k in range(1, 51):
            topk = df_method[df_method["rank"] <= k][["candidate", ID, "rank", "query", "correct"]].drop_duplicates()
            k_correct=topk.groupby("query")["candidate"].apply(lambda x: len(set(x))).median()
            recall = (
                topk.groupby(["query"])["correct"].max().mean()
            )
        
            results_merge.append({
                "method":"All",
                "k": k_correct,
                "recall": recall
            })
    
    metrics_merge = pd.DataFrame(results_merge)
    metrics_recall= pd.concat(
        [metrics_df, metrics_merge],
        axis=0
    )
    return metrics_recall
        
def plot_recall(metrics_recall, entity):
    styles = {
        "bm25": {
            "color": "#56B4E9",
            "marker": "s",
            "markersize": 3,
            "linestyle": "-",
        },
        "fuzzy": {
            "color": "#E69F00",
            "marker": "*",
            "markersize": 4.5,
            "linestyle": "-",
        },
        "ngram": {
            "color": "#009E73",
            "marker": "^",
            "markersize": 3,
            "linestyle": "-",
        },
        "sapbert": {
            "color": "#d73027",
            "marker": "D",
            "markersize": 3,
            "linestyle": "-",
        },
        "combined": {
            "color": "#4a148c",
            "marker": "o",
            "markersize": 3,
            "linestyle": "-",
        },
        "sapbert_cell": {
            "color": "#bd81e6",
            "marker": "s",
            "markersize": 3,
            "linestyle": "-",
        },
        "cellexlink": {
            "color": "#1f6f8b",
            "marker": "o",
            "markersize": 3,
            "linestyle": "--",
        },
    }
    for method, y in metrics_recall.groupby("method"):
        style = styles[method.lower()]
        # y = y[y["k"] <= 25]

        plt.plot(
            y["k"],
            y["recall"],
            color=style["color"],
            # markersize=style["markersize"],
            # marker=style["marker"],
            # markersize=2.5,
            linestyle=style["linestyle"],
            linewidth=1.6,
            label=method

        )
    
    plt.xlabel("Median number of retrieved candidates")
    plt.ylabel("Recall")
    plt.title(f"{entity}")
    plt.xlim(0, 35)
    # plt.ylim(0.3, 1)
    plt.grid(True, color="gray", alpha=0.4, linestyle="-", linewidth=0.5)
    # plt.grid(True)
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


    plt.tight_layout()
    plt.show()
    return 

def plot_accuracy(metrics_table, entity):
    colors = {
        "Combined": "#4a148c",
        "SapBERT": "#d73027"   
    }

    styles = {
        "Borda count": "--",
        "LLM": "-",
        "RRF": ":"
    }
    # colors = {
    #     "Combined": "#6400c0",
    #     "sapbert": "#ff2a2b",
    #     "bm25": "#67A9CF",
    #     "fuzzy": "#F4A340",
    #     "ngram": "#7FB69D"
    # }
    
    reranker_order = ["LLM","Borda count" "RRF"]
    method_order = ["Combined", "SapBERT"]
    
    for metric in ["Reranker accuracy", "Overall accuracy"]:
    
        # plt.figure(figsize=(7, 5))
    
        for reranker in reranker_order:
            for method in method_order:
    
                y = metrics_table[
                    (metrics_table["reranker"] == reranker) &
                    (metrics_table["method"].str.lower() == method.lower())
                ]
    
                if y.empty:
                    continue
    
                plt.plot(
                    y["k"],
                    y[metric],
                    color=colors[method],
                    linestyle=styles[reranker],
                    markersize=3,
                    linewidth=1.6,
                    label=f"{reranker} - {method}"
                )
    
        plt.xlabel("Median number of retrieved candidates")
        plt.ylabel(metric)
        plt.title(entity)
        plt.grid(
            True,
            color="gray",
            alpha=0.3,
            linestyle="-",
            linewidth=0.5
        )
    
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=False
        )
    
        plt.tight_layout()
        plt.show()


def reranker_metrics(df, positive_list_file, parents, children, reranker, method, k, ID):  
    results_list=[]
    df_metrics_reranker = df.merge(positive_list_file, left_on="query", right_on="text", how="inner").drop(columns={"text"})
    df_metrics_reranker["correct_candidate"] = (df_metrics_reranker.apply(lambda row: row["candidate"] in row[ID],axis=1).astype(int))
    df_metrics_reranker["correct_candidate"] = df_metrics_reranker["correct_candidate"].fillna(0).astype(int)
    df_metrics_reranker["correct_accuracy"] = (df_metrics_reranker.apply(lambda row: (row["candidate"] in row[ID]) & (row[ID] in row["candidate_list"]),axis=1).astype(int))
    df_metrics_reranker["correct_accuracy"] = df_metrics_reranker["correct_accuracy"].fillna(0).astype(int)
    df_metrics_reranker["correct_list"] = (df_metrics_reranker.apply(lambda row: (row[ID] in row["candidate_list"]),axis=1).astype(int))
    df_metrics_reranker["correct_list"] = df_metrics_reranker["correct_list"].fillna(0).astype(int)
    df_metrics_reranker["correct_recall"] = df_metrics_reranker['correct_list']
    
    recall = (df_metrics_reranker.groupby("query")["correct_recall"].max().mean())
    sel_accuracy = (df_metrics_reranker.groupby("query")["correct_accuracy"].max().eq(1).sum()) / (df_metrics_reranker.groupby("query")["correct_list"].max().eq(1).sum())
    overall_accuary = (df_metrics_reranker.groupby("query")["correct_candidate"].max().mean()) # also (recall * sel_accuracy)
    ties = df_metrics_reranker[df_metrics_reranker["correct_list"] == 1].groupby(["query"])["candidate"].nunique().reset_index(name="n_candidates")["n_candidates"].gt(1).mean() * 100
    mean_candidates = df_metrics_reranker.groupby("query")["candidate_list"].apply(lambda x: x.str.len().max()).median()
    if parents==True:
            df_metrics_reranker.loc[(df_metrics_reranker["correct_accuracy"] == 0) & df_metrics_reranker.apply(lambda row: (row["candidate"] in row["parents_id"]) & (row[ID] in row["candidate_list"]),axis=1),"correct_accuracy"] = 1
            df_metrics_reranker.loc[(df_metrics_reranker["correct_candidate"] == 0) & df_metrics_reranker.apply(lambda row: row["candidate"] in row["parents_id"],axis=1),"correct_candidate"] = 1
            df_metrics_reranker.loc[(df_metrics_reranker["correct_recall"] == 0) & df_metrics_reranker.apply(lambda row: any(parent in row["candidate_list"] for parent in ast.literal_eval(row["parents_id"])), axis=1), "correct_recall"] = 1
    sel_accuracy_parents = (df_metrics_reranker.groupby("query")["correct_accuracy"].max().eq(1).sum()) / (df_metrics_reranker.groupby("query")["correct_list"].max().eq(1).sum())
    overall_accuary_parents = (df_metrics_reranker.groupby("query")["correct_candidate"].max().mean()) # also (recall * sel_accuracy)
    recall_parents = (df_metrics_reranker.groupby("query")["correct_recall"].max().mean())
    if children==True:
            df_metrics_reranker.loc[(df_metrics_reranker["correct_accuracy"] == 0) & df_metrics_reranker.apply(lambda row: (row["candidate"] in row["children_id"]) & (row[ID] in row["candidate_list"]),axis=1),"correct_accuracy"] = 1
            df_metrics_reranker.loc[(df_metrics_reranker["correct_candidate"] == 0) & df_metrics_reranker.apply(lambda row: row["candidate"] in row["children_id"],axis=1),"correct_candidate"] = 1
            df_metrics_reranker.loc[(df_metrics_reranker["correct_recall"] == 0) & df_metrics_reranker.apply(lambda row: any(parent in row["candidate_list"] for parent in ast.literal_eval(row["children_id"])), axis=1), "correct_recall"] = 1
    
    recall_parents_children = (df_metrics_reranker.groupby("query")["correct_recall"].max().mean())
    sel_accuracy_parents_children = (df_metrics_reranker.groupby("query")["correct_accuracy"].max().eq(1).sum()) / (df_metrics_reranker.groupby("query")["correct_list"].max().eq(1).sum())
    overall_accuary_parents_children = (df_metrics_reranker.groupby("query")["correct_candidate"].max().mean()) # also (recall * sel_accuracy)
                     
    
    results_list.append({
                     
                     "reranker": reranker,
                     "method": method,
                     "k":k,
                     "Recall": round(recall,3),
                     "Recall ascendants": round(recall_parents,3),
                     "Recall ascendants - descendants": round(recall_parents_children, 3),
                     "Reranker accuracy": round(sel_accuracy,3),
                     "Reranker accuracy ascendants": round(sel_accuracy_parents,3),
                     "Reranker accuracy ascendants - descendants": round(sel_accuracy_parents_children,3),
                     "Overall accuracy": round(overall_accuary,3),
                     "Overall accuracy ascendants": round(overall_accuary_parents,3),
                     "Overall accuracy ascendants - descendants": round(overall_accuary_parents_children,3),
                     "median_candidates": round(mean_candidates,3),
                     "% ties": round(ties,3)
    
                 })
         
    return pd.DataFrame(results_list)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluation normalization pipeline"
    )

    parser.add_argument("--positive_list_file", required=True)
    parser.add_argument("--exact_normalization", required=True)
    parser.add_argument("--results_noramlization", required=True)
    parser.add_argument("--ontology_owl_file", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--ID", required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--method", choices=["combined", "sapbert"], default="combined")

    args = parser.parse_args()
    exact_normalization=args.exact_normalization
    positive_list_file=args.positive_list_file
    entity = args.entity
    results_noramlization = args.results_noramlization
    ID = args.ID
    k= args.k
    method=args.method
    
    parents=True
    children=True
    if ID == "CUI":
        ontology_parse = "UMLS"
    else:
        module = importlib.import_module(
            f"ontology_linker.parse_{ID.lower()}_ontology"
        )
        ontology_parse = module.parse_ontology_owl(args.ontology_owl_file)
        
    # 1. Exact matching evaluation
    exact_errors = exact_precision(positive_list_file, exact_normalization, ontology_parse, entity, parents, children, ID)

    # 2. Retrieval evaluation
    recall_metrics = recall_per_method(positive_list_file, results_noramlization, entity, ID)

    # 3. Plot retrieval results
    plot_recall(recall_metrics, entity)

    # 4. Reranking evaluation
    input_borda_file=Path(f"results/reranker/{ID.lower()}_{entity}_borda_reranker_{k}_{method}.tsv")
    input_rrf_file=Path(f"results/reranker/{ID.lower()}_{entity}_RRF_reranker_{k}_{method}.tsv")
    input_llm_file = Path(f"results/reranker/{ID.lower()}_{entity}_LLM_reranker_{k}_{method}.tsv")

    metrics_borda = reranker_metrics(input_borda_file, positive_list_file, parents, children, "Borda", method, k, ID)
    metrics_rrf = reranker_metrics(input_rrf_file, positive_list_file, parents, children, "RRF", method, k, ID)
    metrics_llm = reranker_metrics(input_llm_file, positive_list_file, parents, children, "LLM", method, k, ID)

    # Combine and save metrics
    metrics = pd.concat(
        [metrics_borda, metrics_rrf, metrics_llm],
        ignore_index=True
    )
    plot_accuracy(metrics, entity)
    
    output_metrics_file=Path(f"results/metrics_{entity}_{ID.lower()}_{k}_{method}.tsv")
    metrics.to_csv(output_metrics_file,
                   sep="\t", index=False)


if __name__ == "__main__":
    main()                      