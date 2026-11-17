#==========================================================================================================================

# Normalization of llms entities to different ontologies
# Each term it is processed and normalized by an exact-matching method, 3 lexical and 1 semantic:
    # Exact-matching label and synonym terms
    # BM25
    # Fuzzy
    # N-gram
    # SapBERT
# Two files are generated, one with all exact-matching candidates and the other with all the retrieval method candidates

#==========================================================================================================================

import pandas as pd
from tqdm import tqdm
import os
import argparse
from pathlib import Path
from entity_normalization.retrieve_candidates import LexicalRetriever



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--terms_file", required=True)
    parser.add_argument("--output_file_name", required=True)
    parser.add_argument("--output_exact_name", required=True)
    parser.add_argument("--lexical_jsonl_file", required=True)
    parser.add_argument("--embeddings_path", required=True)

    args = parser.parse_args()

    llm_terms_file = args.terms_file
    output_file = Path(f"results/retrieval/{args.output_file_name}")
    output_exact = Path(f"results/retrieval/{args.output_exact_name}")
    lexical_jsonl_file= args.lexical_jsonl_file
    embeddings_path=args.embeddings_path

    Path("results/retrieval/").mkdir(parents=True, exist_ok=True)

    
    for f in [output_file, output_exact]: # if the output files exists rm them
        if os.path.exists(f):
            os.remove(f)
            
    print("Retrieving candidate concepts...")
    lexical = LexicalRetriever(lexical_jsonl_file, embeddings_path)
        
    entities_llms=pd.read_csv(llm_terms_file, sep="\t", header=None)
    entities_llms=entities_llms[0].to_list()
    for line in tqdm(entities_llms, desc="Processing terms", unit="line"):
        q = str(line).strip()
        res_df = lexical.search(q)
        if isinstance(res_df, pd.DataFrame):
            res_df["query"] = q
            res_df.to_csv(output_file, mode="a",sep="\t", header=not os.path.exists(output_file),index=False)

        else: # if the query is exact --> write output_exact
            res_df=pd.DataFrame(res_df)
            res_df["query"] = q
            res_df.to_csv(output_exact, mode="a",sep="\t", header=not os.path.exists(output_exact),index=False)

if __name__ == "__main__":
    main()
