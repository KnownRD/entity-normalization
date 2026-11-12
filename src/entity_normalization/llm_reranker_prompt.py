from openai import OpenAI
import json
import requests
from tqdm import tqdm # progress bar
import logging
import time
import pandas as pd

### IMPORTAT TO ADD YOUR API KEY AND BASE URL 

SYSTEM_PROMPT = """
You are a professional data annotator and curator.

Given a query and candidate IDs with their associated ontology terms,
select the candidate ID that most specifically and biologically correctly represents
the query. Prefer a candidate whose ontology term denotes the same biological entity
as the query.

You MUST provide an ANSWER from among the candidates

Return ONLY valid JSON:
{"ID": "candidate"}

Do not provide explanations.
"""


USER_TEMPLATE = """
Query:
{query}

Candidates:
{candidates}
"""

YOUR_API_KEY = ""
BASE_URL=""
# Create .log config file


def query_open_ai(model, system_prompt, user_prompt, api_key, base_url):
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            extra_body={
                "chat_template_kwargs": {
                "enable_thinking": False
                }},
            tool_choice="none"
        )
        logging.info("\nREQUEST SENT")
        logging.info(f"PROMPT: {user_prompt[:500]}")
        content = response.choices[0].message.content

        logging.info(f"PROMPT: {response.choices[0].message}")

        logging.info(f"RESPONSE ({time.time()-start:.2f}s): {repr(content[:500])}")

        return content

    except requests.exceptions.Timeout: 
        logging.error("TIMEOUT")
        return None

    except Exception as e:
        logging.error(f"ERROR: {e}")
        return None

def parse_terms(x):
    return [
        term.strip().strip("'\"")
        for term in x.strip("[]").split(",")
        if term.strip()
    ]
def build_candidates(data):
    data = data.copy()
    data["terms"] = data["terms"].apply(parse_terms)

    return "\n".join(
        f"{candidate}: {'; '.join(labels)}"
        for candidate, labels in data[["candidate", "terms"]].itertuples(index=False)
    )


def classify(df_data, output_file, log_file, failed_cases_file):
    logging.basicConfig(
        filename=log_file,
        filemode="w",  
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    llm_results_candidates=[]

    with open(failed_cases_file, "w", encoding="utf-8"):
        pass

    first_item = True

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("[\n")

        for query, data in tqdm(df_data.groupby("query"), desc="Processing candidates"):

            candidates = build_candidates(data)

            user_prompt = USER_TEMPLATE.format(
                query=query,
                candidates=candidates
            )

            raw = query_open_ai(
                "deepseek-ai/DeepSeek-V4-Flash",
                SYSTEM_PROMPT,
                user_prompt,
                YOUR_API_KEY,
                BASE_URL,
            )

            if not raw:
                logging.warning(f"EMPTY RESPONSE QUERY={query}")

                with open(failed_cases_file, "a", encoding="utf-8") as logf:
                    logf.write(f"{query}\tEMPTY_RESPONSE\n")

                continue

            try:
                parsed = json.loads(raw)

                cui = parsed.get("ID")

                if cui is None:
                    raise ValueError("Missing CUI")

            except Exception as e:

                logging.error(f"JSON ERROR: {e}")
                logging.error(f"RAW: {repr(raw)}")

                with open(failed_cases_file, "a", encoding="utf-8") as logf:
                    logf.write(f"{query}\tJSON_ERROR\n")

                continue

            result = {
                "query": query,
                "candidate": cui
            }
            llm_results_candidates.append(result)

            if not first_item:
                f.write(",\n")

            json.dump(result, f)

            f.flush()

            first_item = False

        f.write("\n]\n")

    print("Finished reranking.")
    return pd.DataFrame(llm_results_candidates)

