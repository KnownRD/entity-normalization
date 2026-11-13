from __future__ import annotations
from pathlib import Path
import pandas as pd
def parse_ontology_df(path: str | Path, sty = None)-> pd.DataFrame:
    columns_name=["CUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI","SAUI","SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"]
    umls_parse = pd.read_table(str(path),sep="|",index_col=False, names=columns_name)
    umls_parse = umls_parse[umls_parse["LAT"] == "ENG"]
    sty_file=pd.read_table(Path("data/raw/MRSTY.RRF"),sep="|", index_col=False, header=0, names=["CUI", "TUI", "STN", "STY", "ATUI", "CVF"])

    if sty == "cell":
        sty_cell_types=sty_file[sty_file["TUI"].str.contains("T025")]
        umls_parse = umls_parse.merge(sty_cell_types, on="CUI", how="inner")
    elif sty == "anatomical":
        sty_anatomical=sty_file[sty_file["TUI"].str.contains("T023|T024|T029|T030|T022", na=False) & ~sty_file["TUI"].str.contains("T026|T028|T021", na=False)]
        umls_parse = umls_parse.merge(sty_anatomical, on="CUI", how="inner")
    elif sty == "phenotype":
        sty_phenotype=sty_file[sty_file["TUI"].str.contains("T046|T048")]
        umls_parse = umls_parse.merge(sty_phenotype, on="CUI", how="inner")
    elif sty == "component":
        sty_phenotype=sty_file[sty_file["TUI"].str.contains("T026")]
        umls_parse = umls_parse.merge(sty_phenotype, on="CUI", how="inner")
    elif sty == "function":
        sty_phenotype=sty_file[sty_file["TUI"].str.contains("T044")]
        umls_parse = umls_parse.merge(sty_phenotype, on="CUI", how="inner")
    elif sty == "process":
        sty_phenotype=sty_file[sty_file["TUI"].str.contains("T043|T042")]
        umls_parse = umls_parse.merge(sty_phenotype, on="CUI", how="inner")
    umls_parse = umls_parse[["CUI", "STR", "TTY"]].drop_duplicates()

    return umls_parse