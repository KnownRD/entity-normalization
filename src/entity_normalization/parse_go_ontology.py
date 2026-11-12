from __future__ import annotations
from pathlib import Path
import pandas as pd
import pronto
from entity_normalization.schema import OntologyConcept


def _normalize_concept_id(term_id: str) -> str | None:
    if term_id.startswith("GO:"):
        return term_id
    if "GO_" in term_id:
        return term_id.replace("_", ":")
    return None


def _extract_synonyms(term) -> dict[str, list[str]]:
    out = {
        "exact": [],
        "narrow": [],
        "broad": [],
        "related": [],
    }

    for syn in term.synonyms:
        scope = syn.scope.lower() if syn.scope else "related"
        val = syn.description
        if not val:
            continue

        if scope == "exact":
            out["exact"].append(val)
        elif scope == "narrow":
            out["narrow"].append(val)
        elif scope == "broad":
            out["broad"].append(val)
        else:
            out["related"].append(val)

    return out


def _extract_xrefs(term) -> list[str]:
    xrefs = []
    if hasattr(term, "xrefs"):
        for x in term.xrefs:
            xrefs.append(str(x))
    return sorted(set(xrefs))

def parse_ontology_owl(owl_path: str | Path) -> pd.DataFrame:
    ontology = pronto.Ontology(str(owl_path))
    rows = []

    for term in ontology.terms():
        if term.obsolete:
            continue

        concept_id = _normalize_concept_id(term.id)
        if not concept_id:
            continue

        syns = _extract_synonyms(term)

        parents = [
            parent.name
            for parent in term.superclasses(with_self=False)
            if parent.name
        ]
        children = [
            children.name
            for children in term.subclasses(with_self=False)
            if children.name
        ]
        relationship=[]
        for rel, targets in term.relationships.items():
            if rel.name == "part of":
                for target in targets:
                    relationship.append(target.id)

        rows.append(
            OntologyConcept(
                concept_id=concept_id,
                label=term.name or "",
                exact_synonyms=sorted(set(syns["exact"])),
                narrow_synonyms=sorted(set(syns["narrow"])),
                broad_synonyms=sorted(set(syns["broad"])),
                related_synonyms=sorted(set(syns["related"])),
                definition=str(term.definition) if term.definition else None,
                parents=parents,
                children=children,
                xrefs=_extract_xrefs(term),
                namespace=term.namespace or "",
                relationship=relationship,
                obsolete=False,
                replacement_id=None,
            ).model_dump()
        )

    return pd.DataFrame(rows)
