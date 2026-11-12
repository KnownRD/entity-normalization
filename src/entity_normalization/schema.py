from typing import Optional
from pydantic import BaseModel


class OntologyConcept(BaseModel):
    concept_id: str
    label: str
    exact_synonyms: list[str] = []
    narrow_synonyms: list[str] = []
    broad_synonyms: list[str] = []
    related_synonyms: list[str] = []
    definition: Optional[str] = None
    parents: list[str] = []
    children: list[str] = []
    xrefs: list[str] = []
    namespace: str
    relationship: list[str] = []
    obsolete: bool = False
    replacement_id: Optional[str] = None


class Candidate(BaseModel):
    concept_id: str
    #label: str
    score: float
    # sources: list[str] = []
    #scopes: list[str] = []
    #matched_texts: list[str] = []


class LinkResult(BaseModel):
    query: str
    normalized_query: str
    concept_id: Optional[str]
    label: Optional[str]
    confidence: float
    decision: str
    candidates: list[Candidate]


class LinkResult_lexical(BaseModel):
    query: str
    normalized_query: str
    concept_id: Optional[str]
    label: Optional[str]
    score: float
    candidates: list[Candidate]
