import re
import unicodedata
import spacy

nlp = spacy.load("en_core_web_sm")
_ROMAN = {
    " i ": " 1 ",
    " ii ": " 2 ",
    " iii ": " 3 ",
    " iv ": " 4 ",
    " v ": " 5 ",
}

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\w\s:/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = f" {text} "
    for k, v in _ROMAN.items():
        text = text.replace(k, v)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def singularize(text):
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            tokens.append(token.lemma_)
        else:
            tokens.append(token.text)

    return " ".join(tokens)