import re
import spacy
import warnings

# import gensim
import nltk
import openai

# import pyLDAvis.gensim_models
import pandas as pd
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc

# from gensim import corpora
from nltk.corpus import stopwords

# from nltk.stem import WordNetLemmatizer
nltk.download(["stopwords", "wordnet"])
from spacy.matcher import Matcher
from pdfminer.high_level import extract_text

warnings.filterwarnings("ignore")


def extractTextPDF(path):
    return extract_text(path)


def extractName(text):
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)

    patterns = [
        [{"POS": "PROPN"}, {"POS": "PROPN"}],  # First name and Last name
        [
            {"POS": "PROPN"},
            {"POS": "PROPN"},
            {"POS": "PROPN"},
        ],  # First name, Middle name, and Last name
    ]

    for pattern in patterns:
        matcher.add("NAME", patterns=[pattern])

    doc = nlp(text)
    matches = matcher(doc)

    for match_id, start, end in matches:
        span = doc[start:end]
        return span.text

    return None


def extractContact(text):
    contact = None
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact = match.group()

    return contact


def extractEmail(text):
    email = None
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()

    return email


nlp = spacy.load("en_core_web_sm")
STOPWORDS = set(stopwords.words("english"))
EDUCATION = [
    "BE",
    "B.E.",
    "B.E",
    "BS",
    "B.S",
    "ME",
    "M.E",
    "M.E.",
    "MS",
    "M.S",
    "BTECH",
    "B.TECH",
    "M.TECH",
    "MTECH",
    "SSC",
    "HSC",
    "CBSE",
    "ICSE",
    "X",
    "XII",
]


def extractEducation(text):
    nlp_text = nlp(text)

    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r"[?|$|.|!|,]", r"", tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r"(((20|19)(\d{2})))"), edu[key])
        if year:
            education.append((key, "".join(year[0])))
        else:
            education.append(key)
    return education


def extractSkills(text):
    nlp_text = nlp(text)
    noun_chunks = nlp_text.noun_chunks

    tokens = [token.text for token in nlp_text if not token.is_stop]

    data = pd.read_csv("skills.csv")
    skills = list(data.columns.values)

    skillset = []

    # one-word
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # multi-word
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def main():
    text = extractTextPDF("/Users/swayam/Downloads/resume.pdf")
    print(extractName(text))
    print("\n")
    print(extractContact(text))
    print(extractEmail(text))
    print(extractSkills(text))
    print(extractEducation(text))


if __name__ == "__main__":
    main()
