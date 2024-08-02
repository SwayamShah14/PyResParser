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

nlp = spacy.load("en_core_web_sm")


def extractTextPDF(path):
    return extract_text(path)


def extractName(text):
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


def extract_projects(resume_text):
    # Define keywords related to project sections
    project_keywords = [
        "projects",
        "relevant projects",
        "academic projects",
        "work projects",
        "side projects",
    ]
    end_section_keywords = [
        "experience",
        "work experience",
        "education",
        "skills",
        "certifications",
        "awards",
    ]

    # Process the resume text with spaCy
    doc = nlp(resume_text)

    # Flag to track if we're in the projects section
    in_projects_section = False
    projects = []
    current_section = []

    for sent in doc.sents:
        # Check if this sentence marks the beginning of the projects section
        if any(keyword in sent.text.lower() for keyword in project_keywords):
            in_projects_section = True
            continue

        # If we're in the projects section, check if we reach the end of the section
        if in_projects_section and any(
            keyword in sent.text.lower() for keyword in end_section_keywords
        ):
            break

        # If we're in the projects section, collect the sentences
        if in_projects_section:
            current_section.append(sent.text.strip())

    # Join collected sentences to form the project description
    project_description = "\n".join(current_section).strip()

    # Add project description to the projects list if not empty
    if project_description:
        projects.append(project_description)

    return projects


def extract_work_experience(text):
    # Apply spaCy NLP model to the text
    doc = nlp(text)

    # Define patterns to identify work experience sections
    experience_patterns = [
        r"(?i)\b(work experience|professional experience|employment history)\b"
    ]

    # Find the start of the work experience section
    for pattern in experience_patterns:
        match = re.search(pattern, text)
        if match:
            start_pos = match.end()
            break
    else:
        return "Work experience section not found"

    # Extract sentences related to work experience
    work_experience = []
    for sent in doc.sents:
        if sent.start_char >= start_pos:
            work_experience.append(sent.text)

    return "\n".join(work_experience)


def main():
    text = extractTextPDF("/Users/swayam/Downloads/ConsultingCV.pdf")
    print(extractName(text))
    print(extractContact(text))
    print(extractEmail(text))
    print(extractSkills(text))
    print(extractEducation(text))
    projects = extract_projects(text)
    # for i, project in enumerate(projects, 1):
    #     print(f"Project {i}:\n{project}\n")
    print(extract_work_experience(text))


if __name__ == "__main__":
    main()
