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
from spacy.language import Language
from spacy.tokens import Doc

# from gensim import corpora
from nltk.corpus import stopwords

# from nltk.stem import WordNetLemmatizer
nltk.download(["stopwords", "wordnet"])
from spacy.matcher import Matcher
from pdfminer.high_level import extract_text

warnings.filterwarnings("ignore")


# Load a spaCy model (or create a blank one)
nlp = spacy.load("en_core_web_sm")  # or nlp = spacy.blank("en")


def extractTextPDF(path):
    return ''.join([i for i in extract_text(path) if i != '●' and i != '○'])


def extractName(text):
    matcher = Matcher(nlp.vocab)

    patterns = [
        [{"POS": "PROPN"}, {"POS": "PROPN"}],  # First name and Last name
        [
            {"POS": "PROPN"},
            {"POS": "PROPN"},
            {"POS": "PROPN"},
        ],  # First name, Middle name, and Last name
        [{"POS" : "PROPN"}]  # first name
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
    "Bachelor of Engineering",
    "B.E.",
    "B.E",
    "BS",
    "Bachelor of Science",
    "ME",
    "Master of Engineering",
    "M.E.",
    "Master of Science",
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
                edu[tex] = text + nlp_text[index]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r"(((20|19)(\d{2})))"), edu[key])
        if year:
            education.append((key, "".join(year[0])))
        else:
            education.append(key)
    return education


def extractSkills_empty(text):
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


def remove_leading_whitespace(text):
    return "\n".join(line.lstrip() for line in text.splitlines())

def extractSkills(resume_text):
    # Load the spaCy model
    resume_text = remove_leading_whitespace(resume_text)
    nlp = spacy.load("en_core_web_sm",  exclude=["parser"])
    config = {"punct_chars": ['\n']}
    nlp.add_pipe("sentencizer", config=config)


    # Define keywords related to project sections
    project_keywords = [
        "\nskills",
        "\ntechnical skills",
        "\nother skills",
        "\nkey skills -"
    ]
    end_section_keywords = [
        "projects",
        "\neducation",
        "\nprofessional summary",
        "\nsummary"
        "\ncertifications",
        "\nawards",
        "\norganizations",
    ]

    # Process the resume text with spaCy
    doc = nlp(resume_text)
    # Flag to track if we're in the projects section
    in_work_section = False
    works = []
    current_section = []
    for sent in doc.sents:
        # Check if this sentence marks the beginning of the projects section
        if any(keyword in sent.text.lower() for keyword in project_keywords):
            in_work_section = True
            current_section.append(sent.text.strip())
            continue
        # If we're in the projects section, check if we reach the end of the section
        if in_work_section and any(
            keyword in sent.text.lower() for keyword in end_section_keywords
        ):
            break
        # If we're in the projects section, collect the sentences
        if in_work_section:
            current_section.append(sent.text.strip())
    # Join collected sentences to form the project description
    work_desc = "\n".join(current_section).strip()
    # Add project description to the projects list if not empty
    if work_desc:
        works.append(work_desc)
    if not works:
        return extractSkills_empty(resume_text)
    return ''.join(works)


def extract_projects(resume_text):
    # Load the spaCy model
    resume_text = remove_leading_whitespace(resume_text)
    nlp = spacy.load("en_core_web_sm",  exclude=["parser"])
    config = {"punct_chars": ['\n']}
    nlp.add_pipe("sentencizer", config=config)


    # Define keywords related to project sections
    project_keywords = [
        "\nprojects",
        "\nrelevant projects",
        "\nacademic projects",
        "\nwork projects",
        "\nside projects",
    ]
    end_section_keywords = [
        "\nexperience",
        "\nwork experience",
        "\neducation",
        "\nskills",
        "\ncertifications",
        "\nawards",
        "\norganizations",
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

def extractSummary(resume_text):
    # Load the spaCy model
    resume_text = remove_leading_whitespace(resume_text)
    nlp = spacy.load("en_core_web_sm", exclude= ["parser"])
    config = {"punct_chars": ['\n']}
    nlp.add_pipe("sentencizer", config=config)

    # Define keywords related to project sections
    project_keywords = [
        "\nsummary",
        "\nprofessional summary",
        "\npersonal summary",
        "\nabout me",
    ]
    end_section_keywords = [
        "\nprojects",
        "\neducation",
        "\nskills",
        "\ncertifications",
        "\nawards",
        "\norganizations",
        "\nwork experience"
    ]

    # Process the resume text with spaCy
    doc = nlp(resume_text)
    # Flag to track if we're in the projects section
    in_work_section = False
    works = []
    current_section = []
    for sent in doc.sents:
        # Check if this sentence marks the beginning of the projects section
        if any(keyword in sent.text.lower() for keyword in project_keywords):
            in_work_section = True
            continue
        # If we're in the projects section, check if we reach the end of the section
        if in_work_section and any(
            keyword in sent.text.lower() for keyword in end_section_keywords
        ):
            break
        # If we're in the projects section, collect the sentences
        if in_work_section:
            current_section.append(sent.text.strip())
    # Join collected sentences to form the project description
    work_desc = "\n".join(current_section).strip()
    # Add project description to the projects list if not empty
    if work_desc:
        works.append(work_desc)
    return ''.join(works)


def extract_workex(resume_text):
    # Load the spaCy model
    resume_text = remove_leading_whitespace(resume_text)
    nlp = spacy.load('en_core_web_sm', exclude=["parser"])
    config = {"punct_chars": ['\n']}
    nlp.add_pipe("sentencizer", config=config)

    # Define keywords related to project sections
    project_keywords = [
        "\nwork experience",
        "\nexperience",
        "\nprofessional experience",
        "\nwork history",
    ]
    end_section_keywords = [
        "\nprojects",
        "\neducation",
        "\nskills",
        "\ncertifications",
        "\nawards",
        "\norganizations",
    ]

    # Process the resume text with spaCy
    doc = nlp(resume_text)
    # Flag to track if we're in the projects section
    in_work_section = False
    works = []
    current_section = []
    for sent in doc.sents:
        # Check if this sentence marks the beginning of the projects section
        if any(keyword in sent.text.lower() for keyword in project_keywords):
            in_work_section = True
            continue
        # If we're in the projects section, check if we reach the end of the section
        if in_work_section and any(
            keyword in sent.text.lower() for keyword in end_section_keywords
        ):
            break
        # If we're in the projects section, collect the sentences
        if in_work_section:
            current_section.append(sent.text.strip())
    # Join collected sentences to form the project description
    work_desc = "\n".join(current_section).strip()
    # Add project description to the projects list if not empty
    if work_desc:
        works.append(work_desc)
    return works


def main():
    text = extractTextPDF("/Users/swayam/Downloads/resume-odoo.pdf")
    print(text)
    print(extractName(text))
    # print(extractContact(text))
    # print(extractEmail(text))
    print(extractSummary(text))
    print(extractSkills(text))
    # print(extractEducation(text))
    projects = extract_projects(text)
    for project in projects:
       print(project)
    workex = extract_workex(text)
    for work in workex:
        print(work)


if __name__ == "__main__":
    main()
