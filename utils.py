# utils.py

import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skills import SKILLS


# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# -----------------------------
# Calculate TF-IDF Similarity
# -----------------------------
def calculate_similarity(jd_text, resume_text):

    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform([jd_text, resume_text])

    cosine = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    return round(cosine * 100,2)

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
    

# -----------------------------
# Extract skills
# -----------------------------

def extract_skills(text):

    text = normalize(text)

    aliases = {
        "ai": "artificial intelligence",
        "artificial intelligence": "artificial intelligence",
        "ml": "machine learning",
        "machine learning": "machine learning"
    }

    found = set()

    for skill in SKILLS:

        skill_normalized = normalize(skill)

        if skill_normalized in text:

            canonical = aliases.get(skill.lower(), skill.lower())

            found.add(canonical.title())

    return found


# -----------------------------
# Get matched & missing skills
# -----------------------------
def compare_skills(jd_text, resume_text):

    jd_skills = extract_skills(jd_text)

    resume_skills = extract_skills(resume_text)

    matched = sorted(jd_skills & resume_skills)

    missing = sorted(jd_skills - resume_skills)

    skill_match_percentage = 0

    if len(jd_skills) > 0:
        skill_match_percentage = (len(matched) / len(jd_skills)) * 100

    return matched, missing, round(skill_match_percentage,2)


# -----------------------------
# Recommendation
# -----------------------------
def recommendation(score):
    if score >= 80:
        return "Strong Match"
    elif score >= 60:
        return "Good Match"
    elif score >= 40:
        return "Moderate Match"
    else:
        return "Low Match"
        

def final_score(jd_text, resume_text):

    tfidf_score = calculate_similarity(jd_text, resume_text)

    matched, missing, skill_score = compare_skills(jd_text, resume_text)

    final = (0.30 * tfidf_score) + (0.70 * skill_score)

    return (
        round(final, 2),
        round(tfidf_score, 2),
        round(skill_score, 2),
        matched,
        missing
    )
