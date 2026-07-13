# 🤖 NLP-Based Candidate Job Match System

An NLP-based application that compares a **Job Description (JD)** with a **Candidate Resume** to estimate the candidate's suitability for the role.

The system uses **TF-IDF Vectorization**, **Cosine Similarity**, and **Skill Matching** to generate a match score, identify matched and missing skills, and provide a hiring recommendation.

---

## 🚀 Features

- Upload Job Description (PDF)
- Upload Candidate Resume (PDF)
- PDF Text Extraction
- Text Preprocessing
- TF-IDF Similarity
- Skill Matching
- Match Score (0–100%)
- Matched & Missing Skills
- Hiring Recommendation
- Interactive Streamlit Dashboard

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- Cosine Similarity
- pdfplumber

---

## 📂 Project Structure

```
Candidate-Job-Match-System/
│── app.py
│── utils.py
│── skills.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Workflow

```
JD PDF + Resume PDF
        ↓
Text Extraction
        ↓
Text Cleaning
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Skill Matching
        ↓
Final Match Score
        ↓
Recommendation
```

---

## 📊 Scoring Formula

```
Final Score = (30% × TF-IDF Similarity) + (70% × Skill Match)
```


## 🔮 Future Improvements

- BERT/Sentence Transformers for semantic matching
- Multiple resume ranking
- Dynamic skill extraction
- Experience & education-based scoring

---

## 👩‍💻 Author

**Sayali Sanjay Chidrawar**
