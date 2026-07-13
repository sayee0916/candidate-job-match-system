# 🤖 NLP-Based Candidate Job Match System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![License](https://img.shields.io/badge/License-MIT-blue)

An NLP-based application that compares a **Job Description (JD)** with a **Candidate Resume** to estimate the candidate's suitability for the role.

The system uses **TF-IDF Vectorization**, **Cosine Similarity**, and **Skill Matching** to generate a match score, identify matched and missing skills, and provide a hiring recommendation.

---

## 🚀 Live Demo

👉 **Open the application here:**

[🚀 Launch Live Demo](https://candidate-job-match-system.streamlit.app)

## 📂 GitHub Repository

https://github.com/sayee0916/candidate-job-match-system

---

# 🤖 NLP-Based Candidate Job Match System

Python | Streamlit | NLP | TF-IDF

An NLP-based application that compares a Job Description (JD) with a Candidate Resume to estimate the candidate's suitability for the role.

---

## 🚀 Live Demo

👉 https://candidate-job-match-system.streamlit.app

📂 GitHub Repository

https://github.com/sayee0916/candidate-job-match-system

---

## 📊 Example Output

Tested using my final resume against the provided AI/ML Intern Job Description.

**Match Score:** **48.79%**

**Matched Skills:**
- Data Analysis
- Data Preprocessing
- Machine Learning
- NumPy
- Pandas
- Python
- Scikit-Learn

**Missing Skills:**
- Artificial Intelligence
- Feature Engineering
- PyTorch
- TensorFlow

**Recommendation:** 🟡 Moderate Match

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
## ▶️ Run Locally

1. Clone the repository

```bash
git clone https://github.com/sayee0916/candidate-job-match-system.git
```

2. Navigate to the project folder

```bash
cd candidate-job-match-system
```

3. Install the required packages

```bash
pip install -r requirements.txt
```

4. Launch the application

```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

- BERT/Sentence Transformers for semantic matching
- Multiple resume ranking
- Dynamic skill extraction
- Experience & education-based scoring

---

## 👩‍💻 Author

**Sayali Sanjay Chidrawar**
