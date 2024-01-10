import os
from flask import Flask, render_template, request
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

#Go to this Link in web browser to use/upload your resume --> http://127.0.0.1:5000/

# Load spaCy model for natural language processing
nlp = spacy.load("en_core_web_sm")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Copy and paste job description
job_description = """
We are looking for a Python developer with experience in web development and strong problem-solving skills.
The ideal candidate should have knowledge of Django, Flask, and RESTful API development.
Experience with front-end technologies such as HTML, CSS, and JavaScript is a plus.
"""

# Function to compare text using BERT embeddings and cosine similarity
def compare_text_bert(text1, text2):
    inputs1 = tokenizer(text1, return_tensors="pt", max_length=512, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        logits1 = model(**inputs1).logits
        logits2 = model(**inputs2).logits

    similarity = cosine_similarity(logits1, logits2)
    return similarity[0][0]

# Function to analyze how well the resume fits a job description using BERT
def analyze_fit_bert(job_description, resume_text):
    overall_similarity = compare_text_bert(job_description, resume_text)

    job_tokens = nlp(job_description)
    resume_tokens = nlp(resume_text)
    job_info = " ".join([token.text for token in job_tokens.ents if token.label_ in ["ORG", "GPE", "PERSON", "PRODUCT"]])
    resume_info = " ".join([token.text for token in resume_tokens.ents if token.label_ in ["ORG", "GPE", "PERSON", "PRODUCT"]])

    info_similarity = compare_text_bert(job_info, resume_info)

    return overall_similarity, info_similarity

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if file:
            try:
                resume_text = file.read().decode("utf-8")
            except UnicodeDecodeError:
                resume_text = file.read().decode("latin-1", errors="replace")

            overall_similarity, info_similarity = analyze_fit_bert(job_description, resume_text)
            return render_template("result.html", overall_similarity=overall_similarity, info_similarity=info_similarity)

    return render_template("index.html", error=None)

if __name__ == "__main__":
    app.run(debug=True)
