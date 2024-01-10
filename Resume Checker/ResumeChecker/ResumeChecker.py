# Import necessary libraries
import os
from flask import Flask, render_template, request
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

#http://127.0.0.1:5000/ <- the localhost 

# Initialize Flask app
app = Flask(__name__)

# Load spaCy model for natural language processing
nlp = spacy.load("en_core_web_sm")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Sample job description
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

# Function for Advanced NLP Analysis (Placeholder, needs implementation)
def advanced_nlp_analysis(text):
    # Implement advanced NLP analysis using spaCy or other libraries
    # Include Named Entity Recognition (NER), sentiment analysis, etc.
    # Return insights based on the analysis
    return {"placeholder": "advanced analysis results"}

# Function to analyze how well the resume fits a job description using advanced NLP
def analyze_fit_advanced(job_description, resume_text, user_preferences):
    overall_similarity = compare_text_bert(job_description, resume_text)

    # Advanced NLP analysis on job description and resume
    job_analysis = advanced_nlp_analysis(job_description)
    resume_analysis = advanced_nlp_analysis(resume_text)

    # Additional analysis based on user preferences (hypothetical function)
    user_requirements = analyze_user_preferences(user_preferences)

    # Additional analysis based on user requirements
    additional_analysis = analyze_additional_requirements(job_analysis, resume_analysis, user_requirements)

    return overall_similarity, additional_analysis

# Function to analyze user preferences (Hypothetical, needs implementation)
def analyze_user_preferences(user_preferences):
    # Implement logic to analyze and extract user preferences
    return {"user_preferences": user_preferences}

# Function for additional analysis based on user requirements (Hypothetical, needs implementation)
def analyze_additional_requirements(job_analysis, resume_analysis, user_requirements):
    # Implement logic for additional analysis based on user requirements
    return {"additional_analysis": "additional analysis results"}

# Function to provide detailed feedback and suggestions
def provide_feedback(job_tokens, resume_tokens):
    # Words not found in the resume
    missing_words = [token.text for token in job_tokens if token.text.lower() not in [r_token.text.lower() for r_token in resume_tokens]]

    # Suggestions for improvement
    improvement_suggestions = ["Consider adding more details about your experience with Django, Flask, and RESTful API development.",
                                "Highlight any front-end technologies you have worked with, such as HTML, CSS, and JavaScript.",
                                "Ensure your resume reflects your problem-solving skills with concrete examples from your experience."]

    return missing_words, improvement_suggestions

# Flask route for the index page
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

            user_preferences = request.form.get("user_preferences")

            overall_similarity, additional_analysis = analyze_fit_advanced(job_description, resume_text, user_preferences)

            # Provide detailed feedback and suggestions
            job_tokens = nlp(job_description)
            resume_tokens = nlp(resume_text)
            missing_words, improvement_suggestions = provide_feedback(job_tokens, resume_tokens)

            return render_template("result.html", overall_similarity=overall_similarity, additional_analysis=additional_analysis,
                                   missing_words=missing_words, improvement_suggestions=improvement_suggestions)

    return render_template("index.html", error=None)

if __name__ == "__main__":
    app.run(debug=True)
