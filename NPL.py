# 🚀 Python Text Analyzer: Unleashing Advanced NLP Magic! 🚀

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

class PythonTextAnalyzer:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, I'm {self.name}, your Python NLP Maestro! 🐍✨"

    def analyze_text(self, text):
        # Tokenize the text
        words = word_tokenize(text.lower())

        # Remove stop words and punctuation
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

        # Perform word frequency analysis
        word_freq = Counter(filtered_words)
        top_words = word_freq.most_common(5)

        return f"Text Analysis by {self.name}:\nTop 5 Words: {top_words}"

# Let the Python NLP magic begin!
nlp_maestro = PythonTextAnalyzer("Your Name")
greeting = nlp_maestro.greet()

# Example text for analysis
sample_text = "Unlocking the power of Python for limitless possibilities. Python is truly a versatile language."

# Analyzing the text
analysis_result = nlp_maestro.analyze_text(sample_text)

# Displaying the Python NLP prowess
print(greeting)
print(analysis_result)
