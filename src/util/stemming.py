import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess_text(text: str) -> str:
    words = nltk.word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    
    processed_words = [
        stemmer.stem(w) for w in words 
        if w not in stop_words and w.isalnum()
    ]
    
    return " ".join(processed_words)