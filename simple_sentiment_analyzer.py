import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re

class SimpleSentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def simple_preprocess(self, text):
        """Simple text preprocessing without TextBlob"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove URLs, mentions, special chars
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def train_model(self, texts, labels):
        """Train the model"""
        processed_texts = [self.simple_preprocess(text) for text in texts]
        X = self.vectorizer.fit_transform(processed_texts)
        self.model.fit(X, labels)
        
    def predict_with_confidence(self, texts):
        """Predict with confidence"""
        processed_texts = [self.simple_preprocess(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidence = np.max(probabilities, axis=1)
        return predictions, confidence

def create_simple_dataset():
    """Simple dataset without complex processing"""
    texts = [
        "I love this product amazing quality",
        "Great service highly recommended",
        "Excellent performance very happy",
        "Perfect solution works great",
        "Outstanding results fantastic",
        "Terrible product waste money",
        "Worst service very disappointed", 
        "Poor quality broken",
        "Horrible experience avoid",
        "Awful terrible disaster",
        "Product okay nothing special",
        "Average quality decent price",
        "Fine works normal",
        "Standard typical product",
        "Acceptable meets requirements"
    ]
    
    labels = ['positive'] * 5 + ['negative'] * 5 + ['neutral'] * 5
    return pd.DataFrame({'text': texts, 'sentiment': labels})