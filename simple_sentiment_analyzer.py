"""
Simple Sentiment Analyzer - Fallback version without TextBlob dependencies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
import joblib
from datetime import datetime

class SimpleSentimentAnalyzer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': MultinomialNB()
        }
        self.vectorizers = {
            'tfidf': TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2)),
            'count': CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
        }
        self.best_model = None
        self.best_vectorizer = None
        self.model_performance = {}
        
    def simple_preprocess(self, text):
        """Simple text preprocessing without TextBlob"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def train_and_compare_models(self, texts, labels):
        """Train multiple models and compare performance"""
        processed_texts = [self.simple_preprocess(text) for text in texts]
        
        best_score = 0
        results = []
        
        for vec_name, vectorizer in self.vectorizers.items():
            X = vectorizer.fit_transform(processed_texts)
            
            for model_name, model in self.models.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X, labels, cv=5, scoring='accuracy')
                mean_score = cv_scores.mean()
                
                results.append({
                    'vectorizer': vec_name,
                    'model': model_name,
                    'cv_score': mean_score,
                    'cv_std': cv_scores.std()
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_model = model
                    self.best_vectorizer = vectorizer
                    
        self.model_performance = pd.DataFrame(results)
        
        # Train best model on full dataset
        X_best = self.best_vectorizer.fit_transform(processed_texts)
        self.best_model.fit(X_best, labels)
        
        return self.model_performance
    
    def predict_with_confidence(self, texts):
        """Predict with confidence scores"""
        processed_texts = [self.simple_preprocess(text) for text in texts]
        X = self.best_vectorizer.transform(processed_texts)
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)
        confidence = np.max(probabilities, axis=1)
        
        return predictions, confidence

def create_simple_dataset():
    """Create a simple dataset"""
    texts = [
        # Positive samples
        "I love this product! Great quality and fast delivery.",
        "Amazing service and excellent customer support.",
        "Best purchase ever! Highly recommended.",
        "Outstanding quality and great value for money.",
        "Perfect solution for my needs. Very satisfied!",
        
        # Negative samples
        "Terrible product, complete waste of money.",
        "Worst service ever. Very disappointed.",
        "Poor quality and doesn't work properly.",
        "Overpriced and low quality. Avoid this.",
        "Horrible experience, would not recommend.",
        
        # Neutral samples
        "The product is okay, nothing special.",
        "Average quality, meets basic requirements.",
        "It's fine for the price, decent quality.",
        "Standard product with typical features.",
        "Acceptable performance, could be better."
    ]
    
    labels = ['positive'] * 5 + ['negative'] * 5 + ['neutral'] * 5
    
    return pd.DataFrame({'text': texts, 'sentiment': labels})