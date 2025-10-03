"""
Simple Sentiment Analysis Demo for Interview
No NLTK dependencies - works immediately
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re

class SimpleSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = LogisticRegression()
        self.is_trained = False
    
    def clean_text(self, text):
        """Simple text cleaning"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    def train(self, texts, labels):
        """Train the model"""
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizer.fit_transform(cleaned_texts)
        
        # Train model
        self.model.fit(X, labels)
        self.is_trained = True
        
        print("✅ Model trained successfully!")
        return self
    
    def predict(self, texts):
        """Predict sentiment"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Clean and vectorize
        cleaned_texts = [self.clean_text(text) for text in texts]
        X = self.vectorizer.transform(cleaned_texts)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities

def create_demo_data():
    """Create sample data for demo"""
    data = {
        'text': [
            "I love this product, it's amazing!",
            "This is terrible, worst experience ever",
            "Great service, highly recommended",
            "Poor quality, not worth the money",
            "Excellent work, very satisfied",
            "Bad customer service, disappointed",
            "Outstanding performance, exceeded expectations",
            "Waste of time and money",
            "Perfect solution for my needs",
            "Horrible experience, never again"
        ],
        'sentiment': [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative'
        ]
    }
    return pd.DataFrame(data)

def main():
    print("🤖 Simple Sentiment Analysis Demo")
    print("=" * 50)
    
    # Create demo data
    df = create_demo_data()
    print(f"📊 Training data: {len(df)} samples")
    
    # Initialize and train model
    analyzer = SimpleSentimentAnalyzer()
    analyzer.train(df['text'], df['sentiment'])
    
    # Test predictions
    test_texts = [
        "This is fantastic!",
        "I hate this product",
        "It's okay, nothing special",
        "Absolutely wonderful experience"
    ]
    
    print("\n🔍 Testing predictions:")
    predictions, probabilities = analyzer.predict(test_texts)
    
    for i, text in enumerate(test_texts):
        pred = predictions[i]
        confidence = max(probabilities[i]) * 100
        print(f"Text: '{text}'")
        print(f"Prediction: {pred} (Confidence: {confidence:.1f}%)")
        print("-" * 30)
    
    print("\n✅ Demo completed successfully!")
    print("Ready for interview! 🚀")

if __name__ == "__main__":
    main()