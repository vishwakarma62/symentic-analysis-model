import pandas as pd
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def create_larger_dataset():
    """Create larger training dataset for better model performance"""
    texts = [
        # Strong positive
        "Absolutely amazing product! Best purchase ever made.",
        "Outstanding quality and incredible value for money.",
        "Perfect in every way, exceeded all expectations.",
        "Brilliant service and fantastic results.",
        "Exceptional experience, highly recommend to everyone.",
        
        # Moderate positive  
        "Good product, works as expected.",
        "Nice quality for the price.",
        "Satisfied with the purchase.",
        "Decent service and reasonable quality.",
        "Pretty good overall experience.",
        
        # Strong negative
        "Terrible product, complete waste of money.",
        "Worst experience ever, avoid at all costs.",
        "Horrible quality and awful customer service.",
        "Completely broken and useless.",
        "Fraudulent company with terrible products.",
        
        # Moderate negative
        "Not satisfied with the quality.",
        "Poor value for money.",
        "Disappointing results.",
        "Below average performance.",
        "Could be much better.",
        
        # Neutral
        "Average product, nothing special.",
        "It's okay, meets basic requirements.",
        "Standard quality as expected.",
        "Neither good nor bad.",
        "Acceptable for the price."
    ]
    
    labels = (['positive'] * 10 + ['negative'] * 10 + ['neutral'] * 5)
    return pd.DataFrame({'text': texts, 'sentiment': labels})

def train_and_evaluate():
    """Train model and show detailed evaluation"""
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("=" * 40)
    
    # Create dataset
    df = create_larger_dataset()
    print(f"Dataset: {len(df)} samples")
    print(f"Distribution: {df['sentiment'].value_counts().to_dict()}")
    
    # Train model
    analyzer = AdvancedSentimentAnalyzer()
    performance = analyzer.train_and_compare_models(df['text'], df['sentiment'])
    
    print("\nMODEL COMPARISON:")
    print(performance.sort_values('cv_score', ascending=False))
    
    # Best model details
    best = performance.loc[performance['cv_score'].idxmax()]
    print(f"\nBEST MODEL:")
    print(f"Algorithm: {best['model']}")
    print(f"Vectorizer: {best['vectorizer']}")
    print(f"Accuracy: {best['cv_score']:.3f} (+/- {best['cv_std']:.3f})")
    
    # Test predictions
    test_texts = [
        "This is absolutely fantastic!",
        "Terrible quality and service.",
        "It's okay, nothing special."
    ]
    
    print("\nTEST PREDICTIONS:")
    predictions, confidence = analyzer.predict_with_confidence(test_texts)
    
    for i, text in enumerate(test_texts):
        print(f"Text: '{text}'")
        print(f"Prediction: {predictions[i]} (confidence: {confidence[i]:.1%})")
        print()
    
    # Save model
    analyzer.save_model('production_sentiment_model.pkl')
    print("Model saved to 'production_sentiment_model.pkl'")
    
    return analyzer, performance

if __name__ == "__main__":
    train_and_evaluate()