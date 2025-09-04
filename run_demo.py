from sentiment_analyzer import SentimentAnalyzer, create_sample_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=" * 50)
    print("   AI/ML SENTIMENT ANALYSIS PROJECT")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    print(f"\nDataset created with {len(df)} samples")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['sentiment'], test_size=0.3, random_state=42
    )
    
    # Train model
    print("Training sentiment analysis model...")
    analyzer.train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, report = analyzer.evaluate_model(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)
    
    # Test with new examples
    print("\nTesting with new examples:")
    test_texts = [
        "This AI project is incredible!",
        "I hate waiting in long queues",
        "The weather is fine today"
    ]
    
    for text in test_texts:
        textblob_sentiment = analyzer.analyze_sentiment_textblob(text)
        model_sentiment = analyzer.predict([text])[0]
        print(f"\nText: '{text}'")
        print(f"  TextBlob: {textblob_sentiment}")
        print(f"  ML Model: {model_sentiment}")
    
    print("\n" + "=" * 50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    print("\nNext steps:")
    print("1. Run 'streamlit run streamlit_app.py' for web interface")
    print("2. Check README.md for full documentation")

if __name__ == "__main__":
    main()