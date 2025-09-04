"""
Simple Model Test
================
"""

def test_advanced_model():
    try:
        from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer, create_enhanced_dataset
        
        analyzer = AdvancedSentimentAnalyzer()
        df = create_enhanced_dataset()
        analyzer.train_and_compare_models(df['text'], df['sentiment'])
        
        predictions, confidence = analyzer.predict_with_confidence(["I love this!"])
        
        print("ADVANCED MODEL: Working")
        print(f"Prediction: {predictions[0]} (confidence: {confidence[0]:.3f})")
        return True
        
    except Exception as e:
        print(f"ADVANCED MODEL: Failed - {str(e)}")
        return False

def test_basic_model():
    try:
        from sentiment_analyzer import SentimentAnalyzer, create_sample_data
        
        analyzer = SentimentAnalyzer()
        df = create_sample_data()
        analyzer.train_model(df['text'], df['sentiment'])
        
        prediction = analyzer.predict(["I love this!"])
        
        print("BASIC MODEL: Working")
        print(f"Prediction: {prediction[0]}")
        return True
        
    except Exception as e:
        print(f"BASIC MODEL: Failed - {str(e)}")
        return False

def main():
    print("MODEL COMPATIBILITY TEST")
    print("=" * 25)
    
    advanced_ok = test_advanced_model()
    print()
    basic_ok = test_basic_model()
    
    print("\n" + "=" * 25)
    print("RESULTS:")
    
    if advanced_ok:
        print("SUCCESS: Advanced model is working")
        print("Your Streamlit app should work at: http://localhost:8501")
    elif basic_ok:
        print("FALLBACK: Only basic model works")
        print("Switch to basic model in streamlit_app.py")
    else:
        print("ERROR: Both models failed")
        print("Check your dependencies")

if __name__ == "__main__":
    main()