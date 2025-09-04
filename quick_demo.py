"""
Quick Interview Demo Script
Run this to show key features in 2-3 minutes
"""

from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer, create_enhanced_dataset
from business_sentiment_analyzer import BusinessSentimentAnalyzer

def quick_demo():
    print("SENTIMENT ANALYSIS PROJECT DEMO")
    print("=" * 40)
    
    # 1. Model Training Demo
    print("\n1. TRAINING MULTIPLE MODELS...")
    analyzer = AdvancedSentimentAnalyzer()
    df = create_enhanced_dataset()
    performance = analyzer.train_and_compare_models(df['text'], df['sentiment'])
    
    print("Model Performance Comparison:")
    print(performance.sort_values('cv_score', ascending=False)[['model', 'vectorizer', 'cv_score']].head(3))
    
    best = performance.loc[performance['cv_score'].idxmax()]
    print(f"\nBest Model: {best['model']} + {best['vectorizer']} = {best['cv_score']:.3f}")
    
    # 2. Prediction Demo
    print("\n2. PREDICTION WITH CONFIDENCE...")
    test_texts = [
        "This product is absolutely amazing!",
        "Terrible quality, waste of money.",
        "It's okay, nothing special."
    ]
    
    predictions, confidence = analyzer.predict_with_confidence(test_texts)
    
    for i, text in enumerate(test_texts):
        print(f"'{text[:30]}...' -> {predictions[i]} ({confidence[i]:.1%} confidence)")
    
    # 3. Business Intelligence Demo
    print("\n3. BUSINESS INTELLIGENCE...")
    business_analyzer = BusinessSentimentAnalyzer()
    business_analyzer.best_model = analyzer.best_model
    business_analyzer.best_vectorizer = analyzer.best_vectorizer
    
    business_reviews = [
        "Great product quality but delivery was slow and expensive.",
        "Excellent customer service and fast shipping!",
        "Poor quality product, overpriced for what you get."
    ]
    
    insights = business_analyzer.get_business_insights(business_reviews)
    
    print("Business Recommendations:")
    for rec in insights['recommendations']:
        print(f"  • {rec}")
    
    print("\nAspect Analysis:")
    for aspect, data in insights['aspect_analysis'].items():
        if data['total_mentions'] > 0:
            pos_pct = data['positive'] / data['total_mentions'] * 100
            print(f"  {aspect}: {pos_pct:.0f}% positive")
    
    print("\n4. KEY FEATURES DEMONSTRATED:")
    print("✓ Multi-model comparison and auto-selection")
    print("✓ Confidence scoring for predictions")
    print("✓ Business intelligence and recommendations")
    print("✓ Aspect-based sentiment analysis")
    print("✓ Production-ready architecture")

if __name__ == "__main__":
    quick_demo()