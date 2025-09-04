import pandas as pd
import numpy as np
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer, create_enhanced_dataset
import matplotlib.pyplot as plt
import seaborn as sns

def business_intelligence_demo():
    print("🏢 BUSINESS INTELLIGENCE DEMO")
    print("=" * 50)
    
    # Initialize and train analyzer
    analyzer = AdvancedSentimentAnalyzer()
    df = create_enhanced_dataset()
    analyzer.train_and_compare_models(df['text'], df['sentiment'])
    
    # 1. CUSTOMER FEEDBACK ANALYSIS
    print("\n📊 1. CUSTOMER FEEDBACK ANALYSIS")
    print("-" * 30)
    
    customer_reviews = [
        "Amazing product! Best purchase ever!",
        "Terrible quality, broke after 2 days",
        "Good value for money, satisfied",
        "Outstanding customer service!",
        "Overpriced and poor performance",
        "Love the new features, very intuitive",
        "Shipping was delayed, disappointed",
        "Excellent build quality, recommended!",
        "Average product, nothing special",
        "Fantastic experience, will buy again!"
    ]
    
    predictions, confidence = analyzer.predict_with_confidence(customer_reviews)
    
    # Create business insights
    feedback_df = pd.DataFrame({
        'review': customer_reviews,
        'sentiment': predictions,
        'confidence': confidence
    })
    
    # Business metrics
    satisfaction_rate = (feedback_df['sentiment'] == 'positive').mean() * 100
    dissatisfaction_rate = (feedback_df['sentiment'] == 'negative').mean() * 100
    high_confidence_predictions = (feedback_df['confidence'] > 0.8).sum()
    
    print(f"📈 Customer Satisfaction Rate: {satisfaction_rate:.1f}%")
    print(f"📉 Dissatisfaction Rate: {dissatisfaction_rate:.1f}%")
    print(f"🎯 High Confidence Predictions: {high_confidence_predictions}/10")
    
    # Action items
    negative_reviews = feedback_df[feedback_df['sentiment'] == 'negative']
    print(f"\n🚨 URGENT ACTION NEEDED:")
    for _, row in negative_reviews.iterrows():
        print(f"   • '{row['review'][:40]}...' (Confidence: {row['confidence']:.2f})")
    
    # 2. SOCIAL MEDIA MONITORING
    print("\n\n📱 2. SOCIAL MEDIA BRAND MONITORING")
    print("-" * 35)
    
    social_mentions = [
        "@YourBrand just launched amazing new feature! #love",
        "Worst customer service from @YourBrand ever 😡",
        "@YourBrand app keeps crashing, fix this please",
        "Thank you @YourBrand for quick support response!",
        "@YourBrand pricing is too high compared to competitors",
        "Absolutely loving @YourBrand new update! 🔥",
        "@YourBrand delivery was super fast, impressed!",
        "Having issues with @YourBrand login system",
        "@YourBrand quality has improved significantly",
        "Disappointed with @YourBrand recent changes"
    ]
    
    social_predictions, social_confidence = analyzer.predict_with_confidence(social_mentions)
    
    social_df = pd.DataFrame({
        'mention': social_mentions,
        'sentiment': social_predictions,
        'confidence': social_confidence,
        'platform': ['Twitter'] * 10,
        'urgency': ['High' if s == 'negative' and c > 0.7 else 'Low' 
                   for s, c in zip(social_predictions, social_confidence)]
    })
    
    # Brand health metrics
    brand_sentiment = (social_df['sentiment'] == 'positive').mean() * 100
    crisis_alerts = social_df[social_df['urgency'] == 'High'].shape[0]
    
    print(f"📊 Brand Sentiment Score: {brand_sentiment:.1f}%")
    print(f"🚨 Crisis Alerts: {crisis_alerts} mentions need immediate attention")
    
    # Crisis management
    urgent_mentions = social_df[social_df['urgency'] == 'High']
    if not urgent_mentions.empty:
        print(f"\n⚠️  CRISIS MANAGEMENT ALERTS:")
        for _, row in urgent_mentions.iterrows():
            print(f"   • {row['platform']}: '{row['mention'][:50]}...'")
    
    # 3. PRODUCT COMPARISON ANALYSIS
    print("\n\n🔍 3. COMPETITIVE ANALYSIS")
    print("-" * 25)
    
    your_product_reviews = [
        "Great features and easy to use",
        "Good value for the price",
        "Some bugs but overall satisfied",
        "Excellent customer support"
    ]
    
    competitor_reviews = [
        "Expensive but high quality",
        "Difficult to use interface",
        "Poor customer service experience",
        "Limited features for the price"
    ]
    
    your_sentiment, _ = analyzer.predict_with_confidence(your_product_reviews)
    competitor_sentiment, _ = analyzer.predict_with_confidence(competitor_reviews)
    
    your_score = (your_sentiment == 'positive').mean() * 100
    competitor_score = (competitor_sentiment == 'positive').mean() * 100
    
    print(f"🏆 Your Product Sentiment: {your_score:.1f}%")
    print(f"🏢 Competitor Sentiment: {competitor_score:.1f}%")
    print(f"📈 Competitive Advantage: {your_score - competitor_score:+.1f} percentage points")
    
    # 4. BUSINESS RECOMMENDATIONS
    print("\n\n💡 4. AI-POWERED BUSINESS RECOMMENDATIONS")
    print("-" * 40)
    
    if satisfaction_rate < 70:
        print("🔴 CRITICAL: Customer satisfaction below 70%")
        print("   → Immediate product quality review needed")
        print("   → Enhance customer support training")
    elif satisfaction_rate < 85:
        print("🟡 WARNING: Customer satisfaction needs improvement")
        print("   → Focus on addressing common complaints")
        print("   → Implement feedback collection system")
    else:
        print("🟢 EXCELLENT: High customer satisfaction")
        print("   → Leverage positive reviews for marketing")
        print("   → Maintain current quality standards")
    
    if crisis_alerts > 2:
        print("🚨 URGENT: Multiple negative social mentions detected")
        print("   → Activate crisis communication plan")
        print("   → Respond to negative mentions within 2 hours")
    
    if your_score > competitor_score:
        print("🎯 OPPORTUNITY: Outperforming competitors")
        print("   → Highlight advantages in marketing campaigns")
        print("   → Capture market share with targeted ads")
    
    # 5. ROI CALCULATION
    print("\n\n💰 5. BUSINESS IMPACT & ROI")
    print("-" * 25)
    
    # Simulate business metrics
    monthly_customers = 10000
    avg_customer_value = 50
    churn_reduction = 0.05  # 5% churn reduction from sentiment monitoring
    
    monthly_revenue_impact = monthly_customers * avg_customer_value * churn_reduction
    annual_impact = monthly_revenue_impact * 12
    
    print(f"📊 Monthly Revenue Impact: ${monthly_revenue_impact:,.0f}")
    print(f"📈 Annual Revenue Impact: ${annual_impact:,.0f}")
    print(f"🎯 ROI from Sentiment Analysis: {annual_impact/10000:.0f}x")
    
    print("\n" + "="*50)
    print("✅ BUSINESS INTELLIGENCE TRANSFORMATION COMPLETE!")
    print("Raw text → Actionable insights → Business decisions")

if __name__ == "__main__":
    business_intelligence_demo()