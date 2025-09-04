from business_sentiment_analyzer import BusinessSentimentAnalyzer, create_business_dataset
from advanced_sentiment_analyzer import create_enhanced_dataset

def ecommerce_use_case():
    """E-commerce product review analysis"""
    print("🛒 E-COMMERCE USE CASE")
    print("-" * 30)
    
    reviews = [
        "Fast delivery and great product quality! Customer service was excellent.",
        "Overpriced item, poor quality. Delivery was delayed by 3 days.",
        "Good value for money but customer support was unhelpful.",
        "Amazing product! Worth every penny. Will order again soon.",
        "Terrible experience. Product broke immediately, staff was rude."
    ]
    
    analyzer = BusinessSentimentAnalyzer()
    df = create_enhanced_dataset()
    analyzer.train_and_compare_models(df['text'], df['sentiment'])
    
    insights = analyzer.get_business_insights(reviews)
    
    print(f"📊 Analysis of {len(reviews)} product reviews:")
    for sentiment, count in insights['overall_summary']['sentiment_distribution'].items():
        print(f"  {sentiment}: {count} reviews")
    
    print("\n🎯 Key Issues Found:")
    for rec in insights['recommendations']:
        print(f"  • {rec}")

def restaurant_use_case():
    """Restaurant review analysis"""
    print("\n🍽️ RESTAURANT USE CASE")
    print("-" * 30)
    
    reviews = [
        "Food quality was excellent but service was very slow.",
        "Great staff and amazing food! Prices are reasonable too.",
        "Poor food quality and overpriced menu. Staff was friendly though.",
        "Outstanding service and delicious food. Will definitely return!",
        "Terrible experience. Food was cold, staff was rude, expensive prices."
    ]
    
    analyzer = BusinessSentimentAnalyzer()
    df = create_enhanced_dataset()
    analyzer.train_and_compare_models(df['text'], df['sentiment'])
    
    insights = analyzer.get_business_insights(reviews)
    
    print(f"📊 Analysis of {len(reviews)} restaurant reviews:")
    negative_count = insights['overall_summary']['sentiment_distribution'].get('negative', 0)
    if negative_count > 0:
        print(f"⚠️ {negative_count} negative reviews need attention")
    
    print("\n💡 Business Actions:")
    for rec in insights['recommendations']:
        print(f"  • {rec}")

def saas_product_use_case():
    """SaaS product feedback analysis"""
    print("\n💻 SAAS PRODUCT USE CASE")
    print("-" * 30)
    
    reviews = [
        "Great software features but customer support response is too slow.",
        "Excellent product quality and very responsive support team!",
        "Overpriced for the features offered. Interface needs improvement.",
        "Amazing value and outstanding customer service. Highly recommended!",
        "Poor software quality with many bugs. Support team is unhelpful."
    ]
    
    analyzer = BusinessSentimentAnalyzer()
    df = create_enhanced_dataset()
    analyzer.train_and_compare_models(df['text'], df['sentiment'])
    
    insights = analyzer.get_business_insights(reviews)
    
    print("📈 Product Feedback Summary:")
    aspects = insights['aspect_analysis']
    for aspect, data in aspects.items():
        if data['total_mentions'] > 0:
            pos_pct = data['positive'] / data['total_mentions'] * 100
            print(f"  {aspect.title()}: {pos_pct:.0f}% positive feedback")

def main():
    print("🏢 BUSINESS SENTIMENT ANALYSIS USE CASES")
    print("=" * 50)
    
    # Run different business scenarios
    ecommerce_use_case()
    restaurant_use_case() 
    saas_product_use_case()
    
    print("\n✅ BUSINESS APPLICATIONS:")
    print("• Product improvement prioritization")
    print("• Customer service optimization") 
    print("• Marketing message refinement")
    print("• Competitive analysis")
    print("• Quality control monitoring")
    print("• Pricing strategy validation")

if __name__ == "__main__":
    main()