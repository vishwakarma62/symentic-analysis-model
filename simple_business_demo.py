from business_sentiment_analyzer import BusinessSentimentAnalyzer
from advanced_sentiment_analyzer import create_enhanced_dataset

def main():
    print("BUSINESS SENTIMENT ANALYSIS DEMO")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = BusinessSentimentAnalyzer()
    df = create_enhanced_dataset()
    analyzer.train_and_compare_models(df['text'], df['sentiment'])
    
    # E-commerce reviews
    print("\nE-COMMERCE PRODUCT REVIEWS:")
    print("-" * 30)
    
    ecommerce_reviews = [
        "Fast delivery and great product quality! Customer service was excellent.",
        "Overpriced item, poor quality. Delivery was delayed by 3 days.", 
        "Good value for money but customer support was unhelpful.",
        "Amazing product! Worth every penny. Will order again soon."
    ]
    
    insights = analyzer.get_business_insights(ecommerce_reviews)
    
    print(f"Total Reviews: {insights['overall_summary']['total_reviews']}")
    print("Sentiment Distribution:")
    for sentiment, count in insights['overall_summary']['sentiment_distribution'].items():
        pct = count / insights['overall_summary']['total_reviews'] * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print("\nBusiness Recommendations:")
    for i, rec in enumerate(insights['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\nAspect Analysis:")
    for aspect, data in insights['aspect_analysis'].items():
        if data['total_mentions'] > 0:
            pos_pct = data['positive'] / data['total_mentions'] * 100
            print(f"  {aspect}: {pos_pct:.0f}% positive ({data['total_mentions']} mentions)")
    
    # Sample detailed analysis
    print("\nSample Review Analysis:")
    sample = insights['detailed_results'][0]
    print(f"Review: \"{sample['review'][:50]}...\"")
    print(f"Overall Sentiment: {sample['overall_sentiment']} ({sample['confidence']:.1%} confidence)")
    print("Detected Aspects:")
    for aspect, details in sample['aspects'].items():
        print(f"  - {aspect}: {details['sentiment']} (score: {details['polarity']:.2f})")

if __name__ == "__main__":
    main()