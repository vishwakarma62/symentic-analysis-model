import pandas as pd
import numpy as np
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer
from textblob import TextBlob
import re

class BusinessSentimentAnalyzer(AdvancedSentimentAnalyzer):
    def __init__(self):
        super().__init__()
        self.business_keywords = {
            'product': ['product', 'item', 'goods', 'merchandise'],
            'service': ['service', 'support', 'help', 'assistance'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money'],
            'quality': ['quality', 'durability', 'reliable', 'defective'],
            'delivery': ['delivery', 'shipping', 'fast', 'slow', 'delayed'],
            'staff': ['staff', 'employee', 'representative', 'team']
        }
    
    def analyze_business_aspects(self, text):
        """Analyze specific business aspects in reviews"""
        text_lower = text.lower()
        aspects = {}
        
        for aspect, keywords in self.business_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                # Extract sentences containing aspect keywords
                sentences = text.split('.')
                aspect_sentences = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
                
                if aspect_sentences:
                    aspect_text = '. '.join(aspect_sentences)
                    blob = TextBlob(aspect_text)
                    aspects[aspect] = {
                        'sentiment': 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral',
                        'polarity': blob.sentiment.polarity,
                        'text': aspect_text.strip()
                    }
        
        return aspects
    
    def get_business_insights(self, reviews):
        """Generate business insights from multiple reviews"""
        results = []
        
        for review in reviews:
            prediction, confidence = self.predict_with_confidence([review])
            aspects = self.analyze_business_aspects(review)
            
            results.append({
                'review': review,
                'overall_sentiment': prediction[0],
                'confidence': confidence[0],
                'aspects': aspects
            })
        
        return self.generate_summary(results)
    
    def generate_summary(self, results):
        """Generate business summary and recommendations"""
        df = pd.DataFrame(results)
        
        # Overall sentiment distribution
        sentiment_counts = df['overall_sentiment'].value_counts()
        
        # Aspect analysis
        aspect_summary = {}
        for aspect in self.business_keywords.keys():
            aspect_sentiments = []
            for result in results:
                if aspect in result['aspects']:
                    aspect_sentiments.append(result['aspects'][aspect]['sentiment'])
            
            if aspect_sentiments:
                aspect_df = pd.Series(aspect_sentiments)
                aspect_summary[aspect] = {
                    'positive': (aspect_df == 'positive').sum(),
                    'negative': (aspect_df == 'negative').sum(),
                    'neutral': (aspect_df == 'neutral').sum(),
                    'total_mentions': len(aspect_sentiments)
                }
        
        # Generate recommendations
        recommendations = self.generate_recommendations(sentiment_counts, aspect_summary)
        
        return {
            'overall_summary': {
                'total_reviews': len(results),
                'sentiment_distribution': sentiment_counts.to_dict(),
                'average_confidence': df['confidence'].mean()
            },
            'aspect_analysis': aspect_summary,
            'recommendations': recommendations,
            'detailed_results': results
        }
    
    def generate_recommendations(self, sentiment_counts, aspect_summary):
        """Generate actionable business recommendations"""
        recommendations = []
        
        # Overall sentiment recommendations
        total_reviews = sentiment_counts.sum() if hasattr(sentiment_counts, 'sum') else sum(sentiment_counts.values())
        negative_pct = sentiment_counts.get('negative', 0) / total_reviews * 100
        if negative_pct > 30:
            recommendations.append("HIGH PRIORITY: 30%+ negative reviews - immediate action needed")
        elif negative_pct > 15:
            recommendations.append("MEDIUM PRIORITY: Monitor negative feedback trends")
        
        # Aspect-specific recommendations
        for aspect, data in aspect_summary.items():
            if data['total_mentions'] > 0:
                negative_pct = data['negative'] / data['total_mentions'] * 100
                
                if negative_pct > 40:
                    recommendations.append(f"Fix {aspect.upper()} issues - {negative_pct:.1f}% negative feedback")
                elif data['positive'] > data['negative']:
                    recommendations.append(f"{aspect.upper()} performing well - leverage in marketing")
        
        return recommendations

def create_business_dataset():
    """Create realistic business review dataset"""
    reviews = [
        # Product reviews
        "Great product quality but delivery was very slow. Customer service was helpful though.",
        "Excellent value for money! Fast shipping and amazing quality. Highly recommend!",
        "Poor quality product, broke after one week. Expensive for what you get.",
        "Good product but overpriced. Staff was friendly and helpful during purchase.",
        "Outstanding service and product quality. Will definitely buy again!",
        
        # Service reviews  
        "Terrible customer service experience. Staff was rude and unhelpful.",
        "Amazing support team! They resolved my issue quickly and professionally.",
        "Service was okay, nothing special. Product quality is decent for the price.",
        "Excellent customer service but the product arrived damaged.",
        "Fast delivery and great quality. Customer support was very responsive.",
        
        # Mixed reviews
        "Love the product quality but hate the high prices. Delivery was acceptable.",
        "Good value product with excellent customer service. Shipping could be faster.",
        "Poor quality control - received defective item. Staff helped with replacement.",
        "Expensive but worth it for the quality. Great customer support experience.",
        "Average product, average service, average price. Nothing stands out."
    ]
    
    return reviews

def main():
    print("🏢 Business Sentiment Analysis System")
    print("=" * 50)
    
    # Initialize business analyzer
    analyzer = BusinessSentimentAnalyzer()
    
    # Create and train on business data
    from advanced_sentiment_analyzer import create_enhanced_dataset
    df = create_enhanced_dataset()
    analyzer.train_and_compare_models(df['text'], df['sentiment'])
    
    # Analyze business reviews
    business_reviews = create_business_dataset()
    
    print("📊 Analyzing Business Reviews...")
    insights = analyzer.get_business_insights(business_reviews)
    
    # Display results
    print("\n📈 BUSINESS INSIGHTS REPORT")
    print("=" * 40)
    
    # Overall summary
    summary = insights['overall_summary']
    print(f"Total Reviews Analyzed: {summary['total_reviews']}")
    print(f"Average Confidence: {summary['average_confidence']:.1%}")
    print("\nSentiment Distribution:")
    for sentiment, count in summary['sentiment_distribution'].items():
        pct = count / summary['total_reviews'] * 100
        print(f"  {sentiment.title()}: {count} ({pct:.1f}%)")
    
    # Aspect analysis
    print("\n🎯 ASPECT ANALYSIS")
    print("-" * 20)
    for aspect, data in insights['aspect_analysis'].items():
        print(f"\n{aspect.upper()}:")
        print(f"  Mentions: {data['total_mentions']}")
        if data['total_mentions'] > 0:
            print(f"  Positive: {data['positive']} ({data['positive']/data['total_mentions']*100:.1f}%)")
            print(f"  Negative: {data['negative']} ({data['negative']/data['total_mentions']*100:.1f}%)")
    
    # Recommendations
    print("\n💡 BUSINESS RECOMMENDATIONS")
    print("-" * 30)
    for i, rec in enumerate(insights['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Sample detailed analysis
    print("\n📝 SAMPLE DETAILED ANALYSIS")
    print("-" * 35)
    sample = insights['detailed_results'][0]
    print(f"Review: \"{sample['review'][:60]}...\"")
    print(f"Overall: {sample['overall_sentiment']} ({sample['confidence']:.1%} confidence)")
    print("Aspects detected:")
    for aspect, details in sample['aspects'].items():
        print(f"  {aspect}: {details['sentiment']} (score: {details['polarity']:.2f})")

if __name__ == "__main__":
    main()