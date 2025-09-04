"""
Quick demo script to showcase the sentiment analysis project
Perfect for presentations and demonstrations
"""

from sentiment_analyzer import SentimentAnalyzer, create_sample_data
import time

def animated_print(text, delay=0.03):
    """Print text with typewriter effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def demo():
    print("🎬" + "="*50)
    animated_print("   🤖 AI/ML SENTIMENT ANALYSIS DEMO")
    print("="*52)
    
    # Initialize
    animated_print("\n🔄 Initializing AI model...")
    time.sleep(1)
    
    df = create_sample_data()
    analyzer = SentimentAnalyzer()
    analyzer.train_model(df['text'], df['sentiment'])
    
    animated_print("✅ Model trained successfully!")
    
    # Demo examples
    demo_texts = [
        "This AI project is absolutely amazing! I love it!",
        "The worst experience ever. Completely disappointed.",
        "It's okay, nothing special but works fine.",
        "Incredible technology! This will change everything!",
        "Terrible quality and poor customer service."
    ]
    
    print("\n🧪 LIVE SENTIMENT ANALYSIS DEMO")
    print("-" * 40)
    
    for i, text in enumerate(demo_texts, 1):
        print(f"\n📝 Example {i}:")
        animated_print(f"Text: '{text}'")
        
        # Analyze
        textblob_result = analyzer.analyze_sentiment_textblob(text)
        ml_result = analyzer.predict([text])[0]
        
        # Color coding for results
        colors = {
            'positive': '🟢',
            'negative': '🔴', 
            'neutral': '🟡'
        }
        
        animated_print(f"TextBlob: {colors.get(textblob_result, '⚪')} {textblob_result.upper()}")
        animated_print(f"ML Model: {colors.get(ml_result, '⚪')} {ml_result.upper()}")
        
        time.sleep(1.5)
    
    print("\n" + "="*52)
    animated_print("🎉 Demo completed! Ready for your presentation!")
    print("="*52)
    
    print("\n📋 NEXT STEPS:")
    print("1. Run 'python sentiment_analyzer.py' for full analysis")
    print("2. Run 'streamlit run streamlit_app.py' for web app")
    print("3. Check README.md for detailed documentation")

if __name__ == "__main__":
    demo()