import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression()
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        blob = TextBlob(text)
        return ' '.join([word.lemmatize() for word in blob.words])
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def train_model(self, texts, labels):
        """Train the sentiment analysis model"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.fit_transform(processed_texts)
        self.model.fit(X, labels)
        return self
    
    def predict(self, texts):
        """Predict sentiment for new texts"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        return self.model.predict(X)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return accuracy, report

def create_sample_data():
    """Create sample dataset for demonstration"""
    sample_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst experience I've ever had. Terrible service.",
        "The movie was okay, nothing special but not bad either.",
        "Absolutely fantastic! Highly recommend to everyone.",
        "Poor quality and overpriced. Very disappointed.",
        "Great customer service and fast delivery!",
        "The food was cold and tasteless. Won't come back.",
        "Perfect solution for my needs. Very satisfied.",
        "Average product, meets basic requirements.",
        "Outstanding quality and excellent value for money!"
    ]
    
    sample_labels = [
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'positive', 'negative', 'positive', 'neutral', 'positive'
    ]
    
    return pd.DataFrame({'text': sample_texts, 'sentiment': sample_labels})

def visualize_results(df):
    """Create visualizations for sentiment analysis results"""
    plt.figure(figsize=(12, 8))
    
    # Sentiment distribution
    plt.subplot(2, 2, 1)
    sentiment_counts = df['sentiment'].value_counts()
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    
    # Sentiment by text length
    plt.subplot(2, 2, 2)
    df['text_length'] = df['text'].str.len()
    sns.boxplot(data=df, x='sentiment', y='text_length')
    plt.title('Text Length by Sentiment')
    
    # Word cloud for positive sentiments
    plt.subplot(2, 2, 3)
    positive_texts = ' '.join(df[df['sentiment'] == 'positive']['text'])
    if positive_texts:
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(positive_texts)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Positive Sentiment Words')
    
    # Word cloud for negative sentiments
    plt.subplot(2, 2, 4)
    negative_texts = ' '.join(df[df['sentiment'] == 'negative']['text'])
    if negative_texts:
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(negative_texts)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Negative Sentiment Words')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("AI/ML Sentiment Analysis Project")
    print("=" * 40)
    
    # Create sample data
    df = create_sample_data()
    print(f"Dataset created with {len(df)} samples")
    
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
        # Using both TextBlob and trained model
        textblob_sentiment = analyzer.analyze_sentiment_textblob(text)
        model_sentiment = analyzer.predict([text])[0]
        print(f"Text: '{text}'")
        print(f"  TextBlob: {textblob_sentiment}")
        print(f"  ML Model: {model_sentiment}")
        print()
    
    # Create visualizations
    print("Generating visualizations...")
    visualize_results(df)
    
    print("Project completed! Check 'sentiment_analysis_results.png' for visualizations.")

if __name__ == "__main__":
    main()