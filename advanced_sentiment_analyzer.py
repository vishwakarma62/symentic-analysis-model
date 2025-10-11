import pandas as pd
import numpy as np

# Setup NLTK data
try:
    import nltk
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

from textblob import TextBlob
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import pickle
import joblib
from datetime import datetime

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': MultinomialNB()
        }
        self.vectorizers = {
            'tfidf': TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2)),
            'count': CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
        }
        self.best_model = None
        self.best_vectorizer = None
        self.model_performance = {}
        
    def advanced_preprocess(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lemmatization
        blob = TextBlob(text)
        return ' '.join([word.lemmatize() for word in blob.words])
    
    def train_and_compare_models(self, texts, labels):
        """Train multiple models and compare performance"""
        processed_texts = [self.advanced_preprocess(text) for text in texts]
        
        best_score = 0
        results = []
        
        for vec_name, vectorizer in self.vectorizers.items():
            X = vectorizer.fit_transform(processed_texts)
            
            for model_name, model in self.models.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X, labels, cv=5, scoring='accuracy')
                mean_score = cv_scores.mean()
                
                results.append({
                    'vectorizer': vec_name,
                    'model': model_name,
                    'cv_score': mean_score,
                    'cv_std': cv_scores.std()
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_model = model
                    self.best_vectorizer = vectorizer
                    
        self.model_performance = pd.DataFrame(results)
        
        # Train best model on full dataset
        X_best = self.best_vectorizer.fit_transform(processed_texts)
        self.best_model.fit(X_best, labels)
        
        return self.model_performance
    
    def predict_with_confidence(self, texts):
        """Predict with confidence scores"""
        processed_texts = [self.advanced_preprocess(text) for text in texts]
        X = self.best_vectorizer.transform(processed_texts)
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)
        confidence = np.max(probabilities, axis=1)
        
        return predictions, confidence
    
    def save_model(self, filepath):
        """Save trained model and vectorizer"""
        model_data = {
            'model': self.best_model,
            'vectorizer': self.best_vectorizer,
            'performance': self.model_performance,
            'timestamp': datetime.now()
        }
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath):
        """Load saved model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_vectorizer = model_data['vectorizer']
        self.model_performance = model_data['performance']

def create_enhanced_dataset():
    """Create a larger, more diverse dataset"""
    texts = [
        # Positive samples
        "I absolutely love this product! It exceeded all my expectations.",
        "Amazing quality and fantastic customer service. Highly recommended!",
        "This is the best purchase I've made this year. So happy!",
        "Excellent performance and great value for money.",
        "Outstanding features and user-friendly interface.",
        "Perfect solution for my needs. Works flawlessly!",
        "Incredible results and fast delivery. Very impressed!",
        "Top-notch quality and professional service.",
        "Brilliant design and excellent functionality.",
        "Superb experience from start to finish!",
        
        # Negative samples
        "Terrible product, complete waste of money.",
        "Worst customer service ever. Very disappointed.",
        "Poor quality and doesn't work as advertised.",
        "Overpriced and underdelivered. Avoid at all costs.",
        "Broken after one day. Cheap materials.",
        "Horrible experience, would not recommend to anyone.",
        "Defective product and unhelpful support team.",
        "Complete disaster. Nothing works properly.",
        "Fraudulent company with terrible products.",
        "Awful quality and misleading description.",
        
        # Neutral samples
        "The product is okay, nothing special but functional.",
        "Average quality, meets basic requirements.",
        "It's fine for the price, not amazing but decent.",
        "Standard product with typical features.",
        "Acceptable performance, could be better.",
        "Normal quality, as expected for this price range.",
        "It works but there are better alternatives.",
        "Mediocre product with mixed results.",
        "Fair quality, some good and bad aspects.",
        "Reasonable option if you're on a budget."
    ]
    
    labels = ['positive'] * 10 + ['negative'] * 10 + ['neutral'] * 10
    
    return pd.DataFrame({'text': texts, 'sentiment': labels})

def advanced_visualization(df, model_performance, predictions_df=None):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('Sentiment Distribution')
    
    # 2. Model performance comparison
    pivot_perf = model_performance.pivot(index='model', columns='vectorizer', values='cv_score')
    sns.heatmap(pivot_perf, annot=True, fmt='.3f', ax=axes[0,1], cmap='viridis')
    axes[0,1].set_title('Model Performance Heatmap')
    
    # 3. Text length analysis
    df['text_length'] = df['text'].str.len()
    sns.boxplot(data=df, x='sentiment', y='text_length', ax=axes[0,2])
    axes[0,2].set_title('Text Length by Sentiment')
    
    # 4. Word clouds
    positive_text = ' '.join(df[df['sentiment'] == 'positive']['text'])
    if positive_text:
        wordcloud = WordCloud(width=300, height=200, background_color='white').generate(positive_text)
        axes[1,0].imshow(wordcloud, interpolation='bilinear')
        axes[1,0].axis('off')
        axes[1,0].set_title('Positive Words')
    
    # 5. Performance bar chart
    best_scores = model_performance.groupby('model')['cv_score'].max().sort_values(ascending=True)
    axes[1,1].barh(best_scores.index, best_scores.values)
    axes[1,1].set_title('Best Model Scores')
    axes[1,1].set_xlabel('Cross-Validation Accuracy')
    
    # 6. Confidence distribution (if predictions available)
    if predictions_df is not None and 'confidence' in predictions_df.columns:
        axes[1,2].hist(predictions_df['confidence'], bins=20, alpha=0.7)
        axes[1,2].set_title('Prediction Confidence Distribution')
        axes[1,2].set_xlabel('Confidence Score')
    else:
        axes[1,2].text(0.5, 0.5, 'No prediction data', ha='center', va='center')
        axes[1,2].set_title('Prediction Analysis')
    
    plt.tight_layout()
    plt.savefig('advanced_sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Advanced AI/ML Sentiment Analysis Project")
    print("=" * 50)
    
    # Create enhanced dataset
    df = create_enhanced_dataset()
    print(f"Enhanced dataset created with {len(df)} samples")
    
    # Initialize advanced analyzer
    analyzer = AdvancedSentimentAnalyzer()
    
    # Train and compare models
    print("\nTraining and comparing multiple models...")
    performance_df = analyzer.train_and_compare_models(df['text'], df['sentiment'])
    
    print("\nModel Performance Comparison:")
    print(performance_df.sort_values('cv_score', ascending=False))
    
    # Best model info
    best_result = performance_df.loc[performance_df['cv_score'].idxmax()]
    print(f"\nBest Model: {best_result['model']} with {best_result['vectorizer']}")
    print(f"Cross-validation Score: {best_result['cv_score']:.4f} (+/- {best_result['cv_std']:.4f})")
    
    # Test predictions with confidence
    test_texts = [
        "This AI project is absolutely incredible and revolutionary!",
        "I hate this terrible and buggy software",
        "The application works fine, nothing extraordinary",
        "Machine learning is transforming the world positively",
        "Poor implementation and disappointing results"
    ]
    
    print("\nTesting with confidence scores:")
    predictions, confidence = analyzer.predict_with_confidence(test_texts)
    
    predictions_df = pd.DataFrame({
        'text': test_texts,
        'prediction': predictions,
        'confidence': confidence
    })
    
    for _, row in predictions_df.iterrows():
        print(f"Text: '{row['text'][:50]}...'")
        print(f"  Prediction: {row['prediction']} (Confidence: {row['confidence']:.3f})")
        print()
    
    # Save model
    model_path = 'advanced_sentiment_model.pkl'
    analyzer.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Create advanced visualizations
    print("Generating advanced visualizations...")
    advanced_visualization(df, performance_df, predictions_df)
    
    print("Advanced project completed!")
    print("Key improvements:")
    print("- Multiple model comparison")
    print("- Advanced text preprocessing")
    print("- Confidence scoring")
    print("- Model persistence")
    print("- Enhanced visualizations")

if __name__ == "__main__":
    main()