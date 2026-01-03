import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

class SimpleSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = LogisticRegression()
        self.is_trained = False
    
    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())
    
    def train(self, texts, labels):
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.fit_transform(processed_texts)
        self.model.fit(X, labels)
        self.is_trained = True
    
    def predict(self, texts):
        if not self.is_trained:
            return ['neutral'] * len(texts), [0.5] * len(texts)
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidence = np.max(probabilities, axis=1)
        return predictions, confidence

@st.cache_resource
def load_model():
    texts = [
        "I love this product! Amazing quality!",
        "Great service and fast delivery!",
        "Excellent experience, highly recommended!",
        "Perfect solution for my needs!",
        "Outstanding quality and value!",
        "Terrible product, waste of money!",
        "Worst service ever, very disappointed!",
        "Poor quality, doesn't work properly!",
        "Horrible experience, avoid this!",
        "Overpriced and low quality!",
        "The product is okay, nothing special.",
        "Average quality, meets requirements.",
        "It's fine for the price.",
        "Standard product with typical features.",
        "Acceptable performance, could be better."
    ]
    
    labels = ['positive'] * 5 + ['negative'] * 5 + ['neutral'] * 5
    
    analyzer = SimpleSentimentAnalyzer()
    analyzer.train(texts, labels)
    
    df = pd.DataFrame({'text': texts, 'sentiment': labels})
    return analyzer, df

def simple_sentiment(text):
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'outstanding', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor', 'disappointing']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "Positive", 0.1
    elif neg_count > pos_count:
        return "Negative", -0.1
    else:
        return "Neutral", 0.0

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

def main():
    analyzer, sample_df = load_model()
    
    st.title("📊 Business Sentiment Intelligence")
    st.markdown("### Professional sentiment analysis for customer feedback and reviews")
    st.info("🚀 Professional Sentiment Analysis Platform - Ready for Business Use")
    
    st.sidebar.header("Features")
    st.sidebar.info("""
    🏢 **Business Applications:**
    - Customer review analysis  
    - Social media monitoring  
    - Support ticket prioritization  
    - Brand sentiment tracking  
    - Market research insights
    """)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Analyze Your Text")
        
        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your text here...",
            height=100
        )
        
        if st.button("Analyze Sentiment", type="primary"):
            if user_text:
                simple_sentiment_result, simple_polarity = simple_sentiment(user_text)
                ml_predictions, ml_confidence = analyzer.predict([user_text])
                ml_sentiment = ml_predictions[0].title()
                
                st.subheader("Analysis Results")
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.metric("Rule-Based Sentiment", simple_sentiment_result)
                    st.metric("Polarity Score", f"{simple_polarity:.3f}")
                
                with result_col2:
                    st.metric("ML Model Prediction", ml_sentiment)
                
                with result_col3:
                    st.metric("ML Confidence", f"{ml_confidence[0]:.1%}")
            else:
                st.warning("Please enter some text to analyze!")
    
    with col2:
        st.header("Sample Data")
        st.dataframe(sample_df, use_container_width=True)
        
        sentiment_counts = sample_df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
    
    st.header("Batch Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with text data",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! {len(df)} rows found.")
            
            text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 10:
                        text_columns.append(col)
            
            if text_columns:
                selected_column = st.selectbox(
                    "Select text column to analyze:",
                    text_columns,
                    index=0
                )
                
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Analyze Selected Column"):
                    texts = df[selected_column].astype(str).fillna("")
                    
                    predictions, confidence = analyzer.predict(texts)
                    df['ml_sentiment'] = predictions
                    df['ml_confidence'] = confidence
                    
                    rule_sentiments = []
                    for text in texts:
                        sentiment, _ = simple_sentiment(text)
                        rule_sentiments.append(sentiment.lower())
                    
                    df['rule_sentiment'] = rule_sentiments
                    
                    st.subheader("Batch Analysis Results")
                    st.dataframe(df, use_container_width=True)
                    
                    ml_counts = df['ml_sentiment'].value_counts()
                    st.bar_chart(ml_counts)
            else:
                st.warning("No suitable text columns found.")
                st.write("Available columns:", list(df.columns))
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    main()