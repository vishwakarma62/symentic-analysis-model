import streamlit as st
import pandas as pd
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer, create_enhanced_dataset
from textblob import TextBlob

st.title("Sentiment Analysis - Development Dashboard")

@st.cache_data
def load_model():
    df = create_enhanced_dataset()
    analyzer = AdvancedSentimentAnalyzer()
    performance = analyzer.train_and_compare_models(df['text'], df['sentiment'])
    return analyzer, df, performance

analyzer, sample_df, performance = load_model()

# Model Performance Section
st.header("Model Performance")
st.dataframe(performance.sort_values('cv_score', ascending=False))

best_model = performance.loc[performance['cv_score'].idxmax()]
st.write(f"Best: {best_model['model']} + {best_model['vectorizer']} = {best_model['cv_score']:.3f}")

# Text Analysis
st.header("Text Analysis")
text_input = st.text_area("Enter text:", height=100)

if text_input:
    # ML Model
    predictions, confidence = analyzer.predict_with_confidence([text_input])
    
    # TextBlob
    blob = TextBlob(text_input)
    textblob_sentiment = "positive" if blob.sentiment.polarity > 0.1 else "negative" if blob.sentiment.polarity < -0.1 else "neutral"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ML Model", predictions[0])
        st.metric("Confidence", f"{confidence[0]:.1%}")
    with col2:
        st.metric("TextBlob", textblob_sentiment)
        st.metric("Polarity", f"{blob.sentiment.polarity:.3f}")

# Training Data
st.header("Training Data")
st.dataframe(sample_df)
st.bar_chart(sample_df['sentiment'].value_counts())

# Batch Testing
st.header("Batch Testing")
test_texts = st.text_area("Enter multiple texts (one per line):", height=150)

if test_texts:
    texts = [t.strip() for t in test_texts.split('\n') if t.strip()]
    if texts:
        predictions, confidence = analyzer.predict_with_confidence(texts)
        
        results = pd.DataFrame({
            'text': texts,
            'prediction': predictions,
            'confidence': confidence
        })
        
        st.dataframe(results)
        st.bar_chart(results['prediction'].value_counts())