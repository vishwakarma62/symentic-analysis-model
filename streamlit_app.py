import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

# =====================================================
# 🔧 NLTK + TextBlob Setup (Fix MissingCorpusError)
# =====================================================
import nltk
from textblob import download_corpora

def setup_nltk():
    """Ensure all required NLTK and TextBlob corpora are available."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

    # Fallback: TextBlob built-in corpora
    try:
        download_corpora()
    except Exception:
        pass

# Run setup on startup
setup_nltk()

# =====================================================
# Import your model utilities
# =====================================================
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer, create_enhanced_dataset

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)


# =====================================================
# Model Loader
# =====================================================
@st.cache_data
def load_model():
    """Load and train the sentiment analysis model"""
    df = create_enhanced_dataset()
    analyzer = AdvancedSentimentAnalyzer()
    analyzer.train_and_compare_models(df['text'], df['sentiment'])
    return analyzer, df


# =====================================================
# Main Application
# =====================================================
def main():
    # Load model first
    analyzer, sample_df = load_model()
    
    st.title("📊 Business Sentiment Intelligence")
    st.markdown("### Professional sentiment analysis for customer feedback, reviews, and social media monitoring")
    st.info("🚀 Professional Sentiment Analysis Platform - Ready for Business Use")
    
    # Sidebar
    st.sidebar.header("Features")
    st.sidebar.info("""
    🏢 **Business Applications:**
    - Customer review analysis  
    - Social media monitoring  
    - Support ticket prioritization  
    - Brand sentiment tracking  
    - Market research insights
    """)

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Analyze Your Text")
        
        # Text input
        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your text here...",
            height=100
        )
        
        if st.button("Analyze Sentiment", type="primary"):
            if user_text:
                # --- TextBlob Analysis ---
                blob = TextBlob(user_text)
                textblob_polarity = blob.sentiment.polarity
                textblob_sentiment = (
                    "Positive" if textblob_polarity > 0.1 
                    else "Negative" if textblob_polarity < -0.1 
                    else "Neutral"
                )

                # --- ML Model Analysis ---
                ml_predictions, ml_confidence = analyzer.predict_with_confidence([user_text])
                ml_sentiment = ml_predictions[0].title()
                
                # --- Display Results ---
                st.subheader("Analysis Results")
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.metric("TextBlob Sentiment", textblob_sentiment)
                    st.metric("Polarity Score", f"{textblob_polarity:.3f}")
                
                with result_col2:
                    st.metric("ML Model Prediction", ml_sentiment)
                
                with result_col3:
                    st.metric("ML Confidence", f"{ml_confidence[0]:.1%}")
                    st.metric("TextBlob Polarity", f"{textblob_polarity:.3f}")
                
                # --- Visualization ---
                fig = px.bar(
                    x=[textblob_sentiment, ml_sentiment],
                    y=["TextBlob", "ML Model"],
                    orientation='h',
                    title="Sentiment Comparison",
                    color=[textblob_sentiment, ml_sentiment],
                    color_discrete_map={
                        'Positive': '#00ff00',
                        'Negative': '#ff0000',
                        'Neutral': '#ffff00'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to analyze!")
    
    with col2:
        st.header("Sample Data")
        st.dataframe(sample_df, use_container_width=True)
        
        # Sentiment distribution
        sentiment_counts = sample_df['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Training Data Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # =====================================================
    # Batch Analysis
    # =====================================================
    st.header("Batch Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload any CSV file - we'll auto-detect text columns",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! {len(df)} rows found.")
            
            # Auto-detect text columns
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
                    
                    # ML model analysis
                    predictions, confidence = analyzer.predict_with_confidence(texts)
                    df['ml_sentiment'] = predictions
                    df['ml_confidence'] = confidence
                    
                    # TextBlob analysis
                    def get_textblob_sentiment(text):
                        try:
                            polarity = TextBlob(str(text)).sentiment.polarity
                            if polarity > 0.1:
                                return "positive"
                            elif polarity < -0.1:
                                return "negative"
                            else:
                                return "neutral"
                        except:
                            return "neutral"
                    
                    df['textblob_sentiment'] = texts.apply(get_textblob_sentiment)
                    
                    st.subheader("Batch Analysis Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Visualization
                    ml_counts = df['ml_sentiment'].value_counts()
                    tb_counts = df['textblob_sentiment'].value_counts()
                    
                    fig_comparison = px.bar(
                        x=list(ml_counts.index) + list(tb_counts.index),
                        y=list(ml_counts.values) + list(tb_counts.values),
                        color=['ML Model'] * len(ml_counts) + ['TextBlob'] * len(tb_counts),
                        title="Sentiment Analysis Comparison",
                        barmode='group'
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
            else:
                st.warning("No suitable text columns found. Upload a file with text data (comments, reviews, etc.)")
                st.subheader("Available Columns")
                st.write(list(df.columns))
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    

# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    main()
