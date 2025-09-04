import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer, create_enhanced_dataset

# Change this line:
# from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer, create_enhanced_dataset

# To this:
# from sentiment_analyzer import SentimentAnalyzer, create_sample_data

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)


# # Change this:
# def load_model():
#     df = create_enhanced_dataset()
#     analyzer = AdvancedSentimentAnalyzer()
#     analyzer.train_and_compare_models(df['text'], df['sentiment'])
#     return analyzer, df

# # To this:
# def load_model():
#     df = create_sample_data()
#     analyzer = SentimentAnalyzer()
#     analyzer.train_model(df['text'], df['sentiment'])
#     return analyzer, df

@st.cache_data
def load_model():
    """Load and train the sentiment analysis model"""
    df = create_enhanced_dataset()
    analyzer = AdvancedSentimentAnalyzer()
    analyzer.train_and_compare_models(df['text'], df['sentiment'])
    return analyzer, df

def main():
    # Load model first
    analyzer, sample_df = load_model()
    
    st.title("Sentiment Analysis Dashboard")
    st.markdown("### Analyze text sentiment using Machine Learning")
    st.warning("⚠️ Development Version - Not for Production Use")
    
    # Sidebar
    st.sidebar.header("Features")
    st.sidebar.info("""
    - Text preprocessing and cleaning
    - Machine Learning classification
    - Natural Language Processing
    - Data visualization
    - Interactive web application
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
                # TextBlob analysis
                blob = TextBlob(user_text)
                textblob_polarity = blob.sentiment.polarity
                textblob_sentiment = "Positive" if textblob_polarity > 0.1 else "Negative" if textblob_polarity < -0.1 else "Neutral"
                
# # Change this:
# ml_predictions, ml_confidence = analyzer.predict_with_confidence([user_text])
# ml_sentiment = ml_predictions[0].title()

# # To this:
# ml_sentiment = analyzer.predict([user_text])[0].title()

                

                # ML Model analysis
                ml_predictions, ml_confidence = analyzer.predict_with_confidence([user_text])
                ml_sentiment = ml_predictions[0].title()
                
                # Display results
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
                
                # Sentiment gauge
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
    
    # Batch analysis section
    st.header("Batch Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file with a 'text' column",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' in df.columns:
                st.success(f"File uploaded successfully! {len(df)} rows found.")
                
                if st.button("Analyze All Texts"):
                    # Analyze all texts
                    predictions, confidence = analyzer.predict_with_confidence(df['text'])
                    df['ml_sentiment'] = predictions
                    df['ml_confidence'] = confidence
                    
                    # Convert TextBlob polarity to sentiment labels
                    def get_textblob_sentiment(text):
                        polarity = TextBlob(text).sentiment.polarity
                        if polarity > 0.1:
                            return "positive"
                        elif polarity < -0.1:
                            return "negative"
                        else:
                            return "neutral"
                    
                    df['textblob_sentiment'] = df['text'].apply(get_textblob_sentiment)
                    
                    # Display results
                    st.subheader("Batch Analysis Results")
                    st.dataframe(df)
                    
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
                st.error("CSV file must contain a 'text' column!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    


if __name__ == "__main__":
    main()