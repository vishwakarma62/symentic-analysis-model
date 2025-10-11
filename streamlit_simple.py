import streamlit as st
import pandas as pd
import plotly.express as px
from simple_sentiment_analyzer import SimpleSentimentAnalyzer, create_simple_dataset

st.set_page_config(
    page_title="Sentiment Analysis Platform",
    page_icon="🤖",
    layout="wide"
)

@st.cache_data
def load_simple_model():
    """Load simple model without TextBlob"""
    df = create_simple_dataset()
    analyzer = SimpleSentimentAnalyzer()
    analyzer.train_model(df['text'], df['sentiment'])
    return analyzer, df

def main():
    analyzer, sample_df = load_simple_model()
    
    st.title("📊 Business Sentiment Intelligence Platform")
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
        
        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your text here...",
            height=100
        )
        
        if st.button("Analyze Sentiment", type="primary"):
            if user_text:
                predictions, confidence = analyzer.predict_with_confidence([user_text])
                sentiment = predictions[0].title()
                conf = confidence[0]
                
                st.subheader("Analysis Results")
                
                col_result1, col_result2 = st.columns(2)
                with col_result1:
                    st.metric("Sentiment", sentiment)
                with col_result2:
                    st.metric("Confidence", f"{conf:.1%}")
                
                # Simple visualization
                colors = {'Positive': '#00ff00', 'Negative': '#ff0000', 'Neutral': '#ffff00'}
                fig = px.bar(
                    x=[sentiment], y=[conf],
                    title="Sentiment Analysis Result",
                    color=[sentiment],
                    color_discrete_map=colors
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to analyze!")
    
    with col2:
        st.header("Sample Data")
        st.dataframe(sample_df, use_container_width=True)
        
        # Simple pie chart
        sentiment_counts = sample_df['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Training Data Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()