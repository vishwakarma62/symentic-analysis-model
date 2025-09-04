import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import random
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer

class RealTimeAnalyzer:
    def __init__(self):
        self.analyzer = AdvancedSentimentAnalyzer()
        self.data_buffer = []
        self.max_buffer_size = 100
        
    def simulate_real_time_data(self):
        """Simulate incoming text data"""
        sample_texts = [
            "Love the new AI features!",
            "This update is terrible",
            "Works fine for me",
            "Amazing performance boost!",
            "Buggy and slow",
            "Pretty good overall",
            "Excellent user experience",
            "Needs improvement",
            "Satisfied with results",
            "Disappointing quality"
        ]
        
        text = random.choice(sample_texts)
        timestamp = datetime.now()
        
        return {
            'timestamp': timestamp,
            'text': text,
            'source': random.choice(['Twitter', 'Reviews', 'Feedback', 'Comments'])
        }
    
    def process_real_time_data(self, data_point):
        """Process incoming data and add to buffer"""
        prediction, confidence = self.analyzer.predict_with_confidence([data_point['text']])
        
        processed_data = {
            **data_point,
            'sentiment': prediction[0],
            'confidence': confidence[0]
        }
        
        self.data_buffer.append(processed_data)
        
        # Keep buffer size manageable
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)
            
        return processed_data

def create_real_time_dashboard():
    st.set_page_config(
        page_title="Real-Time Sentiment Monitor",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Real-Time Sentiment Analysis Dashboard")
    st.markdown("### Live monitoring of sentiment trends")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = RealTimeAnalyzer()
        # Train with sample data
        from advanced_sentiment_analyzer import create_enhanced_dataset
        df = create_enhanced_dataset()
        st.session_state.analyzer.analyzer.train_and_compare_models(df['text'], df['sentiment'])
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)
    
    # Manual data input
    st.sidebar.header("📝 Manual Input")
    manual_text = st.sidebar.text_area("Enter text to analyze:")
    if st.sidebar.button("Analyze Text"):
        if manual_text:
            data_point = {
                'timestamp': datetime.now(),
                'text': manual_text,
                'source': 'Manual Input'
            }
            processed = st.session_state.analyzer.process_real_time_data(data_point)
            st.sidebar.success(f"Sentiment: {processed['sentiment']} (Confidence: {processed['confidence']:.3f})")
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # Simulate new data if auto refresh is on
    if auto_refresh:
        new_data = st.session_state.analyzer.simulate_real_time_data()
        st.session_state.analyzer.process_real_time_data(new_data)
    
    # Get current buffer data
    buffer_data = st.session_state.analyzer.data_buffer
    
    if buffer_data:
        df = pd.DataFrame(buffer_data)
        
        # Metrics
        total_samples = len(df)
        positive_pct = (df['sentiment'] == 'positive').mean() * 100
        negative_pct = (df['sentiment'] == 'negative').mean() * 100
        avg_confidence = df['confidence'].mean()
        
        with col1:
            st.metric("Total Samples", total_samples)
        with col2:
            st.metric("Positive %", f"{positive_pct:.1f}%")
        with col3:
            st.metric("Negative %", f"{negative_pct:.1f}%")
        with col4:
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        # Charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Sentiment over time
            fig_time = px.scatter(
                df, 
                x='timestamp', 
                y='confidence',
                color='sentiment',
                title="Sentiment Timeline",
                hover_data=['text', 'source']
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Source distribution
            source_counts = df['source'].value_counts()
            fig_source = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Data Sources"
            )
            st.plotly_chart(fig_source, use_container_width=True)
        
        with col_right:
            # Sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            fig_sentiment = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Sentiment Distribution",
                color=sentiment_counts.index
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Confidence distribution
            fig_conf = px.histogram(
                df,
                x='confidence',
                nbins=20,
                title="Confidence Score Distribution"
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Recent data table
        st.subheader("📋 Recent Analysis Results")
        recent_df = df.tail(10)[['timestamp', 'text', 'sentiment', 'confidence', 'source']]
        st.dataframe(recent_df, use_container_width=True)
        
        # Download data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"sentiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        st.info("No data available yet. Enable auto-refresh or add manual input.")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    create_real_time_dashboard()