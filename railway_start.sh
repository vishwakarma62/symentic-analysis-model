#!/bin/bash

# Railway deployment startup script
echo "Starting Railway deployment..."

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Setup NLTK data
echo "Setting up NLTK data..."
python setup_nltk.py

# Start Streamlit app
echo "Starting Streamlit app..."
streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0