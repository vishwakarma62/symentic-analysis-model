#!/bin/bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true