#!/bin/bash
python nltk_setup.py
streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true