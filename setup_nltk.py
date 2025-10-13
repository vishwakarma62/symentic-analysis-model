#!/usr/bin/env python3
"""
NLTK Setup Script for Deployment
Handles NLTK corpus downloads for production environments
"""

import nltk
import ssl
import os
import sys

def setup_nltk_data():
    """Download all required NLTK data"""
    
    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Set NLTK data directory
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # Required NLTK packages
    packages = [
        'punkt',
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4',
        'brown',
        'movie_reviews',
        'stopwords'
    ]
    
    print("Downloading NLTK packages...")
    
    for package in packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"✓ {package} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {package}: {e}")
    
    # Download TextBlob corpora
    try:
        print("Downloading TextBlob corpora...")
        import textblob
        textblob.download_corpora()
        print("✓ TextBlob corpora downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download TextBlob corpora: {e}")
    
    print("NLTK setup completed!")

if __name__ == "__main__":
    setup_nltk_data()