import nltk
import ssl
import os

def setup_nltk():
    try:
        # Handle SSL issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        print("NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

if __name__ == "__main__":
    setup_nltk()