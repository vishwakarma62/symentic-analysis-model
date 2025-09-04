from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer, create_enhanced_dataset

app = Flask(__name__)

# Global analyzer instance
analyzer = None
model_info = {}

def initialize_model():
    """Initialize the sentiment analysis model"""
    global analyzer, model_info
    
    model_path = 'advanced_sentiment_model.pkl'
    
    if os.path.exists(model_path):
        # Load existing model
        analyzer = AdvancedSentimentAnalyzer()
        analyzer.load_model(model_path)
        model_info = {
            'status': 'loaded',
            'model_path': model_path,
            'loaded_at': datetime.now().isoformat()
        }
    else:
        # Train new model
        analyzer = AdvancedSentimentAnalyzer()
        df = create_enhanced_dataset()
        performance = analyzer.train_and_compare_models(df['text'], df['sentiment'])
        analyzer.save_model(model_path)
        
        best_result = performance.loc[performance['cv_score'].idxmax()]
        model_info = {
            'status': 'trained',
            'model_path': model_path,
            'best_model': best_result['model'],
            'best_vectorizer': best_result['vectorizer'],
            'accuracy': float(best_result['cv_score']),
            'trained_at': datetime.now().isoformat()
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': analyzer is not None
    })

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    return jsonify(model_info)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """Predict sentiment for single text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        if not text or not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Make prediction
        predictions, confidence = analyzer.predict_with_confidence([text])
        
        result = {
            'text': text,
            'sentiment': predictions[0],
            'confidence': float(confidence[0]),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict sentiment for multiple texts"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts field'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts allowed per batch'}), 400
        
        # Filter empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            return jsonify({'error': 'No valid texts provided'}), 400
        
        # Make predictions
        predictions, confidence = analyzer.predict_with_confidence(valid_texts)
        
        results = []
        for i, text in enumerate(valid_texts):
            results.append({
                'text': text,
                'sentiment': predictions[i],
                'confidence': float(confidence[i])
            })
        
        response = {
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/stats', methods=['POST'])
def analyze_statistics():
    """Get sentiment statistics for a batch of texts"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts field'}), 400
        
        texts = data['texts']
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            return jsonify({'error': 'No valid texts provided'}), 400
        
        # Make predictions
        predictions, confidence = analyzer.predict_with_confidence(valid_texts)
        
        # Calculate statistics
        df = pd.DataFrame({
            'sentiment': predictions,
            'confidence': confidence
        })
        
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        sentiment_percentages = (df['sentiment'].value_counts(normalize=True) * 100).to_dict()
        
        stats = {
            'total_texts': len(valid_texts),
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': {k: round(v, 2) for k, v in sentiment_percentages.items()},
            'average_confidence': float(df['confidence'].mean()),
            'confidence_std': float(df['confidence'].std()),
            'high_confidence_count': int((df['confidence'] > 0.8).sum()),
            'low_confidence_count': int((df['confidence'] < 0.5).sum()),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain model with new data"""
    try:
        data = request.get_json()
        
        if not data or 'training_data' not in data:
            return jsonify({'error': 'Missing training_data field'}), 400
        
        training_data = data['training_data']
        
        # Validate training data format
        if not isinstance(training_data, list):
            return jsonify({'error': 'training_data must be a list'}), 400
        
        texts = []
        labels = []
        
        for item in training_data:
            if 'text' not in item or 'sentiment' not in item:
                return jsonify({'error': 'Each training item must have text and sentiment fields'}), 400
            texts.append(item['text'])
            labels.append(item['sentiment'])
        
        if len(texts) < 10:
            return jsonify({'error': 'Minimum 10 training samples required'}), 400
        
        # Retrain model
        global analyzer, model_info
        analyzer = AdvancedSentimentAnalyzer()
        performance = analyzer.train_and_compare_models(texts, labels)
        
        # Save updated model
        model_path = 'advanced_sentiment_model.pkl'
        analyzer.save_model(model_path)
        
        best_result = performance.loc[performance['cv_score'].idxmax()]
        model_info = {
            'status': 'retrained',
            'model_path': model_path,
            'best_model': best_result['model'],
            'best_vectorizer': best_result['vectorizer'],
            'accuracy': float(best_result['cv_score']),
            'training_samples': len(texts),
            'retrained_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'message': 'Model retrained successfully',
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Initializing Sentiment Analysis API...")
    initialize_model()
    print("Model initialized successfully!")
    print("\nAvailable endpoints:")
    print("- GET  /health - Health check")
    print("- GET  /model/info - Model information")
    print("- POST /predict - Single text prediction")
    print("- POST /predict/batch - Batch prediction")
    print("- POST /analyze/stats - Sentiment statistics")
    print("- POST /retrain - Retrain model")
    print("\nStarting API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)