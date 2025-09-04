from flask import Flask, request, jsonify
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer, create_enhanced_dataset
import threading
import time
import requests

app = Flask(__name__)
analyzer = None

def init_model():
    global analyzer
    analyzer = AdvancedSentimentAnalyzer()
    df = create_enhanced_dataset()
    analyzer.train_and_compare_models(df['text'], df['sentiment'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    predictions, confidence = analyzer.predict_with_confidence([text])
    return jsonify({
        'text': text,
        'sentiment': predictions[0],
        'confidence': float(confidence[0])
    })

@app.route('/batch', methods=['POST'])
def batch_predict():
    data = request.get_json()
    texts = data['texts']
    predictions, confidence = analyzer.predict_with_confidence(texts)
    
    results = []
    for i, text in enumerate(texts):
        results.append({
            'text': text,
            'sentiment': predictions[i],
            'confidence': float(confidence[i])
        })
    
    # Business insights
    positive_count = sum(1 for p in predictions if p == 'positive')
    negative_count = sum(1 for p in predictions if p == 'negative')
    satisfaction_rate = (positive_count / len(predictions)) * 100
    
    return jsonify({
        'results': results,
        'business_insights': {
            'satisfaction_rate': f"{satisfaction_rate:.1f}%",
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_analyzed': len(texts)
        }
    })

def run_server():
    print("Initializing API server...")
    init_model()
    print("Model trained! Starting server...")
    app.run(port=5001, debug=False, use_reloader=False)

def test_api():
    time.sleep(3)  # Wait for server to start
    
    print("\nTesting API endpoints...")
    
    # Test single prediction
    response = requests.post('http://localhost:5001/predict', 
                           json={'text': 'I love this AI project!'})
    print("Single Prediction:", response.json())
    
    # Test batch prediction with business insights
    business_data = {
        'texts': [
            'Excellent product quality!',
            'Terrible customer service',
            'Average experience overall',
            'Outstanding support team!',
            'Poor value for money'
        ]
    }
    
    response = requests.post('http://localhost:5001/batch', json=business_data)
    result = response.json()
    
    print("\nBusiness Intelligence Results:")
    print(f"Satisfaction Rate: {result['business_insights']['satisfaction_rate']}")
    print(f"Positive Reviews: {result['business_insights']['positive_count']}")
    print(f"Negative Reviews: {result['business_insights']['negative_count']}")
    
    print("\nDetailed Analysis:")
    for item in result['results']:
        print(f"  {item['sentiment'].upper()} ({item['confidence']:.2f}): '{item['text'][:40]}...'")

if __name__ == "__main__":
    # Start server in background
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Test the API
    test_api()
    
    print("\nAPI Demo Complete! Your sentiment analyzer is now a business intelligence service!")