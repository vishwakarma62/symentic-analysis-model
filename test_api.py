import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_model_info():
    """Test model info endpoint"""
    print("Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_single_prediction():
    """Test single text prediction"""
    print("Testing single prediction...")
    data = {"text": "I love this amazing AI project!"}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_batch_prediction():
    """Test batch prediction"""
    print("Testing batch prediction...")
    data = {
        "texts": [
            "This is absolutely fantastic!",
            "I hate this terrible product",
            "It's okay, nothing special",
            "Amazing quality and service",
            "Worst experience ever"
        ]
    }
    response = requests.post(f"{BASE_URL}/predict/batch", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_statistics():
    """Test sentiment statistics"""
    print("Testing sentiment statistics...")
    data = {
        "texts": [
            "Love it!", "Hate it!", "It's fine", "Amazing!", "Terrible!",
            "Great job!", "Poor quality", "Average product", "Excellent!", "Disappointing"
        ]
    }
    response = requests.post(f"{BASE_URL}/analyze/stats", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def main():
    print("🧪 Testing Sentiment Analysis API")
    print("=" * 40)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    try:
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_statistics()
        
        print("✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server.")
        print("Make sure to run 'python api_service.py' first.")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main()