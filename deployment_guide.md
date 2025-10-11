# 🚀 Deployment Guide - Professional Sentiment Analysis Platform

## Platform Overview
A production-ready sentiment analysis platform designed for businesses to analyze customer feedback, social media sentiment, and market insights at scale.

## 🏗️ Architecture Components

### 1. Core ML Pipeline
- **Advanced Sentiment Analyzer**: Multi-model comparison and selection
- **Text Preprocessing**: Advanced cleaning and feature extraction
- **Model Persistence**: Save/load trained models
- **Performance Metrics**: Cross-validation and confidence scoring

### 2. Web Applications
- **Streamlit Dashboard**: Interactive analysis interface
- **Real-time Monitor**: Live sentiment tracking
- **API Service**: RESTful web service

### 3. API Endpoints
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /predict` - Single text prediction
- `POST /predict/batch` - Batch predictions
- `POST /analyze/stats` - Sentiment statistics
- `POST /retrain` - Model retraining

## 🛠️ Local Development Setup

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Installation
```bash
# Clone/download project
cd sentiment_analysis_project

# Install dependencies
pip install -r requirements_enhanced.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### Running Applications

#### 1. Command Line Analysis
```bash
python advanced_sentiment_analyzer.py
```

#### 2. Interactive Dashboard
```bash
streamlit run streamlit_app.py
```

#### 3. Real-time Monitor
```bash
streamlit run real_time_analyzer.py
```

#### 4. API Service
```bash
python api_service.py
```

#### 5. Test API
```bash
python test_api.py
```

## ☁️ Cloud Deployment Options

### Option 1: AWS Deployment

#### EC2 Instance
```bash
# Launch EC2 instance (t3.medium recommended)
# Install dependencies
sudo yum update -y
sudo yum install python3 python3-pip -y
pip3 install -r requirements_enhanced.txt

# Run API service
python3 api_service.py
```

#### AWS Lambda + API Gateway
```python
# Create Lambda deployment package
zip -r sentiment-api.zip . -x "*.git*" "*.pyc" "__pycache__/*"

# Deploy using AWS CLI or Console
aws lambda create-function \
  --function-name sentiment-analysis \
  --runtime python3.9 \
  --role arn:aws:iam::account:role/lambda-role \
  --handler api_service.lambda_handler \
  --zip-file fileb://sentiment-api.zip
```

#### ECS/Fargate
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_enhanced.txt .
RUN pip install -r requirements_enhanced.txt

COPY . .
EXPOSE 5000

CMD ["python", "api_service.py"]
```

### Option 2: Google Cloud Platform

#### Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/sentiment-api
gcloud run deploy --image gcr.io/PROJECT_ID/sentiment-api --platform managed
```

#### App Engine
```yaml
# app.yaml
runtime: python39

env_variables:
  FLASK_ENV: production

automatic_scaling:
  min_instances: 1
  max_instances: 10
```

### Option 3: Azure Deployment

#### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name sentiment-api \
  --image myregistry.azurecr.io/sentiment-api:latest \
  --ports 5000
```

#### Azure Functions
```python
# function_app.py
import azure.functions as func
from api_service import app

def main(req: func.HttpRequest) -> func.HttpResponse:
    return func.WsgiMiddleware(app).handle(req)
```

### Option 4: Heroku Deployment

```bash
# Create Heroku app
heroku create sentiment-analysis-api

# Add Procfile
echo "web: python api_service.py" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## 🔧 Production Considerations

### 1. Environment Configuration
```python
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key')
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')
    MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', 100))
    REDIS_URL = os.environ.get('REDIS_URL')
```

### 2. Database Integration
```python
# For storing predictions and analytics
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://user:pass@host:port/db')
df.to_sql('predictions', engine, if_exists='append')
```

### 3. Caching Layer
```python
# Redis caching for frequent predictions
import redis
import pickle

redis_client = redis.Redis.from_url(os.environ.get('REDIS_URL'))

def cached_predict(text):
    cache_key = f"prediction:{hash(text)}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return pickle.loads(cached)
    
    result = analyzer.predict([text])
    redis_client.setex(cache_key, 3600, pickle.dumps(result))
    return result
```

### 4. Monitoring & Logging
```python
import logging
from prometheus_client import Counter, Histogram

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### 5. Security
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# API key authentication
@app.before_request
def require_api_key():
    api_key = request.headers.get('X-API-Key')
    if not api_key or api_key != os.environ.get('API_KEY'):
        return jsonify({'error': 'Invalid API key'}), 401
```

## 📊 Performance Optimization

### 1. Model Optimization
- Use model quantization for smaller size
- Implement model caching
- Batch processing for multiple requests

### 2. Infrastructure Scaling
- Auto-scaling based on CPU/memory usage
- Load balancing across multiple instances
- CDN for static assets

### 3. Database Optimization
- Connection pooling
- Query optimization
- Read replicas for analytics

## 🔍 Monitoring & Analytics

### Key Metrics to Track
- Prediction accuracy over time
- Response time percentiles
- Error rates by endpoint
- Model confidence distribution
- Usage patterns by source

### Dashboards
- Grafana for infrastructure metrics
- Custom Streamlit dashboard for ML metrics
- CloudWatch/Stackdriver for cloud metrics

## 🚀 CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy Sentiment API

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements_enhanced.txt
      - name: Run tests
        run: python -m pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Deployment commands here
```

## 📈 Scaling Strategies

### Horizontal Scaling
- Multiple API instances behind load balancer
- Microservices architecture
- Event-driven processing with queues

### Vertical Scaling
- Optimize model inference
- Use GPU acceleration for large models
- Memory optimization techniques

This deployment guide provides a comprehensive roadmap for deploying your sentiment analysis platform for business use across multiple cloud platforms with scalable architecture.