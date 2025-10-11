# 🚂 Railway Deployment Guide

## Quick Deploy (5 minutes)

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO
git push -u origin main
```

2. **Deploy on Railway**:
- Go to [railway.app](https://railway.app)
- Connect GitHub account
- Select your repository
- Railway auto-detects and deploys!

3. **Environment Variables** (if needed):
```
FLASK_ENV=production
PORT=8080
```

## Cost: **FREE** (500 hours/month)