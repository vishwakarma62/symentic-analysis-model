# 📁 GitHub Setup (First Time)

## Step 1: Create Repository on GitHub
1. Go to [github.com](https://github.com)
2. Click "New repository" (green button)
3. Name: `sentiment-analysis-project`
4. Make it **Public** (required for free deployments)
5. Click "Create repository"

## Step 2: Initialize Local Git
```bash
cd sentiment_analysis_project
git init
git add .
git commit -m "Initial sentiment analysis project"
```

## Step 3: Connect to GitHub
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/sentiment-analysis-project.git
git branch -M main
git push -u origin main
```

## Step 4: Deploy Instantly
- **Streamlit Cloud**: Use your new repo URL
- **Railway**: Connect the repo
- **Render**: Select the repo

✅ **Done! Your code is now on GitHub and ready to deploy**