# Streamlit Cloud Deployment Guide

## Prerequisites
- Your code is already on GitHub at: https://github.com/avika124/review-analyzer
- A Streamlit Cloud account (free tier available)

## Step-by-Step Deployment

### 1. Sign up for Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign up" and sign in with your GitHub account
3. Authorize Streamlit Cloud to access your GitHub repositories

### 2. Deploy Your App
1. Click "New app" button
2. Select your GitHub account
3. Select repository: `avika124/review-analyzer`
4. Select branch: `main`
5. Main file path: `app.py`
6. Click "Deploy!"

### 3. Configure Secrets (for Gemini API)
1. After deployment, go to your app's settings (three dots menu → Settings)
2. Click "Secrets" in the sidebar
3. Add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```
4. Save the secrets

### 4. Wait for Deployment
- Streamlit Cloud will install dependencies (this may take 5-10 minutes on first deploy)
- The app will automatically rebuild when you push changes to GitHub

## Important Notes

- **First deployment takes longer** because it needs to download the HuggingFace model (~500MB)
- **Memory usage**: The app is optimized to stay under 1GB (Streamlit Cloud free tier limit)
- **API Key**: Users can also enter their API key directly in the app sidebar if they prefer
- **Auto-deploy**: Every push to the `main` branch will automatically redeploy your app

## Troubleshooting

### If deployment fails:
1. Check the logs in Streamlit Cloud dashboard
2. Verify `requirements.txt` has all dependencies
3. Ensure `app.py` is in the root directory
4. Check that all file paths in code are correct

### If the app runs but shows errors:
1. Check that secrets are properly configured
2. Verify the HuggingFace model downloads correctly
3. Check browser console for any JavaScript errors

## Your App URL
Once deployed, your app will be available at:
`https://review-analyzer-avika124.streamlit.app`

(Or a similar URL based on your username and app name)

