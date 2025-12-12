# Deployment Guide - PlantDocBot

This guide covers deploying PlantDocBot to production environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Backend Deployment](#backend-deployment)
- [Frontend Deployment](#frontend-deployment)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Git account
- GitHub account (for code hosting)
- Hosting platform accounts (choose one for each):
  - **Backend**: Render, Railway, Heroku, or AWS
  - **Frontend**: Netlify, Vercel, or GitHub Pages

---

## Backend Deployment

### Option 1: Render (Recommended - Free Tier Available)

1. **Prepare the Repository**
   ```bash
   # Ensure all changes are committed
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Create Render Account**
   - Visit [render.com](https://render.com)
   - Sign up with GitHub

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `plantdocbot-api`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r Backend/requirements.txt`
     - **Start Command**: `cd Backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Instance Type**: Free

4. **Set Environment Variables**
   - Go to "Environment" tab
   - Add:
     ```
     GEMINI_API_KEY=your_gemini_api_key
     HUGGINGFACE_TOKEN=your_hf_token
     ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)
   - Note your API URL: `https://plantdocbot-api.onrender.com`

### Option 2: Railway

1. **Install Railway CLI** (optional)
   ```bash
   npm install -g @railway/cli
   ```

2. **Deploy via Dashboard**
   - Visit [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Configure:
     - **Root Directory**: `Backend`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Add Environment Variables**
   - Click "Variables" tab
   - Add GEMINI_API_KEY and HUGGINGFACE_TOKEN

### Option 3: Heroku

1. **Create Procfile** in Backend directory:
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

2. **Deploy**
   ```bash
   heroku login
   heroku create plantdocbot-api
   git subtree push --prefix Backend heroku main
   ```

---

## Frontend Deployment

### Option 1: Netlify (Recommended)

1. **Update API Endpoint**
   
   Edit `plantdoc-frontend/src/App.jsx`:
   ```javascript
   const API_BASE_URL = 'https://your-backend-url.onrender.com';
   ```

2. **Build the Frontend**
   ```bash
   cd plantdoc-frontend
   npm run build
   ```

3. **Deploy to Netlify**
   
   **Method A: Drag & Drop**
   - Visit [netlify.com](https://netlify.com)
   - Drag the `dist` folder to Netlify

   **Method B: CLI**
   ```bash
   npm install -g netlify-cli
   netlify login
   netlify deploy --prod
   ```

   **Method C: GitHub Integration**
   - Connect GitHub repository
   - Configure:
     - **Base directory**: `plantdoc-frontend`
     - **Build command**: `npm run build`
     - **Publish directory**: `plantdoc-frontend/dist`

### Option 2: Vercel

1. **Update API Endpoint** (same as Netlify)

2. **Deploy**
   ```bash
   npm install -g vercel
   cd plantdoc-frontend
   vercel --prod
   ```

### Option 3: GitHub Pages

1. **Install gh-pages**
   ```bash
   cd plantdoc-frontend
   npm install --save-dev gh-pages
   ```

2. **Update package.json**
   ```json
   {
     "homepage": "https://yourusername.github.io/plantdocbot",
     "scripts": {
       "predeploy": "npm run build",
       "deploy": "gh-pages -d dist"
     }
   }
   ```

3. **Deploy**
   ```bash
   npm run deploy
   ```

---

## Environment Variables

### Backend (.env)

```env
# Required for advanced chatbot features
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Better rate limits for HuggingFace models
HUGGINGFACE_TOKEN=your_hf_token_here

# Production settings
ENVIRONMENT=production
ALLOWED_ORIGINS=https://your-frontend-url.netlify.app
```

### Frontend

Update in `App.jsx`:
```javascript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

Create `.env.production`:
```env
VITE_API_URL=https://your-backend-url.onrender.com
```

---

## Model Files Hosting

Since model files are large, host them separately:

### Option 1: Google Drive

1. Upload models to Google Drive
2. Make shareable with "Anyone with the link"
3. Update `Backend/models/.gitkeep` with download link

### Option 2: Hugging Face Hub

1. Create account at [huggingface.co](https://huggingface.co)
2. Upload models:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload your-username/plantdocbot-models ./Backend/models
   ```
3. Download in deployment:
   ```python
   from huggingface_hub import hf_hub_download
   model_path = hf_hub_download(repo_id="your-username/plantdocbot-models", 
                                  filename="ImageClassification_model_weights.pth")
   ```

### Option 3: Include in Deployment

For platforms with sufficient storage (Railway, AWS):
- Use Git LFS for large files
- Or download models during build process

---

## Post-Deployment Checklist

- [ ] Backend API is accessible
- [ ] Frontend loads correctly
- [ ] CORS is configured properly
- [ ] Environment variables are set
- [ ] Model files are accessible
- [ ] Image upload works
- [ ] Text prediction works
- [ ] Chatbot responds correctly
- [ ] SSL/HTTPS is enabled
- [ ] Custom domain configured (optional)

---

## Monitoring & Maintenance

### Health Checks

Backend includes a health check endpoint:
```bash
curl https://your-backend-url.onrender.com/health-check
```

### Logs

**Render**: Dashboard → Logs tab
**Railway**: Dashboard → Deployments → View Logs
**Netlify**: Dashboard → Deploys → Deploy log

### Updates

```bash
# Make changes
git add .
git commit -m "Update: description"
git push origin main

# Auto-deploys on most platforms
# Or trigger manual deploy from dashboard
```

---

## Troubleshooting

### Backend Issues

**Problem**: Models not loading
- **Solution**: Ensure models are in correct directory or download during startup

**Problem**: CORS errors
- **Solution**: Add frontend URL to CORS origins in `main.py`:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["https://your-frontend.netlify.app"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

**Problem**: Out of memory
- **Solution**: Upgrade to paid tier or optimize model loading

### Frontend Issues

**Problem**: API calls failing
- **Solution**: Check API_BASE_URL is correct and includes https://

**Problem**: Build fails
- **Solution**: Clear node_modules and reinstall:
  ```bash
  rm -rf node_modules package-lock.json
  npm install
  npm run build
  ```

---

## Security Best Practices

1. **Never commit .env files**
2. **Use environment variables for secrets**
3. **Enable HTTPS only**
4. **Implement rate limiting** (for production)
5. **Validate all inputs**
6. **Keep dependencies updated**

---

## Cost Optimization

### Free Tier Limits

**Render Free**:
- Spins down after 15 min inactivity
- 750 hours/month

**Netlify Free**:
- 100 GB bandwidth/month
- Unlimited sites

**Railway Free Trial**:
- $5 credit/month

### Tips:
- Use free tiers for demo/portfolio
- Upgrade only when needed
- Monitor usage regularly

---

## Support

For deployment issues:
- Check platform documentation
- Review logs carefully
- Open GitHub issue with error details

---

**Happy Deploying! 🚀**
