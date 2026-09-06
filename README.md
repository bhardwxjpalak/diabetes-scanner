# ConjuVascular Web Application: Deployment Guide

This directory contains the complete web application for the **Conjunctival Microvascular Diabetes Screening Pipeline**, featuring a FastAPI backend and a responsive clinical dashboard frontend.

---

## 1. Quick Local Run

To test the application locally on your computer:

```bash
cd website

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
uvicorn app:app --reload --port 8000
```

Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 2. Deploy to Railway (Recommended)

Railway allows you to deploy the full Python container (with OpenCV, ONNX Runtime, and XGBoost) in under 3 minutes with zero configuration.

### Option A: Via GitHub (Easiest)
1. Push this project (or the `website/` folder) to your GitHub repository.
2. Log in to [Railway.app](https://railway.app).
3. Click **"New Project"** $\rightarrow$ **"Deploy from GitHub repo"**.
4. Select your repository.
5. In the service settings:
   - **Root Directory**: Set to `website` (if deploying the whole monorepo) or leave empty if the repo root is `website/`.
6. Railway will automatically detect the `Dockerfile`, build the container, and deploy it.
7. Click **"Generate Domain"** in your Railway dashboard to get your live public URL (e.g., `https://conjuvascular-production.up.railway.app`).

### Option B: Via Railway CLI
```bash
# 1. Install Railway CLI (if not installed)
npm install -g @railway/cli

# 2. Login to Railway
railway login

# 3. Initialize and deploy from the website directory
cd website
railway init
railway up
```

---

## 3. Deploy to Vercel

If you prefer to deploy to Vercel:
1. Install Vercel CLI: `npm install -g vercel`
2. Run from the `website/` folder:
   ```bash
   cd website
   vercel
   ```
*(Note: Because ONNX Runtime and OpenCV dependencies can exceed Vercel Serverless Function limits of 250 MB, Railway Docker deployment is strongly recommended for production stability).*

---

## 4. API Endpoints

- **`GET /`**: Clinical web dashboard (HTML/CSS/JS)
- **`GET /health`**: Healthcheck endpoint returning model status
- **`POST /api/screen`**:
  - Accepts `multipart/form-data` with key `file` (image file) OR `application/json` with `image_base64`.
  - Returns full JSON analysis:
    - `triage`: `risk_level`, `probability`, `cvhi_score`, `threshold`
    - `biomarkers`: `tvl`, `mba`, `lac`, `fd`
    - `visualizations`: base64 encoded ROI overlay, vessel overlay, and skeleton
- **`POST /api/quality-check`**: Quick image quality validation (blur, illumination, contrast, glare)
