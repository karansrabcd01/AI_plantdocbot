# 📖 PlantDocBot - Detailed Setup Guide

This guide will walk you through setting up PlantDocBot from scratch on your local machine.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 16+** - [Download Node.js](https://nodejs.org/)
- **Git** - [Download Git](https://git-scm.com/downloads/)
- **Code Editor** - VS Code, PyCharm, or your preferred editor

---

## 🔧 Step-by-Step Installation

### **Step 1: Clone the Repository**

```bash
# Clone the repository
git clone https://github.com/yourusername/plantdocbot.git

# Navigate to project directory
cd plantdocbot
```

---

### **Step 2: Backend Setup**

#### **2.1 Create Virtual Environment**

**On Windows:**
```bash
cd Backend
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
cd Backend
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

#### **2.2 Install Python Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including:
- FastAPI
- PyTorch
- Transformers
- Pillow
- Uvicorn
- And more...

**Note**: Installation may take 5-10 minutes depending on your internet speed.

#### **2.3 Verify Models are Present**

Check that the `models/` directory contains:
```
Backend/models/
├── ImageClassification_model_weights.pth
└── text_classifier_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer files...
    └── label_encoder.pkl
```

If models are missing, you'll need to train them using the provided notebooks.

#### **2.4 Configure Environment Variables (Optional)**

Create a `.env` file in the `Backend/` directory:

```bash
# Create .env file
touch .env  # On Windows: type nul > .env
```

Add the following (optional):
```env
# Google Gemini API for advanced chatbot (optional)
GEMINI_API_KEY=your_api_key_here

# HuggingFace token for better rate limits (optional)
HUGGINGFACE_TOKEN=your_token_here
```

**Getting Gemini API Key:**
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy and paste into `.env`

---

### **Step 3: Frontend Setup**

#### **3.1 Navigate to Frontend Directory**

```bash
# From project root
cd plantdoc-frontend
```

#### **3.2 Install Node Dependencies**

```bash
npm install
```

This will install:
- React 19
- Vite
- Axios
- ESLint
- And other dependencies

**Note**: Installation may take 2-3 minutes.

#### **3.3 Verify Installation**

Check that `node_modules/` directory was created and `package-lock.json` exists.

---

### **Step 4: Running the Application**

#### **4.1 Start the Backend Server**

**Terminal 1:**
```bash
cd Backend
# Activate venv if not already activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Start the server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Test the API:**
Open browser and visit: `http://localhost:8000/health-check`

You should see:
```json
{"status":"ok","message":"PlantDocBot API is running 🚀"}
```

#### **4.2 Start the Frontend Development Server**

**Terminal 2:**
```bash
cd plantdoc-frontend

# Start Vite dev server
npm run dev
```

You should see:
```
VITE v7.1.12  ready in XXX ms

➜  Local:   http://localhost:5173/
➜  press h + enter to show help
```

#### **4.3 Access the Application**

Open your browser and navigate to: `http://localhost:5173`

You should see the PlantDocBot interface with:
- Header with logo and title
- Image Analysis section
- Text Diagnosis section
- AI Plant Care Assistant chatbot

---

## ✅ Verification Checklist

Use this checklist to ensure everything is working:

### **Backend**
- [ ] Virtual environment activated
- [ ] All dependencies installed without errors
- [ ] Models directory exists with required files
- [ ] Server starts without errors
- [ ] Health check endpoint returns success
- [ ] No error messages in terminal

### **Frontend**
- [ ] Node modules installed
- [ ] Dev server starts without errors
- [ ] Application loads in browser
- [ ] No console errors in browser DevTools
- [ ] All three sections visible

### **Functionality**
- [ ] Can upload an image
- [ ] Image analysis returns results
- [ ] Can type text symptoms
- [ ] Text diagnosis returns results
- [ ] Can send messages in chatbot
- [ ] Chatbot responds with advice

---

## 🧪 Testing the Application

### **Test 1: Image Analysis**

1. Navigate to "Image Analysis" section
2. Click "Choose File"
3. Select a plant image (use `Example/apple.jpg` if available)
4. Click "ANALYZE IMAGE"
5. Wait for results
6. Verify you see:
   - Disease name
   - Confidence percentage
   - Recommendation

### **Test 2: Text Diagnosis**

1. Navigate to "Text Diagnosis" section
2. Type: "Yellow spots on leaves with brown edges"
3. Click "DIAGNOSE"
4. Verify you see:
   - Disease prediction
   - Confidence score
   - Recommendation

### **Test 3: Chatbot**

1. Scroll to "AI Plant Care Assistant"
2. Type: "how to water plants"
3. Click "Send 🚀"
4. Verify you receive a detailed response about watering

---

## 🐛 Troubleshooting

### **Backend Issues**

#### **Problem: ModuleNotFoundError**
```
Solution:
1. Ensure virtual environment is activated
2. Run: pip install -r requirements.txt
3. Check Python version: python --version (should be 3.8+)
```

#### **Problem: Model files not found**
```
Solution:
1. Check Backend/models/ directory exists
2. Verify model files are present
3. If missing, you need to train models using notebooks
```

#### **Problem: Port 8000 already in use**
```
Solution:
1. Change port: uvicorn main:app --reload --port 8001
2. Update frontend API URL in App.jsx
```

### **Frontend Issues**

#### **Problem: npm install fails**
```
Solution:
1. Delete node_modules/ and package-lock.json
2. Run: npm cache clean --force
3. Run: npm install again
```

#### **Problem: Port 5173 already in use**
```
Solution:
Vite will automatically use next available port (5174, 5175, etc.)
```

#### **Problem: API connection refused**
```
Solution:
1. Ensure backend is running on port 8000
2. Check CORS settings in main.py
3. Verify API URL in App.jsx matches backend
```

### **Common Errors**

#### **CORS Error**
```
Error: Access to fetch at 'http://localhost:8000' from origin 
'http://localhost:5173' has been blocked by CORS policy

Solution:
Backend main.py already has CORS configured. 
Ensure backend is running and restart if needed.
```

#### **Gemini API Error**
```
Error: Gemini API error / timeout

Solution:
1. Check GEMINI_API_KEY in .env file
2. Verify API key is valid
3. Chatbot will fall back to knowledge base if API fails
```

---

## 🔄 Updating the Application

### **Update Backend Dependencies**
```bash
cd Backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade -r requirements.txt
```

### **Update Frontend Dependencies**
```bash
cd plantdoc-frontend
npm update
```

---

## 📝 Development Tips

### **Backend Development**

1. **Auto-reload is enabled**: Changes to `main.py` will automatically restart the server

2. **View API docs**: Visit `http://localhost:8000/docs` for interactive API documentation

3. **Check logs**: Terminal shows all API requests and errors

### **Frontend Development**

1. **Hot Module Replacement (HMR)**: Changes to React files update instantly

2. **Browser DevTools**: Press F12 to see console logs and network requests

3. **React DevTools**: Install React DevTools browser extension for debugging

---

## 🚀 Next Steps

After successful setup:

1. **Explore the Application**
   - Try different plant images
   - Test various symptom descriptions
   - Chat with the AI assistant

2. **Customize**
   - Modify UI colors in `App.css`
   - Add new chatbot knowledge topics
   - Enhance API endpoints

3. **Deploy**
   - See deployment section in main README
   - Consider hosting on Render, Railway, or Heroku

---

## 📞 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review error messages carefully
3. Search existing GitHub issues
4. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Your environment (OS, Python version, Node version)

---

## ✅ Setup Complete!

Congratulations! Your PlantDocBot is now set up and running. 

**Quick Start Commands:**

```bash
# Terminal 1 - Backend
cd Backend
source venv/bin/activate  # Windows: venv\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 - Frontend
cd plantdoc-frontend
npm run dev
```

Then open: `http://localhost:5173`

Happy plant disease detecting! 🌿✨
