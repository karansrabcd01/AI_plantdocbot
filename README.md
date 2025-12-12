# 🌿 PlantDocBot - AI-Powered Plant Disease Detection

<div align="center">

![PlantDocBot](https://img.shields.io/badge/PlantDocBot-AI%20Assistant-10b981?style=for-the-badge&logo=leaf)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.120-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An intelligent web application for plant disease detection using deep learning and AI-powered plant care assistance.**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation)

</div>

---

## 📖 Overview

PlantDocBot is a comprehensive plant health management system that combines computer vision, natural language processing, and AI to help users identify and treat plant diseases.

### Core Capabilities

1. **🖼️ Image-based Disease Detection** - Upload plant images for instant disease identification
2. **💬 Text-based Diagnosis** - Describe symptoms to get disease predictions  
3. **🤖 AI Plant Care Assistant** - Interactive chatbot for plant care advice

### Key Features

- **38 Plant Disease Classes** - Comprehensive coverage of common plant diseases
- **High Accuracy** - CNN model with ~95% accuracy, BERT model with ~92% accuracy
- **Real-time Analysis** - Instant predictions and recommendations
- **Smart Chatbot** - Knowledge-based responses with optional Gemini AI integration
- **Modern UI/UX** - Responsive design with glassmorphism and smooth animations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/karansrabcd01/AI_plantdocbot.git
cd AI_plantdocbot

# Backend setup
cd Backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../plantdoc-frontend
npm install
```

### Running the Application

**Terminal 1 - Backend:**
```bash
cd Backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd plantdoc-frontend
npm run dev
```

**Access:** Open `http://localhost:5173` in your browser

---

## 📁 Project Structure

```
plantdocbot/
├── Backend/                    # FastAPI backend
│   ├── models/                 # ML models (download separately)
│   ├── main.py                 # Main API application
│   ├── requirements.txt        # Python dependencies
│   └── .env.example            # Environment template
│
├── plantdoc-frontend/          # React frontend
│   ├── src/
│   │   ├── App.jsx             # Main component
│   │   ├── App.css             # Styling
│   │   └── main.jsx            # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/                  # Jupyter notebooks for training
│   ├── ImageClassification.ipynb
│   └── TextClassifier.ipynb
│
├── docs/                       # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── SETUP_GUIDE.md
│   └── DEPLOYMENT.md
│
├── examples/                   # Sample test images
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## 🛠 Tech Stack

**Backend:**
- FastAPI 0.120.1
- PyTorch 2.9.0
- Transformers (Hugging Face)
- Pillow 12.0.0

**Frontend:**
- React 19.1.1
- Vite 7.1.7
- Axios 1.13.1

**AI/ML:**
- Custom CNN (Image Classification)
- BERT (Text Classification)
- Google Gemini AI (Optional Chatbot)

---

## 🔑 Configuration

### Environment Variables

Create a `.env` file in the `Backend/` directory:

```env
# Optional: Google Gemini API for advanced chatbot
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: HuggingFace token for better rate limits
HUGGINGFACE_TOKEN=your_hf_token_here
```

### Getting API Keys

**Google Gemini API (Free):**
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in and create an API key
3. Add to `.env` file

**HuggingFace Token (Optional):**
1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token
3. Add to `.env` file

---

## 📦 Model Files

Due to file size limitations (~103 MB), model files are not included in the repository.

### Download Models

**Option 1:** Download pre-trained models from [Google Drive Link - Add your link here]

**Option 2:** Train your own models using the notebooks in `notebooks/`

### Model Placement

After downloading, place files in:
```
Backend/models/
├── ImageClassification_model_weights.pth  (~103 MB)
└── text_classifier_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer files...
    └── label_encoder.pkl
```

See `Backend/models/.gitkeep` for detailed instructions.

---

## 🎯 API Endpoints

### Health Check
```http
GET /health-check
```

### Image Prediction
```http
POST /image-prediction
Content-Type: multipart/form-data
Body: file (image)
```

### Text Prediction
```http
POST /text-prediction
Content-Type: application/json
Body: { "input": "description of symptoms" }
```

### Chatbot
```http
POST /chatbot
Content-Type: application/json
Body: { "message": "your question", "conversation_history": [] }
```

See [API Documentation](docs/API_DOCUMENTATION.md) for detailed examples.

---

## 📚 Documentation

- **[Setup Guide](docs/SETUP_GUIDE.md)** - Detailed installation instructions
- **[API Documentation](docs/API_DOCUMENTATION.md)** - Complete API reference
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Deploy to production

---

## 🚢 Deployment

### Backend (Render)
```bash
# Build Command: pip install -r Backend/requirements.txt
# Start Command: cd Backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Frontend (Netlify)
```bash
cd plantdoc-frontend
npm run build
# Deploy dist/ folder to Netlify
```

See [Deployment Guide](docs/DEPLOYMENT.md) for detailed instructions.

---

## 🌱 Supported Plant Diseases

The system can detect **38 plant disease classes** including:

**Plants:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

**Common Diseases:** Scab, Black Rot, Powdery Mildew, Early Blight, Late Blight, Leaf Spot, Bacterial Spot, Mosaic Virus, and more.

<details>
<summary>View all 38 classes</summary>

1. Apple - Scab
2. Apple - Black Rot
3. Apple - Cedar Apple Rust
4. Apple - Healthy
5. Blueberry - Healthy
6. Cherry - Powdery Mildew
7. Cherry - Healthy
8. Corn - Cercospora Leaf Spot
9. Corn - Common Rust
10. Corn - Northern Leaf Blight
11. Corn - Healthy
12. Grape - Black Rot
13. Grape - Esca (Black Measles)
14. Grape - Leaf Blight
15. Grape - Healthy
16. Orange - Huanglongbing
17. Peach - Bacterial Spot
18. Peach - Healthy
19. Pepper - Bacterial Spot
20. Pepper - Healthy
21. Potato - Early Blight
22. Potato - Late Blight
23. Potato - Healthy
24. Raspberry - Healthy
25. Soybean - Healthy
26. Squash - Powdery Mildew
27. Strawberry - Leaf Scorch
28. Strawberry - Healthy
29. Tomato - Bacterial Spot
30. Tomato - Early Blight
31. Tomato - Late Blight
32. Tomato - Leaf Mold
33. Tomato - Septoria Leaf Spot
34. Tomato - Spider Mites
35. Tomato - Target Spot
36. Tomato - Mosaic Virus
37. Tomato - Yellow Leaf Curl Virus
38. Tomato - Healthy

</details>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** - Training data
- **Hugging Face** - Transformer models
- **Google Gemini** - AI capabilities
- **FastAPI** - Web framework
- **React & Vite** - Frontend tools

---

## 📞 Contact

**GitHub Repository:** [https://github.com/karansrabcd01/AI_plantdocbot](https://github.com/karansrabcd01/AI_plantdocbot)

For questions or support, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ and 🌿**

⭐ Star this repo if you find it helpful!

</div>
