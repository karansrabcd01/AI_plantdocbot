# 🌿 PlantDocBot - Complete Project Presentation

## AI-Powered Plant Disease Detection System

**A comprehensive guide covering dataset acquisition, model architecture, implementation, and deployment**

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Sources](#2-dataset-sources)
3. [Model Architecture](#3-model-architecture)
4. [CNN Implementation](#4-cnn-implementation)
5. [Transformer Implementation](#5-transformer-implementation)
6. [Model Training & Weights](#6-model-training--weights)
7. [FastAPI Backend](#7-fastapi-backend)
8. [React Frontend](#8-react-frontend)
9. [Data Flow & Processing](#9-data-flow--processing)
10. [System Architecture](#10-system-architecture)

---

## 1. Project Overview

### 🎯 Objective
Build an intelligent web application that detects plant diseases using:
- **Image-based detection** (CNN)
- **Text-based diagnosis** (BERT Transformer)
- **AI chatbot assistance** (Google Gemini)

### 🏆 Key Features
- 38 plant disease classes
- Dual-model architecture (CNN + Transformer)
- Real-time predictions
- Interactive AI assistant
- Modern responsive UI

### 📊 Performance Metrics
| Model | Accuracy | Task |
|-------|----------|------|
| CNN | ~95% | Image Classification |
| BERT | ~92% | Text Classification |

---

## 2. Dataset Sources

### 📸 Image Dataset

**Source:** PlantVillage Dataset

**Access Method:**
```python
# Available through Kaggle or direct download
# https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
```

**Dataset Structure:**
```
PlantVillage/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
├── Tomato___Bacterial_spot/
├── Tomato___Early_blight/
└── ... (38 classes total)
```

**Dataset Statistics:**
- **Total Images:** ~54,000+
- **Classes:** 38 (14 plant species)
- **Image Size:** 256x256 pixels (RGB)
- **Format:** JPG/PNG
- **Split:** 80% train, 20% validation

**Plants Covered:**
- Apple, Blueberry, Cherry, Corn, Grape
- Orange, Peach, Pepper, Potato, Raspberry
- Soybean, Squash, Strawberry, Tomato

---

### 💬 Text Dataset

**Source:** Hugging Face Dataset

**Dataset:** `ButterChicken98/plantvillage-image-text-pairs`

**Access Method:**
```python
import pandas as pd

# Load from Hugging Face
df = pd.read_parquet(
    "hf://datasets/ButterChicken98/plantvillage-image-text-pairs/data/train-00000-of-00001.parquet"
)
```

**Dataset Structure:**
```
Columns:
- text: Symptom descriptions
- label: Disease class name
- image: Associated image (optional)
```

**Example Data:**
```python
{
    "text": "Yellow spots on leaves with brown halos",
    "label": "Tomato - Early Blight"
}
```

**Dataset Statistics:**
- **Total Samples:** ~10,000+ text descriptions
- **Classes:** 38 disease categories
- **Language:** English
- **Format:** Parquet file

---

## 3. Model Architecture

### 🏗️ Dual-Model System

```
┌─────────────────────────────────────────────────────┐
│              PlantDocBot Architecture                │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────┐              ┌──────────────┐    │
│  │  Image Input │              │  Text Input  │    │
│  └──────┬───────┘              └──────┬───────┘    │
│         │                              │            │
│         ▼                              ▼            │
│  ┌──────────────┐              ┌──────────────┐    │
│  │  CNN Model   │              │ BERT Model   │    │
│  │  (PyTorch)   │              │(Transformers)│    │
│  │              │              │              │    │
│  │ 3 Conv Layers│              │ 12 Layers    │    │
│  │ 2 FC Layers  │              │ 768 Hidden   │    │
│  │ 38 Classes   │              │ 38 Classes   │    │
│  └──────┬───────┘              └──────┬───────┘    │
│         │                              │            │
│         ▼                              ▼            │
│  ┌──────────────────────────────────────────┐      │
│  │         Disease Prediction               │      │
│  │    (Label + Confidence Score)            │      │
│  └──────────────────────────────────────────┘      │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

## 4. CNN Implementation

### 📓 Notebook: `ImageClassification.ipynb`

### 🧱 Model Architecture

```python
class PlantDiseaseModel(torch.nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseModel, self).__init__()
        
        # Convolutional Layers
        self.conv_layers = torch.nn.Sequential(
            # Layer 1: 3 → 32 channels
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # 224→112
            
            # Layer 2: 32 → 64 channels
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # 112→56
            
            # Layer 3: 64 → 128 channels
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)   # 56→28
        )
        
        # Fully Connected Layers
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
```

### 📐 Architecture Details

**Input:** 224×224×3 RGB Image

**Convolutional Layers:**
```
Layer 1: Conv2d(3→32) + ReLU + MaxPool → 112×112×32
Layer 2: Conv2d(32→64) + ReLU + MaxPool → 56×56×64
Layer 3: Conv2d(64→128) + ReLU + MaxPool → 28×28×128
```

**Fully Connected Layers:**
```
Flatten: 28×28×128 → 100,352 features
FC1: 100,352 → 256 + ReLU
FC2: 256 → 38 (output classes)
```

**Total Parameters:** ~25.8 million

### 🔄 Data Preprocessing

```python
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4760, 0.5004, 0.4266],
        std=[0.1775, 0.1509, 0.1960]
    )
])
```

### 🎓 Training Configuration

```python
# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
OPTIMIZER = Adam
LOSS_FUNCTION = CrossEntropyLoss

# Data Augmentation
transforms.RandomHorizontalFlip()
transforms.RandomRotation(10)
transforms.ColorJitter(brightness=0.2)
```

---

## 5. Transformer Implementation

### 📓 Notebook: `TextClassifier.ipynb`

### 🤖 Model: BERT for Sequence Classification

**Base Model:** `bert-base-uncased`

**Architecture:**
- **Embedding Dimension:** 768
- **Attention Heads:** 12
- **Transformer Layers:** 12
- **Total Parameters:** ~110 million
- **Output Classes:** 38

### 🔧 Implementation Steps

#### Step 1: Load Tokenizer
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
```

#### Step 2: Tokenize Dataset
```python
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

#### Step 3: Initialize Model
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=38
)
```

#### Step 4: Training Configuration
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch',
    load_best_model_at_end=True
)
```

#### Step 5: Data Collator
```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

#### Step 6: Train Model
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

trainer.train()
```

### 🎯 Using Hugging Face

**Why Hugging Face?**
- Pre-trained BERT models
- Easy fine-tuning API
- Tokenizer management
- Model serialization
- Community datasets

**Key Libraries:**
```python
transformers==4.36.0  # Main library
torch==2.9.0          # Backend
datasets              # Dataset loading
```

---

## 6. Model Training & Weights

### 💾 Saving Model Weights

#### CNN Model (PyTorch)
```python
# Save model weights
torch.save(
    cnn.state_dict(),
    'models/ImageClassification_model_weights.pth'
)

# File size: ~103 MB
```

#### BERT Model (Hugging Face)
```python
# Save model and tokenizer
model.save_pretrained('./models/text_classifier_model')
tokenizer.save_pretrained('./models/text_classifier_model')

# Save label encoder
import joblib
joblib.dump(encoder, './models/text_classifier_model/label_encoder.pkl')
```

### 📁 Model Files Structure

```
Backend/models/
├── ImageClassification_model_weights.pth  (~103 MB)
└── text_classifier_model/
    ├── config.json                 # Model configuration
    ├── model.safetensors          # Model weights (~440 MB)
    ├── tokenizer_config.json      # Tokenizer config
    ├── vocab.txt                  # BERT vocabulary
    ├── special_tokens_map.json    # Special tokens
    └── label_encoder.pkl          # Label mapping
```

### 📥 Loading Weights in Production

```python
# Load CNN weights
cnn = PlantDiseaseModel(num_classes=38)
cnn.load_state_dict(
    torch.load('models/ImageClassification_model_weights.pth',
               map_location=device)
)
cnn.eval()

# Load BERT model
tokenizer = AutoTokenizer.from_pretrained(
    './models/text_classifier_model',
    local_files_only=True
)
text_model = AutoModelForSequenceClassification.from_pretrained(
    './models/text_classifier_model',
    local_files_only=True
)
text_model.eval()
```

---

## 7. FastAPI Backend

### 🚀 API Structure

**File:** `Backend/main.py`

### 🔌 API Endpoints

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PlantDocBot API 🌿",
    version="1.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

### 📍 Endpoint 1: Health Check

```python
@app.get("/health-check")
def health_check():
    return {
        "status": "ok",
        "message": "PlantDocBot API is running 🚀"
    }
```

### 📍 Endpoint 2: Image Prediction

```python
@app.post("/image-prediction")
async def image_predict(file: UploadFile = File(...)):
    # Read image
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    
    # Preprocess
    img_tensor = image_transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = cnn(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
    
    # Return result
    return {
        "label": class_names[pred_class.item()],
        "confidence": round(confidence.item(), 4),
        "recommendation": get_recommendation(label)
    }
```

### 📍 Endpoint 3: Text Prediction

```python
@app.post("/text-prediction")
def text_predict_endpoint(input_data: TextPredictionInputModel):
    # Tokenize
    inputs = tokenizer(
        input_data.input,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        confidence, pred_id = torch.max(probs, dim=-1)
    
    # Decode label
    label = encoder.inverse_transform([pred_id.item()])[0]
    
    return {
        "label": label,
        "confidence": round(confidence.item(), 4),
        "recommendation": get_recommendation(label)
    }
```

### 📍 Endpoint 4: Chatbot

```python
@app.post("/chatbot")
async def chatbot_endpoint(chat_request: ChatRequest):
    # Configure Gemini AI
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
    
    # Generate response
    response = model.generate_content(chat_request.message)
    
    return {
        "response": response.text,
        "status": "success"
    }
```

### 🏃 Running FastAPI

```bash
# Start server
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Access API docs
http://localhost:8000/docs
```

---

## 8. React Frontend

### ⚛️ Technology Stack

**Framework:** React 19.1.1  
**Build Tool:** Vite 7.1.7  
**HTTP Client:** Axios 1.13.1  
**Styling:** Vanilla CSS

### 📦 Dependencies

```json
{
  "dependencies": {
    "axios": "^1.13.1",
    "react": "^19.1.1",
    "react-dom": "^19.1.1"
  }
}
```

### 🎨 Component Structure

**File:** `src/App.jsx`

```jsx
function App() {
  // State management
  const [imageResponse, setImageResponse] = useState(null);
  const [textResponse, setTextResponse] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  
  return (
    <div className="app-card">
      {/* Header */}
      <Header />
      
      {/* Image Analysis Section */}
      <ImageAnalysis />
      
      {/* Text Diagnosis Section */}
      <TextDiagnosis />
      
      {/* AI Chatbot Section */}
      <Chatbot />
    </div>
  );
}
```

### 🖼️ Image Upload Handler

```jsx
const handleImageSubmit = async (e) => {
  e.preventDefault();
  const file = e.target.elements.image.files[0];
  
  const formData = new FormData();
  formData.append('file', file);
  
  setLoading(true);
  try {
    const res = await axios.post(
      'http://localhost:8000/image-prediction',
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
    setImageResponse(res.data);
  } catch (err) {
    setImageResponse({ error: err.message });
  }
  setLoading(false);
};
```

### 💬 Text Input Handler

```jsx
const handleTextSubmit = async (e) => {
  e.preventDefault();
  const text = e.target.elements.input.value.trim();
  
  setLoading(true);
  try {
    const res = await axios.post(
      'http://localhost:8000/text-prediction',
      { input: text }
    );
    setTextResponse(res.data);
  } catch (err) {
    setTextResponse({ error: err.message });
  }
  setLoading(false);
};
```

### 🤖 Chatbot Handler

```jsx
const handleChatSubmit = async (e) => {
  e.preventDefault();
  
  const userMessage = { role: 'user', content: chatInput };
  setChatMessages(prev => [...prev, userMessage]);
  
  const res = await axios.post('http://localhost:8000/chatbot', {
    message: chatInput,
    conversation_history: chatMessages
  });
  
  const botMessage = { role: 'assistant', content: res.data.response };
  setChatMessages(prev => [...prev, botMessage]);
};
```

### 🎨 UI Features

- **Glassmorphism Design**
- **Smooth Animations**
- **Responsive Layout**
- **Real-time Updates**
- **Loading States**
- **Error Handling**

---

## 9. Data Flow & Processing

### 🔄 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                          │
└────────────┬────────────────────────────┬───────────────────┘
             │                            │
             ▼                            ▼
    ┌────────────────┐          ┌────────────────┐
    │  Upload Image  │          │  Enter Text    │
    │   (JPG/PNG)    │          │  (Symptoms)    │
    └────────┬───────┘          └────────┬───────┘
             │                            │
             ▼                            ▼
    ┌────────────────┐          ┌────────────────┐
    │ React Frontend │          │ React Frontend │
    │  (Axios POST)  │          │  (Axios POST)  │
    └────────┬───────┘          └────────┬───────┘
             │                            │
             ▼                            ▼
    ┌────────────────────────────────────────────┐
    │         FastAPI Backend (Port 8000)        │
    │  /image-prediction    /text-prediction     │
    └────────┬───────────────────────┬───────────┘
             │                       │
             ▼                       ▼
    ┌────────────────┐      ┌────────────────┐
    │ Image Process  │      │ Text Tokenize  │
    │ - Resize 224²  │      │ - BERT Tokens  │
    │ - Normalize    │      │ - Padding      │
    │ - To Tensor    │      │ - Truncation   │
    └────────┬───────┘      └────────┬───────┘
             │                       │
             ▼                       ▼
    ┌────────────────┐      ┌────────────────┐
    │   CNN Model    │      │  BERT Model    │
    │  (PyTorch)     │      │ (Transformers) │
    │  Forward Pass  │      │  Forward Pass  │
    └────────┬───────┘      └────────┬───────┘
             │                       │
             ▼                       ▼
    ┌────────────────┐      ┌────────────────┐
    │  Softmax       │      │  Softmax       │
    │  Get Max Prob  │      │  Get Max Prob  │
    └────────┬───────┘      └────────┬───────┘
             │                       │
             ▼                       ▼
    ┌────────────────────────────────────────────┐
    │          JSON Response                      │
    │  {                                          │
    │    "label": "Disease Name",                 │
    │    "confidence": 0.95,                      │
    │    "recommendation": "Treatment advice"     │
    │  }                                          │
    └────────┬───────────────────────────────────┘
             │
             ▼
    ┌────────────────┐
    │ Display Result │
    │  in Frontend   │
    └────────────────┘
```

### 📊 Processing Steps

#### Image Processing Pipeline
```
1. User uploads image → FormData
2. Frontend sends to /image-prediction
3. Backend receives file
4. PIL opens image → RGB conversion
5. Transform: Resize(224,224) → ToTensor → Normalize
6. Add batch dimension: (1, 3, 224, 224)
7. Move to device (CPU/GPU)
8. CNN forward pass
9. Softmax activation
10. Get argmax for prediction
11. Map to class name
12. Return JSON response
```

#### Text Processing Pipeline
```
1. User enters symptoms → String
2. Frontend sends to /text-prediction
3. Backend receives text
4. Tokenizer processes text
5. Convert to token IDs
6. Add padding/truncation
7. Create attention masks
8. Move to device (CPU/GPU)
9. BERT forward pass
10. Softmax on logits
11. Get argmax for prediction
12. Inverse transform label
13. Return JSON response
```

---

## 10. System Architecture

### 🏛️ Complete System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     PLANTDOCBOT SYSTEM                         │
├───────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              FRONTEND LAYER (Port 5173)              │      │
│  │  ┌─────────────────────────────────────────────┐    │      │
│  │  │         React 19 + Vite 7                   │    │      │
│  │  │  - App.jsx (Main Component)                 │    │      │
│  │  │  - State Management (useState)              │    │      │
│  │  │  - Axios HTTP Client                        │    │      │
│  │  │  - CSS Styling (Glassmorphism)              │    │      │
│  │  └─────────────────────────────────────────────┘    │      │
│  └────────────────────┬────────────────────────────────┘      │
│                       │ HTTP Requests                          │
│                       │ (Axios)                                │
│                       ▼                                        │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              BACKEND LAYER (Port 8000)               │      │
│  │  ┌─────────────────────────────────────────────┐    │      │
│  │  │         FastAPI Framework                   │    │      │
│  │  │  - CORS Middleware                          │    │      │
│  │  │  - 4 API Endpoints                          │    │      │
│  │  │  - Request Validation (Pydantic)            │    │      │
│  │  │  - Error Handling                           │    │      │
│  │  └─────────────────────────────────────────────┘    │      │
│  └────────────────────┬────────────────────────────────┘      │
│                       │                                        │
│                       ▼                                        │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              AI MODEL LAYER                          │      │
│  │  ┌──────────────────┐    ┌──────────────────┐      │      │
│  │  │   CNN Model      │    │   BERT Model     │      │      │
│  │  │   (PyTorch)      │    │ (Transformers)   │      │      │
│  │  │                  │    │                  │      │      │
│  │  │  - 3 Conv Layers │    │  - 12 Layers     │      │      │
│  │  │  - 2 FC Layers   │    │  - 768 Hidden    │      │      │
│  │  │  - 38 Classes    │    │  - 38 Classes    │      │      │
│  │  │  - ~26M params   │    │  - ~110M params  │      │      │
│  │  └──────────────────┘    └──────────────────┘      │      │
│  │                                                      │      │
│  │  ┌──────────────────────────────────────────┐      │      │
│  │  │      Google Gemini AI (Chatbot)          │      │      │
│  │  │      - gemini-pro model                  │      │      │
│  │  │      - API Key authentication            │      │      │
│  │  └──────────────────────────────────────────┘      │      │
│  └─────────────────────────────────────────────────────┘      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              DATA LAYER                              │      │
│  │  ┌──────────────────┐    ┌──────────────────┐      │      │
│  │  │  Model Weights   │    │  Configurations  │      │      │
│  │  │                  │    │                  │      │      │
│  │  │  - CNN .pth      │    │  - .env file     │      │      │
│  │  │  - BERT model    │    │  - API keys      │      │      │
│  │  │  - Tokenizer     │    │  - Class labels  │      │      │
│  │  │  - Label encoder │    │  - Normalization │      │      │
│  │  └──────────────────┘    └──────────────────┘      │      │
│  └─────────────────────────────────────────────────────┘      │
│                                                                 │
└───────────────────────────────────────────────────────────────┘
```

### 🔐 Technology Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | React 19 + Vite | User interface |
| **HTTP Client** | Axios | API communication |
| **Backend** | FastAPI | REST API server |
| **Image Model** | PyTorch CNN | Image classification |
| **Text Model** | BERT (Hugging Face) | Text classification |
| **Chatbot** | Google Gemini AI | Conversational AI |
| **Deployment** | Uvicorn | ASGI server |

---

## 📝 Summary

### ✅ What We Built

1. **Dataset Collection**
   - PlantVillage images (54K+ images)
   - Hugging Face text dataset (10K+ descriptions)

2. **Model Development**
   - Custom CNN (95% accuracy)
   - Fine-tuned BERT (92% accuracy)

3. **Backend API**
   - FastAPI with 4 endpoints
   - Model loading and inference
   - CORS enabled

4. **Frontend Interface**
   - React with modern UI
   - Three interaction modes
   - Real-time results

5. **Complete System**
   - End-to-end disease detection
   - Dual-model predictions
   - AI-powered assistance

### 🎯 Key Achievements

- ✅ 38 plant disease classes
- ✅ Dual-model architecture
- ✅ 95% image accuracy
- ✅ 92% text accuracy
- ✅ Real-time predictions
- ✅ Modern responsive UI
- ✅ Production-ready API

---

**Project Repository:** [GitHub - PlantDocBot](https://github.com/karansrabcd01/AI_plantdocbot)

**Made with 🌿 and 🤖**
