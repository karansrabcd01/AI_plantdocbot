from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F
from torchvision import transforms
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# ------------------------------------------------------
# FastAPI Initialization
# ------------------------------------------------------
app = FastAPI(
    title="PlantDocBot API 🌿",
    description="API for Plant Disease Detection using Image and Text Models",
    version="1.0"
)

# ------------------------------------------------------
# CORS Configuration (to connect with React Frontend)
# ------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use ["http://localhost:5173"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# Device Configuration
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------
# Image Classification Model
# ------------------------------------------------------
class PlantDiseaseModel(torch.nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseModel, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
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


# Load Image Model
cnn = PlantDiseaseModel(num_classes=38)
image_model_path = BASE_DIR / "models" / "ImageClassification_model_weights.pth"

try:
    cnn.load_state_dict(torch.load(str(image_model_path), map_location=device))
    cnn.to(device)
    cnn.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load image model from {image_model_path}: {e}")

# Image Preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4760, 0.5004, 0.4266],
                         std=[0.1775, 0.1509, 0.1960])
])

# Class Labels
class_names = [
    "Apple - Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery Mildew", "Cherry - Healthy",
    "Corn (Maize) - Cercospora Leaf Spot (Gray Leaf Spot)", "Corn (Maize) - Common Rust",
    "Corn (Maize) - Northern Leaf Blight", "Corn (Maize) - Healthy",
    "Grape - Black Rot", "Grape - Esca (Black Measles)", "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape - Healthy", "Orange - Huanglongbing (Citrus Greening)", "Peach - Bacterial Spot",
    "Peach - Healthy", "Pepper (Bell) - Bacterial Spot", "Pepper (Bell) - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Raspberry - Healthy", "Soybean - Healthy", "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch", "Strawberry - Healthy", "Tomato - Bacterial Spot",
    "Tomato - Early Blight", "Tomato - Late Blight", "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato - Target Spot", "Tomato - Mosaic Virus", "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Healthy"
]

# ------------------------------------------------------
# Text Classification Model
# ------------------------------------------------------
text_model_path = BASE_DIR / "models" / "text_classifier_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(str(text_model_path), local_files_only=True)
    text_model = AutoModelForSequenceClassification.from_pretrained(
        str(text_model_path), local_files_only=True
    ).to(device)
    text_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load text model/tokenizer from {text_model_path}: {e}")

# Load Label Encoder
label_encoder_path = text_model_path / "label_encoder.pkl"
try:
    encoder = joblib.load(str(label_encoder_path))
except Exception as e:
    raise RuntimeError(f"Failed to load label encoder from {label_encoder_path}: {e}")


def predict_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        confidence, pred_id = torch.max(probs, dim=-1)

    label = encoder.inverse_transform([pred_id.item()])[0]
    confidence_score = round(confidence.item(), 4)
    recommendation = (
        "Your plant looks healthy 🌱"
        if "healthy" in label.lower()
        else "Your plant seems affected. Consider proper diagnosis and treatment."
    )

    return {"label": label, "confidence": confidence_score, "recommendation": recommendation}


class TextPredictionInputModel(BaseModel):
    input: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []



# ------------------------------------------------------
# API Endpoints
# ------------------------------------------------------
@app.get("/health-check")
def health_check():
    return {"status": "ok", "message": "PlantDocBot API is running 🚀"}


@app.post("/image-prediction")
async def image_predict(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
        img_tensor = image_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = cnn(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

        label = class_names[pred_class.item()]
        confidence_score = round(confidence.item(), 4)
        recommendation = (
            "Your plant looks healthy 🌱"
            if "healthy" in label.lower()
            else "Your plant seems affected. Consider proper diagnosis and treatment."
        )

        return {
            "filename": file.filename,
            "label": label,
            "confidence": confidence_score,
            "recommendation": recommendation
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/text-prediction")
def text_predict_endpoint(input_data: TextPredictionInputModel):
    try:
        result = predict_text(input_data.input)
        return {"input_text": input_data.input, **result}
    except Exception as e:
        return {"error": str(e)}


@app.post("/chatbot")
async def chatbot_endpoint(chat_request: ChatRequest):
    """
    Plant care chatbot powered by Google Gemini AI
    Falls back to knowledge base if API is unavailable
    """
    try:
        # Try to use Gemini API first
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if gemini_api_key:
            try:
                # Configure Gemini
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(
                    'gemini-pro',
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'top_k': 40,
                        'max_output_tokens': 500,
                    }
                )
                
                # Build conversation context
                conversation_context = ""
                if chat_request.conversation_history:
                    for msg in chat_request.conversation_history[-5:]:
                        role = "User" if msg.get("role") == "user" else "Assistant"
                        content = msg.get("content", "")
                        conversation_context += f"{role}: {content}\n"
                
                # Create plant-focused prompt
                system_prompt = """You are PlantDocBot, an expert AI assistant specializing in plant care, plant diseases, and gardening advice. 
You provide helpful, accurate, and friendly advice about:
- Plant disease identification and treatment
- Watering, fertilizing, and general plant care
- Indoor and outdoor gardening
- Pest control
- Soil and repotting
- Pruning techniques
- Specific plant species care

Keep your responses concise (2-3 paragraphs max), practical, and easy to understand. Use emojis when appropriate to make responses friendly."""

                full_prompt = f"{system_prompt}\n\n{conversation_context}User: {chat_request.message}\nAssistant:"
                
                # Generate response with timeout handling
                import asyncio
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(model.generate_content, full_prompt),
                        timeout=10.0  # 10 second timeout
                    )
                    
                    if response and response.text:
                        return {
                            "response": response.text,
                            "status": "success"
                        }
                except asyncio.TimeoutError:
                    print("Gemini API timeout - falling back to knowledge base")
                except Exception as api_error:
                    print(f"Gemini API error: {api_error}")
                # Fall through to knowledge base
            except Exception as e:
                print(f"Gemini configuration error: {e}")
                # Fall through to knowledge base
        
        # Fallback: Knowledge base for common questions
        message_lower = chat_request.message.lower()
        
        plant_knowledge = {
            "water": "💧 Watering tips: Most plants need water when the top 1-2 inches of soil feels dry. Overwatering is more harmful than underwatering. Ensure pots have drainage holes. Water deeply but less frequently to encourage strong root growth.",
            
            "yellow": "🍂 Yellow leaves can indicate: 1) Overwatering (most common), 2) Nutrient deficiency (especially nitrogen), 3) Poor drainage, 4) Natural aging of lower leaves. Check soil moisture and adjust watering schedule.",
            
            "sunlight": "☀️ Light requirements vary by plant: Full sun (6+ hours), Partial sun (3-6 hours), Shade (less than 3 hours). Most houseplants prefer bright, indirect light. Rotate plants weekly for even growth.",
            
            "fertilizer": "🌱 Fertilizer guide: Use balanced fertilizer (10-10-10) during growing season (spring/summer). Dilute to half strength. Fertilize every 2-4 weeks. Reduce or stop in fall/winter when growth slows.",
            
            "pest": "🐛 Common pests: Aphids, spider mites, mealybugs, scale. Solutions: Neem oil spray, insecticidal soap, or mix 1 tsp dish soap in 1 quart water. Spray weekly until pests are gone. Isolate infected plants.",
            
            "disease": "🦠 Common diseases: Fungal spots, powdery mildew, root rot. Prevention: Good air circulation, avoid overhead watering, remove infected leaves. Treatment: Fungicide spray, improve drainage, reduce humidity.",
            
            "soil": "🌍 Good soil should: Drain well, retain some moisture, contain organic matter. Mix potting soil with perlite or vermiculite for better drainage. Repot every 1-2 years with fresh soil.",
            
            "prune": "✂️ Pruning tips: Remove dead/diseased parts immediately. Prune for shape in early spring. Use clean, sharp tools. Cut just above a leaf node at 45° angle. Don't remove more than 1/3 of plant at once.",
            
            "repot": "🪴 Repotting guide: Repot when roots circle the pot or grow through drainage holes. Choose pot 1-2 inches larger. Best time: early spring. Water thoroughly after repotting. Expect some shock initially.",
            
            "tomato": "🍅 Tomato care: Full sun (6-8 hours), deep watering, support with stakes/cages. Common issues: blight (remove infected leaves), blossom end rot (calcium deficiency - add crushed eggshells).",
            
            "indoor": "🏠 Indoor plant tips: Most prefer bright indirect light, humidity 40-60%, temps 65-75°F. Popular easy plants: pothos, snake plant, spider plant, ZZ plant. Avoid cold drafts and heating vents.",
            
            "outdoor": "🌳 Outdoor gardening: Know your hardiness zone, plant after last frost, mulch to retain moisture, companion planting helps pest control. Water early morning to reduce disease.",
        }
        
        # Check for keywords
        for keyword, response_text in plant_knowledge.items():
            if keyword in message_lower:
                return {
                    "response": response_text,
                    "status": "success"
                }
        
        # Default response
        return {
            "response": f"🌿 I'm PlantDocBot, your plant care expert! I can help you with:\n\n• Watering schedules and techniques\n• Yellow/brown leaves diagnosis\n• Sunlight requirements\n• Fertilizing guidance\n• Pest and disease control\n• Soil and repotting advice\n• Pruning techniques\n• Specific plants (tomatoes, indoor plants, etc.)\n\nTo get better responses, please add your **GEMINI_API_KEY** to the .env file. You can get a free API key at: https://makersuite.google.com/app/apikey\n\nWhat would you like to know about plant care?",
            "status": "success"
        }
            
    except Exception as e:
        return {
            "response": "I'm here to help with plant care! Ask me about watering, sunlight, pests, diseases, or any plant topic.",
            "status": "success"
        }

