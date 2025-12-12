# 📡 PlantDocBot API Documentation

Complete API reference for the PlantDocBot backend server.

---

## 🌐 Base URL

```
http://localhost:8000
```

For production, replace with your deployed backend URL.

---

## 📚 Table of Contents

- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Image Prediction](#image-prediction)
  - [Text Prediction](#text-prediction)
  - [Chatbot](#chatbot)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

---

## 🔐 Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

**Note**: For production deployment, consider adding API key authentication.

---

## 📍 Endpoints

### **Health Check**

Check if the API server is running.

#### **Request**
```http
GET /health-check
```

#### **Response**
```json
{
  "status": "ok",
  "message": "PlantDocBot API is running 🚀"
}
```

#### **Status Codes**
- `200 OK` - Server is running

#### **Example**
```bash
curl http://localhost:8000/health-check
```

---

### **Image Prediction**

Upload a plant image for disease detection.

#### **Request**
```http
POST /image-prediction
Content-Type: multipart/form-data
```

#### **Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Plant image (JPEG, PNG, JPG) |

#### **Response**
```json
{
  "filename": "tomato_leaf.jpg",
  "label": "Tomato - Early Blight",
  "confidence": 0.9542,
  "recommendation": "Your plant seems affected. Consider proper diagnosis and treatment."
}
```

#### **Response Fields**
| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Original filename |
| `label` | string | Predicted disease class |
| `confidence` | float | Confidence score (0-1) |
| `recommendation` | string | Treatment recommendation |

#### **Status Codes**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid file format
- `500 Internal Server Error` - Processing error

#### **Example (cURL)**
```bash
curl -X POST http://localhost:8000/image-prediction \
  -F "file=@/path/to/plant_image.jpg"
```

#### **Example (Python)**
```python
import requests

url = "http://localhost:8000/image-prediction"
files = {"file": open("plant_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

#### **Example (JavaScript)**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/image-prediction', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### **Text Prediction**

Predict plant disease from text description of symptoms.

#### **Request**
```http
POST /text-prediction
Content-Type: application/json
```

#### **Request Body**
```json
{
  "input": "Yellow spots on leaves with brown edges and wilting"
}
```

#### **Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string | Yes | Description of plant symptoms |

#### **Response**
```json
{
  "input_text": "Yellow spots on leaves with brown edges and wilting",
  "label": "Tomato - Septoria Leaf Spot",
  "confidence": 0.8734,
  "recommendation": "Your plant seems affected. Consider proper diagnosis and treatment."
}
```

#### **Response Fields**
| Field | Type | Description |
|-------|------|-------------|
| `input_text` | string | Original input text |
| `label` | string | Predicted disease class |
| `confidence` | float | Confidence score (0-1) |
| `recommendation` | string | Treatment recommendation |

#### **Status Codes**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid input
- `500 Internal Server Error` - Processing error

#### **Example (cURL)**
```bash
curl -X POST http://localhost:8000/text-prediction \
  -H "Content-Type: application/json" \
  -d '{"input": "Yellow spots on leaves"}'
```

#### **Example (Python)**
```python
import requests

url = "http://localhost:8000/text-prediction"
data = {"input": "Yellow spots on leaves with brown edges"}
response = requests.post(url, json=data)
print(response.json())
```

#### **Example (JavaScript)**
```javascript
fetch('http://localhost:8000/text-prediction', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    input: 'Yellow spots on leaves with brown edges'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### **Chatbot**

Interactive AI assistant for plant care advice.

#### **Request**
```http
POST /chatbot
Content-Type: application/json
```

#### **Request Body**
```json
{
  "message": "How often should I water tomatoes?",
  "conversation_history": [
    {
      "role": "user",
      "content": "I have a tomato plant"
    },
    {
      "role": "assistant",
      "content": "Great! Tomatoes are wonderful to grow..."
    }
  ]
}
```

#### **Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | Yes | User's question or message |
| `conversation_history` | array | No | Previous conversation messages |

#### **Conversation History Format**
```json
[
  {
    "role": "user" | "assistant",
    "content": "message text"
  }
]
```

#### **Response**
```json
{
  "response": "💧 Watering tips: Most plants need water when the top 1-2 inches of soil feels dry. Overwatering is more harmful than underwatering. Ensure pots have drainage holes. Water deeply but less frequently to encourage strong root growth.",
  "status": "success"
}
```

#### **Response Fields**
| Field | Type | Description |
|-------|------|-------------|
| `response` | string | Bot's response message |
| `status` | string | Response status (success, loading, error) |

#### **Status Codes**
- `200 OK` - Response generated
- `400 Bad Request` - Invalid input
- `500 Internal Server Error` - Processing error

#### **Knowledge Base Topics**
The chatbot has built-in knowledge about:
- Watering schedules
- Yellow/brown leaves diagnosis
- Sunlight requirements
- Fertilizing guidance
- Pest control
- Disease prevention
- Soil management
- Pruning techniques
- Repotting
- Specific plants (tomatoes, indoor plants, etc.)

#### **Example (cURL)**
```bash
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How to water plants?",
    "conversation_history": []
  }'
```

#### **Example (Python)**
```python
import requests

url = "http://localhost:8000/chatbot"
data = {
    "message": "How often should I water tomatoes?",
    "conversation_history": []
}
response = requests.post(url, json=data)
print(response.json())
```

#### **Example (JavaScript)**
```javascript
fetch('http://localhost:8000/chatbot', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'How to prevent pests?',
    conversation_history: []
  })
})
.then(response => response.json())
.then(data => console.log(data.response));
```

---

## ⚠️ Error Handling

### **Error Response Format**
```json
{
  "error": "Error description"
}
```

### **Common Errors**

#### **400 Bad Request**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "input"],
      "msg": "Field required"
    }
  ]
}
```

#### **500 Internal Server Error**
```json
{
  "error": "Failed to process image: Invalid image format"
}
```

### **Error Handling Best Practices**

```javascript
// JavaScript example
try {
  const response = await fetch('http://localhost:8000/image-prediction', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  
  if (data.error) {
    console.error('API Error:', data.error);
    // Handle error
  } else {
    console.log('Success:', data);
    // Process result
  }
} catch (error) {
  console.error('Network Error:', error);
  // Handle network error
}
```

---

## 🚦 Rate Limiting

Currently, there are no rate limits on the API.

**For production:**
- Consider implementing rate limiting
- Recommended: 100 requests per minute per IP
- Use libraries like `slowapi` for FastAPI

---

## 📝 Examples

### **Complete Workflow Example**

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Check API health
health = requests.get(f"{BASE_URL}/health-check")
print("Health:", health.json())

# 2. Analyze image
with open("plant_image.jpg", "rb") as f:
    files = {"file": f}
    image_result = requests.post(f"{BASE_URL}/image-prediction", files=files)
    print("Image Analysis:", image_result.json())

# 3. Text diagnosis
text_data = {"input": "Yellow spots on leaves"}
text_result = requests.post(f"{BASE_URL}/text-prediction", json=text_data)
print("Text Diagnosis:", text_result.json())

# 4. Chat with bot
chat_data = {
    "message": "How to treat early blight?",
    "conversation_history": []
}
chat_result = requests.post(f"{BASE_URL}/chatbot", json=chat_data)
print("Chatbot:", chat_result.json())
```

### **Batch Processing Example**

```python
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
image_dir = Path("plant_images/")

results = []
for image_path in image_dir.glob("*.jpg"):
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/image-prediction", files=files)
        result = response.json()
        results.append({
            "filename": image_path.name,
            "prediction": result
        })

# Save results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 🔧 Interactive API Documentation

FastAPI provides automatic interactive API documentation:

### **Swagger UI**
Visit: `http://localhost:8000/docs`

Features:
- Try out endpoints directly in browser
- See request/response schemas
- Download OpenAPI specification

### **ReDoc**
Visit: `http://localhost:8000/redoc`

Features:
- Alternative documentation interface
- Better for reading and sharing
- Cleaner layout

---

## 📊 Response Times

Typical response times (on average hardware):

| Endpoint | Average Time | Notes |
|----------|-------------|-------|
| `/health-check` | < 10ms | Simple status check |
| `/image-prediction` | 200-500ms | Depends on image size |
| `/text-prediction` | 100-300ms | Depends on text length |
| `/chatbot` | 50-200ms | Knowledge base responses |
| `/chatbot` (Gemini) | 1-3s | When using Gemini API |

---

## 🔄 Versioning

Current API version: **v1.0**

Future versions will be accessible via URL prefix:
- v1: `http://localhost:8000/api/v1/`
- v2: `http://localhost:8000/api/v2/`

---

## 📞 Support

For API issues or questions:
- Check the [Setup Guide](SETUP_GUIDE.md)
- Review [Troubleshooting](SETUP_GUIDE.md#troubleshooting)
- Open an issue on GitHub

---

**Last Updated**: December 2025
**API Version**: 1.0
