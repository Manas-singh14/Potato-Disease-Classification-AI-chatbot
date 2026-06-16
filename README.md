# 🌿 Potato Disease Classification & AI Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-API-F55036?style=for-the-badge&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white)

**An end-to-end AI-powered potato disease detection system with an integrated LLM chatbot.**
Upload a leaf image → get instant diagnosis → chat with an AI plant pathologist for treatment advice.

[🔴 Live Demo](#-live-demo) · [✨ Features](#-features) · [🛠 Tech Stack](#-tech-stack) · [🚀 Getting Started](#-getting-started) · [📡 API](#-api-endpoints)

</div>

---

## 🔴 Live Demo

| Service | URL |
|---|---|
| 🌐 Frontend | https://potato-disease-classification-ai-chatbot.onrender.com |
| ⚙️ Backend API | https://plantscan-3ct6.onrender.com |
| 📖 API Docs | https://plantscan-3ct6.onrender.com/docs |

> ⚠️ Hosted on Render free tier — first request may take 30–60 seconds to wake up.

---

## 🎯 What is this project?

This is a full-stack deep learning project that detects diseases in potato leaves using a custom-trained Convolutional Neural Network (CNN). It features:

- A **REST API** built with FastAPI that serves predictions from the trained model
- An **interactive web frontend** with real-time diagnosis and probability visualization
- An **AI chatbot** powered by Groq (Llama3) via LangChain that knows your scan result and gives specific treatment advice
- Full **Docker containerization** and deployment on **Render**

---

## ✨ Features

### 🔬 Disease Detection
- Custom CNN model trained on PlantVillage dataset
- **95% accuracy** across 3 classes
- Animated confidence bars showing probability for each class
- Treatment recommendations shown instantly after diagnosis
- Scan history with thumbnails (last 5 scans)

### 🤖 AI Chatbot
- Powered by **Groq API** (Llama3-8b) via **LangChain**
- Bot automatically knows your current scan result
- Ask follow-up questions — *"How do I treat this?"*, *"Will it spread?"*
- **Conversation memory** — remembers full chat per browser session
- **Streaming responses** — words appear one by one like ChatGPT
- Chat history saved to localStorage — persists on page refresh
- Multiple model options — Llama3, Mixtral, Gemma2

### 🎨 Frontend
- Drag & drop image upload
- Live API status indicator
- Responsive design — works on mobile and desktop
- No framework — pure HTML, CSS, JavaScript

---

## 🧠 Detected Classes

| Class | Description | Severity |
|---|---|---|
| 🥬 **Potato — Healthy** | No disease detected | ✅ None |
| 🍂 **Potato — Early Blight** | Alternaria solani fungus | ⚠️ Moderate |
| 🟤 **Potato — Late Blight** | Phytophthora infestans | 🚨 Critical |

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **ML Model** | TensorFlow / Keras | Custom CNN for disease classification |
| **Dataset** | PlantVillage | Training data (3 classes) |
| **Backend** | FastAPI + Uvicorn | REST API server |
| **LLM** | Groq API (Llama3-8b) | Powers the AI chatbot |
| **LLM Framework** | LangChain | Memory, prompt templates, chain management |
| **Memory** | InMemoryChatMessageHistory | Per-session conversation history |
| **Frontend** | HTML + CSS + JavaScript | User interface |
| **Streaming** | Server-Sent Events (SSE) | Real-time token streaming |
| **Containerization** | Docker | Packaging and deployment |
| **Hosting** | Render | Backend (Docker) + Frontend (Static) |
| **Image Registry** | Docker Hub | Stores Docker image |

---

## 📁 Project Structure

```
Potato-Disease-Classification-AI-chatbot/
│
├── backend_final.py          # FastAPI app — /predict and /chat routes
├── Dockerfile                # Docker container configuration
├── requirements.txt          # Python dependencies
├── .dockerignore             # Files excluded from Docker image
├── .gitignore                # Files excluded from git
│
├── frontend/
│   └── index.html            # Frontend — diagnosis UI + chatbot
│
├── chat/
│   ├── __init__.py
│   ├── chain.py              # LangChain chain with Groq LLM
│   ├── chain_ollama.py       # Alternative chain using local Ollama
│   ├── memory.py             # Per-session conversation memory
│   └── prompt.py             # Prompt templates
│
└── models/
    └── new_best_model.keras  # Trained CNN model (not in repo — too large)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com) (free)
- Docker Desktop (for containerized deployment)

### 1. Clone the repository
```bash
git clone https://github.com/Manas-singh14/Potato-Disease-Classification-AI-chatbot.git
cd Potato-Disease-Classification-AI-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 5. Add your trained model
Place your trained model at:
```
models/new_best_model.keras
```

### 6. Run the backend
```bash
uvicorn backend_final:app --reload
```

### 7. Open the frontend
Open `frontend/index.html` in your browser.

---

## 🐳 Docker Deployment

### Build and run locally
```bash
# Build image
docker build -t plantscan .

# Run container
docker run -p 10000:10000 -e GROQ_API_KEY=your_key plantscan
```

### Push to Docker Hub
```bash
docker tag plantscan yourusername/plantscan:latest
docker push yourusername/plantscan:latest
```

### Deploy on Render
1. Go to [render.com](https://render.com) → New → Web Service
2. Select **Deploy an existing image**
3. Enter: `yourusername/plantscan:latest`
4. Add environment variable: `GROQ_API_KEY`
5. Set Health Check Path: `/ping`
6. Deploy

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/ping` | Health check |
| `POST` | `/predict` | Upload image → disease prediction |
| `POST` | `/chat` | Send message → LLM reply (SSE stream) |
| `DELETE` | `/chat/{session_id}` | Clear chat memory for session |
| `GET` | `/chat/sessions` | List active sessions (debug) |
| `GET` | `/docs` | Interactive API documentation |

### Example `/predict` Response
```json
{
  "class": "Potato___Early_blight",
  "confidence": 94.23,
  "predictions": {
    "Potato___Early_blight": 0.942312,
    "Potato___healthy": 0.034521,
    "Potato___Late_blight": 0.023167
  }
}
```

### Example `/chat` Request
```json
{
  "session_id": "uuid-generated-per-tab",
  "message": "How do I treat this disease?",
  "model": "llama-3.1-8b-instant",
  "scan_result": {
    "class": "Potato___Early_blight",
    "confidence": 94.23
  }
}
```

---

## 🧪 Model Details

| Property | Value |
|---|---|
| Architecture | Custom CNN (6 Convolutional layers) |
| Input size | 256 × 256 × 3 |
| Output classes | 3 |
| Training accuracy | ~95% |
| Dataset | PlantVillage |
| Framework | TensorFlow / Keras |
| Preprocessing | Rescaling(1/255) baked into model |
| Optimizer | Adam |

---

## 🔮 Future Improvements

- [ ] Add PostgreSQL for persistent scan and chat history
- [ ] User authentication with JWT tokens
- [ ] Expand to full PlantVillage dataset (38 classes, 14 crops)
- [ ] Grad-CAM heatmap — highlight which leaf area the model focused on
- [ ] TensorFlow Lite model for mobile/offline use
- [ ] LangGraph agents for multi-step reasoning
- [ ] Rate limiting with `slowapi`
- [ ] UptimeRobot to keep Render free tier awake

---

## 📄 License

This project is for educational purposes.
Dataset credit: [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

---

<div align="center">

Made with 🌿 by [Manas Singh](https://github.com/Manas-singh14)

⭐ Star this repo if you found it useful!

</div>
