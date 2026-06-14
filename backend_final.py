import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json

# ── Chat imports ──────────────────────────────────────────────────────
from chat.memory import clear_memory, delete_session, list_sessions
from chat.chain  import build_chain

# ── Config ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "models" / "new_best_model.keras"
IMAGE_SIZE  = 256
CLASS_NAMES = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

# ── Load model at startup ─────────────────────────────────────────────
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Loading model, please wait...")
    ml_models["model"] = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model ready! Server is fully up.")
    yield
    ml_models.clear()

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Potato Disease Classifier + Chat",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ──────────────────────────────────────────────────────
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# ── Image preprocessing ───────────────────────────────────────────────
def read_file_as_image(data: bytes) -> np.ndarray:
    """
    Model has Rescaling(1./255) baked in as first layer.
    Do NOT normalize here — pass raw uint8 to avoid double normalization.
    """
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Please upload a valid image file.") from exc
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)    # raw uint8 [0, 255] — model rescales internally
    return image

# ── Predict ───────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload a JPEG or PNG image."
        )
    image     = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions   = ml_models["model"].predict(img_batch, verbose=0)
    predicted_idx = int(np.argmax(predictions[0]))

    return {
        "class":      CLASS_NAMES[predicted_idx],
        "confidence": round(float(np.max(predictions[0])) * 100, 2),
        "predictions": {
            CLASS_NAMES[i]: round(float(predictions[0][i]), 6)
            for i in range(len(CLASS_NAMES))
        }
    }

# ══════════════════════════════════════════════════════════════════════
#  CHAT ROUTES
# ══════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    session_id:  str                 # unique ID per browser tab — generated in frontend
    message:     str                 # what the user typed
    model:       str = "llama3.2"   # which Ollama model to use
    scan_result: dict | None = None  # current disease result from /predict, optional


def _format_scan_context(scan_result: dict | None) -> tuple[str, str, str]:
    """
    Extracts disease, confidence, and all_scores strings from scan_result.
    Returns safe defaults if no scan has been done yet.
    """
    if not scan_result:
        return "No scan performed yet", "N/A", "N/A"

    disease    = scan_result.get("class", "Unknown")
    confidence = str(round(scan_result.get("confidence", 0), 1))
    preds      = scan_result.get("predictions", {})
    all_scores = ", ".join(
        f"{k.replace('___', ' ').replace('_', ' ')}: {round(v * 100, 1)}%"
        for k, v in preds.items()
    )
    return disease, confidence, all_scores


# ── POST /chat  — streaming response ─────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Receives a user message, runs it through LangChain + Ollama,
    and streams the reply back using Server-Sent Events.
    Uses the modern RunnableWithMessageHistory pattern.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    disease, confidence, all_scores = _format_scan_context(req.scan_result)

    # build chain — session memory is handled inside build_chain via
    # RunnableWithMessageHistory using req.session_id
    chain = build_chain(session_id=req.session_id, model=req.model)

    # async def token_stream():
    #     try:
    #         result = chain.invoke(
    #             {
    #                 "input":      req.message,
    #                 "disease":    disease,
    #                 "confidence": confidence,
    #                 "all_scores": all_scores,
    #             },
    #             config={"configurable": {"session_id": req.session_id}}
    #         )

    #         # result is a plain string from OllamaLLM
    #         reply = result if isinstance(result, str) else str(result)

    #         yield f"data: {json.dumps({'token': reply, 'done': False})}\n\n"
    #         yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    #     except Exception as e:
    #         error_msg = f"Error: {str(e)}. Make sure Ollama is running with: ollama serve"
    #         yield f"data: {json.dumps({'token': error_msg, 'done': True})}\n\n"
    async def token_stream():
        try:
            import httpx

            # Build the messages list manually for Ollama API
            system_prompt = f"""You are PlantScan AI, an expert plant pathologist assistant.
    Current scan: {disease} detected with {confidence}% confidence.
    All scores: {all_scores}
    Answer specifically based on this scan result. Be concise and practical."""

            # get existing history from memory
            from chat.memory import get_or_create_memory
            history = get_or_create_memory(req.session_id)
            messages = [{"role": "system", "content": system_prompt}]
            for msg in history.messages:
                role = "user" if msg.type == "human" else "assistant"
                messages.append({"role": role, "content": msg.content})
            messages.append({"role": "user", "content": req.message})

            # call Ollama streaming API directly
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", "http://localhost:11434/api/chat", json={
                    "model": req.model,
                    "messages": messages,
                    "stream": True
                }) as response:
                    full_reply = ""
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                full_reply += token
                                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                        except:
                            continue

                    # save to LangChain memory after streaming completes
                    history.add_user_message(req.message)
                    history.add_ai_message(full_reply)
                    yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

        except Exception as e:
            error_msg = f"Error: {str(e)}. Make sure Ollama is running with: ollama serve"
            yield f"data: {json.dumps({'token': error_msg, 'done': True})}\n\n"
    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ── DELETE /chat/{session_id}  — clear memory ────────────────────────
@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    """
    Clears the conversation memory for this session.
    Called when user clicks the 'Clear' button in the frontend.
    """
    clear_memory(session_id)
    return {"message": f"Chat history cleared for session {session_id}"}


# ── GET /chat/sessions  — debug endpoint ─────────────────────────────
@app.get("/chat/sessions")
async def list_chat_sessions():
    """Shows all active session IDs. Useful during development."""
    return {"active_sessions": list_sessions()}


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)