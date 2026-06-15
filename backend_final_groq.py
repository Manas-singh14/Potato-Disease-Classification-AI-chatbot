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
from dotenv import load_dotenv
load_dotenv()

# ── Chat imports ──────────────────────────────────────────────────────
from chat.memory import get_or_create_memory, clear_memory, delete_session, list_sessions
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
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Please upload a valid image file.") from exc
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)
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
    session_id:  str
    message:     str
    model:       str = "llama3-8b-8192"   # Groq model name
    scan_result: dict | None = None


def _format_scan_context(scan_result: dict | None) -> tuple[str, str, str]:
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


# ── POST /chat ────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    disease, confidence, all_scores = _format_scan_context(req.scan_result)
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
    #         # Groq via LangChain returns an AIMessage object
    #         reply = result.content if hasattr(result, "content") else str(result)

    #         yield f"data: {json.dumps({'token': reply, 'done': False})}\n\n"
    #         yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    #     except Exception as e:
    #         error_msg = f"Error: {str(e)}"
    #         yield f"data: {json.dumps({'token': error_msg, 'done': True})}\n\n"

     # using this code function for streaming
    async def token_stream():
            try:
                full_reply = ""
                # stream() yields chunks instead of waiting for full response
                for chunk in chain.stream(
                    {
                        "input":      req.message,
                        "disease":    disease,
                        "confidence": confidence,
                        "all_scores": all_scores,
                    },
                    config={"configurable": {"session_id": req.session_id}}
                ):
                    # Groq returns AIMessageChunk objects with .content
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_reply += token
                        yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                yield f"data: {json.dumps({'token': error_msg, 'done': True})}\n\n"
    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── DELETE /chat/{session_id} ─────────────────────────────────────────
@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    clear_memory(session_id)
    return {"message": f"Chat history cleared for session {session_id}"}


# ── GET /chat/sessions ────────────────────────────────────────────────
@app.get("/chat/sessions")
async def list_chat_sessions():
    return {"active_sessions": list_sessions()}


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
