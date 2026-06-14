import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "models" / "new_best_model.keras"
IMAGE_SIZE  = 256
CLASS_NAMES = ['Potato___Early_blight','Potato___healthy','Potato___Late_blight']

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
    title="Potato Disease Classifier",
    version="1.0.0",
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
    IMPORTANT FIX: The model already contains a Rescaling(1./255) layer as its
    first layer (baked in during training). So we must NOT normalize here —
    just resize and return raw uint8 values. Double-normalizing caused the
    model to always predict the same class.
    """
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Please upload a valid image file.") from exc
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)          # raw uint8 [0, 255] — model rescales internally
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
    img_batch = np.expand_dims(image, 0)                     # shape → (1, 256, 256, 3)

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

# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
