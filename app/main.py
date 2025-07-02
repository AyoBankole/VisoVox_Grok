import sys
assert sys.version_info >= (3, 9), "Python 3.9+ is required"
import os
# Ensure required directories exist at startup
for d in ["app/static", "app/static/audio_output", "app/static/uploads", "app/temp"]:
    os.makedirs(d, exist_ok=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ✅ Load environment variables early
load_dotenv()

# ✅ Debug print to verify .env loading (optional, remove later)
print("✅ GROQ_API_KEY loaded:", os.getenv("GROQ_API_KEY"))
print("✅ OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI(title="VisoVox Backend")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount static files for serving uploaded images
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Import routers
from app.routes import caption, ocr, vqa, audio

# ✅ Include API routes
app.include_router(caption.router, prefix="/api/caption", tags=["Captioning"])
app.include_router(ocr.router, prefix="/api/ocr", tags=["OCR"])
app.include_router(vqa.router, prefix="/api/vqa", tags=["VQA"])
app.include_router(audio.router, prefix="/api/audio", tags=["Audio I/O"])