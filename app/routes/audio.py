from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import FileResponse
from app.models.audio_model import transcribe_audio, generate_speech
import os

router = APIRouter(prefix="/audio", tags=["Audio"])

AUDIO_DIR = "static/audio_output"
os.makedirs(AUDIO_DIR, exist_ok=True)


@router.post("/transcribe")
async def transcribe_audio_endpoint(audio: UploadFile = File(...)):
    """Convert user's audio to text using Whisper (via OpenAI)."""
    result = await transcribe_audio(audio)
    return {"transcript": result}


@router.post("/speak")
async def speak_text_endpoint(text: str = Form(...)):
    """Convert model text output to speech using GPT-4o TTS."""
    audio_path = await generate_speech(text)
    filename = os.path.basename(audio_path)
    return FileResponse(path=audio_path, filename=filename, media_type="audio/mpeg")