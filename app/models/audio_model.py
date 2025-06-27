import openai
import os
from fastapi import UploadFile
import aiofiles

openai.api_key = os.getenv("OPENAI_API_KEY")

AUDIO_OUTPUT_PATH = "static/audio_output/response.mp3"


async def transcribe_audio(audio_file: UploadFile) -> str:
    """Uses OpenAI Whisper to transcribe speech from uploaded audio."""
    temp_path = f"temp_{audio_file.filename}"
    async with aiofiles.open(temp_path, 'wb') as out_file:
        content = await audio_file.read()
        await out_file.write(content)

    with open(temp_path, "rb") as f:
        transcript = openai.Audio.transcribe(model="whisper-1", file=f)

    os.remove(temp_path)
    return transcript["text"]


async def generate_speech(text: str) -> str:
    """Uses GPT-4o TTS to generate speech audio for a given text."""
    response = openai.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        input=text,
    )
    with open(AUDIO_OUTPUT_PATH, "wb") as out_file:
        out_file.write(response.read())

    return AUDIO_OUTPUT_PATH