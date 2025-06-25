import yaml
from gtts import gTTS
import pyttsx3
import tempfile
import os

with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

tts_engine = config.get("tts_engine", "gTTS")

def text_to_speech(text: str, lang: str = "en"):
    if tts_engine == "gTTS":
        tts = gTTS(text=text, lang=lang)
        path = tempfile.mktemp(suffix=".mp3")
        tts.save(path)
        return path
    else:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return None