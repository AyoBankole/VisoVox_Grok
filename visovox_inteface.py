import yaml
from PIL import Image

from implementation.captioning import generate_caption
from implementation.ocr_processing import extract_text_from_image
from implementation.reasoning import answer_question
from implementation.tts_module import text_to_speech

with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

voice_enabled = config.get("voice_input_enabled", True)
language = config.get("output_language", "en")

def process_image_caption(image: Image.Image, speak: bool = False):
    caption = generate_caption(image)
    audio_path = text_to_speech(caption, language) if speak else None
    return caption, audio_path

def process_image_ocr(image: Image.Image, speak: bool = False):
    text = extract_text_from_image(image)
    audio_path = text_to_speech(text, language) if speak else None
    return text, audio_path

def process_image_question(image: Image.Image, question: str, speak: bool = False):
    answer = answer_question(image, question)
    audio_path = text_to_speech(answer, language) if speak else None
    return answer, audio_path