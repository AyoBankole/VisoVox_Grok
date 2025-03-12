import os
import io
import random
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import gdown
from gtts import gTTS
import gradio as gr

def load_trained_model():
    model_path = "model/blip_image_captioning_model.pth"
    drive_link = "https://drive.google.com/file/d/1UyHlFI5EVWskNh_p3ZgQvHKA9IdG6pIY/view?usp=drive_link"  # Replace with your actual file ID or direct link

    # Download the model if it does not exist
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        gdown.download(drive_link, model_path, quiet=False)

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
    model.load_state_dict(state_dict)
    return model

# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = load_trained_model()

def get_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp.getvalue()  # Return audio as bytes

def process_image(image):
    caption = get_caption(image)
    audio_bytes = text_to_speech(caption)
    # Return the input image along with caption and audio
    return image, caption, audio_bytes

# Gradio Interface with three outputs: Image, Textbox, and Audio.
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload or Capture Image"),
    outputs=[
        gr.Image(label="Selected Image"),
        gr.Textbox(label="Generated Caption"),
        gr.Audio(label="Text-to-Speech")
    ],
    title="VisoVox AI",
    description="Upload an image or capture one using your camera to generate a caption and listen to it.",
    flagging_mode="never"  # Updated parameter instead of deprecated allow_flagging
)

if __name__ == '__main__':
    iface.launch(share=True)
