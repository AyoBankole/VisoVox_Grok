import os
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from gtts import gTTS
import gradio as gr

def load_trained_model():
    # Specify a directory where the model will be cached
    model_cache_dir = "Model"
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Load the model directly from Hugging Face Hub.
    # This will download the model if it's not already in the cache.
    try:
        model = BlipForConditionalGeneration.from_pretrained(
            'Salesforce/blip-image-captioning-base',
            cache_dir=model_cache_dir
        )
    except Exception as e:
        raise RuntimeError(f"Error loading model from Hugging Face: {e}")
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
    # Return the input image along with the caption and audio
    return image, caption, audio_bytes

# Create Gradio Interface with three outputs: Image, Textbox, and Audio.
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