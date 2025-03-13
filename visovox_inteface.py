import os
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from gtts import gTTS
import streamlit as st

# Function to load the BLIP model and processor from Hugging Face Hub
@st.cache_resource
def initialize_model():
    model_cache_dir = "Model"
    os.makedirs(model_cache_dir, exist_ok=True)
    try:
        # Load processor and model directly from Hugging Face
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=model_cache_dir)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=model_cache_dir)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e
    return processor, model

# Initialize once and cache
processor, model = initialize_model()

def get_caption(image):
    """Generates a caption for the input image using the BLIP model."""
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def text_to_speech(text):
    """Converts the given text to speech using gTTS and returns audio bytes."""
    tts = gTTS(text=text, lang='en')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp.getvalue()

def process_image(image):
    """Processes the image: generates a caption and converts it to speech."""
    caption = get_caption(image)
    audio_bytes = text_to_speech(caption)
    return caption, audio_bytes

def main():
    st.title("VisoVox AI - Image Captioning and TTS")
    st.write("Upload an image to generate a caption and listen to it.")

    # File uploader widget for image input
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process the image to get caption and audio
        with st.spinner("Generating caption and audio..."):
            caption, audio_bytes = process_image(image)
        
        st.subheader("Generated Caption")
        st.write(caption)
        
        st.subheader("Text-to-Speech")
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.info("Please upload an image to get started.")

if __name__ == '__main__':
    main()