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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

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
    return audio_fp.getvalue()

def process_inputs(name, image):
    """
    Processes the input by greeting the user using their name and,
    if an image is provided, generating a caption and text-to-speech audio.
    """
    greeting = f"Welcome, {name}!" if name and name.strip() != "" else ""
    if image is None:
        return None, greeting, None
    caption = get_caption(image)
    audio_bytes = text_to_speech(caption)
    # Combine greeting and caption, separated by a newline if both exist.
    full_caption = f"{greeting}\n\n{caption}" if greeting else caption
    return image, full_caption, audio_bytes

# Custom CSS for a creative artistic interface
custom_css = """
body {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    font-family: 'Arial', sans-serif;
}
h1, h2 {
    color: #333;
    text-align: center;
    text-shadow: 1px 1px 2px #fff;
}
.gradio-container {
    border-radius: 15px;
    background-color: rgba(255, 255, 255, 0.95);
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
}
.gradio-button {
    background-color: #fcb69f !important;
    border: none !important;
    color: #fff !important;
    font-weight: bold;
}
.gradio-input {
    margin-bottom: 20px;
}
"""

# Create Gradio Interface with two inputs (Name and Image) and three outputs.
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Textbox(label="Enter Your Name (Onboarding)", placeholder="e.g., Alex"),
        gr.Image(type="pil", label="Upload or Capture Image")
    ],
    outputs=[
        gr.Image(label="Selected Image"),
        gr.Textbox(label="Generated Caption & Greeting"),
        gr.Audio(label="Text-to-Speech")
    ],
    title="VisoVox AI",
    description="Enter your name to get started, then upload an image to generate a caption and listen to it.",
    css=custom_css,
    flagging_mode="never"
)

if __name__ == '__main__':
    iface.launch(share=True)