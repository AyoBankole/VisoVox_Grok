import gradio as gr
from PIL import Image
import yaml

from visovox_inteface import (
    process_image_caption,
    process_image_ocr,
    process_image_question
)
from implementation.voice_input import listen_from_microphone

# Load config
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

voice_enabled = config.get("voice_input_enabled", True)
language = config.get("output_language", "en")

# ----------- Gradio Functions -----------

def caption_image(image: Image.Image):
    caption, audio_path = process_image_caption(image, speak=voice_enabled)
    return caption, audio_path

def ocr_image(image: Image.Image):
    text, audio_path = process_image_ocr(image, speak=voice_enabled)
    return text, audio_path

def answer_image_question(image: Image.Image, question: str):
    answer, audio_path = process_image_question(image, question, speak=voice_enabled)
    return answer, audio_path

def use_microphone(image: Image.Image):
    question = listen_from_microphone()
    answer, audio_path = process_image_question(image, question, speak=voice_enabled)
    return question, answer, audio_path

# ----------- Gradio Interface -----------

with gr.Blocks(title="VisoVox AI") as demo:
    gr.Markdown("## 🧠 VisoVox AI – Voice-Enabled Image Intelligence for the Visually Impaired")

    with gr.Tab("🖼️ Image Captioning"):
        with gr.Row():
            image_input1 = gr.Image(label="Upload Image", type="pil")
            caption_output = gr.Textbox(label="Generated Caption")
        audio_output1 = gr.Audio(label="TTS (if enabled)", interactive=False)
        caption_button = gr.Button("Generate Caption")
        caption_button.click(fn=caption_image, inputs=image_input1, outputs=[caption_output, audio_output1])

    with gr.Tab("❓ Visual Q&A"):
        with gr.Row():
            image_input2 = gr.Image(label="Upload Image", type="pil")
            question_input = gr.Textbox(label="Enter Your Question")
        vqa_output = gr.Textbox(label="Answer")
        audio_output2 = gr.Audio(label="TTS (if enabled)", interactive=False)
        vqa_button = gr.Button("Ask")
        vqa_button.click(fn=answer_image_question, inputs=[image_input2, question_input], outputs=[vqa_output, audio_output2])

        if voice_enabled:
            mic_button = gr.Button("🎤 Ask with Microphone")
            mic_question = gr.Textbox(label="Recognized Question")
            mic_answer = gr.Textbox(label="Answer")
            mic_audio = gr.Audio(label="TTS", interactive=False)
            mic_button.click(fn=use_microphone, inputs=image_input2, outputs=[mic_question, mic_answer, mic_audio])

    with gr.Tab("📝 OCR & Number Insight"):
        image_input3 = gr.Image(label="Upload Image", type="pil")
        ocr_output = gr.Textbox(label="Extracted Text")
        audio_output3 = gr.Audio(label="TTS (if enabled)", interactive=False)
        ocr_button = gr.Button("Extract Text")
        ocr_button.click(fn=ocr_image, inputs=image_input3, outputs=[ocr_output, audio_output3])

# ----------- Launch App -----------
demo.launch(share=True)