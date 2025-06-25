
import yaml
from PIL import Image

# Import both inference modes
from implementation.onnx_inference import run_onnx_ocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Load settings
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

ocr_backend = config.get("ocr_backend", "hf").lower()

# Load Hugging Face OCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.eval()

def run_hf_ocr(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def extract_text_from_image(image: Image.Image) -> str:
    if ocr_backend == "onnx":
        try:
            return run_onnx_ocr(image)
        except Exception as e:
            print("⚠️ ONNX OCR failed. Falling back to Hugging Face OCR.")
            print(f"🔍 Error: {e}")
            return run_hf_ocr(image)

    # Default to Hugging Face
    return run_hf_ocr(image)