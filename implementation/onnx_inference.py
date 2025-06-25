import onnxruntime as ort
from transformers import BlipProcessor, TrOCRProcessor, AutoProcessor
from PIL import Image
import numpy as np
import torch
import yaml

with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

onnx_paths = config.get("onnx_models")

def preprocess_image(image: Image.Image):
    return np.array(image.resize((224, 224))).astype("float32").transpose(2, 0, 1)[None, :]

def run_onnx_caption(image: Image.Image) -> str:
    session = ort.InferenceSession(onnx_paths["captioning"])
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(images=image, return_tensors="pt")
    out = session.run(None, {"pixel_values": inputs["pixel_values"].numpy()})
    # Placeholder: In ONNX, generation decoding is trickier. Use logits or integrate full loop.
    return "Image Caption (ONNX mode)"

def run_onnx_ocr(image: Image.Image) -> str:
    session = ort.InferenceSession(onnx_paths["ocr"])
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    inputs = processor(images=image, return_tensors="pt")
    out = session.run(None, {"pixel_values": inputs["pixel_values"].numpy()})
    return "Extracted Text (ONNX mode)"

def run_onnx_vqa(image: Image.Image, question: str) -> str:
    session = ort.InferenceSession(onnx_paths["vqa"])
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    inputs = processor(image, question, return_tensors="pt")
    out = session.run(None, {
        "input_ids": inputs["input_ids"].numpy(),
        "pixel_values": inputs["pixel_values"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    })
    return "Answer (ONNX mode)"