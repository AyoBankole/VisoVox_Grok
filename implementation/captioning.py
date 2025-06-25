
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import yaml

# Load config
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

inference_mode = config.get("inference_mode", "hf")

# Always load HF since ONNX is not available for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

def generate_caption(image: Image.Image) -> str:
    if inference_mode == "onnx":
        print("⚠️ Warning: ONNX mode not supported for captioning. Using Hugging Face model instead.")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(**inputs)
    return processor.decode(output_ids[0], skip_special_tokens=True)