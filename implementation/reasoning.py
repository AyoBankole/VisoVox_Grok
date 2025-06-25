from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image
import torch
import yaml

# Load settings
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

inference_mode = config.get("inference_mode", "hf")

# Load processor & model once
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
model.eval()

def answer_question(image: Image.Image, question: str) -> str:
    inputs = processor(image, question, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    return processor.decode(generated_ids[0], skip_special_tokens=True)