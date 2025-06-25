import os
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    TrOCRProcessor, VisionEncoderDecoderModel,
    AutoProcessor, AutoModelForVisualQuestionAnswering
)
from PIL import Image

# Paths
onnx_dir = os.path.join("models", "onnx")
os.makedirs(onnx_dir, exist_ok=True)

# Dummy input for models
dummy_image = Image.new("RGB", (224, 224), color="white")

# --- 1. Convert BLIP for Image Captioning ---
# def convert_blip_caption():
#     print("🔄 Converting BLIP Captioning model to ONNX...")
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()

#     inputs = processor(images=dummy_image, return_tensors="pt")
#     pixel_values = inputs["pixel_values"]

#     torch.onnx.export(
#         model,
#         args=(pixel_values,),
#         f=os.path.join(onnx_dir, "blip_caption.onnx"),
#         input_names=["pixel_values"],
#         output_names=["logits"],
#         dynamic_axes={"pixel_values": {0: "batch_size"}},
#         opset_version=14
#     )
#     print("✅ BLIP Captioning model converted.")

# --- 2. Convert BLIP for VQA ---
def convert_blip_vqa():
    print("🔄 Converting BLIP VQA model to ONNX...")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").eval()

    inputs = processor(dummy_image, "What is in the picture?", return_tensors="pt")
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs["attention_mask"]

    torch.onnx.export(
        model,
        args=(input_ids, pixel_values, attention_mask),
        f=os.path.join(onnx_dir, "blip_vqa.onnx"),
        input_names=["input_ids", "pixel_values", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "pixel_values": {0: "batch_size"},
            "attention_mask": {0: "batch_size"}
        },
        opset_version=14
    )
    print("✅ BLIP VQA model converted.")

# --- 3. Convert TrOCR for OCR ---
# def convert_trocr_ocr():
#     print("🔄 Converting TrOCR model to ONNX...")
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").eval()

#     inputs = processor(images=dummy_image, return_tensors="pt")
#     pixel_values = inputs["pixel_values"]

#     torch.onnx.export(
#         model,
#         args=(pixel_values,),
#         f=os.path.join(onnx_dir, "trocr_ocr.onnx"),
#         input_names=["pixel_values"],
#         output_names=["logits"],
#         dynamic_axes={"pixel_values": {0: "batch_size"}},
#         opset_version=14
#     )
#     print("✅ TrOCR model converted.")

# Run all conversions
if __name__ == "__main__":
    # convert_blip_caption()
    convert_blip_vqa()
    # convert_trocr_ocr()
    print("🎉 All models successfully converted and saved in models/onnx/")