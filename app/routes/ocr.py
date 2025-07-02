from fastapi import APIRouter, UploadFile, File
from app.models.ocr_model import extract_text
from app.utils.upload import upload_image_to_cloudinary
import shutil
import os
import uuid

router = APIRouter()

@router.post("/")
async def ocr_image(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.png"
    temp_path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = upload_image_to_cloudinary(temp_path)
    os.remove(temp_path)

    text = extract_text(image_url)
    return {
        "status": "success",
        "message": "Text extracted from image successfully.",
        "data": {
            "image_url": image_url,
            "extracted_text": text
        }
    }