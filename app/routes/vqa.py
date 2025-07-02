from fastapi import APIRouter, UploadFile, File, Form
from app.models.vqa_model import answer_question
from app.utils.upload import upload_image_to_cloudinary
import shutil
import os
import uuid

router = APIRouter()

@router.post("/")
async def vqa_image(file: UploadFile = File(...), question: str = Form(...)):
    filename = f"{uuid.uuid4()}.png"
    temp_path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = upload_image_to_cloudinary(temp_path)
    os.remove(temp_path)

    answer = answer_question(image_url, question)
    return {
        "status": "success",
        "message": "Answer generated from image and question successfully.",
        "data": {
            "image_url": image_url,
            "question": question,
            "answer": answer
        }
    }