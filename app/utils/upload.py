import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_image_to_cloudinary(file_path: str):
    try:
        result = cloudinary.uploader.upload(file_path)
        return result["secure_url"]
    except Exception as e:
        raise RuntimeError(f"Cloudinary upload failed: {e}")