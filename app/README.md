<p align="center">
  <img src="visovox-frontend/vivox.png" alt="VisoVox Logo" width="200"/>
</p>

# VisoVox AI Backend

The backend API for **VisoVox AI** built with **FastAPI**. It serves as the core service layer powering:

- 🧠 Image Captioning
- 🔍 OCR (Optical Character Recognition)
- ❓ Visual Question Answering
- 🎙️ Audio Transcription & Speech Synthesis

---

## ⚙️ Tech Stack

- FastAPI
- Python 3.9+
- OpenAI Whisper + TTS
- Groq LLaMA 4 model
- Cloudinary for image hosting
- Uvicorn server

---

## 📦 Installation

### Prerequisites

- Python 3.9 or later
- `pip` (Python package manager)
- Cloudinary and OpenAI API keys

### Setup

```bash
cd app
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file inside the `app/` directory:

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
CLOUDINARY_CLOUD_NAME=your_cloudinary_name
CLOUDINARY_API_KEY=your_cloudinary_key
CLOUDINARY_API_SECRET=your_cloudinary_secret
BASE_URL=http://localhost:10000
```

---

## 🚀 Running the Server

```bash
uvicorn app.main:app --reload --port 10000
```

---

## 📂 API Endpoints

| Method | Endpoint                  | Description                           |
|--------|---------------------------|---------------------------------------|
| POST   | `/api/caption/`           | Generate caption for uploaded image   |
| POST   | `/api/ocr/`               | Extract text from uploaded image      |
| POST   | `/api/vqa/`               | Ask question about uploaded image     |
| POST   | `/api/audio/transcribe`   | Transcribe speech to text             |
| POST   | `/api/audio/speak`        | Convert text to speech (returns .mp3) |

---

## 🧠 Models & Logic

- `caption_model.py`: Uses GROQ LLaMA-4 to describe images
- `ocr_model.py`: Uses VLM to extract visible text
- `vqa_model.py`: Handles Visual Question Answering with images + queries
- `audio_model.py`: Integrates OpenAI Whisper and GPT-4o TTS

---

## 📁 Folder Structure

```
app/
├── main.py                 # FastAPI app entry
├── models/                 # ML logic for caption, OCR, VQA, audio
├── routes/                 # API endpoints
├── utils/                  # Cloudinary upload logic
├── static/                 # Audio and image output
├── render.yaml             # Deployment config (e.g., Render.com)
```

---

## 🧪 Testing

Use tools like Postman, Insomnia, or CURL to test the endpoints. Upload images or audio files as `multipart/form-data`.

---

## 🧰 Deployment

To deploy on platforms like **Render**, use the `render.yaml` file as a service blueprint.

---

## 📜 License

MIT — you're free to use, modify, and distribute this backend with attribution.
