<p align="center">
  <img src="visovox-frontend/vivox.png" alt="VisoVox Logo" width="300"/>
</p>

<h1 align="center">VisoVox AI</h1>
<p align="center">An AI-Powered Visual Assistant for the Visually Impaired</p>

---

## 🧩 Overview

**VisoVox** is a full-stack AI-powered accessibility tool that empowers visually impaired users through:

- 🧠 Image Captioning
- 📄 Text Extraction (OCR)
- ❓ Visual Question Answering
- 🔊 Speech Output and Audio Input

It comprises a **React frontend** and a **FastAPI backend**, connected through REST APIs and enhanced with powerful models from **GROQ**, **OpenAI**, and **Cloudinary** for media management.

---

## 📁 Project Structure

```
ayobankole-visovox_grok/
├── README.md                 ← This file
├── app/                      ← FastAPI backend
├── visovox-frontend/         ← React frontend
└── requirements.txt          ← Backend dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<ayobankole>/visovox.git
cd visovox
```

### 2. Backend Setup

```bash
cd app
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
uvicorn app.main:app --reload --port 10000
```

### 3. Frontend Setup

```bash
cd visovox-frontend
npm install
npm run dev
```

Make sure the backend is running on port `10000` and the frontend on `3000`.

---

## 🔑 Environment Variables

Both frontend and backend require environment variables. Check their respective README files:

- Backend: [`app/README.md`](app/README.md)
- Frontend: [`visovox-frontend/README.md`](visovox-frontend/README.md)

---

## 📦 Features Summary

| Feature                  | Description                                  |
|--------------------------|----------------------------------------------|
| 📷 Image Captioning      | Describes uploaded images using BLIP/VLM     |
| 🔎 OCR                   | Extracts text from images                    |
| 🧠 Visual Q&A            | Answers questions based on image content     |
| 🗣️ Speech-to-Text        | Transcribes recorded speech via Whisper      |
| 🔊 Text-to-Speech        | Converts captions/responses into speech      |

---

## 🤝 Contributing

We welcome contributions! Please fork the repo and submit a pull request. Make sure to write clear commit messages and follow the coding style.

---

## 📜 License

MIT License — see `LICENSE.md` for details.
