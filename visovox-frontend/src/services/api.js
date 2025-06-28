// Handles all backend API calls
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:10000";

export async function captionImage(image) {
  const formData = new FormData();
  formData.append("image", image);
  const res = await fetch(`${API_URL}/api/caption/`, { method: "POST", body: formData });
  return res.json();
}

export async function extractText(image) {
  const formData = new FormData();
  formData.append("image", image);
  const res = await fetch(`${API_URL}/api/ocr/`, { method: "POST", body: formData });
  return res.json();
}

export async function visualQA(image, question) {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("question", question);
  const res = await fetch(`${API_URL}/api/vqa/`, { method: "POST", body: formData });
  return res.json();
}

export async function transcribeAudio(audio) {
  const formData = new FormData();
  formData.append("audio", audio);
  const res = await fetch(`${API_URL}/api/audio/transcribe`, { method: "POST", body: formData });
  return res.json();
}

export async function textToSpeech(text) {
  const res = await fetch(`${API_URL}/api/audio/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.blob();
}