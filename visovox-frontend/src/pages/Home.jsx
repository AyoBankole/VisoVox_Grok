import React, { useState } from "react";
import UploadForm from "../components/UploadForm";
import AudioInput from "../components/AudioInput";
import OutputDisplay from "../components/OutputDisplay";
import { captionImage, extractText, visualQA, transcribeAudio, textToSpeech } from "../services/api";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [mode, setMode] = useState("caption");
  const [imageUrl, setImageUrl] = useState(null);

  async function handleImageSubmit({ file, mode, question }) {
    setLoading(true);
    setImageUrl(URL.createObjectURL(file));
    let res;
    if (mode === "caption") res = await captionImage(file);
    else if (mode === "ocr") res = await extractText(file);
    else if (mode === "vqa") res = await visualQA(file, question);
    setResult(res);
    setMode(mode);
    setLoading(false);
  }

  async function handleAudioTranscribe(audioBlob) {
    setLoading(true);
    const res = await transcribeAudio(audioBlob);
    setResult({ text: res.text });
    setMode("audio");
    setLoading(false);
  }

  async function handleSpeak(text) {
    const blob = await textToSpeech(text);
    const audio = new Audio(URL.createObjectURL(blob));
    audio.play();
  }

  return (
    <main className="min-h-screen flex flex-col items-center p-4 bg-gray-50">
      <h1 className="app-header text-2xl md:text-4xl font-bold mb-6 text-center">VisoVox AI Visual Assistant</h1>
      <div className="main-grid grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-5xl">
        <section className="flex flex-col gap-4">
          <UploadForm onSubmit={handleImageSubmit} loading={loading} />
          <AudioInput onTranscribe={handleAudioTranscribe} loading={loading} />
        </section>
        <section className="flex flex-col gap-4">
          <OutputDisplay result={result} imageUrl={imageUrl} mode={mode} onSpeak={handleSpeak} />
        </section>
      </div>
    </main>
  );
}