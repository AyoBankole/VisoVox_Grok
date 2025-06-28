import React, { useState } from "react";

export default function OutputDisplay({ result, imageUrl, mode, onSpeak }) {
  const [playing, setPlaying] = useState(false);

  async function handleSpeak() {
    setPlaying(true);
    await onSpeak(result.text);
    setPlaying(false);
  }

  if (!result) return null;

  return (
    <div className="output-card">
      {imageUrl && <img src={imageUrl} alt="" className="output-img" />}
      <div className="output-content">
        {mode === "caption" && <h3>Caption:</h3>}
        {mode === "ocr" && <h3>Extracted Text:</h3>}
        {mode === "vqa" && <h3>Answer:</h3>}
        <p>{result.text}</p>
        <button className="btn-secondary" onClick={handleSpeak} disabled={playing}>
          {playing ? "Playing..." : "Speak"}
        </button>
      </div>
    </div>
  );
}