import React, { useRef, useState } from "react";

export default function UploadForm({ onSubmit, loading }) {
  const fileInput = useRef();
  const [mode, setMode] = useState("caption");
  const [question, setQuestion] = useState("");

  function handleFileChange(e) {
    const file = e.target.files[0];
    if (file) onSubmit({ file, mode, question });
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!fileInput.current.files[0]) return;
    onSubmit({ file: fileInput.current.files[0], mode, question });
  }

  return (
    <form className="form-card" onSubmit={handleSubmit}>
      <label>
        <span>Choose Image:</span>
        <input type="file" accept="image/*" ref={fileInput} required disabled={loading} />
      </label>
      <div className="modes">
        <label>
          <input type="radio" value="caption" checked={mode === "caption"} onChange={() => setMode("caption")} />
          Caption
        </label>
        <label>
          <input type="radio" value="ocr" checked={mode === "ocr"} onChange={() => setMode("ocr")} />
          OCR
        </label>
        <label>
          <input type="radio" value="vqa" checked={mode === "vqa"} onChange={() => setMode("vqa")} />
          Visual Q&A
        </label>
      </div>
      {mode === "vqa" && (
        <input
          type="text"
          placeholder="Ask a question about the image"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          required
          disabled={loading}
        />
      )}
      <button className="btn-primary" type="submit" disabled={loading}>
        {loading ? "Processing..." : "Submit"}
      </button>
    </form>
  );
}