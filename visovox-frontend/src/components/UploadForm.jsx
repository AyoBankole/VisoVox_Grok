import React, { useState } from 'react';
import { uploadImage } from '../services/api';

export default function UploadForm({ onResult }) {
  const [image, setImage] = useState(null);

  const handleFileChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleSubmit = async (task) => {
    if (!image) {
      alert("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const data = await uploadImage(formData, task);
      onResult(data.result || "No response received.");
    } catch (error) {
      onResult("Error processing your request.");
      console.error(error);
    }
  };

  return (
    <div role="form" aria-labelledby="form-label" className="w-full max-w-md mx-auto p-4 bg-white rounded-lg shadow-md flex flex-col gap-4">
      <label id="form-label" className="label mb-2 text-lg font-semibold" htmlFor="image-upload">
        Upload Image
      </label>
      <input
        id="image-upload"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        aria-label="Upload an image from your device"
        className="mb-4"
      />
      <div className="button-group flex flex-col sm:flex-row gap-2 sm:gap-4 w-full" role="group" aria-label="AI Tasks">
        <button onClick={() => handleSubmit("caption")} className="btn flex-1 py-2 px-4 text-base rounded" aria-label="Generate image caption">
          📝 Caption
        </button>
        <button onClick={() => handleSubmit("vqa")} className="btn flex-1 py-2 px-4 text-base rounded" aria-label="Ask a question about the image">
          ❓ Ask
        </button>
        <button onClick={() => handleSubmit("ocr")} className="btn flex-1 py-2 px-4 text-base rounded" aria-label="Read text in the image">
          🔊 Read
        </button>
      </div>
    </div>
  );
}