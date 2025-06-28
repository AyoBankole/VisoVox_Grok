import React, { useRef, useState } from "react";

export default function AudioInput({ onTranscribe, loading }) {
  const audioRef = useRef();
  const [audioBlob, setAudioBlob] = useState(null);
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new window.MediaRecorder(stream);
    setMediaRecorder(recorder);
    let chunks = [];
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = () => {
      setAudioBlob(new Blob(chunks, { type: "audio/webm" }));
    };
    recorder.start();
    setRecording(true);
  }

  function stopRecording() {
    mediaRecorder.stop();
    setRecording(false);
  }

  function handleTranscribe() {
    onTranscribe(audioBlob);
  }

  return (
    <div className="form-card">
      <p>Record Audio:</p>
      <div>
        <button className="btn-primary" type="button" onClick={recording ? stopRecording : startRecording} disabled={loading}>
          {recording ? "Stop Recording" : "Start Recording"}
        </button>
        {audioBlob && (
          <button className="btn-secondary" type="button" onClick={handleTranscribe} disabled={loading}>
            {loading ? "Transcribing..." : "Transcribe"}
          </button>
        )}
      </div>
      {audioBlob && <audio controls src={URL.createObjectURL(audioBlob)} />}
    </div>
  );
}