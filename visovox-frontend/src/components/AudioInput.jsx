import React from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

export default function AudioInput({ onVoiceCommand }) {
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  const handleStart = () => {
    resetTranscript();
    SpeechRecognition.startListening({ continuous: true });
  };

  const handleStop = () => {
    SpeechRecognition.stopListening();
    onVoiceCommand(transcript);
  };

  if (!browserSupportsSpeechRecognition) {
    return <p>Your browser does not support speech recognition.</p>;
  }

  return (
    <div className="voice-input flex flex-col sm:flex-row gap-2 sm:gap-4 items-center justify-center mt-2" role="form" aria-labelledby="voice-label">
      <p id="voice-label" className="sr-only">Voice Control Section</p>

      <button onClick={handleStart} className="btn" aria-label="Start voice recognition">
        🎙️ Start Voice
      </button>
      <button onClick={handleStop} className="btn" aria-label="Stop voice recognition">
        🛑 Stop
      </button>

      <p className="transcript-text" aria-live="polite">
        Heard: {transcript || "No input yet."}
      </p>
    </div>
  );
}