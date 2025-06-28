import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

export default function AudioAssistant({ onVoiceCommand, speakText }) {
  const { transcript, listening, resetTranscript } = useSpeechRecognition();

  const startVoice = () => {
    resetTranscript();
    SpeechRecognition.startListening({ continuous: true });
  };

  const stopVoice = () => {
    SpeechRecognition.stopListening();
    onVoiceCommand(transcript);
  };

  return (
    <div className="voice-assistant" aria-label="Voice assistant controls">
      <button onClick={startVoice} className="btn" aria-label="Start voice assistant">🎙 Start</button>
      <button onClick={stopVoice} className="btn" aria-label="Stop voice assistant">🛑 Stop</button>
      <button onClick={() => speakText(transcript)} className="btn" aria-label="Repeat spoken text">🔈 Speak</button>
      <p aria-live="polite">Heard: {transcript}</p>
    </div>
  );
}