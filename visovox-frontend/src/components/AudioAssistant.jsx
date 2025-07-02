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
    <div className="voice-assistant flex flex-col sm:flex-row gap-2 sm:gap-4 items-center justify-center mt-2">
      <button onClick={startVoice} className="btn" aria-label="Start voice assistant">🎙 Start</button>
      <button onClick={stopVoice} className="btn" aria-label="Stop voice assistant">🛑 Stop</button>
      <button onClick={() => speakText(transcript)} className="btn" aria-label="Repeat spoken text">🔈 Speak</button>
      <p aria-live="polite">Heard: {transcript}</p>
    </div>
  );
}