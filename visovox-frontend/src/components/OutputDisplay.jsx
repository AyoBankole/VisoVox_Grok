import React from 'react';

export default function OutputDisplay({ responseText }) {
  const speak = () => {
    const utterance = new SpeechSynthesisUtterance(responseText);
    utterance.lang = "en-US";
    speechSynthesis.speak(utterance);
  };

  return (
    <div className="output-display" role="region" aria-labelledby="output-label">
      <h2 id="output-label" className="sr-only">AI Output</h2>

      <p className="output-text" aria-live="assertive" aria-atomic="true">
        {responseText}
      </p>

      <button onClick={speak} className="btn" aria-label="Read response aloud">
        🔈 Read Aloud
      </button>
    </div>
  );
}