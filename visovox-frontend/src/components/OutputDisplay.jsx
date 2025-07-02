import React from 'react';

export default function OutputDisplay({ responseText }) {
  const speak = () => {
    const utterance = new SpeechSynthesisUtterance(responseText);
    utterance.lang = "en-US";
    speechSynthesis.speak(utterance);
  };

  return (
    <div className="output-display p-4 rounded-lg shadow-md bg-white w-full max-w-xl mx-auto mt-4">
      <h2 id="output-label" className="sr-only">AI Output</h2>
      <p className="output-text text-base md:text-lg mb-4" aria-live="assertive" aria-atomic="true">
        {responseText}
      </p>
      <button onClick={speak} className="btn py-2 px-6 text-lg rounded w-full sm:w-auto" aria-label="Read response aloud">
        🔈 Read Aloud
      </button>
    </div>
  );
}