import React, { useState, useEffect, useRef } from 'react';
import HamburgerMenu from './components/HamburgerMenu';
import ExitButton from './components/ExitButton';
import CameraFeed from './components/CameraFeed';
import ActionButtons from './components/ActionButtons';
import AudioAssistant from './components/AudioAssistant';
import { uploadImage } from './services/api';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import logo from '../vivox.png';

const APP_NAME = 'VisoVox AI';

export default function App() {
  const [screen, setScreen] = useState('home'); // home, camera, actions, ask, results
  const [currentImage, setCurrentImage] = useState(null);
  const [output, setOutput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [capturedImages, setCapturedImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [question, setQuestion] = useState("");
  const outputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const askInputRef = useRef(null);
  const [showAskBar, setShowAskBar] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [cameraFacingMode, setCameraFacingMode] = useState('environment');
  const { transcript, listening, resetTranscript, browserSupportsSpeechRecognition } = useSpeechRecognition();

  // Text-to-speech helper
  const speakText = (text) => {
    try {
      window.speechSynthesis.cancel();
      const utterance = new window.SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      utterance.rate = 1;
      utterance.pitch = 1;
      utterance.volume = 1;
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      setOutput(`Speech synthesis not supported: ${error.message}`);
    }
  };

  // Capture image from camera
  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = canvas.toDataURL('image/png');
      setCurrentImage({ data: imageData, timestamp: new Date().toLocaleString() });
      setOutput("");
      setScreen('actions');
    }
  };

  // Handle gallery file selection
  const handleGalleryClick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setCurrentImage({ data: ev.target.result, timestamp: new Date().toLocaleString() });
        setOutput("");
        setScreen('actions');
      };
      reader.readAsDataURL(file);
    }
  };

  // Camera switch handler
  const handleSwitchCamera = () => {
    setCameraFacingMode((prev) => (prev === 'environment' ? 'user' : 'environment'));
  };

  // Handle action buttons
  const handleAction = async (taskType) => {
    if (!currentImage) {
      setOutput("Please capture or select an image first.");
      speakText("Please capture or select an image first.");
      return;
    }
    setIsLoading(true);
    setOutput(`Processing ${taskType}...`);
    try {
      const res = await fetch(currentImage.data);
      const blob = await res.blob();
      const formData = new FormData();
      formData.append("file", blob, `image.png`);
      let apiType = "";
      let apiRes;
      if (taskType === "caption") {
        apiType = "caption";
        apiRes = await uploadImage(formData, apiType);
        setOutput(apiRes.data.caption || "No caption returned.");
        speakText(apiRes.data.caption || "No caption returned.");
      } else if (taskType === "read") {
        apiType = "ocr";
        apiRes = await uploadImage(formData, apiType);
        setOutput(apiRes.data.extracted_text || "No text found.");
        speakText(apiRes.data.extracted_text || "No text found.");
      } else if (taskType === "ask") {
        apiType = "vqa";
        formData.append("question", question);
        apiRes = await uploadImage(formData, apiType);
        setOutput(apiRes.data.answer || "No answer returned.");
        speakText(apiRes.data.answer || "No answer returned.");
      }
      setScreen('results');
    } catch (err) {
      setOutput("Error processing image. Please try again.");
      speakText("Error processing image. Please try again.");
      setScreen('results');
    }
    setIsLoading(false);
  };

  // Voice input for Ask using react-speech-recognition
  const handleVoiceInput = () => {
    if (!browserSupportsSpeechRecognition) {
      setOutput("Speech recognition not supported in this browser.");
      return;
    }
    setIsRecording(true);
    resetTranscript();
    SpeechRecognition.startListening({ continuous: false });
  };
  // When listening stops, update question
  React.useEffect(() => {
    if (!listening && isRecording) {
      setQuestion(transcript);
      setIsRecording(false);
    }
  }, [listening]);

  // Focus ask input when Ask bar is shown
  React.useEffect(() => {
    if (screen === 'ask' && askInputRef.current) {
      askInputRef.current.focus();
    }
  }, [screen]);

  // Notification system
  const addNotification = (message, type = 'info') => {
    const id = Date.now();
    const notification = { id, message, type };
    setNotifications(prev => [...prev, notification]);
    
    // Auto-remove notification after 3 seconds
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 3000);
  };

  // Auto-scroll output area when content changes
  useEffect(() => {
    if (outputRef.current && output) {
      outputRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [output]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (screen === 'actions' && (event.ctrlKey || event.metaKey)) {
        switch (event.key) {
          case '1':
            event.preventDefault();
            handleAction('caption');
            break;
          case '2':
            event.preventDefault();
            handleAction('read');
            break;
          case '3':
            event.preventDefault();
            handleAction('ask');
            break;
          case 'Enter':
            event.preventDefault();
            // Trigger voice command
            break;
          case 'c':
            event.preventDefault();
            captureImage();
            break;
          default:
            break;
        }
      }
      
      // ESC to go back to home
      if (event.key === 'Escape' && screen === 'actions') {
        setScreen('home');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [screen]);

  // Home Screen
  if (screen === 'home') {
    return (
      <div className="min-h-screen flex flex-col items-center justify-between bg-gray-50 p-2 w-full max-w-md mx-auto">
        <div className="flex flex-col items-center w-full mt-8">
          <h1 className="text-2xl font-bold mb-2 text-center">VisoVox AI</h1>
          <img src={logo} alt="VisoVox Logo" className="w-40 h-40 object-contain mx-auto mb-4" />
          <div className="text-lg font-semibold text-center mb-2">Welcome to VisoVox AI</div>
          <button className="bg-sky-500 text-white rounded-lg px-6 py-2 text-base font-bold mt-2 mb-4" onClick={() => setScreen('camera')}>START</button>
        </div>
        <div className="w-full max-w-xs mx-auto bg-white p-3 rounded-lg border mb-6 text-center text-sm text-gray-700">
          To begin, capture or upload an image by tapping the camera icon. Once your image is ready, you can proceed to edit or analyze it using the tools provided in the app. Enjoy exploring the features!
        </div>
      </div>
    );
  }

  // Camera Screen
  if (screen === 'camera') {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-2 w-full max-w-md mx-auto">
        <div className="w-full flex justify-center">
          <div className="bg-gray-300 rounded-lg w-72 h-48 flex items-center justify-center mb-4">
            <CameraFeed videoRef={videoRef} facingMode={cameraFacingMode} />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>
        </div>
        <div className="flex flex-row justify-center items-center gap-6 mb-4">
          <button className="bg-white border border-gray-400 rounded-full w-12 h-12 flex items-center justify-center text-2xl" onClick={handleSwitchCamera} aria-label="Switch camera" disabled={isLoading}>🔄</button>
          <button className="bg-sky-500 text-white rounded-full w-16 h-16 flex items-center justify-center text-3xl border-4 border-white shadow-lg" onClick={captureImage} aria-label="Capture image from camera" disabled={isLoading}>📸</button>
          <button className="bg-white border border-gray-400 rounded-full w-12 h-12 flex items-center justify-center text-2xl" onClick={handleGalleryClick} aria-label="Upload from gallery" disabled={isLoading}>🖼️</button>
          <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
        </div>
        <button className="bg-sky-500 text-white rounded-lg px-6 py-2 text-base font-bold mt-2" onClick={() => currentImage ? setScreen('actions') : null} disabled={!currentImage}>Proceed</button>
      </div>
    );
  }

  // Actions Screen
  if (screen === 'actions') {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-2 w-full max-w-md mx-auto">
        <div className="w-full flex justify-center">
          <div className="bg-gray-300 rounded-lg w-72 h-48 flex items-center justify-center mb-4">
            {currentImage && <img src={currentImage.data} alt="Captured" className="w-full h-full object-contain rounded-lg" />}
          </div>
        </div>
        <div className="flex flex-row gap-3 mb-4 w-full justify-center">
          <button className="bg-sky-500 text-white rounded-lg w-24 h-16 flex flex-col items-center justify-center text-lg" onClick={() => handleAction('read')} aria-label="Read text in image" disabled={!currentImage || isLoading}>
            <span className="text-2xl">🔊</span>
            Read
          </button>
          <button className="bg-sky-500 text-white rounded-lg w-24 h-16 flex flex-col items-center justify-center text-lg" onClick={() => setScreen('ask')} aria-label="Ask a question about image" disabled={!currentImage || isLoading}>
            <span className="text-2xl">❓</span>
            Ask
          </button>
          <button className="bg-sky-500 text-white rounded-lg w-24 h-16 flex flex-col items-center justify-center text-lg" onClick={() => handleAction('caption')} aria-label="Generate caption for image" disabled={!currentImage || isLoading}>
            <span className="text-2xl">📝</span>
            Caption
          </button>
        </div>
      </div>
    );
  }

  // Ask Screen
  if (screen === 'ask') {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-2 w-full max-w-md mx-auto">
        <div className="w-full flex justify-center">
          <div className="bg-gray-300 rounded-lg w-72 h-48 flex items-center justify-center mb-4">
            {currentImage && <img src={currentImage.data} alt="Captured" className="w-full h-full object-contain rounded-lg" />}
          </div>
        </div>
        <input
          ref={askInputRef}
          type="text"
          placeholder="Type your question here..."
          value={question}
          onChange={e => setQuestion(e.target.value)}
          className="question-input p-2 border rounded w-full max-w-xs mb-4"
          aria-label="Type your question"
          disabled={isLoading}
        />
        <button className="bg-sky-500 text-white rounded-full w-16 h-16 flex items-center justify-center text-3xl shadow-lg mb-2" onClick={handleVoiceInput} aria-label="Record a question for Ask" disabled={isLoading}>
          🎤
        </button>
        <button className="bg-sky-500 text-white rounded-lg px-6 py-2 text-base font-bold" onClick={() => handleAction('ask')} disabled={!question || isLoading}>Ask</button>
      </div>
    );
  }

  // Results Screen
  if (screen === 'results') {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-2 w-full max-w-md mx-auto">
        <header className="w-full flex justify-between items-center mb-2 px-2">
          <button className="text-2xl" aria-label="Back" onClick={() => setScreen('actions')}>←</button>
          <span className="text-lg font-semibold">Results</span>
        </header>
        <div className="w-full flex justify-center">
          <div className="bg-white rounded-lg w-full max-w-xs p-4 mb-4 border shadow">
            <div className="text-lg font-bold text-center mb-2">AI Response: {output ? `"${output}"` : 'No response.'}</div>
            <div className="text-sm text-gray-700 text-center">The AI has successfully completed your request. You can proceed with further actions or go back to refine your input.</div>
          </div>
        </div>
        <div className="flex flex-row gap-3 mt-4 w-full max-w-xs mx-auto justify-center">
          <button className="bg-sky-500 text-white rounded-lg w-20 h-12 flex flex-col items-center justify-center text-base" onClick={() => setScreen('actions')}>↻<span className="text-xs">Retry</span></button>
          <button className="bg-sky-500 text-white rounded-lg w-20 h-12 flex flex-col items-center justify-center text-base" onClick={() => navigator.share ? navigator.share({ text: output }) : null}>⇪<span className="text-xs">Share</span></button>
          <button className="bg-sky-500 text-white rounded-lg w-20 h-12 flex flex-col items-center justify-center text-base" onClick={() => setScreen('home')}>⌂<span className="text-xs">Home</span></button>
        </div>
      </div>
    );
  }

  // Fallback
  return null;
}

// Notification Component
const NotificationContainer = ({ notifications }) => {
  if (notifications.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 w-full max-w-xs sm:max-w-sm md:max-w-md z-50">
      {notifications.map(notification => (
        <div 
          key={notification.id}
          className={`notification notification-${notification.type}`}
          role="alert"
        >
          <span className="notification-icon">
            {notification.type === 'success' && '✓'}
            {notification.type === 'error' && '✕'}
            {notification.type === 'info' && 'ℹ'}
          </span>
          {notification.message}
        </div>
      ))}
    </div>
  );
};

// Status Indicator Component
const StatusIndicator = ({ isLoading, isVoiceActive }) => {
  if (!isLoading && !isVoiceActive) return null;

  return (
    <div className="status-indicator">
      {isLoading && (
        <div className="status-item">
          <div className="spinner"></div>
          <span>Processing...</span>
        </div>
      )}
      {isVoiceActive && (
        <div className="status-item voice-active">
          <div className="voice-wave">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <span>Voice Active</span>
        </div>
      )}
    </div>
  );
};

// Keyboard Shortcuts Component
const KeyboardShortcuts = () => {
  const [showShortcuts, setShowShortcuts] = useState(false);

  return (
    <>
      <button 
        className="shortcuts-toggle"
        onClick={() => setShowShortcuts(!showShortcuts)}
        title="Keyboard Shortcuts"
        aria-label="Toggle keyboard shortcuts"
      >
        ⌨️
      </button>
      
      {showShortcuts && (
        <div className="shortcuts-panel">
          <h4>Keyboard Shortcuts</h4>
          <ul>
            <li><kbd>Ctrl</kbd> + <kbd>1</kbd> - Caption</li>
            <li><kbd>Ctrl</kbd> + <kbd>2</kbd> - Text Recognition</li>
            <li><kbd>Ctrl</kbd> + <kbd>3</kbd> - Ask</li>
            <li><kbd>Ctrl</kbd> + <kbd>C</kbd> - Capture Image</li>
            <li><kbd>Ctrl</kbd> + <kbd>Enter</kbd> - Voice Command</li>
            <li><kbd>ESC</kbd> - Back to Home</li>
          </ul>
        </div>
      )}
    </>
  );
};