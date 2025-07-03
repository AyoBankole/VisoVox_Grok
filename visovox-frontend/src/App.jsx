import React, { useState, useEffect, useRef } from 'react';
import HamburgerMenu from './components/HamburgerMenu';
import ExitButton from './components/ExitButton';
import CameraFeed from './components/CameraFeed';
import ActionButtons from './components/ActionButtons';
import AudioAssistant from './components/AudioAssistant';
import { uploadImage } from './services/api';

export default function App() {
  const [currentView, setCurrentView] = useState('home'); // 'home' or 'main'
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
  const fileInputRef = useRef(null); // For gallery
  const [currentImage, setCurrentImage] = useState(null);
  const [showAskBar, setShowAskBar] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  // Helper to speak text aloud
  const speakText = (text) => {
    try {
      speechSynthesis.cancel();
      const utterance = new window.SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      utterance.rate = 1;
      utterance.pitch = 1;
      utterance.volume = 1;
      speechSynthesis.speak(utterance);
    } catch (error) {
      setOutput(`Speech synthesis not supported: ${error.message}`);
    }
  };

  // Handle action buttons to send currentImage to backend
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
    } catch (err) {
      setOutput("Error processing image. Please try again.");
      speakText("Error processing image. Please try again.");
    }
    setIsLoading(false);
  };

  // Enhanced voice command handler
  const handleVoiceCommand = (text) => {
    setOutput(`Processing voice command: "${text}"`);
    setIsVoiceActive(true);
    
    // Simulate voice processing
    setTimeout(() => {
      setOutput(`Voice command "${text}" processed successfully!`);
      setIsVoiceActive(false);
      addNotification('Voice command processed', 'info');
    }, 1500);
  };

  // Capture image from camera and set as currentImage
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
    }
  };

  // Handle gallery file selection and set as currentImage
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
      };
      reader.readAsDataURL(file);
    }
  };

  // Retake: clear current image and show camera feed
  const handleRetake = () => {
    setCurrentImage(null);
    setOutput("");
    setShowAskBar(false);
    setQuestion("");
  };

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
      if (currentView === 'main' && (event.ctrlKey || event.metaKey)) {
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
      if (event.key === 'Escape' && currentView === 'main') {
        setCurrentView('home');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentView]);

  // Home Page Component
  const HomePage = () => (
    <div className="home-container min-h-screen flex flex-col items-center justify-center p-4 sm:p-8 bg-gray-50">
      <div className="home-logo">
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
          {/* VisoVox Logo Recreation */}
          {/* Left magnifying glass with speaker */}
          <circle cx="120" cy="120" r="60" fill="#2E5F98" stroke="#4682B4" strokeWidth="8"/>
          <circle cx="120" cy="120" r="40" fill="white"/>
          
          {/* Speaker icon */}
          <rect x="100" y="105" width="15" height="30" fill="#2E5F98" rx="2"/>
          <path d="M115 105 L130 95 L130 145 L115 135" fill="#2E5F98"/>
          <path d="M132 100 Q140 110 132 130" stroke="#2E5F98" strokeWidth="3" fill="none"/>
          <path d="M135 95 Q148 110 135 135" stroke="#2E5F98" strokeWidth="3" fill="none"/>
          
          {/* Left handle */}
          <line x1="65" y1="175" x2="90" y2="150" stroke="#2E5F98" strokeWidth="12" strokeLinecap="round"/>
          <circle cx="60" cy="180" r="8" fill="#4682B4"/>
          
          {/* Right magnifying glass with eye */}
          <circle cx="280" cy="120" r="60" fill="#2E5F98" stroke="#4682B4" strokeWidth="8"/>
          <circle cx="280" cy="120" r="40" fill="white"/>
          
          {/* Eye */}
          <circle cx="280" cy="120" r="25" fill="white"/>
          <circle cx="280" cy="120" r="15" fill="#5A9BD4"/>
          <circle cx="280" cy="120" r="8" fill="#2E5F98"/>
          <ellipse cx="285" cy="115" rx="3" ry="4" fill="white" opacity="0.8"/>
          
          {/* Right handle */}
          <line x1="335" y1="175" x2="310" y2="150" stroke="#2E5F98" strokeWidth="12" strokeLinecap="round"/>
          <circle cx="340" cy="180" r="8" fill="#4682B4"/>
          
          {/* VISOVOX Text */}
          <text x="200" y="240" textAnchor="middle" fill="#2E5F98" fontSize="36" fontWeight="bold" fontFamily="Arial, sans-serif">
            VISOVOX
          </text>
        </svg>
      </div>
      
      <h1 className="home-title">VisoVox AI</h1>
      <p className="home-subtitle">
        Intelligent Vision & Voice Assistant - Experience the future of AI-powered visual recognition and voice interaction
      </p>
      
      <button 
        className="start-button"
        onClick={() => setCurrentView('main')}
        aria-label="Start VisoVox AI Application"
      >
        🚀 Start Experience
      </button>
    </div>
  );

  // Main App Component
  const MainApp = () => (
    <div className="main-app-container min-h-screen flex flex-col items-center justify-center p-2 sm:p-4 bg-white">
      {/* Header */}
      <header className="w-full flex justify-between items-center mb-2">
        <div></div>
        <button className="btn exit-button text-lg px-4 py-2" aria-label="Exit the app">EXIT</button>
      </header>
      {/* Image or Camera */}
      <div className="w-full flex flex-col items-center">
        {!currentImage ? (
          <>
            <CameraFeed videoRef={videoRef} />
            <button className="capture-button rounded-full bg-blue-600 text-white text-3xl w-24 h-24 flex items-center justify-center mt-4 mb-2 shadow-lg" onClick={captureImage} aria-label="Capture image from camera" disabled={isLoading}>📸</button>
            <button className="gallery-button btn mt-2 w-full max-w-xs" onClick={handleGalleryClick} aria-label="Upload from gallery" disabled={isLoading}>Upload from Gallery</button>
          </>
        ) : (
          <div className="flex flex-col items-center w-full">
            <img src={currentImage.data} alt="Captured" className="image-preview mb-2 w-full max-w-xs rounded-lg border shadow" style={{ maxHeight: 300 }} />
            <div className="flex gap-2 mb-2 w-full justify-center">
              <button className="btn w-1/2" onClick={handleRetake} aria-label="Retake photo" disabled={isLoading}>Retake</button>
              <button className="btn w-1/2" onClick={handleGalleryClick} aria-label="Pick from gallery" disabled={isLoading}>Pick from Gallery</button>
            </div>
          </div>
        )}
        {/* Hidden canvas for image capture */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        {/* Hidden file input for gallery */}
        <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
      </div>
      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-3 mt-4 w-full max-w-xs mx-auto">
        <button className="btn w-full py-3 text-lg" onClick={() => handleAction('read')} aria-label="Read text in image" disabled={!currentImage || isLoading}>🔊 Read</button>
        <button className="btn w-full py-3 text-lg" onClick={() => setShowAskBar(true)} aria-label="Ask a question about image" disabled={!currentImage || isLoading}>❓ Ask</button>
        <button className="btn w-full py-3 text-lg" onClick={() => handleAction('caption')} aria-label="Generate caption for image" disabled={!currentImage || isLoading}>📝 Caption</button>
      </div>
      {/* Big Record Button for Voice Input (Ask) */}
      <div className="flex flex-col items-center mt-4 w-full max-w-xs mx-auto">
        <button className={`btn rounded-full bg-green-600 text-white text-4xl w-20 h-20 flex items-center justify-center shadow-lg ${isRecording ? 'animate-pulse' : ''}`} onClick={() => setIsRecording(true)} aria-label="Record a question for Ask" disabled={!currentImage || isLoading}>
          🎤
        </button>
        <span className="text-sm mt-1">Voice input for Ask</span>
      </div>
      {/* Ask Bar (text or voice) */}
      {showAskBar && (
        <div className="flex flex-col items-center mt-3 w-full max-w-xs mx-auto bg-white p-3 rounded shadow">
          <input
            type="text"
            placeholder="Ask a question about the image..."
            value={question}
            onChange={e => setQuestion(e.target.value)}
            className="question-input p-2 border rounded w-full mb-2"
            aria-label="Type your question"
            disabled={!currentImage || isLoading}
          />
          <button className="btn w-full mb-2" onClick={() => handleAction('ask')} aria-label="Submit question" disabled={!currentImage || isLoading || !question}>Ask</button>
          <button className="btn w-full" onClick={() => setShowAskBar(false)} aria-label="Close ask bar">Close</button>
        </div>
      )}
      {/* Output Section */}
      <div className="output-section mt-4 w-full max-w-xs mx-auto p-3 bg-gray-100 rounded-lg min-h-[60px] text-lg" aria-live="assertive">
        {output || "Ready to assist you..."}
      </div>
    </div>
  );

  return (
    <>
      {/* Notifications */}
      <NotificationContainer notifications={notifications} />
      
      {/* Image Modal */}
      {selectedImage && (
        <div className="image-modal" onClick={() => setSelectedImage(null)}>
          <img src={selectedImage.data} alt={`Captured at ${selectedImage.timestamp}`} />
          <button 
            className="modal-close"
            onClick={() => setSelectedImage(null)}
            aria-label="Close image"
          >
            ×
          </button>
        </div>
      )}
      
      {/* Render current view */}
      {currentView === 'home' ? <HomePage /> : <MainApp />}
    </>
  );
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