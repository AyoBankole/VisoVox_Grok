import React, { useState, useEffect, useRef } from 'react';
import HamburgerMenu from './components/HamburgerMenu';
import ExitButton from './components/ExitButton';
import CameraFeed from './components/CameraFeed';
import ActionButtons from './components/ActionButtons';
import AudioAssistant from './components/AudioAssistant';

export default function App() {
  const [currentView, setCurrentView] = useState('home'); // 'home' or 'main'
  const [output, setOutput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [capturedImages, setCapturedImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const outputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null); // For gallery

  // Enhanced action handler with loading states and animations
  const handleAction = async (taskType) => {
    setIsLoading(true);
    setOutput(`Initializing ${taskType}...`);
    
    // Simulate processing time
    setTimeout(() => {
      setOutput(`${taskType} task completed successfully!`);
      setIsLoading(false);
      addNotification(`${taskType} executed`, 'success');
    }, 2000);
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

  // Enhanced text-to-speech with better error handling
  const speakText = (text) => {
    try {
      // Cancel any ongoing speech
      speechSynthesis.cancel();
      
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 0.8;
      
      // Add event listeners for better UX
      utterance.onstart = () => {
        setOutput(`Speaking: "${text}"`);
        addNotification('Speech started', 'info');
      };
      
      utterance.onend = () => {
        setOutput('Speech completed');
        addNotification('Speech completed', 'success');
      };
      
      utterance.onerror = (event) => {
        setOutput(`Speech error: ${event.error}`);
        addNotification('Speech failed', 'error');
      };
      
      speechSynthesis.speak(utterance);
    } catch (error) {
      setOutput(`Speech synthesis not supported: ${error.message}`);
      addNotification('Speech not available', 'error');
    }
  };

  // Image capture functionality
  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert to image data
      const imageData = canvas.toDataURL('image/png');
      const timestamp = new Date().toLocaleString();
      
      // Add to captured images
      const newImage = {
        id: Date.now(),
        data: imageData,
        timestamp: timestamp
      };
      
      setCapturedImages(prev => [newImage, ...prev]);
      addNotification('Image captured successfully!', 'success');
      setOutput(`Image captured at ${timestamp}`);
      
      // Add capture animation
      const captureBtn = document.querySelector('.capture-button');
      if (captureBtn) {
        captureBtn.classList.add('capturing');
        setTimeout(() => {
          captureBtn.classList.remove('capturing');
        }, 300);
      }
    }
  };

  // Handler for gallery file selection
  const handleGalleryClick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        const imageData = ev.target.result;
        const timestamp = new Date().toLocaleString();
        const newImage = {
          id: Date.now(),
          data: imageData,
          timestamp: timestamp
        };
        setCapturedImages(prev => [newImage, ...prev]);
        addNotification('Image selected from gallery!', 'success');
        setOutput(`Image selected at ${timestamp}`);
      };
      reader.readAsDataURL(file);
    }
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
            handleAction('Object Detection');
            break;
          case '2':
            event.preventDefault();
            handleAction('Text Recognition');
            break;
          case '3':
            event.preventDefault();
            handleAction('Face Recognition');
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
    <div className="main-app-container min-h-screen flex flex-col md:flex-row gap-4 p-4 md:p-8 bg-white">
      {/* Navigation */}
      <nav className="app-nav">
        <button 
          className="back-button"
          onClick={() => setCurrentView('home')}
          aria-label="Back to Home"
          title="Back to Home (ESC)"
        >
          ←
        </button>
      </nav>

      <div className="app-container" role="main" aria-label="VisoVox AI App">
        
        {/* Header */}
        <header className="app-header">
          <HamburgerMenu onGalleryClick={handleGalleryClick} />
          <div className="app-title-container">
            <h1 className="app-title" role="heading" aria-level="1">
              VisoVox AI
            </h1>
            <p className="app-subtitle">Intelligent Vision & Voice Assistant</p>
          </div>
          <ExitButton />
        </header>

        {/* Status Indicator */}
        <StatusIndicator 
          isLoading={isLoading} 
          isVoiceActive={isVoiceActive} 
        />

        {/* Camera View */}
        <section className="camera-section" aria-labelledby="camera-label">
          <h2 id="camera-label" className="sr-only">Camera Live Feed</h2>
          <div className="camera-container">
            <CameraFeed videoRef={videoRef} />
            <div className="camera-overlay">
              <div className="camera-status">
                <span className="status-dot"></span>
                <span>Live</span>
              </div>
            </div>
            
            {/* Camera Controls */}
            <div className="camera-controls">
              <button 
                className="gallery-button"
                onClick={handleGalleryClick}
                title="View captured images"
              >
                📷 Gallery ({capturedImages.length})
              </button>
              
              <button 
                className="capture-button"
                onClick={captureImage}
                disabled={isLoading}
                title="Capture Image (Ctrl+C)"
                aria-label="Capture current video frame"
              >
                📸
              </button>
              
              <button 
                className="gallery-button"
                onClick={() => setCapturedImages([])}
                title="Clear all images"
                disabled={capturedImages.length === 0}
              >
                🗑️ Clear
              </button>
            </div>
          </div>
        </section>

        {/* Action Buttons */}
        <section className="action-section" role="region" aria-label="Task Options">
          <ActionButtons 
            onAction={handleAction} 
            disabled={isLoading}
            className={isLoading ? 'loading' : ''}
          />
        </section>

        {/* Voice Assistant */}
        <section className="voice-section" role="region" aria-label="Voice Assistant">
          <h3 className="section-title">Voice Assistant</h3>
          <AudioAssistant
            onVoiceCommand={handleVoiceCommand}
            speakText={speakText}
            isActive={isVoiceActive}
            disabled={isLoading}
          />
        </section>

        {/* Enhanced Output */}
        <section className="output-section" aria-live="assertive">
          <h3 className="section-title">System Output</h3>
          <div 
            ref={outputRef}
            className={`output-area ${output ? 'has-content' : ''} ${isLoading ? 'loading' : ''}`}
            role="log"
          >
            {output || "Ready to assist you..."}
          </div>
        </section>

        {/* Hidden canvas for image capture */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        {/* Hidden file input for gallery */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />

        {/* Keyboard Shortcuts Helper */}
        <KeyboardShortcuts />
      </div>
    </div>
  );

  return (
    <>
      {/* Notifications */}
      <NotificationContainer notifications={notifications} />
      
      {/* Captured Images Display */}
      {capturedImages.length > 0 && currentView === 'main' && (
        <div className="captured-images">
          <h4 style={{ color: 'white', marginBottom: '1rem' }}>Recent Captures</h4>
          {capturedImages.slice(0, 3).map(image => (
            <img
              key={image.id}
              src={image.data}
              alt={`Captured at ${image.timestamp}`}
              className="image-preview"
              onClick={() => setSelectedImage(image)}
              title={`Captured: ${image.timestamp}`}
            />
          ))}
        </div>
      )}
      
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
            <li><kbd>Ctrl</kbd> + <kbd>1</kbd> - Object Detection</li>
            <li><kbd>Ctrl</kbd> + <kbd>2</kbd> - Text Recognition</li>
            <li><kbd>Ctrl</kbd> + <kbd>3</kbd> - Face Recognition</li>
            <li><kbd>Ctrl</kbd> + <kbd>C</kbd> - Capture Image</li>
            <li><kbd>Ctrl</kbd> + <kbd>Enter</kbd> - Voice Command</li>
            <li><kbd>ESC</kbd> - Back to Home</li>
          </ul>
        </div>
      )}
    </>
  );
};