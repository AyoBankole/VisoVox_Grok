import React, { useState, useRef, useEffect } from 'react';
import { Volume2, VolumeX, Copy, Download, Play, Pause, Loader2 } from 'lucide-react';
import { apiService } from '../services/api';

const OutputDisplay = ({ result, mode, imageUrl }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const audioRef = useRef(null);

  useEffect(() => {
    return () => {
      // Cleanup audio URL when component unmounts
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const getDisplayText = () => {
    if (!result) return '';
    
    switch (mode) {
      case 'caption':
        return result.caption || '';
      case 'ocr':
        return result.text || '';
      case 'vqa':
        return result.answer || '';
      default:
        return '';
    }
  };

  const getTitle = () => {
    switch (mode) {
      case 'caption':
        return 'Image Caption';
      case 'ocr':
        return 'Extracted Text';
      case 'vqa':
        return 'Answer';
      default:
        return 'Result';
    }
  };

  const handleTextToSpeech = async () => {
    const text = getDisplayText();
    if (!text) return;

    try {
      setIsGeneratingAudio(true);
      
      // Clean up previous audio URL
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }

      const newAudioUrl = await apiService.speakText(text);
      setAudioUrl(newAudioUrl);
      
      // Auto-play the generated audio
      setTimeout(() => {
        if (audioRef.current) {
          audioRef.current.play();
          setIsPlaying(true);
        }
      }, 100);

    } catch (error) {
      console.error('Text-to-speech error:', error);
      alert('Failed to generate speech. Please try again.');
    } finally {
      setIsGeneratingAudio(false);
    }
  };

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        audioRef.current.play();
        setIsPlaying(true);
      }
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  const copyToClipboard = async () => {
    const text = getDisplayText();
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (error) {
      console.error('Copy failed:', error);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    }
  };

  const downloadText = () => {
    const text = getDisplayText();
    if (!text) return;

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `visovox-${mode}-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadAudio = () => {
    if (!audioUrl) return;

    const a = document.createElement('a');
    a.href = audioUrl;
    a.download = `visovox-audio-${Date.now()}.mp3`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  if (!result) return null;

  const displayText = getDisplayText();

  return (
    <div className="w-full max-w-4xl mx-auto mt-8 animate-fade-in">
      {/* Image Preview */}
      {imageUrl && (
        <div className="mb-6 text-center">
          <img
            src={imageUrl}
            alt="Processed"
            className="max-h-64 mx-auto rounded-lg shadow-lg border border-gray-200"
          />
        </div>
      )}

      {/* Result Card */}
      <div className="bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
        {/* Header */}
        <div className="bg-primary-50 px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-primary-800">{getTitle()}</h3>
        </div>

        {/* Content */}
        <div className="p-6">
          {displayText ? (
            <div className="space-y-4">
              {/* Text Content */}
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
                  {displayText}
                </p>
              </div>

              {/* Audio Player */}
              {audioUrl && (
                <div className="bg-blue-50 rounded-lg p-4">
                  <audio
                    ref={audioRef}
                    src={audioUrl}
                    onEnded={handleAudioEnded}
                    className="hidden"
                  />
                  
                  <div className="flex items-center space-x-4">
                    <button
                      onClick={handlePlayPause}
                      className="w-10 h-10 bg-primary-500 text-white rounded-full flex items-center justify-center hover:bg-primary-600 transition-colors"
                    >
                      {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5 ml-1" />}
                    </button>
                    
                    <div className="flex-1">
                      <p className="text-sm text-gray-600">Audio generated from text</p>
                    </div>

                    <button
                      onClick={downloadAudio}
                      className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
                      title="Download Audio"
                    >
                      <Download className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex flex-wrap gap-3">
                {/* Text to Speech */}
                <button
                  onClick={handleTextToSpeech}
                  disabled={isGeneratingAudio}
                  className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isGeneratingAudio ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Generating...
                    </>
                  ) : audioUrl ? (
                    <>
                      <VolumeX className="h-4 w-4 mr-2" />
                      Regenerate Audio
                    </>
                  ) : (
                    <>
                      <Volume2 className="h-4 w-4 mr-2" />
                      Speak Text
                    </>
                  )}
                </button>

                {/* Copy to Clipboard */}
                <button
                  onClick={copyToClipboard}
                  className={`inline-flex items-center px-4 py-2 rounded-lg transition-colors ${
                    isCopied
                      ? 'bg-green-100 text-green-700 border border-green-300'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <Copy className="h-4 w-4 mr-2" />
                  {isCopied ? 'Copied!' : 'Copy Text'}
                </button>

                {/* Download Text */}
                <button
                  onClick={downloadText}
                  className="inline-flex items-center px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </button>
              </div>
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">No result to display</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default OutputDisplay;