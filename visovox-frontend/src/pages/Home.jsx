import React, { useState } from 'react';
import { Eye, Volume2, MessageCircle, FileText, Sparkles } from 'lucide-react';
import UploadForm from '../components/UploadForm';
import AudioInput from '../components/AudioInput';
import OutputDisplay from '../components/OutputDisplay';
import { apiService } from '../services/api';

const Home = () => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [processedImageUrl, setProcessedImageUrl] = useState(null);
  const [currentMode, setCurrentMode] = useState(null);
  const [transcript, setTranscript] = useState('');

  const handleImageSubmit = async (data) => {
    const { file, mode, question } = data;
    
    setLoading(true);
    setError(null);
    setResult(null);
    setCurrentMode(mode);

    // Create preview URL for the uploaded image
    const imageUrl = URL.createObjectURL(file);
    setProcessedImageUrl(imageUrl);

    try {
      let response;
      
      switch (mode) {
        case 'caption':
          response = await apiService.captionImage(file);
          break;
        case 'ocr':
          response = await apiService.extractText(file);
          break;
        case 'vqa':
          response = await apiService.answerQuestion(file, question);
          break;
        default:
          throw new Error('Invalid mode selected');
      }

      setResult(response);
    } catch (err) {
      console.error('Processing error:', err);
      setError(err.message || 'An error occurred while processing the image');
    } finally {
      setLoading(false);
    }
  };

  const handleAudioTranscript = async (audioFile) => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.transcribeAudio(audioFile);
      setTranscript(response.transcript || '');
    } catch (err) {
      console.error('Transcription error:', err);
      setError(err.message || 'An error occurred while transcribing audio');
    } finally {
      setLoading(false);
    }
  };

  const clearResults = () => {
    setResult(null);
    setError(null);
    setTranscript('');
    setCurrentMode(null);
    if (processedImageUrl) {
      URL.revokeObjectURL(processedImageUrl);
      setProcessedImageUrl(null);
    }
  };

  const features = [
    {
      icon: Eye,
      title: 'Image Captioning',
      description: 'AI-powered descriptions of images for accessibility',
      color: 'text-blue-600'
    },
    {
      icon: FileText,
      title: 'Text Extraction',
      description: 'OCR technology to extract text from images',
      color: 'text-green-600'
    },
    {
      icon: MessageCircle,
      title: 'Visual Q&A',
      description: 'Ask questions about images and get AI answers',
      color: 'text-purple-600'
    },
    {
      icon: Volume2,
      title: 'Text-to-Speech',
      description: 'Convert any text output to natural speech',
      color: 'text-orange-600'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center">
            <div className="flex items-center justify-center mb-2">
              <Sparkles className="h-8 w-8 text-primary-600 mr-2" />
              <h1 className="text-3xl font-bold text-gray-900">VisoVox AI</h1>
            </div>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Empowering the visually impaired with intelligent image captioning and speech output
            </p>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {features.map((feature, index) => {
            const IconComponent = feature.icon;
            return (
              <div
                key={index}
                className="bg-white rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow duration-200"
              >
                <div className="flex items-center mb-3">
                  <IconComponent className={`h-6 w-6 ${feature.color} mr-2`} />
                  <h3 className="font-semibold text-gray-900">{feature.title}</h3>
                </div>
                <p className="text-sm text-gray-600">{feature.description}</p>
              </div>
            );
          })}
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Image Upload */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-6 text-gray-800">
                Upload Image for Analysis
              </h2>
              <UploadForm onSubmit={handleImageSubmit} loading={loading} />
            </div>
          </div>

          {/* Right Column - Audio Input */}
          <div className="lg:col-span-1">
            <div className="space-y-6">
              <AudioInput onTranscript={handleAudioTranscript} loading={loading} />
              
              {/* Transcript Display */}
              {transcript && (
                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <h4 className="font-medium text-gray-800 mb-2">Transcript:</h4>
                  <p className="text-gray-600 text-sm bg-gray-50 rounded p-3">
                    {transcript}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-8 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
              <div className="ml-auto">
                <button
                  onClick={() => setError(null)}
                  className="text-red-400 hover:text-red-600"
                >
                  <span className="sr-only">Dismiss</span>
                  <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Results Display */}
        {result && (
          <div className="mt-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-800">Results</h2>
              <button
                onClick={clearResults}
                className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
              >
                Clear Results
              </button>
            </div>
            <OutputDisplay
              result={result}
              mode={currentMode}
              imageUrl={processedImageUrl}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <p className="text-gray-600 mb-2">
              Built with ❤️ for accessibility
            </p>
            <p className="text-sm text-gray-500">
              © 2025 VisoVox AI - Ayobankole (Grok Member)
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;