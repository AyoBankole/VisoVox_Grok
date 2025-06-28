import React, { useState, useRef } from 'react';
import { Upload, Camera, Image, FileText, MessageCircleQuestion, Loader2 } from 'lucide-react';

const UploadForm = ({ onSubmit, loading }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mode, setMode] = useState('caption'); // caption, ocr, vqa
  const [question, setQuestion] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  const modes = [
    { id: 'caption', label: 'Image Caption', icon: Image, description: 'Get AI description of the image' },
    { id: 'ocr', label: 'Extract Text', icon: FileText, description: 'Extract text from the image' },
    { id: 'vqa', label: 'Ask Question', icon: MessageCircleQuestion, description: 'Ask questions about the image' },
  ];

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setPreviewUrl(e.target.result);
      reader.readAsDataURL(file);
    } else {
      alert('Please select a valid image file.');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragActive(false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!selectedFile) {
      alert('Please select an image first.');
      return;
    }

    if (mode === 'vqa' && !question.trim()) {
      alert('Please enter a question for the image.');
      return;
    }

    onSubmit({
      file: selectedFile,
      mode,
      question: mode === 'vqa' ? question : null,
    });
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Mode Selection */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3 text-gray-800">Choose Action</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {modes.map((m) => {
            const IconComponent = m.icon;
            return (
              <button
                key={m.id}
                type="button"
                onClick={() => setMode(m.id)}
                className={`p-4 rounded-lg border-2 transition-all duration-200 text-left ${
                  mode === m.id
                    ? 'border-primary-500 bg-primary-50 text-primary-700'
                    : 'border-gray-200 hover:border-gray-300 text-gray-600'
                }`}
              >
                <div className="flex items-center mb-2">
                  <IconComponent className="h-5 w-5 mr-2" />
                  <span className="font-medium">{m.label}</span>
                </div>
                <p className="text-sm opacity-75">{m.description}</p>
              </button>
            );
          })}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File Upload Area */}
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive
              ? 'border-primary-400 bg-primary-50'
              : selectedFile
              ? 'border-green-400 bg-green-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          {previewUrl ? (
            <div className="space-y-4">
              <img
                src={previewUrl}
                alt="Preview"
                className="max-h-64 mx-auto rounded-lg shadow-md"
              />
              <div className="flex justify-center space-x-2">
                <p className="text-sm text-green-600 font-medium">
                  ✓ {selectedFile.name}
                </p>
                <button
                  type="button"
                  onClick={clearFile}
                  className="text-sm text-red-500 hover:text-red-700 underline"
                >
                  Remove
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <Upload className="h-12 w-12 mx-auto text-gray-400" />
              <div>
                <p className="text-lg font-medium text-gray-700 mb-2">
                  Upload or capture an image
                </p>
                <p className="text-sm text-gray-500 mb-4">
                  Drag and drop an image here, or click to select
                </p>
                
                <div className="flex justify-center space-x-4">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    Choose File
                  </button>
                  
                  <button
                    type="button"
                    onClick={() => cameraInputRef.current?.click()}
                    className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    <Camera className="h-4 w-4 mr-2" />
                    Take Photo
                  </button>
                </div>
              </div>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => handleFileSelect(e.target.files[0])}
            className="hidden"
          />
          
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={(e) => handleFileSelect(e.target.files[0])}
            className="hidden"
          />
        </div>

        {/* Question Input for VQA */}
        {mode === 'vqa' && (
          <div className="animate-slide-up">
            <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-2">
              What would you like to know about this image?
            </label>
            <input
              type="text"
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., What objects are in this image? What color is the car?"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors"
              disabled={loading}
            />
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || !selectedFile || (mode === 'vqa' && !question.trim())}
          className="w-full flex items-center justify-center px-6 py-3 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
        >
          {loading ? (
            <>
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              {mode === 'caption' && 'Generate Caption'}
              {mode === 'ocr' && 'Extract Text'}
              {mode === 'vqa' && 'Ask Question'}
            </>
          )}
        </button>
      </form>
    </div>
  );
};

export default UploadForm;