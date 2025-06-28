import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:10000',
  timeout: 30000,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// Request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.message || 'Server error occurred';
      throw new Error(message);
    } else if (error.request) {
      // Request made but no response received
      throw new Error('No response from server. Please check your connection.');
    } else {
      // Something else happened
      throw new Error('An unexpected error occurred');
    }
  }
);

// API service functions
export const apiService = {
  // Image captioning
  captionImage: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/caption/', formData);
    return response.data;
  },

  // OCR - Extract text from image
  extractText: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/ocr/', formData);
    return response.data;
  },

  // VQA - Visual Question Answering
  answerQuestion: async (file, question) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('question', question);
    
    const response = await api.post('/api/vqa/', formData);
    return response.data;
  },

  // Audio transcription
  transcribeAudio: async (audioFile) => {
    const formData = new FormData();
    formData.append('audio', audioFile);
    
    const response = await api.post('/api/audio/transcribe', formData);
    return response.data;
  },

  // Text to speech
  speakText: async (text) => {
    const formData = new FormData();
    formData.append('text', text);
    
    const response = await api.post('/api/audio/speak', formData, {
      responseType: 'blob',
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    // Create blob URL for audio playback
    const audioBlob = new Blob([response.data], { type: 'audio/mpeg' });
    return URL.createObjectURL(audioBlob);
  },
};

export default api;