<p align="center">
  <img src="vivox.png" alt="VisoVox Logo" width="200"/>
</p>

# VisoVox AI Frontend

A React-based frontend for the VisoVox AI visual assistant application that provides accessibility features through AI-powered image analysis and text-to-speech capabilities.

## 🚀 Features

- **Image Upload & Capture**: Upload images or capture them using device camera
- **AI Image Captioning**: Generate descriptive captions for images
- **OCR Text Extraction**: Extract text from images using AI
- **Visual Question Answering**: Ask questions about uploaded images
- **Audio Recording**: Record voice questions and commands
- **Text-to-Speech**: Convert any text output to natural speech
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Accessibility First**: Built with screen readers and keyboard navigation in mind

## 🛠️ Tech Stack

- **React 18** - Modern React with hooks
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client for API calls

## 📋 Prerequisites

- Node.js 18 or higher
- npm or yarn package manager
- Running VisoVox backend server

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd visovox-frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and update the API URL:
   ```
   VITE_API_URL=http://localhost:10000
   ```

4. **Start development server**
   ```bash
   npm run dev
   ```

5. **Open your browser**
   Navigate to `http://localhost:3000`

## 🏗️ Project Structure

```
visovox-frontend/
├── public/
│   └── favicon.ico
├── src/
│   ├── assets/                    # Static assets
│   ├── components/               # Reusable components
│   │   ├── UploadForm.jsx        # Image upload and mode selection
│   │   ├── AudioInput.jsx        # Voice recording component
│   │   └── OutputDisplay.jsx     # Results display with TTS
│   ├── pages/
│   │   └── Home.jsx              # Main application page
│   ├── services/
│   │   └── api.js                # API integration layer
│   ├── App.jsx                   # Root component
│   ├── main.jsx                  # Entry point
│   ├── index.css                 # Global styles
│   └── App.css                   # Component styles
├── .env                          # Environment variables
├── netlify.toml                  # Netlify deployment config
├── package.json
├── tailwind.config.js
├── postcss.config.js
├── vite.config.js
└── README.md
```

## 🔧 API Integration

The frontend communicates with the VisoVox backend through these endpoints:

- `POST /api/caption/` - Image captioning
- `POST /api/ocr/` - Text extraction from images
- `POST /api/vqa/` - Visual question answering
- `POST /api/audio/transcribe` - Audio transcription
- `POST /api/audio/speak` - Text-to-speech generation

## 🎨 Component Overview

### UploadForm
- Handles image upload via file picker or camera
- Mode selection (Caption, OCR, VQA)
- Question input for VQA mode
- Drag & drop support

### AudioInput
- Voice recording functionality
- Audio file upload
- Playback controls
- Integration with Whisper transcription

### OutputDisplay
- Results presentation
- Text-to-speech playback
- Copy to clipboard
- Download functionality

## 🚀 Build & Deployment

### Development
```bash
npm run dev
```

### Production Build
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

### Deploy to Netlify
1. Connect your repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `dist`
4. Add environment variables in Netlify dashboard
5. Deploy!

## 🔐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API base URL | `http://localhost:10000` |
| `VITE_APP_NAME` | Application name | `VisoVox AI` |
