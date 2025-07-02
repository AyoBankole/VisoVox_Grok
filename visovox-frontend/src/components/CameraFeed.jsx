import React, { useEffect, useRef } from 'react';

export default function CameraFeed({ videoRef: externalVideoRef }) {
  const internalVideoRef = useRef();
  const videoRef = externalVideoRef || internalVideoRef;

  useEffect(() => {
    async function enableCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
    }

    enableCamera();
  }, [videoRef]);

  return (
    <div className="camera-area w-full max-w-md mx-auto rounded-lg overflow-hidden shadow-md bg-black aspect-video" aria-label="Live camera feed">
      <video ref={videoRef} autoPlay playsInline className="camera-video w-full h-auto object-cover" />
    </div>
  );
}