import React, { useEffect, useRef } from 'react';

export default function CameraFeed() {
  const videoRef = useRef();

  useEffect(() => {
    async function enableCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
    }

    enableCamera();
  }, []);

  return (
    <div className="camera-area" aria-label="Live camera feed">
      <video ref={videoRef} autoPlay playsInline className="camera-video" />
    </div>
  );
}