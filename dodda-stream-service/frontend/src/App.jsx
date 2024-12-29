import React, { useState, useEffect } from 'react';
import './App.css';
import PhoneInterface from './components/PhoneInterface';
import TranscriptView from './components/TranscriptView';

function App() {
  const [transcripts, setTranscripts] = useState([]);
  const [isRecording, setIsRecording] = useState(false);

  return (
    <div className="app-container">
      <PhoneInterface 
        isRecording={isRecording}
        setIsRecording={setIsRecording}
      />
      <TranscriptView transcripts={transcripts} />
    </div>
  );
}

export default App; 