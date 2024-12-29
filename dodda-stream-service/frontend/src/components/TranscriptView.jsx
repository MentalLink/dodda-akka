import React from 'react';
import './TranscriptView.css';

const TranscriptView = ({ transcripts }) => {
  return (
    <div className="transcript-container">
      <h2>Conversation</h2>
      <div className="transcript-messages">
        {transcripts.map((message, index) => (
          <div 
            key={index} 
            className={`message ${message.speaker.toLowerCase()}`}
          >
            <div className="message-speaker">{message.speaker}</div>
            <div className="message-text">{message.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TranscriptView; 