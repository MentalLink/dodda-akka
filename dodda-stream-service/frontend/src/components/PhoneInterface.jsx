import React from 'react';
import './PhoneInterface.css';

const PhoneInterface = ({ isRecording, setIsRecording }) => {
  const toggleRecording = () => {
    setIsRecording(!isRecording);
    // TODO: Implement WebSocket connection to Python backend
  };

  return (
    <div className="phone-container">
      <div className="phone">
        <div className="phone-screen">
          <div className="call-status">
            {isRecording ? 'In Call' : 'Ready'}
          </div>
        </div>
        <div className="phone-controls">
          <button 
            className={`call-button ${isRecording ? 'active' : ''}`}
            onClick={toggleRecording}
          >
            {isRecording ? 'End Call' : 'Start Call'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default PhoneInterface; 