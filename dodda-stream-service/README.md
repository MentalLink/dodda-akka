# Voice-Based First Responder Assistant

A real-time voice interaction system that provides assistance through voice conversations. The system uses Gemini AI for processing responses and supports multiple languages.

## Project Overview

This system provides:
- Real-time voice transcription
- Multilingual support (English, Hindi, Tamil, Kannada)
- AI responses using Google's Gemini
- Text-to-speech output
- Conversation history tracking

## Dependencies

Required packages:
bash
sounddevice
numpy
wave
webrtcvad
google-generativeai
google-cloud-texttospeech
soundfile
requests


## Setup Instructions

1. Install required packages:

bash
pip install sounddevice numpy wave webrtcvad google-generativeai google-cloud-texttospeech soundfile requests

2. Set up credentials:
   - Place Google Cloud credentials JSON file in project directory
   - Update Gemini API key in code 
   - Update Sarvam API key in code 

3. Ensure you have:
   - Working microphone
   - Working speakers
   - Internet connection

## Running the Application

1. Run the script:
bash
python beta.py


2. The system will start with "Recording... Press Ctrl+C to stop"

3. Start speaking when ready

4. Press Ctrl+C to stop recording

## Usage Examples

### Basic Conversation Flow:
1. Start the program
2. Speak into your microphone
3. System will:
   - Detect when you stop speaking
   - Process your speech
   - Generate a response
   - Play the response through speakers

### Language Support:
- Speak in any supported language (English, Hindi, Tamil, Kannada)
- System automatically detects the language
- Responds in the same language

### End Conversation:
- Say "okay thank you, i want the call to end"
- System will provide closing message with FIR details

## Features

- Voice activity detection
- Real-time speech processing
- Multilingual support
- Text-to-speech response
- Conversation history tracking
- Context-aware responses
- FIR information support

## Troubleshooting

If you encounter issues:
1. Check microphone connection and permissions
2. Verify speaker/headphone connection
3. Ensure internet connection is stable
4. Verify API credentials are correct
