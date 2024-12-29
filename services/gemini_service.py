import os
import logging
import google.generativeai as genai
from typing import Optional, Dict, Any
import tempfile
import soundfile as sf
import numpy as np
import pathlib

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        # Initialize Gemini API
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize Gemini model for audio processing and chat
        self.flash_model = genai.GenerativeModel('gemini-1.5-flash')
        self.chat_model = genai.GenerativeModel('gemini-pro')
        
        # System prompt for conversation
        self.system_prompt = """You are a helpful AI assistant engaged in a voice conversation. 
        Keep your responses clear, concise, and natural. Avoid technical jargon unless specifically asked."""
        
        # Initialize conversation history
        self.chat = self.chat_model.start_chat(history=[])

    async def transcribe_audio(self, audio_chunk: bytes) -> Optional[str]:
        """
        Transcribe audio using Gemini Flash
        """
        try:
            # Save audio chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                # Convert bytes to numpy array and save as WAV
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                sf.write(temp_file.name, audio_data, samplerate=16000)
                
                # Create the prompt for transcription
                prompt = "Generate a transcript of the speech."
                
                # Load the audio file and pass it to Gemini
                audio_data = {
                    "mime_type": "audio/wav",
                    "data": pathlib.Path(temp_file.name).read_bytes()
                }
                
                # Generate transcript
                response = self.flash_model.generate_content([prompt, audio_data])
                
                if response and response.text:
                    return response.text.strip()
                
                return None

        except Exception as e:
            logger.error(f"Error in speech transcription: {str(e)}")
            return None

    async def generate_response(self, user_input: str, conversation_state: Dict[str, Any]) -> str:
        """
        Generate response using Gemini Pro
        """
        try:
            # Add conversation context if needed
            if conversation_state.get("history"):
                history = conversation_state["history"]
                # Update chat history with previous interactions
                for interaction in history[-3:]:  # Keep last 3 turns for context
                    self.chat.history.append({
                        "role": "user",
                        "parts": [interaction["user"]]
                    })
                    self.chat.history.append({
                        "role": "model",
                        "parts": [interaction["assistant"]]
                    })
            
            # Generate response
            response = self.chat.send_message(
                f"{self.system_prompt}\n\nUser: {user_input}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=150,
                    top_p=0.95,
                )
            )
            
            return response.text
                    
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            return "I apologize, but I'm having trouble processing your request at the moment. Could you please try again?"

    async def process_stream(self, audio_stream):
        """
        Process streaming audio input
        """
        buffer = b""
        chunk_size = 4096  # Adjust based on your needs
        
        async for chunk in audio_stream:
            buffer += chunk
            if len(buffer) >= chunk_size:
                transcript = await self.transcribe_audio(buffer)
                if transcript:
                    yield transcript
                buffer = b""
        
        # Process remaining buffer
        if buffer:
            transcript = await self.transcribe_audio(buffer)
            if transcript:
                yield transcript