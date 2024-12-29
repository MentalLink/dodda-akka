import os
import logging
import openai
from typing import Optional
import tempfile
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

class WhisperASRService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        openai.api_key = self.api_key

    async def transcribe_audio(self, audio_chunk: bytes) -> Optional[str]:
        """
        Transcribe audio chunk using Whisper API
        """
        try:
            # Save audio chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                # Convert bytes to numpy array and save as WAV
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                sf.write(temp_file.name, audio_data, samplerate=16000)
                
                # Transcribe using Whisper API
                with open(temp_file.name, "rb") as audio_file:
                    response = await openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                    
                return response

        except Exception as e:
            logger.error(f"Error in ASR transcription: {str(e)}")
            return None

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