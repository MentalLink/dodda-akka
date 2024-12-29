import os
import logging
import torch
import numpy as np
from TTS.api import TTS
import tempfile
import soundfile as sf
from typing import Optional

logger = logging.getLogger(__name__)

class CoquiTTSService:
    def __init__(self):
        try:
            # Initialize TTS with a fast and high-quality model
            self.tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch",
                          progress_bar=False,
                          gpu=torch.cuda.is_available())
            
            # Set parameters for voice generation
            self.sample_rate = 22050  # Standard sample rate for this model
            
        except Exception as e:
            logger.error(f"Error initializing TTS service: {str(e)}")
            raise

    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """
        Convert text to speech using Coqui TTS
        """
        try:
            # Generate speech
            wav = self.tts.tts(text)
            
            # Convert numpy array to bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                sf.write(temp_file.name, wav, self.sample_rate)
                with open(temp_file.name, "rb") as audio_file:
                    return audio_file.read()
                    
        except Exception as e:
            logger.error(f"Error in speech synthesis: {str(e)}")
            # Return error tone or pre-recorded message
            return self._generate_error_audio()
            
    def _generate_error_audio(self) -> bytes:
        """
        Generate a simple error tone as fallback
        """
        # Generate a simple beep sound
        duration = 0.5  # seconds
        frequency = 440  # Hz
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Convert to bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            sf.write(temp_file.name, audio, self.sample_rate)
            with open(temp_file.name, "rb") as audio_file:
                return audio_file.read()

    async def process_stream(self, text_stream):
        """
        Process streaming text input
        """
        async for text in text_stream:
            audio = await self.synthesize_speech(text)
            if audio:
                yield audio 