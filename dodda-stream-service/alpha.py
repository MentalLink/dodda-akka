import sounddevice as sd
import numpy as np
import wave
import webrtcvad
import collections
import google.generativeai as genai
import sys
import time
from pathlib import Path

class LiveAudioTranscriber:
    def __init__(self, sample_rate=16000, vad_mode=2):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_mode)
        self.buffer = collections.deque(maxlen=30)  # 30 frames buffer
        self.recording = []
        self.is_speaking = False
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 30  # Number of silent frames before stopping
        
        # Configure Gemini
        genai.configure(api_key='API_KEY')
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print('Error:', status)
        
        # Convert to mono if needed and ensure correct format
        audio_data = np.mean(indata, axis=1) if len(indata.shape) > 1 else indata
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Process frame for VAD
        frame = audio_data.tobytes()
        is_speech = self.vad.is_speech(frame, self.sample_rate)
        
        self.buffer.append(frame)
        
        if is_speech:
            if not self.is_speaking:
                print("Speech started")
                self.is_speaking = True
                # Add buffer contents to capture pre-speech audio
                self.recording.extend(list(self.buffer))
            self.recording.append(frame)
            self.silence_frames = 0
        else:
            if self.is_speaking:
                self.silence_frames += 1
                self.recording.append(frame)
                if self.silence_frames >= self.SILENCE_THRESHOLD:
                    if len(self.recording)>100:
                        print("Speech ended")
                        self.save_and_transcribe()
                        self.recording = []
                        self.is_speaking = False
                        self.silence_frames = 0
                    else:
                        print("Speech Ended")
                        self.recording = []
                        self.is_speaking = False
                        self.silence_frames = 0


    def save_and_transcribe(self):
        """Save recorded audio and send to Gemini API"""
        if not self.recording:
            return
        
        # Save to WAV file
        temp_file = "temp_recording.wav"
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.recording))
        
        # Read audio file for Gemini
        audio_data = {
            "mime_type": "audio/wav",
            "data": open(temp_file, "rb").read()
        }
        
        # Generate transcript
        prompt = "You are an helpful assistant whose purpose is to transcribe speech flawlessly. You need to figure out what language the audio is being spoken in. If it is an Indian Language, you need to translate it and output in English. If it is in English, you should output a word for word transcription. You must STRICTLY output only the translation or transcription of what was said. No explanations."
        try:
            response = self.model.generate_content([prompt, audio_data])
            print("\nTranscription:", response.text)
        except Exception as e:
            print(f"Error in transcription: {e}")
        
        # Clean up
        Path(temp_file).unlink()

    def start_recording(self):
        """Start the audio stream"""
        try:
            with sd.InputStream(callback=self.audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=int(self.sample_rate * 30/1000),  # 30ms frames
                              dtype=np.float32):
                print("Recording... Press Ctrl+C to stop")
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nRecording stopped")

if __name__ == "__main__":
    transcriber = LiveAudioTranscriber()
    transcriber.start_recording()