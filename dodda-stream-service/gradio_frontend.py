import gradio as gr
import numpy as np
import sounddevice as sd
import tempfile
import os
from pathlib import Path
from beta import LiveAudioTranscriber  # Assuming your provided code is saved as live_audio_transcriber.py
import wave
class GradioFrontend:
    def __init__(self):
        self.transcriber = LiveAudioTranscriber()
        self.audio_buffer = []
        self.sample_rate = self.transcriber.sample_rate
        self.is_recording = False

    def start_recording(self):
        """Start recording audio."""
        print("YOOOOOOOOOOO")
        self.audio_buffer = []
        self.is_recording = True
        try:
            print("recording boutta begin")
            with sd.InputStream(callback=self.audio_callback,
                                channels=1,
                                samplerate=self.sample_rate,
                                blocksize=int(self.sample_rate * 30 / 1000),
                                dtype=np.float32):
                while self.is_recording:
                    sd.sleep(100)
        except Exception as e:
            return f"Error: {str(e)}"

    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        return self.process_audio()

    def audio_callback(self, indata, frames, time, status):
        """Callback function to handle incoming audio data."""
        if status:
            print(f"Error: {status}")
        audio_data = np.mean(indata, axis=1) if len(indata.shape) > 1 else indata
        self.audio_buffer.append(audio_data)

    def process_audio(self):
        """Process the recorded audio and get the transcriber response."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(np.concatenate(self.audio_buffer).astype(np.int16).tobytes())

        try:
            response = self.transcriber.process_speech(temp_file.name)
            return response
        finally:
            Path(temp_file.name).unlink()  # Clean up temporary file

    def run(self):
        """Run the Gradio app."""
        def record_and_transcribe():
            """Record audio and process transcription and response."""
            self.start_recording()

        def stop_record_and_transcribe():
            """Stop recording and process the response."""
            response = self.stop_recording()
            return response

        with gr.Blocks() as app:
            with gr.Row():
                gr.Markdown("# Voice-to-Voice Assistant")

            with gr.Row():
                record_btn = gr.Button("Start Recording")
                stop_btn = gr.Button("Stop Recording")

            with gr.Row():
                transcript_output = gr.Textbox(label="Transcription", lines=5)
                response_output = gr.Textbox(label="Assistant Response", lines=5)

            with gr.Row():
                play_audio_btn = gr.Audio(label="Playback Response", type="filepath")

            # Set up event handlers
            record_btn.click(fn=record_and_transcribe, inputs=None, outputs=None)
            stop_btn.click(fn=stop_record_and_transcribe, inputs=None, outputs=[transcript_output, response_output])

        app.launch()

if __name__ == "__main__":
    frontend = GradioFrontend()
    frontend.run()
