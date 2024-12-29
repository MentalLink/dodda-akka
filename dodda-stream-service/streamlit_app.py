import streamlit as st
import threading
import time
from beta import LiveAudioTranscriber
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue

class StreamlitInterface:
    def __init__(self):
        # Initialize transcriber as a regular instance variable
        self.transcriber = LiveAudioTranscriber()
        
        # Message queue for thread-safe communication
        self.message_queue = queue.Queue()
        
        # Thread-safe variables
        self._recording = False
        self._muted = False
        self._thread = None
        
        # Initialize session state variables
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize all session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.call_active = False
            st.session_state.is_muted = False
            st.session_state.messages = []
            st.session_state.last_message_count = 0
        
    def play_audio_response(self, file_path="output_audio.mp3"):
        try:
            if Path(file_path).exists():
                print(f"Playing audio from {file_path}")
                data, samplerate = sf.read(file_path)
                sd.play(data, samplerate)
                sd.wait()
                print("Audio playback completed")
        except Exception as e:
            print(f"Error playing audio: {e}")

    def recording_thread(self):
        while self._recording:
            if not self._muted:
                try:
                    with sd.InputStream(callback=self.transcriber.audio_callback,
                                    channels=1,
                                    samplerate=self.transcriber.sample_rate,
                                    blocksize=int(self.transcriber.sample_rate * 30/1000),
                                    dtype=np.float32):
                        while self._recording and not self._muted:
                            time.sleep(0.1)
                            # Check if new message is available
                            if self.transcriber.response_ready.is_set():
                                if hasattr(self.transcriber, 'history') and self.transcriber.history.turns:
                                    # Put new messages in queue
                                    new_messages = self.transcriber.history.turns[len(st.session_state.messages):]
                                    for turn in new_messages:
                                        self.message_queue.put((turn.speaker, turn.text))
                                self.transcriber.response_ready.clear()
                except Exception as e:
                    print(f"Recording error: {e}")
            time.sleep(0.1)

    def start_call(self):
        st.session_state.call_active = True
        self._recording = True
        self._muted = False
        self._thread = threading.Thread(target=self.recording_thread)
        self._thread.daemon = True
        self._thread.start()

    def end_call(self):
        st.session_state.call_active = False
        self._recording = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)

    def toggle_mute(self):
        st.session_state.is_muted = not st.session_state.is_muted
        self._muted = st.session_state.is_muted

    def update_conversation(self):
        """Update conversation from message queue"""
        updated = False
        while not self.message_queue.empty():
            try:
                speaker, text = self.message_queue.get_nowait()
                st.session_state.messages.append((speaker, text))
                updated = True
            except queue.Empty:
                break
        return updated

    def display_messages(self):
        for speaker, text in st.session_state.messages:
            if speaker == "User":
                st.markdown(f"""
                    <div style='background-color: #e6f3ff; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                        🗣️ <b>You:</b> {text}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                        🤖 <b>Assistant:</b> {text}
                    </div>
                """, unsafe_allow_html=True)
                # Play audio for assistant responses
                self.play_audio_response()

    def run(self):
        st.title("Voice Assistant Call Interface")
        
        # Call control buttons in a horizontal layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not st.session_state.call_active:
                if st.button("Start Call", type="primary"):
                    self.start_call()
            else:
                if st.button("End Call", type="secondary"):
                    self.end_call()
        
        with col2:
            if st.session_state.call_active:
                if not st.session_state.is_muted:
                    if st.button("Mute"):
                        self.toggle_mute()
                else:
                    if st.button("Unmute"):
                        self.toggle_mute()

        # Display call status
        st.markdown("---")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.markdown(f"**Call Status:** {'Active' if st.session_state.call_active else 'Inactive'}")
        with status_col2:
            st.markdown(f"**Microphone:** {'Muted' if st.session_state.is_muted else 'Active'}")

        # Display conversation history
        st.markdown("---")
        st.markdown("### Conversation History")
        
        # Update and display messages
        self.update_conversation()
        self.display_messages()

        # Add custom CSS
        st.markdown("""
            <style>
            .stButton button {
                width: 100%;
                border-radius: 20px;
                height: 50px;
            }
            .css-1d391kg {
                padding: 1rem;
                border-radius: 10px;
                background-color: #f0f2f6;
                margin-bottom: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)

        # Auto-refresh when call is active
        if st.session_state.call_active:
            time.sleep(0.1)
            st.rerun()

if __name__ == "__main__":
    interface = StreamlitInterface()
    interface.run() 