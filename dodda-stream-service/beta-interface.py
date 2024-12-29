import sys
import sounddevice as sd
import typing
import numpy as np
import wave
import webrtcvad
import collections
import google.generativeai as genai
from google.cloud import texttospeech
from google.oauth2 import service_account
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import typing_extensions as typing
import json
import streamlit as st
import io
from pydub import AudioSegment
from pydub.playback import play
import threading
import queue

@dataclass
class ConversationTurn:
    speaker: str
    text: str
    original_language: str
    english_translation: Optional[str] = None
    is_question: bool = False

class TranscriptionPrompt(typing.TypedDict):
    english_transcription: str

class ConversationHistory:
    def __init__(self, max_turns=10):
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns
    
    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
    
    def get_context_string(self):
        return "\n".join([
            f"{turn.speaker}: {turn.english_translation or turn.text}"
            for turn in self.turns[-self.max_turns:]
        ])

class TTSService:
    def __init__(self, credentials_path="../durable-pulsar-413916-03082340547d.json"):
        self.credentials_path = credentials_path
        self.client = texttospeech.TextToSpeechClient(credentials=service_account.Credentials.from_service_account_file(credentials_path))
    
    def synthesize_text(self, text: str, language_code: str) -> bytes:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        # breakpoint()
        return response.audio_content

class LiveAudioTranscriberWeb:
    def __init__(self, sample_rate=16000, vad_mode=2):
        # Initialize all the same components as before
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_mode)
        self.buffer = collections.deque(maxlen=30)
        self.recording = []
        self.is_speaking = False
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 30
        
        self.tts_service = TTSService()
        self.history = ConversationHistory()
        
        # Keep the same domain context and model configuration
        self.context_domain="""
The following is your context - Fundamentals of Becoming an Active Bystander for Gender Equality

Gender equality is a cornerstone of a just and equitable society, yet achieving it requires collective effort. Active bystanders play a crucial role in challenging gender-based discrimination, harassment, and inequality. To effectively become an active bystander, one must cultivate awareness, skills, and strategies. Below are the key fundamentals to guide this journey:

1. Understanding the Role of an Active Bystander

An active bystander is someone who recognizes a problematic situation and takes action to prevent or intervene in behavior that perpetuates inequality or harm. This includes standing up against:

Gender-based discrimination.

Harassment and violence.

Stereotyping and microaggressions.

2. Educating Oneself on Gender Equality and Diversity

To act effectively, it is essential to understand the issues surrounding gender equality and diversity.

Learn the Basics: Familiarize yourself with key terms such as gender identity, intersectionality, and unconscious bias.

Study the Data: Understand the disparities in employment, education, and representation that persist across genders.

Seek Out Perspectives: Read books, attend workshops, or listen to voices from diverse gender experiences.

Engage in Reflection: Examine your own biases and privileges to recognize areas for growth.

3. Recognizing Situations Requiring Intervention

The first step in intervention is identifying situations where action is needed. These may include:

Overt Crimes: Sexual harassment, assault, or gender-based violence.

Subtle Discrimination: Unequal opportunities, pay gaps, or biased language.

Microaggressions: Seemingly small but impactful acts of exclusion or stereotyping.

4. Adopting Effective Strategies for Intervention

Intervening as an active bystander can take different forms, often referred to as the "4 D's of Bystander Intervention":

Direct Action:

Confront the behavior directly but safely.

Example: "That comment seems inappropriate. Let's discuss why it might be harmful."

Distraction:

Defuse the situation without direct confrontation.

Example: Changing the topic or creating a diversion to interrupt the behavior.

Delegation:

Seek help from others who may be better equipped to address the situation, such as supervisors or authorities.

Example: Reporting harassment to HR or law enforcement.

Delay:

If immediate action isn't possible, follow up later.

Example: Checking in with the affected individual after the incident to offer support.

5. Ensuring Safety When Intervening

Safety is paramount for both the bystander and the individuals involved.

Assess the Risk: Evaluate whether the situation poses physical danger.

Stay Calm: Avoid escalating tension.

Call for Help: If the situation involves immediate danger, contact authorities.

6. Building Allyship and Advocacy Skills

Active bystanders also work proactively to prevent gender inequality by being allies and advocates.

Challenge Everyday Bias: Speak up when you notice discriminatory jokes, stereotypes, or assumptions.

Support Representation: Advocate for policies and practices that promote gender diversity in leadership and decision-making.

Mentor and Sponsor: Support individuals from underrepresented genders in professional and personal growth.

7. Supporting Those Affected

Providing support to victims or those impacted by gender-based inequality is essential.

Listen Actively: Validate their experiences without judgment or interruption.

Offer Resources: Guide them to counseling services, legal aid, or support groups.

Respect Their Choices: Allow them to decide how they want to proceed.

8. Practicing Resilience and Self-Care

Engaging in active bystander behavior can be emotionally taxing. To sustain efforts:

Seek Support: Share your experiences with like-minded individuals or groups.

Educate Continuously: Stay informed to feel empowered in addressing issues.

Maintain Balance: Take time for self-care to avoid burnout.

9. Creating a Culture of Accountability

Active bystanders contribute to broader cultural change by fostering accountability:

Promote Dialogue: Encourage open discussions about gender equality in your community.

Model Behavior: Demonstrate respect, inclusivity, and equality in your actions.

Hold Institutions Accountable: Advocate for transparent policies that address inequality and discrimination.
        """
        # Sample domain-specific context (can be loaded from files)
        self.domain_context = f"""
        You are a calm, empathetic, and professional assistant for an active bystander and first responder hotline, seamlessly integrated into a voice-to-voice system. Your role is to assist users by providing brief, clear, and actionable responses, focusing on the immediate needs of the conversation. You are knowledgeable in topics related to bystander intervention, crisis response, and supporting individuals in distress.

        Always maintain a supportive and non-judgmental tone, prioritizing user comfort and clarity. Your responses should be concise, limited to one or two sentences, and relevant to the user's immediate context, similar to natural phone call turns. Avoid overwhelming users with information—offer only what is necessary and actionable. If further assistance is needed, guide the user towards next steps or suggest connecting with a human representative when appropriate.
        {self.context_domain}
        """
        
        genai.configure(api_key='AIzaSyBco7C-oBK-6mx6qYIznGrTw4z9Ky2NdYk')
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.model_pro = genai.GenerativeModel('gemini-1.5-pro')
        
        self.supported_languages = {
            "hindi": "hi",
            "tamil": "ta",
            "kannada": "kn",
        }
        
        # Add audio processing queue
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None  # Initialize stream as None

    def start_recording_stream(self):
        if self.stream is not None:  # Check if stream already exists
            self.stop_recording_stream()  # Stop existing stream
            
        self.is_recording = True
        def audio_callback(indata, frames, time, status):
            if status:
                print('Error:', status)
            
            audio_data = np.mean(indata, axis=1) if len(indata.shape) > 1 else indata
            audio_data = (audio_data * 32767).astype(np.int16)
            
            frame = audio_data.tobytes()
            self.audio_queue.put(frame)

        try:
            self.stream = sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 30/1000),
                dtype=np.float32
            )
            self.stream.start()
        except Exception as e:
            st.error(f"Error starting audio stream: {e}")
            self.is_recording = False

    def stop_recording_stream(self):
        self.is_recording = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                st.error(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None

    def process_audio_frames(self):
        while self.is_recording:
            try:
                if not self.audio_queue.empty():
                    frame = self.audio_queue.get()
                    is_speech = self.vad.is_speech(frame, self.sample_rate)
                    
                    self.buffer.append(frame)
                    
                    if is_speech:
                        if not self.is_speaking:
                            st.session_state.status = "Recording Started..."
                            self.is_speaking = True
                            self.recording.extend(list(self.buffer))
                        self.recording.append(frame)
                        self.silence_frames = 0
                    else:
                        if self.is_speaking:
                            self.silence_frames += 1
                            self.recording.append(frame)
                            
                            if self.silence_frames >= self.SILENCE_THRESHOLD:
                                if len(self.recording) > 100:
                                    st.session_state.status = "Processing Speech..."
                                    # Stop recording before processing
                                    self.stop_recording_stream()
                                    self.process_speech()
                                    self.recording = []
                                    self.is_speaking = False
                                    self.silence_frames = 0
                                    # Restart recording after TTS finishes
                                    self.start_recording_stream()
                                    st.session_state.status = "Listening..."
                                else:
                                    st.session_state.status = "Recording too short, discarded"
                                    self.recording = []
                                    self.is_speaking = False
                                    self.silence_frames = 0
            except Exception as e:
                st.error(f"Error processing audio frame: {e}")
                break

    def process_speech(self):
        if not self.recording:
            return
        print("MUSTARDDDDDDDDD")
        temp_file = "temp_recording.wav"
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.recording))
        
        audio_data = {
            "mime_type": "audio/wav",
            "data": open(temp_file, "rb").read()
        }

        try:
            print("hvgcghc")
            transcription_prompt = "Please provide a word-for-word transcription of the speech as it is spoken in English. Ensure that the output is in JSON format. Strictly stick to the format which is {\"english_transcription\": \"<transcribed_text>\"}}."
            response = self.model.generate_content(
                [transcription_prompt, audio_data],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=TranscriptionPrompt
                )
            )
            
            transcribed_text = json.loads(response.text)["english_transcription"]
            # Use rerun context to safely update session state
            # with st.rerun_context():
            #     st.session_state.messages.append({"role": "user", "content": transcribed_text})
            print(transcribed_text)
            ai_response = self.generate_response(transcribed_text, "english")
            # with st.rerun_context():
            #     st.session_state.messages.append({"role": "assistant", "content": ai_response})
            print(ai_response)
            # Play TTS response
            audio_content = self.tts_service.synthesize_text(ai_response, "en-US")
            
            # Save audio content to temporary file
            temp_audio = "temp_response.mp3"
            with open(temp_audio, "wb") as out:
                out.write(audio_content)
            
            # Read and play the audio file
            audio_segment = AudioSegment.from_file(temp_audio)
            play(audio_segment)
            
            # Clean up temp file
            if Path(temp_audio).exists():
                Path(temp_audio).unlink()
            
        except Exception as e:
            st.error(f"Error processing speech: {e}")
        finally:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
    
    def generate_response(self, text, source_language):
        """Generate contextual response based on conversation history"""
        # Translate input to English if needed
        # english_text = self.translate_to_english(text, source_language)
        
        # Analyze input
        # analysis = self.analyze_input(english_text)
        
        # Add user turn to history
        user_turn = ConversationTurn(
            speaker="User",
            text=text,
            original_language=source_language,
            english_translation=text,
            is_question="NA"
        )
        self.history.add_turn(user_turn)
        
        # Generate response prompt
        prompt = f"""
        {self.domain_context}
        
        Conversation history:
        {self.history.get_context_string()}
        
        Based on the above context and conversation history, provide a detailed response to the user query or statement. If it's a question, answer it with relevant information. If it's a statement, acknowledge and provide additional insights. You need to ensure that the response is accurate and informative and directly from the provided context. Do not be very verbose and stick to the point. Be very concise and provide as much help as possible and have the kindest and most helpful tone possible.
        
        Remember to:
        1. Stay within the provided domain context
        2. Reference relevant parts of the conversation history but keep it extremely concise.
        """
        
        # Generate response in English
        response = self.model.generate_content(prompt, generation_config=genai.GenerationConfig(max_output_tokens=100))
        final_response = english_response = response.text.strip()
        
        # Translate response back to source language if needed
        # final_response = self.translate_from_english(
        #     english_response, source_language
        # )
        # breakpoint()
        # Add assistant's response to history
        assistant_turn = ConversationTurn(
            speaker="Assistant",
            text=final_response,
            original_language=source_language,
            english_translation=english_response
        )
        self.history.add_turn(assistant_turn)
        
        return final_response

def main():
    st.set_page_config(page_title="Voice Assistant Interface", layout="wide")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "status" not in st.session_state:
        st.session_state.status = "Idle"
    
    if "transcriber" not in st.session_state:
        st.session_state.transcriber = LiveAudioTranscriberWeb()
    
    st.title("Voice Assistant Interface")
    
    # Status indicator
    st.markdown(f"**Status:** {st.session_state.status}")
    
    # Recording controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording"):
            st.session_state.status = "Listening..."
            st.session_state.transcriber.start_recording_stream()
            threading.Thread(target=st.session_state.transcriber.process_audio_frames, daemon=True).start()
    
    with col2:
        if st.button("Stop Recording"):
            st.session_state.status = "Stopped"
            st.session_state.transcriber.stop_recording_stream()
    
    # Conversation history
    st.markdown("### Conversation History")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if __name__ == "__main__":
    main()