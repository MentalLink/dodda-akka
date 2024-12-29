import json
import sounddevice as sd
import numpy as np
import wave
import webrtcvad
import collections
import google.generativeai as genai
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List
import typing_extensions as typing

@dataclass
class ConversationTurn:
    speaker: str
    original_text: str
    english_text: str = ""
    response: str = ""

class AudioProcessingResult(typing.TypedDict):
    transcription: str
    detected_language: str
    english_translation: str
    is_question: bool
    english_response: str
    translated_response: str

class ConversationHistory:
    def __init__(self, max_turns=10):
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns
    
    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
    
    def get_formatted_history(self):
        formatted = []
        for turn in self.turns:
            formatted.append(f"User: {turn.english_text}")
            if turn.response:
                formatted.append(f"Assistant: {turn.response}")
        return "\n".join(formatted)

class LiveAudioTranscriber:
    def __init__(self, sample_rate=16000, vad_mode=2):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_mode)
        self.buffer = collections.deque(maxlen=30)
        self.recording = []
        self.is_speaking = False
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 30
        
        # Initialize conversation history
        self.history = ConversationHistory()
        
        # Domain-specific context
        self.domain_context = """
        You are an AI assistant that helps users by providing information strictly within these guidelines:
        1. Only provide information from authorized sources
        2. When unsure, acknowledge limitations
        3. For technical questions, provide detailed explanations with examples
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

Intervening as an active bystander can take different forms, often referred to as the "4 D’s of Bystander Intervention":

Direct Action:

Confront the behavior directly but safely.

Example: “That comment seems inappropriate. Let’s discuss why it might be harmful.”

Distraction:

Defuse the situation without direct confrontation.

Example: Changing the topic or creating a diversion to interrupt the behavior.

Delegation:

Seek help from others who may be better equipped to address the situation, such as supervisors or authorities.

Example: Reporting harassment to HR or law enforcement.

Delay:

If immediate action isn’t possible, follow up later.

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
        You can understand and respond in multiple Indian languages including Hindi, Tamil, Telugu, Kannada, and Malayalam.
        """
        
        # Configure Gemini
        genai.configure(api_key='API_KEY')
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def process_audio(self, audio_data):
        """Process audio with a single comprehensive prompt"""
        conversation_history = self.history.get_formatted_history()
        
        unified_prompt = f"""
        {self.domain_context}
        
        Previous conversation:
        {conversation_history}

        Instructions:
        1. First, transcribe the audio input exactly as spoken
        2. Detect the language of the transcribed text
        3. If the language is not English, translate it to English
        4. Analyze if the input is a question or statement
        5. Generate an appropriate response based on the conversation history and domain context
        6. If the original input was not in English, translate the response back to the original language
        
        Format your response as a JSON object with these fields:
        {{
            "transcription": "original transcribed text",
            "detected_language": "language name",
            "english_translation": "English translation if needed, otherwise same as transcription",
            "is_question": boolean,
            "english_response": "your response in English",
            "translated_response": "response in original language if needed, otherwise same as english_response"
        }}
        """
        
        try:
            # response = self.model.generate_content([unified_prompt, audio_data])
            response = self.model.generate_content(
            unified_prompt, generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=AudioProcessingResult  # Enforces the TypedDict structure
                ),
            )   
            result = json.loads(response.text)
            breakpoint()
            # Add to conversation history
            turn = ConversationTurn(
                speaker="User",
                original_text=result["transcription"],
                english_text=result["english_translation"],
                response=result["translated_response"]
            )
            self.history.add_turn(turn)
            
            # Print results
            print(f"\nTranscribed text: {result['transcription']}")
            print(f"Detected language: {result['detected_language']}")
            if result['transcription'] != result['english_translation']:
                print(f"English translation: {result['english_translation']}")
            print(f"\nResponse: {result['translated_response']}")
            
            return result
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print('Error:', status)
        
        audio_data = np.mean(indata, axis=1) if len(indata.shape) > 1 else indata
        audio_data = (audio_data * 32767).astype(np.int16)
        
        frame = audio_data.tobytes()
        is_speech = self.vad.is_speech(frame, self.sample_rate)
        
        self.buffer.append(frame)
        
        if is_speech:
            if not self.is_speaking:
                print("Speech started")
                self.is_speaking = True
                self.recording.extend(list(self.buffer))
            self.recording.append(frame)
            self.silence_frames = 0
        else:
            if self.is_speaking:
                self.silence_frames += 1
                self.recording.append(frame)
                
                if self.silence_frames >= self.SILENCE_THRESHOLD:
                    print("Speech ended")
                    self.process_recording()
                    self.recording = []
                    self.is_speaking = False
                    self.silence_frames = 0

    def process_recording(self):
        """Process the recorded audio"""
        if not self.recording:
            return
        
        # Save to temporary WAV file
        temp_file = "temp_recording.wav"
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.recording))
        
        # Prepare audio data for Gemini
        audio_data = {
            "mime_type": "audio/wav",
            "data": open(temp_file, "rb").read()
        }
        
        # Process audio with unified prompt
        self.process_audio(audio_data)
        
        # Clean up
        Path(temp_file).unlink()

    def start_recording(self):
        """Start the audio stream"""
        try:
            with sd.InputStream(callback=self.audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=int(self.sample_rate * 30/1000),
                              dtype=np.float32):
                print("Recording... Press Ctrl+C to stop")
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nRecording stopped")

if __name__ == "__main__":
    transcriber = LiveAudioTranscriber()
    transcriber.start_recording()