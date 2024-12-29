import typing
import requests
import sounddevice as sd
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
import soundfile as sf
import requests

@dataclass
class ConversationTurn:
    speaker: str
    text: str
    original_language: str
    english_translation: Optional[str] = None
    is_question: bool = False

class TranscriptionPrompt(typing.TypedDict):
    english_transcription: str

class TranslateFromEnglishPrompt(typing.TypedDict):
    translated_text: str

class TranscriptionPromptMultilingual(typing.TypedDict):
    multilingual_transcription: str
    language: str
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
    
    def synthesize_text(self, text: str, language_code: str
    ) -> bytes:
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content
class LiveAudioTranscriber:
    def __init__(self, sample_rate=16000, vad_mode=2):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_mode)
        self.buffer = collections.deque(maxlen=30)
        self.recording = []
        self.is_speaking = False
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 30
        
        self.tts_service = TTSService()
        # Initialize conversation history
        self.history = ConversationHistory()
        with open("Dataset.txt", "r") as file:
            self.context_data = file.read()

        # Sample domain-specific context (can be loaded from files)
        self.domain_context = f"""
        You are a calm, empathetic, and professional assistant for an active bystander and first responder hotline, seamlessly integrated into a voice-to-voice system. Your role is to assist users by providing brief, clear, and actionable responses, focusing on the immediate needs of the conversation. You are knowledgeable in topics related to bystander intervention, crisis response, and supporting individuals in distress.

        Always maintain a supportive and non-judgmental tone, prioritizing user comfort and clarity. Your responses should be concise, limited to one or two sentences, and relevant to the user's immediate context, similar to natural phone call turns. Avoid overwhelming users with information—offer only what is necessary and actionable. If further assistance is needed, guide the user towards next steps or suggest connecting with a human representative when appropriate.
        Don't make it feel like an interrogation, don't repeat questions too often, move the conversation based on the user's questions, do not get impatient.
        {self.context_data}
        """
        
        # Configure Gemini
        genai.configure(api_key='AIzaSyBco7C-oBK-6mx6qYIznGrTw4z9Ky2NdYk')
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.model_pro = genai.GenerativeModel('gemini-1.5-pro')
        
        # Supported languages with their codes
        self.supported_languages = {
            "hindi": "hi",
            "tamil": "ta",
            "kannada": "kn",
        }

    def detect_language(self, text):
        """Detect input language using Gemini"""
        prompt = f"""
        Detect the language of the following text and return only the language name in lowercase:
        Text: {text}
        """
        response = self.model.generate_content(prompt)
        detected_lang = response.text.strip().lower()
        return detected_lang if detected_lang in self.supported_languages else "english"

    def translate_to_english(self, text, source_language):
        """Translate text to English using Gemini"""
        if source_language == "english":
            return text
            
        prompt = f""" You can potentially have the input language as Hindi, Tamil, Kannada or English. You need to recognise which language it is based on what script you see.
        Translate the following text from input text to English. Always text:
        Text: {text}
        Language: {source_language}
        """
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def translate_from_english(self, text, target_language):
        """Translate text from English to target language using Gemini"""
        if target_language == "english":
            return text
            
        prompt = f"""
        Translate the following text from English to {target_language} and return the output in JSON format:
        
        Text: {text}
        
        Output format:
        {{
            "translated_text": "<translated_text_here>"
        }}
        """
        response = self.model.generate_content(
                prompt, generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=TranslateFromEnglishPrompt
                )
            )
        return response.text.strip()

    def analyze_input(self, text):
        """Analyze input and generate FIR-ready summary of conversation"""
        prompt = f"""
        Analyze the following text and return a JSON object with these fields:
        - is_question: boolean
        - intent: string (question/statement/command) 
        - topic: string
        - evidence_details: object containing:
            - incident_description: string (key details of any incident described)
            - date_time: string (any mentioned dates/times)
            - location: string (any mentioned locations)
            - persons_involved: array of strings (names/descriptions of people mentioned)
            - witness_details: array of strings (any witness information)
            - physical_evidence: array of strings (any physical evidence mentioned)
            - additional_notes: string (other relevant details for FIR)
        
        Text: {text}
        """
        
        # Get basic analysis
        response = self.model.generate_content(prompt)
        analysis = json.loads(response.text)
        
        # Generate conversation summary for FIR
        summary_prompt = f"""
        Review the entire conversation history and provide a comprehensive summary suitable for an FIR filing.
        Focus on capturing all relevant details about any incidents, evidence, dates, locations, and persons involved.
        Format the response as a structured JSON object.
        
        Conversation history:
        {self.history.get_context_string()}
        """
        
        summary_response = self.model.generate_content(summary_prompt)
        summary = json.loads(summary_response.text)
        
        # Combine analysis and summary
        analysis["conversation_summary"] = summary
        
        return analysis

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
        
        Based on the above context and conversation history, provide a detailed response to the user query or statement. If it's a question, answer it with relevant information. If it's a statement, acknowledge and provide additional insights. You need to ensure that the response is accurate and informative and directly from the provided context. Do not be very verbose and stick to the point. Be very concise and provide as much help as possible and have the kindest and most helpful tone possible. You need to speak as minimal as possible, you need to keep your responses very information dense and very helpful. if the person says "okay thank you, i want the call to end", acknowledge that and say "okay, thank you for calling, take care and feel to call back. i will share the details for an FIR which you can easily go and approach the authorities" and end the call.
        
        Remember to:
        1. Stay within the provided domain context
        2. Reference relevant parts of the conversation history but keep it extremely concise.
        3. Try to keep the response as short as possible while being informative.
        4. Maintain a supportive and non-judgmental tone.
        5. When you feel the caller isn't sharing much in a response, ask them a question to get more information. Stick to the questions in the provided context and do not ask any other questions. You goal is to extract as much information as possible and provide as much good information as possible.
        6. Remember you are in a phone call, you need to minimise the dialogue and not speak too much, keep it very much a back and forth conversation. 
        """
        
        # Generate response in English
        response = self.model.generate_content(prompt, generation_config=genai.GenerationConfig(max_output_tokens=75))
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
                    if len(self.recording)>100:
                        print("Speech ended")
                        self.process_speech()
                        self.recording = []
                        self.is_speaking = False
                        self.silence_frames = 0
                    else:
                        print("Speech Ended")
                        self.recording = []
                        self.is_speaking = False
                        self.silence_frames = 0

    def process_speech(self):
        """Process recorded speech and generate response"""
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
        # conversation_history = self.history.get_context_string()

        # unified_prompt = f"""
        # {self.domain_context}
        
        # Previous Conversation:
        # {conversation_history}

        # """
        try:
            # First, get transcription
            # transcription_prompt_multilingual = f"Please provide a word-for-word transcription of the speech as it is spoken in English, Hindi, Tamil, or Kannada. Stick to the native script and make sure you get every syllable correct. Figure out the language based on the input and return the transcription in that native language script. Output the language of the audio and give the audio in BCP47 code for eg. en-US, hi-IN, etc. DO NOT hallucinate. I do not want any extra characters, just whatever you hear, transcribe that. Stick to all -IN codes. Ensure that the output is in JSON format. Strictly stick to the format which is {{\"multilingual_transcription\": \"<transcribed_text>\",\"language\": \"<language_identified>\" }}. "
            # transcription_prompt = f"Please provide a word-for-word transcription of the speech as it is spoken in English. Ensure that the output is in JSON format. Strictly stick to the format which is {{\"english_transcription\": \"<transcribed_text>\"}}."
            # response = self.model.generate_content(
            #     [transcription_prompt, audio_data], generation_config=genai.GenerationConfig(
            #         response_mime_type="application/json",
            #         response_schema=TranscriptionPrompt
            #     )
            # )
            # language_id = self.model.generate_content(
            #     [transcription_prompt, audio_data], generation_config=genai.GenerationConfig(
            #         response_mime_type="application/json",
            # )

            # response = self.model.generate_content(
            #     [transcription_prompt_multilingual, audio_data], generation_config=genai.GenerationConfig(
            #         response_mime_type="application/json",
            #         response_schema=TranscriptionPromptMultilingual,
            #         max_output_tokens=75
            #     )
            # )
            # print(f"\n Transcription output: {response.text}")
            # transcribed_text = json.loads(response.text)["multilingual_transcription"]
            # indentified_language = json.loads(response.text)["language"]

            url = "https://api.sarvam.ai/speech-to-text-translate"

            # Read the audio file
            files = {
                'file': ('temp_recording.wav', open('temp_recording.wav', 'rb'), 'audio/wav'),
                'model': (None, 'saaras:v2'),
                'prompt': (None, '')
            }
            headers = {
                'accept': 'application/json',
                'api-subscription-key': '4d3340f6-8600-420b-a69b-bb87ced80f7f'  # Replace with actual API key
            }
            response = requests.post(url, files=files, headers=headers)

            print(response.text)
            # breakpoint()
            response = json.loads(response.text)
            indentified_language = response["language_code"]
            translated_text = response["transcript"]
            print(indentified_language)
            # breakpoint()
            # print(f"\n Language identified: {indentified_language}")
            # transcribed_text_multilingual = json.loads(response.text)["multilingual_transcription"]
            # translated_text = self.translate_to_english(transcribed_text, indentified_language)
            # print(f"\n Translation gemini output: {transcribed_text}")
            print(f"\n Translation gemini output: {translated_text}")
            response = self.generate_response(translated_text, "english")
            print(f"\nResponse: {response}")
            # tts_input = self.translate_from_english(response, indentified_language)
            # tts_input = json.loads(tts_input)["translated_text"]
            # Translate text using Sarvam API
            if indentified_language != "en-IN":
                url = "https://api.sarvam.ai/translate"
                
                # Map language codes to Sarvam API format
                sarvam_lang_codes = {
                    "hi": "hi-IN",
                    "ta": "ta-IN", 
                    "kn": "kn-IN"
                }
                
                # Only translate if language is supported by Sarvam API
                # if indentified_language in sarvam_lang_codes.values():
                payload = {
                    "input": response,
                    "source_language_code": "en-IN",
                    "target_language_code": indentified_language,
                    "speaker_gender": "Male",
                    "mode": "formal",
                    "model": "mayura:v1",
                    "enable_preprocessing": False
                }
                # breakpoint()
                headers = {
                    "Content-Type": "application/json",
                    "api-subscription-key": "4d3340f6-8600-420b-a69b-bb87ced80f7f"
                }

                response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    tts_input = response.json()["translated_text"]
            else:
                tts_input = response
            # breakpoint()
            audio_tts_output = self.tts_service.synthesize_text(tts_input, indentified_language)
            with open("output_audio.mp3", "wb") as audio_file:
                audio_file.write(audio_tts_output)
            
            # Play the audio file out loud
            audio_data = sf.read("output_audio.mp3", dtype='float32')
            sd.play(audio_data[0], audio_data[1])
            sd.wait()  # Wait until the audio is finished playing
        except Exception as e:
            print(f"Error processing speech: {e}")
        
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
            analysis = self.analyze_input(self.history.get_context_string())
            print(analysis)

if __name__ == "__main__":
    transcriber = LiveAudioTranscriber()
    transcriber.start_recording()


# make responses crisper and nice. do not add I Understand over and over again. 
# dont make it feel like an interrogation, dont repeat questions, move the conversation based on tbhe user, do not get impatient.
# add the functionality to be able to file FIR.
# ability to switch to
