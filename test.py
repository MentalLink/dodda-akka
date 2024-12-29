import os
import google.generativeai as genai

genai.configure(api_key='AIzaSyBco7C-oBK-6mx6qYIznGrTw4z9Ky2NdYk')

model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Read audio file
audio_path = "/Users/pranav/Desktop/genderxgenai/DMK MP.wav"  # Replace with your audio file path
audio_data = {
    "mime_type": "audio/wav",
    "data": open(audio_path, "rb").read()
}

# Create prompt for transcription
prompt = "Generate a transcript and translation in english of the speech."

# Generate transcript from audio
response = model.generate_content([prompt, audio_data])

print(response.text)