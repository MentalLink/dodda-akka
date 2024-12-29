import requests

url = "https://api.sarvam.ai/speech-to-text-translate"

# Read the audio file
files = {
    'file': ('temp_recording.wav', open('/Users/pranav/Desktop/genderxgenai/DMK MP.wav', 'rb'), 'audio/wav'),
    'model': (None, 'saaras:v2'),
    'prompt': (None, '')
}

headers = {
    'accept': 'application/json',
    'api-subscription-key': '4d3340f6-8600-420b-a69b-bb87ced80f7f'  # Replace with actual API key
}

response = requests.post(url, files=files, headers=headers)

print(response.text)