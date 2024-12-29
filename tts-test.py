import requests

url = "https://api.ttsopenai.com/uapi/v1/text-to-speech"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "<your api key>"
}
data = {
    "model": "tts-1",
    "voice_id": "OA001",
    "speed": 1,
    "input": "Hello world!"
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
