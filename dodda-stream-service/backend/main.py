from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beta import LiveAudioTranscriber

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the React build directory
app.mount("/", StaticFiles(directory="../frontend/build", html=True))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    transcriber = LiveAudioTranscriber()
    
    try:
        while True:
            # Receive audio data from the frontend
            audio_data = await websocket.receive_bytes()
            
            # Process the audio data
            response = transcriber.process_audio_data(audio_data)
            
            # Send the response back to the frontend
            await websocket.send_json({
                "transcription": response["transcription"],
                "response": response["response"],
                "audio": response["audio_base64"]
            })
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 