from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import asyncio
import logging
import os
from dotenv import load_dotenv

from services.gemini_service import GeminiService
# from services.tts import CoquiTTSService
from services.twilio_service import TwilioService
from utils.conversation_manager import ConversationManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="V2V Conversational System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
gemini_service = GeminiService()
# tts_service = CoquiTTSService()
twilio_service = TwilioService()
conversation_manager = ConversationManager()

class TwilioWebhookRequest(BaseModel):
    CallSid: str
    From: str
    To: str
    RecordingUrl: Optional[str]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/webhook/twilio")
async def twilio_webhook(request: TwilioWebhookRequest):
    try:
        # Handle incoming Twilio webhook
        response = twilio_service.handle_incoming_call(request.CallSid)
        return response
    except Exception as e:
        logger.error(f"Error handling Twilio webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/conversation/{call_sid}")
async def websocket_endpoint(websocket: WebSocket, call_sid: str):
    await websocket.accept()
    
    try:
        # Initialize conversation state
        conversation_state = conversation_manager.initialize_conversation(call_sid)
        
        while True:
            # Receive audio chunks
            audio_chunk = await websocket.receive_bytes()
            
            # Process audio through Gemini (ASR + Response Generation)
            transcript = await gemini_service.transcribe_audio(audio_chunk)
            
            if transcript:
                # Get response from Gemini
                response = await gemini_service.generate_response(
                    transcript, 
                    conversation_state
                )
                
                # Convert response to speech
                audio_response = await tts_service.synthesize_speech(response)
                
                # Send audio response back
                await websocket.send_bytes(audio_response)
                
                # Update conversation state
                conversation_manager.update_conversation(
                    call_sid, 
                    transcript, 
                    response
                )
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()
    finally:
        conversation_manager.end_conversation(call_sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 