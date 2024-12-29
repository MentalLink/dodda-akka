import os
import logging
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TwilioService:
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        
        if not self.account_sid or not self.auth_token:
            raise ValueError("Twilio credentials not set in environment variables")
            
        self.client = Client(self.account_sid, self.auth_token)
        self.websocket_url = os.getenv("WEBSOCKET_URL", "wss://your-domain/ws/conversation")

    def handle_incoming_call(self, call_sid: str) -> Dict[str, Any]:
        """
        Handle incoming Twilio call and return TwiML response
        """
        try:
            response = VoiceResponse()
            
            # Connect the call to our WebSocket endpoint
            response.connect().stream(url=f"{self.websocket_url}/{call_sid}")
            
            return {"twiml": str(response)}
            
        except Exception as e:
            logger.error(f"Error handling incoming call: {str(e)}")
            response = VoiceResponse()
            response.say("We're sorry, but we're experiencing technical difficulties. Please try again later.")
            return {"twiml": str(response)}

    def end_call(self, call_sid: str):
        """
        End an active call
        """
        try:
            self.client.calls(call_sid).update(status="completed")
        except Exception as e:
            logger.error(f"Error ending call {call_sid}: {str(e)}")

    def get_call_status(self, call_sid: str) -> str:
        """
        Get the status of a call
        """
        try:
            call = self.client.calls(call_sid).fetch()
            return call.status
        except Exception as e:
            logger.error(f"Error getting call status for {call_sid}: {str(e)}")
            return "unknown" 