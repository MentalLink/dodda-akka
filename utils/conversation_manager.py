import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.max_history_length = 10
        self.conversation_timeout = timedelta(minutes=30)

    def initialize_conversation(self, call_sid: str) -> Dict[str, Any]:
        """
        Initialize a new conversation state
        """
        conversation_state = {
            "call_sid": call_sid,
            "start_time": datetime.now(),
            "last_activity": datetime.now(),
            "history": [],
            "metadata": {}
        }
        
        self.conversations[call_sid] = conversation_state
        return conversation_state

    def update_conversation(self, call_sid: str, user_input: str, assistant_response: str):
        """
        Update conversation history and metadata
        """
        if call_sid not in self.conversations:
            logger.warning(f"Attempting to update non-existent conversation: {call_sid}")
            return
            
        conversation = self.conversations[call_sid]
        conversation["last_activity"] = datetime.now()
        
        # Add to history
        conversation["history"].append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if too long
        if len(conversation["history"]) > self.max_history_length:
            conversation["history"] = conversation["history"][-self.max_history_length:]

    def get_conversation(self, call_sid: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation state
        """
        conversation = self.conversations.get(call_sid)
        
        if conversation:
            # Check if conversation has timed out
            if datetime.now() - conversation["last_activity"] > self.conversation_timeout:
                self.end_conversation(call_sid)
                return None
                
            return conversation
            
        return None

    def end_conversation(self, call_sid: str):
        """
        End and cleanup conversation
        """
        if call_sid in self.conversations:
            # Log conversation metrics
            conversation = self.conversations[call_sid]
            duration = datetime.now() - conversation["start_time"]
            turns = len(conversation["history"])
            
            logger.info(f"Conversation {call_sid} ended. Duration: {duration}, Turns: {turns}")
            
            # Cleanup
            del self.conversations[call_sid]

    def cleanup_stale_conversations(self):
        """
        Cleanup conversations that have timed out
        """
        current_time = datetime.now()
        stale_conversations = [
            call_sid for call_sid, conv in self.conversations.items()
            if current_time - conv["last_activity"] > self.conversation_timeout
        ]
        
        for call_sid in stale_conversations:
            self.end_conversation(call_sid) 