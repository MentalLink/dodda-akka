import os
import logging
from typing import Dict, Any
import aiohttp
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class MistralLLMService:
    def __init__(self):
        self.api_key = os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError("HF_API_KEY environment variable is not set")
        
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        
        # System prompt for conversation
        self.system_prompt = """You are a helpful AI assistant engaged in a voice conversation. 
        Keep your responses clear, concise, and natural. Avoid technical jargon unless specifically asked."""
    
    def _format_prompt(self, user_input: str, conversation_state: Dict[str, Any]) -> str:
        """Format the prompt with conversation history and system prompt"""
        history = conversation_state.get("history", [])
        formatted_history = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" 
                                     for h in history[-3:]])  # Keep last 3 turns
        
        return f"""<s>[INST] {self.system_prompt}

{formatted_history}

User: {user_input} [/INST]</s>"""

    async def generate_response(self, user_input: str, conversation_state: Dict[str, Any]) -> str:
        """
        Generate response using Mistral model
        """
        try:
            prompt = self._format_prompt(user_input, conversation_state)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": prompt, "parameters": {
                        "max_new_tokens": 150,
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "return_full_text": False
                    }}
                ) as response:
                    result = await response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                        # Clean up the response
                        generated_text = generated_text.strip()
                        return generated_text
                    
                    raise ValueError(f"Unexpected response format: {result}")
                    
        except Exception as e:
            logger.error(f"Error in LLM generation: {str(e)}")
            return "I apologize, but I'm having trouble processing your request at the moment. Could you please try again?"

    def update_conversation_history(self, conversation_state: Dict[str, Any], 
                                 user_input: str, assistant_response: str):
        """
        Update the conversation history in the state
        """
        if "history" not in conversation_state:
            conversation_state["history"] = []
            
        conversation_state["history"].append({
            "user": user_input,
            "assistant": assistant_response
        }) 