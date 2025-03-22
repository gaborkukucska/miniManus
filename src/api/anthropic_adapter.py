#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anthropic API Adapter for miniManus

This module implements the adapter for the Anthropic API, which provides
access to Claude language models.
"""

import os
import sys
import json
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Union

# Import local modules
try:
    from ..api.api_manager import APIProvider, APIRequestType
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from api.api_manager import APIProvider, APIRequestType
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager

logger = logging.getLogger("miniManus.AnthropicAdapter")

class AnthropicAdapter:
    """
    Adapter for the Anthropic API.
    
    This adapter provides methods to interact with the Anthropic API,
    which offers access to Claude language models.
    """
    
    def __init__(self):
        """Initialize the Anthropic adapter."""
        self.logger = logger
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        
        # API configuration
        self.base_url = self.config_manager.get_config(
            "api.providers.anthropic.base_url", 
            "https://api.anthropic.com/v1"
        )
        self.timeout = self.config_manager.get_config(
            "api.providers.anthropic.timeout", 
            30
        )
        self.default_model = self.config_manager.get_config(
            "api.providers.anthropic.default_model", 
            "claude-instant-1"
        )
        
        # Available models (hardcoded since Anthropic doesn't have a models endpoint)
        self.available_models = [
            {"id": "claude-instant-1", "name": "Claude Instant 1.2", "context_length": 100000},
            {"id": "claude-2", "name": "Claude 2", "context_length": 100000},
            {"id": "claude-3-opus", "name": "Claude 3 Opus", "context_length": 200000},
            {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "context_length": 200000},
            {"id": "claude-3-haiku", "name": "Claude 3 Haiku", "context_length": 200000}
        ]
        
        self.logger.info("AnthropicAdapter initialized")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.
        
        Returns:
            Dictionary of headers
        """
        api_key = self.config_manager.get_api_key("anthropic")
        if not api_key:
            self.logger.warning("No API key found for Anthropic")
            return {}
        
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    
    async def check_availability(self) -> bool:
        """
        Check if the Anthropic API is available.
        
        Returns:
            True if available, False otherwise
        """
        api_key = self.config_manager.get_api_key("anthropic")
        if not api_key:
            self.logger.warning("No API key found for Anthropic")
            return False
        
        try:
            # Anthropic doesn't have a dedicated health check endpoint,
            # so we'll make a minimal request to check availability
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                payload = {
                    "model": self.default_model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1
                }
                
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status in (200, 401, 403):  # Even auth errors mean API is up
                        return True
                    else:
                        self.logger.warning(f"Anthropic API returned status {response.status}")
                        return False
        except Exception as e:
            self.logger.warning(f"Error checking Anthropic availability: {str(e)}")
            return False
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from Anthropic.
        
        Returns:
            List of model information dictionaries
        """
        # Anthropic doesn't have a models endpoint, so we return hardcoded models
        return self.available_models
    
    async def send_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a chat completion request to Anthropic.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # Ensure we have the required fields
        if "messages" not in request_data:
            return {"error": "Missing required field: messages"}
        
        # Prepare request payload
        payload = {
            "messages": request_data["messages"],
            "model": request_data.get("model", self.default_model),
            "max_tokens": request_data.get("max_tokens", 1024),
        }
        
        # Add optional parameters if provided
        if "temperature" in request_data:
            payload["temperature"] = request_data["temperature"]
        
        if "top_p" in request_data:
            payload["top_p"] = request_data["top_p"]
        
        # Stream parameter (Anthropic uses 'stream' parameter)
        if "stream" in request_data:
            payload["stream"] = request_data["stream"]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        anthropic_response = await response.json()
                        
                        # Convert Anthropic response format to OpenAI-like format for consistency
                        return {
                            "id": anthropic_response.get("id", ""),
                            "object": "chat.completion",
                            "created": anthropic_response.get("created_at", 0),
                            "model": anthropic_response.get("model", ""),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": anthropic_response.get("content", [{}])[0].get("text", "")
                                    },
                                    "finish_reason": anthropic_response.get("stop_reason", "stop")
                                }
                            ],
                            "usage": anthropic_response.get("usage", {})
                        }
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error in chat request: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status} - {error_text}"}
        except Exception as e:
            error_msg = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "anthropic", "action": "chat_completion"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_completion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a text completion request to Anthropic.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # For Anthropic, we'll convert this to a chat request
        if "prompt" not in request_data:
            return {"error": "Missing required field: prompt"}
        
        # Convert to chat format
        chat_request = {
            "messages": [{"role": "user", "content": request_data["prompt"]}],
            "model": request_data.get("model", self.default_model),
        }
        
        # Add optional parameters if provided
        for param in ["temperature", "max_tokens", "top_p"]:
            if param in request_data:
                chat_request[param] = request_data[param]
        
        # Send as chat request
        response = await self.send_chat_request(chat_request)
        
        # Convert response format if needed
        if "error" not in response and "choices" in response:
            # Extract text from chat response
            try:
                text = response["choices"][0]["message"]["content"]
                return {
                    "id": response.get("id", ""),
                    "object": "text_completion",
                    "created": response.get("created", 0),
                    "model": response.get("model", ""),
                    "choices": [{"text": text, "index": 0, "finish_reason": response["choices"][0].get("finish_reason", "stop")}],
                    "usage": response.get("usage", {})
                }
            except (KeyError, IndexError) as e:
                self.logger.error(f"Error converting chat response to completion format: {str(e)}")
                return {"error": "Error processing response"}
        
        return response
    
    async def send_embedding_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an embedding request to Anthropic.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # Anthropic doesn't currently offer a public embeddings API
        return {"error": "Embedding not supported by Anthropic"}
    
    async def send_image_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an image generation request to Anthropic.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # Anthropic doesn't currently offer an image generation API
        return {"error": "Image generation not supported by Anthropic"}
    
    async def send_audio_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an audio processing request to Anthropic.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # Anthropic doesn't currently offer an audio processing API
        return {"error": "Audio processing not supported by Anthropic"}

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    config_manager = ConfigurationManager.get_instance()
    error_handler = ErrorHandler.get_instance()
    
    # Set API key for testing
    config_manager.set_api_key("anthropic", "your_api_key_here")
    
    # Initialize adapter
    adapter = AnthropicAdapter()
    
    # Example request
    async def test_request():
        # Check availability
        available = await adapter.check_availability()
        print(f"Anthropic available: {available}")
        
        if available:
            # Get models
            models = await adapter.get_available_models()
            print(f"Available models: {len(models)}")
            
            # Send chat request
            response = await adapter.send_chat_request({
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "model": "claude-instant-1"
            })
            print(f"Chat response: {json.dumps(response, indent=2)}")
    
    # Run test request
    asyncio.run(test_request())
