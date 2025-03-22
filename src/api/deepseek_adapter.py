#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek API Adapter for miniManus

This module implements the adapter for the DeepSeek API, which provides
access to DeepSeek's language models.
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

logger = logging.getLogger("miniManus.DeepSeekAdapter")

class DeepSeekAdapter:
    """
    Adapter for the DeepSeek API.
    
    This adapter provides methods to interact with the DeepSeek API,
    which offers access to DeepSeek's language models.
    """
    
    def __init__(self):
        """Initialize the DeepSeek adapter."""
        self.logger = logger
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        
        # API configuration
        self.base_url = self.config_manager.get_config(
            "api.providers.deepseek.base_url", 
            "https://api.deepseek.com/v1"
        )
        self.timeout = self.config_manager.get_config(
            "api.providers.deepseek.timeout", 
            30
        )
        self.default_model = self.config_manager.get_config(
            "api.providers.deepseek.default_model", 
            "deepseek-chat"
        )
        
        # Available models cache
        self.models_cache = None
        self.models_cache_timestamp = 0
        self.models_cache_ttl = 3600  # 1 hour
        
        self.logger.info("DeepSeekAdapter initialized")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.
        
        Returns:
            Dictionary of headers
        """
        api_key = self.config_manager.get_api_key("deepseek")
        if not api_key:
            self.logger.warning("No API key found for DeepSeek")
            return {}
        
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    async def check_availability(self) -> bool:
        """
        Check if the DeepSeek API is available.
        
        Returns:
            True if available, False otherwise
        """
        api_key = self.config_manager.get_api_key("deepseek")
        if not api_key:
            self.logger.warning("No API key found for DeepSeek")
            return False
        
        try:
            # Try to fetch models as an availability check
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        self.logger.warning(f"DeepSeek API returned status {response.status}")
                        return False
        except Exception as e:
            self.logger.warning(f"Error checking DeepSeek availability: {str(e)}")
            return False
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from DeepSeek.
        
        Returns:
            List of model information dictionaries
        """
        # Check cache first
        current_time = asyncio.get_event_loop().time()
        if (self.models_cache is not None and 
            current_time - self.models_cache_timestamp < self.models_cache_ttl):
            return self.models_cache
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.models_cache = data.get("data", [])
                        self.models_cache_timestamp = current_time
                        return self.models_cache
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error fetching models: {response.status} - {error_text}")
                        return []
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "deepseek", "action": "get_models"}
            )
            return []
    
    async def send_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a chat completion request to DeepSeek.
        
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
            "stream": request_data.get("stream", False),
        }
        
        # Add optional parameters if provided
        for param in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
            if param in request_data:
                payload[param] = request_data[param]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error in chat request: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status} - {error_text}"}
        except Exception as e:
            error_msg = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "deepseek", "action": "chat_completion"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_completion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a text completion request to DeepSeek.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # For DeepSeek, we'll convert this to a chat request
        if "prompt" not in request_data:
            return {"error": "Missing required field: prompt"}
        
        # Convert to chat format
        chat_request = {
            "messages": [{"role": "user", "content": request_data["prompt"]}],
            "model": request_data.get("model", self.default_model),
            "stream": request_data.get("stream", False),
        }
        
        # Add optional parameters if provided
        for param in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
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
        Send an embedding request to DeepSeek.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # Ensure we have the required fields
        if "input" not in request_data:
            return {"error": "Missing required field: input"}
        
        # Prepare request payload
        payload = {
            "input": request_data["input"],
            "model": request_data.get("model", "deepseek-embedding"),  # Default embedding model
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error in embedding request: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status} - {error_text}"}
        except Exception as e:
            error_msg = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "deepseek", "action": "embedding"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_image_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an image generation request to DeepSeek.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # Check if DeepSeek supports image generation
        if "prompt" not in request_data:
            return {"error": "Missing required field: prompt"}
        
        # Prepare request payload
        payload = {
            "prompt": request_data["prompt"],
            "model": request_data.get("model", "deepseek-image"),  # Default image model
            "n": request_data.get("n", 1),
            "size": request_data.get("size", "1024x1024"),
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/images/generations",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        return {"error": "Image generation not supported by this DeepSeek API endpoint"}
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error in image request: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status} - {error_text}"}
        except Exception as e:
            error_msg = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "deepseek", "action": "image_generation"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_audio_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an audio processing request to DeepSeek.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # DeepSeek may not support audio processing
        return {"error": "Audio processing not supported by DeepSeek"}

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    config_manager = ConfigurationManager.get_instance()
    error_handler = ErrorHandler.get_instance()
    
    # Set API key for testing
    config_manager.set_api_key("deepseek", "your_api_key_here")
    
    # Initialize adapter
    adapter = DeepSeekAdapter()
    
    # Example request
    async def test_request():
        # Check availability
        available = await adapter.check_availability()
        print(f"DeepSeek available: {available}")
        
        if available:
            # Get models
            models = await adapter.get_available_models()
            print(f"Available models: {len(models)}")
            
            # Send chat request
            response = await adapter.send_chat_request({
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "model": "deepseek-chat"
            })
            print(f"Chat response: {json.dumps(response, indent=2)}")
    
    # Run test request
    asyncio.run(test_request())
