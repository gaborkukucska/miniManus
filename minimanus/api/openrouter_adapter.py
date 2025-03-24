#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenRouter API Adapter for miniManus

This module implements the adapter for the OpenRouter API, which provides
access to various LLM models through a unified API interface.
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

logger = logging.getLogger("miniManus.OpenRouterAdapter")

class OpenRouterAdapter:
    """
    Adapter for the OpenRouter API.
    
    This adapter provides methods to interact with the OpenRouter API,
    which offers access to various LLM models through a unified interface.
    """
    
    def __init__(self):
        """Initialize the OpenRouter adapter."""
        self.logger = logger
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        
        # API configuration
        self.base_url = self.config_manager.get_config(
            "api.providers.openrouter.base_url", 
            "https://openrouter.ai/api/v1"
        )
        self.timeout = self.config_manager.get_config(
            "api.providers.openrouter.timeout", 
            30
        )
        self.default_model = self.config_manager.get_config(
            "api.providers.openrouter.default_model", 
            "gpt-3.5-turbo"
        )
        
        # Available models cache
        self.models_cache = None
        self.models_cache_timestamp = 0
        self.models_cache_ttl = 3600  # 1 hour
        
        self.logger.info("OpenRouterAdapter initialized")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.
        
        Returns:
            Dictionary of headers
        """
        api_key = self.config_manager.get_api_key("openrouter")
        if not api_key:
            self.logger.warning("No API key found for OpenRouter")
            return {}
        
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://minimanus.app",  # Required by OpenRouter
            "X-Title": "miniManus",  # Optional but recommended
        }
    
    def check_availability(self) -> bool:
        """
        Check if the OpenRouter API is available.
        
        Returns:
            True if available, False otherwise
        """
        api_key = self.config_manager.get_config("api.openrouter.api_key", "")
        if not api_key:
            self.logger.warning("No API key found for OpenRouter")
            return False
        
        # If we have an API key, assume it's available
        # In a real implementation, we would make a test request
        return True
    
    def generate_text(self, messages: List[Dict[str, str]], model: str = "", 
                     temperature: float = 0.7, max_tokens: int = 1024,
                     api_key: str = "") -> str:
        """
        Generate text using the OpenRouter API.
        
        Args:
            messages: List of message dictionaries with role and content
            model: Model to use
            temperature: Temperature parameter (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            api_key: API key to use (overrides config)
            
        Returns:
            Generated text
        """
        try:
            # Use provided API key or get from config
            if not api_key:
                api_key = self.config_manager.get_config("api.openrouter.api_key", "")
            
            if not api_key:
                return "Error: No API key provided for OpenRouter. Please configure your API key in settings."
            
            # Use provided model or get default
            if not model:
                model = self.config_manager.get_config("api.openrouter.default_model", "openai/gpt-3.5-turbo")
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://minimanus.app",  # Required by OpenRouter
                "X-Title": "miniManus",  # Optional but recommended
            }
            
            # Prepare request payload
            payload = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # In a real implementation, we would use asyncio and aiohttp
            # For simplicity in this synchronous context, we'll use the requests library
            import requests
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    return "Error: No response content received from OpenRouter."
            else:
                return f"Error: OpenRouter API returned status {response.status_code}: {response.text}"
        
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error generating text with OpenRouter: {error_msg}")
            return f"Error communicating with OpenRouter: {error_msg}"
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter.
        
        Returns:
            List of model information dictionaries
        """
        # Check cache first
        current_time = time.time()
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
                {"provider": "openrouter", "action": "get_models"}
            )
            return []
    
    async def send_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenRouter.
        
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
                {"provider": "openrouter", "action": "chat_completion"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_completion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a text completion request to OpenRouter.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # For OpenRouter, we'll convert this to a chat request
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
        Send an embedding request to OpenRouter.
        
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
            "model": request_data.get("model", "text-embedding-ada-002"),  # Default embedding model
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
                {"provider": "openrouter", "action": "embedding"}
            )
            return {"error": f"Request error: {error_msg}"}

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize adapter
    adapter = OpenRouterAdapter()
    
    # Example chat request
    request = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    # Run async function
    async def test():
        response = await adapter.send_chat_request(request)
        print(json.dumps(response, indent=2))
    
    asyncio.run(test())
