#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenRouter API Adapter for miniManus

This module implements the adapter for the OpenRouter API, which provides
access to various commercial language models.
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
    which offers access to various commercial language models.
    """
    
    def __init__(self):
        """Initialize the OpenRouter adapter."""
        self.logger = logger
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        
        # API configuration
        self.base_url = "https://openrouter.ai/api/v1"
        self.timeout = self.config_manager.get_config(
            "api.providers.openrouter.timeout", 
            60
        )
        self.default_model = self.config_manager.get_config(
            "api.providers.openrouter.default_model", 
            "openai/gpt-4-turbo"
        )
        
        # Available models cache
        self.models_cache = None
        self.models_cache_timestamp = 0
        self.models_cache_ttl = 3600  # 1 hour
        
        self.logger.info("OpenRouterAdapter initialized")
    
    def check_availability(self) -> bool:
        """
        Check if the OpenRouter API is available.
        
        Returns:
            True if available, False otherwise
        """
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._check_availability_async())
        finally:
            loop.close()
    
    async def _check_availability_async(self) -> bool:
        """
        Async implementation of check_availability.
        
        Returns:
            True if available, False otherwise
        """
        # Get API key
        api_key = self.config_manager.get_config("api.openrouter.api_key", "")
        if not api_key:
            self.logger.warning("OpenRouter API key not configured")
            return False
        
        # Check if API is available
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://minimanus.app",
                        "X-Title": "miniManus"
                    },
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error checking OpenRouter availability: {response.status} - {error_text}")
                        return False
        except Exception as e:
            self.logger.error(f"Exception checking OpenRouter availability: {str(e)}")
            return False
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter.
        
        Returns:
            List of model information dictionaries
        """
        # Check cache first
        current_time = asyncio.get_event_loop().time()
        if (self.models_cache is not None and 
            current_time - self.models_cache_timestamp < self.models_cache_ttl):
            return self.models_cache
        
        # Get API key
        api_key = self.config_manager.get_config("api.openrouter.api_key", "")
        if not api_key:
            self.logger.warning("OpenRouter API key not configured")
            return []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://minimanus.app",
                        "X-Title": "miniManus"
                    },
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("data", [])
                        
                        # Convert to standard format
                        formatted_models = []
                        for model in models:
                            formatted_models.append({
                                "id": model.get("id", ""),
                                "name": model.get("name", ""),
                                "context_length": model.get("context_length", 0),
                                "pricing": {
                                    "prompt": model.get("pricing", {}).get("prompt", 0),
                                    "completion": model.get("pricing", {}).get("completion", 0)
                                }
                            })
                        
                        self.models_cache = formatted_models
                        self.models_cache_timestamp = current_time
                        return formatted_models
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
    
    def generate_text(self, messages: List[Dict[str, str]], model: str = None, 
                     temperature: float = 0.7, max_tokens: int = 1024, 
                     api_key: str = None) -> str:
        """
        Generate text using the OpenRouter API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use (defaults to configured default model)
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            api_key: API key (optional, will use configured key if not provided)
            
        Returns:
            Generated text
        """
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._generate_text_async(messages, model, temperature, max_tokens, api_key)
            )
        finally:
            loop.close()
    
    async def _generate_text_async(self, messages: List[Dict[str, str]], 
                                 model: str = None, temperature: float = 0.7, 
                                 max_tokens: int = 1024, api_key: str = None) -> str:
        """
        Async implementation of generate_text.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use (defaults to configured default model)
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            api_key: API key (optional, will use configured key if not provided)
            
        Returns:
            Generated text
        """
        if not model:
            model = self.default_model
        
        if not api_key:
            api_key = self.config_manager.get_config("api.openrouter.api_key", "")
            if not api_key:
                self.logger.error("OpenRouter API key not configured")
                return "Error: OpenRouter API key not configured. Please set your API key in the settings."
        
        # Prepare request data
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://minimanus.app",
                        "X-Title": "miniManus",
                        "Content-Type": "application/json"
                    },
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error generating text: {response.status} - {error_text}")
                        return f"Error: {error_text}"
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Exception in generate_text: {error_msg}")
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "openrouter", "action": "generate_text"}
            )
            return f"Error: {error_msg}"
