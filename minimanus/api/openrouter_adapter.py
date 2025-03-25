#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenRouter API Adapter for miniManus

This module implements the adapter for the OpenRouter API, which provides
access to various AI models through a unified interface.
"""

import os
import sys
import json
import logging
import aiohttp
import asyncio
import requests
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
    which offers access to various AI models through a unified interface.
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
            "api.openrouter.default_model", 
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
        api_key = self.config_manager.get_config("api.openrouter.api_key", "")
        if not api_key:
            self.logger.warning("OpenRouter API key not configured")
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://minimanus.app",
                "X-Title": "miniManus"
            }
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"OpenRouter API not available: {str(e)}")
            return False
    
    async def _check_availability_async(self) -> bool:
        """
        Async implementation of check_availability.
        
        Returns:
            True if available, False otherwise
        """
        api_key = self.config_manager.get_config("api.openrouter.api_key", "")
        if not api_key:
            self.logger.warning("OpenRouter API key not configured")
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://minimanus.app",
                "X-Title": "miniManus"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=self.timeout
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.warning(f"OpenRouter API not available: {str(e)}")
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
        
        api_key = self.config_manager.get_config("api.openrouter.api_key", "")
        if not api_key:
            self.logger.warning("OpenRouter API key not configured")
            return []
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://minimanus.app",
                "X-Title": "miniManus"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
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
        try:
            # Use the provided API key or fall back to configured key
            key = api_key or self.config_manager.get_config("api.openrouter.api_key", "")
            if not key:
                return "I'm sorry, but the OpenRouter API key is not configured. Please check your API settings."
            
            # Use the provided model or fall back to default
            model_name = model or self.default_model
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://minimanus.app",
                "X-Title": "miniManus"
            }
            
            # Make the API request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    return message.get("content", "")
                else:
                    return "I'm sorry, but the API response did not contain any choices."
            else:
                error_message = f"OpenRouter API error: {response.status_code} - {response.text}"
                self.logger.error(error_message)
                return f"I'm sorry, but the OpenRouter API is not available. Please check your API settings and ensure you've entered a valid API key."
                
        except Exception as e:
            error_message = f"Error generating text with OpenRouter: {str(e)}"
            self.logger.error(error_message)
            return f"I'm sorry, but there was an error communicating with the language model: {str(e)}"
    
    async def _generate_text_async(self, messages: List[Dict[str, str]], model: str = None,
                                 temperature: float = 0.7, max_tokens: int = 1024,
                                 api_key: str = None) -> str:
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
        try:
            # Use the provided API key or fall back to configured key
            key = api_key or self.config_manager.get_config("api.openrouter.api_key", "")
            if not key:
                return "I'm sorry, but the OpenRouter API key is not configured. Please check your API settings."
            
            # Use the provided model or fall back to default
            model_name = model or self.default_model
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://minimanus.app",
                "X-Title": "miniManus"
            }
            
            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        choices = result.get("choices", [])
                        if choices:
                            message = choices[0].get("message", {})
                            return message.get("content", "")
                        else:
                            return "I'm sorry, but the API response did not contain any choices."
                    else:
                        error_text = await response.text()
                        error_message = f"OpenRouter API error: {response.status} - {error_text}"
                        self.logger.error(error_message)
                        return f"I'm sorry, but the OpenRouter API is not available. Please check your API settings and ensure you've entered a valid API key."
                        
        except Exception as e:
            error_message = f"Error generating text with OpenRouter: {str(e)}"
            self.logger.error(error_message)
            return f"I'm sorry, but there was an error communicating with the language model: {str(e)}"
