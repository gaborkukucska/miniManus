#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenRouter API Adapter for miniManus

This module implements the adapter for the OpenRouter API, which provides
access to various AI models through a unified API interface.
"""

import os
import sys
import json
import logging
import aiohttp
import asyncio
import time
from typing import Dict, List, Optional, Any, Union

# Import local modules
try:
    from ..api.api_manager import APIProvider, APIRequestType # Use relative import
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
except ImportError as e:
    # Handle potential import errors during early startup or testing
    logging.getLogger("miniManus.OpenRouterAdapter").critical(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(f"ImportError in openrouter_adapter.py: {e}. Ensure all components exist.")

logger = logging.getLogger("miniManus.OpenRouterAdapter")

class OpenRouterAdapter:
    """
    Adapter for the OpenRouter API.

    Provides asynchronous methods to interact with the OpenRouter API,
    offering access to various AI models through a unified interface.
    """

    def __init__(self):
        """Initialize the OpenRouter adapter."""
        self.logger = logger
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()

        # API configuration - read from ConfigManager
        self.provider_name = "openrouter" # Consistent key for config/secrets
        self.base_url = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.base_url",
            "https://openrouter.ai/api/v1" # Default value
        )
        self.timeout = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.timeout",
            60 # Increased default timeout for potentially longer generations
        )
        # Note: Default model for the *provider* is in config,
        #       but the adapter might use a specific model per request type if needed.

        # Available models cache
        self.models_cache: Optional[List[Dict[str, Any]]] = None
        self.models_cache_timestamp: float = 0
        self.models_cache_ttl: int = 3600  # 1 hour

        self.logger.info("OpenRouterAdapter initialized")

    def _get_headers(self) -> Optional[Dict[str, str]]:
        """
        Get headers for API requests, including authentication.

        Returns:
            Dictionary of headers or None if API key is missing.
        """
        api_key = self.config_manager.get_api_key(self.provider_name)
        if not api_key:
            self.logger.warning(f"API key for {self.provider_name} not configured.")
            return None # Indicate missing key

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config_manager.get_config("api.openrouter.referer", "https://minimanus.app"), # Make referer configurable
            "X-Title": self.config_manager.get_config("api.openrouter.x_title", "miniManus"), # Make X-Title configurable
        }

    async def check_availability(self) -> bool:
        """
        Asynchronously check if the OpenRouter API is available and authenticated.

        Returns:
            True if available and key is valid, False otherwise.
        """
        headers = self._get_headers()
        if not headers:
            return False # Not available if key is missing

        try:
            # Use the /models endpoint as a simple check
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(f"{self.base_url}/models", timeout=self.timeout / 2) as response: # Shorter timeout for availability check
                    if response.status == 200:
                        self.logger.info(f"{self.provider_name} API is available.")
                        return True
                    elif response.status == 401:
                         self.logger.warning(f"{self.provider_name} API key is invalid (401 Unauthorized).")
                         return False # Key invalid, treat as unavailable
                    else:
                        self.logger.warning(f"{self.provider_name} API check failed with status {response.status}.")
                        return False
        except asyncio.TimeoutError:
            self.logger.warning(f"{self.provider_name} API availability check timed out.")
            return False
        except aiohttp.ClientConnectionError as e:
             self.logger.warning(f"Connection error during {self.provider_name} availability check: {e}")
             return False
        except Exception as e:
            self.logger.error(f"Unexpected error during {self.provider_name} availability check: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.WARNING, {"provider": self.provider_name, "action": "check_availability"})
            return False

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Asynchronously get the list of available models from OpenRouter, using cache.

        Returns:
            List of model information dictionaries, or an empty list on failure.
        """
        # Check cache first
        current_time = time.time()
        if (self.models_cache is not None and
            current_time - self.models_cache_timestamp < self.models_cache_ttl):
            self.logger.debug(f"Returning cached models for {self.provider_name}")
            return self.models_cache

        headers = self._get_headers()
        if not headers:
            return [] # Cannot fetch without API key

        self.logger.info(f"Fetching available models from {self.provider_name}...")
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(f"{self.base_url}/models", timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("data", [])
                        # Update cache
                        self.models_cache = models
                        self.models_cache_timestamp = current_time
                        self.logger.info(f"Successfully fetched {len(models)} models from {self.provider_name}.")
                        return models
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error fetching models from {self.provider_name}: {response.status} - {error_text}")
                        # Don't update cache on error
                        return [] # Return empty list on API error
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout fetching models from {self.provider_name}.")
            self.error_handler.handle_error(asyncio.TimeoutError("Timeout fetching models"), ErrorCategory.API, ErrorSeverity.WARNING, {"provider": self.provider_name, "action": "get_available_models"})
            return [] # Return empty list on timeout
        except aiohttp.ClientError as e:
            self.logger.error(f"Client error fetching models from {self.provider_name}: {e}")
            self.error_handler.handle_error(e, ErrorCategory.NETWORK, ErrorSeverity.WARNING, {"provider": self.provider_name, "action": "get_available_models"})
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching models from {self.provider_name}: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "get_available_models"})
            return [] # Return empty list on unexpected error

    async def send_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send a chat completion request to OpenRouter.

        Args:
            request_data: Dictionary containing request parameters like 'model', 'messages', etc.

        Returns:
            Dictionary with the API response or an error dictionary.
        """
        headers = self._get_headers()
        if not headers:
            return {"error": f"API key for {self.provider_name} is missing."}

        required_fields = ["model", "messages"]
        if not all(field in request_data for field in required_fields):
            return {"error": f"Missing required fields for chat request: {required_fields}"}

        # Add default params from config if not present in request_data (optional)
        # payload = {
        #     "temperature": self.config_manager.get_config(f"api.{self.provider_name}.temperature", 0.7),
        #     "max_tokens": self.config_manager.get_config(f"api.{self.provider_name}.max_tokens", 1024),
        #     **request_data # request_data overrides defaults
        # }
        payload = request_data # Assume request_data is complete for now

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    response_data = await response.json()
                    if response.status == 200:
                        return response_data # Return successful response directly
                    else:
                        error_msg = response_data.get("error", {}).get("message", "Unknown API error")
                        self.logger.error(f"{self.provider_name} chat request failed: {response.status} - {error_msg}")
                        # You might want to parse specific OpenRouter errors here
                        return {"error": f"API Error ({response.status}): {error_msg}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} chat request.")
            self.error_handler.handle_error(asyncio.TimeoutError("Timeout during chat request"), ErrorCategory.API, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "send_chat_request"})
            return {"error": "Request timed out."}
        except aiohttp.ClientError as e:
            self.logger.error(f"Client error during {self.provider_name} chat request: {e}")
            self.error_handler.handle_error(e, ErrorCategory.NETWORK, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "send_chat_request"})
            return {"error": f"Network error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error during {self.provider_name} chat request: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "send_chat_request"})
            return {"error": f"An unexpected error occurred: {e}"}

    async def send_completion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a legacy text completion request (Not recommended for most OpenRouter models).
        Adapts the prompt to the chat format.
        """
        self.logger.warning(f"Using legacy completion request for {self.provider_name}. Adapting to chat format.")
        if "prompt" not in request_data:
            return {"error": "Missing required field: prompt"}

        # Adapt to chat format
        chat_request_data = request_data.copy()
        chat_request_data["messages"] = [{"role": "user", "content": chat_request_data.pop("prompt")}]

        # Call the chat request method
        chat_response = await self.send_chat_request(chat_request_data)

        # Convert chat response back to completion format (if needed by caller)
        if "error" not in chat_response and "choices" in chat_response:
             try:
                 text_content = chat_response["choices"][0]["message"]["content"]
                 completion_response = {
                     "id": chat_response.get("id"),
                     "object": "text_completion", # Mimic completion object
                     "created": chat_response.get("created"),
                     "model": chat_response.get("model"),
                     "choices": [
                         {
                             "text": text_content,
                             "index": 0,
                             "logprobs": None, # Not available in chat format
                             "finish_reason": chat_response["choices"][0].get("finish_reason"),
                         }
                     ],
                     "usage": chat_response.get("usage"),
                 }
                 return completion_response
             except (KeyError, IndexError) as e:
                 self.logger.error(f"Error converting chat response to completion format: {e}")
                 return {"error": "Failed to process chat response into completion format."}
        else:
            # Return the error from the chat request
            return chat_response

    async def send_embedding_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send an embedding request to OpenRouter.
        """
        headers = self._get_headers()
        if not headers:
            return {"error": f"API key for {self.provider_name} is missing."}

        required_fields = ["model", "input"]
        if not all(field in request_data for field in required_fields):
            return {"error": f"Missing required fields for embedding request: {required_fields}"}

        payload = request_data

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    response_data = await response.json()
                    if response.status == 200:
                        return response_data
                    else:
                        error_msg = response_data.get("error", {}).get("message", "Unknown API error")
                        self.logger.error(f"{self.provider_name} embedding request failed: {response.status} - {error_msg}")
                        return {"error": f"API Error ({response.status}): {error_msg}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} embedding request.")
            self.error_handler.handle_error(asyncio.TimeoutError("Timeout during embedding request"), ErrorCategory.API, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "send_embedding_request"})
            return {"error": "Request timed out."}
        except aiohttp.ClientError as e:
            self.logger.error(f"Client error during {self.provider_name} embedding request: {e}")
            self.error_handler.handle_error(e, ErrorCategory.NETWORK, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "send_embedding_request"})
            return {"error": f"Network error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error during {self.provider_name} embedding request: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "send_embedding_request"})
            return {"error": f"An unexpected error occurred: {e}"}


    async def send_image_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for image generation requests."""
        self.logger.warning(f"{self.provider_name} adapter does not currently support image generation requests.")
        return {"error": "Image generation not supported by this adapter."}

    async def send_audio_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for audio processing requests."""
        self.logger.warning(f"{self.provider_name} adapter does not currently support audio processing requests.")
        return {"error": "Audio processing not supported by this adapter."}

# Example usage (if run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for testing

    async def test_openrouter_adapter():
        # Minimal setup for testing
        config_manager = ConfigurationManager.get_instance()
        # Manually set API key for testing - REPLACE with your actual key
        # config_manager.secrets = {"api_keys": {"openrouter": "sk-or-v1-YOUR-API-KEY"}}
        if not config_manager.get_api_key("openrouter"):
             print("WARNING: OpenRouter API key not found in config/secrets. Tests may fail.")
             # Provide a dummy key for basic tests if needed, but API calls will fail auth
             # config_manager.secrets = {"api_keys": {"openrouter": "dummy-key"}}


        error_handler = ErrorHandler.get_instance()
        event_bus = EventBus.get_instance() # Error handler needs event bus
        event_bus.startup()

        adapter = OpenRouterAdapter()

        print("--- Checking Availability ---")
        is_available = await adapter.check_availability()
        print(f"OpenRouter Available: {is_available}")

        if is_available:
            print("\n--- Getting Available Models ---")
            models = await adapter.get_available_models()
            print(f"Found {len(models)} models.")
            if models:
                print("First 3 models:")
                for m in models[:3]:
                    print(f"  - {m.get('id')} ({m.get('name')})")

                print("\n--- Sending Chat Request ---")
                # Use a known free model if possible, or default
                test_model = config_manager.get_config("api.providers.openrouter.default_model", "mistralai/mistral-7b-instruct")
                chat_data = {
                    "model": test_model,
                    "messages": [{"role": "user", "content": "Hello, OpenRouter!"}]
                }
                chat_response = await adapter.send_chat_request(chat_data)
                print("Chat Response:")
                print(json.dumps(chat_response, indent=2))

                # print("\n--- Sending Embedding Request ---")
                # embedding_model = "thenlper/gte-large" # Example embedding model
                # embedding_data = {
                #     "model": embedding_model,
                #     "input": "This is a test sentence."
                # }
                # embedding_response = await adapter.send_embedding_request(embedding_data)
                # print("Embedding Response (omitting vector):")
                # if "data" in embedding_response and embedding_response["data"]:
                #      embedding_response["data"][0]["embedding"] = "[...vector omitted...]"
                # print(json.dumps(embedding_response, indent=2))

            else:
                print("Skipping further tests as no models were retrieved.")
        else:
            print("Skipping further tests as OpenRouter is not available.")

        event_bus.shutdown()

    try:
        asyncio.run(test_openrouter_adapter())
    except Exception as e:
        print(f"\nStandalone test failed: {e}")
