#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anthropic API Adapter for miniManus

This module implements the adapter for the Anthropic API, providing
asynchronous access to Claude language models.
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
    logging.getLogger("miniManus.AnthropicAdapter").critical(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(f"ImportError in anthropic_adapter.py: {e}. Ensure all components exist.")


logger = logging.getLogger("miniManus.AnthropicAdapter")

class AnthropicAdapter:
    """
    Adapter for the Anthropic API (Claude models).

    Provides asynchronous methods to interact with the Anthropic API,
    primarily for chat completions. Note: Anthropic uses different API
    headers and response structures compared to OpenAI-like APIs.
    """

    # Hardcoded models as Anthropic doesn't have a public /models endpoint
    # (Keep this updated based on Anthropic documentation)
    _AVAILABLE_MODELS = [
        {"id": "claude-3-5-sonnet-20240620", "name": "Claude 3.5 Sonnet", "context_length": 200000},
        {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "context_length": 200000},
        {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet", "context_length": 200000},
        {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "context_length": 200000},
        # Older models (might be deprecated)
        # {"id": "claude-2.1", "name": "Claude 2.1", "context_length": 200000},
        # {"id": "claude-2.0", "name": "Claude 2.0", "context_length": 100000},
        # {"id": "claude-instant-1.2", "name": "Claude Instant 1.2", "context_length": 100000},
    ]
    # Recommended default model
    _DEFAULT_MODEL_ID = "claude-3-5-sonnet-20240620"
    _API_VERSION = "2023-06-01" # Anthropic API version

    def __init__(self):
        """Initialize the Anthropic adapter."""
        self.logger = logger
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()

        # API configuration
        self.provider_name = "anthropic" # Consistent key for config/secrets
        self.base_url = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.base_url",
            "https://api.anthropic.com/v1" # Default value
        )
        self.timeout = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.timeout",
            60 # Default timeout
        )
        self.default_model = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.default_model",
            self._DEFAULT_MODEL_ID # Use class constant default
        )

        self.logger.info("AnthropicAdapter initialized")

    def _get_headers(self) -> Optional[Dict[str, str]]:
        """
        Get headers required for Anthropic API requests.

        Returns:
            Dictionary of headers or None if API key is missing.
        """
        api_key = self.config_manager.get_api_key(self.provider_name)
        if not api_key:
            self.logger.warning(f"API key for {self.provider_name} not configured.")
            return None

        return {
            "x-api-key": api_key,
            "anthropic-version": self._API_VERSION,
            "Content-Type": "application/json",
        }

    async def check_availability(self) -> bool:
        """
        Asynchronously check if the Anthropic API is available and authenticated.
        Makes a minimal request to the messages endpoint.

        Returns:
            True if available and key is valid, False otherwise.
        """
        headers = self._get_headers()
        if not headers:
            return False

        # Minimal payload for testing the endpoint
        test_payload = {
            "model": self.default_model, # Use configured default
            "messages": [{"role": "user", "content": "Ping"}],
            "max_tokens": 1 # Request minimal generation
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=test_payload,
                    timeout=self.timeout / 3 # Shorter timeout for check
                ) as response:
                    # 200 is success, 401/403 indicate auth issues but API is reachable
                    if response.status == 200:
                         self.logger.info(f"{self.provider_name} API is available.")
                         return True
                    elif response.status in [401, 403]:
                        self.logger.warning(f"{self.provider_name} API key is invalid or lacks permissions ({response.status}).")
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
        Get the list of available models (hardcoded for Anthropic).

        Returns:
            List of model information dictionaries.
        """
        self.logger.debug(f"Returning hardcoded models for {self.provider_name}")
        return self._AVAILABLE_MODELS

    async def send_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send a chat completion request to Anthropic's /messages endpoint.

        Args:
            request_data: Dictionary containing request parameters like 'model', 'messages', 'max_tokens', etc.
                          Uses self.default_model if 'model' is not provided.

        Returns:
            Dictionary with the API response (potentially adapted to a common format) or an error dictionary.
        """
        headers = self._get_headers()
        if not headers:
            return {"error": f"API key for {self.provider_name} is missing."}

        required_fields = ["messages", "max_tokens"] # 'model' is optional, max_tokens is required by Anthropic
        if not all(field in request_data for field in required_fields):
            return {"error": f"Missing required fields for Anthropic chat request: {required_fields}"}

        # Ensure system prompt is handled correctly (Anthropic has a dedicated 'system' parameter)
        system_prompt = request_data.get("system")
        messages = request_data.get("messages", [])
        if messages and messages[0].get("role") == "system":
             # If system prompt is passed in messages, extract it
             if not system_prompt: system_prompt = messages[0].get("content")
             messages = messages[1:] # Remove system message from list

        payload = {
            "model": request_data.get("model", self.default_model),
            "messages": messages,
            "max_tokens": request_data["max_tokens"],
            # Add optional parameters
            "temperature": request_data.get("temperature"),
            "top_p": request_data.get("top_p"),
            "top_k": request_data.get("top_k"), # Anthropic specific
            "stop_sequences": request_data.get("stop_sequences"),
            "stream": request_data.get("stream", False),
            # Add system prompt if present
            "system": system_prompt
        }
        # Remove None values before sending
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    # Check content type before decoding json
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        response_data = await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"{self.provider_name} chat request failed: {response.status} - Non-JSON response: {error_text}")
                        return {"error": f"API Error ({response.status}): Received non-JSON response."}

                    if response.status == 200:
                         # Adapt Anthropic response to a more common format (like OpenAI's)
                         return self._adapt_anthropic_response(response_data)
                    else:
                        error_msg = response_data.get("error", {}).get("message", f"Unknown API error. Response: {response_data}")
                        error_type = response_data.get("error", {}).get("type", "api_error")
                        self.logger.error(f"{self.provider_name} chat request failed: {response.status} - {error_type}: {error_msg}")
                        return {"error": f"API Error ({response.status} {error_type}): {error_msg}"}

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

    def _adapt_anthropic_response(self, anthropic_response: Dict[str, Any]) -> Dict[str, Any]:
        """Converts Anthropic /messages response format to resemble OpenAI's chat completion format."""
        try:
             content_block = anthropic_response.get("content", [{}])[0]
             # Handle different content types if needed (e.g., tool_use)
             response_text = content_block.get("text", "") if content_block.get("type") == "text" else ""

             # Map stop reason
             finish_reason_map = {
                 "end_turn": "stop",
                 "max_tokens": "length",
                 "stop_sequence": "stop",
                 # Add mappings for other Anthropic stop reasons if necessary
             }
             finish_reason = finish_reason_map.get(anthropic_response.get("stop_reason"), "stop")

             adapted_response = {
                 "id": anthropic_response.get("id"),
                 "object": "chat.completion", # Mimic OpenAI format
                 "created": int(time.time()), # Anthropic doesn't provide creation time in response
                 "model": anthropic_response.get("model"),
                 "choices": [
                     {
                         "index": 0,
                         "message": {
                             "role": "assistant",
                             "content": response_text,
                             # Add tool calls here if adapting tool use responses
                         },
                         "finish_reason": finish_reason,
                     }
                 ],
                 "usage": { # Map usage stats
                     "prompt_tokens": anthropic_response.get("usage", {}).get("input_tokens"),
                     "completion_tokens": anthropic_response.get("usage", {}).get("output_tokens"),
                     "total_tokens": (
                         anthropic_response.get("usage", {}).get("input_tokens", 0) +
                         anthropic_response.get("usage", {}).get("output_tokens", 0)
                     ),
                 },
                 "_raw_anthropic_response": anthropic_response # Optional: include original for debugging
             }
             return adapted_response
        except (IndexError, KeyError, TypeError) as e:
             self.logger.error(f"Error adapting Anthropic response: {e}. Raw response: {anthropic_response}", exc_info=True)
             return {"error": "Failed to process API response.", "_raw_response": anthropic_response}


    async def send_completion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a legacy text completion request. NOT SUPPORTED by modern Anthropic API.
        This method will return an error.
        """
        self.logger.error(f"Legacy text completion endpoint is not supported by the {self.provider_name} adapter.")
        return {"error": "Anthropic adapter only supports the chat completions (/messages) API."}

    async def send_embedding_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for embedding requests."""
        self.logger.warning(f"{self.provider_name} adapter does not currently support embedding requests.")
        return {"error": "Embedding requests not supported by this adapter."}

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

    async def test_anthropic_adapter():
        # Minimal setup for testing
        config_manager = ConfigurationManager.get_instance()
        # Manually set API key for testing - REPLACE with your actual key
        # config_manager.secrets = {"api_keys": {"anthropic": "sk-ant-..."}}
        if not config_manager.get_api_key("anthropic"):
             print("WARNING: Anthropic API key not found in config/secrets. Tests may fail.")
             # config_manager.secrets = {"api_keys": {"anthropic": "dummy-key"}}

        error_handler = ErrorHandler.get_instance()
        event_bus = EventBus.get_instance()
        event_bus.startup()

        adapter = AnthropicAdapter()

        print("--- Checking Availability ---")
        is_available = await adapter.check_availability()
        print(f"Anthropic Available: {is_available}")

        if is_available:
            print("\n--- Getting Available Models (Hardcoded) ---")
            models = await adapter.get_available_models()
            print(f"Found {len(models)} models.")
            if models:
                print("Example models:")
                for m in models[:3]:
                    print(f"  - {m.get('id')}")

                print("\n--- Sending Chat Request ---")
                test_model = adapter.default_model
                chat_data = {
                    "model": test_model,
                    "messages": [{"role": "user", "content": "Explain the concept of large language models in simple terms."}],
                    "max_tokens": 150 # Required by Anthropic
                }
                chat_response = await adapter.send_chat_request(chat_data)
                print("Chat Response (Adapted Format):")
                print(json.dumps(chat_response, indent=2))

            else:
                print("No models configured for Anthropic.")
        else:
            print("Skipping further tests as Anthropic is not available.")

        event_bus.shutdown()

    try:
        asyncio.run(test_anthropic_adapter())
    except Exception as e:
        print(f"\nStandalone test failed: {e}")
