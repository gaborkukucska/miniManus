#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama API Adapter for miniManus

This module implements the adapter for the Ollama API, providing
asynchronous access to locally hosted language models.
"""

import os
import sys
import json
import logging
import aiohttp
import asyncio
import socket
import time
from typing import Dict, List, Optional, Any, Union, Tuple

# Import local modules
try:
    from ..api.api_manager import APIProvider, APIRequestType # Use relative import
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
except ImportError as e:
    # Handle potential import errors during early startup or testing
    logging.getLogger("miniManus.OllamaAdapter").critical(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(f"ImportError in ollama_adapter.py: {e}. Ensure all components exist.")

logger = logging.getLogger("miniManus.OllamaAdapter")

class OllamaAdapter:
    """
    Adapter for the Ollama API.

    Provides asynchronous methods to interact with a local or remote Ollama instance
    for model listing, chat completions, legacy completions, and embeddings.
    Includes network discovery for local Ollama servers.
    """

    def __init__(self):
        """Initialize the Ollama adapter."""
        self.logger = logger
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()

        # API configuration
        self.provider_name = "ollama" # Consistent key for config/secrets
        # Base URL: Read from config, try discovery if initial connection fails
        self._base_url = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.base_url",
            "http://localhost:11434" # Default Ollama URL (without /api path)
        )
        self.timeout = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.timeout",
            120 # Longer default timeout for potentially slow local models
        )
        self.default_model = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.default_model",
            "llama3" # Common default
        )

        # Available models cache
        self.models_cache: Optional[List[Dict[str, Any]]] = None
        self.models_cache_timestamp: float = 0
        self.models_cache_ttl: int = 120  # 2 minutes (shorter cache for local models)

        # Network discovery settings
        self.discovery_enabled = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.discovery_enabled", True
        )
        self.discovery_ports = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.discovery_ports", [11434]
        )
        self.discovery_timeout = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.discovery_timeout", 1.0 # Slightly longer discovery timeout
        )

        self._discovered_url: Optional[str] = None # Store discovered URL if found
        self._availability_checked = False # Flag to avoid redundant discovery

        self.logger.info("OllamaAdapter initialized")

    @property
    def base_url(self) -> str:
         """Returns the currently active base URL (discovered or from config)."""
         return self._discovered_url or self._base_url

    def _get_api_endpoint(self, path: str) -> str:
         """Constructs the full API endpoint URL."""
         # Ensure no double slashes and path starts with /api/
         clean_base = self.base_url.rstrip('/')
         if not path.startswith('/api/'):
              path = '/api/' + path.lstrip('/')
         return f"{clean_base}{path}"


    async def discover_ollama_servers(self) -> List[str]:
        """
        Asynchronously discover Ollama servers on the local network.

        Returns:
            List of discovered Ollama server base URLs (e.g., http://<ip>:11434).
        """
        if not self.discovery_enabled:
            self.logger.info("Ollama discovery disabled by configuration.")
            return []

        discovered_servers = []
        checked_ips = set()

        # --- Check localhost first ---
        for port in self.discovery_ports:
            url = f"http://localhost:{port}"
            checked_ips.add("127.0.0.1") # Mark localhost as checked
            try:
                endpoint = f"{url}/api/tags" # Check /api/tags endpoint
                async with aiohttp.ClientSession() as session:
                    # Short timeout for discovery probes
                    probe_timeout = aiohttp.ClientTimeout(total=self.discovery_timeout)
                    async with session.get(endpoint, timeout=probe_timeout) as response:
                        if response.status == 200:
                            self.logger.info(f"Discovered local Ollama server at {url}")
                            discovered_servers.append(url)
                            # Prefer localhost if found, return immediately
                            return discovered_servers
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                 self.logger.debug(f"No Ollama server found at {url} (connection/timeout error).")
            except Exception as e:
                 self.logger.debug(f"Error checking {url}: {e}")
        # --- End localhost check ---


        # --- Scan local network (if localhost failed) ---
        local_ip = self._get_local_ip()
        if not local_ip:
            self.logger.warning("Could not determine local IP for network scan.")
            return discovered_servers # Return empty list if local IP fails

        ip_parts = local_ip.split('.')
        if len(ip_parts) != 4:
            self.logger.warning(f"Invalid local IP format obtained: {local_ip}")
            return discovered_servers

        network_prefix = '.'.join(ip_parts[:3])
        max_hosts = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.discovery_max_hosts", 20
        )

        # Prioritize common IPs and scan others up to max_hosts
        common_last_octets = [1, 2, 254] # Router/gateway often end in these
        scan_range = list(range(1, 255)) # Full range initially
        tasks = []

        # Create tasks for scanning
        scanned_count = 0
        checked_ips.add(local_ip) # Don't scan self

        # Check common IPs first
        for i in common_last_octets:
             ip = f"{network_prefix}.{i}"
             if ip not in checked_ips:
                 tasks.append(self._probe_ollama_ip(ip, self.discovery_ports))
                 checked_ips.add(ip)
                 scanned_count += 1

        # Check remaining IPs up to the limit
        for i in scan_range:
             if scanned_count >= max_hosts: break
             ip = f"{network_prefix}.{i}"
             if ip not in checked_ips:
                 tasks.append(self._probe_ollama_ip(ip, self.discovery_ports))
                 checked_ips.add(ip)
                 scanned_count += 1

        # Run probes concurrently
        self.logger.info(f"Scanning up to {scanned_count} local network hosts for Ollama...")
        results = await asyncio.gather(*tasks)

        # Collect successful probes
        for url in results:
            if url:
                discovered_servers.append(url)

        self.logger.info(f"Ollama discovery finished. Found {len(discovered_servers)} potential servers.")
        return discovered_servers

    async def _probe_ollama_ip(self, ip: str, ports: List[int]) -> Optional[str]:
        """Probes a single IP address on given ports for Ollama."""
        for port in ports:
            url = f"http://{ip}:{port}"
            endpoint = f"{url}/api/tags"
            try:
                async with aiohttp.ClientSession() as session:
                    probe_timeout = aiohttp.ClientTimeout(total=self.discovery_timeout)
                    async with session.get(endpoint, timeout=probe_timeout) as response:
                        if response.status == 200:
                            self.logger.info(f"Discovered Ollama server at {url}")
                            return url # Return the base URL
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                 pass # Expected errors for non-responsive hosts
            except Exception as e:
                 self.logger.debug(f"Error probing {url}: {e}")
        return None # Not found on this IP


    def _get_local_ip(self) -> Optional[str]:
        """Helper to get the local IP address."""
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            # Doesn't need to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
            return IP
        except Exception as e:
            self.logger.warning(f"Could not determine local IP: {e}")
            return None
        finally:
            if s: s.close()

    async def check_availability(self) -> bool:
        """
        Asynchronously check if the configured or discovered Ollama API is available.
        Will attempt discovery if the configured URL fails and discovery is enabled.

        Returns:
            True if available, False otherwise.
        """
        # Check current base URL first
        endpoint = self._get_api_endpoint("tags") # Use /api/tags as health check
        try:
            async with aiohttp.ClientSession() as session:
                # Use a shorter timeout for availability check
                check_timeout = aiohttp.ClientTimeout(total=max(1.0, self.timeout / 10))
                async with session.get(endpoint, timeout=check_timeout) as response:
                    if response.status == 200:
                         self.logger.info(f"Ollama server confirmed available at {self.base_url}")
                         self._availability_checked = True
                         return True
                    else:
                         self.logger.warning(f"Ollama check at {self.base_url} failed with status {response.status}.")
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
             self.logger.warning(f"Ollama server not reachable at configured URL: {self.base_url}")
        except Exception as e:
             self.logger.error(f"Unexpected error checking Ollama at {self.base_url}: {e}", exc_info=True)

        # If initial check failed and discovery hasn't run yet
        if self.discovery_enabled and not self._discovered_url and not self._availability_checked:
            self.logger.info("Configured Ollama URL failed, attempting discovery...")
            discovered = await self.discover_ollama_servers()
            self._availability_checked = True # Mark discovery as attempted
            if discovered:
                self._discovered_url = discovered[0] # Use the first discovered URL
                self.logger.info(f"Discovery successful. Using Ollama server at {self.base_url}")
                # Save the discovered URL to config for persistence
                self.config_manager.set_config(f"api.providers.{self.provider_name}.base_url", self.base_url)
                return True
            else:
                self.logger.warning("Ollama discovery failed to find any servers.")
                return False
        elif self._discovered_url:
             # If we previously discovered a URL, assume it's still the one to use, but it failed the check.
             return False
        else:
             # Discovery disabled or already attempted, and configured URL failed.
             return False


    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Asynchronously get list of available models from Ollama, using cache.

        Returns:
            List of model information dictionaries, or empty list on failure.
        """
        # Check cache first
        current_time = time.time()
        if (self.models_cache is not None and
            current_time - self.models_cache_timestamp < self.models_cache_ttl):
            self.logger.debug(f"Returning cached models for {self.provider_name}")
            return self.models_cache

        endpoint = self._get_api_endpoint("tags")
        self.logger.info(f"Fetching available models from {endpoint}...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        # Ollama format is slightly different, but usable directly
                        # Example: {"name":"llama3:latest","modified_at":"...","size":...}
                        self.models_cache = models
                        self.models_cache_timestamp = current_time
                        self.logger.info(f"Successfully fetched {len(models)} models from {self.provider_name}.")
                        # Adapt to common format if needed by ModelSelectionInterface
                        adapted_models = [{
                            "id": m.get("name"),
                            "name": m.get("name"), # Often includes tag
                            "provider": self.provider_name.upper(),
                             # Add other fields if available or needed
                            "metadata": m
                        } for m in models if m.get("name")]
                        return adapted_models
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error fetching models from {self.provider_name}: {response.status} - {error_text}")
                        return []
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout fetching models from {self.provider_name}.")
            self.error_handler.handle_error(asyncio.TimeoutError("Timeout fetching models"), ErrorCategory.API, ErrorSeverity.WARNING, {"provider": self.provider_name, "action": "get_available_models"})
            return []
        except aiohttp.ClientError as e:
            self.logger.error(f"Client error fetching models from {self.provider_name}: {e}")
            self.error_handler.handle_error(e, ErrorCategory.NETWORK, ErrorSeverity.WARNING, {"provider": self.provider_name, "action": "get_available_models"})
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching models from {self.provider_name}: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "get_available_models"})
            return []

    async def send_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send a chat completion request to Ollama (/api/chat).

        Args:
            request_data: Dictionary containing 'model', 'messages', and optionally 'stream', 'options'.

        Returns:
            Dictionary with the API response (adapted) or an error dictionary.
        """
        required_fields = ["model", "messages"]
        if not all(field in request_data for field in required_fields):
            return {"error": f"Missing required fields for Ollama chat request: {required_fields}"}

        payload = {
            "model": request_data["model"],
            "messages": request_data["messages"],
            "stream": request_data.get("stream", False),
            "options": request_data.get("options", {}) # Pass options dict directly
        }
        # Add common parameters to options if not already present
        if "temperature" not in payload["options"] and request_data.get("temperature") is not None:
             payload["options"]["temperature"] = request_data["temperature"]
        if "top_p" not in payload["options"] and request_data.get("top_p") is not None:
             payload["options"]["top_p"] = request_data["top_p"]
        if "num_predict" not in payload["options"] and request_data.get("max_tokens") is not None:
             payload["options"]["num_predict"] = request_data["max_tokens"] # Map max_tokens

        endpoint = self._get_api_endpoint("chat")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload, timeout=self.timeout) as response:
                    # Ollama streams responses by default if stream=True, handle that later if needed
                    # For now, assume stream=False or handle non-streaming response
                    if response.status == 200:
                        ollama_response = await response.json()
                        # Adapt response to common format
                        return self._adapt_ollama_chat_response(ollama_response)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Ollama chat request failed: {response.status} - {error_text}")
                        return {"error": f"API Error ({response.status}): {error_text}"}

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

    def _adapt_ollama_chat_response(self, ollama_response: Dict[str, Any]) -> Dict[str, Any]:
         """Adapts Ollama's /api/chat non-streaming response to resemble OpenAI's."""
         try:
             created_time = int(time.time()) # Approximate creation time
             model_name = ollama_response.get("model", "unknown")
             message_content = ollama_response.get("message", {}).get("content", "")
             finish_reason = "stop" if ollama_response.get("done", True) else "incomplete" # Simple mapping

             adapted = {
                 "id": f"ollama-{model_name}-{created_time}", # Generate an ID
                 "object": "chat.completion",
                 "created": created_time,
                 "model": model_name,
                 "choices": [{
                     "index": 0,
                     "message": {
                         "role": "assistant",
                         "content": message_content
                     },
                     "finish_reason": finish_reason
                 }],
                 "usage": {
                     "prompt_tokens": ollama_response.get("prompt_eval_count"),
                     "completion_tokens": ollama_response.get("eval_count"),
                     "total_tokens": ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
                 },
                 "_raw_ollama_response": ollama_response
             }
             # Filter out usage fields if they are None
             adapted["usage"] = {k: v for k, v in adapted["usage"].items() if v is not None}
             return adapted
         except Exception as e:
             self.logger.error(f"Error adapting Ollama chat response: {e}. Raw: {ollama_response}", exc_info=True)
             return {"error": "Failed to process Ollama response", "_raw_response": ollama_response}


    async def send_completion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send a legacy completion request to Ollama (/api/generate).

        Args:
            request_data: Dictionary containing 'model', 'prompt', and optionally 'stream', 'options'.

        Returns:
            Dictionary with the API response (adapted) or an error dictionary.
        """
        required_fields = ["model", "prompt"]
        if not all(field in request_data for field in required_fields):
            return {"error": f"Missing required fields for Ollama completion request: {required_fields}"}

        payload = {
            "model": request_data["model"],
            "prompt": request_data["prompt"],
            "stream": request_data.get("stream", False),
            "options": request_data.get("options", {}),
            "system": request_data.get("system"), # Pass system prompt if provided
            "template": request_data.get("template"), # Pass template if provided
            "context": request_data.get("context") # Pass context if provided
        }
        payload = {k: v for k, v in payload.items() if v is not None} # Clean None values

        endpoint = self._get_api_endpoint("generate")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        ollama_response = await response.json()
                        # Adapt response
                        return self._adapt_ollama_completion_response(ollama_response)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Ollama completion request failed: {response.status} - {error_text}")
                        return {"error": f"API Error ({response.status}): {error_text}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} completion request.")
            return {"error": "Request timed out."}
        except aiohttp.ClientError as e:
             self.logger.error(f"Client error during {self.provider_name} completion request: {e}")
             return {"error": f"Network error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error during {self.provider_name} completion request: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}

    def _adapt_ollama_completion_response(self, ollama_response: Dict[str, Any]) -> Dict[str, Any]:
         """Adapts Ollama's /api/generate non-streaming response to resemble OpenAI's completion."""
         try:
             created_time = int(time.time()) # Approximate
             model_name = ollama_response.get("model", "unknown")
             text_content = ollama_response.get("response", "")
             finish_reason = "stop" if ollama_response.get("done", True) else "incomplete"

             adapted = {
                 "id": f"ollama-compl-{model_name}-{created_time}",
                 "object": "text_completion",
                 "created": created_time,
                 "model": model_name,
                 "choices": [{
                     "text": text_content,
                     "index": 0,
                     "logprobs": None,
                     "finish_reason": finish_reason
                 }],
                  "usage": {
                     "prompt_tokens": ollama_response.get("prompt_eval_count"),
                     "completion_tokens": ollama_response.get("eval_count"),
                     "total_tokens": ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
                 },
                 "_raw_ollama_response": ollama_response
             }
             adapted["usage"] = {k: v for k, v in adapted["usage"].items() if v is not None}
             return adapted
         except Exception as e:
             self.logger.error(f"Error adapting Ollama completion response: {e}. Raw: {ollama_response}", exc_info=True)
             return {"error": "Failed to process Ollama response", "_raw_response": ollama_response}

    async def send_embedding_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send an embedding request to Ollama (/api/embeddings).

        Args:
            request_data: Dictionary containing 'model', 'prompt' (the text to embed).

        Returns:
            Dictionary with the API response (adapted) or an error dictionary.
        """
        required_fields = ["model", "prompt"] # Ollama uses 'prompt' for input text
        if not all(field in request_data for field in required_fields):
            # Adapt: if 'input' is provided instead of 'prompt', use it
            if "input" in request_data and "prompt" not in request_data:
                request_data["prompt"] = request_data["input"]
            else:
                return {"error": f"Missing required field 'prompt' for Ollama embedding request."}

        payload = {
            "model": request_data["model"],
            "prompt": request_data["prompt"],
            "options": request_data.get("options", {}) # Pass options if provided
        }

        endpoint = self._get_api_endpoint("embeddings")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        ollama_response = await response.json()
                        # Adapt response
                        return self._adapt_ollama_embedding_response(ollama_response, request_data["model"])
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Ollama embedding request failed: {response.status} - {error_text}")
                        return {"error": f"API Error ({response.status}): {error_text}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} embedding request.")
            return {"error": "Request timed out."}
        except aiohttp.ClientError as e:
             self.logger.error(f"Client error during {self.provider_name} embedding request: {e}")
             return {"error": f"Network error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error during {self.provider_name} embedding request: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}

    def _adapt_ollama_embedding_response(self, ollama_response: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Adapts Ollama's /api/embeddings response to resemble OpenAI's."""
        try:
            embedding_vector = ollama_response.get("embedding")
            if embedding_vector is None:
                 raise ValueError("Embedding vector not found in Ollama response")

            adapted = {
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "embedding": embedding_vector,
                    "index": 0
                }],
                "model": model_name, # Ollama response doesn't include model name
                "usage": { # Ollama response doesn't include usage for embeddings
                    "prompt_tokens": None, # Or estimate based on input length?
                    "total_tokens": None,
                },
                 "_raw_ollama_response": ollama_response
            }
            return adapted
        except Exception as e:
             self.logger.error(f"Error adapting Ollama embedding response: {e}. Raw: {ollama_response}", exc_info=True)
             return {"error": "Failed to process Ollama embedding response", "_raw_response": ollama_response}


    async def send_image_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder: Ollama doesn't support image generation via this API."""
        self.logger.warning(f"{self.provider_name} adapter does not support image generation.")
        return {"error": "Image generation not supported by Ollama."}

    async def send_audio_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder: Ollama doesn't support audio processing via this API."""
        self.logger.warning(f"{self.provider_name} adapter does not support audio processing.")
        return {"error": "Audio processing not supported by Ollama."}

# Example usage (if run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    async def test_ollama_adapter():
        config_manager = ConfigurationManager.get_instance()
        # Ensure base_url is set correctly for testing, or rely on discovery
        # config_manager.set_config("api.providers.ollama.base_url", "http://your_ollama_ip:11434")

        error_handler = ErrorHandler.get_instance()
        event_bus = EventBus.get_instance()
        event_bus.startup()

        adapter = OllamaAdapter()

        print("--- Checking Availability (might trigger discovery) ---")
        is_available = await adapter.check_availability()
        print(f"Ollama Available: {is_available} at {adapter.base_url}")

        if is_available:
            print("\n--- Getting Available Models ---")
            models = await adapter.get_available_models()
            print(f"Found {len(models)} models.")
            if models:
                print("Example models:")
                for m in models[:min(len(models), 5)]:
                    print(f"  - {m.get('id')} (Name: {m.get('name')})") # Display adapted format

                test_chat_model = adapter.default_model # Use default from config
                print(f"\n--- Sending Chat Request (Model: {test_chat_model}) ---")
                chat_data = {
                    "model": test_chat_model,
                    "messages": [{"role": "user", "content": "Why is the sky blue?"}]
                }
                chat_response = await adapter.send_chat_request(chat_data)
                print("Chat Response (Adapted):")
                print(json.dumps(chat_response, indent=2))

                # Test embedding only if a model seems suitable (or use default)
                test_embed_model = test_chat_model # Often chat models work for embeddings in Ollama
                print(f"\n--- Sending Embedding Request (Model: {test_embed_model}) ---")
                embedding_data = {
                    "model": test_embed_model,
                    "prompt": "This is a test sentence for embedding."
                    # "input": "..." # Test if adapter handles 'input' key
                }
                embedding_response = await adapter.send_embedding_request(embedding_data)
                print("Embedding Response (Adapted, omitting vector):")
                if "data" in embedding_response and isinstance(embedding_response["data"], list) and embedding_response["data"]:
                     printable_response = json.loads(json.dumps(embedding_response))
                     printable_response["data"][0]["embedding"] = "[...vector omitted...]"
                     print(json.dumps(printable_response, indent=2))
                else:
                     print(json.dumps(embedding_response, indent=2))


            else:
                print("Skipping further tests as no models were retrieved.")
        else:
            print("Skipping further tests as Ollama is not available.")

        event_bus.shutdown()

    try:
        asyncio.run(test_ollama_adapter())
    except Exception as e:
        print(f"\nStandalone test failed: {e}", file=sys.stderr)
