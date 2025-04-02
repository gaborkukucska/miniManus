#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LiteLLM API Adapter for miniManus

This module implements the adapter for the LiteLLM proxy API, which provides
a unified OpenAI-compatible interface to multiple LLM providers.
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
    logging.getLogger("miniManus.LiteLLMAdapter").critical(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(f"ImportError in litellm_adapter.py: {e}. Ensure all components exist.")


logger = logging.getLogger("miniManus.LiteLLMAdapter")

class LiteLLMAdapter:
    """
    Adapter for the LiteLLM Proxy API.

    Provides asynchronous methods to interact with a LiteLLM proxy instance,
    leveraging its OpenAI compatibility for chat, completions, and embeddings.
    Includes network discovery for local LiteLLM proxy servers.
    """

    def __init__(self):
        """Initialize the LiteLLM adapter."""
        self.logger = logger
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()

        # API configuration
        self.provider_name = "litellm" # Consistent key for config/secrets
        # Base URL: Read from config, try discovery if initial connection fails
        self._base_url = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.base_url",
            "http://localhost:8000" # Default LiteLLM proxy URL
        )
        self.timeout = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.timeout",
            60 # Default timeout, adjust as needed
        )
        # Default model: LiteLLM often maps requests, but we can set a preferred one
        self.default_model = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.default_model",
            "gpt-3.5-turbo" # A common default
        )
        self.default_embedding_model = self.config_manager.get_config(
             f"api.providers.{self.provider_name}.embedding_model",
             "text-embedding-ada-002" # Common embedding default
        )

        # Available models cache
        self.models_cache: Optional[List[Dict[str, Any]]] = None
        self.models_cache_timestamp: float = 0
        self.models_cache_ttl: int = 300  # 5 minutes cache for potentially dynamic models

        # Network discovery settings
        self.discovery_enabled = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.discovery_enabled", True
        )
        self.discovery_ports = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.discovery_ports", [8000, 4000] # Common LiteLLM ports
        )
        self.discovery_timeout = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.discovery_timeout", 1.0
        )

        self._discovered_url: Optional[str] = None
        self._availability_checked = False

        self.logger.info("LiteLLMAdapter initialized")

    @property
    def base_url(self) -> str:
         """Returns the currently active base URL (discovered or from config)."""
         return self._discovered_url or self._base_url

    def _get_headers(self) -> Optional[Dict[str, str]]:
        """
        Get headers for API requests. LiteLLM proxy might require a key.

        Returns:
            Dictionary of headers or None if required API key is missing.
        """
        # LiteLLM proxy *can* be configured to require a master key
        api_key = self.config_manager.get_api_key(self.provider_name) # Use 'litellm' key from secrets

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # No key needed if proxy isn't secured
        return headers


    async def discover_litellm_servers(self) -> List[str]:
        """
        Asynchronously discover LiteLLM proxy servers on the local network.
        Checks for OpenAI-compatible /models endpoint.

        Returns:
            List of discovered LiteLLM server base URLs (e.g., http://<ip>:8000).
        """
        if not self.discovery_enabled:
            self.logger.info("LiteLLM discovery disabled by configuration.")
            return []

        discovered_servers = []
        checked_ips = set()
        headers = self._get_headers() # Include headers in probe if key is configured

        # --- Check localhost first ---
        for port in self.discovery_ports:
            url = f"http://localhost:{port}"
            checked_ips.add("127.0.0.1")
            try:
                endpoint = f"{url}/models" # Standard OpenAI endpoint
                async with aiohttp.ClientSession() as session:
                    probe_timeout = aiohttp.ClientTimeout(total=self.discovery_timeout)
                    async with session.get(endpoint, headers=headers, timeout=probe_timeout) as response:
                        # LiteLLM might return 200 even without models if running
                        if response.status == 200:
                            self.logger.info(f"Discovered potential LiteLLM server at {url}")
                            discovered_servers.append(url)
                            # Prefer localhost if found
                            return discovered_servers
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                 self.logger.debug(f"No LiteLLM server found at {url} (connection/timeout error).")
            except Exception as e:
                 self.logger.debug(f"Error checking {url}: {e}")
        # --- End localhost check ---

        # --- Scan local network (if localhost failed) ---
        local_ip = self._get_local_ip()
        if not local_ip:
            self.logger.warning("Could not determine local IP for LiteLLM network scan.")
            return discovered_servers

        ip_parts = local_ip.split('.')
        if len(ip_parts) != 4:
            self.logger.warning(f"Invalid local IP format for LiteLLM scan: {local_ip}")
            return discovered_servers

        network_prefix = '.'.join(ip_parts[:3])
        max_hosts = self.config_manager.get_config(
            f"api.providers.{self.provider_name}.discovery_max_hosts", 20
        )

        common_last_octets = [1, 2, 254]
        scan_range = list(range(1, 255))
        tasks = []
        scanned_count = 0
        checked_ips.add(local_ip) # Don't scan self

        # Check common IPs first
        for i in common_last_octets:
             ip = f"{network_prefix}.{i}"
             if ip not in checked_ips:
                 tasks.append(self._probe_litellm_ip(ip, self.discovery_ports, headers))
                 checked_ips.add(ip)
                 scanned_count += 1

        # Check remaining IPs up to the limit
        for i in scan_range:
             if scanned_count >= max_hosts: break
             ip = f"{network_prefix}.{i}"
             if ip not in checked_ips:
                 tasks.append(self._probe_litellm_ip(ip, self.discovery_ports, headers))
                 checked_ips.add(ip)
                 scanned_count += 1

        # Run probes concurrently
        self.logger.info(f"Scanning up to {scanned_count} local network hosts for LiteLLM...")
        results = await asyncio.gather(*tasks)

        # Collect successful probes
        for url in results:
            if url:
                discovered_servers.append(url)

        self.logger.info(f"LiteLLM discovery finished. Found {len(discovered_servers)} potential servers.")
        return discovered_servers

    async def _probe_litellm_ip(self, ip: str, ports: List[int], headers: Optional[Dict[str,str]]) -> Optional[str]:
        """Probes a single IP address on given ports for LiteLLM proxy."""
        for port in ports:
            url = f"http://{ip}:{port}"
            endpoint = f"{url}/models"
            try:
                async with aiohttp.ClientSession() as session:
                    probe_timeout = aiohttp.ClientTimeout(total=self.discovery_timeout)
                    async with session.get(endpoint, headers=headers, timeout=probe_timeout) as response:
                        if response.status == 200:
                            self.logger.info(f"Discovered potential LiteLLM server at {url}")
                            return url # Return the base URL
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                 pass
            except Exception as e:
                 self.logger.debug(f"Error probing LiteLLM at {url}: {e}")
        return None


    def _get_local_ip(self) -> Optional[str]:
        """Helper to get the local IP address."""
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
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
        Asynchronously check if the configured or discovered LiteLLM API is available.
        Attempts discovery if the configured URL fails.

        Returns:
            True if available, False otherwise.
        """
        headers = self._get_headers() # Headers might be needed even for check

        # Check current base URL first
        endpoint = f"{self.base_url}/models"
        try:
            async with aiohttp.ClientSession() as session:
                check_timeout = aiohttp.ClientTimeout(total=max(1.0, self.timeout / 10))
                async with session.get(endpoint, headers=headers, timeout=check_timeout) as response:
                    if response.status == 200:
                         self.logger.info(f"LiteLLM server confirmed available at {self.base_url}")
                         self._availability_checked = True
                         return True
                    else:
                         # Handle 401 if master key is required but wrong/missing
                         if response.status == 401 and headers and "Authorization" in headers:
                              self.logger.warning(f"LiteLLM check at {self.base_url} failed: 401 Unauthorized (Check Master Key).")
                         else:
                              self.logger.warning(f"LiteLLM check at {self.base_url} failed with status {response.status}.")
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
             self.logger.warning(f"LiteLLM server not reachable at configured URL: {self.base_url}")
        except Exception as e:
             self.logger.error(f"Unexpected error checking LiteLLM at {self.base_url}: {e}", exc_info=True)

        # Attempt discovery if needed
        if self.discovery_enabled and not self._discovered_url and not self._availability_checked:
            self.logger.info("Configured LiteLLM URL failed, attempting discovery...")
            discovered = await self.discover_litellm_servers()
            self._availability_checked = True
            if discovered:
                self._discovered_url = discovered[0]
                self.logger.info(f"Discovery successful. Using LiteLLM server at {self.base_url}")
                self.config_manager.set_config(f"api.providers.{self.provider_name}.base_url", self.base_url)
                return True
            else:
                self.logger.warning("LiteLLM discovery failed to find any servers.")
                return False
        elif self._discovered_url:
             return False # Previously discovered URL failed the check
        else:
             return False # Discovery disabled or already failed


    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Asynchronously get list of available models from LiteLLM (/models), using cache.

        Returns:
            List of model information dictionaries (OpenAI format), or empty list on failure.
        """
        current_time = time.time()
        if (self.models_cache is not None and
            current_time - self.models_cache_timestamp < self.models_cache_ttl):
            self.logger.debug(f"Returning cached models for {self.provider_name}")
            return self.models_cache

        headers = self._get_headers()
        # Don't necessarily need headers for /models unless proxy requires key

        endpoint = f"{self.base_url}/models"
        self.logger.info(f"Fetching available models from {endpoint}...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        # LiteLLM should return OpenAI format: {"object": "list", "data": [...]}
                        models = data.get("data", [])
                        self.models_cache = models
                        self.models_cache_timestamp = current_time
                        self.logger.info(f"Successfully fetched {len(models)} models from {self.provider_name}.")
                        return models
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error fetching models from {self.provider_name}: {response.status} - {error_text}")
                        return []
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout fetching models from {self.provider_name}.")
            return []
        except aiohttp.ClientError as e:
            self.logger.error(f"Client error fetching models from {self.provider_name}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching models from {self.provider_name}: {e}", exc_info=True)
            return []

    async def send_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send a chat completion request to LiteLLM (/chat/completions).

        Args:
            request_data: OpenAI-compatible request dictionary.

        Returns:
            OpenAI-compatible response dictionary or an error dictionary.
        """
        headers = self._get_headers()
        # Headers (esp. Auth) might be required by the proxy

        required_fields = ["model", "messages"]
        if not all(field in request_data for field in required_fields):
            # Use default model if not provided
            if "messages" in request_data and "model" not in request_data:
                 request_data["model"] = self.default_model
            else:
                 return {"error": f"Missing required fields for chat request: {required_fields}"}

        payload = request_data # Assumes request_data is already in OpenAI format

        endpoint = f"{self.base_url}/chat/completions"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload, timeout=self.timeout) as response:
                    response_data = await response.json() # Assume JSON response
                    if response.status == 200:
                        return response_data # LiteLLM should return OpenAI format
                    else:
                        error_msg = response_data.get("error", {}).get("message", f"Unknown API error. Response: {response_data}")
                        self.logger.error(f"{self.provider_name} chat request failed: {response.status} - {error_msg}")
                        return {"error": f"API Error ({response.status}): {error_msg}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} chat request.")
            return {"error": "Request timed out."}
        except aiohttp.ClientError as e:
            self.logger.error(f"Client error during {self.provider_name} chat request: {e}")
            return {"error": f"Network error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error during {self.provider_name} chat request: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}

    async def send_completion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send a legacy completion request to LiteLLM (/completions).
        Use this only if the underlying model specifically requires it.
        """
        headers = self._get_headers()

        required_fields = ["model", "prompt"]
        if not all(field in request_data for field in required_fields):
            return {"error": f"Missing required fields for completion request: {required_fields}"}

        payload = request_data

        endpoint = f"{self.base_url}/completions"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload, timeout=self.timeout) as response:
                    response_data = await response.json()
                    if response.status == 200:
                        return response_data
                    else:
                        error_msg = response_data.get("error", {}).get("message", f"Unknown API error. Response: {response_data}")
                        self.logger.error(f"{self.provider_name} completion request failed: {response.status} - {error_msg}")
                        return {"error": f"API Error ({response.status}): {error_msg}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} completion request.")
            return {"error": "Request timed out."}
        except aiohttp.ClientError as e:
             self.logger.error(f"Client error during {self.provider_name} completion request: {e}")
             return {"error": f"Network error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error during {self.provider_name} completion request: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}

    async def send_embedding_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send an embedding request to LiteLLM (/embeddings).

        Args:
            request_data: OpenAI-compatible embedding request dictionary.

        Returns:
            OpenAI-compatible embedding response or an error dictionary.
        """
        headers = self._get_headers()

        required_fields = ["model", "input"]
        if not all(field in request_data for field in required_fields):
             # Use default embedding model if not provided
            if "input" in request_data and "model" not in request_data:
                 request_data["model"] = self.default_embedding_model
            else:
                 return {"error": f"Missing required fields for embedding request: {required_fields}"}

        payload = request_data

        endpoint = f"{self.base_url}/embeddings"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload, timeout=self.timeout) as response:
                    response_data = await response.json()
                    if response.status == 200:
                        return response_data
                    else:
                        error_msg = response_data.get("error", {}).get("message", f"Unknown API error. Response: {response_data}")
                        self.logger.error(f"{self.provider_name} embedding request failed: {response.status} - {error_msg}")
                        return {"error": f"API Error ({response.status}): {error_msg}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} embedding request.")
            return {"error": "Request timed out."}
        except aiohttp.ClientError as e:
             self.logger.error(f"Client error during {self.provider_name} embedding request: {e}")
             return {"error": f"Network error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error during {self.provider_name} embedding request: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}

    async def send_image_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send an image generation request via LiteLLM (/images/generations).
        Requires the proxy to be configured with an image model provider (e.g., OpenAI DALL-E).
        """
        headers = self._get_headers()

        required_fields = ["model", "prompt"] # 'model' specifies the underlying image model (e.g., dall-e-3)
        if not all(field in request_data for field in required_fields):
            return {"error": f"Missing required fields for image generation request: {required_fields}"}

        payload = request_data

        endpoint = f"{self.base_url}/images/generations"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload, timeout=self.timeout) as response:
                    response_data = await response.json()
                    if response.status == 200:
                        return response_data
                    else:
                        error_msg = response_data.get("error", {}).get("message", f"Unknown API error. Response: {response_data}")
                        self.logger.error(f"{self.provider_name} image request failed: {response.status} - {error_msg}")
                        return {"error": f"API Error ({response.status}): {error_msg}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} image request.")
            return {"error": "Request timed out."}
        except aiohttp.ClientError as e:
             self.logger.error(f"Client error during {self.provider_name} image request: {e}")
             return {"error": f"Network error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error during {self.provider_name} image request: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}

    async def send_audio_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder: LiteLLM proxy may not commonly support audio endpoints."""
        # Check LiteLLM documentation for audio transcription/speech endpoints if needed.
        self.logger.warning(f"{self.provider_name} adapter does not currently support audio processing.")
        return {"error": "Audio processing not supported by LiteLLM adapter."}


# Example usage (if run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    async def test_litellm_adapter():
        config_manager = ConfigurationManager.get_instance()
        # Optionally set a specific URL or API key for testing
        # config_manager.set_config("api.providers.litellm.base_url", "http://your_litellm_ip:4000")
        # config_manager.set_api_key("litellm", "your-master-key-if-needed")

        error_handler = ErrorHandler.get_instance()
        event_bus = EventBus.get_instance()
        event_bus.startup()

        adapter = LiteLLMAdapter()

        print("--- Checking Availability (might trigger discovery) ---")
        is_available = await adapter.check_availability()
        print(f"LiteLLM Available: {is_available} at {adapter.base_url}")

        if is_available:
            print("\n--- Getting Available Models ---")
            models = await adapter.get_available_models()
            print(f"Found {len(models)} models.")
            if models:
                print("Example models:")
                for m in models[:min(len(models), 5)]:
                    # Models should be in OpenAI format {'id': ..., 'object': 'model', ...}
                    print(f"  - {m.get('id')}")

                test_model = adapter.default_model
                print(f"\n--- Sending Chat Request (Model: {test_model}) ---")
                chat_data = {
                    "model": test_model,
                    "messages": [{"role": "user", "content": "What is LiteLLM?"}]
                }
                chat_response = await adapter.send_chat_request(chat_data)
                print("Chat Response:")
                print(json.dumps(chat_response, indent=2))

                # Test embedding
                embed_model = adapter.default_embedding_model
                print(f"\n--- Sending Embedding Request (Model: {embed_model}) ---")
                embedding_data = {
                    "model": embed_model,
                    "input": "Testing LiteLLM embeddings."
                }
                embedding_response = await adapter.send_embedding_request(embedding_data)
                print("Embedding Response (omitting vector):")
                if "data" in embedding_response and isinstance(embedding_response["data"], list) and embedding_response["data"]:
                     printable_response = json.loads(json.dumps(embedding_response))
                     printable_response["data"][0]["embedding"] = "[...vector omitted...]"
                     print(json.dumps(printable_response, indent=2))
                else:
                     print(json.dumps(embedding_response, indent=2))

            else:
                print("Skipping further tests as no models were retrieved.")
        else:
            print("Skipping further tests as LiteLLM is not available.")

        event_bus.shutdown()

    try:
        asyncio.run(test_litellm_adapter())
    except Exception as e:
        print(f"\nStandalone test failed: {e}", file=sys.stderr)
