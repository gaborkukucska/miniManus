# START OF FILE miniManus-main/minimanus/api/ollama_adapter.py
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
import traceback # Added for direct test traceback printing
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path # Added for direct test

# *** Add sys.path modification for direct execution ***
if __name__ == "__main__":
    # Add the parent directory (miniManus-main) to sys.path
    # Assumes the script is run from the miniManus-main directory
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    print(f"Temporarily added to sys.path: {parent_dir}")
# *** End sys.path modification ***


# Import local modules
try:
    from minimanus.api.api_manager import APIProvider, APIRequestType # Use absolute import now
    from minimanus.core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from minimanus.core.config_manager import ConfigurationManager
    from minimanus.core.event_bus import EventBus # Needed for DummyEventBus below
except ImportError as e:
    # Handle potential import errors during early startup or testing
    logging.getLogger("miniManus.OllamaAdapter").critical(f"Failed to import required modules: {e}", exc_info=True)
    # Allow direct test execution even if imports fail initially
    if __name__ != "__main__":
        sys.exit(f"ImportError in ollama_adapter.py: {e}. Ensure all components exist.")
    else:
        # Define dummy classes if running directly and imports fail
        print("ImportError occurred, defining dummy classes for direct test.")
        class DummyEnum: pass
        class APIProvider(DummyEnum): OLLAMA=1
        class APIRequestType(DummyEnum): CHAT=1; COMPLETION=2; EMBEDDING=3 # Add necessary types
        class ErrorCategory(DummyEnum): API = 1; NETWORK = 2; STORAGE = 3; SYSTEM = 4
        class ErrorSeverity(DummyEnum): WARNING = 1; ERROR = 2; CRITICAL=3

        class DummyErrorHandler:
             _instance = None
             def handle_error(self, *args, **kwargs): print(f"DUMMY ERROR HANDLED: {args}")
             @classmethod
             def get_instance(cls): # Add get_instance
                  if cls._instance is None: cls._instance = cls()
                  return cls._instance

        class DummyConfigManager:
            _instance = None
            _config = { # Simulate config structure needed by adapter
                 "api.providers.ollama.base_url": "http://localhost:11434",
                 "api.providers.ollama.timeout": 60,
                 "api.providers.ollama.default_model": "llama3",
                 "api.providers.ollama.discovery_enabled": False,
                 "api.providers.ollama.discovery_ports": [11434],
                 "api.providers.ollama.discovery_max_hosts": 10,
                 "api.providers.ollama.discovery_timeout": 0.5
             }
            def get_config(self, key, default=None):
                val = self._config.get(key, default)
                # print(f"DummyConfig get: {key} -> {val}")
                return val
            def get_api_key(self, provider): return None
            def set_config(self, key, value): self._config[key] = value
            @classmethod
            def get_instance(cls): # Add get_instance
                 if cls._instance is None: cls._instance = cls()
                 return cls._instance

        class DummyEventBus:
             _instance = None
             def publish_event(self, *args, **kwargs): pass
             def startup(self): pass
             def shutdown(self): pass
             @classmethod
             def get_instance(cls): # Add get_instance
                  if cls._instance is None: cls._instance = cls()
                  return cls._instance

        # Assign dummy classes to the expected names
        ErrorHandler = DummyErrorHandler
        ConfigurationManager = DummyConfigManager
        EventBus = DummyEventBus # Assign EventBus dummy as well


logger = logging.getLogger("miniManus.OllamaAdapter")

class OllamaAdapter:
    """
    Adapter for the Ollama API.

    Provides asynchronous methods to interact with a local or remote Ollama instance
    for model listing, chat completions, legacy completions, and embeddings.
    Reads configuration dynamically for flexibility.
    Includes network discovery for local Ollama servers.
    """

    def __init__(self):
        """Initialize the Ollama adapter."""
        self.logger = logger
        # Use get_instance for potentially dummy classes
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()

        self.provider_name = "ollama" # Consistent key for config/secrets

        # Store config keys instead of values
        self._config_key_base_url = f"api.providers.{self.provider_name}.base_url"
        self._config_key_timeout = f"api.providers.{self.provider_name}.timeout"
        self._config_key_default_model = f"api.providers.{self.provider_name}.default_model"
        self._config_key_discovery_enabled = f"api.providers.{self.provider_name}.discovery_enabled"
        self._config_key_discovery_ports = f"api.providers.{self.provider_name}.discovery_ports"
        self._config_key_discovery_timeout = f"api.providers.{self.provider_name}.discovery_timeout"
        self._config_key_discovery_max_hosts = f"api.providers.{self.provider_name}.discovery_max_hosts"

        # Model cache settings
        self.models_cache: Optional[List[Dict[str, Any]]] = None
        self.models_cache_timestamp: float = 0
        self.models_cache_ttl: int = 120  # 2 minutes cache

        self.logger.info("OllamaAdapter initialized (dynamic config reading)")

    def _get_current_base_url(self) -> str:
        """Gets the current base URL from ConfigManager."""
        return self.config_manager.get_config(self._config_key_base_url, "http://localhost:11434")

    def _get_current_timeout(self) -> float:
         """Gets the current timeout from ConfigManager."""
         return self.config_manager.get_config(self._config_key_timeout, 120)

    def _get_current_default_model(self) -> str:
         """Gets the current default model from ConfigManager."""
         return self.config_manager.get_config(self._config_key_default_model, "llama3")

    def _get_api_endpoint(self, path: str) -> str:
        """Constructs the full API endpoint URL using the current base URL."""
        current_base_url = self._get_current_base_url() # Get current value
        clean_base = current_base_url.rstrip('/')
        if not path.startswith('/api/'):
            path = '/api/' + path.lstrip('/')
        return f"{clean_base}{path}"

    async def discover_ollama_servers(self) -> List[str]:
        """
        Asynchronously discover Ollama servers on the local network.
        Returns a list of discovered base URLs. Does not automatically update config.
        """
        # Use config manager to get discovery settings dynamically
        discovery_enabled = self.config_manager.get_config(self._config_key_discovery_enabled, True)
        discovery_ports = self.config_manager.get_config(self._config_key_discovery_ports, [11434])
        discovery_timeout_val = self.config_manager.get_config(self._config_key_discovery_timeout, 1.0)
        discovery_max_hosts = self.config_manager.get_config(self._config_key_discovery_max_hosts, 20)

        if not discovery_enabled:
            self.logger.info("Ollama discovery disabled by configuration.")
            return []

        discovered_servers = []
        checked_ips = set()

        # --- Check localhost first ---
        for port in discovery_ports:
            url = f"http://localhost:{port}"
            checked_ips.add("127.0.0.1")
            try:
                endpoint = f"{url}/api/tags"
                async with aiohttp.ClientSession() as session:
                    probe_timeout = aiohttp.ClientTimeout(total=discovery_timeout_val)
                    async with session.get(endpoint, timeout=probe_timeout) as response:
                        if response.status == 200:
                            self.logger.info(f"Discovered local Ollama server at {url}")
                            discovered_servers.append(url)
                            # Prefer localhost if found
                            return discovered_servers
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                 self.logger.debug(f"No Ollama server found at {url} (connection/timeout error).")
            except Exception as e:
                 self.logger.debug(f"Error checking {url}: {e}")

        # --- Scan local network (if localhost failed) ---
        local_ip = self._get_local_ip()
        if not local_ip:
            self.logger.warning("Could not determine local IP for network scan.")
            return discovered_servers

        ip_parts = local_ip.split('.')
        if len(ip_parts) != 4:
            self.logger.warning(f"Invalid local IP format obtained: {local_ip}")
            return discovered_servers

        network_prefix = '.'.join(ip_parts[:3])

        common_last_octets = [1, 2, 254]
        scan_range = list(range(1, 255))
        tasks = []
        scanned_count = 0
        checked_ips.add(local_ip)

        # Create tasks for common IPs
        for i in common_last_octets:
             ip = f"{network_prefix}.{i}"
             if ip not in checked_ips:
                 tasks.append(self._probe_ollama_ip(ip, discovery_ports))
                 checked_ips.add(ip)
                 scanned_count += 1

        # Create tasks for remaining IPs up to the limit
        for i in scan_range:
             if scanned_count >= discovery_max_hosts: break
             ip = f"{network_prefix}.{i}"
             if ip not in checked_ips:
                 tasks.append(self._probe_ollama_ip(ip, discovery_ports))
                 checked_ips.add(ip)
                 scanned_count += 1

        # Run probes concurrently
        self.logger.info(f"Scanning up to {scanned_count} local network hosts for Ollama...")
        results = await asyncio.gather(*tasks)

        # Collect successful probes
        for url in results:
            if url:
                discovered_servers.append(url)

        if discovered_servers:
            self.logger.info(f"Ollama discovery found servers: {discovered_servers}")
            # Optionally publish an event
            # self.event_bus.publish_event("ollama.discovered", {"servers": discovered_servers})
        else:
             self.logger.info("Ollama discovery finished. Found no servers on the network.")

        return discovered_servers

    async def _probe_ollama_ip(self, ip: str, ports: List[int]) -> Optional[str]:
        """Probes a single IP address on given ports for Ollama."""
        discovery_timeout_val = self.config_manager.get_config(self._config_key_discovery_timeout, 1.0)
        for port in ports:
            url = f"http://{ip}:{port}"
            endpoint = f"{url}/api/tags"
            try:
                async with aiohttp.ClientSession() as session:
                    probe_timeout = aiohttp.ClientTimeout(total=discovery_timeout_val)
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
        Asynchronously check if the Ollama API (at the currently configured URL) is available.
        Does NOT automatically trigger discovery on failure here anymore. Discovery is separate.
        """
        current_base_url = self._get_current_base_url()
        current_timeout = self._get_current_timeout()
        endpoint = self._get_api_endpoint("tags") # Use /api/tags as health check
        self.logger.debug(f"Checking Ollama availability at {current_base_url}")
        try:
            async with aiohttp.ClientSession() as session:
                check_timeout = aiohttp.ClientTimeout(total=max(1.0, current_timeout / 10))
                async with session.get(endpoint, timeout=check_timeout) as response:
                    if response.status == 200:
                         self.logger.debug(f"Ollama server confirmed available at {current_base_url}")
                         return True
                    else:
                         self.logger.warning(f"Ollama check at {current_base_url} failed with status {response.status}.")
                         return False
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
             self.logger.warning(f"Ollama server not reachable at configured URL: {current_base_url}")
             return False
        except Exception as e:
             self.logger.error(f"Unexpected error checking Ollama at {current_base_url}: {e}", exc_info=True)
             return False

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Asynchronously get list of available models from Ollama, using cache and current config.

        Returns:
            List of model information dictionaries (adapted format), or empty list on failure.
        """
        # Check cache first
        current_time = time.time()
        if (self.models_cache is not None and
            current_time - self.models_cache_timestamp < self.models_cache_ttl):
            self.logger.debug(f"Returning cached models for {self.provider_name}")
            # Ensure cache contains adapted data
            if self.models_cache and isinstance(self.models_cache[0], dict) and "provider" in self.models_cache[0]:
                 return self.models_cache
            else: # If cache contains raw data, adapt it
                 valid_models = [self._adapt_ollama_model_data(m) for m in self.models_cache if m]
                 return [m for m in valid_models if m]

        endpoint = self._get_api_endpoint("tags") # Uses current URL
        current_timeout = self._get_current_timeout()
        self.logger.info(f"Fetching available models from {endpoint}...")
        try:
            async with aiohttp.ClientSession() as session:
                fetch_timeout = aiohttp.ClientTimeout(total=max(5.0, current_timeout / 4))
                async with session.get(endpoint, timeout=fetch_timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        raw_models = data.get("models", [])
                        # Adapt Ollama format to common format
                        adapted_models = [
                            self._adapt_ollama_model_data(m) for m in raw_models if m
                        ]
                        # Filter out None results from adaptation errors
                        valid_models = [m for m in adapted_models if m]

                        self.models_cache = valid_models # Cache the adapted format
                        self.models_cache_timestamp = current_time
                        self.logger.info(f"Successfully fetched and adapted {len(valid_models)} models from {self.provider_name}.")
                        return valid_models
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

    def _adapt_ollama_model_data(self, ollama_model_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
         """Adapts Ollama's /api/tags model format to a more common one used internally."""
         model_id = ollama_model_data.get("name")
         if not model_id:
             self.logger.warning(f"Skipping Ollama model due to missing 'name': {ollama_model_data}")
             return None
         return {
             "id": model_id,
             "name": model_id,
             "provider": self.provider_name.upper(),
             "metadata": ollama_model_data
         }

    async def send_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send a chat completion request to Ollama (/api/chat) using current config.

        Args:
            request_data: Dictionary containing 'model', 'messages', and optionally 'stream', 'options'.

        Returns:
            Dictionary with the API response (adapted) or an error dictionary.
        """
        required_fields = ["messages"]
        if not all(field in request_data for field in required_fields):
            return {"error": f"Missing required fields for Ollama chat request: {required_fields}"}

        payload = {
            "model": request_data.get("model", self._get_current_default_model()),
            "messages": request_data["messages"],
            "stream": request_data.get("stream", False),
            "options": request_data.get("options", {})
        }

        # Add common parameters to options if not already present
        if "temperature" not in payload["options"] and request_data.get("temperature") is not None:
             payload["options"]["temperature"] = request_data["temperature"]
        if "top_p" not in payload["options"] and request_data.get("top_p") is not None:
             payload["options"]["top_p"] = request_data["top_p"]
        if "num_predict" not in payload["options"] and request_data.get("max_tokens") is not None:
             payload["options"]["num_predict"] = request_data["max_tokens"]

        if not payload["model"]:
             self.logger.error("Ollama chat request cannot proceed: model name is empty.")
             return {"error": "Model name is required but was empty."}

        endpoint = self._get_api_endpoint("chat")
        current_timeout = self._get_current_timeout()
        try:
            async with aiohttp.ClientSession() as session:
                req_timeout = aiohttp.ClientTimeout(total=current_timeout)
                async with session.post(endpoint, json=payload, timeout=req_timeout) as response:
                    if response.status == 200:
                        ollama_response = await response.json()
                        return self._adapt_ollama_chat_response(ollama_response)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Ollama chat request failed: {response.status} - {error_text}. Payload: {payload}")
                        return {"error": f"API Error ({response.status}): {error_text}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} chat request (timeout={current_timeout}s).")
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
             created_time = int(time.time())
             model_name = ollama_response.get("model", "unknown")
             message_content = ollama_response.get("message", {}).get("content", "")
             finish_reason = "stop" if ollama_response.get("done", True) else "incomplete"

             adapted = {
                 "id": f"ollama-{model_name}-{created_time}",
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
             adapted["usage"] = {k: v for k, v in adapted["usage"].items() if v is not None}
             return adapted
         except Exception as e:
             self.logger.error(f"Error adapting Ollama chat response: {e}. Raw: {ollama_response}", exc_info=True)
             return {"error": "Failed to process Ollama response", "_raw_response": ollama_response}

    async def send_completion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously send a legacy completion request to Ollama (/api/generate) using current config.

        Args:
            request_data: Dictionary containing 'model', 'prompt', and optionally 'stream', 'options'.

        Returns:
            Dictionary with the API response (adapted) or an error dictionary.
        """
        required_fields = ["prompt"]
        if not all(field in request_data for field in required_fields):
            return {"error": f"Missing required fields for Ollama completion request: {required_fields}"}

        payload = {
            "model": request_data.get("model", self._get_current_default_model()),
            "prompt": request_data["prompt"],
            "stream": request_data.get("stream", False),
            "options": request_data.get("options", {}),
            "system": request_data.get("system"),
            "template": request_data.get("template"),
            "context": request_data.get("context")
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        if not payload["model"]:
             self.logger.error("Ollama completion request cannot proceed: model name is empty.")
             return {"error": "Model name is required but was empty."}

        endpoint = self._get_api_endpoint("generate")
        current_timeout = self._get_current_timeout()
        try:
            async with aiohttp.ClientSession() as session:
                req_timeout = aiohttp.ClientTimeout(total=current_timeout)
                async with session.post(endpoint, json=payload, timeout=req_timeout) as response:
                    if response.status == 200:
                        ollama_response = await response.json()
                        return self._adapt_ollama_completion_response(ollama_response)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Ollama completion request failed: {response.status} - {error_text}. Payload: {payload}")
                        return {"error": f"API Error ({response.status}): {error_text}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} completion request (timeout={current_timeout}s).")
            self.error_handler.handle_error(asyncio.TimeoutError("Timeout during completion request"), ErrorCategory.API, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "send_completion_request"})
            return {"error": "Request timed out."}
        except aiohttp.ClientError as e:
             self.logger.error(f"Client error during {self.provider_name} completion request: {e}")
             self.error_handler.handle_error(e, ErrorCategory.NETWORK, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "send_completion_request"})
             return {"error": f"Network error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error during {self.provider_name} completion request: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.ERROR, {"provider": self.provider_name, "action": "send_completion_request"})
            return {"error": f"An unexpected error occurred: {e}"}

    def _adapt_ollama_completion_response(self, ollama_response: Dict[str, Any]) -> Dict[str, Any]:
         """Adapts Ollama's /api/generate non-streaming response to resemble OpenAI's completion."""
         try:
             created_time = int(time.time())
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
        Asynchronously send an embedding request to Ollama (/api/embeddings) using current config.

        Args:
            request_data: Dictionary containing 'model', 'prompt' (the text to embed).

        Returns:
            Dictionary with the API response (adapted) or an error dictionary.
        """
        required_fields = ["prompt"]
        if "input" in request_data and "prompt" not in request_data:
             request_data["prompt"] = request_data["input"]
        elif "prompt" not in request_data:
             return {"error": f"Missing required field 'prompt' (or 'input') for Ollama embedding request."}

        payload = {
            "model": request_data.get("model", self._get_current_default_model()),
            "prompt": request_data["prompt"],
            "options": request_data.get("options", {})
        }

        if not payload["model"]:
             self.logger.error("Ollama embedding request cannot proceed: model name is empty.")
             return {"error": "Model name is required but was empty."}

        endpoint = self._get_api_endpoint("embeddings")
        current_timeout = self._get_current_timeout()
        try:
            async with aiohttp.ClientSession() as session:
                req_timeout = aiohttp.ClientTimeout(total=current_timeout)
                async with session.post(endpoint, json=payload, timeout=req_timeout) as response:
                    if response.status == 200:
                        ollama_response = await response.json()
                        return self._adapt_ollama_embedding_response(ollama_response, payload["model"])
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Ollama embedding request failed: {response.status} - {error_text}. Payload: {payload}")
                        return {"error": f"API Error ({response.status}): {error_text}"}

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout during {self.provider_name} embedding request (timeout={current_timeout}s).")
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
                "model": model_name,
                "usage": {
                    "prompt_tokens": None,
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
    print("--- Running Direct Ollama Adapter Test ---")

    # Dummies are defined at the top if imports failed

    # Ensure singletons are replaced ONLY if they weren't imported correctly
    if 'ConfigurationManager' not in globals() or not hasattr(ConfigurationManager, 'get_instance'):
         print("Using DummyConfigManager")
         ConfigurationManager = DummyConfigManager
    if 'ErrorHandler' not in globals() or not hasattr(ErrorHandler, 'get_instance'):
         print("Using DummyErrorHandler")
         ErrorHandler = DummyErrorHandler
    if 'EventBus' not in globals() or not hasattr(EventBus, 'get_instance'):
         print("Using DummyEventBus")
         EventBus = DummyEventBus

    # Now create the adapter instance
    adapter = OllamaAdapter()

    async def test_ollama_adapter_direct():
        # Get config dynamically
        print(f"Adapter Config Base URL: {adapter._get_current_base_url()}")
        print(f"Adapter Config Default Model: {adapter._get_current_default_model()}")
        print(f"Adapter Config Timeout: {adapter._get_current_timeout()}")

        print("\n--- Checking Availability ---")
        try:
            is_available = await adapter.check_availability()
            print(f"Ollama Available via Adapter: {is_available}")
        except Exception as e:
            print(f"Error during check_availability: {e}")
            is_available = False

        if is_available:
            print("\n--- Getting Available Models ---")
            models = []
            try:
                models = await adapter.get_available_models()
                print(f"Adapter found {len(models)} models.")
                if models:
                    print("Models (Adapted Format):")
                    for m in models[:min(len(models), 5)]:
                        print(f"  - ID: {m.get('id')}, Name: {m.get('name')}, Provider: {m.get('provider')}, Metadata: {m.get('metadata')}")
                else:
                    print("No models retrieved by adapter.")
            except Exception as e:
                 print(f"Error during get_available_models: {e}")

            print("\n--- Testing Chat Request ---")
            if models:
                chat_model_to_use = models[0]['id']
                print(f"Using model: {chat_model_to_use}")
                chat_data = {
                    "model": chat_model_to_use,
                    "messages": [{"role": "user", "content": "Why is the sky blue?"}]
                }
                try:
                    chat_response = await adapter.send_chat_request(chat_data)
                    print("Chat Response (Adapted):")
                    print(json.dumps(chat_response, indent=2))
                except Exception as e:
                    print(f"Error during send_chat_request: {e}")
            else:
                 # Try with default model if no models were listed
                 print(f"No models listed, trying default: {adapter._get_current_default_model()}")
                 chat_data = {
                    # "model": adapter._get_current_default_model(), # Let send_chat_request use default
                    "messages": [{"role": "user", "content": "Why is the sky blue?"}]
                 }
                 try:
                    chat_response = await adapter.send_chat_request(chat_data)
                    print("Chat Response (Adapted, using default model):")
                    print(json.dumps(chat_response, indent=2))
                 except Exception as e:
                    print(f"Error during send_chat_request (using default): {e}")


        else:
            print("Skipping further tests as adapter check failed or returned False.")


    # Run the async test function
    try:
        asyncio.run(test_ollama_adapter_direct())
    except Exception as e:
        print(f"\nDirect adapter test failed: {e}", file=sys.stderr)
        traceback.print_exc()

    print("\n--- Direct Ollama Adapter Test Finished ---")
# END OF FILE miniManus-main/minimanus/api/ollama_adapter.py
