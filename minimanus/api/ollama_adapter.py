#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama API Adapter for miniManus

This module implements the adapter for the Ollama API, which provides
access to locally hosted language models.
"""

import os
import sys
import json
import logging
import aiohttp
import asyncio
import socket
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

logger = logging.getLogger("miniManus.OllamaAdapter")

class OllamaAdapter:
    """
    Adapter for the Ollama API.
    
    This adapter provides methods to interact with the Ollama API,
    which offers access to locally hosted language models.
    """
    
    def __init__(self):
        """Initialize the Ollama adapter."""
        self.logger = logger
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        
        # API configuration
        self.base_url = self.config_manager.get_config(
            "api.providers.ollama.base_url", 
            "http://localhost:11434/api"
        )
        self.timeout = self.config_manager.get_config(
            "api.providers.ollama.timeout", 
            30
        )
        self.default_model = self.config_manager.get_config(
            "api.providers.ollama.default_model", 
            "llama2"
        )
        
        # Available models cache
        self.models_cache = None
        self.models_cache_timestamp = 0
        self.models_cache_ttl = 60  # 1 minute (shorter for local models that might change)
        
        # Network discovery settings
        self.discovery_enabled = self.config_manager.get_config(
            "api.providers.ollama.discovery_enabled",
            True
        )
        self.discovery_ports = self.config_manager.get_config(
            "api.providers.ollama.discovery_ports",
            [11434]
        )
        self.discovery_timeout = self.config_manager.get_config(
            "api.providers.ollama.discovery_timeout",
            0.5  # seconds
        )
        
        self.logger.info("OllamaAdapter initialized")
    
    async def discover_ollama_servers(self) -> List[str]:
        """
        Discover Ollama servers on the local network.
        
        Returns:
            List of discovered Ollama server URLs
        """
        if not self.discovery_enabled:
            return []
        
        discovered_servers = []
        
        # First check localhost
        for port in self.discovery_ports:
            try:
                url = f"http://localhost:{port}/api"
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{url}/tags",
                        timeout=self.discovery_timeout
                    ) as response:
                        if response.status == 200:
                            self.logger.info(f"Discovered local Ollama server at {url}")
                            discovered_servers.append(url)
            except Exception:
                pass
        
        # If we found a local server, no need to scan network
        if discovered_servers:
            return discovered_servers
        
        # Get local IP address
        local_ip = self._get_local_ip()
        if not local_ip:
            return []
        
        # Get network prefix
        ip_parts = local_ip.split('.')
        if len(ip_parts) != 4:
            return []
        
        network_prefix = '.'.join(ip_parts[:3])
        
        # Scan network (limited to last octet to avoid excessive scanning)
        max_hosts = self.config_manager.get_config(
            "api.providers.ollama.discovery_max_hosts",
            20  # Limit to 20 hosts by default
        )
        
        # Prioritize common IP addresses
        common_last_octets = [1, 2, 254]  # Common router/gateway IPs
        scan_range = common_last_octets + list(range(3, min(254, max_hosts + 3)))
        
        for i in scan_range[:max_hosts]:
            ip = f"{network_prefix}.{i}"
            if ip == local_ip:
                continue  # Skip our own IP
                
            for port in self.discovery_ports:
                try:
                    url = f"http://{ip}:{port}/api"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{url}/tags",
                            timeout=self.discovery_timeout
                        ) as response:
                            if response.status == 200:
                                self.logger.info(f"Discovered Ollama server at {url}")
                                discovered_servers.append(url)
                except Exception:
                    pass
        
        return discovered_servers
    
    def _get_local_ip(self) -> Optional[str]:
        """
        Get the local IP address.
        
        Returns:
            Local IP address or None if not found
        """
        try:
            # Create a socket to determine the outgoing IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            s.connect(("8.8.8.8", 80))  # Google's DNS server
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception as e:
            self.logger.warning(f"Error getting local IP: {str(e)}")
            return None
    
    async def check_availability(self) -> bool:
        """
        Check if the Ollama API is available.
        
        Returns:
            True if available, False otherwise
        """
        # First try the configured base URL
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/tags",
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return True
        except Exception:
            pass
        
        # If not available, try to discover servers
        servers = await self.discover_ollama_servers()
        if servers:
            # Update base URL to the first discovered server
            self.base_url = servers[0]
            self.config_manager.set_config("api.providers.ollama.base_url", self.base_url)
            self.logger.info(f"Updated Ollama base URL to {self.base_url}")
            return True
        
        self.logger.warning("No Ollama servers found")
        return False
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from Ollama.
        
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
                    f"{self.base_url}/tags",
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        
                        # Convert to standard format
                        formatted_models = []
                        for model in models:
                            formatted_models.append({
                                "id": model.get("name", ""),
                                "name": model.get("name", ""),
                                "size": model.get("size", 0),
                                "modified_at": model.get("modified_at", ""),
                                "quantization_level": model.get("details", {}).get("quantization_level", "")
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
                {"provider": "ollama", "action": "get_models"}
            )
            return []
    
    async def send_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a chat completion request to Ollama.
        
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
            "model": request_data.get("model", self.default_model),
            "messages": request_data["messages"],
            "stream": request_data.get("stream", False),
            "options": {}
        }
        
        # Add optional parameters if provided
        if "temperature" in request_data:
            payload["options"]["temperature"] = request_data["temperature"]
        
        if "top_p" in request_data:
            payload["options"]["top_p"] = request_data["top_p"]
        
        if "max_tokens" in request_data:
            payload["options"]["num_predict"] = request_data["max_tokens"]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        ollama_response = await response.json()
                        
                        # Convert Ollama response format to OpenAI-like format for consistency
                        return {
                            "id": ollama_response.get("id", ""),
                            "object": "chat.completion",
                            "created": int(asyncio.get_event_loop().time()),
                            "model": ollama_response.get("model", ""),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": ollama_response.get("message", {}).get("content", "")
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                                "completion_tokens": ollama_response.get("eval_count", 0),
                                "total_tokens": (
                                    ollama_response.get("prompt_eval_count", 0) + 
                                    ollama_response.get("eval_count", 0)
                                )
                            }
                        }
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error in chat request: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status} - {error_text}"}
        except Exception as e:
            error_msg = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "ollama", "action": "chat_completion"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_completion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a text completion request to Ollama.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # Ensure we have the required fields
        if "prompt" not in request_data:
            return {"error": "Missing required field: prompt"}
        
        # Prepare request payload
        payload = {
            "model": request_data.get("model", self.default_model),
            "prompt": request_data["prompt"],
            "stream": request_data.get("stream", False),
            "options": {}
        }
        
        # Add optional parameters if provided
        if "temperature" in request_data:
            payload["options"]["temperature"] = request_data["temperature"]
        
        if "top_p" in request_data:
            payload["options"]["top_p"] = request_data["top_p"]
        
        if "max_tokens" in request_data:
            payload["options"]["num_predict"] = request_data["max_tokens"]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/generate",
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        ollama_response = await response.json()
                        
                        # Convert Ollama response format to OpenAI-like format for consistency
                        return {
                            "id": ollama_response.get("id", ""),
                            "object": "text_completion",
                            "created": int(asyncio.get_event_loop().time()),
                            "model": ollama_response.get("model", ""),
                            "choices": [
                                {
                                    "text": ollama_response.get("response", ""),
                                    "index": 0,
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                                "completion_tokens": ollama_response.get("eval_count", 0),
                                "total_tokens": (
                                    ollama_response.get("prompt_eval_count", 0) + 
                                    ollama_response.get("eval_count", 0)
                                )
                            }
                        }
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error in completion request: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status} - {error_text}"}
        except Exception as e:
            error_msg = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "ollama", "action": "completion"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_embedding_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an embedding request to Ollama.
        
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
            "model": request_data.get("model", self.default_model),
            "prompt": request_data["input"],
            "options": {}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        ollama_response = await response.json()
                        
                        # Convert Ollama response format to OpenAI-like format for consistency
                        return {
                            "object": "list",
                            "data": [
                                {
                                    "object": "embedding",
                                    "embedding": ollama_response.get("embedding", []),
                                    "index": 0
                                }
                            ],
                            "model": ollama_response.get("model", ""),
                            "usage": {
                                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                                "total_tokens": ollama_response.get("prompt_eval_count", 0)
                            }
                        }
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error in embedding request: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status} - {error_text}"}
        except Exception as e:
            error_msg = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "ollama", "action": "embedding"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_image_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an image generation request to Ollama.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # Ollama doesn't support image generation directly
        return {"error": "Image generation not supported by Ollama"}
    
    async def send_audio_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an audio processing request to Ollama.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # Ollama doesn't support audio processing
        return {"error": "Audio processing not supported by Ollama"}

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    config_manager = ConfigurationManager.get_instance()
    error_handler = ErrorHandler.get_instance()
    
    # Initialize adapter
    adapter = OllamaAdapter()
    
    # Example request
    async def test_request():
        # Check availability
        available = await adapter.check_availability()
        print(f"Ollama available: {available}")
        
        if available:
            # Get models
            models = await adapter.get_available_models()
            print(f"Available models: {len(models)}")
            
            # Send chat request
            response = await adapter.send_chat_request({
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "model": "llama2"
            })
            print(f"Chat response: {json.dumps(response, indent=2)}")
    
    # Run test request
    asyncio.run(test_request())
