#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LiteLLM API Adapter for miniManus

This module implements the adapter for the LiteLLM API, which provides
a unified interface to multiple LLM providers.
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

logger = logging.getLogger("miniManus.LiteLLMAdapter")

class LiteLLMAdapter:
    """
    Adapter for the LiteLLM API.
    
    This adapter provides methods to interact with the LiteLLM API,
    which offers a unified interface to multiple LLM providers.
    """
    
    def __init__(self):
        """Initialize the LiteLLM adapter."""
        self.logger = logger
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        
        # API configuration
        self.base_url = self.config_manager.get_config(
            "api.providers.litellm.base_url", 
            "http://localhost:8000"  # Default LiteLLM proxy server URL
        )
        self.timeout = self.config_manager.get_config(
            "api.providers.litellm.timeout", 
            30
        )
        self.default_model = self.config_manager.get_config(
            "api.providers.litellm.default_model", 
            "gpt-3.5-turbo"  # Default model
        )
        
        # Available models cache
        self.models_cache = None
        self.models_cache_timestamp = 0
        self.models_cache_ttl = 300  # 5 minutes
        
        # Network discovery settings
        self.discovery_enabled = self.config_manager.get_config(
            "api.providers.litellm.discovery_enabled",
            True
        )
        self.discovery_ports = self.config_manager.get_config(
            "api.providers.litellm.discovery_ports",
            [8000, 8080]  # Common ports for LiteLLM proxy
        )
        self.discovery_timeout = self.config_manager.get_config(
            "api.providers.litellm.discovery_timeout",
            0.5  # seconds
        )
        
        self.logger.info("LiteLLMAdapter initialized")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.
        
        Returns:
            Dictionary of headers
        """
        api_key = self.config_manager.get_api_key("litellm")
        headers = {
            "Content-Type": "application/json",
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        return headers
    
    async def discover_litellm_servers(self) -> List[str]:
        """
        Discover LiteLLM servers on the local network.
        
        Returns:
            List of discovered LiteLLM server URLs
        """
        if not self.discovery_enabled:
            return []
        
        discovered_servers = []
        
        # First check localhost
        for port in self.discovery_ports:
            try:
                url = f"http://localhost:{port}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{url}/models",
                        headers=self._get_headers(),
                        timeout=self.discovery_timeout
                    ) as response:
                        if response.status == 200:
                            self.logger.info(f"Discovered local LiteLLM server at {url}")
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
            "api.providers.litellm.discovery_max_hosts",
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
                    url = f"http://{ip}:{port}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{url}/models",
                            headers=self._get_headers(),
                            timeout=self.discovery_timeout
                        ) as response:
                            if response.status == 200:
                                self.logger.info(f"Discovered LiteLLM server at {url}")
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
            import socket
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
        Check if the LiteLLM API is available.
        
        Returns:
            True if available, False otherwise
        """
        # First try the configured base URL
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return True
        except Exception:
            pass
        
        # If not available, try to discover servers
        servers = await self.discover_litellm_servers()
        if servers:
            # Update base URL to the first discovered server
            self.base_url = servers[0]
            self.config_manager.set_config("api.providers.litellm.base_url", self.base_url)
            self.logger.info(f"Updated LiteLLM base URL to {self.base_url}")
            return True
        
        self.logger.warning("No LiteLLM servers found")
        return False
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from LiteLLM.
        
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
                        models = data.get("data", [])
                        
                        self.models_cache = models
                        self.models_cache_timestamp = current_time
                        return models
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error fetching models: {response.status} - {error_text}")
                        return []
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "litellm", "action": "get_models"}
            )
            return []
    
    async def send_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a chat completion request to LiteLLM.
        
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
                {"provider": "litellm", "action": "chat_completion"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_completion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a text completion request to LiteLLM.
        
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
            "prompt": request_data["prompt"],
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
                    f"{self.base_url}/completions",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error in completion request: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status} - {error_text}"}
        except Exception as e:
            error_msg = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "litellm", "action": "completion"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_embedding_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an embedding request to LiteLLM.
        
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
                {"provider": "litellm", "action": "embedding"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_image_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an image generation request to LiteLLM.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # LiteLLM might support image generation through DALL-E models
        if "prompt" not in request_data:
            return {"error": "Missing required field: prompt"}
        
        # Prepare request payload
        payload = {
            "prompt": request_data["prompt"],
            "model": request_data.get("model", "dall-e-3"),  # Default image model
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
                        return {"error": "Image generation not supported by this LiteLLM server"}
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error in image request: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status} - {error_text}"}
        except Exception as e:
            error_msg = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "litellm", "action": "image_generation"}
            )
            return {"error": f"Request error: {error_msg}"}
    
    async def send_audio_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an audio processing request to LiteLLM.
        
        Args:
            request_data: Request parameters
            
        Returns:
            Response from the API
        """
        # LiteLLM might support audio processing through Whisper models
        if "file" not in request_data:
            return {"error": "Missing required field: file"}
        
        # LiteLLM proxy might not support audio processing
        return {"error": "Audio processing not supported by LiteLLM"}

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    config_manager = ConfigurationManager.get_instance()
    error_handler = ErrorHandler.get_instance()
    
    # Set API key for testing (optional for LiteLLM)
    config_manager.set_api_key("litellm", "your_api_key_here")
    
    # Initialize adapter
    adapter = LiteLLMAdapter()
    
    # Example request
    async def test_request():
        # Check availability
        available = await adapter.check_availability()
        print(f"LiteLLM available: {available}")
        
        if available:
            # Get models
            models = await adapter.get_available_models()
            print(f"Available models: {len(models)}")
            
            # Send chat request
            response = await adapter.send_chat_request({
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "model": "gpt-3.5-turbo"
            })
            print(f"Chat response: {json.dumps(response, indent=2)}")
    
    # Run test request
    asyncio.run(test_request())
