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
    
    def check_availability(self) -> bool:
        """
        Check if the Ollama API is available.
        
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
    
    def generate_text(self, messages: List[Dict[str, str]], model: str = None, 
                     temperature: float = 0.7, max_tokens: int = 1024, 
                     api_key: str = None) -> str:
        """
        Generate text using the Ollama API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use (defaults to configured default model)
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            api_key: API key (not used for Ollama but included for compatibility)
            
        Returns:
            Generated text
        """
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._generate_text_async(messages, model, temperature, max_tokens)
            )
        finally:
            loop.close()
    
    async def _generate_text_async(self, messages: List[Dict[str, str]], 
                                 model: str = None, temperature: float = 0.7, 
                                 max_tokens: int = 1024) -> str:
        """
        Async implementation of generate_text.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use (defaults to configured default model)
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        if not model:
            model = self.default_model
        
        # Format messages for Ollama API
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            # Ollama uses 'user' and 'assistant' roles
            if role == "system":
                # For system messages, we'll prepend to the first user message
                # or add as a user message if there are no user messages
                system_content = content
                continue
            
            formatted_messages.append({
                "role": role,
                "content": content
            })
        
        # Add system message as a prefix to the first user message if it exists
        if 'system_content' in locals() and formatted_messages:
            for i, msg in enumerate(formatted_messages):
                if msg["role"] == "user":
                    formatted_messages[i]["content"] = f"{system_content}\n\n{msg['content']}"
                    break
            else:
                # No user messages found, add system as a user message
                formatted_messages.append({
                    "role": "user",
                    "content": system_content
                })
        
        # Prepare request data
        request_data = {
            "model": model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat",
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("message", {}).get("content", "")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error generating text: {response.status} - {error_text}")
                        return f"Error: {error_text}"
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Exception in generate_text: {error_msg}")
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": "ollama", "action": "generate_text"}
            )
            return f"Error: {error_msg}"
