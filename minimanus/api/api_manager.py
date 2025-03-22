#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Manager for miniManus

This module implements the API Manager component, which coordinates all API interactions,
manages API keys and authentication, implements rate limiting and quota management,
and monitors API health and availability.
"""

import os
import sys
import time
import logging
import threading
import json
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum, auto

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager

logger = logging.getLogger("miniManus.APIManager")

class APIProvider(Enum):
    """Supported API providers."""
    OPENROUTER = auto()
    DEEPSEEK = auto()
    ANTHROPIC = auto()
    OLLAMA = auto()
    LITELLM = auto()
    CUSTOM = auto()

class APIRequestType(Enum):
    """Types of API requests."""
    COMPLETION = auto()
    CHAT = auto()
    EMBEDDING = auto()
    IMAGE = auto()
    AUDIO = auto()

class APIManager:
    """
    APIManager coordinates all API interactions with LLM providers.
    
    It handles:
    - API key management and authentication
    - Provider selection and fallback
    - Rate limiting and quota management
    - Request formatting and response parsing
    - Error handling and recovery
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'APIManager':
        """Get or create the singleton instance of APIManager."""
        if cls._instance is None:
            cls._instance = APIManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the APIManager."""
        if APIManager._instance is not None:
            raise RuntimeError("APIManager is a singleton. Use get_instance() instead.")
        
        self.logger = logger
        self.event_bus = EventBus.get_instance()
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        
        # Provider adapters
        self.adapters = {}
        
        # Provider status
        self.provider_status = {}
        
        # Rate limiting
        self.rate_limits = {}
        self.request_counts = {}
        self.request_timestamps = {}
        self.rate_limit_lock = threading.RLock()
        
        # Request cache
        self.cache_enabled = True
        self.cache = {}
        self.cache_ttl = 86400  # 24 hours
        self.cache_lock = threading.RLock()
        self.max_cache_size = 1000  # Maximum number of cached responses
        
        # Initialize provider status
        for provider in APIProvider:
            self.provider_status[provider] = {
                "available": False,
                "last_check": 0,
                "error": None,
            }
        
        self.logger.info("APIManager initialized")
    
    def register_adapter(self, provider: APIProvider, adapter: Any) -> None:
        """
        Register an adapter for a provider.
        
        Args:
            provider: Provider to register adapter for
            adapter: Adapter instance
        """
        self.adapters[provider] = adapter
        self.logger.debug(f"Registered adapter for {provider.name}")
    
    def get_adapter(self, provider: APIProvider) -> Optional[Any]:
        """
        Get the adapter for a provider.
        
        Args:
            provider: Provider to get adapter for
            
        Returns:
            Adapter instance or None if not registered
        """
        return self.adapters.get(provider)
    
    def check_provider_availability(self, provider: APIProvider) -> bool:
        """
        Check if a provider is available.
        
        Args:
            provider: Provider to check
            
        Returns:
            True if available, False otherwise
        """
        # Check if we have an adapter for this provider
        if provider not in self.adapters:
            return False
        
        # Check if we have a recent status check
        status = self.provider_status[provider]
        current_time = time.time()
        
        # If we checked recently and it was available, return cached result
        if current_time - status["last_check"] < 300 and status["available"]:  # 5 minutes
            return True
        
        # Otherwise, perform a new check
        adapter = self.adapters[provider]
        try:
            # Call the adapter's availability check method
            if hasattr(adapter, 'check_availability'):
                available = adapter.check_availability()
            else:
                # Default to assuming available if we have an API key
                api_key = self.config_manager.get_api_key(provider.name.lower())
                available = api_key is not None
            
            # Update status
            status["available"] = available
            status["last_check"] = current_time
            status["error"] = None if available else "Provider not available"
            
            return available
        
        except Exception as e:
            # Update status with error
            status["available"] = False
            status["last_check"] = current_time
            status["error"] = str(e)
            
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.WARNING,
                {"provider": provider.name, "action": "check_availability"}
            )
            
            return False
    
    def get_available_providers(self) -> List[APIProvider]:
        """
        Get a list of available providers.
        
        Returns:
            List of available providers
        """
        available = []
        
        for provider in APIProvider:
            if self.check_provider_availability(provider):
                available.append(provider)
        
        return available
    
    def select_provider(self, request_type: APIRequestType, 
                       preferred_provider: Optional[APIProvider] = None) -> Optional[APIProvider]:
        """
        Select a provider for a request.
        
        Args:
            request_type: Type of request
            preferred_provider: Preferred provider to use
            
        Returns:
            Selected provider or None if no provider is available
        """
        # Check if preferred provider is available
        if preferred_provider and self.check_provider_availability(preferred_provider):
            return preferred_provider
        
        # Get default provider from config
        default_provider_name = self.config_manager.get_config("api.default_provider")
        if default_provider_name:
            try:
                default_provider = APIProvider[default_provider_name.upper()]
                if self.check_provider_availability(default_provider):
                    return default_provider
            except (KeyError, ValueError):
                pass
        
        # Fall back to any available provider
        available_providers = self.get_available_providers()
        if available_providers:
            return available_providers[0]
        
        return None
    
    def check_rate_limit(self, provider: APIProvider) -> bool:
        """
        Check if a provider is rate limited.
        
        Args:
            provider: Provider to check
            
        Returns:
            True if not rate limited, False if rate limited
        """
        with self.rate_limit_lock:
            if provider not in self.rate_limits:
                return True
            
            limit = self.rate_limits[provider]
            current_time = time.time()
            
            # Clean up old timestamps
            if provider in self.request_timestamps:
                self.request_timestamps[provider] = [
                    ts for ts in self.request_timestamps[provider]
                    if current_time - ts < limit["window"]
                ]
            
            # Check if we're under the limit
            if provider not in self.request_counts:
                self.request_counts[provider] = 0
            
            if provider not in self.request_timestamps:
                self.request_timestamps[provider] = []
            
            count = len(self.request_timestamps[provider])
            
            return count < limit["requests"]
    
    def update_rate_limit(self, provider: APIProvider, requests: int, window: int) -> None:
        """
        Update rate limit for a provider.
        
        Args:
            provider: Provider to update rate limit for
            requests: Maximum number of requests
            window: Time window in seconds
        """
        with self.rate_limit_lock:
            self.rate_limits[provider] = {
                "requests": requests,
                "window": window,
            }
    
    def record_request(self, provider: APIProvider) -> None:
        """
        Record a request to a provider for rate limiting.
        
        Args:
            provider: Provider to record request for
        """
        with self.rate_limit_lock:
            current_time = time.time()
            
            if provider not in self.request_counts:
                self.request_counts[provider] = 0
            
            if provider not in self.request_timestamps:
                self.request_timestamps[provider] = []
            
            self.request_counts[provider] += 1
            self.request_timestamps[provider].append(current_time)
    
    def get_cache_key(self, provider: APIProvider, request_type: APIRequestType, 
                     request_data: Dict[str, Any]) -> str:
        """
        Generate a cache key for a request.
        
        Args:
            provider: Provider for the request
            request_type: Type of request
            request_data: Request data
            
        Returns:
            Cache key string
        """
        # Create a deterministic string representation of the request
        key_parts = [
            provider.name,
            request_type.name,
        ]
        
        # Add relevant request data to the key
        if request_type == APIRequestType.CHAT:
            if "messages" in request_data:
                # For chat, include the messages
                messages_str = json.dumps(request_data["messages"], sort_keys=True)
                key_parts.append(messages_str)
            
            # Include model if specified
            if "model" in request_data:
                key_parts.append(request_data["model"])
        
        elif request_type == APIRequestType.COMPLETION:
            # For completion, include the prompt
            if "prompt" in request_data:
                key_parts.append(request_data["prompt"])
            
            # Include model if specified
            if "model" in request_data:
                key_parts.append(request_data["model"])
        
        elif request_type == APIRequestType.EMBEDDING:
            # For embedding, include the input text
            if "input" in request_data:
                key_parts.append(request_data["input"])
            
            # Include model if specified
            if "model" in request_data:
                key_parts.append(request_data["model"])
        
        # Join parts and create a hash
        key_str = "|".join(key_parts)
        return f"cache:{hash(key_str)}"
    
    def get_from_cache(self, provider: APIProvider, request_type: APIRequestType,
                      request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get a response from the cache.
        
        Args:
            provider: Provider for the request
            request_type: Type of request
            request_data: Request data
            
        Returns:
            Cached response or None if not in cache
        """
        if not self.cache_enabled:
            return None
        
        cache_key = self.get_cache_key(provider, request_type, request_data)
        
        with self.cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check if entry is expired
                if time.time() - entry["timestamp"] > self.cache_ttl:
                    del self.cache[cache_key]
                    return None
                
                self.logger.debug(f"Cache hit for {provider.name} {request_type.name}")
                return entry["response"]
        
        return None
    
    def add_to_cache(self, provider: APIProvider, request_type: APIRequestType,
                    request_data: Dict[str, Any], response: Dict[str, Any]) -> None:
        """
        Add a response to the cache.
        
        Args:
            provider: Provider for the request
            request_type: Type of request
            request_data: Request data
            response: Response to cache
        """
        if not self.cache_enabled:
            return
        
        # Don't cache error responses
        if "error" in response:
            return
        
        cache_key = self.get_cache_key(provider, request_type, request_data)
        
        with self.cache_lock:
            # Check if cache is full
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
                del self.cache[oldest_key]
            
            # Add new entry
            self.cache[cache_key] = {
                "timestamp": time.time(),
                "response": response,
            }
            
            self.logger.debug(f"Added to cache: {provider.name} {request_type.name}")
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        with self.cache_lock:
            self.cache = {}
            self.logger.debug("Cache cleared")
    
    async def send_request(self, request_type: APIRequestType, request_data: Dict[str, Any],
                         preferred_provider: Optional[APIProvider] = None,
                         use_cache: bool = True) -> Dict[str, Any]:
        """
        Send a request to an API provider.
        
        Args:
            request_type: Type of request
            request_data: Request data
            preferred_provider: Preferred provider to use
            use_cache: Whether to use the cache
            
        Returns:
            Response from the provider
        """
        # Select provider
        provider = self.select_provider(request_type, preferred_provider)
        if not provider:
            error_msg = "No available API providers"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        # Check cache if enabled
        if use_cache and self.cache_enabled:
            cached_response = self.get_from_cache(provider, request_type, request_data)
            if cached_response:
                return cached_response
        
        # Check rate limit
        if not self.check_rate_limit(provider):
            error_msg = f"Rate limit exceeded for {provider.name}"
            self.logger.warning(error_msg)
            
            # Try another provider
            other_providers = [p for p in self.get_available_providers() if p != provider]
            if other_providers:
                self.logger.info(f"Trying alternate provider: {other_providers[0].name}")
                return await self.send_request(request_type, request_data, other_providers[0], use_cache)
            
            return {"error": error_msg}
        
        # Get adapter
        adapter = self.adapters.get(provider)
        if not adapter:
            error_msg = f"No adapter registered for {provider.name}"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            # Record request for rate limiting
            self.record_request(provider)
            
            # Send request through adapter
            if request_type == APIRequestType.CHAT:
                response = await adapter.send_chat_request(request_data)
            elif request_type == APIRequestType.COMPLETION:
                response = await adapter.send_completion_request(request_data)
            elif request_type == APIRequestType.EMBEDDING:
                response = await adapter.send_embedding_request(request_data)
            elif request_type == APIRequestType.IMAGE:
                response = await adapter.send_image_request(request_data)
            elif request_type == APIRequestType.AUDIO:
                response = await adapter.send_audio_request(request_data)
            else:
                error_msg = f"Unsupported request type: {request_type.name}"
                self.logger.error(error_msg)
                return {"error": error_msg}
            
            # Cache successful response
            if use_cache and self.cache_enabled and "error" not in response:
                self.add_to_cache(provider, request_type, request_data, response)
            
            return response
        
        except Exception as e:
            error_msg = f"Error sending {request_type.name} request to {provider.name}: {str(e)}"
            self.logger.error(error_msg)
            
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": provider.name, "request_type": request_type.name}
            )
            
            # Try another provider
            other_providers = [p for p in self.get_available_providers() if p != provider]
            if other_providers:
                self.logger.info(f"Trying alternate provider: {other_providers[0].name}")
                return await self.send_request(request_type, request_data, other_providers[0], use_cache)
            
            return {"error": error_msg}
    
    def startup(self) -> None:
        """Start the API manager."""
        # Load cache settings from config
        self.cache_enabled = self.config_manager.get_config("api.cache.enabled", True)
        self.cache_ttl = self.config_manager.get_config("api.cache.ttl_seconds", 86400)
        max_cache_size_mb = self.config_manager.get_config("api.cache.max_size_mb", 50)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024 // 1000  # Rough estimate
        
        self.logger.info("APIManager started")
    
    def shutdown(self) -> None:
        """Stop the API manager."""
        # Nothing special to do here
        self.logger.info("APIManager stopped")

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    error_handler = ErrorHandler.get_instance()
    
    config_manager = ConfigurationManager.get_instance()
    
    # Initialize APIManager
    api_manager = APIManager.get_instance()
    api_manager.startup()
    
    # Example adapter class
    class DummyAdapter:
        async def check_availability(self):
            return True
        
        async def send_chat_request(self, request_data):
            return {"response": "This is a dummy response"}
    
    # Register adapter
    api_manager.register_adapter(APIProvider.OPENROUTER, DummyAdapter())
    
    # Example request
    async def test_request():
        response = await api_manager.send_request(
            APIRequestType.CHAT,
            {"messages": [{"role": "user", "content": "Hello"}]}
        )
        print(f"Response: {response}")
    
    # Run test request
    import asyncio
    asyncio.run(test_request())
    
    # Shutdown
    api_manager.shutdown()
    event_bus.shutdown()
