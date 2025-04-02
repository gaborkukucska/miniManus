import os
import sys
import time
import logging
import threading
import json
import asyncio
import hashlib
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum, auto

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
except ImportError as e:
     # Handle potential import errors during early startup or testing
    logging.getLogger("miniManus.APIManager").critical(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(f"ImportError in api_manager.py: {e}. Ensure all components exist.")


logger = logging.getLogger("miniManus.APIManager")

class APIProvider(Enum):
    """Supported API providers."""
    OPENROUTER = auto()
    DEEPSEEK = auto()
    ANTHROPIC = auto()
    OLLAMA = auto()
    LITELLM = auto()
    CUSTOM = auto() # For future extensibility

class APIRequestType(Enum):
    """Types of API requests."""
    CHAT = auto()
    COMPLETION = auto() # Legacy/alternative to chat
    EMBEDDING = auto()
    IMAGE = auto()
    AUDIO = auto()
    # Add more types as needed (e.g., MODERATION)

class APIManager:
    """
    APIManager coordinates all API interactions with LLM providers.

    Handles:
    - API key management via ConfigManager.
    - Loading and managing provider adapters.
    - Selecting appropriate providers based on availability and preferences.
    - Routing requests to the correct adapter method.
    - Basic caching of API responses.
    - (Future) Rate limiting and quota management.
    - Centralized error handling for API interactions.
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

        # Provider adapters registry
        self.adapters: Dict[APIProvider, Any] = {}

        # Provider status cache (last check time, availability, error message)
        self.provider_status: Dict[APIProvider, Dict[str, Any]] = {}
        self._status_lock = asyncio.Lock() # Lock for accessing provider_status
        self.status_check_interval = 300 # Re-check availability every 5 minutes

        # Rate limiting (placeholders for future implementation)
        # self.rate_limits = {}
        # self.request_counts = {}
        # self.request_timestamps = {}
        # self.rate_limit_lock = asyncio.Lock()

        # Request cache
        self.cache_enabled = self.config_manager.get_config("api.cache.enabled", True)
        self.cache: Dict[str, Tuple[float, Any]] = {} # Store (timestamp, response)
        self.cache_ttl = self.config_manager.get_config("api.cache.ttl_seconds", 3600) # Default 1 hour TTL
        self.max_cache_size = self.config_manager.get_config("api.cache.max_items", 500) # Max items
        self._cache_lock = asyncio.Lock()

        # Initialize provider status structure
        for provider in APIProvider:
             # Don't initialize status for CUSTOM initially
             if provider != APIProvider.CUSTOM:
                self.provider_status[provider] = {
                    "available": False,
                    "last_check": 0,
                    "error": None,
                }

        self.logger.info("APIManager initialized")

    def startup(self) -> None:
        """Start the API manager."""
        self._initialize_adapters()
        # Optionally start background tasks (e.g., cache cleanup, status polling)
        if self.cache_enabled:
            asyncio.create_task(self._cleanup_cache_periodically())
        self.logger.info("APIManager started")

    def shutdown(self) -> None:
        """Shut down the API manager."""
        # Cancel any background tasks if necessary
        self.logger.info("APIManager shutting down")
        # Clear cache? Or persist it depending on design
        # async with self._cache_lock:
        #     self.cache.clear()
        self.logger.info("APIManager shut down")


    def _initialize_adapters(self) -> None:
        """Initialize and register API adapters for configured providers."""
        # Define adapter classes corresponding to APIProvider enums
        adapter_map = {}
        try:
            # Import adapters here to avoid potential circular imports at module level
            from .ollama_adapter import OllamaAdapter
            from .openrouter_adapter import OpenRouterAdapter
            from .anthropic_adapter import AnthropicAdapter
            from .deepseek_adapter import DeepSeekAdapter
            from .litellm_adapter import LiteLLMAdapter
            adapter_map = {
                APIProvider.OLLAMA: OllamaAdapter,
                APIProvider.OPENROUTER: OpenRouterAdapter,
                APIProvider.ANTHROPIC: AnthropicAdapter,
                APIProvider.DEEPSEEK: DeepSeekAdapter,
                APIProvider.LITELLM: LiteLLMAdapter,
            }
        except ImportError as e:
            self.logger.error(f"Failed to import one or more API adapters: {e}. Some providers may be unavailable.", exc_info=True)


        for provider, AdapterClass in adapter_map.items():
            # Check if provider is generally enabled (optional)
            # if not self.config_manager.get_config(f"api.providers.{provider.name.lower()}.enabled", True):
            #     self.logger.info(f"Skipping initialization for disabled provider: {provider.name}")
            #     continue

            try:
                adapter_instance = AdapterClass()
                self.register_adapter(provider, adapter_instance)
            except Exception as e:
                self.logger.error(f"Failed to initialize adapter for {provider.name}: {e}", exc_info=True)
                self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.ERROR, {"provider": provider.name, "action": "initialize_adapter"})

        self.logger.info(f"Initialized {len(self.adapters)} API adapters.")

    def register_adapter(self, provider: APIProvider, adapter: Any) -> None:
        """Register an adapter instance for a specific provider."""
        if provider in self.adapters:
             self.logger.warning(f"Replacing existing adapter for {provider.name}")
        self.adapters[provider] = adapter
        if provider not in self.provider_status: # Initialize status if needed (e.g., for CUSTOM)
             self.provider_status[provider] = {"available": False, "last_check": 0, "error": None}
        self.logger.debug(f"Registered adapter for {provider.name}")

    def get_adapter(self, provider: APIProvider) -> Optional[Any]:
        """Get the registered adapter instance for a provider."""
        return self.adapters.get(provider)

    async def check_provider_availability(self, provider: APIProvider, force_check: bool = False) -> bool:
        """
        Check if a provider is available, using cached status if recent.

        Args:
            provider: The APIProvider enum member to check.
            force_check: If True, bypass the cache and perform a live check.

        Returns:
            True if the provider is considered available, False otherwise.
        """
        if provider == APIProvider.CUSTOM or provider not in self.adapters:
            return False # Cannot check custom or non-registered providers

        adapter = self.adapters[provider]
        current_time = time.time()

        async with self._status_lock:
            status = self.provider_status[provider]
            # Use cached status if recent and not forcing a check
            if not force_check and (current_time - status["last_check"] < self.status_check_interval):
                 # self.logger.debug(f"Using cached availability for {provider.name}: {status['available']}")
                 return status["available"]

            # Perform a live check
            self.logger.debug(f"Performing live availability check for {provider.name}")
            available = False
            error_msg = "Check not performed"
            try:
                if hasattr(adapter, 'check_availability') and callable(getattr(adapter, 'check_availability')):
                    # Adapter has its own check method (preferred)
                    available = await adapter.check_availability()
                    error_msg = None if available else "Adapter check failed"
                else:
                    # Basic check: Does the provider have an API key configured?
                    api_key = self.config_manager.get_api_key(provider.name.lower())
                    if api_key:
                        available = True # Assume available if key exists (less reliable)
                        error_msg = None
                    else:
                         available = False
                         error_msg = "API key not configured"
                         self.logger.warning(f"Provider {provider.name} has no check_availability method and no API key configured.")

                # Update status cache
                status["available"] = available
                status["last_check"] = current_time
                status["error"] = error_msg
                self.logger.info(f"Provider {provider.name} availability updated: {available}")

            except Exception as e:
                self.logger.error(f"Error checking availability for {provider.name}: {e}", exc_info=False)
                status["available"] = False
                status["last_check"] = current_time
                status["error"] = str(e)
                self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.WARNING, {"provider": provider.name, "action": "check_availability"})
                available = False

            return available


    async def get_available_providers(self) -> List[APIProvider]:
        """
        Get a list of currently available providers based on recent checks.

        Returns:
            List of available APIProvider enum members.
        """
        available_providers = []
        tasks = []
        providers_to_check = list(self.adapters.keys()) # Check all registered providers

        for provider in providers_to_check:
             tasks.append(self.check_provider_availability(provider)) # Use cached check

        results = await asyncio.gather(*tasks)

        for provider, is_available in zip(providers_to_check, results):
             if is_available:
                 available_providers.append(provider)

        return available_providers

    async def select_provider(self, request_type: APIRequestType,
                       preferred_provider: Optional[APIProvider] = None) -> Optional[APIProvider]:
        """
        Selects the best available provider for a given request type.

        Args:
            request_type: The type of API request (e.g., CHAT, EMBEDDING).
            preferred_provider: An optional specific provider to try first.

        Returns:
            The selected APIProvider enum member, or None if no suitable provider is found.
        """
        # 1. Try the preferred provider if specified and available
        if preferred_provider:
            if await self.check_provider_availability(preferred_provider):
                 # Optional: Add check if provider supports the request_type
                 self.logger.debug(f"Selected preferred provider: {preferred_provider.name}")
                 return preferred_provider
            else:
                 self.logger.warning(f"Preferred provider {preferred_provider.name} is not available.")

        # 2. Try the default provider from config if available
        default_provider_name = self.config_manager.get_config("api.default_provider")
        if default_provider_name:
            try:
                default_provider = APIProvider[default_provider_name.upper()]
                if await self.check_provider_availability(default_provider):
                     # Optional: Add check if provider supports the request_type
                     self.logger.debug(f"Selected default provider from config: {default_provider.name}")
                     return default_provider
                else:
                     self.logger.debug(f"Default provider {default_provider.name} is not available.")
            except KeyError:
                self.logger.warning(f"Default provider name '{default_provider_name}' in config is invalid.")

        # 3. Fallback: Iterate through all available providers and return the first one
        #    (Could add more sophisticated logic here, e.g., based on cost, capability)
        available_providers = await self.get_available_providers()
        if available_providers:
             # Optional: Add check if provider supports the request_type
             selected = available_providers[0]
             self.logger.debug(f"Selected first available provider as fallback: {selected.name}")
             return selected
        else:
             self.logger.error("No API providers available for the request.")
             return None

    async def send_request(self, request_type: APIRequestType,
                      request_data: Dict[str, Any],
                      preferred_provider: Optional[APIProvider] = None) -> Dict[str, Any]:
        """
        Sends an API request to the best available provider.

        Args:
            request_type: The type of request (e.g., CHAT, EMBEDDING).
            request_data: The data payload for the request (e.g., messages, prompt).
            preferred_provider: An optional specific provider to try first.

        Returns:
            The API response dictionary, or an error dictionary.
        """
        selected_provider = await self.select_provider(request_type, preferred_provider)

        if not selected_provider:
            error_msg = "No available API provider found for this request."
            self.error_handler.handle_error(RuntimeError(error_msg), ErrorCategory.API, ErrorSeverity.ERROR, {"request_type": request_type.name})
            return {"error": error_msg}

        adapter = self.get_adapter(selected_provider)
        if not adapter:
            error_msg = f"Adapter not found for selected provider: {selected_provider.name}"
            self.error_handler.handle_error(RuntimeError(error_msg), ErrorCategory.SYSTEM, ErrorSeverity.ERROR, {"provider": selected_provider.name})
            return {"error": error_msg}

        # --- Caching Logic ---
        cache_key = self._generate_cache_key(selected_provider, request_type, request_data)
        if self.cache_enabled and cache_key:
            async with self._cache_lock:
                cached_entry = self.cache.get(cache_key)
                if cached_entry:
                    timestamp, cached_response = cached_entry
                    if time.time() - timestamp < self.cache_ttl:
                        self.logger.info(f"Returning cached response for {selected_provider.name}/{request_type.name}")
                        return cached_response # Return the cached response directly
                    else:
                        # Cache expired, remove it
                        del self.cache[cache_key]
                        self.logger.debug(f"Cache expired for key: {cache_key}")

        # --- Map Request Type to Adapter Method ---
        method_name_map = {
            APIRequestType.CHAT: "send_chat_request",
            APIRequestType.COMPLETION: "send_completion_request",
            APIRequestType.EMBEDDING: "send_embedding_request",
            APIRequestType.IMAGE: "send_image_request",
            APIRequestType.AUDIO: "send_audio_request",
        }
        method_name = method_name_map.get(request_type)

        if not method_name or not hasattr(adapter, method_name):
            error_msg = f"Adapter for {selected_provider.name} does not support request type: {request_type.name}"
            self.error_handler.handle_error(NotImplementedError(error_msg), ErrorCategory.API, ErrorSeverity.WARNING, {"provider": selected_provider.name, "request_type": request_type.name})
            return {"error": error_msg}

        # --- Execute Request ---
        try:
            self.logger.info(f"Sending {request_type.name} request to {selected_provider.name}...")
            api_method = getattr(adapter, method_name)
            response = await api_method(request_data)

            # --- Caching Response ---
            if self.cache_enabled and cache_key and "error" not in response:
                async with self._cache_lock:
                     # Evict oldest item if cache is full
                     if len(self.cache) >= self.max_cache_size:
                         oldest_key = min(self.cache, key=lambda k: self.cache[k][0])
                         del self.cache[oldest_key]
                         self.logger.debug(f"Cache full, evicted oldest key: {oldest_key}")
                     # Store new response
                     self.cache[cache_key] = (time.time(), response)
                     self.logger.debug(f"Cached response for key: {cache_key}")

            return response

        except Exception as e:
            self.logger.error(f"Error during {request_type.name} request to {selected_provider.name}: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.ERROR, {"provider": selected_provider.name, "request_type": request_type.name, "request_data": request_data})
            # Mark provider as potentially unavailable after error
            async with self._status_lock:
                 self.provider_status[selected_provider]["available"] = False
                 self.provider_status[selected_provider]["last_check"] = time.time()
                 self.provider_status[selected_provider]["error"] = str(e)
            return {"error": f"API request failed: {str(e)}"}

    def _generate_cache_key(self, provider: APIProvider, request_type: APIRequestType, request_data: Dict[str, Any]) -> Optional[str]:
        """Generates a cache key based on request parameters."""
        try:
             # Create a stable string representation (sort dict keys)
             # Exclude potentially sensitive or highly variable parts if needed
             stable_data = json.dumps(request_data, sort_keys=True)
             key_string = f"{provider.name}:{request_type.name}:{stable_data}"
             # Use SHA256 hash for a fixed-size key
             return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
        except Exception as e:
             self.logger.warning(f"Failed to generate cache key: {e}")
             return None # Don't cache if key generation fails

    async def _cleanup_cache_periodically(self):
        """Background task to periodically clean up expired cache items."""
        while True:
            await asyncio.sleep(self.cache_ttl / 2) # Check roughly twice per TTL interval
            if not self.cache_enabled: continue

            async with self._cache_lock:
                current_time = time.time()
                expired_keys = [
                    key for key, (timestamp, _) in self.cache.items()
                    if current_time - timestamp >= self.cache_ttl
                ]
                if expired_keys:
                     for key in expired_keys:
                         del self.cache[key]
                     self.logger.info(f"Cleaned up {len(expired_keys)} expired cache items.")
