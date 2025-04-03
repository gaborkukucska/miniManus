# START OF FILE miniManus-main/minimanus/api/api_manager.py
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
    - Provider status tracking and availability checks.
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

        # Rate limiting (placeholders)
        # self.rate_limits = {} ...

        # Request cache
        self.cache_enabled = self.config_manager.get_config("api.cache.enabled", True)
        self.cache: Dict[str, Tuple[float, Any]] = {} # Store (timestamp, response)
        self.cache_ttl = self.config_manager.get_config("api.cache.ttl_seconds", 3600) # Default 1 hour TTL
        self.max_cache_size = self.config_manager.get_config("api.cache.max_items", 500) # Max items
        self._cache_lock = asyncio.Lock()

        # Initialize provider status structure
        for provider in APIProvider:
             if provider != APIProvider.CUSTOM:
                self.provider_status[provider] = {
                    "available": False,
                    "last_check": 0,
                    "error": None, # Store last error message here
                }

        self.logger.info("APIManager initialized")

    def startup(self) -> None:
        """Start the API manager."""
        self._initialize_adapters()
        if self.cache_enabled:
            # Ensure loop is running before creating task
            try:
                 loop = asyncio.get_running_loop()
                 loop.create_task(self._cleanup_cache_periodically())
            except RuntimeError:
                 self.logger.warning("No running event loop, cache cleanup task not started.")

        self.logger.info("APIManager started")

    def shutdown(self) -> None:
        """Shut down the API manager."""
        # Cancel background tasks if necessary (needs task tracking)
        self.logger.info("APIManager shutting down")
        # Clear cache? Or persist? Current logic doesn't clear.
        self.logger.info("APIManager shut down")


    def _initialize_adapters(self) -> None:
        """Initialize and register API adapters for configured providers."""
        adapter_map = {}
        try:
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
            # Optional: Check if provider is enabled in config
            # if not self.config_manager.get_config(f"api.providers.{provider.name.lower()}.enabled", True): continue
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
        if provider not in self.provider_status:
             self.provider_status[provider] = {"available": False, "last_check": 0, "error": None}
        self.logger.debug(f"Registered adapter for {provider.name}")

    def get_adapter(self, provider: APIProvider) -> Optional[Any]:
        """Get the registered adapter instance for a provider."""
        return self.adapters.get(provider)

    async def check_provider_availability(self, provider: APIProvider, force_check: bool = False) -> bool:
        """
        Check if a provider is available, using cached status if recent and successful.

        Args:
            provider: The APIProvider enum member to check.
            force_check: If True, bypass the cache and perform a live check.

        Returns:
            True if the provider is considered available, False otherwise.
        """
        if provider == APIProvider.CUSTOM or provider not in self.adapters:
            self.logger.debug(f"Provider {provider.name} is CUSTOM or has no adapter. Reporting unavailable.")
            return False

        adapter = self.adapters[provider]
        current_time = time.time()

        async with self._status_lock:
            status = self.provider_status[provider]
            # Use cached status IF:
            # - Not forcing a check AND
            # - Check is recent AND
            # - Last check did NOT result in an error (status['error'] is None)
            if not force_check and \
               (current_time - status["last_check"] < self.status_check_interval) and \
               status["error"] is None:
                 # Return cached availability only if last check was successful
                 # self.logger.debug(f"Using cached availability for {provider.name}: {status['available']}")
                 return status["available"]

            # Perform a live check (or re-check if cache expired or last check failed)
            self.logger.debug(f"Performing live availability check for {provider.name} (Force: {force_check}, Last Status: {status})")
            available = False
            error_msg = None # Reset error message for new check
            adapter_last_error = None # Store potential error details from adapter

            try:
                if hasattr(adapter, 'check_availability') and callable(getattr(adapter, 'check_availability')):
                    available = await adapter.check_availability() # Adapter handles its internal logging
                    if not available:
                         # Try to get a reason if possible, otherwise generic message
                         adapter_last_error = getattr(adapter, 'last_check_error', "Adapter check failed")
                         error_msg = adapter_last_error # Store the specific error
                else:
                    # Basic check: API key configured? (Less reliable)
                    api_key = self.config_manager.get_api_key(provider.name.lower())
                    if api_key:
                        available = True
                        self.logger.debug(f"Provider {provider.name} assumed available (API key configured, no check method).")
                    else:
                         available = False
                         error_msg = "API key not configured and no check method"
                         self.logger.warning(f"Provider {provider.name} unavailable: {error_msg}")

            except Exception as e:
                error_msg = f"Exception during check: {str(e)}"
                self.logger.error(f"Error checking availability for {provider.name}: {e}", exc_info=False)
                self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.WARNING, {"provider": provider.name, "action": "check_availability"})
                available = False # Ensure unavailable on exception

            # Update status cache regardless of outcome
            status["available"] = available
            status["last_check"] = current_time
            status["error"] = error_msg # Store None if successful, error message otherwise
            self.logger.info(f"Provider {provider.name} availability updated: {available}" + (f" (Reason: {error_msg})" if error_msg else ""))

            return available


    async def get_available_providers(self) -> List[APIProvider]:
        """
        Get a list of currently available providers based on recent checks.
        Performs checks concurrently.

        Returns:
            List of available APIProvider enum members.
        """
        available_providers = []
        tasks = []
        # Check all registered providers except CUSTOM
        providers_to_check = [p for p in self.adapters.keys() if p != APIProvider.CUSTOM]

        for provider in providers_to_check:
             # Use cached check (check_provider_availability handles cache logic)
             tasks.append(self.check_provider_availability(provider))

        results = await asyncio.gather(*tasks)

        for provider, is_available in zip(providers_to_check, results):
             if is_available:
                 available_providers.append(provider)

        self.logger.debug(f"Available providers determined: {[p.name for p in available_providers]}")
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
            # Force a check if preferred? Or rely on cache? Rely on cache for now.
            if await self.check_provider_availability(preferred_provider):
                 # TODO: Add check if provider supports the request_type (adapter capability check)
                 self.logger.debug(f"Selected preferred provider: {preferred_provider.name}")
                 return preferred_provider
            else:
                 self.logger.warning(f"Preferred provider {preferred_provider.name} is not available.")

        # 2. Try the default provider from config if available
        default_provider_name = self.config_manager.get_config("api.default_provider")
        if default_provider_name:
            try:
                default_provider = APIProvider[default_provider_name.upper()]
                if default_provider != preferred_provider: # Avoid re-checking preferred
                     if await self.check_provider_availability(default_provider):
                         # TODO: Add check if provider supports the request_type
                         self.logger.debug(f"Selected default provider from config: {default_provider.name}")
                         return default_provider
                     else:
                         self.logger.debug(f"Default provider {default_provider.name} is not available.")
            except KeyError:
                self.logger.warning(f"Default provider name '{default_provider_name}' in config is invalid.")

        # 3. Fallback: Iterate through *all* registered adapters and check availability
        #    (Could add more sophisticated logic based on cost, capability later)
        self.logger.debug("Falling back to checking all registered providers.")
        all_adapters = list(self.adapters.items()) # Get items to iterate
        for provider, adapter in all_adapters:
            if provider == APIProvider.CUSTOM: continue # Skip CUSTOM
            # Avoid re-checking preferred or default if they were already checked above
            if provider != preferred_provider and provider.name.lower() != default_provider_name:
                 if await self.check_provider_availability(provider):
                     # TODO: Add check if provider supports the request_type
                     self.logger.debug(f"Selected first available provider as fallback: {provider.name}")
                     return provider

        # If we reach here, no provider was found
        self.logger.error("No suitable and available API providers found for the request.")
        return None

    async def send_request(self, request_type: APIRequestType,
                      request_data: Dict[str, Any],
                      preferred_provider: Optional[APIProvider] = None) -> Dict[str, Any]:
        """
        Sends an API request to the best available provider. Ensures provider availability
        is checked before sending. Updates provider status cache on errors.

        Args:
            request_type: The type of request (e.g., CHAT, EMBEDDING).
            request_data: The data payload for the request (e.g., messages, prompt).
                          Should contain the desired 'model' ID.
            preferred_provider: An optional specific provider to try first.

        Returns:
            The API response dictionary, or an error dictionary. Provider info can be added.
        """
        selected_provider = await self.select_provider(request_type, preferred_provider)

        if not selected_provider:
            error_msg = "No available API provider found for this request."
            self.error_handler.handle_error(RuntimeError(error_msg), ErrorCategory.API, ErrorSeverity.ERROR, {"request_type": request_type.name})
            return {"error": error_msg, "_provider_used": None}

        adapter = self.get_adapter(selected_provider)
        if not adapter:
            error_msg = f"Adapter not found for selected provider: {selected_provider.name}"
            self.error_handler.handle_error(RuntimeError(error_msg), ErrorCategory.SYSTEM, ErrorSeverity.ERROR, {"provider": selected_provider.name})
            return {"error": error_msg, "_provider_used": selected_provider.name} # Include provider even if adapter missing

        # Ensure Model ID is set correctly before sending to adapter
        # Priority: request_data['model'] > adapter's default for the type
        model_key = "model"
        default_model_getter = None

        # Find appropriate default model getter/attribute in the adapter
        default_attr_options = []
        if request_type == APIRequestType.EMBEDDING:
            default_attr_options.append('_get_current_default_embedding_model') # Dynamic method preferred
            default_attr_options.append('default_embedding_model') # Static attribute fallback
        # Default for CHAT, COMPLETION, etc.
        default_attr_options.append('_get_current_default_model') # Dynamic method preferred
        default_attr_options.append('default_model') # Static attribute fallback

        for attr_name in default_attr_options:
             if hasattr(adapter, attr_name):
                  default_model_getter = getattr(adapter, attr_name)
                  break # Use the first one found in priority order

        effective_model_name = request_data.get(model_key)
        if not effective_model_name:
            if default_model_getter:
                 if callable(default_model_getter):
                     effective_model_name = default_model_getter() # Call dynamic getter
                 else:
                     effective_model_name = default_model_getter # Access static attribute
                 self.logger.debug(f"Using default model '{effective_model_name}' for {selected_provider.name}/{request_type.name}")
            else:
                self.logger.warning(f"No model specified in request and no default model getter/attribute found in adapter for {selected_provider.name}/{request_type.name}. Adapter must handle.")

        # Ensure the effective model name is in request_data sent to adapter
        if effective_model_name:
            request_data[model_key] = effective_model_name
        elif model_key in request_data:
             # Remove explicit None or empty string if present originally
             del request_data[model_key]
             self.logger.debug(f"Removed explicit None/empty '{model_key}' from request_data.")


        # --- Caching Logic ---
        cache_key = self._generate_cache_key(selected_provider, request_type, request_data)
        if self.cache_enabled and cache_key:
            async with self._cache_lock:
                cached_entry = self.cache.get(cache_key)
                if cached_entry:
                    timestamp, cached_response = cached_entry
                    if time.time() - timestamp < self.cache_ttl:
                        self.logger.info(f"Returning cached response for {selected_provider.name}/{request_type.name}")
                        cached_response['_provider_used'] = selected_provider.name # Add provider info
                        cached_response['_cached'] = True
                        return cached_response
                    else:
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
            return {"error": error_msg, "_provider_used": selected_provider.name}

        # --- Execute Request ---
        try:
            self.logger.info(f"Sending {request_type.name} request to {selected_provider.name} (Model: {request_data.get('model', 'N/A')})...")
            api_method = getattr(adapter, method_name)
            response = await api_method(request_data) # Pass request_data with effective_model_name

            # Add provider info to the response
            response['_provider_used'] = selected_provider.name
            response['_cached'] = False

            # --- Caching Response ---
            if self.cache_enabled and cache_key and "error" not in response:
                async with self._cache_lock:
                     # Evict oldest if full
                     if len(self.cache) >= self.max_cache_size:
                         oldest_key = min(self.cache, key=lambda k: self.cache[k][0], default=None)
                         if oldest_key:
                              del self.cache[oldest_key]
                              self.logger.debug(f"Cache full, evicted oldest key: {oldest_key}")
                     # Store new response (don't store internal flags like _provider_used in cache)
                     response_to_cache = {k:v for k,v in response.items() if not k.startswith('_')}
                     self.cache[cache_key] = (time.time(), response_to_cache)
                     self.logger.debug(f"Cached response for key: {cache_key}")

            return response

        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Error during {request_type.name} request to {selected_provider.name}: {error_str}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.ERROR, {"provider": selected_provider.name, "request_type": request_type.name, "request_data": {"model": request_data.get("model")}}) # Avoid logging full data

            # Mark provider as unavailable after error
            async with self._status_lock:
                 self.provider_status[selected_provider]["available"] = False
                 self.provider_status[selected_provider]["last_check"] = time.time()
                 self.provider_status[selected_provider]["error"] = error_str # Store the error message
                 self.logger.info(f"Marked provider {selected_provider.name} as unavailable due to error: {error_str}")

            return {"error": f"API request failed: {error_str}", "_provider_used": selected_provider.name}


    def _generate_cache_key(self, provider: APIProvider, request_type: APIRequestType, request_data: Dict[str, Any]) -> Optional[str]:
        """Generates a cache key based on request parameters."""
        try:
             # Create a stable string representation (sort dict keys)
             # Exclude potentially sensitive or highly variable parts if needed (e.g., user ID)
             stable_data = json.dumps(request_data, sort_keys=True)
             key_string = f"{provider.name}:{request_type.name}:{stable_data}"
             return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
        except Exception as e:
             self.logger.warning(f"Failed to generate cache key: {e}")
             return None

    async def _cleanup_cache_periodically(self):
        """Background task to periodically clean up expired cache items."""
        while True:
            await asyncio.sleep(self.cache_ttl / 2 if self.cache_ttl > 0 else 3600) # Check interval
            if not self.cache_enabled: continue

            async with self._cache_lock:
                current_time = time.time()
                keys_to_delete = [
                    key for key, (timestamp, _) in self.cache.items()
                    if current_time - timestamp >= self.cache_ttl
                ]
                if keys_to_delete:
                     for key in keys_to_delete:
                         # Check existence before deleting (might have been removed)
                         if key in self.cache:
                              del self.cache[key]
                     self.logger.info(f"Cleaned up {len(keys_to_delete)} expired cache items.")

# END OF FILE miniManus-main/minimanus/api/api_manager.py
