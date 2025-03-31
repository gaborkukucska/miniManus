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
    
    def startup(self) -> None:
        """Start the API manager."""
        # Initialize adapters
        self._initialize_adapters()
        
        self.logger.info("APIManager started")
    
    def _initialize_adapters(self) -> None:
        """Initialize API adapters for all providers."""
        try:
            # Import adapters here to avoid circular imports
            from .ollama_adapter import OllamaAdapter
            from .openrouter_adapter import OpenRouterAdapter
            from .anthropic_adapter import AnthropicAdapter
            from .deepseek_adapter import DeepSeekAdapter
            from .litellm_adapter import LiteLLMAdapter
            
            # Register adapters
            self.register_adapter(APIProvider.OLLAMA, OllamaAdapter())
            self.register_adapter(APIProvider.OPENROUTER, OpenRouterAdapter())
            self.register_adapter(APIProvider.ANTHROPIC, AnthropicAdapter())
            self.register_adapter(APIProvider.DEEPSEEK, DeepSeekAdapter())
            self.register_adapter(APIProvider.LITELLM, LiteLLMAdapter())
            
            self.logger.info("API adapters initialized successfully")
        except ImportError as e:
            self.logger.warning(f"Error importing adapters: {str(e)}")
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"action": "initialize_adapters"}
            )
    
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
    
    def shutdown(self) -> None:
        """Shut down the API manager."""
        # Nothing to do here for now
        self.logger.info("APIManager shut down")
