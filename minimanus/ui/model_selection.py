#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Selection Interface for miniManus

This module implements the Model Selection Interface component, which provides
methods for discovering, managing, and selecting LLM models from various providers.
"""

import os
import sys
import logging
import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum, auto
from pathlib import Path

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
    # UIManager might not be strictly needed here unless interacting directly with UI elements
    # from ..ui.ui_manager import UIManager
    from ..api.api_manager import APIManager, APIProvider
except ImportError as e:
    logging.getLogger("miniManus.ModelSelectionInterface").critical(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(f"ImportError in model_selection.py: {e}. Ensure all components exist.")


logger = logging.getLogger("miniManus.ModelSelectionInterface")

class ModelCategory(Enum):
    """Model categories."""
    GENERAL = auto()
    CODE = auto()
    CHAT = auto()
    EMBEDDING = auto()
    IMAGE = auto()
    AUDIO = auto()
    SPECIALIZED = auto()
    CUSTOM = auto()

class ModelInfo:
    """Represents information about a model."""

    def __init__(self, id: str, name: str, provider: APIProvider,
                category: ModelCategory = ModelCategory.GENERAL,
                description: Optional[str] = None, context_length: Optional[int] = None,
                capabilities: Optional[List[str]] = None, tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize model information.

        Args:
            id: Unique Model ID (usually provider-specific)
            name: Display name
            provider: API provider (APIProvider enum)
            category: Model category (ModelCategory enum)
            description: Model description
            context_length: Maximum context length (optional)
            capabilities: List of capabilities (optional)
            tags: List of tags (optional)
            metadata: Additional provider-specific metadata (optional)
        """
        self.id = id
        self.name = name
        self.provider = provider
        self.category = category
        self.description = description or ""
        self.context_length = context_length
        self.capabilities = capabilities or []
        self.tags = tags or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model info to dictionary suitable for serialization or API responses.

        Returns:
            Dictionary representation of the model info
        """
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider.name, # Store enum name as string
            "category": self.category.name, # Store enum name as string
            "description": self.description,
            "context_length": self.context_length,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['ModelInfo']:
        """
        Create model info from dictionary. Handles potential errors.

        Args:
            data: Dictionary representation of the model info

        Returns:
            ModelInfo instance or None if essential data is missing/invalid
        """
        model_id = data.get("id")
        name = data.get("name")
        provider_str = data.get("provider")

        if not model_id or not name or not provider_str:
             logger.warning(f"Skipping model creation from dict due to missing essential fields (id, name, provider): {data}")
             return None

        try:
            provider = APIProvider[provider_str]
        except KeyError:
            logger.warning(f"Invalid provider name '{provider_str}' found for model {model_id}. Setting provider to CUSTOM.")
            provider = APIProvider.CUSTOM # Default to CUSTOM if provider name is unknown

        category_str = data.get("category", "GENERAL")
        try:
            category = ModelCategory[category_str]
        except KeyError:
            logger.warning(f"Invalid category name '{category_str}' for model {model_id}. Defaulting to GENERAL.")
            category = ModelCategory.GENERAL

        return cls(
            id=model_id,
            name=name,
            provider=provider,
            category=category,
            description=data.get("description", ""),
            context_length=data.get("context_length"), # Allow None
            capabilities=data.get("capabilities", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )

class ModelParameter:
    """Represents a configurable parameter for a model (Not fully utilized yet)."""
    # This class remains largely the same as it's mostly a data holder.
    # Full implementation would involve linking these to specific models/providers.

    def __init__(self, id: str, name: str, description: str,
                type: str, default_value: Any, min_value: Optional[Any] = None,
                max_value: Optional[Any] = None, options: Optional[List[Dict[str, Any]]] = None):
        self.id = id
        self.name = name
        self.description = description
        self.type = type # e.g., "number", "string", "boolean", "select"
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.options = options or [] # For "select" type

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "default_value": self.default_value
        }
        if self.min_value is not None: result["min_value"] = self.min_value
        if self.max_value is not None: result["max_value"] = self.max_value
        if self.options: result["options"] = self.options
        return result

class ModelSelectionInterface:
    """
    Manages discovery, selection, and configuration of LLM models.

    Handles:
    - Discovering models from configured API providers.
    - Storing model information.
    - Managing favorite and recent models.
    - Providing methods to query and filter models.
    """

    _instance = None  # Singleton instance

    @classmethod
    def get_instance(cls) -> 'ModelSelectionInterface':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = ModelSelectionInterface()
        return cls._instance

    def __init__(self):
        """Initialize the ModelSelectionInterface."""
        if ModelSelectionInterface._instance is not None:
            raise RuntimeError("ModelSelectionInterface is a singleton. Use get_instance() instead.")

        self.logger = logger
        self.event_bus = EventBus.get_instance()
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        self.api_manager = APIManager.get_instance()

        # Model registry: Store ModelInfo objects, keyed by model ID
        self.models: Dict[str, ModelInfo] = {}
        self._models_lock = asyncio.Lock() # Use asyncio lock for async operations

        # Model parameters (future use)
        # self.parameters: Dict[str, List[ModelParameter]] = {}

        # User preferences for models
        self.favorite_models: List[str] = []
        self.recent_models: List[str] = []
        self.max_recent_models = self.config_manager.get_config("models.max_recents", 10)

        self.logger.info("ModelSelectionInterface initialized")

    def startup(self) -> None:
        """Start the model selection interface."""
        self._load_preferences() # Load favorites and recents from config

        # Subscribe to relevant events after startup
        self.event_bus.subscribe("models.discovered", self._handle_models_discovered)
        self.event_bus.subscribe("models.selected", self._handle_model_selected)

        self.logger.info("ModelSelectionInterface started")

    def shutdown(self) -> None:
        """Stop the model selection interface."""
        self._save_preferences() # Save favorites and recents
        self.logger.info("ModelSelectionInterface stopped")

    async def register_model(self, model: ModelInfo) -> None:
        """
        Register or update a model in the registry.

        Args:
            model: ModelInfo object to register.
        """
        async with self._models_lock:
            self.models[model.id] = model
        self.logger.debug(f"Registered/Updated model: {model.provider.name}/{model.id}")

    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a model by its ID."""
        async with self._models_lock:
            return self.models.get(model_id)

    async def get_all_models(self) -> List[ModelInfo]:
        """Get a list of all registered models."""
        async with self._models_lock:
            # Return a copy to avoid external modification
            return list(self.models.values())

    async def get_models_by_provider(self, provider: APIProvider) -> List[ModelInfo]:
        """Get models filtered by provider."""
        async with self._models_lock:
            return [model for model in self.models.values() if model.provider == provider]

    async def get_models_by_category(self, category: ModelCategory) -> List[ModelInfo]:
        """Get models filtered by category."""
        async with self._models_lock:
            return [model for model in self.models.values() if model.category == category]

    async def search_models(self, query: str) -> List[ModelInfo]:
        """Search models by ID, name, description, tags, provider, or category."""
        query_lower = query.lower()
        results = []
        async with self._models_lock:
            for model in self.models.values():
                if (query_lower in model.id.lower() or
                    query_lower in model.name.lower() or
                    query_lower in model.description.lower() or
                    query_lower in model.provider.name.lower() or
                    query_lower in model.category.name.lower() or
                    any(query_lower in tag.lower() for tag in model.tags)):
                    results.append(model)
        return results

    # --- Favorite and Recent Model Management ---

    async def add_favorite_model(self, model_id: str) -> bool:
        """Add a model ID to favorites."""
        async with self._models_lock:
             if model_id not in self.models:
                 self.logger.warning(f"Cannot favorite non-existent model: {model_id}")
                 return False
             if model_id in self.favorite_models:
                 return True # Already favorited

             self.favorite_models.append(model_id)
             self._save_preferences() # Save immediately
             self.event_bus.publish_event("models.favorite.added", {"model_id": model_id})
             self.logger.info(f"Added model to favorites: {model_id}")
             return True

    async def remove_favorite_model(self, model_id: str) -> bool:
        """Remove a model ID from favorites."""
        if model_id not in self.favorite_models:
            return False
        self.favorite_models.remove(model_id)
        self._save_preferences() # Save immediately
        self.event_bus.publish_event("models.favorite.removed", {"model_id": model_id})
        self.logger.info(f"Removed model from favorites: {model_id}")
        return True

    async def get_favorite_models(self) -> List[ModelInfo]:
        """Get ModelInfo objects for favorited models."""
        favs = []
        async with self._models_lock:
             for model_id in self.favorite_models:
                 if model_id in self.models:
                     favs.append(self.models[model_id])
        return favs

    async def add_recent_model(self, model_id: str) -> bool:
        """Add a model ID to the top of the recent models list."""
        async with self._models_lock:
             if model_id not in self.models:
                 self.logger.warning(f"Cannot add non-existent model to recents: {model_id}")
                 return False

             # Remove if already exists to move it to the front
             if model_id in self.recent_models:
                 self.recent_models.remove(model_id)

             self.recent_models.insert(0, model_id)

             # Trim list if it exceeds max size
             self.recent_models = self.recent_models[:self.max_recent_models]

             self._save_preferences() # Save immediately
             self.logger.debug(f"Added model to recents: {model_id}")
             return True

    async def get_recent_models(self) -> List[ModelInfo]:
        """Get ModelInfo objects for recently used models."""
        recents = []
        async with self._models_lock:
             for model_id in self.recent_models:
                 if model_id in self.models:
                     recents.append(self.models[model_id])
        return recents

    def _save_preferences(self) -> None:
        """Save favorite and recent models lists to config."""
        try:
            # Use separate keys for favorites and recents
            self.config_manager.set_config("models.favorites", self.favorite_models)
            self.config_manager.set_config("models.recents", self.recent_models)
            # ConfigManager's set_config should handle saving the file
            self.logger.debug("Saved model preferences (favorites/recents).")
        except Exception as e:
            self.logger.error(f"Error saving model preferences: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.WARNING, {"action": "save_model_preferences"})

    def _load_preferences(self) -> None:
        """Load favorite and recent models lists from config."""
        try:
            favorites = self.config_manager.get_config("models.favorites", [])
            recents = self.config_manager.get_config("models.recents", [])

            if isinstance(favorites, list):
                 self.favorite_models = [fav for fav in favorites if isinstance(fav, str)] # Basic type check
            else:
                 self.logger.warning("Invalid format for 'models.favorites' in config, resetting.")
                 self.favorite_models = []

            if isinstance(recents, list):
                 self.recent_models = [rec for rec in recents if isinstance(rec, str)][:self.max_recent_models] # Check type and limit size
            else:
                 self.logger.warning("Invalid format for 'models.recents' in config, resetting.")
                 self.recent_models = []

            self.logger.debug("Loaded model preferences (favorites/recents).")
        except Exception as e:
            self.logger.error(f"Error loading model preferences: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.WARNING, {"action": "load_model_preferences"})
            self.favorite_models = []
            self.recent_models = []


    async def select_model(self, model_id: str) -> bool:
        """
        Select a model, adding it to recents and publishing an event.

        Args:
            model_id: ID of the model to select.

        Returns:
            True if selection was successful (model exists), False otherwise.
        """
        model_exists = False
        async with self._models_lock:
             model_exists = model_id in self.models

        if not model_exists:
            self.logger.warning(f"Attempted to select non-existent model: {model_id}")
            return False

        await self.add_recent_model(model_id) # Add to recents
        self.event_bus.publish_event("models.selected", {"model_id": model_id})
        self.logger.info(f"Selected model: {model_id}")
        return True

    async def discover_models(self) -> None:
        """Discover available models from all enabled and available providers."""
        self.logger.info("Starting model discovery...")
        discovered_count = 0
        all_provider_models = {} # Store models per provider temporarily

        # Get available providers (checks API key presence, basic connectivity)
        # Note: APIManager.get_available_providers() needs refinement to be async
        # For now, we assume it returns a list synchronously or adapt it.
        # Let's assume api_manager needs an async version or we call check_availability async
        
        tasks = []
        for provider in APIProvider:
            if provider == APIProvider.CUSTOM: continue # Skip custom for now

            # Check if provider is generally enabled in config (optional)
            if not self.config_manager.get_config(f"api.providers.{provider.name.lower()}.enabled", True):
                 self.logger.info(f"Skipping model discovery for disabled provider: {provider.name}")
                 continue

            adapter = self.api_manager.get_adapter(provider)
            if not adapter or not hasattr(adapter, 'get_available_models'):
                self.logger.warning(f"No adapter or get_available_models method for provider: {provider.name}")
                continue

            # Create a task to fetch models for this provider
            tasks.append(self._discover_provider_models(provider, adapter))

        # Run discovery tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and update registry
        async with self._models_lock:
            self.models.clear() # Clear existing models before adding newly discovered ones
            for provider, result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error discovering models for {provider.name}: {result}")
                    self.error_handler.handle_error(result, ErrorCategory.API, ErrorSeverity.WARNING, {"provider": provider.name, "action": "discover_models"})
                elif isinstance(result, list):
                    self.logger.info(f"Discovered {len(result)} models from {provider.name}")
                    for model_data in result:
                        model = self._create_model_info_from_data(provider, model_data)
                        if model:
                            self.models[model.id] = model
                            discovered_count += 1
                else:
                    self.logger.warning(f"Unexpected result type for {provider.name} discovery: {type(result)}")

        self.logger.info(f"Model discovery complete. Total models registered: {discovered_count}")
        # Publish event after all discoveries are processed
        self.event_bus.publish_event("models.discovered", {"count": discovered_count})

    async def _discover_provider_models(self, provider: APIProvider, adapter: Any) -> Tuple[APIProvider, Any]:
        """Helper coroutine to discover models for a single provider."""
        try:
            # Optional: Check availability again if needed, though get_available_models might do it
            # if hasattr(adapter, 'check_availability') and not await adapter.check_availability():
            #     self.logger.warning(f"Provider {provider.name} is not available.")
            #     return provider, [] # Return empty list if unavailable

            models_data = await adapter.get_available_models()
            return provider, models_data
        except Exception as e:
             # Let gather handle the exception logging/reporting
             # Log it here too for immediate feedback during discovery
             self.logger.error(f"Exception during model discovery for {provider.name}: {e}", exc_info=False) # Avoid excessive traceback logs here
             return provider, e # Return exception to be handled by gather

    def _create_model_info_from_data(self, provider: APIProvider, model_data: Dict[str, Any]) -> Optional[ModelInfo]:
        """Creates a ModelInfo object from provider data, attempting categorization."""
        model_id = model_data.get("id")
        if not model_id:
             self.logger.warning(f"Skipping model from {provider.name} due to missing 'id': {model_data}")
             return None

        name = model_data.get("name", model_id) # Default name to ID if missing
        description = model_data.get("description", model_data.get("summary", "")) # Use summary as fallback desc
        context_length = model_data.get("context_length", model_data.get("max_context_size")) # Look for variations

        # Attempt categorization based on ID/name
        category = self._guess_category(model_id, name)

        # Add tags based on provider and category
        tags = [provider.name.lower(), category.name.lower()]
        if 'instruct' in model_id.lower() or 'chat' in model_id.lower():
             tags.append('instruct')
        if 'code' in model_id.lower():
             tags.append('code')
        # Add more tag logic as needed

        model = ModelInfo(
            id=model_id,
            name=name,
            provider=provider,
            category=category,
            description=description,
            context_length=context_length,
            tags=list(set(tags)), # Ensure unique tags
            metadata=model_data # Store original data
        )
        return model

    def _guess_category(self, model_id: str, name: str) -> ModelCategory:
        """Attempt to guess model category based on name/id."""
        id_lower = model_id.lower()
        name_lower = name.lower()

        if "embed" in id_lower: return ModelCategory.EMBEDDING
        if "image" in id_lower or "dall-e" in id_lower or "sdxl" in id_lower: return ModelCategory.IMAGE
        if "audio" in id_lower or "whisper" in id_lower: return ModelCategory.AUDIO
        if "code" in id_lower or "coder" in name_lower: return ModelCategory.CODE
        if "chat" in id_lower or "instruct" in id_lower or "claude" in id_lower or "gemini" in id_lower or "gpt" in id_lower: return ModelCategory.CHAT
        # Add more rules based on observed model names

        return ModelCategory.GENERAL # Default


    # --- Event Handlers ---

    def _handle_models_discovered(self, event: Event) -> None:
        """Handle the models.discovered event."""
        count = event.data.get("count", 0)
        self.logger.debug(f"Event received: Models discovered ({count}).")
        # Potentially trigger UI updates here if needed

    def _handle_model_selected(self, event: Event) -> None:
        """Handle the models.selected event."""
        model_id = event.data.get("model_id")
        self.logger.debug(f"Event received: Model selected ({model_id}).")
        # Potentially update default model in config or UI state

# Example usage (if run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for testing

    async def test_model_interface():
        # Initialize required components for standalone testing
        event_bus = EventBus.get_instance()
        event_bus.startup()
        error_handler = ErrorHandler.get_instance()
        config_manager = ConfigurationManager.get_instance()
        # Ensure config dir is set for testing if needed
        # config_manager.config_dir = Path('./test_config')
        # config_manager.secrets_file = config_manager.config_dir / 'secrets.json'
        # config_manager.config_dir.mkdir(exist_ok=True)
        # config_manager._load_config()
        # config_manager._load_secrets()

        api_manager = APIManager.get_instance()
        api_manager.startup() # This initializes adapters

        # Initialize ModelSelectionInterface
        model_interface = ModelSelectionInterface.get_instance()
        model_interface.startup()

        print("--- Initial State ---")
        print(f"Favorites: {model_interface.favorite_models}")
        print(f"Recents: {model_interface.recent_models}")

        print("\n--- Discovering Models ---")
        await model_interface.discover_models()
        all_models = await model_interface.get_all_models()
        print(f"Total models discovered: {len(all_models)}")
        if all_models:
            print("Example models:")
            for m in all_models[:3]:
                print(f"  - {m.provider.name}/{m.id} ({m.category.name})")

            test_model_id = all_models[0].id

            print(f"\n--- Testing Favorites (Model: {test_model_id}) ---")
            await model_interface.add_favorite_model(test_model_id)
            favs = await model_interface.get_favorite_models()
            print(f"Favorites: {[m.id for m in favs]}")
            await model_interface.remove_favorite_model(test_model_id)
            favs = await model_interface.get_favorite_models()
            print(f"Favorites after removal: {[m.id for m in favs]}")

            print(f"\n--- Testing Recents & Selection (Model: {test_model_id}) ---")
            await model_interface.select_model(test_model_id)
            recents = await model_interface.get_recent_models()
            print(f"Recents: {[m.id for m in recents]}")

            # Select another if available
            if len(all_models) > 1:
                 second_model_id = all_models[1].id
                 await model_interface.select_model(second_model_id)
                 recents = await model_interface.get_recent_models()
                 print(f"Recents after selecting another: {[m.id for m in recents]}")


            print("\n--- Testing Search ('gpt') ---")
            search_results = await model_interface.search_models("gpt")
            print(f"Search results for 'gpt': {[m.id for m in search_results]}")

        # Shutdown components
        model_interface.shutdown()
        api_manager.shutdown()
        event_bus.shutdown()

    try:
        asyncio.run(test_model_interface())
    except Exception as e:
         print(f"\nStandalone test failed: {e}")
