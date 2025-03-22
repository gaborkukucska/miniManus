#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Selection Interface for miniManus

This module implements the Model Selection Interface component, which provides a mobile-optimized
interface for selecting and configuring LLM models.
"""

import os
import sys
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum, auto
from pathlib import Path

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
    from ..ui.ui_manager import UIManager
    from ..api.api_manager import APIManager, APIProvider
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager
    from ui.ui_manager import UIManager
    from api.api_manager import APIManager, APIProvider

logger = logging.getLogger("miniManus.ModelSelectionInterface")

class ModelCategory(Enum):
    """Model categories."""
    GENERAL = auto()
    CODE = auto()
    CHAT = auto()
    EMBEDDING = auto()
    SPECIALIZED = auto()
    CUSTOM = auto()

class ModelInfo:
    """Represents information about a model."""
    
    def __init__(self, id: str, name: str, provider: APIProvider, 
                category: ModelCategory = ModelCategory.GENERAL,
                description: Optional[str] = None, context_length: int = 4096,
                capabilities: Optional[List[str]] = None, tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize model information.
        
        Args:
            id: Model ID
            name: Display name
            provider: API provider
            category: Model category
            description: Model description
            context_length: Maximum context length
            capabilities: List of capabilities
            tags: List of tags
            metadata: Additional metadata
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
        Convert model info to dictionary.
        
        Returns:
            Dictionary representation of the model info
        """
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider.name,
            "category": self.category.name,
            "description": self.description,
            "context_length": self.context_length,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """
        Create model info from dictionary.
        
        Args:
            data: Dictionary representation of the model info
            
        Returns:
            ModelInfo instance
        """
        try:
            provider = APIProvider[data["provider"]]
        except (KeyError, ValueError):
            provider = APIProvider.CUSTOM
        
        try:
            category = ModelCategory[data["category"]]
        except (KeyError, ValueError):
            category = ModelCategory.GENERAL
        
        return cls(
            id=data["id"],
            name=data["name"],
            provider=provider,
            category=category,
            description=data.get("description", ""),
            context_length=data.get("context_length", 4096),
            capabilities=data.get("capabilities", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )

class ModelParameter:
    """Represents a configurable parameter for a model."""
    
    def __init__(self, id: str, name: str, description: str, 
                type: str, default_value: Any, min_value: Optional[Any] = None,
                max_value: Optional[Any] = None, options: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize model parameter.
        
        Args:
            id: Parameter ID
            name: Display name
            description: Parameter description
            type: Parameter type (string, number, boolean, select)
            default_value: Default value
            min_value: Minimum value (for number type)
            max_value: Maximum value (for number type)
            options: Options (for select type)
        """
        self.id = id
        self.name = name
        self.description = description
        self.type = type
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.options = options or []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter to dictionary.
        
        Returns:
            Dictionary representation of the parameter
        """
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "default_value": self.default_value
        }
        
        if self.min_value is not None:
            result["min_value"] = self.min_value
        
        if self.max_value is not None:
            result["max_value"] = self.max_value
        
        if self.options:
            result["options"] = self.options
        
        return result

class ModelSelectionInterface:
    """
    ModelSelectionInterface provides a mobile-optimized model selection interface for miniManus.
    
    It handles:
    - Model discovery and listing
    - Model filtering and searching
    - Model parameter configuration
    - Model selection and persistence
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'ModelSelectionInterface':
        """Get or create the singleton instance of ModelSelectionInterface."""
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
        self.ui_manager = UIManager.get_instance()
        self.api_manager = APIManager.get_instance()
        
        # Model registry
        self.models: Dict[str, ModelInfo] = {}
        
        # Model parameters
        self.parameters: Dict[str, List[ModelParameter]] = {}
        
        # Favorite models
        self.favorite_models: List[str] = []
        
        # Recent models
        self.recent_models: List[str] = []
        self.max_recent_models = 10
        
        # Register event handlers
        self.event_bus.subscribe("models.discovered", self._handle_models_discovered)
        self.event_bus.subscribe("models.selected", self._handle_model_selected)
        
        self.logger.info("ModelSelectionInterface initialized")
    
    def register_model(self, model: ModelInfo) -> None:
        """
        Register a model.
        
        Args:
            model: Model to register
        """
        self.models[model.id] = model
        self.logger.debug(f"Registered model: {model.id}")
    
    def register_model_parameter(self, model_id: str, parameter: ModelParameter) -> None:
        """
        Register a model parameter.
        
        Args:
            model_id: Model ID
            parameter: Parameter to register
        """
        if model_id not in self.parameters:
            self.parameters[model_id] = []
        
        self.parameters[model_id].append(parameter)
        self.logger.debug(f"Registered parameter {parameter.id} for model {model_id}")
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model info or None if not found
        """
        return self.models.get(model_id)
    
    def get_all_models(self) -> List[ModelInfo]:
        """
        Get all models.
        
        Returns:
            List of all models
        """
        return list(self.models.values())
    
    def get_models_by_provider(self, provider: APIProvider) -> List[ModelInfo]:
        """
        Get models by provider.
        
        Args:
            provider: API provider
            
        Returns:
            List of models from the provider
        """
        return [
            model for model in self.models.values()
            if model.provider == provider
        ]
    
    def get_models_by_category(self, category: ModelCategory) -> List[ModelInfo]:
        """
        Get models by category.
        
        Args:
            category: Model category
            
        Returns:
            List of models in the category
        """
        return [
            model for model in self.models.values()
            if model.category == category
        ]
    
    def get_model_parameters(self, model_id: str) -> List[ModelParameter]:
        """
        Get parameters for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            List of parameters for the model
        """
        return self.parameters.get(model_id, [])
    
    def add_favorite_model(self, model_id: str) -> bool:
        """
        Add a model to favorites.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if added, False if already in favorites or not found
        """
        if model_id not in self.models:
            return False
        
        if model_id in self.favorite_models:
            return False
        
        self.favorite_models.append(model_id)
        self._save_favorites()
        
        # Publish event
        self.event_bus.publish_event("models.favorite_added", {
            "model_id": model_id
        })
        
        self.logger.debug(f"Added model {model_id} to favorites")
        return True
    
    def remove_favorite_model(self, model_id: str) -> bool:
        """
        Remove a model from favorites.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if removed, False if not in favorites
        """
        if model_id not in self.favorite_models:
            return False
        
        self.favorite_models.remove(model_id)
        self._save_favorites()
        
        # Publish event
        self.event_bus.publish_event("models.favorite_removed", {
            "model_id": model_id
        })
        
        self.logger.debug(f"Removed model {model_id} from favorites")
        return True
    
    def get_favorite_models(self) -> List[ModelInfo]:
        """
        Get favorite models.
        
        Returns:
            List of favorite models
        """
        return [
            self.models[model_id] for model_id in self.favorite_models
            if model_id in self.models
        ]
    
    def add_recent_model(self, model_id: str) -> bool:
        """
        Add a model to recent models.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if added, False if not found
        """
        if model_id not in self.models:
            return False
        
        # Remove if already in list
        if model_id in self.recent_models:
            self.recent_models.remove(model_id)
        
        # Add to front of list
        self.recent_models.insert(0, model_id)
        
        # Trim list if needed
        if len(self.recent_models) > self.max_recent_models:
            self.recent_models = self.recent_models[:self.max_recent_models]
        
        self._save_recents()
        
        self.logger.debug(f"Added model {model_id} to recent models")
        return True
    
    def get_recent_models(self) -> List[ModelInfo]:
        """
        Get recent models.
        
        Returns:
            List of recent models
        """
        return [
            self.models[model_id] for model_id in self.recent_models
            if model_id in self.models
        ]
    
    def search_models(self, query: str) -> List[ModelInfo]:
        """
        Search for models.
        
        Args:
            query: Search query
            
        Returns:
            List of matching models
        """
        query = query.lower()
        return [
            model for model in self.models.values()
            if (
                query in model.id.lower() or
                query in model.name.lower() or
                query in model.description.lower() or
                any(query in tag.lower() for tag in model.tags) or
                query in model.provider.name.lower() or
                query in model.category.name.lower()
            )
        ]
    
    def select_model(self, model_id: str) -> bool:
        """
        Select a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if selected, False if not found
        """
        if model_id not in self.models:
            return False
        
        # Add to recent models
        self.add_recent_model(model_id)
        
        # Publish event
        self.event_bus.publish_event("models.selected", {
            "model_id": model_id
        })
        
        self.logger.info(f"Selected model: {model_id}")
        return True
    
    async def discover_models(self) -> None:
        """Discover available models from all providers."""
        self.logger.info("Discovering models...")
        
        # Get available providers
        providers = self.api_manager.get_available_providers()
        
        for provider in providers:
            try:
                # Get adapter
                adapter = self.api_manager.get_adapter(provider)
                if not adapter:
                    continue
                
                # Get models
                models = await adapter.get_available_models()
                
                # Register models
                for model_data in models:
                    model_id = model_data.get("id")
                    if not model_id:
                        continue
                    
                    # Create model info
                    model = ModelInfo(
                        id=model_id,
                        name=model_data.get("name", model_id),
                        provider=provider,
                        description=model_data.get("description", ""),
                        context_length=model_data.get("context_length", 4096),
                        metadata=model_data
                    )
                    
                    # Register model
                    self.register_model(model)
                
                self.logger.info(f"Discovered {len(models)} models from {provider.name}")
            
            except Exception as e:
                self.error_handler.handle_error(
                    e, ErrorCategory.API, ErrorSeverity.WARNING,
                    {"provider": provider.name, "action": "discover_models"}
                )
        
        # Publish event
        self.event_bus.publish_event("models.discovered", {
            "count": len(self.models)
        })
    
    def _register_default_parameters(self) -> None:
        """Register default parameters for models."""
        # Common parameters for all models
        common_parameters = [
            ModelParameter(
                "temperature",
                "Temperature",
                "Controls randomness: lower values are more deterministic, higher values are more random",
                "number",
                0.7,
                0.0,
                2.0
            ),
            ModelParameter(
                "max_tokens",
                "Max Tokens",
                "Maximum number of tokens to generate",
                "number",
                1024,
                1,
                32000
            ),
            ModelParameter(
                "top_p",
                "Top P",
                "Controls diversity via nucleus sampling",
                "number",
                1.0,
                0.0,
                1.0
            ),
            ModelParameter(
                "frequency_penalty",
                "Frequency Penalty",
                "Penalizes repeated tokens",
                "number",
                0.0,
                -2.0,
                2.0
            ),
            ModelParameter(
                "presence_penalty",
                "Presence Penalty",
                "Penalizes repeated topics",
                "number",
                0.0,
                -2.0,
                2.0
            )
        ]
        
        # Register common parameters for all models
        for model_id in self.models:
            for param in common_parameters:
                self.register_model_parameter(model_id, param)
    
    def _save_favorites(self) -> None:
        """Save favorite models to storage."""
        try:
            self.config_manager.set_config("models.favorites", self.favorite_models)
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.STORAGE, ErrorSeverity.WARNING,
                {"action": "save_favorite_models"}
            )
    
    def _load_favorites(self) -> None:
        """Load favorite models from storage."""
        try:
            favorites = self.config_manager.get_config("models.favorites", [])
            if isinstance(favorites, list):
                self.favorite_models = favorites
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.STORAGE, ErrorSeverity.WARNING,
                {"action": "load_favorite_models"}
            )
    
    def _save_recents(self) -> None:
        """Save recent models to storage."""
        try:
            self.config_manager.set_config("models.recents", self.recent_models)
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.STORAGE, ErrorSeverity.WARNING,
                {"action": "save_recent_models"}
            )
    
    def _load_recents(self) -> None:
        """Load recent models from storage."""
        try:
            recents = self.config_manager.get_config("models.recents", [])
            if isinstance(recents, list):
                self.recent_models = recents
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.STORAGE, ErrorSeverity.WARNING,
                {"action": "load_recent_models"}
            )
    
    def _handle_models_discovered(self, event: Dict[str, Any]) -> None:
        """
        Handle models discovered event.
        
        Args:
            event: Event data
        """
        count = event.get("count", 0)
        self.logger.debug(f"Models discovered: {count}")
        
        # Register default parameters
        self._register_default_parameters()
    
    def _handle_model_selected(self, event: Dict[str, Any]) -> None:
        """
        Handle model selected event.
        
        Args:
            event: Event data
        """
        model_id = event.get("model_id")
        if model_id:
            self.logger.debug(f"Model selected: {model_id}")
    
    def startup(self) -> None:
        """Start the model selection interface."""
        # Load favorites and recents
        self._load_favorites()
        self._load_recents()
        
        self.logger.info("ModelSelectionInterface started")
    
    def shutdown(self) -> None:
        """Stop the model selection interface."""
        # Save favorites and recents
        self._save_favorites()
        self._save_recents()
        
        self.logger.info("ModelSelectionInterface stopped")

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    error_handler = ErrorHandler.get_instance()
    
    config_manager = ConfigurationManager.get_instance()
    
    ui_manager = UIManager.get_instance()
    ui_manager.startup()
    
    api_manager = APIManager.get_instance()
    api_manager.startup()
    
    # Initialize ModelSelectionInterface
    model_interface = ModelSelectionInterface.get_instance()
    model_interface.startup()
    
    # Example usage
    async def test_model_interface():
        # Register some example models
        model1 = ModelInfo(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider=APIProvider.OPENROUTER,
            category=ModelCategory.CHAT,
            description="A good all-around model for chat",
            context_length=4096
        )
        model_interface.register_model(model1)
        
        model2 = ModelInfo(
            id="llama2",
            name="Llama 2",
            provider=APIProvider.OLLAMA,
            category=ModelCategory.GENERAL,
            description="Open source large language model",
            context_length=4096
        )
        model_interface.register_model(model2)
        
        # Add to favorites
        model_interface.add_favorite_model("gpt-3.5-turbo")
        
        # Select a model
        model_interface.select_model("llama2")
        
        # Search models
        results = model_interface.search_models("gpt")
        print(f"Search results: {[model.name for model in results]}")
        
        # Get favorites
        favorites = model_interface.get_favorite_models()
        print(f"Favorites: {[model.name for model in favorites]}")
        
        # Get recents
        recents = model_interface.get_recent_models()
        print(f"Recents: {[model.name for model in recents]}")
    
    # Run test
    asyncio.run(test_model_interface())
    
    # Shutdown
    model_interface.shutdown()
    api_manager.shutdown()
    ui_manager.shutdown()
    event_bus.shutdown()
