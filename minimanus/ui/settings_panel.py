#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Settings Panel for miniManus

This module implements the Settings Panel component, which provides a mobile-optimized
interface for configuring miniManus settings.
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
    from ..ui.ui_manager import UIManager, UITheme
    from ..api.api_manager import APIProvider, APIRequestType
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager
    from ui.ui_manager import UIManager, UITheme
    from api.api_manager import APIProvider, APIRequestType

logger = logging.getLogger("miniManus.SettingsPanel")

class SettingType(Enum):
    """Types of settings."""
    BOOLEAN = auto()
    STRING = auto()
    NUMBER = auto()
    SELECT = auto()
    COLOR = auto()
    SECTION = auto()
    PASSWORD = auto()  # Added for API keys

class Setting:
    """Represents a setting."""
    
    def __init__(self, key: str, name: str, description: str, type: SettingType,
                default_value: Any, options: Optional[List[Dict[str, Any]]] = None,
                section: Optional[str] = None, order: int = 0):
        """
        Initialize a setting.
        
        Args:
            key: Setting key
            name: Display name
            description: Setting description
            type: Setting type
            default_value: Default value
            options: Options for SELECT type
            section: Section name
            order: Display order
        """
        self.key = key
        self.name = name
        self.description = description
        self.type = type
        self.default_value = default_value
        self.options = options or []
        self.section = section
        self.order = order
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert setting to dictionary.
        
        Returns:
            Dictionary representation of the setting
        """
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "type": self.type.name.lower(),
            "default_value": self.default_value,
            "options": self.options,
            "section": self.section,
            "order": self.order
        }

class SettingsPanel:
    """
    SettingsPanel provides a mobile-optimized settings interface for miniManus.
    
    It handles:
    - Settings definition and organization
    - Settings persistence
    - Mobile-optimized UI rendering
    - Settings validation
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'SettingsPanel':
        """Get or create the singleton instance of SettingsPanel."""
        if cls._instance is None:
            cls._instance = SettingsPanel()
        return cls._instance
    
    def __init__(self):
        """Initialize the SettingsPanel."""
        if SettingsPanel._instance is not None:
            raise RuntimeError("SettingsPanel is a singleton. Use get_instance() instead.")
        
        self.logger = logger
        self.event_bus = EventBus.get_instance()
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        self.ui_manager = UIManager.get_instance()
        
        # Settings definitions
        self.settings: Dict[str, Setting] = {}
        self.sections: Dict[str, Dict[str, Any]] = {}
        
        # Register event handlers
        self.event_bus.subscribe("settings.changed", self._handle_setting_changed)
        
        self.logger.info("SettingsPanel initialized")
    
    def register_setting(self, setting: Setting) -> None:
        """
        Register a setting.
        
        Args:
            setting: Setting to register
        """
        self.settings[setting.key] = setting
        self.logger.debug(f"Registered setting: {setting.key}")
    
    def register_section(self, id: str, name: str, description: str, icon: Optional[str] = None,
                       order: int = 0) -> None:
        """
        Register a settings section.
        
        Args:
            id: Section ID
            name: Section name
            description: Section description
            icon: Section icon
            order: Display order
        """
        self.sections[id] = {
            "id": id,
            "name": name,
            "description": description,
            "icon": icon,
            "order": order
        }
        self.logger.debug(f"Registered settings section: {id}")
    
    def get_setting(self, key: str) -> Optional[Setting]:
        """
        Get a setting.
        
        Args:
            key: Setting key
            
        Returns:
            Setting or None if not found
        """
        return self.settings.get(key)
    
    def get_section(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get a settings section.
        
        Args:
            id: Section ID
            
        Returns:
            Section or None if not found
        """
        return self.sections.get(id)
    
    def get_all_settings(self) -> List[Setting]:
        """
        Get all settings.
        
        Returns:
            List of all settings
        """
        return list(self.settings.values())
    
    def get_all_sections(self) -> List[Dict[str, Any]]:
        """
        Get all settings sections.
        
        Returns:
            List of all sections
        """
        return list(self.sections.values())
    
    def get_section_settings(self, section_id: str) -> List[Setting]:
        """
        Get settings for a section.
        
        Args:
            section_id: Section ID
            
        Returns:
            List of settings in the section
        """
        return [
            setting for setting in self.settings.values()
            if setting.section == section_id
        ]
    
    def get_setting_value(self, key: str) -> Any:
        """
        Get the current value of a setting.
        
        Args:
            key: Setting key
            
        Returns:
            Setting value
        """
        setting = self.get_setting(key)
        if setting is None:
            return None
        
        return self.config_manager.get_config(key, setting.default_value)
    
    def set_setting_value(self, key: str, value: Any) -> bool:
        """
        Set the value of a setting.
        
        Args:
            key: Setting key
            value: New value
            
        Returns:
            True if set, False if not found or invalid
        """
        setting = self.get_setting(key)
        if setting is None:
            return False
        
        # Validate value
        if not self._validate_setting_value(setting, value):
            return False
        
        # Set value
        self.config_manager.set_config(key, value)
        
        # Publish event
        self.event_bus.publish_event("settings.changed", {
            "key": key,
            "value": value
        })
        
        self.logger.debug(f"Set setting {key} to {value}")
        return True
    
    def reset_setting(self, key: str) -> bool:
        """
        Reset a setting to its default value.
        
        Args:
            key: Setting key
            
        Returns:
            True if reset, False if not found
        """
        setting = self.get_setting(key)
        if setting is None:
            return False
        
        # Set to default value
        self.config_manager.set_config(key, setting.default_value)
        
        # Publish event
        self.event_bus.publish_event("settings.changed", {
            "key": key,
            "value": setting.default_value
        })
        
        self.logger.debug(f"Reset setting {key} to default value")
        return True
    
    def _validate_setting_value(self, setting: Setting, value: Any) -> bool:
        """
        Validate a setting value.
        
        Args:
            setting: Setting to validate
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if setting.type == SettingType.BOOLEAN:
            return isinstance(value, bool)
        
        elif setting.type == SettingType.STRING or setting.type == SettingType.PASSWORD:
            return isinstance(value, str)
        
        elif setting.type == SettingType.NUMBER:
            return isinstance(value, (int, float))
        
        elif setting.type == SettingType.SELECT:
            if not setting.options:
                return False
            
            valid_values = [option["value"] for option in setting.options]
            return value in valid_values
        
        elif setting.type == SettingType.COLOR:
            if not isinstance(value, str):
                return False
            
            # Simple validation for hex color
            return value.startswith("#") and len(value) in (4, 7, 9)
        
        return False
    
    def _handle_setting_changed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle setting changed event.
        
        Args:
            event_data: Event data
        """
        key = event_data.get("key")
        value = event_data.get("value")
        
        if key and key in self.settings:
            setting = self.settings[key]
            
            # Handle special settings
            if key == "ui.theme":
                try:
                    theme = UITheme[value]
                    self.ui_manager.set_theme(theme)
                except (KeyError, ValueError):
                    pass
            
            elif key == "ui.font_size":
                try:
                    font_size = int(value)
                    self.ui_manager.set_font_size(font_size)
                except (ValueError, TypeError):
                    pass
            
            elif key == "ui.animations_enabled":
                try:
                    enabled = bool(value)
                    self.ui_manager.toggle_animations(enabled)
                except (ValueError, TypeError):
                    pass
    
    def _register_default_settings(self) -> None:
        """Register default settings."""
        # Register sections
        self.register_section(
            "general",
            "General",
            "General settings",
            "settings",
            0
        )
        
        self.register_section(
            "appearance",
            "Appearance",
            "Appearance settings",
            "palette",
            1
        )
        
        self.register_section(
            "chat",
            "Chat",
            "Chat settings",
            "chat",
            2
        )
        
        self.register_section(
            "api",
            "API",
            "API settings",
            "cloud",
            3
        )
        
        # Add API provider subsections
        self.register_section(
            "api_openrouter",
            "OpenRouter",
            "OpenRouter API settings",
            "cloud",
            0
        )
        
        self.register_section(
            "api_anthropic",
            "Anthropic",
            "Anthropic API settings",
            "cloud",
            1
        )
        
        self.register_section(
            "api_deepseek",
            "DeepSeek",
            "DeepSeek API settings",
            "cloud",
            2
        )
        
        self.register_section(
            "api_ollama",
            "Ollama",
            "Ollama API settings",
            "cloud",
            3
        )
        
        self.register_section(
            "api_litellm",
            "LiteLLM",
            "LiteLLM API settings",
            "cloud",
            4
        )
        
        self.register_section(
            "advanced",
            "Advanced",
            "Advanced settings",
            "code",
            4
        )
        
        # General settings
        self.register_setting(Setting(
            "general.startup_action",
            "Startup Action",
            "Action to perform on startup",
            SettingType.SELECT,
            "new_chat",
            [
                {"label": "New Chat", "value": "new_chat"},
                {"label": "Continue Last Chat", "value": "continue_last"},
                {"label": "Show Chat List", "value": "chat_list"}
            ],
            "general",
            0
        ))
        
        self.register_setting(Setting(
            "general.confirm_exit",
            "Confirm Exit",
            "Show confirmation dialog before exiting",
            SettingType.BOOLEAN,
            True,
            None,
            "general",
            1
        ))
        
        # Appearance settings
        self.register_setting(Setting(
            "ui.theme",
            "Theme",
            "UI theme",
            SettingType.SELECT,
            UITheme.SYSTEM.name,
            [
                {"label": "Light", "value": UITheme.LIGHT.name},
                {"label": "Dark", "value": UITheme.DARK.name},
                {"label": "System", "value": UITheme.SYSTEM.name}
            ],
            "appearance",
            0
        ))
        
        self.register_setting(Setting(
            "ui.font_size",
            "Font Size",
            "UI font size",
            SettingType.NUMBER,
            14,
            None,
            "appearance",
            1
        ))
        
        self.register_setting(Setting(
            "ui.animations_enabled",
            "Enable Animations",
            "Enable UI animations",
            SettingType.BOOLEAN,
            True,
            None,
            "appearance",
            2
        ))
        
        # Chat settings
        self.register_setting(Setting(
            "chat.auto_send_on_enter",
            "Auto-send on Enter",
            "Automatically send message when Enter key is pressed",
            SettingType.BOOLEAN,
            True,
            None,
            "chat",
            0
        ))
        
        self.register_setting(Setting(
            "chat.show_timestamps",
            "Show Timestamps",
            "Show timestamps for messages",
            SettingType.BOOLEAN,
            False,
            None,
            "chat",
            1
        ))
        
        self.register_setting(Setting(
            "chat.max_history",
            "Max History",
            "Maximum number of messages to keep in history",
            SettingType.NUMBER,
            100,
            None,
            "chat",
            2
        ))
        
        # API settings
        self.register_setting(Setting(
            "api.default_provider",
            "Default Provider",
            "Default API provider",
            SettingType.SELECT,
            "openrouter",
            [
                {"label": "OpenRouter", "value": "openrouter"},
                {"label": "DeepSeek", "value": "deepseek"},
                {"label": "Anthropic", "value": "anthropic"},
                {"label": "Ollama", "value": "ollama"},
                {"label": "LiteLLM", "value": "litellm"}
            ],
            "api",
            0
        ))
        
        # OpenRouter API settings
        self.register_setting(Setting(
            "api.openrouter.api_key",
            "API Key",
            "OpenRouter API key",
            SettingType.PASSWORD,
            "",
            None,
            "api_openrouter",
            0
        ))
        
        self.register_setting(Setting(
            "api.openrouter.default_model",
            "Default Model",
            "Default model to use with OpenRouter",
            SettingType.SELECT,
            "openai/gpt-3.5-turbo",
            [
                {"label": "GPT-3.5 Turbo", "value": "openai/gpt-3.5-turbo"},
                {"label": "GPT-4", "value": "openai/gpt-4"},
                {"label": "Claude 3 Opus", "value": "anthropic/claude-3-opus"},
                {"label": "Claude 3 Sonnet", "value": "anthropic/claude-3-sonnet"},
                {"label": "Llama 3 70B", "value": "meta-llama/llama-3-70b-instruct"},
                {"label": "Mistral Large", "value": "mistralai/mistral-large"}
            ],
            "api_openrouter",
            1
        ))
        
        # Anthropic API settings
        self.register_setting(Setting(
            "api.anthropic.api_key",
            "API Key",
            "Anthropic API key",
            SettingType.PASSWORD,
            "",
            None,
            "api_anthropic",
            0
        ))
        
        self.register_setting(Setting(
            "api.anthropic.default_model",
            "Default Model",
            "Default model to use with Anthropic",
            SettingType.SELECT,
            "claude-3-opus-20240229",
            [
                {"label": "Claude 3 Opus", "value": "claude-3-opus-20240229"},
                {"label": "Claude 3 Sonnet", "value": "claude-3-sonnet-20240229"},
                {"label": "Claude 3 Haiku", "value": "claude-3-haiku-20240307"}
            ],
            "api_anthropic",
            1
        ))
        
        # DeepSeek API settings
        self.register_setting(Setting(
            "api.deepseek.api_key",
            "API Key",
            "DeepSeek API key",
            SettingType.PASSWORD,
            "",
            None,
            "api_deepseek",
            0
        ))
        
        self.register_setting(Setting(
            "api.deepseek.default_model",
            "Default Model",
            "Default model to use with DeepSeek",
            SettingType.SELECT,
            "deepseek-chat",
            [
                {"label": "DeepSeek Chat", "value": "deepseek-chat"},
                {"label": "DeepSeek Coder", "value": "deepseek-coder"}
            ],
            "api_deepseek",
            1
        ))
        
        # Ollama API settings
        self.register_setting(Setting(
            "api.ollama.host",
            "Host",
            "Ollama host URL (e.g., http://localhost:11434)",
            SettingType.STRING,
            "http://localhost:11434",
            None,
            "api_ollama",
            0
        ))
        
        self.register_setting(Setting(
            "api.ollama.default_model",
            "Default Model",
            "Default model to use with Ollama",
            SettingType.STRING,
            "llama3",
            None,
            "api_ollama",
            1
        ))
        
        # LiteLLM API settings
        self.register_setting(Setting(
            "api.litellm.host",
            "Host",
            "LiteLLM host URL",
            SettingType.STRING,
            "http://localhost:8000",
            None,
            "api_litellm",
            0
        ))
        
        self.register_setting(Setting(
            "api.litellm.api_key",
            "API Key",
            "LiteLLM API key (if required)",
            SettingType.PASSWORD,
            "",
            None,
            "api_litellm",
            1
        ))
        
        self.register_setting(Setting(
            "api.litellm.default_model",
            "Default Model",
            "Default model to use with LiteLLM",
            SettingType.STRING,
            "gpt-3.5-turbo",
            None,
            "api_litellm",
            2
        ))
        
        # Common API settings
        self.register_setting(Setting(
            "api.temperature",
            "Temperature",
            "Model temperature (0.0 to 1.0)",
            SettingType.NUMBER,
            0.7,
            None,
            "api",
            1
        ))
        
        self.register_setting(Setting(
            "api.max_tokens",
            "Max Tokens",
            "Maximum number of tokens to generate",
            SettingType.NUMBER,
            1024,
            None,
            "api",
            2
        ))
        
        self.register_setting(Setting(
            "api.cache.enabled",
            "Enable Cache",
            "Enable API response caching",
            SettingType.BOOLEAN,
            True,
            None,
            "api",
            3
        ))
        
        self.register_setting(Setting(
            "api.cache.ttl_seconds",
            "Cache TTL",
            "Time to live for cached responses (seconds)",
            SettingType.NUMBER,
            86400,  # 24 hours
            None,
            "api",
            4
        ))
        
        # Advanced settings
        self.register_setting(Setting(
            "advanced.debug_mode",
            "Debug Mode",
            "Enable debug mode",
            SettingType.BOOLEAN,
            False,
            None,
            "advanced",
            0
        ))
        
        self.register_setting(Setting(
            "advanced.log_level",
            "Log Level",
            "Logging level",
            SettingType.SELECT,
            "info",
            [
                {"label": "Debug", "value": "debug"},
                {"label": "Info", "value": "info"},
                {"label": "Warning", "value": "warning"},
                {"label": "Error", "value": "error"},
                {"label": "Critical", "value": "critical"}
            ],
            "advanced",
            1
        ))
    
    def startup(self) -> None:
        """Start the settings panel."""
        # Register default settings
        self._register_default_settings()
        
        self.logger.info("SettingsPanel started")
    
    def shutdown(self) -> None:
        """Stop the settings panel."""
        self.logger.info("SettingsPanel stopped")

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
    
    # Initialize SettingsPanel
    settings_panel = SettingsPanel.get_instance()
    settings_panel.startup()
    
    # Example usage
    print(f"Theme: {settings_panel.get_setting_value('ui.theme')}")
    settings_panel.set_setting_value('ui.theme', UITheme.DARK.name)
