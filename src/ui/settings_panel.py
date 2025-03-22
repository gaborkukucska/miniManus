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
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager
    from ui.ui_manager import UIManager, UITheme

logger = logging.getLogger("miniManus.SettingsPanel")

class SettingType(Enum):
    """Types of settings."""
    BOOLEAN = auto()
    STRING = auto()
    NUMBER = auto()
    SELECT = auto()
    COLOR = auto()
    SECTION = auto()

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
    
    def reset_all_settings(self) -> None:
        """Reset all settings to their default values."""
        for key, setting in self.settings.items():
            self.config_manager.set_config(key, setting.default_value)
            
            # Publish event
            self.event_bus.publish_event("settings.changed", {
                "key": key,
                "value": setting.default_value
            })
        
        self.logger.info("Reset all settings to default values")
    
    def _validate_setting_value(self, setting: Setting, value: Any) -> bool:
        """
        Validate a setting value.
        
        Args:
            setting: Setting to validate
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if setting.type == SettingType.BOOLEAN:
                return isinstance(value, bool)
            
            elif setting.type == SettingType.STRING:
                return isinstance(value, str)
            
            elif setting.type == SettingType.NUMBER:
                return isinstance(value, (int, float))
            
            elif setting.type == SettingType.SELECT:
                return any(opt.get("value") == value for opt in setting.options)
            
            elif setting.type == SettingType.COLOR:
                # Check if valid color string (hex, rgb, etc.)
                return isinstance(value, str) and (
                    value.startswith("#") or 
                    value.startswith("rgb") or 
                    value.startswith("hsl")
                )
            
            elif setting.type == SettingType.SECTION:
                # Sections don't have values
                return False
            
            return True
        
        except Exception as e:
            self.logger.warning(f"Error validating setting {setting.key}: {str(e)}")
            return False
    
    def _handle_setting_changed(self, event: Dict[str, Any]) -> None:
        """
        Handle setting changed event.
        
        Args:
            event: Event data
        """
        key = event.get("key")
        value = event.get("value")
        
        if key:
            self.logger.debug(f"Setting changed: {key} = {value}")
    
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
        
        self.register_section(
            "advanced",
            "Advanced",
            "Advanced settings",
            "code",
            4
        )
        
        # Register settings
        
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
                {"label": "Show Chat List", "value": "show_list"}
            ],
            "general",
            0
        ))
        
        self.register_setting(Setting(
            "general.confirm_exit",
            "Confirm Exit",
            "Show confirmation dialog when exiting",
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
            "ui.enable_animations",
            "Enable Animations",
            "Enable UI animations",
            SettingType.BOOLEAN,
            True,
            None,
            "appearance",
            2
        ))
        
        self.register_setting(Setting(
            "ui.compact_mode",
            "Compact Mode",
            "Use compact UI mode",
            SettingType.BOOLEAN,
            False,
            None,
            "appearance",
            3
        ))
        
        self.register_setting(Setting(
            "ui.accent_color",
            "Accent Color",
            "UI accent color",
            SettingType.COLOR,
            "#007bff",
            None,
            "appearance",
            4
        ))
        
        # Chat settings
        self.register_setting(Setting(
            "chat.default_model",
            "Default Model",
            "Default model for new chats",
            SettingType.STRING,
            "gpt-3.5-turbo",
            None,
            "chat",
            0
        ))
        
        self.register_setting(Setting(
            "chat.default_system_prompt",
            "Default System Prompt",
            "Default system prompt for new chats",
            SettingType.STRING,
            "You are a helpful assistant.",
            None,
            "chat",
            1
        ))
        
        self.register_setting(Setting(
            "chat.auto_save",
            "Auto Save",
            "Automatically save chat history",
            SettingType.BOOLEAN,
            True,
            None,
            "chat",
            2
        ))
        
        self.register_setting(Setting(
            "chat.max_history_length",
            "Max History Length",
            "Maximum number of messages to keep in history",
            SettingType.NUMBER,
            100,
            None,
            "chat",
            3
        ))
        
        # API settings
        self.register_setting(Setting(
            "api.default_provider",
            "Default Provider",
            "Default API provider",
            SettingType.SELECT,
            "openai",
            [
                {"label": "OpenAI", "value": "openai"},
                {"label": "OpenRouter", "value": "openrouter"},
                {"label": "DeepSeek", "value": "deepseek"},
                {"label": "Anthropic", "value": "anthropic"},
                {"label": "Ollama", "value": "ollama"},
                {"label": "LiteLLM", "value": "litellm"}
            ],
            "api",
            0
        ))
        
        self.register_setting(Setting(
            "api.cache.enabled",
            "Enable Cache",
            "Enable API response caching",
            SettingType.BOOLEAN,
            True,
            None,
            "api",
            1
        ))
        
        self.register_setting(Setting(
            "api.cache.ttl_seconds",
            "Cache TTL",
            "Time to live for cached responses (seconds)",
            SettingType.NUMBER,
            86400,  # 24 hours
            None,
            "api",
            2
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
                {"label": "Error", "value": "error"}
            ],
            "advanced",
            1
        ))
        
        self.register_setting(Setting(
            "advanced.max_memory_mb",
            "Max Memory (MB)",
            "Maximum memory usage (MB)",
            SettingType.NUMBER,
            256,
            None,
            "advanced",
            2
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
    print(f"Theme after change: {settings_panel.get_setting_value('ui.theme')}")
    
    # Shutdown
    settings_panel.shutdown()
    ui_manager.shutdown()
    event_bus.shutdown()
