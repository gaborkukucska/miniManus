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
    # REMOVED: from ..ui.ui_manager import UIManager, UITheme # No longer needed directly
    from ..api.api_manager import APIProvider # Keep APIProvider if used in default settings options
except ImportError as e:
    # Handle potential import errors during early startup or testing
    logging.getLogger("miniManus.SettingsPanel").critical(f"Failed to import required modules: {e}", exc_info=True)
    # Define dummy classes if running directly and imports fail
    if __name__ == "__main__":
        print("ImportError occurred, defining dummy classes for direct test.")
        class DummyEnum: pass
        class EventPriority(DummyEnum): NORMAL=1
        class ErrorCategory(DummyEnum): SYSTEM=1
        class ErrorSeverity(DummyEnum): INFO=1
        class APIProvider(DummyEnum): OPENROUTER=1; DEEPSEEK=2; ANTHROPIC=3; OLLAMA=4; LITELLM=5
        class UITheme(DummyEnum): SYSTEM=1; LIGHT=2; DARK=3 # Define dummy UITheme too
        class Event:
             def __init__(self, *args, **kwargs): pass
        class EventBus:
            _instance=None
            def publish_event(self,*args, **kwargs): print(f"DummyEventBus: Published {args}")
            def subscribe(self, *args, **kwargs): print(f"DummyEventBus: Subscribed {args}")
            @classmethod
            def get_instance(cls):
                if cls._instance is None: cls._instance = cls()
                return cls._instance
        class ErrorHandler:
            _instance=None
            def handle_error(self,*args, **kwargs): print(f"DummyErrorHandler: Handled {args}")
            @classmethod
            def get_instance(cls):
                if cls._instance is None: cls._instance = cls()
                return cls._instance
        class ConfigurationManager:
             _instance=None
             # Simulate getting defaults
             def _get_default_config(self): return SettingsPanel._get_mock_default_config() # Use mock defaults
             def get_config(self, key, default=None):
                 # Basic dot notation getter for mock config
                 parts = key.split('.')
                 val = self._get_default_config()
                 try:
                     for part in parts: val = val[part]
                     return val
                 except (KeyError, TypeError):
                     setting = SettingsPanel.get_instance().get_setting(key) # Get default from Setting definition
                     return setting.default_value if setting else default

             def set_config(self, key, value): print(f"DummyConfigManager: Set {key}={value}"); return True
             def get_api_key(self, provider): return None
             def set_api_key(self, provider, key): print(f"DummyConfigManager: Set API Key for {provider}"); return True
             @classmethod
             def get_instance(cls):
                 if cls._instance is None: cls._instance = cls()
                 return cls._instance


logger = logging.getLogger("miniManus.SettingsPanel")

class SettingType(Enum):
    """Types of settings."""
    BOOLEAN = auto()
    STRING = auto()
    NUMBER = auto()
    SELECT = auto()
    COLOR = auto()
    SECTION = auto()
    PASSWORD = auto()

class Setting:
    """Represents a setting."""

    def __init__(self, key: str, name: str, description: str, type: SettingType,
                default_value: Any, options: Optional[List[Dict[str, Any]]] = None,
                section: Optional[str] = None, order: int = 0,
                min_value: Optional[Union[int, float]] = None, # Add min/max/step for numbers
                max_value: Optional[Union[int, float]] = None,
                step: Optional[Union[int, float]] = None):
        self.key = key
        self.name = name
        self.description = description
        self.type = type
        self.default_value = default_value
        self.options = options or []
        self.section = section
        self.order = order
        self.min_value = min_value
        self.max_value = max_value
        self.step = step

    def to_dict(self) -> Dict[str, Any]:
        """Convert setting to dictionary."""
        data = {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "type": self.type.name.lower(),
            "default_value": self.default_value,
            "options": self.options,
            "section": self.section,
            "order": self.order
        }
        # Add optional fields if they exist
        if self.min_value is not None: data["min_value"] = self.min_value
        if self.max_value is not None: data["max_value"] = self.max_value
        if self.step is not None: data["step"] = self.step
        return data

class SettingsPanel:
    """
    SettingsPanel provides a mobile-optimized settings interface for miniManus.
    """

    _instance = None

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
        # REMOVED: self.ui_manager = UIManager.get_instance()

        self.settings: Dict[str, Setting] = {}
        self.sections: Dict[str, Dict[str, Any]] = {}

        # Subscribe directly to the event bus for settings changes
        self.event_bus.subscribe("settings.changed", self._handle_setting_changed)

        self.logger.info("SettingsPanel initialized")

    def register_setting(self, setting: Setting) -> None:
        """Register a setting."""
        if setting.key in self.settings:
             self.logger.warning(f"Setting key '{setting.key}' is already registered. Overwriting.")
        self.settings[setting.key] = setting
        self.logger.debug(f"Registered setting: {setting.key}")

    def register_section(self, id: str, name: str, description: str, icon: Optional[str] = None,
                       order: int = 0) -> None:
        """Register a settings section."""
        self.sections[id] = {
            "id": id,
            "name": name,
            "description": description,
            "icon": icon,
            "order": order
        }
        self.logger.debug(f"Registered settings section: {id}")

    def get_setting(self, key: str) -> Optional[Setting]:
        """Get a setting."""
        return self.settings.get(key)

    def get_section(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a settings section."""
        return self.sections.get(id)

    def get_all_settings(self) -> List[Setting]:
        """Get all settings."""
        # Return sorted by section order, then setting order
        return sorted(list(self.settings.values()), key=lambda s: (self.sections.get(s.section, {}).get("order", 99), s.order))

    def get_all_sections(self) -> List[Dict[str, Any]]:
        """Get all settings sections."""
        # Return sorted by order
        return sorted(list(self.sections.values()), key=lambda s: s.get("order", 99))

    def get_section_settings(self, section_id: str) -> List[Setting]:
        """Get settings for a section."""
        # Return sorted by setting order
        return sorted(
            [setting for setting in self.settings.values() if setting.section == section_id],
            key=lambda s: s.order
        )

    def get_setting_value(self, key: str) -> Any:
        """Get the current value of a setting from ConfigManager."""
        setting = self.get_setting(key)
        if setting is None:
            self.logger.warning(f"Attempted to get value for unknown setting: {key}")
            return None
        # Always fetch from ConfigManager to get the current value
        return self.config_manager.get_config(key, setting.default_value)

    def set_setting_value(self, key: str, value: Any) -> bool:
        """Set the value of a setting via ConfigManager."""
        setting = self.get_setting(key)
        if setting is None:
            self.logger.warning(f"Attempted to set unknown setting: {key}")
            return False

        if not self._validate_setting_value(setting, value):
            self.logger.warning(f"Invalid value '{value}' for setting {key} (type: {setting.type.name})")
            return False

        # Use ConfigManager to set and save
        if self.config_manager.set_config(key, value):
            # Publish event only if saving succeeded
            self.event_bus.publish_event("settings.changed", {
                "key": key,
                "value": value
            }, priority=EventPriority.NORMAL)
            self.logger.debug(f"Set setting {key} to {value}")
            return True
        else:
            self.logger.error(f"Failed to save setting {key} via ConfigManager.")
            return False

    def reset_setting(self, key: str) -> bool:
        """Reset a setting to its default value via ConfigManager."""
        setting = self.get_setting(key)
        if setting is None:
            return False

        if self.config_manager.set_config(key, setting.default_value):
            self.event_bus.publish_event("settings.changed", {
                "key": key,
                "value": setting.default_value
            }, priority=EventPriority.NORMAL)
            self.logger.debug(f"Reset setting {key} to default value")
            return True
        else:
            self.logger.error(f"Failed to save reset setting {key} via ConfigManager.")
            return False

    def _validate_setting_value(self, setting: Setting, value: Any) -> bool:
        """Validate a setting value."""
        if setting.type == SettingType.BOOLEAN:
            return isinstance(value, bool)
        elif setting.type in (SettingType.STRING, SettingType.PASSWORD, SettingType.COLOR):
            # Allow empty strings for passwords/strings
            return isinstance(value, str)
        elif setting.type == SettingType.NUMBER:
            is_num = isinstance(value, (int, float))
            if not is_num: return False
            # Check range if defined
            if setting.min_value is not None and value < setting.min_value: return False
            if setting.max_value is not None and value > setting.max_value: return False
            return True
        elif setting.type == SettingType.SELECT:
            if not setting.options: return False
            valid_values = [option.get("value") for option in setting.options]
            return value in valid_values
        return False # Unknown type

    def _handle_setting_changed(self, event_data: Dict[str, Any]) -> None:
        """Handle setting changed event."""
        key = event_data.get("key")
        value = event_data.get("value")
        self.logger.debug(f"Internal handler: Setting changed - {key} = {value}")

    # --- Restored Explicit Settings Registration ---
    def _register_default_settings(self) -> None:
        """Register default settings explicitly."""
        self.settings.clear() # Clear any previous registrations
        self.sections.clear()

        # Define sections
        self.register_section("general", "General", "General application settings", order=0)
        self.register_section("ui", "Appearance", "User interface settings", order=1)
        self.register_section("chat", "Chat", "Chat behavior settings", order=2)
        self.register_section("api", "API Configuration", "LLM Provider settings", order=3)
        self.register_section("agent", "Agent", "Agent behavior settings", order=4)
        self.register_section("resources", "Resources", "Resource monitoring", order=5)

        # General Settings
        self.register_setting(Setting(
            key="general.log_level", name="Log Level", type=SettingType.SELECT,
            description="Application logging detail level",
            default_value="INFO", section="general", order=0,
            options=[{"label": lvl, "value": lvl} for lvl in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]]
        ))

        # UI Settings
        self.register_setting(Setting(
            key="ui.host", name="Listen Host", type=SettingType.STRING,
            description="Network address for the UI server (use 0.0.0.0 for external access)",
            default_value="localhost", section="ui", order=0
        ))
        self.register_setting(Setting(
            key="ui.port", name="Listen Port", type=SettingType.NUMBER,
            description="Network port for the UI server",
            default_value=8080, section="ui", order=1, min_value=1025, max_value=65535
        ))
        self.register_setting(Setting(
            key="ui.theme", name="Theme", type=SettingType.SELECT,
            description="Select the UI theme", default_value="dark", section="ui", order=2,
            options=[{"label": "System", "value": "system"}, {"label": "Light", "value": "light"}, {"label": "Dark", "value": "dark"}]
        ))
        self.register_setting(Setting(
            key="ui.font_size", name="Font Size", type=SettingType.NUMBER,
            description="Adjust UI font size (pixels)", default_value=14, section="ui", order=3,
            min_value=10, max_value=20, step=1 # Added range/step for UI slider hint
        ))
        self.register_setting(Setting(
            key="ui.animations_enabled", name="Enable Animations", type=SettingType.BOOLEAN,
            description="Enable UI transition animations", default_value=True, section="ui", order=4
        ))
        self.register_setting(Setting(
            key="ui.compact_mode", name="Compact Mode", type=SettingType.BOOLEAN,
            description="Use a more compact layout", default_value=False, section="ui", order=5
        ))
        # self.register_setting(Setting("ui.max_output_lines", ...)) # Add if needed

        # Chat Settings
        self.register_setting(Setting(
            key="chat.max_history_for_llm", name="Context History Length", type=SettingType.NUMBER,
            description="Max number of past messages sent to LLM", default_value=20, section="chat", order=0, min_value=0, max_value=100
        ))
        self.register_setting(Setting(
            key="chat.auto_save", name="Auto-Save Sessions", type=SettingType.BOOLEAN,
            description="Automatically save chat sessions", default_value=True, section="chat", order=1
        ))

        # API Settings (General)
        self.register_setting(Setting(
            key="api.default_provider", name="Default Provider", type=SettingType.SELECT,
            description="Primary LLM provider to use if preferred fails", default_value="openrouter", section="api", order=0,
            options=[{"label": p.name.capitalize(), "value": p.name.lower()} for p in APIProvider if p != APIProvider.CUSTOM]
        ))
        self.register_setting(Setting(
            key="api.cache.enabled", name="Enable API Cache", type=SettingType.BOOLEAN,
            description="Cache identical API requests to reduce calls", default_value=True, section="api", order=1
        ))
        self.register_setting(Setting(
            key="api.cache.ttl_seconds", name="Cache TTL (seconds)", type=SettingType.NUMBER,
            description="How long to keep cached API responses", default_value=3600, section="api", order=2, min_value=0
        ))
        self.register_setting(Setting(
             key="api.cache.max_items", name="Max Cache Items", type=SettingType.NUMBER,
             description="Maximum number of API responses to cache", default_value=500, section="api", order=3, min_value=0
        ))

        # --- Provider Specific Sections & Settings ---
        provider_index = 0
        for provider in APIProvider:
             if provider == APIProvider.CUSTOM: continue
             provider_name = provider.name.lower()
             section_id = f"api_{provider_name}"
             self.register_section(section_id, provider.name.capitalize(), f"Settings for {provider.name.capitalize()}", order=provider_index)
             provider_index += 1

             # Common provider settings
             self.register_setting(Setting(f"api.{provider_name}.enabled", "Enable Provider", SettingType.BOOLEAN, f"Use {provider.name.capitalize()} for API calls", default_value=True, section=section_id, order=0))
             self.register_setting(Setting(f"api.{provider_name}.api_key", "API Key", SettingType.PASSWORD, f"{provider.name.capitalize()} API Key (Stored Securely)", default_value="", section=section_id, order=1)) # API Key is always password
             if provider not in [APIProvider.OLLAMA]: # Ollama doesn't use model list from API in the same way typically
                 self.register_setting(Setting(f"api.{provider_name}.default_model", "Default Model", SettingType.STRING, f"Default {provider.name.capitalize()} model ID for new chats", default_value=self.config_manager.get_config(f"api.{provider_name}.default_model"), section=section_id, order=2)) # Use default from config

             # Specific settings
             if provider == APIProvider.OPENROUTER:
                  self.register_setting(Setting(f"api.{provider_name}.base_url", "Base URL", SettingType.STRING, f"{provider.name.capitalize()} API endpoint URL", default_value="https://openrouter.ai/api/v1", section=section_id, order=10))
                  self.register_setting(Setting(f"api.{provider_name}.timeout", "Timeout (s)", SettingType.NUMBER, "Request timeout", default_value=60, section=section_id, order=11, min_value=1))
                  self.register_setting(Setting(f"api.{provider_name}.referer", "HTTP Referer", SettingType.STRING, "Referer header (often required)", default_value="https://minimanus.app", section=section_id, order=12))
                  self.register_setting(Setting(f"api.{provider_name}.x_title", "X-Title", SettingType.STRING, "X-Title header (optional)", default_value="miniManus", section=section_id, order=13))
             elif provider == APIProvider.ANTHROPIC:
                  self.register_setting(Setting(f"api.{provider_name}.base_url", "Base URL", SettingType.STRING, f"{provider.name.capitalize()} API endpoint URL", default_value="https://api.anthropic.com/v1", section=section_id, order=10))
                  self.register_setting(Setting(f"api.{provider_name}.timeout", "Timeout (s)", SettingType.NUMBER, "Request timeout", default_value=60, section=section_id, order=11, min_value=1))
                  # Note: Anthropic models are often hardcoded in adapter, but setting default still useful
             elif provider == APIProvider.DEEPSEEK:
                  self.register_setting(Setting(f"api.{provider_name}.base_url", "Base URL", SettingType.STRING, f"{provider.name.capitalize()} API endpoint URL", default_value="https://api.deepseek.com/v1", section=section_id, order=10))
                  self.register_setting(Setting(f"api.{provider_name}.timeout", "Timeout (s)", SettingType.NUMBER, "Request timeout", default_value=30, section=section_id, order=11, min_value=1))
                  self.register_setting(Setting(f"api.{provider_name}.embedding_model", "Embedding Model", SettingType.STRING, "Default embedding model ID", default_value=self.config_manager.get_config(f"api.{provider_name}.embedding_model"), section=section_id, order=3))
             elif provider == APIProvider.OLLAMA:
                  self.register_setting(Setting(f"api.{provider_name}.base_url", "Host URL", SettingType.STRING, f"{provider.name.capitalize()} Server URL (e.g., http://localhost:11434)", default_value="http://localhost:11434", section=section_id, order=10))
                  self.register_setting(Setting(f"api.{provider_name}.timeout", "Timeout (s)", SettingType.NUMBER, "Request timeout (longer for local models)", default_value=120, section=section_id, order=11, min_value=1))
                  self.register_setting(Setting(f"api.{provider_name}.default_model", "Default Model", SettingType.STRING, "Default Ollama model name (e.g., llama3)", default_value="llama3", section=section_id, order=2))
                  self.register_setting(Setting(f"api.{provider_name}.discovery_enabled", "Enable Discovery", SettingType.BOOLEAN, "Scan network for Ollama servers", default_value=True, section=section_id, order=12))
             elif provider == APIProvider.LITELLM:
                  self.register_setting(Setting(f"api.{provider_name}.base_url", "Host URL", SettingType.STRING, f"{provider.name.capitalize()} Proxy URL (e.g., http://localhost:8000)", default_value="http://localhost:8000", section=section_id, order=10))
                  # Key for LiteLLM is optional, handled by api_key setting order=1
                  self.register_setting(Setting(f"api.{provider_name}.timeout", "Timeout (s)", SettingType.NUMBER, "Request timeout", default_value=60, section=section_id, order=11, min_value=1))
                  self.register_setting(Setting(f"api.{provider_name}.default_model", "Default Model", SettingType.STRING, f"Default model passed to LiteLLM", default_value=self.config_manager.get_config(f"api.{provider_name}.default_model"), section=section_id, order=2))
                  self.register_setting(Setting(f"api.{provider_name}.embedding_model", "Embedding Model", SettingType.STRING, "Default embedding model passed to LiteLLM", default_value=self.config_manager.get_config(f"api.{provider_name}.embedding_model"), section=section_id, order=3))
                  self.register_setting(Setting(f"api.{provider_name}.discovery_enabled", "Enable Discovery", SettingType.BOOLEAN, "Scan network for LiteLLM servers", default_value=True, section=section_id, order=12))

        # Agent Settings
        self.register_setting(Setting(
             key="agent.max_iterations", name="Max Agent Iterations", type=SettingType.NUMBER,
             description="Max planning/tool cycles per request", default_value=5, section="agent", order=0, min_value=1, max_value = 20
        ))
        self.register_setting(Setting(
             key="agent.default_provider", name="Agent Default Provider", type=SettingType.SELECT,
             description="LLM provider the agent uses for reasoning/planning", default_value="openrouter", section="agent", order=1,
             options=[{"label": p.name.capitalize(), "value": p.name.lower()} for p in APIProvider if p != APIProvider.CUSTOM]
        ))
        self.register_setting(Setting(
             key="agent.files.allowed_read_dir", name="Allowed Read Directory", type=SettingType.STRING,
             description="Base directory agent can read files from (Use with caution!)", default_value=str(Path.home() / "minimanus_files"), section="agent", order=2
        ))
        self.register_setting(Setting(
             key="agent.files.allowed_write_dir", name="Allowed Write Directory", type=SettingType.STRING,
             description="Base directory agent can write files to (Use with extreme caution!)", default_value=str(Path.home() / "minimanus_files" / "agent_writes"), section="agent", order=3
        ))

        # Resource Settings
        self.register_setting(Setting("resources.monitoring_interval", "Monitor Interval (s)", SettingType.NUMBER, "How often to check resources", 30, section="resources", order=0, min_value=5))
        self.register_setting(Setting("resources.memory_warning_threshold", "Memory Warning (%)", SettingType.NUMBER, "Memory usage warning level", 75, section="resources", order=1, min_value=0, max_value=100))
        self.register_setting(Setting("resources.memory_critical_threshold", "Memory Critical (%)", SettingType.NUMBER, "Memory usage critical level", 85, section="resources", order=2, min_value=0, max_value=100))
        # Add other resource thresholds...

        self.logger.info(f"Registered {len(self.settings)} default settings explicitly.")

    def startup(self) -> None:
        """Start the settings panel."""
        # Register default settings explicitly
        self._register_default_settings()
        self.logger.info("SettingsPanel started")

    def shutdown(self) -> None:
        """Stop the settings panel."""
        self.logger.info("SettingsPanel stopped")

    # --- Mock default config for standalone testing ---
    @staticmethod
    def _get_mock_default_config():
         # Provide a minimal structure similar to ConfigManager's defaults
         # This is only used if run directly (`if __name__ == "__main__"`) and imports fail
         return {
             "general": {"log_level": "INFO"},
             "ui": {"host": "localhost", "port": 8080, "theme": "dark", "font_size": 14},
             "chat": {"max_history_for_llm": 20, "auto_save": True},
             "api": {
                 "default_provider": "openrouter",
                 "cache": {"enabled": True, "ttl_seconds": 3600, "max_items": 500},
                 "providers": {
                     "openrouter": {"enabled": True, "base_url": "...", "default_model": "openai/gpt-3.5-turbo", "timeout": 60, "referer": "...", "x_title": "..."},
                     "anthropic": {"enabled": True, "base_url": "...", "default_model": "claude-3-5-sonnet-20240620", "timeout": 60},
                     "deepseek": {"enabled": True, "base_url": "...", "default_model": "deepseek-chat", "embedding_model": "deepseek-embedding", "timeout": 30},
                     "ollama": {"enabled": True, "base_url": "http://localhost:11434", "default_model": "llama3", "timeout": 120, "discovery_enabled": True},
                     "litellm": {"enabled": True, "base_url": "http://localhost:8000", "default_model": "gpt-3.5-turbo", "embedding_model": "text-embedding-ada-002", "timeout": 60, "discovery_enabled": True}
                 }
             },
             "agent": {
                  "max_iterations": 5, "default_provider": "openrouter",
                  "files": {"allowed_read_dir": str(Path.home() / "minimanus_files"), "allowed_write_dir": str(Path.home() / "minimanus_files" / "agent_writes")}
              },
             "resources": {"monitoring_interval": 30, "memory_warning_threshold": 75, "memory_critical_threshold": 85}
         }


# Example usage (If needed for direct testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("--- Running SettingsPanel Standalone Test ---")
    # Use dummy instances defined in the ImportError block
    settings_panel = SettingsPanel.get_instance()
    settings_panel.startup()

    print("\n--- Sections ---")
    for section in settings_panel.get_all_sections():
        print(f"- {section['id']} (Order: {section['order']})")

    print("\n--- Settings (Example: ui.theme) ---")
    theme_setting = settings_panel.get_setting("ui.theme")
    if theme_setting:
         print(f"Setting: {theme_setting.to_dict()}")
         print(f"Current Value: {settings_panel.get_setting_value('ui.theme')}")
         settings_panel.set_setting_value('ui.theme', 'light')
         print(f"New Value: {settings_panel.get_setting_value('ui.theme')}")
         settings_panel.reset_setting('ui.theme')
         print(f"Value after reset: {settings_panel.get_setting_value('ui.theme')}")
    else:
        print("ui.theme setting not found.")

    print("\n--- All Settings ---")
    all_settings_list = settings_panel.get_all_settings()
    print(f"Total settings registered: {len(all_settings_list)}")
    # for setting in all_settings_list:
    #      print(f"  - {setting.key} (Section: {setting.section}, Order: {setting.order})")


    settings_panel.shutdown()
    print("--- SettingsPanel Standalone Test Finished ---")
