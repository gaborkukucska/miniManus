# START OF FILE miniManus-main/minimanus/core/config_manager.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Manager for miniManus

This module implements the Configuration Manager component, responsible for
loading, saving, and providing access to application settings and secrets.
"""

import os
import sys # Keep sys import for sys.platform check
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Union, Set, Tuple
from pathlib import Path

# REMOVED: Direct import of UITheme and its placeholder to break circular dependency

logger = logging.getLogger("miniManus.ConfigManager")

# Define default base paths, assuming standard structure relative to this file
DEFAULT_BASE_DIR = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'minimanus'
DEFAULT_CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'minimanus'
DEFAULT_CONFIG_DIR_CONSISTENT = DEFAULT_BASE_DIR / 'config'


class ConfigurationManager:
    """
    ConfigurationManager handles loading, saving, and accessing settings.

    It separates general configuration (`config.json`) from sensitive data
    like API keys (`secrets.json`).
    """

    _instance = None
    _lock = threading.RLock() # Lock for thread-safe access to config/secrets dicts

    @classmethod
    def get_instance(cls) -> 'ConfigurationManager':
        """Get or create the singleton instance of ConfigurationManager."""
        if cls._instance is None:
            with cls._lock: # Ensure thread-safe singleton creation
                if cls._instance is None:
                    cls._instance = ConfigurationManager()
        return cls._instance

    def __init__(self):
        """Initialize the ConfigurationManager."""
        if ConfigurationManager._instance is not None:
            raise RuntimeError("ConfigurationManager is a singleton. Use get_instance() instead.")

        self.logger = logger

        # Configuration paths - can be overridden after instantiation
        self.config_dir = DEFAULT_CONFIG_DIR_CONSISTENT
        self.config_file = self.config_dir / 'config.json'
        self.secrets_file = self.config_dir / 'secrets.json'

        # Initialize configuration and secrets dictionaries
        self.config: Dict[str, Any] = self._get_default_config()
        self.secrets: Dict[str, Any] = {"api_keys": {}} # Default empty secrets structure

        # Ensure directories exist (might be redundant if __main__ does it, but safe)
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions on the config directory itself
            try:
                 os.chmod(self.config_dir, 0o700)
            except OSError as e:
                 self.logger.warning(f"Could not set permissions on config directory {self.config_dir}: {e}")
            except Exception as e:
                 self.logger.warning(f"Unexpected error setting permissions on config directory {self.config_dir}: {e}")


        except OSError as e:
             self.logger.error(f"Failed to create config directory {self.config_dir}: {e}", exc_info=True)
             # Continue with defaults in memory, saving might fail

        # Load existing configuration and secrets
        self._load_config()
        self._load_secrets()

        self.logger.info(f"ConfigurationManager initialized. Config path: {self.config_file}, Secrets path: {self.secrets_file}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Define the default application configuration."""
        default_agent_read_dir = Path.home() / "minimanus_files"

        # Use a simple string for the default theme value
        theme_default = 'dark'

        return {
            "general": {
                "log_level": "INFO",
            },
            "ui": {
                "host": "localhost",
                "port": 8080,
                "theme": theme_default, # Use the string default
                "max_output_lines": 1000,
                "animations_enabled": True,
                "compact_mode": False,
                "font_size": 14,
            },
            "chat": {
                 "max_history_for_llm": 20,
                 "auto_save": True,
            },
            "api": {
                "default_provider": "openrouter",
                "cache": {
                    "enabled": True,
                    "ttl_seconds": 3600,
                    "max_items": 500,
                },
                "providers": {
                    "openrouter": {
                        "enabled": True,
                        "base_url": "https://openrouter.ai/api/v1",
                        "timeout": 60,
                        "default_model": "openai/gpt-3.5-turbo",
                        "referer": "https://minimanus.app",
                        "x_title": "miniManus",
                    },
                    "deepseek": {
                        "enabled": True,
                        "base_url": "https://api.deepseek.com/v1",
                        "timeout": 30,
                        "default_model": "deepseek-chat",
                        "embedding_model": "deepseek-embedding",
                    },
                    "anthropic": {
                        "enabled": True,
                        "base_url": "https://api.anthropic.com/v1",
                        "timeout": 60,
                        "default_model": "claude-3-5-sonnet-20240620",
                    },
                    "ollama": {
                        "enabled": True,
                        "base_url": "http://localhost:11434",
                        "timeout": 120,
                        "default_model": "llama3",
                        "discovery_enabled": True,
                        "discovery_ports": [11434],
                        "discovery_max_hosts": 20,
                        "discovery_timeout": 1.0,
                    },
                    "litellm": {
                        "enabled": True,
                        "base_url": "http://localhost:8000",
                        "timeout": 60,
                        "default_model": "gpt-3.5-turbo",
                        "embedding_model": "text-embedding-ada-002",
                        "discovery_enabled": True,
                        "discovery_ports": [8000, 4000],
                        "discovery_max_hosts": 20,
                        "discovery_timeout": 1.0,
                    }
                }
            },
            "agent": {
                 "max_iterations": 5,
                 "default_provider": "openrouter",
                 "files": {
                      "allowed_read_dir": str(default_agent_read_dir),
                      "allowed_write_dir": str(default_agent_read_dir / "agent_writes"),
                 },
            },
            "models": {
                 "favorites": [],
                 "recents": [],
                 "max_recents": 10,
            },
            "resources": {
                "monitoring_interval": 30,
                "memory_warning_threshold": 75,
                "memory_critical_threshold": 85,
                "memory_emergency_threshold": 90,
                "cpu_warning_threshold": 80,
                "cpu_critical_threshold": 90,
                "cpu_emergency_threshold": 95,
                "storage_warning_threshold": 85,
                "storage_critical_threshold": 95,
                "storage_emergency_threshold": 98,
                "battery_warning_threshold": 20,
                "battery_critical_threshold": 10,
                "battery_emergency_threshold": 5,
            }
        }

    def _load_config(self) -> None:
        """Load configuration from config_file, merging with defaults."""
        if not self.config_file.exists():
            self.logger.info(f"Configuration file not found ({self.config_file}). Using defaults.")
            if not self.save_config():
                self.logger.error("Failed to save default configuration file.")
            return

        try:
            with self.config_file.open('r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            with self._lock:
                 self.config = self._deep_update(self._get_default_config(), loaded_config)

            self.logger.info(f"Configuration loaded successfully from {self.config_file}")

        except json.JSONDecodeError as e:
             self.logger.error(f"Error decoding JSON from configuration file {self.config_file}: {e}", exc_info=True)
             self.logger.warning("Using default configuration due to load error.")
             self.config = self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration from {self.config_file}: {e}", exc_info=True)
            self.logger.warning("Using default configuration due to load error.")
            self.config = self._get_default_config()

    def _load_secrets(self) -> None:
        """Load secrets from secrets_file."""
        if not self.secrets_file.exists():
            self.logger.info(f"Secrets file not found ({self.secrets_file}). Creating empty structure.")
            self.secrets = {"api_keys": {}}
            if not self.save_secrets():
                self.logger.error("Failed to save empty secrets file.")
            return

        try:
            if sys.platform != "win32":
                 try:
                    file_stat = os.stat(self.secrets_file)
                    if file_stat.st_mode & 0o077:
                        self.logger.warning(f"Secrets file {self.secrets_file} has potentially insecure permissions ({oct(file_stat.st_mode & 0o777)}). Recommended: 600.")
                 except OSError as e:
                     self.logger.warning(f"Could not check permissions for secrets file {self.secrets_file}: {e}")

            with self.secrets_file.open('r', encoding='utf-8') as f:
                loaded_secrets = json.load(f)

            with self._lock:
                if "api_keys" not in loaded_secrets or not isinstance(loaded_secrets["api_keys"], dict):
                     self.logger.warning("Secrets file missing 'api_keys' dictionary. Resetting.")
                     self.secrets = {"api_keys": {}}
                else:
                     self.secrets = loaded_secrets

            self.logger.info(f"Secrets loaded successfully from {self.secrets_file}")

        except json.JSONDecodeError as e:
             self.logger.error(f"Error decoding JSON from secrets file {self.secrets_file}: {e}", exc_info=True)
             self.secrets = {"api_keys": {}}
        except Exception as e:
            self.logger.error(f"Error loading secrets from {self.secrets_file}: {e}", exc_info=True)
            self.secrets = {"api_keys": {}}

    def save_config(self) -> bool:
        """Save current configuration (excluding secrets) to config_file."""
        config_to_save = None
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)

            with self._lock:
                 config_to_save = json.loads(json.dumps(self.config))
                 api_settings = config_to_save.get('api')
                 if isinstance(api_settings, dict):
                      if 'api_keys' in api_settings:
                         self.logger.warning("Removing 'api_keys' from main config before saving.")
                         del api_settings['api_keys']

            temp_file_path = self.config_file.with_suffix(".tmp")
            self.logger.debug(f"Saving config dictionary to temp file {temp_file_path}. Root keys: {list(config_to_save.keys())}")
            with temp_file_path.open('w', encoding='utf-8') as f:
                 json.dump(config_to_save, f, indent=2, ensure_ascii=False)

            os.replace(temp_file_path, self.config_file)
            self.logger.debug(f"Configuration saved successfully to {self.config_file}")
            return True
        except json.JSONDecodeError as e:
             self.logger.error(f"Error creating copy of config for saving: {e}", exc_info=True)
             return False
        except Exception as e:
            self.logger.error(f"Error saving configuration to {self.config_file}: {e}", exc_info=True)
            if 'temp_file_path' in locals() and temp_file_path.exists():
                try: temp_file_path.unlink()
                except OSError: pass
            return False

    def save_secrets(self) -> bool:
        """Save current secrets dictionary to secrets_file with secure permissions."""
        secrets_to_save = None
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            if sys.platform != "win32":
                try:
                    os.chmod(self.config_dir, 0o700)
                except OSError as e:
                    self.logger.warning(f"Could not set permissions on config directory {self.config_dir}: {e}")

            with self._lock:
                 secrets_to_save = json.loads(json.dumps(self.secrets))

            temp_file_path = self.secrets_file.with_suffix(".tmp")
            self.logger.debug(f"Saving secrets to temp file {temp_file_path}.")
            with temp_file_path.open('w', encoding='utf-8') as f:
                 if sys.platform != "win32":
                     try:
                        os.chmod(temp_file_path, 0o600)
                     except OSError as e:
                         self.logger.warning(f"Could not set permissions on temp secrets file {temp_file_path}: {e}")
                 json.dump(secrets_to_save, f, indent=2, ensure_ascii=False)

            os.replace(temp_file_path, self.secrets_file)
            if sys.platform != "win32":
                try:
                    os.chmod(self.secrets_file, 0o600)
                except OSError as e:
                    self.logger.warning(f"Could not set final permissions on secrets file {self.secrets_file}: {e}")

            self.logger.debug(f"Secrets saved successfully to {self.secrets_file}")
            return True
        except json.JSONDecodeError as e:
             self.logger.error(f"Error creating copy of secrets for saving: {e}", exc_info=True)
             return False
        except Exception as e:
            self.logger.error(f"Error saving secrets to {self.secrets_file}: {e}", exc_info=True)
            if 'temp_file_path' in locals() and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except OSError: pass
            return False

    def get_config(self, path: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation. Returns copies for mutable types.
        """
        with self._lock:
             if path is None:
                 try:
                     safe_config = json.loads(json.dumps(self.config))
                     api_settings = safe_config.get('api')
                     if isinstance(api_settings, dict):
                         api_settings.pop('api_keys', None)
                         for p_key in list(api_settings.keys()):
                            if "api_key" in p_key:
                                api_settings.pop(p_key, None)
                     return safe_config
                 except Exception as e:
                     self.logger.error(f"Error creating deep copy of config: {e}", exc_info=True)
                     return {}

             keys = path.split('.')
             current_level = self.config
             for key in keys:
                 if isinstance(current_level, dict) and key in current_level:
                     current_level = current_level[key]
                 else:
                     return default

             if isinstance(current_level, (dict, list)):
                 try:
                    return json.loads(json.dumps(current_level))
                 except Exception:
                     return default
             else:
                 return current_level

    def set_config(self, path: str, value: Any) -> bool:
        """
        Set a configuration value using dot notation and save the config.
        Prevents setting API keys directly.
        """
        if not path:
            self.logger.error("Cannot set config with empty path.")
            return False

        path_lower = path.lower()
        if "api_key" in path_lower or "secret" in path_lower:
             self.logger.error(f"Attempted to set potentially sensitive key '{path}' via set_config. Use set_api_key or remove_api_key instead.")
             return False

        needs_save = False # Flag to track if save is needed
        with self._lock:
            keys = path.split('.')
            current_level = self.config
            try:
                for i, key in enumerate(keys[:-1]):
                    node = current_level.get(key)
                    if not isinstance(node, dict):
                        self.logger.debug(f"Creating/overwriting intermediate path '{key}' in '{path}'")
                        current_level[key] = {}
                    current_level = current_level[key]

                final_key = keys[-1]
                if not isinstance(current_level, dict):
                    self.logger.error(f"Cannot set key '{final_key}' because parent path '{'.'.join(keys[:-1])}' is not a dictionary.")
                    return False

                old_value = current_level.get(final_key, '__NOT_SET__')
                value_repr = f"type:{type(value)}" if isinstance(value, (dict, list)) else repr(value)
                old_value_repr = f"type:{type(old_value)}" if isinstance(old_value, (dict, list)) else repr(old_value)

                # Only update if the value actually changed
                if old_value != value:
                    current_level[final_key] = value
                    self.logger.debug(f"Internal config dict updated for '{path}'. Old={old_value_repr}, New={value_repr}")
                    needs_save = True
                else:
                    self.logger.debug(f"Value for '{path}' unchanged. Skipping save.")
                    needs_save = False

            except Exception as e:
                 self.logger.error(f"Error traversing config path '{path}': {e}", exc_info=True)
                 return False

        if needs_save:
            return self.save_config()
        else:
            return True # Report success even if no save was needed

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider from secrets."""
        with self._lock:
            api_keys_dict = self.secrets.get("api_keys", {})
            key = api_keys_dict.get(provider) if isinstance(api_keys_dict, dict) else None
        if key:
             self.logger.debug(f"Retrieved API key for provider '{provider}'.")
        else:
             self.logger.debug(f"No API key found for provider '{provider}'.")
        return key


    def set_api_key(self, provider: str, api_key: str) -> bool:
        """Set API key for a specific provider in secrets and save."""
        if not isinstance(provider, str) or not provider:
             self.logger.error("Invalid provider name for setting API key.")
             return False
        if not isinstance(api_key, str):
             self.logger.error("Invalid API key value (must be a string).")
             return False
        if not api_key:
             self.logger.warning(f"Attempted to set empty API key for '{provider}'. Use remove_api_key() to clear.")
             return self.remove_api_key(provider)

        needs_save = False
        with self._lock:
            if "api_keys" not in self.secrets or not isinstance(self.secrets["api_keys"], dict):
                 self.secrets["api_keys"] = {}
            if self.secrets["api_keys"].get(provider) != api_key:
                self.secrets["api_keys"][provider] = api_key
                self.logger.info(f"Set API key for provider '{provider}'.")
                needs_save = True
            else:
                self.logger.debug(f"API key for '{provider}' unchanged. Skipping save.")

        return self.save_secrets() if needs_save else True

    def remove_api_key(self, provider: str) -> bool:
        """Remove API key for a specific provider from secrets and save."""
        removed = False
        with self._lock:
            if "api_keys" in self.secrets and isinstance(self.secrets["api_keys"], dict):
                 if provider in self.secrets["api_keys"]:
                     del self.secrets["api_keys"][provider]
                     removed = True
                     self.logger.info(f"Removed API key for provider '{provider}'.")
                 else:
                      self.logger.debug(f"API key for provider '{provider}' not found, nothing to remove.")
            else:
                 self.logger.debug(f"'api_keys' structure missing or invalid in secrets, cannot remove key for '{provider}'.")

        if removed:
            return self.save_secrets()
        else:
            return True

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge source dict into target dict.
        Modifies target in place and also returns it.
        """
        for key, value in source.items():
            if isinstance(value, dict):
                node = target.get(key)
                if not isinstance(node, dict):
                     target[key] = {}
                     node = target[key]
                self._deep_update(node, value)
            else:
                target[key] = value
        return target

    def reset_to_defaults(self) -> bool:
        """Reset configuration (excluding secrets) to defaults and save."""
        self.logger.warning("Resetting configuration to default values.")
        with self._lock:
            self.config = self._get_default_config()
        return self.save_config()

# Example usage (if run directly) - Keep this for standalone testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    config_manager = ConfigurationManager.get_instance()
    print(f"\n--- Initial Values ---")
    print(f"Default Provider: {config_manager.get_config('api.default_provider')}")
    print(f"Ollama Timeout: {config_manager.get_config('api.providers.ollama.timeout')}")
    print(f"Non-existent Key: {config_manager.get_config('some.missing.key', default='Not Found')}")
    print("\n--- Setting Values ---")
    config_manager.set_config('ui.theme', 'light')
    config_manager.set_config('api.providers.ollama.timeout', 180)
    config_manager.set_config('new.nested.setting', True)
    print(f"New UI Theme: {config_manager.get_config('ui.theme')}")
    print(f"New Ollama Timeout: {config_manager.get_config('api.providers.ollama.timeout')}")
    print(f"New Nested Setting: {config_manager.get_config('new.nested.setting')}")
    print("\n--- API Keys ---")
    print(f"Initial Ollama Key: {config_manager.get_api_key('ollama')}")
    config_manager.set_api_key('ollama', 'ollama-test-key-123')
    print(f"Ollama Key Set: {'Yes' if config_manager.get_api_key('ollama') else 'No'}")
    config_manager.remove_api_key('ollama')
    print(f"Ollama Key After Removal: {config_manager.get_api_key('ollama')}")
    print("\n--- Resetting Config ---")
    config_manager.reset_to_defaults()
    print(f"UI Theme after reset: {config_manager.get_config('ui.theme')}")
    print(f"Ollama Timeout after reset: {config_manager.get_config('api.providers.ollama.timeout')}")
    print(f"New Nested Setting after reset: {config_manager.get_config('new.nested.setting', 'Not Found')}")
    print("\n--- Test Complete ---")

# END OF FILE miniManus-main/minimanus/core/config_manager.py
