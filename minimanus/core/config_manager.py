#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Manager for miniManus

This module implements the Configuration Manager component, responsible for
loading, saving, and providing access to application settings and secrets.
"""

import os
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Union, Set
from pathlib import Path

logger = logging.getLogger("miniManus.ConfigManager")

# Define default base paths, assuming standard structure relative to this file
# These can be overridden after instantiation if needed (e.g., by __main__.py)
DEFAULT_BASE_DIR = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'minimanus'
DEFAULT_CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'minimanus'
# For consistency with __main__.py, let's prioritize the XDG_DATA_HOME location for config too.
# If you prefer ~/.config, change DEFAULT_CONFIG_DIR definition.
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
            # Note: This might fail if the parent dirs don't have correct permissions
            try:
                 os.chmod(self.config_dir, 0o700)
            except OSError as e:
                 self.logger.warning(f"Could not set permissions on config directory {self.config_dir}: {e}")

        except OSError as e:
             self.logger.error(f"Failed to create config directory {self.config_dir}: {e}", exc_info=True)
             # Depending on severity, might raise exception or continue with defaults

        # Load existing configuration and secrets
        self._load_config()
        self._load_secrets()

        self.logger.info(f"ConfigurationManager initialized. Config path: {self.config_file}, Secrets path: {self.secrets_file}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Define the default application configuration."""
        # Define file paths using Path for better cross-platform handling
        default_agent_read_dir = Path.home() / "minimanus_files" # Example default

        return {
            "general": {
                "log_level": "INFO",
            },
            "ui": {
                "host": "localhost",
                "port": 8080,
                "theme": UITheme.DARK.name.lower() if 'UITheme' in globals() else "dark", # Default dark
                "max_output_lines": 1000,
                "animations_enabled": True,
                "compact_mode": False,
                "font_size": 14,
            },
            "chat": {
                 "max_history_for_llm": 20, # Max messages sent to LLM context
                 "auto_save": True,
                 # Default model selection is now per-provider
            },
            "api": {
                "default_provider": "openrouter", # Overall default if specific preferred fails
                "cache": {
                    "enabled": True,
                    "ttl_seconds": 3600,  # 1 hour
                    "max_items": 500,
                },
                "providers": {
                    # Settings specific to each provider adapter
                    "openrouter": {
                        "enabled": True, # Can disable providers entirely
                        "base_url": "https://openrouter.ai/api/v1",
                        "timeout": 60,
                        "default_model": "openai/gpt-3.5-turbo", # Default for this provider
                        "referer": "https://minimanus.app", # OpenRouter required header
                        "x_title": "miniManus", # OpenRouter optional header
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
                        "default_model": "claude-3-5-sonnet-20240620", # Updated default
                    },
                    "ollama": {
                        "enabled": True,
                        "base_url": "http://localhost:11434", # Base URL, paths added by adapter
                        "timeout": 120,
                        "default_model": "llama3",
                        "discovery_enabled": True,
                        "discovery_ports": [11434],
                        "discovery_max_hosts": 20,
                        "discovery_timeout": 1.0,
                    },
                    "litellm": {
                        "enabled": True,
                        "base_url": "http://localhost:8000", # Base URL
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
                 # Default provider used by agent if not overridden
                 "default_provider": "openrouter",
                 "files": {
                      # SECURITY: Define allowed base paths for file operations
                      "allowed_read_dir": str(default_agent_read_dir),
                      "allowed_write_dir": str(default_agent_read_dir / "agent_writes"), # Separate write dir
                 },
                 # Add other agent settings as needed
            },
            "models": { # For storing user preferences related to models
                 "favorites": [],
                 "recents": [],
                 "max_recents": 10,
            },
            "resources": { # Resource monitor thresholds
                "monitoring_interval": 30, # Check every 30 seconds
                # Percentages
                "memory_warning_threshold": 75,
                "memory_critical_threshold": 85,
                "memory_emergency_threshold": 90, # Termux OOM risk increases past ~50% of *device* RAM, use process % carefully
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
            self.save_config() # Create default config file
            return

        try:
            with self.config_file.open('r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            with self._lock:
                 # Deep merge loaded config onto default config
                 self.config = self._deep_update(self._get_default_config(), loaded_config)

            self.logger.info(f"Configuration loaded successfully from {self.config_file}")

        except json.JSONDecodeError as e:
             self.logger.error(f"Error decoding JSON from configuration file {self.config_file}: {e}", exc_info=True)
             # Decide behavior: Use defaults? Raise error? Backup corrupt file?
             self.logger.warning("Using default configuration due to load error.")
             self.config = self._get_default_config() # Reset to defaults
        except Exception as e:
            self.logger.error(f"Error loading configuration from {self.config_file}: {e}", exc_info=True)
            # Fallback to defaults might be safest
            self.logger.warning("Using default configuration due to load error.")
            self.config = self._get_default_config()

    def _load_secrets(self) -> None:
        """Load secrets from secrets_file."""
        if not self.secrets_file.exists():
            self.logger.info(f"Secrets file not found ({self.secrets_file}). Creating empty structure.")
            self.secrets = {"api_keys": {}}
            self.save_secrets() # Create empty secrets file with secure permissions
            return

        try:
            # Check permissions before reading (optional but good practice)
            if os.stat(self.secrets_file).st_mode & 0o077:
                 self.logger.warning(f"Secrets file {self.secrets_file} has insecure permissions. Consider setting to 600.")

            with self.secrets_file.open('r', encoding='utf-8') as f:
                loaded_secrets = json.load(f)

            with self._lock:
                # Ensure basic structure exists
                if "api_keys" not in loaded_secrets or not isinstance(loaded_secrets["api_keys"], dict):
                     self.logger.warning("Secrets file missing 'api_keys' dictionary. Resetting.")
                     self.secrets = {"api_keys": {}}
                else:
                     self.secrets = loaded_secrets

            self.logger.info(f"Secrets loaded successfully from {self.secrets_file}")

        except json.JSONDecodeError as e:
             self.logger.error(f"Error decoding JSON from secrets file {self.secrets_file}: {e}", exc_info=True)
             self.secrets = {"api_keys": {}} # Reset on error
        except Exception as e:
            self.logger.error(f"Error loading secrets from {self.secrets_file}: {e}", exc_info=True)
            self.secrets = {"api_keys": {}} # Reset on error

    def save_config(self) -> bool:
        """Save current configuration (excluding secrets) to config_file."""
        try:
            # Ensure directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)

            with self._lock:
                 # Create a copy to avoid modifying the live config dict directly during serialization
                 config_to_save = self.config.copy()
                 # Ensure secrets are not accidentally saved in config.json
                 if 'api_keys' in config_to_save.get('api', {}):
                      self.logger.warning("Attempting to save api_keys in main config, removing.")
                      del config_to_save['api']['api_keys']

            with self.config_file.open('w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Configuration saved successfully to {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration to {self.config_file}: {e}", exc_info=True)
            return False

    def save_secrets(self) -> bool:
        """Save current secrets dictionary to secrets_file with secure permissions."""
        try:
            # Ensure directory exists and has secure permissions
            self.config_dir.mkdir(parents=True, exist_ok=True)
            try:
                 os.chmod(self.config_dir, 0o700)
            except OSError as e:
                 self.logger.warning(f"Could not set permissions on config directory {self.config_dir}: {e}")


            with self._lock:
                 secrets_to_save = self.secrets.copy()

            # Write secrets atomically (write to temp file, then rename)
            temp_file_path = self.secrets_file.with_suffix(".tmp")
            with temp_file_path.open('w', encoding='utf-8') as f:
                 # Set restrictive permissions *before* writing sensitive data
                 os.chmod(temp_file_path, 0o600)
                 json.dump(secrets_to_save, f, indent=2, ensure_ascii=False)

            # Atomically replace the old secrets file with the new one
            os.replace(temp_file_path, self.secrets_file)

            self.logger.debug(f"Secrets saved successfully to {self.secrets_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving secrets to {self.secrets_file}: {e}", exc_info=True)
            # Clean up temp file if rename failed
            if temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except OSError: pass
            return False

    def get_config(self, path: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            path: Dot-separated path (e.g., "api.providers.ollama.timeout").
                  If None, returns the entire config dictionary (excluding secrets).
            default: Value to return if path is not found.

        Returns:
            The configuration value or the default.
        """
        with self._lock:
             # Return a copy of the whole config if no path specified
             if path is None:
                 # Ensure secrets aren't included in the returned full config
                 safe_config = self.config.copy()
                 if 'api_keys' in safe_config.get('api', {}):
                     del safe_config['api']['api_keys']
                 return safe_config

             keys = path.split('.')
             current_level = self.config
             for key in keys:
                 if isinstance(current_level, dict) and key in current_level:
                     current_level = current_level[key]
                 else:
                     # Log if a requested key isn't found? Optional.
                     # self.logger.debug(f"Config path '{path}' not found, returning default: {default}")
                     return default
             return current_level

    def set_config(self, path: str, value: Any) -> bool:
        """
        Set a configuration value using dot notation and save the config.

        Args:
            path: Dot-separated path (e.g., "ui.theme").
            value: The value to set.

        Returns:
            True if successful, False otherwise.
        """
        if not path:
            self.logger.error("Cannot set config with empty path.")
            return False

        # Prevent setting secrets via this method
        if "api_key" in path.lower() or "secret" in path.lower():
             self.logger.error(f"Attempted to set potentially sensitive key '{path}' via set_config. Use set_api_key instead.")
             return False

        with self._lock:
            keys = path.split('.')
            current_level = self.config

            for i, key in enumerate(keys[:-1]):
                if key not in current_level or not isinstance(current_level[key], dict):
                    # If intermediate path doesn't exist or isn't a dict, create it
                    current_level[key] = {}
                current_level = current_level[key]

            # Set the final value
            final_key = keys[-1]
            if not isinstance(current_level, dict):
                 self.logger.error(f"Cannot set key '{final_key}' because parent path '{'.'.join(keys[:-1])}' is not a dictionary.")
                 return False

            current_level[final_key] = value
            self.logger.debug(f"Set config '{path}' to: {value}")

        # Save the updated configuration
        return self.save_config()

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider from secrets."""
        with self._lock:
            key = self.secrets.get("api_keys", {}).get(provider)
        if key:
            # Log retrieval carefully, avoid logging the key itself
             self.logger.debug(f"Retrieved API key for provider '{provider}'.")
        else:
             self.logger.debug(f"No API key found for provider '{provider}'.")
        return key


    def set_api_key(self, provider: str, api_key: str) -> bool:
        """Set API key for a specific provider in secrets and save."""
        if not isinstance(provider, str) or not provider:
             self.logger.error("Invalid provider name for setting API key.")
             return False
        if not isinstance(api_key, str): # Allow empty string to clear key
             self.logger.error("Invalid API key value (must be a string).")
             return False

        with self._lock:
            # Ensure the api_keys dictionary exists
            if "api_keys" not in self.secrets or not isinstance(self.secrets["api_keys"], dict):
                 self.secrets["api_keys"] = {}
            self.secrets["api_keys"][provider] = api_key
            # Avoid logging the key value
            self.logger.info(f"Set API key for provider '{provider}'.")

        return self.save_secrets()

    def remove_api_key(self, provider: str) -> bool:
        """Remove API key for a specific provider from secrets and save."""
        removed = False
        with self._lock:
            if "api_keys" in self.secrets and provider in self.secrets["api_keys"]:
                del self.secrets["api_keys"][provider]
                removed = True
                self.logger.info(f"Removed API key for provider '{provider}'.")

        if removed:
            return self.save_secrets()
        else:
            self.logger.debug(f"API key for provider '{provider}' not found, nothing to remove.")
            return True # Operation succeeded as key wasn't there

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge source dict into target dict.
        Modifies target in place and also returns it.
        """
        for key, value in source.items():
            if isinstance(value, dict):
                # Get node or create one
                node = target.setdefault(key, {})
                if isinstance(node, dict):
                     self._deep_update(node, value)
                else:
                     # If target's key exists but isn't a dict, overwrite it
                     target[key] = value
            else:
                target[key] = value
        return target

    def reset_to_defaults(self) -> bool:
        """Reset configuration (excluding secrets) to defaults and save."""
        self.logger.warning("Resetting configuration to default values.")
        with self._lock:
            self.config = self._get_default_config()
        return self.save_config()

# Example usage (if run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Create instance
    config_manager = ConfigurationManager.get_instance()

    # --- Test Basic Get/Set ---
    print(f"\n--- Initial Values ---")
    print(f"Default Provider: {config_manager.get_config('api.default_provider')}")
    print(f"Ollama Timeout: {config_manager.get_config('api.providers.ollama.timeout')}")
    print(f"Non-existent Key: {config_manager.get_config('some.missing.key', default='Not Found')}")

    print("\n--- Setting Values ---")
    config_manager.set_config('ui.theme', 'light')
    config_manager.set_config('api.providers.ollama.timeout', 180)
    config_manager.set_config('new.nested.setting', True) # Creates nested dicts

    print(f"New UI Theme: {config_manager.get_config('ui.theme')}")
    print(f"New Ollama Timeout: {config_manager.get_config('api.providers.ollama.timeout')}")
    print(f"New Nested Setting: {config_manager.get_config('new.nested.setting')}")

    # --- Test API Keys ---
    print("\n--- API Keys ---")
    print(f"Initial Ollama Key: {config_manager.get_api_key('ollama')}") # Should be None initially
    config_manager.set_api_key('ollama', 'ollama-test-key-123')
    # Avoid printing the actual key here in real logs
    # print(f"Set Ollama Key: {config_manager.get_api_key('ollama')}")
    print(f"Ollama Key Set: {'Yes' if config_manager.get_api_key('ollama') else 'No'}")
    config_manager.remove_api_key('ollama')
    print(f"Ollama Key After Removal: {config_manager.get_api_key('ollama')}")

    # --- Test Reset ---
    # print("\n--- Resetting Config ---")
    # config_manager.reset_to_defaults()
    # print(f"UI Theme after reset: {config_manager.get_config('ui.theme')}")
    # print(f"Ollama Timeout after reset: {config_manager.get_config('api.providers.ollama.timeout')}")

    print("\n--- Test Complete ---")
