# START OF FILE miniManus-main/minimanus/core/config_manager.py
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
from typing import Dict, Any, Optional, List, Union, Set, Tuple
from pathlib import Path

# Import UITheme locally if needed for default config structure
# This might cause issues if ui_manager isn't importable yet,
# handle potential ImportError during default config creation.
try:
    from ..ui.ui_manager import UITheme
except ImportError:
    # Define a placeholder if UITheme cannot be imported (e.g., during early startup/tests)
    class UITheme: # type: ignore
        DARK = 'dark'
        LIGHT = 'light'
        SYSTEM = 'system'
        # Map names to values if needed by default config logic
        DARK = type('Enum', (), {'name': 'dark'})()
        LIGHT = type('Enum', (), {'name': 'light'})()
        SYSTEM = type('Enum', (), {'name': 'system'})()


logger = logging.getLogger("miniManus.ConfigManager")

# Define default base paths, assuming standard structure relative to this file
# These can be overridden after instantiation if needed (e.g., by __main__.py)
DEFAULT_BASE_DIR = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'minimanus'
DEFAULT_CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'minimanus'
# For consistency with __main__.py, use the XDG_DATA_HOME location for config.
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
        # Define file paths using Path for better cross-platform handling
        default_agent_read_dir = Path.home() / "minimanus_files" # Example default

        # Safely get theme default value
        try:
            theme_default = UITheme.DARK.name.lower()
        except NameError:
            theme_default = 'dark' # Fallback if UITheme wasn't imported

        return {
            "general": {
                "log_level": "INFO",
            },
            "ui": {
                "host": "localhost",
                "port": 8080,
                "theme": theme_default,
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
            # Attempt to save default config, handle potential errors
            if not self.save_config():
                self.logger.error("Failed to save default configuration file.")
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
             self.logger.warning("Using default configuration due to load error.")
             self.config = self._get_default_config() # Reset to defaults
        except Exception as e:
            self.logger.error(f"Error loading configuration from {self.config_file}: {e}", exc_info=True)
            self.logger.warning("Using default configuration due to load error.")
            self.config = self._get_default_config()

    def _load_secrets(self) -> None:
        """Load secrets from secrets_file."""
        if not self.secrets_file.exists():
            self.logger.info(f"Secrets file not found ({self.secrets_file}). Creating empty structure.")
            self.secrets = {"api_keys": {}}
            # Attempt to save default secrets file, handle potential errors
            if not self.save_secrets():
                self.logger.error("Failed to save empty secrets file.")
            return

        try:
            # Check permissions before reading
            if sys.platform != "win32": # Skip permission check on Windows
                 try:
                    file_stat = os.stat(self.secrets_file)
                    # Check if group or others have any permissions
                    if file_stat.st_mode & 0o077:
                        self.logger.warning(f"Secrets file {self.secrets_file} has potentially insecure permissions ({oct(file_stat.st_mode & 0o777)}). Recommended: 600.")
                 except OSError as e:
                     self.logger.warning(f"Could not check permissions for secrets file {self.secrets_file}: {e}")

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
        config_to_save = None # Define outside lock
        try:
            # Ensure directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)

            with self._lock:
                 # Create a deep copy to avoid modifying the live config dict directly during serialization
                 config_to_save = json.loads(json.dumps(self.config)) # Simple deep copy via JSON
                 # Ensure secrets are not accidentally saved in config.json
                 # Check within the copied dictionary
                 api_settings = config_to_save.get('api')
                 if isinstance(api_settings, dict):
                      if 'api_keys' in api_settings:
                         self.logger.warning("Removing 'api_keys' from main config before saving.")
                         del api_settings['api_keys']

            # Perform file I/O outside the lock
            temp_file_path = self.config_file.with_suffix(".tmp")
            self.logger.debug(f"Saving config dictionary to temp file {temp_file_path}. Root keys: {list(config_to_save.keys())}")
            with temp_file_path.open('w', encoding='utf-8') as f:
                 json.dump(config_to_save, f, indent=2, ensure_ascii=False)

            # Atomically replace the old config file
            os.replace(temp_file_path, self.config_file)

            self.logger.debug(f"Configuration saved successfully to {self.config_file}")
            return True
        except json.JSONDecodeError as e: # Catch error during deep copy
             self.logger.error(f"Error creating copy of config for saving: {e}", exc_info=True)
             return False
        except Exception as e:
            self.logger.error(f"Error saving configuration to {self.config_file}: {e}", exc_info=True)
             # Clean up temp file if rename failed
            if 'temp_file_path' in locals() and temp_file_path.exists():
                try: temp_file_path.unlink()
                except OSError: pass
            return False

    def save_secrets(self) -> bool:
        """Save current secrets dictionary to secrets_file with secure permissions."""
        secrets_to_save = None # Define outside lock
        try:
            # Ensure directory exists and has secure permissions
            self.config_dir.mkdir(parents=True, exist_ok=True)
            if sys.platform != "win32":
                try:
                    os.chmod(self.config_dir, 0o700)
                except OSError as e:
                    self.logger.warning(f"Could not set permissions on config directory {self.config_dir}: {e}")

            with self._lock:
                 # Create a deep copy to avoid issues if secrets dict is modified while saving
                 secrets_to_save = json.loads(json.dumps(self.secrets))

            # Write secrets atomically (write to temp file, then rename)
            temp_file_path = self.secrets_file.with_suffix(".tmp")
            self.logger.debug(f"Saving secrets to temp file {temp_file_path}.")
            with temp_file_path.open('w', encoding='utf-8') as f:
                 # Set restrictive permissions *before* writing sensitive data (on platforms that support it)
                 if sys.platform != "win32":
                     try:
                        os.chmod(temp_file_path, 0o600)
                     except OSError as e:
                         self.logger.warning(f"Could not set permissions on temp secrets file {temp_file_path}: {e}")
                 json.dump(secrets_to_save, f, indent=2, ensure_ascii=False)

            # Atomically replace the old secrets file with the new one
            os.replace(temp_file_path, self.secrets_file)
            # Ensure final file has correct permissions (replace might alter them)
            if sys.platform != "win32":
                try:
                    os.chmod(self.secrets_file, 0o600)
                except OSError as e:
                    self.logger.warning(f"Could not set final permissions on secrets file {self.secrets_file}: {e}")

            self.logger.debug(f"Secrets saved successfully to {self.secrets_file}")
            return True
        except json.JSONDecodeError as e: # Catch error during deep copy
             self.logger.error(f"Error creating copy of secrets for saving: {e}", exc_info=True)
             return False
        except Exception as e:
            self.logger.error(f"Error saving secrets to {self.secrets_file}: {e}", exc_info=True)
            # Clean up temp file if rename failed
            if 'temp_file_path' in locals() and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except OSError: pass
            return False

    def get_config(self, path: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            path: Dot-separated path (e.g., "api.providers.ollama.timeout").
                  If None, returns a deep copy of the entire config dictionary (excluding secrets).
            default: Value to return if path is not found.

        Returns:
            The configuration value or the default. Returns a copy if path is None.
        """
        with self._lock:
             if path is None:
                 # Return a deep copy of the whole config, ensuring secrets aren't included
                 try:
                     safe_config = json.loads(json.dumps(self.config)) # Simple deep copy
                     api_settings = safe_config.get('api')
                     if isinstance(api_settings, dict):
                         api_settings.pop('api_keys', None)
                         # Also remove individual provider keys if they exist at this level
                         for p_key in list(api_settings.keys()):
                            if "api_key" in p_key:
                                api_settings.pop(p_key, None)
                     return safe_config
                 except Exception as e:
                     self.logger.error(f"Error creating deep copy of config: {e}", exc_info=True)
                     return {} # Return empty dict on failure

             keys = path.split('.')
             current_level = self.config
             for key in keys:
                 if isinstance(current_level, dict) and key in current_level:
                     current_level = current_level[key]
                 else:
                     return default
             # Return a copy of the value if it's mutable (dict/list) to prevent modification
             if isinstance(current_level, (dict, list)):
                 try:
                    return json.loads(json.dumps(current_level))
                 except Exception: # Fallback for non-JSON serializable types
                     return default # Or maybe copy.deepcopy? For now, default.
             else:
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
        path_lower = path.lower()
        if "api_key" in path_lower or "secret" in path_lower:
             # Allow setting api.providers.xxx.api_key explicitly to empty string
             # This is handled by the specialized API key handler in ui_manager now.
             # Let's be strict here: do not allow setting any api_key via set_config.
             self.logger.error(f"Attempted to set potentially sensitive key '{path}' via set_config. Use set_api_key or remove_api_key instead.")
             return False

        with self._lock:
            keys = path.split('.')
            current_level = self.config

            for i, key in enumerate(keys[:-1]):
                # If a key exists but isn't a dict, or doesn't exist, create it
                node = current_level.get(key)
                if not isinstance(node, dict):
                    self.logger.debug(f"Creating/overwriting intermediate path '{key}' in '{path}'")
                    current_level[key] = {}
                current_level = current_level[key]

            # Set the final value
            final_key = keys[-1]
            if not isinstance(current_level, dict):
                 # This should not happen with the creation logic above, but check defensively
                 self.logger.error(f"Cannot set key '{final_key}' because parent path '{'.'.join(keys[:-1])}' is not a dictionary.")
                 return False

            old_value = current_level.get(final_key, '__NOT_SET__')
            # Avoid logging potentially large values
            value_repr = f"type:{type(value)}" if isinstance(value, (dict, list)) else repr(value)
            old_value_repr = f"type:{type(old_value)}" if isinstance(old_value, (dict, list)) else repr(old_value)

            current_level[final_key] = value
            self.logger.debug(f"Internal config dict updated for '{path}'. Old={old_value_repr}, New={value_repr}")


        # Save the updated configuration
        return self.save_config()

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider from secrets."""
        with self._lock:
            key = self.secrets.get("api_keys", {}).get(provider)
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
        if not api_key: # Prevent setting empty strings via set_api_key, use remove instead
             self.logger.warning(f"Attempted to set empty API key for '{provider}'. Use remove_api_key() to clear.")
             return self.remove_api_key(provider)

        with self._lock:
            # Ensure the api_keys dictionary exists
            if "api_keys" not in self.secrets or not isinstance(self.secrets["api_keys"], dict):
                 self.secrets["api_keys"] = {}
            self.secrets["api_keys"][provider] = api_key
            self.logger.info(f"Set API key for provider '{provider}'.")

        return self.save_secrets()

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


        # Only save if a change was actually made
        if removed:
            return self.save_secrets()
        else:
            return True # Operation "succeeded" as the key wasn't there or couldn't be removed

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge source dict into target dict.
        Modifies target in place and also returns it.
        """
        for key, value in source.items():
            if isinstance(value, dict):
                # Get node or create one if it doesn't exist or isn't a dict
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
    print(f"Ollama Key Set: {'Yes' if config_manager.get_api_key('ollama') else 'No'}")
    config_manager.set_api_key('openrouter', 'sk-or-test-456')
    print(f"OpenRouter Key Set: {'Yes' if config_manager.get_api_key('openrouter') else 'No'}")
    config_manager.remove_api_key('ollama')
    print(f"Ollama Key After Removal: {config_manager.get_api_key('ollama')}")
    print(f"OpenRouter Key After Ollama Removal: {config_manager.get_api_key('openrouter')}")
    # Test removing non-existent key
    config_manager.remove_api_key('deepseek')
    # Test setting empty key (should remove)
    config_manager.set_api_key('openrouter', '')
    print(f"OpenRouter Key After Setting to Empty: {config_manager.get_api_key('openrouter')}")


    # --- Test Reset ---
    print("\n--- Resetting Config ---")
    config_manager.reset_to_defaults()
    print(f"UI Theme after reset: {config_manager.get_config('ui.theme')}")
    print(f"Ollama Timeout after reset: {config_manager.get_config('api.providers.ollama.timeout')}")
    print(f"New Nested Setting after reset: {config_manager.get_config('new.nested.setting', 'Not Found')}")

    print("\n--- Test Complete ---")
# END OF FILE miniManus-main/minimanus/core/config_manager.py
