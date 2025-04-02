#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Manager for miniManus

This module implements the Configuration Manager component, which is responsible for
loading and persisting user configurations, managing API keys and endpoints,
providing defaults appropriate for mobile environment, and validating configuration changes.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Set
from pathlib import Path

logger = logging.getLogger("miniManus.ConfigManager")

class ConfigurationManager:
    """
    ConfigurationManager handles all configuration-related operations.
    
    It handles:
    - Loading and saving configuration from/to storage
    - Providing default configurations
    - Validating configuration changes
    - Secure storage of sensitive information (API keys)
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'ConfigurationManager':
        """Get or create the singleton instance of ConfigurationManager."""
        if cls._instance is None:
            cls._instance = ConfigurationManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the ConfigurationManager."""
        if ConfigurationManager._instance is not None:
            raise RuntimeError("ConfigurationManager is a singleton. Use get_instance() instead.")
        
        self.logger = logger
        
        # Determine configuration directory based on Termux environment
        self.config_dir = os.path.join(os.environ.get('HOME', ''), '.config', 'minimanus')
        self.config_file = os.path.join(self.config_dir, 'config.json')
        self.secrets_file = os.path.join(self.config_dir, 'secrets.json')
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Initialize configuration with defaults
        self.config = self._get_default_config()
        self.secrets = {}
        
        # Load existing configuration if available
        self._load_config()
        self._load_secrets()
        
        self.logger.info("ConfigurationManager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns:
            Dictionary containing default configuration
        """
        return {
            "general": {
                "log_level": "INFO",
                "theme": "dark",  # Default to dark theme for battery efficiency
                "max_history": 100,  # Limit history for memory efficiency
            },
            "api": {
                "default_provider": "openrouter",
                "providers": {
                    "openrouter": {
                        "enabled": True,
                        "base_url": "https://openrouter.ai/api/v1",
                        "timeout": 30,
                        "default_model": "gpt-3.5-turbo",
                    },
                    "deepseek": {
                        "enabled": True,
                        "base_url": "https://api.deepseek.com/v1",
                        "timeout": 30,
                        "default_model": "deepseek-chat",
                    },
                    "anthropic": {
                        "enabled": True,
                        "base_url": "https://api.anthropic.com/v1",
                        "timeout": 30,
                        "default_model": "claude-instant-1",
                    },
                    "ollama": {
                        "enabled": True,
                        "base_url": "http://localhost:11434/api",
                        "timeout": 60,
                        "default_model": "llama2",
                    },
                    "litellm": {
                        "enabled": True,
                        "base_url": "http://localhost:8000/v1",
                        "timeout": 30,
                    }
                },
                "cache": {
                    "enabled": True,
                    "max_size_mb": 50,  # Limit cache size for storage efficiency
                    "ttl_seconds": 86400,  # 24 hours
                }
            },
            "ui": {
                "max_output_lines": 1000,  # Limit output lines for memory efficiency
                "show_typing_animation": True,
                "compact_mode": False,
                "font_size": "medium",
            },
            "resources": {
                "max_memory_percent": 40,  # Stay under 50% to avoid OOM killer
                "max_cpu_percent": 80,
                "background_processing": True,
                "low_power_mode": False,
            }
        }
    
    def _load_config(self) -> None:
        """Load configuration from file if it exists."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Update default config with loaded values
                    self._deep_update(self.config, loaded_config)
                self.logger.info("Configuration loaded successfully")
            else:
                self.logger.info("No configuration file found, using defaults")
                self.save_config()  # Create default config file
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
    
    def _load_secrets(self) -> None:
        """Load secrets from file if it exists."""
        try:
            if os.path.exists(self.secrets_file):
                with open(self.secrets_file, 'r') as f:
                    self.secrets = json.load(f)
                self.logger.info("Secrets loaded successfully")
            else:
                self.logger.info("No secrets file found, creating empty one")
                self.secrets = {"api_keys": {}}
                self.save_secrets()  # Create empty secrets file
        except Exception as e:
            self.logger.error(f"Error loading secrets: {str(e)}")
            self.secrets = {"api_keys": {}}
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def save_secrets(self) -> bool:
        """
        Save current secrets to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory has proper permissions
            os.chmod(self.config_dir, 0o700)  # Only user can access
            
            with open(self.secrets_file, 'w') as f:
                json.dump(self.secrets, f, indent=2)
            
            # Set restrictive permissions on secrets file
            os.chmod(self.secrets_file, 0o600)  # Only user can read/write
            
            self.logger.info("Secrets saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error saving secrets: {str(e)}")
            return False
    
    def get_config(self, path: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value at specified path.
        
        Args:
            path: Dot-separated path to configuration value (e.g., "api.default_provider")
            default: Default value to return if path not found
            
        Returns:
            Configuration value or default if not found
        """
        if path is None:
            return self.config
        
        # Add debug logging for model configs
        if "default_model" in path:
            self.logger.debug(f"Getting config for path: {path}, default: {default}")
        
        current = self.config
        for key in path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                if "default_model" in path:
                    self.logger.debug(f"Path {path} not found, returning default: {default}")
                return default
        
        if "default_model" in path:
            self.logger.debug(f"Returning config value for {path}: {current}")
        return current
    
    def set_config(self, path: str, value: Any) -> bool:
        """
        Set configuration value at specified path.
        
        Args:
            path: Dot-separated path to configuration value
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        if not path:
            return False
        
        # Add debug logging for model configs
        if "default_model" in path:
            self.logger.info(f"Setting config path: {path} to value: {value}")
        
        keys = path.split('.')
        current = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        
        # Save the updated configuration
        return self.save_config()
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for specified provider.
        
        Args:
            provider: Name of the provider
            
        Returns:
            API key or None if not found
        """
        return self.secrets.get("api_keys", {}).get(provider)
    
    def set_api_key(self, provider: str, api_key: str) -> bool:
        """
        Set API key for specified provider.
        
        Args:
            provider: Name of the provider
            api_key: API key to set
            
        Returns:
            True if successful, False otherwise
        """
        if "api_keys" not in self.secrets:
            self.secrets["api_keys"] = {}
        
        self.secrets["api_keys"][provider] = api_key
        return self.save_secrets()
    
    def remove_api_key(self, provider: str) -> bool:
        """
        Remove API key for specified provider.
        
        Args:
            provider: Name of the provider
            
        Returns:
            True if successful, False otherwise
        """
        if provider in self.secrets.get("api_keys", {}):
            del self.secrets["api_keys"][provider]
            return self.save_secrets()
        return True  # Key didn't exist, so removal is technically successful
    
    def validate_config(self, config: Dict[str, Any] = None) -> List[str]:
        """
        Validate configuration for errors.
        
        Args:
            config: Configuration to validate, or current config if None
            
        Returns:
            List of error messages, empty if no errors
        """
        errors = []
        config = config or self.config
        
        # Validate general settings
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.get("general", {}).get("log_level") not in log_levels:
            errors.append(f"Invalid log_level. Must be one of: {', '.join(log_levels)}")
        
        # Validate API settings
        providers = config.get("api", {}).get("providers", {})
        default_provider = config.get("api", {}).get("default_provider")
        
        if default_provider and default_provider not in providers:
            errors.append(f"Default provider '{default_provider}' not found in configured providers")
        
        for provider, settings in providers.items():
            if "base_url" not in settings:
                errors.append(f"Missing base_url for provider '{provider}'")
            
            if "timeout" in settings and not isinstance(settings["timeout"], (int, float)):
                errors.append(f"Invalid timeout for provider '{provider}'. Must be a number")
        
        # Validate resource settings
        resources = config.get("resources", {})
        
        if "max_memory_percent" in resources:
            mem_percent = resources["max_memory_percent"]
            if not isinstance(mem_percent, (int, float)) or mem_percent <= 0 or mem_percent > 90:
                errors.append("max_memory_percent must be between 1 and 90")
        
        if "max_cpu_percent" in resources:
            cpu_percent = resources["max_cpu_percent"]
            if not isinstance(cpu_percent, (int, float)) or cpu_percent <= 0 or cpu_percent > 100:
                errors.append("max_cpu_percent must be between 1 and 100")
        
        return errors
    
    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to default values.
        
        Returns:
            True if successful, False otherwise
        """
        self.config = self._get_default_config()
        return self.save_config()
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep update target dict with source dict.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    config_manager = ConfigurationManager.get_instance()
    
    # Get and print a configuration value
    default_provider = config_manager.get_config("api.default_provider")
    print(f"Default provider: {default_provider}")
    
    # Set a configuration value
    config_manager.set_config("ui.font_size", "large")
    
    # Set an API key
    config_manager.set_api_key("openrouter", "demo-api-key")
    
    # Get and print the API key
    api_key = config_manager.get_api_key("openrouter")
    print(f"OpenRouter API key: {api_key}")
