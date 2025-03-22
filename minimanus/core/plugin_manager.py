#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plugin Manager for miniManus

This module implements the Plugin Manager component, which supports extensibility through plugins,
manages plugin lifecycle and dependencies, enforces resource limits for plugins,
and provides isolation for plugin failures.
"""

import os
import sys
import logging
import importlib
import importlib.util
import inspect
import threading
import json
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Type
from enum import Enum, auto
from pathlib import Path

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.resource_monitor import ResourceMonitor, ResourceType, ResourceWarningLevel
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.resource_monitor import ResourceMonitor, ResourceType, ResourceWarningLevel

logger = logging.getLogger("miniManus.PluginManager")

class PluginState(Enum):
    """States of a plugin."""
    UNLOADED = auto()  # Plugin is not loaded
    LOADED = auto()    # Plugin is loaded but not initialized
    ENABLED = auto()   # Plugin is loaded and initialized
    DISABLED = auto()  # Plugin is loaded but disabled
    ERROR = auto()     # Plugin encountered an error

class PluginInfo:
    """Information about a plugin."""
    
    def __init__(self, name: str, version: str, description: str, author: str,
                dependencies: List[str], path: str):
        """
        Initialize plugin information.
        
        Args:
            name: Name of the plugin
            version: Version of the plugin
            description: Description of the plugin
            author: Author of the plugin
            dependencies: List of plugin dependencies
            path: Path to the plugin directory or file
        """
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies
        self.path = path
        self.state = PluginState.UNLOADED
        self.error = None
        self.module = None
        self.instance = None

class PluginManager:
    """
    PluginManager handles plugin loading, initialization, and lifecycle management.
    
    It handles:
    - Plugin discovery and loading
    - Plugin dependency resolution
    - Plugin initialization and cleanup
    - Resource limit enforcement for plugins
    - Isolation of plugin failures
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'PluginManager':
        """Get or create the singleton instance of PluginManager."""
        if cls._instance is None:
            cls._instance = PluginManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the PluginManager."""
        if PluginManager._instance is not None:
            raise RuntimeError("PluginManager is a singleton. Use get_instance() instead.")
        
        self.logger = logger
        self.event_bus = EventBus.get_instance()
        self.error_handler = ErrorHandler.get_instance()
        self.resource_monitor = ResourceMonitor.get_instance()
        
        # Plugin directories
        self.plugin_dirs = []
        
        # Plugin registry
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugins_lock = threading.RLock()
        
        # Plugin interface class (will be set later)
        self.plugin_interface = None
        
        # Resource limits for plugins
        self.resource_limits = {
            "memory_mb": 50,  # Maximum memory usage per plugin
            "cpu_percent": 20,  # Maximum CPU usage per plugin
        }
        
        self.logger.info("PluginManager initialized")
    
    def set_plugin_interface(self, interface_class: Type) -> None:
        """
        Set the plugin interface class that plugins must implement.
        
        Args:
            interface_class: Class that defines the plugin interface
        """
        self.plugin_interface = interface_class
        self.logger.debug(f"Plugin interface set to {interface_class.__name__}")
    
    def add_plugin_directory(self, directory: str) -> None:
        """
        Add a directory to search for plugins.
        
        Args:
            directory: Directory path to add
        """
        if os.path.isdir(directory) and directory not in self.plugin_dirs:
            self.plugin_dirs.append(directory)
            self.logger.debug(f"Added plugin directory: {directory}")
    
    def discover_plugins(self) -> List[PluginInfo]:
        """
        Discover plugins in the registered plugin directories.
        
        Returns:
            List of discovered plugin information
        """
        discovered_plugins = []
        
        for plugin_dir in self.plugin_dirs:
            self.logger.debug(f"Searching for plugins in {plugin_dir}")
            
            # Look for Python modules and packages
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                
                # Check if it's a Python file
                if item.endswith('.py') and os.path.isfile(item_path):
                    plugin_name = item[:-3]  # Remove .py extension
                    plugin_info = self._load_plugin_info(item_path, plugin_name)
                    if plugin_info:
                        discovered_plugins.append(plugin_info)
                
                # Check if it's a Python package
                elif os.path.isdir(item_path) and os.path.isfile(os.path.join(item_path, '__init__.py')):
                    plugin_info = self._load_plugin_info(item_path, item)
                    if plugin_info:
                        discovered_plugins.append(plugin_info)
        
        self.logger.info(f"Discovered {len(discovered_plugins)} plugins")
        return discovered_plugins
    
    def _load_plugin_info(self, path: str, name: str) -> Optional[PluginInfo]:
        """
        Load plugin information from a plugin file or directory.
        
        Args:
            path: Path to the plugin file or directory
            name: Name of the plugin
            
        Returns:
            PluginInfo object or None if not a valid plugin
        """
        try:
            # Check for plugin metadata file
            metadata_path = os.path.join(path, 'plugin.json') if os.path.isdir(path) else None
            
            if metadata_path and os.path.isfile(metadata_path):
                # Load metadata from JSON file
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                return PluginInfo(
                    name=metadata.get('name', name),
                    version=metadata.get('version', '0.1.0'),
                    description=metadata.get('description', ''),
                    author=metadata.get('author', 'Unknown'),
                    dependencies=metadata.get('dependencies', []),
                    path=path
                )
            else:
                # Try to load the module to extract metadata
                spec = importlib.util.spec_from_file_location(
                    name,
                    path if path.endswith('.py') else os.path.join(path, '__init__.py')
                )
                
                if spec is None or spec.loader is None:
                    return None
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check if it's a valid plugin
                if not hasattr(module, 'plugin_info'):
                    return None
                
                info = module.plugin_info
                return PluginInfo(
                    name=info.get('name', name),
                    version=info.get('version', '0.1.0'),
                    description=info.get('description', ''),
                    author=info.get('author', 'Unknown'),
                    dependencies=info.get('dependencies', []),
                    path=path
                )
        
        except Exception as e:
            self.logger.warning(f"Error loading plugin info from {path}: {str(e)}")
            return None
    
    def load_plugin(self, plugin_info: PluginInfo) -> bool:
        """
        Load a plugin module.
        
        Args:
            plugin_info: Information about the plugin to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Check if already loaded
            if plugin_info.state != PluginState.UNLOADED:
                self.logger.warning(f"Plugin {plugin_info.name} is already loaded")
                return True
            
            # Load the module
            if os.path.isdir(plugin_info.path):
                # Load as package
                module_name = os.path.basename(plugin_info.path)
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    os.path.join(plugin_info.path, '__init__.py')
                )
            else:
                # Load as module
                module_name = os.path.basename(plugin_info.path)[:-3]  # Remove .py extension
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    plugin_info.path
                )
            
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load plugin module: {plugin_info.path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    self.plugin_interface is not None and 
                    issubclass(obj, self.plugin_interface) and 
                    obj != self.plugin_interface):
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                raise ValueError(f"No plugin class found in {plugin_info.path}")
            
            # Store module in plugin info
            plugin_info.module = module
            plugin_info.state = PluginState.LOADED
            
            # Register plugin
            with self.plugins_lock:
                self.plugins[plugin_info.name] = plugin_info
            
            self.logger.info(f"Loaded plugin: {plugin_info.name} v{plugin_info.version}")
            return True
        
        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.PLUGIN, ErrorSeverity.ERROR,
                {"plugin": plugin_info.name, "action": "load"}
            )
            return False
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable a loaded plugin.
        
        Args:
            plugin_name: Name of the plugin to enable
            
        Returns:
            True if enabled successfully, False otherwise
        """
        with self.plugins_lock:
            if plugin_name not in self.plugins:
                self.logger.warning(f"Plugin {plugin_name} not found")
                return False
            
            plugin_info = self.plugins[plugin_name]
            
            # Check if already enabled
            if plugin_info.state == PluginState.ENABLED:
                return True
            
            # Check if in error state
            if plugin_info.state == PluginState.ERROR:
                self.logger.warning(f"Cannot enable plugin {plugin_name} due to previous error")
                return False
            
            # Check if not loaded
            if plugin_info.state == PluginState.UNLOADED:
                if not self.load_plugin(plugin_info):
                    return False
            
            # Check dependencies
            for dependency in plugin_info.dependencies:
                if dependency not in self.plugins:
                    self.logger.warning(f"Plugin {plugin_name} depends on {dependency} which is not loaded")
                    plugin_info.state = PluginState.ERROR
                    plugin_info.error = f"Missing dependency: {dependency}"
                    return False
                
                if self.plugins[dependency].state != PluginState.ENABLED:
                    self.logger.info(f"Enabling dependency {dependency} for plugin {plugin_name}")
                    if not self.enable_plugin(dependency):
                        plugin_info.state = PluginState.ERROR
                        plugin_info.error = f"Failed to enable dependency: {dependency}"
                        return False
        
        try:
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(plugin_info.module):
                if (inspect.isclass(obj) and 
                    self.plugin_interface is not None and 
                    issubclass(obj, self.plugin_interface) and 
                    obj != self.plugin_interface):
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                raise ValueError(f"No plugin class found in {plugin_info.name}")
            
            # Create plugin instance
            plugin_info.instance = plugin_class()
            
            # Initialize plugin
            if hasattr(plugin_info.instance, 'initialize'):
                plugin_info.instance.initialize()
            
            # Update state
            plugin_info.state = PluginState.ENABLED
            
            # Register resource cleanup callback
            self.resource_monitor.register_cleanup_callback(
                ResourceType.MEMORY,
                lambda level: self._handle_resource_warning(plugin_name, ResourceType.MEMORY, level)
            )
            
            self.logger.info(f"Enabled plugin: {plugin_name}")
            
            # Publish event
            self.event_bus.publish_event(
                "plugin.enabled",
                {"plugin": plugin_name, "version": plugin_info.version}
            )
            
            return True
        
        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.PLUGIN, ErrorSeverity.ERROR,
                {"plugin": plugin_name, "action": "enable"}
            )
            return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """
        Disable an enabled plugin.
        
        Args:
            plugin_name: Name of the plugin to disable
            
        Returns:
            True if disabled successfully, False otherwise
        """
        with self.plugins_lock:
            if plugin_name not in self.plugins:
                self.logger.warning(f"Plugin {plugin_name} not found")
                return False
            
            plugin_info = self.plugins[plugin_name]
            
            # Check if already disabled
            if plugin_info.state in (PluginState.LOADED, PluginState.DISABLED, PluginState.UNLOADED):
                return True
            
            # Check for dependent plugins
            dependent_plugins = []
            for name, info in self.plugins.items():
                if name != plugin_name and plugin_name in info.dependencies and info.state == PluginState.ENABLED:
                    dependent_plugins.append(name)
            
            # Disable dependent plugins first
            for dependent in dependent_plugins:
                self.logger.info(f"Disabling dependent plugin {dependent}")
                if not self.disable_plugin(dependent):
                    self.logger.warning(f"Failed to disable dependent plugin {dependent}")
                    # Continue anyway
        
        try:
            # Cleanup plugin
            if plugin_info.instance and hasattr(plugin_info.instance, 'cleanup'):
                plugin_info.instance.cleanup()
            
            # Update state
            plugin_info.state = PluginState.DISABLED
            
            # Unregister resource cleanup callback
            self.resource_monitor.unregister_cleanup_callback(
                ResourceType.MEMORY,
                lambda level: self._handle_resource_warning(plugin_name, ResourceType.MEMORY, level)
            )
            
            self.logger.info(f"Disabled plugin: {plugin_name}")
            
            # Publish event
            self.event_bus.publish_event(
                "plugin.disabled",
                {"plugin": plugin_name}
            )
            
            return True
        
        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            self.error_handler.handle_error(
                e, ErrorCategory.PLUGIN, ErrorSeverity.ERROR,
                {"plugin": plugin_name, "action": "disable"}
            )
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        with self.plugins_lock:
            if plugin_name not in self.plugins:
                self.logger.warning(f"Plugin {plugin_name} not found")
                return False
            
            plugin_info = self.plugins[plugin_name]
            
            # Disable first if enabled
            if plugin_info.state == PluginState.ENABLED:
                if not self.disable_plugin(plugin_name):
                    return False
            
            # Remove from registry
            del self.plugins[plugin_name]
            
            # Remove from sys.modules if possible
            module_name = plugin_info.module.__name__ if plugin_info.module else None
            if module_name and module_name in sys.modules:
                del sys.modules[module_name]
            
            self.logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """
        Get information about a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            PluginInfo object or None if not found
        """
        with self.plugins_lock:
            return self.plugins.get(plugin_name)
    
    def get_all_plugins(self) -> Dict[str, PluginInfo]:
        """
        Get information about all plugins.
        
        Returns:
            Dictionary mapping plugin names to PluginInfo objects
        """
        with self.plugins_lock:
            return dict(self.plugins)
    
    def _handle_resource_warning(self, plugin_name: str, resource_type: ResourceType,
                               warning_level: ResourceWarningLevel) -> None:
        """
        Handle resource warning for a plugin.
        
        Args:
            plugin_name: Name of the plugin
            resource_type: Type of resource with warning
            warning_level: Warning level
        """
        self.logger.warning(f"Resource warning for plugin {plugin_name}: {resource_type.name} {warning_level.name}")
        
        # For critical and emergency warnings, disable the plugin
        if warning_level in (ResourceWarningLevel.CRITICAL, ResourceWarningLevel.EMERGENCY):
            self.logger.warning(f"Disabling plugin {plugin_name} due to {resource_type.name} usage")
            self.disable_plugin(plugin_name)
    
    def startup(self) -> None:
        """Start the plugin manager."""
        # Create default plugin directories if they don't exist
        default_dirs = [
            os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus', 'plugins'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'plugins')
        ]
        
        for directory in default_dirs:
            os.makedirs(directory, exist_ok=True)
            self.add_plugin_directory(directory)
        
        self.logger.info("PluginManager started")
    
    def shutdown(self) -> None:
        """Stop the plugin manager and cleanup plugins."""
        # Disable all enabled plugins
        with self.plugins_lock:
            enabled_plugins = [name for name, info in self.plugins.items() if info.state == PluginState.ENABLED]
        
        for plugin_name in enabled_plugins:
            self.disable_plugin(plugin_name)
        
        self.logger.info("PluginManager stopped")

# Example plugin interface
class PluginInterface:
    """Base interface that all plugins must implement."""
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources used by the plugin."""
        pass

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    error_handler = ErrorHandler.get_instance()
    
    resource_monitor = ResourceMonitor.get_instance()
    resource_monitor.startup()
    
    # Initialize PluginManager
    plugin_manager = PluginManager.get_instance()
    plugin_manager.set_plugin_interface(PluginInterface)
    plugin_manager.startup()
    
    # Create a test plugin directory
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_plugins')
    os.makedirs(test_dir, exist_ok=True)
    plugin_manager.add_plugin_directory(test_dir)
    
    # Discover plugins
    plugins = plugin_manager.discover_plugins()
    print(f"Discovered {len(plugins)} plugins")
    
    # Shutdown
    plugin_manager.shutdown()
    resource_monitor.shutdown()
    event_bus.shutdown()
