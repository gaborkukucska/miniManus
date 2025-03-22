#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UI Manager for miniManus

This module implements the UI Manager component, which coordinates all UI interactions,
manages UI state, handles user input and output, and provides a responsive mobile interface.
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
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager

logger = logging.getLogger("miniManus.UIManager")

class UITheme(Enum):
    """UI theme options."""
    LIGHT = auto()
    DARK = auto()
    SYSTEM = auto()

class UIManager:
    """
    UIManager coordinates all UI interactions for miniManus.
    
    It handles:
    - UI state management
    - User input and output
    - Theme and appearance settings
    - Mobile responsiveness
    - UI component coordination
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'UIManager':
        """Get or create the singleton instance of UIManager."""
        if cls._instance is None:
            cls._instance = UIManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the UIManager."""
        if UIManager._instance is not None:
            raise RuntimeError("UIManager is a singleton. Use get_instance() instead.")
        
        self.logger = logger
        self.event_bus = EventBus.get_instance()
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        
        # UI settings
        self.theme = UITheme[self.config_manager.get_config(
            "ui.theme", 
            UITheme.SYSTEM.name
        )]
        
        self.font_size = self.config_manager.get_config("ui.font_size", 14)
        self.enable_animations = self.config_manager.get_config("ui.enable_animations", True)
        self.compact_mode = self.config_manager.get_config("ui.compact_mode", False)
        
        # UI components
        self.components = {}
        
        # UI state
        self.state = {
            "current_view": "chat",
            "is_processing": False,
            "notification_count": 0,
            "error_count": 0,
        }
        
        # Register event handlers
        self.event_bus.subscribe("ui.theme_changed", self._handle_theme_changed)
        self.event_bus.subscribe("ui.font_size_changed", self._handle_font_size_changed)
        self.event_bus.subscribe("ui.animations_toggled", self._handle_animations_toggled)
        self.event_bus.subscribe("ui.compact_mode_toggled", self._handle_compact_mode_toggled)
        self.event_bus.subscribe("ui.view_changed", self._handle_view_changed)
        self.event_bus.subscribe("ui.notification", self._handle_notification)
        self.event_bus.subscribe("ui.error", self._handle_error)
        
        self.logger.info("UIManager initialized")
    
    def register_component(self, name: str, component: Any) -> None:
        """
        Register a UI component.
        
        Args:
            name: Name of the component
            component: Component instance
        """
        self.components[name] = component
        self.logger.debug(f"Registered UI component: {name}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a UI component.
        
        Args:
            name: Name of the component
            
        Returns:
            Component instance or None if not registered
        """
        return self.components.get(name)
    
    def set_theme(self, theme: UITheme) -> None:
        """
        Set the UI theme.
        
        Args:
            theme: Theme to set
        """
        if theme != self.theme:
            self.theme = theme
            self.config_manager.set_config("ui.theme", theme.name)
            self.event_bus.publish_event("ui.theme_changed", {"theme": theme.name})
            self.logger.info(f"Theme changed to {theme.name}")
    
    def set_font_size(self, size: int) -> None:
        """
        Set the UI font size.
        
        Args:
            size: Font size to set
        """
        if size != self.font_size:
            self.font_size = size
            self.config_manager.set_config("ui.font_size", size)
            self.event_bus.publish_event("ui.font_size_changed", {"size": size})
            self.logger.info(f"Font size changed to {size}")
    
    def toggle_animations(self, enable: bool) -> None:
        """
        Toggle UI animations.
        
        Args:
            enable: Whether to enable animations
        """
        if enable != self.enable_animations:
            self.enable_animations = enable
            self.config_manager.set_config("ui.enable_animations", enable)
            self.event_bus.publish_event("ui.animations_toggled", {"enabled": enable})
            self.logger.info(f"Animations {'enabled' if enable else 'disabled'}")
    
    def toggle_compact_mode(self, enable: bool) -> None:
        """
        Toggle UI compact mode.
        
        Args:
            enable: Whether to enable compact mode
        """
        if enable != self.compact_mode:
            self.compact_mode = enable
            self.config_manager.set_config("ui.compact_mode", enable)
            self.event_bus.publish_event("ui.compact_mode_toggled", {"enabled": enable})
            self.logger.info(f"Compact mode {'enabled' if enable else 'disabled'}")
    
    def change_view(self, view: str) -> None:
        """
        Change the current UI view.
        
        Args:
            view: View to change to
        """
        if view != self.state["current_view"]:
            self.state["current_view"] = view
            self.event_bus.publish_event("ui.view_changed", {"view": view})
            self.logger.info(f"View changed to {view}")
    
    def show_notification(self, message: str, level: str = "info", duration: int = 3000) -> None:
        """
        Show a notification.
        
        Args:
            message: Notification message
            level: Notification level (info, warning, error)
            duration: Duration in milliseconds
        """
        self.state["notification_count"] += 1
        self.event_bus.publish_event("ui.notification", {
            "message": message,
            "level": level,
            "duration": duration,
        })
        self.logger.debug(f"Notification shown: {message}")
    
    def show_error(self, message: str, details: Optional[str] = None) -> None:
        """
        Show an error message.
        
        Args:
            message: Error message
            details: Error details
        """
        self.state["error_count"] += 1
        self.event_bus.publish_event("ui.error", {
            "message": message,
            "details": details,
        })
        self.logger.debug(f"Error shown: {message}")
    
    def set_processing_state(self, is_processing: bool) -> None:
        """
        Set the processing state.
        
        Args:
            is_processing: Whether the system is processing
        """
        if is_processing != self.state["is_processing"]:
            self.state["is_processing"] = is_processing
            self.event_bus.publish_event("ui.processing_state_changed", {"is_processing": is_processing})
            self.logger.debug(f"Processing state changed to {is_processing}")
    
    def _handle_theme_changed(self, event: Dict[str, Any]) -> None:
        """
        Handle theme changed event.
        
        Args:
            event: Event data
        """
        theme_name = event.get("theme")
        try:
            self.theme = UITheme[theme_name]
            self.logger.debug(f"Theme updated to {theme_name}")
        except (KeyError, ValueError):
            self.logger.warning(f"Invalid theme: {theme_name}")
    
    def _handle_font_size_changed(self, event: Dict[str, Any]) -> None:
        """
        Handle font size changed event.
        
        Args:
            event: Event data
        """
        size = event.get("size")
        if isinstance(size, (int, float)):
            self.font_size = size
            self.logger.debug(f"Font size updated to {size}")
    
    def _handle_animations_toggled(self, event: Dict[str, Any]) -> None:
        """
        Handle animations toggled event.
        
        Args:
            event: Event data
        """
        enabled = event.get("enabled")
        if isinstance(enabled, bool):
            self.enable_animations = enabled
            self.logger.debug(f"Animations {'enabled' if enabled else 'disabled'}")
    
    def _handle_compact_mode_toggled(self, event: Dict[str, Any]) -> None:
        """
        Handle compact mode toggled event.
        
        Args:
            event: Event data
        """
        enabled = event.get("enabled")
        if isinstance(enabled, bool):
            self.compact_mode = enabled
            self.logger.debug(f"Compact mode {'enabled' if enabled else 'disabled'}")
    
    def _handle_view_changed(self, event: Dict[str, Any]) -> None:
        """
        Handle view changed event.
        
        Args:
            event: Event data
        """
        view = event.get("view")
        if isinstance(view, str):
            self.state["current_view"] = view
            self.logger.debug(f"View updated to {view}")
    
    def _handle_notification(self, event: Dict[str, Any]) -> None:
        """
        Handle notification event.
        
        Args:
            event: Event data
        """
        self.state["notification_count"] += 1
        self.logger.debug(f"Notification received: {event.get('message')}")
    
    def _handle_error(self, event: Dict[str, Any]) -> None:
        """
        Handle error event.
        
        Args:
            event: Event data
        """
        self.state["error_count"] += 1
        self.logger.debug(f"Error received: {event.get('message')}")
    
    def startup(self) -> None:
        """Start the UI manager."""
        self.logger.info("UIManager started")
    
    def shutdown(self) -> None:
        """Stop the UI manager."""
        self.logger.info("UIManager stopped")

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    error_handler = ErrorHandler.get_instance()
    
    config_manager = ConfigurationManager.get_instance()
    
    # Initialize UIManager
    ui_manager = UIManager.get_instance()
    ui_manager.startup()
    
    # Example usage
    ui_manager.set_theme(UITheme.DARK)
    ui_manager.set_font_size(16)
    ui_manager.toggle_animations(True)
    ui_manager.toggle_compact_mode(False)
    ui_manager.change_view("settings")
    ui_manager.show_notification("This is a test notification")
    ui_manager.show_error("This is a test error")
    
    # Shutdown
    ui_manager.shutdown()
    event_bus.shutdown()
