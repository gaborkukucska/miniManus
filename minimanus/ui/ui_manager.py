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
import json
import threading
import http.server
import socketserver
import time
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum, auto
from pathlib import Path

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
    from ..api.api_manager import APIManager, APIProvider, APIRequestType
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager
    from api.api_manager import APIManager, APIProvider, APIRequestType

logger = logging.getLogger("miniManus.UIManager")

class UITheme(Enum):
    """UI themes."""
    LIGHT = auto()
    DARK = auto()
    SYSTEM = auto()

class UIManager:
    """
    UIManager coordinates all UI interactions for miniManus.
    
    It handles:
    - UI state management
    - User input and output
    - Theme management
    - UI component coordination
    - Web server for UI rendering
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
        
        # Web server
        self.server = None
        self.server_thread = None
        self.port = self.config_manager.get_config("ui.port", 8080)
        self.host = self.config_manager.get_config("ui.host", "localhost")
        self.httpd = None
        
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
    
    def _get_static_file_path(self, path: str) -> str:
        """
        Get the absolute path to a static file.
        
        Args:
            path: Relative path to the file
            
        Returns:
            Absolute path to the file
        """
        # Find the static directory
        module_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join(module_dir, "..", "..", "static")
        
        # If static directory doesn't exist at the expected location, try alternative locations
        if not os.path.isdir(static_dir):
            static_dir = os.path.join(module_dir, "..", "static")
            if not os.path.isdir(static_dir):
                static_dir = os.path.join(os.path.dirname(module_dir), "static")
                if not os.path.isdir(static_dir):
                    # Last resort, create a basic static dir
                    os.makedirs(static_dir, exist_ok=True)
                    self._create_default_static_files(static_dir)
        
        return os.path.join(static_dir, path.lstrip('/'))
    
    def _create_default_static_files(self, static_dir: str) -> None:
        """
        Create default static files if they don't exist.
        
        Args:
            static_dir: Directory to create files in
        """
        # Create default index.html
        index_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>miniManus</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            background-color: #f5f5f5;
        }
        header {
            background-color: #4a90e2;
            color: white;
            padding: 1rem;
            text-align: center;
        }
        main {
            flex: 1;
            padding: 1rem;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 70vh;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background-color: white;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            background-color: #f9f9f9;
        }
        .message {
            margin-bottom: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 18px;
            max-width: 80%;
        }
        .user-message {
            background-color: #4a90e2;
            color: white;
            align-self: flex-end;
            margin-left: auto;
            text-align: right;
        }
        .assistant-message {
            background-color: #e5e5ea;
            color: black;
            align-self: flex-start;
        }
        .chat-input {
            display: flex;
            padding: 0.5rem;
            border-top: 1px solid #ddd;
            background-color: white;
        }
        .chat-input input {
            flex: 1;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 18px;
            margin-right: 0.5rem;
        }
        .chat-input button {
            padding: 0.5rem 1rem;
            background-color: #4a90e2;
            color: white;
            border: none;
            border-radius: 18px;
            cursor: pointer;
        }
        .chat-messages-container {
            display: flex;
            flex-direction: column;
        }
        .timestamp {
            font-size: 0.8em;
            color: #888;
            margin-top: 0.2rem;
        }
        .settings-panel {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .settings-panel h2 {
            margin-top: 0;
        }
        .settings-group {
            margin-bottom: 1rem;
        }
        .settings-group h3 {
            margin-bottom: 0.5rem;
        }
        .setting-item {
            margin-bottom: 0.5rem;
            display: flex;
            flex-direction: column;
        }
        .setting-item label {
            margin-bottom: 0.3rem;
        }
        .models-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
        }
        .model-card {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .model-card:hover {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .model-card h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        .model-card p {
            margin: 0;
            font-size: 0.9rem;
            color: #555;
        }
        .navigation {
            display: flex;
            justify-content: space-between;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 1rem;
        }
        .nav-item {
            flex: 1;
            text-align: center;
            padding: 0.8rem;
            cursor: pointer;
        }
        .nav-item.active {
            background-color: #4a90e2;
            color: white;
        }
        .notification {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.8rem 1rem;
            border-radius: 4px;
            background-color: #4a90e2;
            color: white;
            z-index: 1000;
            animation: fadeOut 0.3s ease-in-out 2.7s forwards;
        }
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }
    </style>
</head>
<body>
    <header>
        <h1>miniManus</h1>
    </header>
    <main>
        <div class="navigation">
            <div class="nav-item active" data-view="chat">Chat</div>
            <div class="nav-item" data-view="models">Models</div>
            <div class="nav-item" data-view="settings">Settings</div>
            <div class="nav-item" data-view="info">Info</div>
        </div>
        
        <div id="chat-view">
            <div class="chat-container">
                <div class="chat-messages" id="chat-messages">
                    <div class="chat-messages-container">
                        <div class="message assistant-message">
                            <p>Hello! I'm miniManus. How can I help you today?</p>
                            <div class="timestamp">Just now</div>
                        </div>
                    </div>
                </div>
                <div class="chat-input">
                    <input type="text" id="message-input" placeholder="Type your message...">
                    <button id="send-button">Send</button>
                </div>
            </div>
        </div>
        
        <div id="models-view" style="display: none;">
            <h2>Available Models</h2>
            <div class="models-list" id="models-list">
                <div class="model-card">
                    <h3>Loading models...</h3>
                    <p>Please wait while we fetch available models</p>
                </div>
            </div>
        </div>
        
        <div id="settings-view" style="display: none;">
            <div class="settings-panel">
                <h2>Settings</h2>
                
                <div class="settings-group">
                    <h3>Appearance</h3>
                    <div class="setting-item">
                        <label for="theme-select">Theme</label>
                        <select id="theme-select">
                            <option value="LIGHT">Light</option>
                            <option value="DARK" selected>Dark</option>
                            <option value="SYSTEM">System</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label for="font-size">Font Size</label>
                        <input type="range" id="font-size" min="12" max="24" value="16">
                        <span id="font-size-value">16px</span>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="animations-toggle" checked>
                            Enable Animations
                        </label>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="compact-mode-toggle">
                            Compact Mode
                        </label>
                    </div>
                </div>
                
                <div class="settings-group">
                    <h3>API Settings</h3>
                    <div class="setting-item">
                        <label for="api-provider">Default Provider</label>
                        <select id="api-provider">
                            <option value="openrouter" selected>OpenRouter</option>
                            <option value="deepseek">DeepSeek</option>
                            <option value="anthropic">Anthropic</option>
                            <option value="ollama">Ollama</option>
                            <option value="litellm">LiteLLM</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label for="api-key">API Key</label>
                        <input type="password" id="api-key" placeholder="Enter API key">
                    </div>
                </div>
            </div>
        </div>
        
        <div id="info-view" style="display: none;">
            <div class="settings-panel">
                <h2>About miniManus</h2>
                <p>miniManus is a mobile-focused framework that runs on Linux in Termux for Android phones.</p>
                <p>Version: 0.1.0</p>
                <p>Created by: miniManus Team</p>
                
                <h3>Support</h3>
                <p>For help and support, please visit:</p>
                <p><a href="https://github.com/yourusername/miniManus" target="_blank">GitHub Repository</a></p>
            </div>
        </div>
    </main>
    
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const messagesContainer = document.getElementById('chat-messages');
            const messageInput = document.getElementById('message-input');
            const sendButton = document.getElementById('send-button');
            const navItems = document.querySelectorAll('.nav-item');
            const views = {
                chat: document.getElementById('chat-view'),
                models: document.getElementById('models-view'),
                settings: document.getElementById('settings-view'),
                info: document.getElementById('info-view')
            };
            
            // Navigation
            navItems.forEach(item => {
                item.addEventListener('click', () => {
                    const view = item.getAttribute('data-view');
                    
                    // Update active nav item
                    navItems.forEach(navItem => navItem.classList.remove('active'));
                    item.classList.add('active');
                    
                    // Show selected view, hide others
                    Object.keys(views).forEach(key => {
                        views[key].style.display = key === view ? 'block' : 'none';
                    });
                });
            });
            
            // Chat functionality
            function addMessage(content, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
                
                const messagePara = document.createElement('p');
                messagePara.textContent = content;
                
                const timestamp = document.createElement('div');
                timestamp.className = 'timestamp';
                timestamp.textContent = 'Just now';
                
                messageDiv.appendChild(messagePara);
                messageDiv.appendChild(timestamp);
                
                const container = document.createElement('div');
                container.className = 'chat-messages-container';
                container.appendChild(messageDiv);
                
                messagesContainer.appendChild(container);
                
                // Scroll to bottom
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                // Add user message to UI
                addMessage(message, true);
                
                // Clear input
                messageInput.value = '';
                
                // Send to backend
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message })
                })
                .then(response => response.json())
                .then(data => {
                    // Add assistant response to UI
                    addMessage(data.response);
                })
                .catch(error => {
                    console.error('Error:', error);
                    addMessage('Sorry, there was an error processing your request.');
                });
            }
            
            // Event listeners
            sendButton.addEventListener('click', sendMessage);
            
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Settings event listeners
            document.getElementById('theme-select').addEventListener('change', function() {
                // Implementation would send theme change to backend
                console.log('Theme changed to:', this.value);
            });
            
            document.getElementById('font-size').addEventListener('input', function() {
                document.getElementById('font-size-value').textContent = this.value + 'px';
                // Implementation would send font size change to backend
                console.log('Font size changed to:', this.value);
            });
            
            document.getElementById('animations-toggle').addEventListener('change', function() {
                // Implementation would send animations toggle to backend
                console.log('Animations toggled:', this.checked);
            });
            
            document.getElementById('compact-mode-toggle').addEventListener('change', function() {
                // Implementation would send compact mode toggle to backend
                console.log('Compact mode toggled:', this.checked);
            });
            
            document.getElementById('api-provider').addEventListener('change', function() {
                // Implementation would send API provider change to backend
                console.log('API provider changed to:', this.value);
            });
            
            document.getElementById('api-key').addEventListener('blur', function() {
                if (this.value) {
                    // Implementation would send API key to backend
                    console.log('API key updated');
                }
            });
            
            // Show a notification function
            function showNotification(message, level = 'info') {
                const notification = document.createElement('div');
                notification.className = `notification ${level}`;
                notification.textContent = message;
                document.body.appendChild(notification);
                
                // Remove after 3 seconds
                setTimeout(() => {
                    notification.remove();
                }, 3000);
            }
            
            // Example notification
            // showNotification('Welcome to miniManus!');
        });
    </script>
</body>
</html>
"""
        with open(os.path.join(static_dir, 'index.html'), 'w') as f:
            f.write(index_html)
        
        self.logger.info(f"Created default static files in {static_dir}")
    
    def startup(self) -> None:
        """Start the UI manager."""
        # Start web server
        try:
            # Enable address reuse to prevent "Address already in use" errors
            socketserver.TCPServer.allow_reuse_address = True
            
            class CustomHandler(http.server.SimpleHTTPRequestHandler):
                """Custom HTTP request handler."""
                
                def __init__(self, *args, **kwargs):
                    # Store reference to UI manager
                    self.ui_manager = UIManager.get_instance()
                    super().__init__(*args, directory=None, **kwargs)
                
                def log_message(self, format, *args):
                    """Override to use our logger."""
                    self.ui_manager.logger.debug(format % args)
                
                def translate_path(self, path):
                    """Override to serve files from our static directory."""
                    return self.ui_manager._get_static_file_path(path)
                
                def do_GET(self):
                    """Handle GET requests."""
                    # Serve API endpoints
                    if self.path.startswith('/api/'):
                        if self.path == '/api/state':
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps(self.ui_manager.state).encode('utf-8'))
                            return
                        elif self.path == '/api/models':
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            
                            # Get model selection interface
                            from minimanus.ui.model_selection import ModelSelectionInterface
                            model_interface = ModelSelectionInterface.get_instance()
                            
                            # Get all models
                            models = model_interface.get_all_models()
                            
                            # Convert to serializable format
                            model_data = [model.to_dict() for model in models]
                            
                            self.wfile.write(json.dumps({"models": model_data}).encode('utf-8'))
                            return
                    
                    # Serve static files
                    try:
                        # If requesting a directory (like /), serve index.html
                        if self.path == '/' or self.path == '':
                            self.path = '/index.html'
                        
                        # Try to serve the file
                        f = open(self.translate_path(self.path), 'rb')
                    except FileNotFoundError:
                        # If file not found, serve index.
