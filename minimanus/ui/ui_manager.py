#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UI Manager for miniManus

This module implements the UI Manager component, which manages the user interface
for the miniManus framework.
"""

import os
import sys
import logging
import json
import threading
import http.server
import socketserver
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
    """UI themes."""
    LIGHT = auto()
    DARK = auto()
    SYSTEM = auto()

class UIManager:
    """
    UIManager manages the user interface for the miniManus framework.
    
    It handles:
    - UI initialization
    - Theme management
    - UI event handling
    - UI state management
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
        self.theme = UITheme[self.config_manager.get_config("ui.theme", "SYSTEM")]
        self.font_size = self.config_manager.get_config("ui.font_size", 14)
        self.animations_enabled = self.config_manager.get_config("ui.animations_enabled", True)
        
        # UI state
        self.ui_state = {
            "theme": self.theme.name,
            "font_size": self.font_size,
            "animations_enabled": self.animations_enabled,
            "current_view": "chat",
            "is_sidebar_open": False,
            "is_settings_open": False,
        }
        
        # Web server
        self.httpd = None
        self.server_thread = None
        self.server_port = self.config_manager.get_config("ui.server_port", 8080)
        
        # Register event handlers
        self.event_bus.subscribe("ui.theme_changed", self._handle_theme_changed)
        self.event_bus.subscribe("ui.font_size_changed", self._handle_font_size_changed)
        self.event_bus.subscribe("ui.animations_toggled", self._handle_animations_toggled)
        
        self.logger.info("UIManager initialized")
    
    def get_ui_state(self) -> Dict[str, Any]:
        """
        Get current UI state.
        
        Returns:
            Dictionary with UI state information
        """
        return self.ui_state.copy()
    
    def set_theme(self, theme: UITheme) -> None:
        """
        Set UI theme.
        
        Args:
            theme: UI theme to set
        """
        self.theme = theme
        self.ui_state["theme"] = theme.name
        self.config_manager.set_config("ui.theme", theme.name)
        self.event_bus.publish_event("ui.theme_changed", {"theme": theme.name})
        self.logger.debug(f"Theme set to {theme.name}")
    
    def set_font_size(self, font_size: int) -> None:
        """
        Set font size.
        
        Args:
            font_size: Font size to set
        """
        self.font_size = font_size
        self.ui_state["font_size"] = font_size
        self.config_manager.set_config("ui.font_size", font_size)
        self.event_bus.publish_event("ui.font_size_changed", {"font_size": font_size})
        self.logger.debug(f"Font size set to {font_size}")
    
    def toggle_animations(self, enabled: bool) -> None:
        """
        Toggle animations.
        
        Args:
            enabled: Whether animations should be enabled
        """
        self.animations_enabled = enabled
        self.ui_state["animations_enabled"] = enabled
        self.config_manager.set_config("ui.animations_enabled", enabled)
        self.event_bus.publish_event("ui.animations_toggled", {"enabled": enabled})
        self.logger.debug(f"Animations {'enabled' if enabled else 'disabled'}")
    
    def set_current_view(self, view: str) -> None:
        """
        Set current view.
        
        Args:
            view: View to set
        """
        self.ui_state["current_view"] = view
        self.event_bus.publish_event("ui.view_changed", {"view": view})
        self.logger.debug(f"Current view set to {view}")
    
    def toggle_sidebar(self, is_open: bool) -> None:
        """
        Toggle sidebar.
        
        Args:
            is_open: Whether sidebar should be open
        """
        self.ui_state["is_sidebar_open"] = is_open
        self.event_bus.publish_event("ui.sidebar_toggled", {"is_open": is_open})
        self.logger.debug(f"Sidebar {'opened' if is_open else 'closed'}")
    
    def toggle_settings(self, is_open: bool) -> None:
        """
        Toggle settings.
        
        Args:
            is_open: Whether settings should be open
        """
        self.ui_state["is_settings_open"] = is_open
        self.event_bus.publish_event("ui.settings_toggled", {"is_open": is_open})
        self.logger.debug(f"Settings {'opened' if is_open else 'closed'}")
    
    def _handle_theme_changed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle theme changed event.
        
        Args:
            event_data: Event data
        """
        # This is just to handle events from other components
        # We don't need to do anything here since we already handled the theme change
        pass
    
    def _handle_font_size_changed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle font size changed event.
        
        Args:
            event_data: Event data
        """
        # This is just to handle events from other components
        # We don't need to do anything here since we already handled the font size change
        pass
    
    def _handle_animations_toggled(self, event_data: Dict[str, Any]) -> None:
        """
        Handle animations toggled event.
        
        Args:
            event_data: Event data
        """
        # This is just to handle events from other components
        # We don't need to do anything here since we already handled the animations toggle
        pass
    
    def start_web_server(self) -> None:
        """Start the web server for the UI."""
        # Create UI directory if it doesn't exist
        ui_dir = os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus', 'ui')
        os.makedirs(ui_dir, exist_ok=True)
        
        # Create a simple index.html if it doesn't exist
        index_path = os.path.join(ui_dir, 'index.html')
        if not os.path.exists(index_path):
            with open(index_path, 'w') as f:
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>miniManus</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }
                        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
                        .header { background-color: #4a90e2; color: white; padding: 20px; text-align: center; }
                        .chat-container { background-color: white; border-radius: 5px; padding: 20px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                        .message { margin-bottom: 15px; }
                        .user-message { text-align: right; }
                        .user-bubble { background-color: #4a90e2; color: white; border-radius: 20px; padding: 10px 15px; display: inline-block; max-width: 80%; }
                        .bot-message { text-align: left; }
                        .bot-bubble { background-color: #e5e5ea; color: black; border-radius: 20px; padding: 10px 15px; display: inline-block; max-width: 80%; }
                        .input-area { display: flex; margin-top: 20px; }
                        #message-input { flex-grow: 1; padding: 10px; border: 1px solid #ddd; border-radius: 20px; margin-right: 10px; }
                        #send-button { background-color: #4a90e2; color: white; border: none; border-radius: 20px; padding: 10px 20px; cursor: pointer; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>miniManus</h1>
                    </div>
                    <div class="container">
                        <div class="chat-container" id="chat-container">
                            <div class="message bot-message">
                                <div class="bot-bubble">Hello! I'm miniManus. How can I help you today?</div>
                            </div>
                        </div>
                        <div class="input-area">
                            <input type="text" id="message-input" placeholder="Type your message...">
                            <button id="send-button">Send</button>
                        </div>
                    </div>
                    
                    <script>
                        document.getElementById('send-button').addEventListener('click', sendMessage);
                        document.getElementById('message-input').addEventListener('keypress', function(e) {
                            if (e.key === 'Enter') {
                                sendMessage();
                            }
                        });
                        
                        function sendMessage() {
                            const input = document.getElementById('message-input');
                            const message = input.value.trim();
                            
                            if (message) {
                                // Add user message to chat
                                addMessage('user', message);
                                
                                // Clear input
                                input.value = '';
                                
                                // Send message to server
                                fetch('/api/chat', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json',
                                    },
                                    body: JSON.stringify({
                                        message: message
                                    }),
                                })
                                .then(response => {
                                    if (!response.ok) {
                                        throw new Error('Network response was not ok');
                                    }
                                    return response.json();
                                })
                                .then(data => {
                                    // Add bot response to chat
                                    addMessage('bot', data.response);
                                })
                                .catch((error) => {
                                    console.error('Error:', error);
                                    addMessage('bot', 'Sorry, there was an error processing your request.');
                                });
                            }
                        }
                        
                        function addMessage(role, content) {
                            const chatContainer = document.getElementById('chat-container');
                            const messageDiv = document.createElement('div');
                            messageDiv.className = `message ${role}-message`;
                            
                            const bubbleDiv = document.createElement('div');
                            bubbleDiv.className = `${role}-bubble`;
                            bubbleDiv.textContent = content;
                            
                            messageDiv.appendChild(bubbleDiv);
                            chatContainer.appendChild(messageDiv);
                            
                            // Scroll to bottom
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    </script>
                </body>
                </html>
                """)
        
        # Set up the HTTP server
        PORT = self.server_port
        
        # Reference to the UIManager instance for the handler
        ui_manager = self
        
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=ui_dir, **kwargs)
            
            def do_POST(self):
                """Handle POST requests."""
                try:
                    # Check if this is an API request
                    if self.path == '/api/chat':
                        # Get content length
                        content_length = int(self.headers['Content-Length'])
                        
                        # Read the request body
                        post_data = self.rfile.read(content_length)
                        
                        # Parse JSON data
                        request_data = json.loads(post_data.decode('utf-8'))
                        
                        # Get the message from the request
                        message = request_data.get('message', '')
                        
                        # Get the chat interface
                        from ..ui.chat_interface import ChatInterface, MessageRole, ChatMessage
                        chat_interface = ChatInterface.get_instance()
                        
                        # Create a new session if none exists
                        if not chat_interface.current_session_id:
                            session = chat_interface.create_session(title="New Chat")
                        else:
                            session = chat_interface.get_session(chat_interface.current_session_id)
                        
                        # Add user message to session
                        user_message = ChatMessage(MessageRole.USER, message)
                        session.add_message(user_message)
                        
                        # Process the message (in a real implementation, this would call the LLM API)
                        # For now, we'll just echo the message back with a prefix
                        response_text = f"I received your message: {message}"
                        
                        # Add assistant message to session
                        assistant_message = ChatMessage(MessageRole.ASSISTANT, response_text)
                        session.add_message(assistant_message)
                        
                        # Prepare response
                        response = {
                            'status': 'success',
                            'response': response_text
                        }
                        
                        # Send response
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                        
                        # Log the interaction
                        ui_manager.logger.info(f"Chat message processed: {message}")
                        
                    else:
                        # Handle other POST requests
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b'Not Found')
                
                except Exception as e:
                    # Log the error
                    ui_manager.logger.error(f"Error handling POST request: {str(e)}")
                    
                    # Send error response
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    error_response = {
                        'status': 'error',
                        'message': 'Internal server error'
                    }
                    self.wfile.write(json.dumps(error_response).encode('utf-8'))
        
        try:
            # Use ThreadingTCPServer with allow_reuse_address to prevent "Address already in use" errors
            socketserver.TCPServer.allow_reuse_address = True
            self.httpd = socketserver.TCPServer(("0.0.0.0", PORT), CustomHandler)
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.server_thread.start()
            self.logger.info(f"Web server started at http://localhost:{PORT}")
            print(f"\n\n*** miniManus UI is now available at http://localhost:{PORT} ***\n")
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            self.error_handler.handle_error(
                e, ErrorCategory.UI, ErrorSeverity.ERROR,
                {"action": "start_web_server"}
            )
    
    def startup(self) -> None:
        """Start the UI manager."""
        # Start web server
        self.start_web_server()
        self.logger.info("UIManager started")
    
    def shutdown(self) -> None:
        """Stop the UI manager."""
        # Stop web server if it's running
        if hasattr(self, 'httpd') and self.httpd:
            try:
                # First shutdown the server (stops serve_forever loop)
                self.httpd.shutdown()
                
                # Then close the socket to release the port
                self.httpd.server_close()
                
                # Wait for server thread to terminate
                if self.server_thread and self.server_thread.is_alive():
                    self.server_thread.join(timeout=2.0)
                
                self.logger.info("Web server stopped and port released")
            except Exception as e:
                self.logger.error(f"Error shutting down web server: {e}")
        
        self.logger.info("UIManager stopped")
