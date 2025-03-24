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
        
        # UI state
        self.theme = UITheme.SYSTEM
        self.font_size = 14
        self.animations_enabled = True
        
        # Web server
        self.server = None
        self.server_thread = None
        self.port = 8080
        self.host = "localhost"
        
        self.logger.info("UIManager initialized")
    
    def set_theme(self, theme: UITheme) -> None:
        """
        Set the UI theme.
        
        Args:
            theme: Theme to set
        """
        self.theme = theme
        self.logger.debug(f"Set theme to {theme.name}")
    
    def set_font_size(self, size: int) -> None:
        """
        Set the UI font size.
        
        Args:
            size: Font size to set
        """
        self.font_size = size
        self.logger.debug(f"Set font size to {size}")
    
    def toggle_animations(self, enabled: bool) -> None:
        """
        Toggle UI animations.
        
        Args:
            enabled: Whether animations are enabled
        """
        self.animations_enabled = enabled
        self.logger.debug(f"Set animations enabled to {enabled}")
    
    def startup(self) -> None:
        """Start the UI manager."""
        # Start web server
        try:
            # Create custom handler with access to UIManager
            ui_manager = self
            
            class CustomHandler(http.server.SimpleHTTPRequestHandler):
                """Custom HTTP request handler."""
                
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=os.path.join(os.path.dirname(__file__), "../static"), **kwargs)
                
                def log_message(self, format, *args):
                    """Override to use our logger."""
                    ui_manager.logger.debug(format % args)
                
                def do_GET(self):
                    """Handle GET requests."""
                    if self.path == "/":
                        # Serve index.html
                        self.path = "/index.html"
                    
                    try:
                        super().do_GET()
                    except FileNotFoundError:
                        # If file not found in static directory, serve index.html
                        self.send_response(200)
                        self.send_header("Content-type", "text/html")
                        self.end_headers()
                        
                        # Read index.html
                        index_path = os.path.join(os.path.dirname(__file__), "../static/index.html")
                        if os.path.exists(index_path):
                            with open(index_path, "rb") as f:
                                self.wfile.write(f.read())
                        else:
                            # Fallback to minimal HTML
                            html = """
                            <!DOCTYPE html>
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
                                </style>
                            </head>
                            <body>
                                <header>
                                    <h1>miniManus</h1>
                                </header>
                                <main>
                                    <div class="chat-container">
                                        <div class="chat-messages" id="chat-messages">
                                            <div class="message assistant-message">
                                                <p>Hello! I'm miniManus. How can I help you today?</p>
                                            </div>
                                        </div>
                                        <div class="chat-input">
                                            <input type="text" id="message-input" placeholder="Type your message...">
                                            <button id="send-button">Send</button>
                                        </div>
                                    </div>
                                </main>
                                <script>
                                    document.addEventListener('DOMContentLoaded', () => {
                                        const messagesContainer = document.getElementById('chat-messages');
                                        const messageInput = document.getElementById('message-input');
                                        const sendButton = document.getElementById('send-button');
                                        
                                        function addMessage(content, isUser = false) {
                                            const messageDiv = document.createElement('div');
                                            messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
                                            
                                            const messagePara = document.createElement('p');
                                            messagePara.textContent = content;
                                            
                                            messageDiv.appendChild(messagePara);
                                            messagesContainer.appendChild(messageDiv);
                                            
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
                                    });
                                </script>
                            </body>
                            </html>
                            """
                            self.wfile.write(html.encode())
                
                def do_POST(self):
                    """Handle POST requests."""
                    try:
                        if self.path == "/api/chat":
                            # Get request body
                            content_length = int(self.headers['Content-Length'])
                            post_data = self.rfile.read(content_length)
                            request_data = json.loads(post_data.decode('utf-8'))
                            
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
                            
                            # Get API manager
                            api_manager = APIManager.get_instance()
                            
                            # Get settings from config
                            default_provider_name = ui_manager.config_manager.get_config("api.default_provider", "openrouter")
                            try:
                                default_provider = APIProvider[default_provider_name.upper()]
                            except (KeyError, ValueError):
                                default_provider = APIProvider.OPENROUTER
                            
                            # Check if the provider is available
                            if api_manager.check_provider_availability(default_provider):
                                # Get provider-specific settings
                                provider_key = default_provider_name.lower()
                                api_key = ui_manager.config_manager.get_config(f"api.{provider_key}.api_key", "")
                                model = ui_manager.config_manager.get_config(f"api.{provider_key}.default_model", "")
                                temperature = ui_manager.config_manager.get_config("api.temperature", 0.7)
                                max_tokens = ui_manager.config_manager.get_config("api.max_tokens", 1024)
                                
                                # Prepare messages for API
                                messages = []
                                
                                # Add system message if available
                                if session.system_prompt:
                                    messages.append({
                                        "role": "system",
                                        "content": session.system_prompt
                                    })
                                
                                # Add conversation history (limited to last 10 messages)
                                for msg in session.messages[-10:]:
                                    messages.append({
                                        "role": msg.role.name.lower(),
                                        "content": msg.content
                                    })
                                
                                try:
                                    # Get the appropriate adapter
                                    adapter = api_manager.get_adapter(default_provider)
                                    
                                    if adapter:
                                        # Call the API
                                        response_text = adapter.generate_text(
                                            messages=messages,
                                            model=model,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            api_key=api_key
                                        )
                                    else:
                                        # Fallback if no adapter
                                        response_text = "I'm sorry, but I couldn't connect to the language model. Please check your API settings."
                                except Exception as e:
                                    # Handle API errors
                                    ui_manager.logger.error(f"API error: {str(e)}")
                                    response_text = f"I'm sorry, but there was an error communicating with the language model: {str(e)}"
                            else:
                                # Provider not available
                                response_text = f"I'm sorry, but the {default_provider_name} API is not available. Please check your API settings and ensure you've entered a valid API key."
                            
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
                        self.wfile.write(json.dumps({
                            'status': 'error',
                            'message': str(e)
                        }).encode('utf-8'))
            
            # Enable address reuse to prevent "Address already in use" errors
            socketserver.TCPServer.allow_reuse_address = True
            
            # Create and start server
            self.server = socketserver.TCPServer((self.host, self.port), CustomHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.logger.info(f"Web server started at http://{self.host}:{self.port}")
            print(f"\n\n*** miniManus UI is now available at http://{self.host}:{self.port} ***\n")
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {str(e)}")
            self.error_handler.handle_error(
                e, ErrorCategory.UI, ErrorSeverity.ERROR,
                {"component": "UIManager", "action": "startup"}
            )
        
        self.logger.info("UIManager started")
    
    def shutdown(self) -> None:
        """Stop the UI manager."""
        # Stop web server
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()  # Release the socket
                if self.server_thread and self.server_thread.is_alive():
                    self.server_thread.join(timeout=2.0)  # Wait for thread to terminate
                self.logger.info("Web server stopped")
            except Exception as e:
                self.logger.error(f"Error stopping web server: {str(e)}")
        
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
    
    # Keep main thread alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        ui_manager.shutdown()
