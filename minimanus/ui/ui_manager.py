import os
import sys
import json
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from enum import Enum, auto

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
    from ..api.api_manager import APIManager, APIProvider
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager
    from api.api_manager import APIManager, APIProvider

logger = logging.getLogger("miniManus.UIManager")

class UITheme(Enum):
    """UI theme options."""
    LIGHT = auto()
    DARK = auto()
    SYSTEM = auto()

class UIManager:
    """
    UIManager handles the web-based user interface for miniManus.
    
    It provides:
    - Web server for the UI
    - API endpoints for the UI to interact with the backend
    - Static file serving for the UI assets
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
        self.api_manager = APIManager.get_instance()
        
        # Web server configuration
        self.host = self.config_manager.get_config("ui.host", "localhost")
        self.port = self.config_manager.get_config("ui.port", 8080)
        
        # Static file directory
        self.static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
        
        # Server instance
        self.server = None
        self.server_thread = None
        
        self.logger.info("UIManager initialized")
    
    def startup(self):
        """Start the UI manager (alias for start method for consistency with other components)."""
        self.start()
    
    def start(self):
        """Start the UI manager."""
        # Start the web server
        self._start_web_server()
        
        # Subscribe to events
        self.event_bus.subscribe("settings.updated", self._on_settings_updated)
        
        self.logger.info("UIManager started")
    
    def stop(self):
        """Stop the UI manager."""
        # Stop the web server
        if self.server:
            self.server.shutdown()
            self.server_thread.join()
            self.server = None
            self.server_thread = None
        
        # Unsubscribe from events
        self.event_bus.unsubscribe("settings.updated", self._on_settings_updated)
        
        self.logger.info("UIManager stopped")
    
    def _start_web_server(self):
        """Start the web server."""
        # Create a request handler with access to the UIManager instance
        ui_manager = self
        
        class RequestHandler(BaseHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.ui_manager = ui_manager
                super().__init__(*args, **kwargs)
            
            def log_message(self, format, *args):
                # Redirect log messages to our logger
                ui_manager.logger.debug(format % args)
            
            def do_GET(self):
                """Handle GET requests."""
                try:
                    # Parse the URL
                    parsed_url = urlparse(self.path)
                    path = parsed_url.path
                    
                    # API endpoints
                    if path.startswith("/api/"):
                        if path == "/api/settings":
                            self._handle_get_settings()
                        elif path == "/api/models":
                            self._handle_get_models(parsed_url)
                        else:
                            self.send_error(404, "API endpoint not found")
                    
                    # Static files
                    else:
                        self._serve_static_file(path)
                
                except Exception as e:
                    ui_manager.error_handler.handle_error(
                        e, ErrorCategory.UI, ErrorSeverity.ERROR,
                        {"path": self.path, "method": "GET"}
                    )
                    self.send_error(500, str(e))
            
            def do_POST(self):
                """Handle POST requests."""
                try:
                    # Parse the URL
                    parsed_url = urlparse(self.path)
                    path = parsed_url.path
                    
                    # Get the request body
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length).decode("utf-8")
                    
                    # Parse JSON body
                    if content_length > 0:
                        try:
                            body = json.loads(body)
                        except json.JSONDecodeError:
                            self.send_error(400, "Invalid JSON")
                            return
                    
                    # API endpoints
                    if path == "/api/settings":
                        self._handle_post_settings(body)
                    elif path == "/api/chat":
                        self._handle_post_chat(body)
                    elif path == "/api/model":
                        self._handle_post_model(body)
                    else:
                        self.send_error(404, "API endpoint not found")
                
                except Exception as e:
                    ui_manager.error_handler.handle_error(
                        e, ErrorCategory.UI, ErrorSeverity.ERROR,
                        {"path": self.path, "method": "POST"}
                    )
                    self.send_error(500, str(e))
            
            def _handle_get_settings(self):
                """Handle GET /api/settings."""
                # Get the settings from the config manager
                settings = ui_manager.config_manager.get_all_config()
                
                # Send the response
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(settings).encode("utf-8"))
            
            def _handle_get_models(self, parsed_url):
                """Handle GET /api/models."""
                # Parse query parameters
                query = parse_qs(parsed_url.query)
                provider = query.get("provider", ["openrouter"])[0]
                
                # Get models for the provider
                models = ui_manager._get_models_for_provider(provider)
                
                # Send the response
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"models": models}).encode("utf-8"))
            
            def _handle_post_settings(self, body):
                """Handle POST /api/settings."""
                # Update the settings in the config manager
                ui_manager.config_manager.update_config(body)
                
                # Save the settings
                ui_manager.config_manager.save_config()
                
                # Publish an event
                ui_manager.event_bus.publish(
                    Event("settings.updated", {"settings": body})
                )
                
                # Send the response
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
                
                ui_manager.logger.info("Settings updated")
            
            def _handle_post_chat(self, body):
                """Handle POST /api/chat."""
                # Get the message from the request body
                message = body.get("message", "")
                
                # Process the message
                response = ui_manager._process_chat_message(message)
                
                # Send the response
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"response": response}).encode("utf-8"))
                
                ui_manager.logger.info(f"Chat message processed: {message}")
            
            def _handle_post_model(self, body):
                """Handle POST /api/model."""
                # Get the model from the request body
                model = body.get("model", "")
                provider = body.get("provider", "")
                
                # Update the model in the config manager
                if provider:
                    ui_manager.config_manager.update_config({
                        "providers": {
                            provider: {
                                "model": model
                            }
                        }
                    })
                    
                    # Save the settings
                    ui_manager.config_manager.save_config()
                    
                    # Publish an event
                    ui_manager.event_bus.publish(
                        Event("model.updated", {"provider": provider, "model": model})
                    )
                
                # Send the response
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
                
                ui_manager.logger.info(f"Model updated: {provider}/{model}")
            
            def _serve_static_file(self, path):
                """Serve a static file."""
                # Map the URL path to a file path
                if path == "/" or path == "":
                    file_path = os.path.join(ui_manager.static_dir, "index.html")
                else:
                    # Remove leading slash
                    path = path[1:] if path.startswith("/") else path
                    file_path = os.path.join(ui_manager.static_dir, path)
                
                # Check if the file exists
                if not os.path.isfile(file_path):
                    # Try index.html for directories
                    if os.path.isdir(file_path):
                        file_path = os.path.join(file_path, "index.html")
                        if not os.path.isfile(file_path):
                            self.send_error(404, "File not found")
                            return
                    else:
                        self.send_error(404, "File not found")
                        return
                
                # Determine the content type
                content_type = self._get_content_type(file_path)
                
                # Send the file
                try:
                    with open(file_path, "rb") as f:
                        content = f.read()
                    
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                
                except Exception as e:
                    ui_manager.error_handler.handle_error(
                        e, ErrorCategory.UI, ErrorSeverity.ERROR,
                        {"path": path, "file_path": file_path}
                    )
                    self.send_error(500, str(e))
            
            def _get_content_type(self, file_path):
                """Get the content type for a file."""
                # Map file extensions to content types
                content_types = {
                    ".html": "text/html",
                    ".css": "text/css",
                    ".js": "application/javascript",
                    ".json": "application/json",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".svg": "image/svg+xml",
                    ".ico": "image/x-icon",
                }
                
                # Get the file extension
                _, ext = os.path.splitext(file_path)
                
                # Return the content type
                return content_types.get(ext.lower(), "application/octet-stream")
        
        # Create and start the server
        self.server = HTTPServer((self.host, self.port), RequestHandler)
        
        # Start the server in a separate thread
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.logger.info(f"Web server started at http://{self.host}:{self.port}")
    
    def _on_settings_updated(self, event):
        """Handle settings.updated event."""
        # Nothing to do here for now
        pass
    
    def _process_chat_message(self, message):
        """Process a chat message."""
        try:
            # Get the default provider from config
            provider_name = self.config_manager.get_config("defaultProvider", "openrouter")
            
            # Map provider name to APIProvider enum
            provider_map = {
                "openrouter": APIProvider.OPENROUTER,
                "anthropic": APIProvider.ANTHROPIC,
                "deepseek": APIProvider.DEEPSEEK,
                "ollama": APIProvider.OLLAMA,
                "litellm": APIProvider.LITELLM,
            }
            
            provider = provider_map.get(provider_name, APIProvider.OPENROUTER)
            
            # Get the adapter for the provider
            adapter = self.api_manager.get_adapter(provider)
            
            if not adapter:
                return "I'm sorry, but the selected API provider is not available. Please check your settings."
            
            # Get the model from config
            model = None
            if provider_name == "openrouter":
                model = self.config_manager.get_config("providers.openrouter.model")
            elif provider_name == "anthropic":
                model = self.config_manager.get_config("providers.anthropic.model")
            elif provider_name == "deepseek":
                model = self.config_manager.get_config("providers.deepseek.model")
            elif provider_name == "ollama":
                model = self.config_manager.get_config("providers.ollama.model")
            elif provider_name == "litellm":
                model = self.config_manager.get_config("providers.litellm.model")
            
            # Format the message for the provider
            messages = [
                {"role": "system", "content": "You are miniManus, a helpful AI assistant running on a mobile device."},
                {"role": "user", "content": message}
            ]
            
            # Generate a response
            if hasattr(adapter, 'generate_text'):
                response = adapter.generate_text(messages, model=model)
                return response
            else:
                return "I'm sorry, but the selected API provider does not support text generation."
        
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"message": message}
            )
            return f"I'm sorry, but there was an error processing your request: {str(e)}"
    
    def _get_models_for_provider(self, provider_name):
        """Get available models for a provider."""
        try:
            # Map provider name to APIProvider enum
            provider_map = {
                "openrouter": APIProvider.OPENROUTER,
                "anthropic": APIProvider.ANTHROPIC,
                "deepseek": APIProvider.DEEPSEEK,
                "ollama": APIProvider.OLLAMA,
                "litellm": APIProvider.LITELLM,
            }
            
            provider = provider_map.get(provider_name)
            if not provider:
                return []
            
            # Get the adapter for the provider
            adapter = self.api_manager.get_adapter(provider)
            if not adapter:
                return []
            
            # Get available models
            if hasattr(adapter, 'get_available_models'):
                # Run the async method in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    models = loop.run_until_complete(adapter.get_available_models())
                finally:
                    loop.close()
                return models
            else:
                # Return default models for providers without a get_available_models method
                if provider_name == "openrouter":
                    return [
                        {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus"},
                        {"id": "openai/gpt-4-turbo", "name": "GPT-4 Turbo"},
                        {"id": "mistralai/mistral-large", "name": "Mistral Large"}
                    ]
                elif provider_name == "anthropic":
                    return [
                        {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
                        {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
                        {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"}
                    ]
                elif provider_name == "deepseek":
                    return [
                        {"id": "deepseek-chat", "name": "DeepSeek Chat"},
                        {"id": "deepseek-coder", "name": "DeepSeek Coder"}
                    ]
                elif provider_name == "ollama":
                    return [
                        {"id": "llama3", "name": "Llama 3"},
                        {"id": "mistral", "name": "Mistral"},
                        {"id": "llama2", "name": "Llama 2"}
                    ]
                elif provider_name == "litellm":
                    return [
                        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
                        {"id": "gpt-4", "name": "GPT-4"}
                    ]
                else:
                    return []
        
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": provider_name}
            )
            return []

# For standalone testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ui_manager = UIManager.get_instance()
    ui_manager.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ui_manager.stop()
