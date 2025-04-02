import os
import sys
import json
import logging
import asyncio
import threading
import socketserver
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
        self.httpd = None
        self.server_thread = None
        
        self.logger.info("UIManager initialized")
    
    def startup(self) -> None:
        """Start the UI manager."""
        # Start the web server
        self._start_web_server()
        
        # Subscribe to events
        self.event_bus.subscribe("settings.updated", self._on_settings_updated)
        
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
        
        # Unsubscribe from events
        self.event_bus.unsubscribe("settings.updated", self._on_settings_updated)
        
        self.logger.info("UIManager stopped")
    
    def _start_web_server(self):
        """Start the web server."""
        # Create a request handler with access to the UIManager instance
        ui_manager = self
        
        class CustomHandler(BaseHTTPRequestHandler):
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
                settings = ui_manager.config_manager.get_config()
                
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
                if isinstance(body, dict):
                    for key, value in body.items():
                        if key == "providers":
                            # Handle nested provider settings
                            for provider, provider_settings in value.items():
                                for setting_key, setting_value in provider_settings.items():
                                    # Make sure we handle model selection properly
                                    if setting_key == "model":
                                        config_path = f"api.{provider}.default_model"
                                        ui_manager.config_manager.set_config(config_path, setting_value)
                                        ui_manager.logger.info(f"Set {provider} model to {setting_value}")
                                    else:
                                        config_path = f"api.providers.{provider}.{setting_key}"
                                        ui_manager.config_manager.set_config(config_path, setting_value)
                        else:
                            ui_manager.config_manager.set_config(key, value)
                
                # Save the settings
                ui_manager.config_manager.save_config()
                
                # Publish an event
                ui_manager.event_bus.publish_event("settings.updated", {"settings": body})
                
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
                
                if provider and model:
                    # Update the model in the config manager
                    config_path = f"api.{provider}.default_model"
                    ui_manager.config_manager.set_config(config_path, model)
                    ui_manager.logger.info(f"Updated model for {provider} to: {model}")
                    
                    # Also update the matching provider in adapters
                    if provider == "ollama":
                        adapter = ui_manager.api_manager.get_adapter(APIProvider.OLLAMA)
                        if adapter:
                            adapter.default_model = model
                            ui_manager.logger.info(f"Updated Ollama adapter default model to: {model}")
                
                # Save the settings
                ui_manager.config_manager.save_config()
                
                # Send the response
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
                
                ui_manager.logger.info(f"Model updated: {provider}/{model}")
            
            def _serve_static_file(self, path):
                """Serve a static file."""
                # Map URL path to file path
                if path == "/" or path == "":
                    file_path = os.path.join(ui_manager.static_dir, "index.html")
                else:
                    # Remove leading slash
                    path = path[1:] if path.startswith("/") else path
                    file_path = os.path.join(ui_manager.static_dir, path)
                
                # Check if file exists
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
                
                # Determine content type
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
                
                # Return the content type or default to binary
                return content_types.get(ext.lower(), "application/octet-stream")
        
        # Enable address reuse to prevent "Address already in use" errors on restart
        socketserver.TCPServer.allow_reuse_address = True
        
        # Create and start the server
        self.httpd = HTTPServer((self.host, self.port), CustomHandler)
        
        # Start the server in a separate thread
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.logger.info(f"Web server started at http://{self.host}:{self.port}")
    
    def _on_settings_updated(self, event):
        """Handle settings updated event."""
        # Reload configuration when settings are updated
        settings = event.data.get("settings", {})
        
        # Handle theme changes
        if "theme" in settings:
            theme_name = settings["theme"]
            try:
                theme = UITheme[theme_name.upper()]
                self.set_theme(theme)
            except (KeyError, AttributeError):
                self.logger.warning(f"Invalid theme: {theme_name}")
        
        # Handle API provider changes
        if "defaultProvider" in settings:
            provider = settings["defaultProvider"]
            self.logger.info(f"Default provider changed to {provider}")
    
    def _process_chat_message(self, message):
        """Process a chat message."""
        # Get the default provider from settings
        provider_name = self.config_manager.get_config("api.default_provider", "openrouter")
        
        # Map provider name to APIProvider enum
        provider_map = {
            "openrouter": APIProvider.OPENROUTER,
            "anthropic": APIProvider.ANTHROPIC,
            "deepseek": APIProvider.DEEPSEEK,
            "ollama": APIProvider.OLLAMA,
            "litellm": APIProvider.LITELLM,
        }
        
        provider = provider_map.get(provider_name.lower())
        if not provider:
            return "Sorry, the selected API provider is not available. Please check your settings."
        
        # Get the adapter for the provider
        adapter = self.api_manager.get_adapter(provider)
        if not adapter:
            return "Sorry, the selected API provider is not available. Please check your settings."
        
        # Get the model for the provider - Fixed config path
        model = self.config_manager.get_config(f"api.{provider_name.lower()}.default_model")
        
        # Check if the provider is available
        if not self.api_manager.check_provider_availability(provider):
            return "Sorry, the selected API provider is not available. Please check your settings."
        
        # Process the message
        try:
            # Create a simple message format
            messages = [
                {"role": "user", "content": message}
            ]
            
            # Call the adapter - Log what we're using
            self.logger.info(f"Using provider: {provider_name}, model: {model}")
            response = adapter.generate_text(messages, model=model)
            
            return response
        
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": provider_name, "action": "generate_text"}
            )
            
            return f"Sorry, there was an error processing your request: {str(e)}"
    
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
                self.logger.warning(f"Unknown provider: {provider_name}")
                return []
            
            # Get the adapter for the provider
            adapter = self.api_manager.get_adapter(provider)
            if not adapter:
                self.logger.warning(f"No adapter found for provider: {provider_name}")
                return []
            
            # Check if the provider is available
            if not self.api_manager.check_provider_availability(provider):
                self.logger.warning(f"Provider not available: {provider_name}")
                # Return hardcoded models as fallback
                return self._get_fallback_models(provider_name)
            
            # Get available models
            if hasattr(adapter, 'get_available_models'):
                try:
                    # Run the async method in a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        models = loop.run_until_complete(adapter.get_available_models())
                        if models:
                            self.logger.info(f"Successfully fetched {len(models)} models from {provider_name}")
                            return models
                        else:
                            self.logger.warning(f"No models returned from {provider_name}, using fallback")
                            return self._get_fallback_models(provider_name)
                    except Exception as e:
                        self.logger.error(f"Error fetching models from {provider_name}: {str(e)}")
                        return self._get_fallback_models(provider_name)
                    finally:
                        loop.close()
                except Exception as e:
                    self.logger.error(f"Error setting up async loop for {provider_name}: {str(e)}")
                    return self._get_fallback_models(provider_name)
            else:
                self.logger.warning(f"Adapter for {provider_name} does not have get_available_models method")
                return self._get_fallback_models(provider_name)
        
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.API, ErrorSeverity.ERROR,
                {"provider": provider_name}
            )
            return self._get_fallback_models(provider_name)
    
    def _get_fallback_models(self, provider_name):
        """Get fallback models for a provider when API call fails."""
        self.logger.info(f"Using fallback models for {provider_name}")
        
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
    
    def set_theme(self, theme: UITheme) -> None:
        """
        Set the UI theme.
        
        Args:
            theme: Theme to set
        """
        self.config_manager.set_config("ui.theme", theme.name.lower())
        self.logger.debug(f"Set theme to {theme.name}")
    
    def set_font_size(self, font_size: int) -> None:
        """
        Set the UI font size.
        
        Args:
            font_size: Font size to set
        """
        self.config_manager.set_config("ui.font_size", font_size)
        self.logger.debug(f"Set font size to {font_size}")
    
    def toggle_animations(self, enabled: bool) -> None:
        """
        Toggle UI animations.
        
        Args:
            enabled: Whether animations are enabled
        """
        self.config_manager.set_config("ui.animations_enabled", enabled)
        self.logger.debug(f"Set animations enabled to {enabled}")
    
    def toggle_compact_mode(self, enabled: bool) -> None:
        """
        Toggle UI compact mode.
        
        Args:
            enabled: Whether compact mode is enabled
        """
        self.config_manager.set_config("ui.compact_mode", enabled)
        self.logger.debug(f"Set compact mode to {enabled}")
    
    def show_notification(self, message: str, duration: int = 3000) -> None:
        """
        Show a notification in the UI.
        
        Args:
            message: Notification message
            duration: Duration in milliseconds
        """
        self.event_bus.publish_event("ui.notification", {
            "message": message,
            "duration": duration
        })
        self.logger.debug(f"Showed notification: {message}")
    
    def show_error(self, message: str) -> None:
        """
        Show an error message in the UI.
        
        Args:
            message: Error message
        """
        self.event_bus.publish_event("ui.error", {
            "message": message
        })
        self.logger.debug(f"Showed error: {message}")
