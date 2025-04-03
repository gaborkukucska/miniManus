# START OF FILE miniManus-main/minimanus/ui/ui_manager.py
import os
import sys
import json
import logging
import asyncio
import threading
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple # Added Tuple

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
    from ..api.api_manager import APIManager, APIProvider
    # Import components needed for API handlers
    from ..ui.model_selection import ModelSelectionInterface
    from ..ui.chat_interface import ChatInterface, MessageRole, ChatMessage # Import ChatInterface components
except ImportError as e:
    # Handle potential import errors during early startup or testing
    logging.getLogger("miniManus.UIManager").critical(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(f"ImportError in ui_manager.py: {e}. Ensure all components exist.")

logger = logging.getLogger("miniManus.UIManager")

# Define BASE_DIR for static file path resolution
# Assumes ui_manager.py is in minimanus/ui/
MINIMANUS_ROOT_DIR = Path(__file__).parent.parent.parent
STATIC_DIR_DEFAULT = MINIMANUS_ROOT_DIR / 'static'


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
        # No direct APIManager needed here, interact via Chat/Model interfaces
        # self.api_manager = APIManager.get_instance()
        self.model_selection_interface = ModelSelectionInterface.get_instance()
        self.chat_interface = ChatInterface.get_instance()

        # Web server configuration
        self.host = self.config_manager.get_config("ui.host", "localhost")
        self.port = self.config_manager.get_config("ui.port", 8080)
        self.request_timeout = 120 # Timeout for waiting for async results (seconds)

        # Static file directory (use default, can be overridden)
        self.static_dir = str(STATIC_DIR_DEFAULT)

        # Server instance
        self.httpd = None
        self.server_thread = None

        # Event loop (store the loop where startup runs)
        self._loop = None

        self.logger.info("UIManager initialized")

    def get_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Returns the event loop the UIManager was started in."""
        return self._loop

    def startup(self) -> None:
        """Start the UI manager."""
        self._loop = asyncio.get_running_loop() # Capture the loop
        # Start the web server
        self._start_web_server()

        # Subscribe to events (example)
        self.event_bus.subscribe("settings.updated", self._on_settings_updated)

        self.logger.info("UIManager started")

    def shutdown(self) -> None:
        """Stop the UI manager."""
        # Stop web server if it's running
        if hasattr(self, 'httpd') and self.httpd:
            try:
                # First shutdown the server (stops serve_forever loop)
                self.logger.info("Shutting down web server...")
                self.httpd.shutdown() # Blocks until loop ends

                # Then close the socket to release the port
                self.logger.info("Closing web server socket...")
                self.httpd.server_close()

                # Wait for server thread to terminate
                if self.server_thread and self.server_thread.is_alive():
                    self.logger.info("Waiting for server thread to join...")
                    self.server_thread.join(timeout=5.0) # Increased timeout
                    if self.server_thread.is_alive():
                         self.logger.warning("Server thread did not join!")

                self.logger.info("Web server stopped and port released.")
            except Exception as e:
                self.logger.error(f"Error shutting down web server: {e}", exc_info=True)
                self.error_handler.handle_error(e, ErrorCategory.UI, ErrorSeverity.ERROR, {"action": "shutdown_webserver"})

        # Unsubscribe from events
        self.event_bus.unsubscribe("settings.updated", self._on_settings_updated)

        self.logger.info("UIManager stopped")

    def _start_web_server(self):
        """Start the web server in a separate thread."""
        if self.server_thread and self.server_thread.is_alive():
            self.logger.warning("Web server thread already running.")
            return

        # Create a request handler with access to the UIManager instance
        ui_manager_instance = self

        class CustomHandler(BaseHTTPRequestHandler):
            # --- Request Handler Class ---
            def __init__(self, *args, **kwargs):
                self.ui_manager = ui_manager_instance
                # Get component instances needed by handlers
                self.config_manager = self.ui_manager.config_manager
                self.model_selection_interface = self.ui_manager.model_selection_interface
                self.chat_interface = self.ui_manager.chat_interface
                self.error_handler = self.ui_manager.error_handler
                self.logger = self.ui_manager.logger
                super().__init__(*args, **kwargs)

            def log_message(self, format, *args):
                # Redirect log messages to our logger
                self.logger.debug(format % args)

            def send_json_response(self, status_code: int, data: Dict[str, Any]):
                """Helper to send JSON responses."""
                try:
                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, OSError) as sock_err:
                     self.logger.warning(f"Socket error sending JSON response (client disconnected?): {sock_err}")
                except Exception as e:
                     self.logger.error(f"Unexpected error sending JSON response: {e}", exc_info=True)

            def send_error_response(self, status_code: int, message: str, error_details: Optional[str] = None):
                """Helper to send JSON error responses."""
                error_data = {"status": "error", "message": message}
                if error_details:
                    error_data["details"] = error_details
                self.send_json_response(status_code, error_data)

            def run_async_task(self, coro) -> Tuple[Optional[Any], Optional[Exception]]:
                """Runs a coroutine in the main event loop and waits for the result."""
                loop = self.ui_manager.get_event_loop()
                if not loop or not loop.is_running():
                    self.logger.error("Event loop not available for async task.")
                    return None, RuntimeError("Event loop unavailable")

                future = asyncio.run_coroutine_threadsafe(coro, loop)
                try:
                    # Wait for the result with a timeout
                    result = future.result(timeout=self.ui_manager.request_timeout)
                    return result, None
                except asyncio.TimeoutError as e:
                    self.logger.error(f"Timeout waiting for async task result ({self.ui_manager.request_timeout}s). Task: {coro.__name__}")
                    return None, e
                except Exception as e:
                    self.logger.error(f"Exception during async task execution: {e}. Task: {coro.__name__}", exc_info=True)
                    return None, e

            # --- GET Handlers ---
            def do_GET(self):
                """Handle GET requests."""
                try:
                    parsed_url = urlparse(self.path)
                    path = parsed_url.path

                    if path.startswith("/api/"):
                        # Route API GET requests
                        if path == "/api/settings":
                            self._handle_get_settings()
                        elif path == "/api/models":
                            self._handle_get_models(parsed_url)
                        elif path == "/api/sessions":
                            self._handle_get_sessions()
                        else:
                            self.send_error_response(404, "API endpoint not found")
                    else:
                        # Serve static files
                        self._serve_static_file(path)

                except (BrokenPipeError, ConnectionResetError) as client_err:
                    self.logger.warning(f"Client connection error during GET {self.path}: {client_err}")
                except Exception as e:
                    self.logger.error(f"GET request error for {self.path}: {e}", exc_info=True)
                    self.error_handler.handle_error(e, ErrorCategory.UI, ErrorSeverity.ERROR, {"path": self.path, "method": "GET"})
                    try:
                        # Only send error if headers haven't been sent (unlikely here but good practice)
                        if not getattr(self, 'headers_sent', False):
                             self.send_error_response(500, "Internal Server Error", str(e))
                    except Exception:
                         self.logger.error(f"Failed to send error response for GET {self.path}")


            def _handle_get_settings(self):
                """Handle GET /api/settings (Synchronous)."""
                try:
                    settings = self.config_manager.get_config()
                    if 'secrets' in settings: del settings['secrets']
                    if 'api_keys' in settings.get('api', {}): del settings['api']['api_keys']
                    self.send_json_response(200, settings)
                except Exception as e:
                    self.logger.error(f"Error getting settings: {e}", exc_info=True)
                    self.send_error_response(500, "Failed to retrieve settings", str(e))

            def _handle_get_models(self, parsed_url):
                """Handle GET /api/models (Runs async task and waits)."""
                query = parse_qs(parsed_url.query)
                provider_name = query.get("provider", [None])[0]

                # Define the async function to call
                async def get_models_async():
                    if not provider_name:
                        models_info = await self.model_selection_interface.get_all_models()
                    else:
                        try:
                            provider_enum = APIProvider[provider_name.upper()]
                            models_info = await self.model_selection_interface.get_models_by_provider(provider_enum)
                        except KeyError:
                            # Raise specific error for invalid provider
                            raise ValueError(f"Invalid provider name: {provider_name}")
                        except Exception as e:
                             # Re-raise other exceptions from model fetching
                             raise RuntimeError(f"Failed to retrieve models for {provider_name}") from e
                    return [m.to_dict() for m in models_info]

                # Run the async task and wait for result
                result, error = self.run_async_task(get_models_async())

                # Handle result or error
                if error:
                    if isinstance(error, ValueError): # Invalid provider
                        self.send_error_response(400, str(error))
                    elif isinstance(error, asyncio.TimeoutError):
                        self.send_error_response(504, "Request timed out while fetching models.")
                    else: # Other internal errors
                        self.send_error_response(500, "Internal server error processing models request.", str(error))
                elif result is not None:
                    self.send_json_response(200, {"models": result})
                else:
                    # Should not happen if error is None, but handle defensively
                     self.send_error_response(500, "Internal server error: No result from async task.")


            def _handle_get_sessions(self):
                """Handle GET /api/sessions (Synchronous)."""
                try:
                    sessions = self.chat_interface.get_all_sessions()
                    sessions_data = [s.to_dict() for s in sessions]
                    current_id = self.chat_interface.current_session_id
                    self.send_json_response(200, {"sessions": sessions_data, "current_session_id": current_id})
                except Exception as e:
                    self.logger.error(f"Error getting sessions: {e}", exc_info=True)
                    self.send_error_response(500, "Failed to retrieve sessions", str(e))

            # --- POST Handlers ---
            def do_POST(self):
                """Handle POST requests."""
                try:
                    parsed_url = urlparse(self.path)
                    path = parsed_url.path

                    content_length = int(self.headers.get("Content-Length", 0))
                    body = {}
                    if content_length > 0:
                        body_raw = self.rfile.read(content_length).decode("utf-8")
                        try:
                            body = json.loads(body_raw)
                        except json.JSONDecodeError:
                            self.send_error_response(400, "Invalid JSON format in request body")
                            return

                    # Route API POST requests
                    if path == "/api/settings":
                        self._handle_post_settings(body) # Synchronous
                    elif path == "/api/chat":
                        self._handle_post_chat(body) # Runs async task and waits
                    elif path == "/api/discover_models":
                        self._handle_post_discover_models() # Runs async task and waits
                    else:
                        self.send_error_response(404, "API endpoint not found")

                except (BrokenPipeError, ConnectionResetError) as client_err:
                    self.logger.warning(f"Client connection error during POST {self.path}: {client_err}")
                except Exception as e:
                    self.logger.error(f"POST request error for {self.path}: {e}", exc_info=True)
                    self.error_handler.handle_error(e, ErrorCategory.UI, ErrorSeverity.ERROR, {"path": self.path, "method": "POST"})
                    try:
                        if not getattr(self, 'headers_sent', False):
                             self.send_error_response(500, "Internal Server Error", str(e))
                    except Exception:
                         self.logger.error(f"Failed to send error response for POST {self.path}")


            def _handle_post_settings(self, body):
                """Handle POST /api/settings (Synchronous)."""
                # (Code is the same as previous version - synchronous config/secret updates)
                self.logger.info("Received POST /api/settings request.")
                keys_saved = True
                api_keys_failed = []
                settings_failed = []
                try:
                    # Separate API keys from regular settings
                    api_keys_to_set = {}
                    regular_settings = {}
                    # self.logger.debug(f"Processing settings body: {body}") # Avoid logging potentially sensitive values

                    for key, value in body.items():
                        is_api_key_field = (
                            (isinstance(key, str) and "api_key" in key.lower()) or
                             (key == "api.openrouter.api_key") or
                             (key == "api.anthropic.api_key") or
                             (key == "api.deepseek.api_key") or
                             (key == "api.litellm.api_key")
                        )

                        if is_api_key_field and isinstance(value, str):
                            provider_name = None
                            if key.startswith("api.") and key.endswith(".api_key"):
                                parts = key.split('.')
                                if len(parts) == 3: provider_name = parts[1]
                            elif "-api-key" in key: provider_name = key.replace("-api-key", "")

                            if provider_name:
                                # Only add if value is not empty
                                if value: api_keys_to_set[provider_name] = value
                                else: # If value is empty, treat as removal
                                     self.logger.info(f"Removing API key for provider {provider_name} due to empty value.")
                                     if not self.config_manager.remove_api_key(provider_name):
                                         self.logger.warning(f"Failed to remove API key for {provider_name} (already removed or error).")
                            else:
                                self.logger.warning(f"Could not determine provider for API key field: {key}")
                                regular_settings[key] = value
                        else:
                           regular_settings[key] = value

                    # Update regular settings
                    self.logger.debug(f"Updating regular settings: {regular_settings.keys()}")
                    for key, value in regular_settings.items():
                        if not self.config_manager.set_config(key, value):
                            self.logger.error(f"Failed to set regular setting: {key}")
                            settings_failed.append(key)
                    self.logger.debug("Finished updating regular settings.")

                    # Update API keys (secrets)
                    self.logger.debug(f"Updating API keys for providers: {api_keys_to_set.keys()}")
                    for provider, key_value in api_keys_to_set.items():
                        if not self.config_manager.set_api_key(provider, key_value):
                            keys_saved = False
                            api_keys_failed.append(provider)
                            self.logger.error(f"Failed to save API key for {provider}")
                    self.logger.debug("Finished updating API keys.")

                    if settings_failed or api_keys_failed:
                         error_msg = "Failed to update some settings."
                         if settings_failed: error_msg += f" Settings: {', '.join(settings_failed)}."
                         if api_keys_failed: error_msg += f" API Keys: {', '.join(api_keys_failed)}."
                         self.logger.warning(error_msg)
                         self.send_error_response(500, error_msg)
                         return

                    # Publish a general settings updated event
                    self.ui_manager.event_bus.publish_event("settings.updated", {"keys_updated": list(regular_settings.keys()) + list(api_keys_to_set.keys())})

                    self.logger.info("Settings update processed successfully. Sending success response.")
                    self.send_json_response(200, {"status": "success", "message": "Settings updated"})
                    self.logger.info("Settings updated via API (response sent).")

                except Exception as e:
                    self.logger.error(f"Error processing POST /api/settings: {e}", exc_info=True)
                    self.send_error_response(500, "Failed to update settings", str(e))


            def _handle_post_chat(self, body):
                """Handle POST /api/chat (Runs async task and waits)."""
                message_text = body.get("message")
                session_id = body.get("session_id") # Optional

                if not message_text:
                    self.send_error_response(400, "Missing 'message' field")
                    return

                # Define the async function to call
                async def process_chat_async():
                    return await self.chat_interface.process_message(message_text, session_id)

                # Run the async task and wait for result
                result, error = self.run_async_task(process_chat_async())

                # Handle result or error
                if error:
                    if isinstance(error, asyncio.TimeoutError):
                        self.send_error_response(504, "Request timed out while waiting for chat response.")
                    else:
                        self.send_error_response(500, "Error processing message", str(error))
                elif result is not None:
                    self.send_json_response(200, {"status": "success", "response": result})
                else:
                    self.send_error_response(500, "Internal server error: No result from chat processing task.")


            def _handle_post_discover_models(self):
                """Handle POST /api/discover_models (Runs async task and waits)."""

                # Define the async function to call
                async def discover_models_async():
                    await self.model_selection_interface.discover_models()
                    # Maybe return the count or list of models? For now, just return success.
                    return {"message": "Model discovery initiated and completed."} # Return data for JSON response

                # Run the async task and wait for result
                result, error = self.run_async_task(discover_models_async())

                # Handle result or error
                if error:
                    if isinstance(error, asyncio.TimeoutError):
                        self.send_error_response(504, "Request timed out while discovering models.")
                    else:
                        self.send_error_response(500, "Failed to run model discovery", str(error))
                elif result is not None:
                     # Send the result data back
                    self.send_json_response(200, {"status": "success", **result})
                else:
                    self.send_error_response(500, "Internal server error: No result from discovery task.")


            # --- Static File Serving ---
            def _serve_static_file(self, path):
                """Serve a static file."""
                if ".." in path:
                    self.send_error_response(403, "Forbidden path")
                    return

                if path == "/" or path == "":
                    file_path = Path(self.ui_manager.static_dir) / "index.html"
                else:
                    path = path[1:] if path.startswith("/") else path
                    file_path = Path(self.ui_manager.static_dir) / path

                if not file_path.is_file() or not str(file_path.resolve()).startswith(str(Path(self.ui_manager.static_dir).resolve())):
                    # Log before sending error
                    self.logger.debug(f"Static file not found or outside base directory: {file_path}")
                    self.send_error_response(404, f"File not found: {path}")
                    return

                content_type = self._get_content_type(file_path)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(file_path.stat().st_size))
                    if content_type in ["text/css", "application/javascript", "image/png", "image/jpeg", "image/gif", "image/svg+xml", "image/x-icon"]:
                        self.send_header("Cache-Control", "public, max-age=3600")
                    else:
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.end_headers()
                    with file_path.open("rb") as f:
                        self.wfile.write(f.read())
                except (BrokenPipeError, ConnectionResetError) as client_err:
                     self.logger.warning(f"Client connection error serving static file {file_path}: {client_err}")
                except Exception as e:
                    self.logger.error(f"Error serving static file {file_path}: {e}", exc_info=True)
                    # Avoid sending error if headers are already sent/broken


            def _get_content_type(self, file_path: Path) -> str:
                """Get the content type for a file."""
                content_types = { ".html": "text/html", ".htm": "text/html", ".css": "text/css", ".js": "application/javascript", ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".svg": "image/svg+xml", ".ico": "image/x-icon", ".txt": "text/plain"}
                return content_types.get(file_path.suffix.lower(), "application/octet-stream")
        # --- End of RequestHandler Class ---

        try:
            socketserver.TCPServer.allow_reuse_address = True
            self.httpd = HTTPServer((self.host, self.port), CustomHandler)
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, name="WebServerThread")
            self.server_thread.daemon = True
            self.server_thread.start()
            self.logger.info(f"Web server starting on http://{self.host}:{self.port}")
        except OSError as e:
             self.logger.critical(f"Failed to start web server on {self.host}:{self.port}: {e}", exc_info=True)
             self.error_handler.handle_error(e, ErrorCategory.NETWORK, ErrorSeverity.FATAL, {"host": self.host, "port": self.port})
             raise
        except Exception as e:
             self.logger.critical(f"Unexpected error starting web server: {e}", exc_info=True)
             self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.FATAL, {"action": "start_webserver"})
             raise

    def _on_settings_updated(self, event: Event) -> None:
        """Handle settings updated event."""
        self.logger.debug(f"Settings updated event received: {event.data}")

    # --- Public Methods to Control UI State ---
    def set_theme(self, theme: UITheme) -> None:
        self.config_manager.set_config("ui.theme", theme.name.lower())
        self.event_bus.publish_event("ui.theme.changed", {"theme": theme.name.lower()})
        self.logger.debug(f"Set theme to {theme.name}")

    def set_font_size(self, font_size: int) -> None:
        self.config_manager.set_config("ui.font_size", font_size)
        self.event_bus.publish_event("ui.fontsize.changed", {"size": font_size})
        self.logger.debug(f"Set font size to {font_size}")

    def toggle_animations(self, enabled: bool) -> None:
        self.config_manager.set_config("ui.animations_enabled", enabled)
        self.event_bus.publish_event("ui.animations.toggled", {"enabled": enabled})
        self.logger.debug(f"Set animations enabled to {enabled}")

    def toggle_compact_mode(self, enabled: bool) -> None:
        self.config_manager.set_config("ui.compact_mode", enabled)
        self.event_bus.publish_event("ui.compactmode.toggled", {"enabled": enabled})
        self.logger.debug(f"Set compact mode to {enabled}")

    def show_notification(self, message: str, type: str = "info", duration: int = 3000) -> None:
        self.event_bus.publish_event("ui.notification.show", {"message": message, "type": type, "duration": duration})
        self.logger.debug(f"Sent notification: {message}")

    def show_error(self, message: str) -> None:
        self.show_notification(message, type="error", duration=5000)
# END OF FILE miniManus-main/minimanus/ui/ui_manager.py
