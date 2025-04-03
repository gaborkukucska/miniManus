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
from typing import Dict, List, Optional, Any # Correct import added previously

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
            # Needs access to the UIManager instance to interact with other components
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
                # Add a check if headers are already sent (less likely with http.server but good practice)
                # if self.headers_sent:
                #     self.logger.warning("Attempted to send JSON response after headers were sent.")
                #     return
                try:
                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, OSError) as sock_err:
                     # Catch common errors when client disconnects during response sending
                     self.logger.warning(f"Socket error sending JSON response (client disconnected?): {sock_err}")
                except Exception as e:
                     self.logger.error(f"Unexpected error sending JSON response: {e}", exc_info=True)


            def send_error_response(self, status_code: int, message: str, error_details: Optional[str] = None):
                """Helper to send JSON error responses."""
                error_data = {"status": "error", "message": message}
                if error_details:
                    error_data["details"] = error_details
                self.send_json_response(status_code, error_data)

            # --- GET Handlers ---
            def do_GET(self):
                """Handle GET requests."""
                try:
                    parsed_url = urlparse(self.path)
                    path = parsed_url.path

                    if path.startswith("/api/"):
                        self._handle_api_get(path, parsed_url)
                    else:
                        self._serve_static_file(path)

                except (BrokenPipeError, ConnectionResetError) as client_err:
                    self.logger.warning(f"Client connection error during GET {self.path}: {client_err}")
                except Exception as e:
                    self.logger.error(f"GET request error for {self.path}: {e}", exc_info=True)
                    self.error_handler.handle_error(e, ErrorCategory.UI, ErrorSeverity.ERROR, {"path": self.path, "method": "GET"})
                    # Avoid sending error if headers might already be broken
                    # self.send_error_response(500, "Internal Server Error", str(e))

            def _handle_api_get(self, path: str, parsed_url):
                """Route GET API requests."""
                loop = self.ui_manager.get_event_loop() # Get the main loop
                if not loop or not loop.is_running():
                    self.logger.error("Event loop not available for async GET handler.")
                    self.send_error_response(500, "Internal server error: Event loop unavailable")
                    return

                if path == "/api/settings":
                    self._handle_get_settings()
                elif path == "/api/models":
                    # Run async handler in the main loop
                    asyncio.run_coroutine_threadsafe(self._handle_get_models(parsed_url), loop)
                elif path == "/api/sessions":
                    self._handle_get_sessions()
                else:
                    self.send_error_response(404, "API endpoint not found")

            def _handle_get_settings(self):
                """Handle GET /api/settings."""
                try:
                    settings = self.config_manager.get_config()
                    # Exclude sensitive data if secrets were mistakenly loaded into main config
                    if 'secrets' in settings: del settings['secrets']
                    if 'api_keys' in settings.get('api', {}): del settings['api']['api_keys']
                    self.send_json_response(200, settings)
                except Exception as e:
                    self.logger.error(f"Error getting settings: {e}", exc_info=True)
                    self.send_error_response(500, "Failed to retrieve settings", str(e))

            # --- Updated _handle_get_models with robust error handling ---
            async def _handle_get_models(self, parsed_url):
                """Handle GET /api/models (async)."""
                # Added outer try block for overall safety
                try:
                    query = parse_qs(parsed_url.query)
                    provider_name = query.get("provider", [None])[0]
                    models_info = []
                    models_dict = [] # Initialize models_dict

                    # Log entry
                    self.logger.debug(f"Handling GET /api/models?provider={provider_name}")

                    # Inner try block for the async operation
                    try:
                        if not provider_name:
                            models_info = await self.model_selection_interface.get_all_models()
                        else:
                            try:
                                provider_enum = APIProvider[provider_name.upper()]
                                models_info = await self.model_selection_interface.get_models_by_provider(provider_enum)
                            except KeyError:
                                # Send error response within the async context if possible
                                self.send_error_response(400, f"Invalid provider name: {provider_name}")
                                return # Exit async function
                            # Catch potential errors during model fetching itself
                            except Exception as model_fetch_error:
                                self.logger.error(f"Error getting models for provider {provider_name}: {model_fetch_error}", exc_info=True)
                                self.send_error_response(500, f"Failed to retrieve models for {provider_name}", str(model_fetch_error))
                                return # Exit async function

                        # Convert to dict *after* successful fetch
                        models_dict = [m.to_dict() for m in models_info]
                        self.logger.debug(f"Successfully fetched {len(models_dict)} models for provider '{provider_name or 'all'}'.")

                    except Exception as e:
                        # Catch errors from await calls or model processing
                        self.logger.error(f"Async error handling GET /api/models internal logic: {e}", exc_info=True)
                        self.send_error_response(500, "Internal server error processing models request.", str(e))
                        return # Exit async function

                    # Try sending the response, catching socket errors
                    try:
                         self.send_json_response(200, {"models": models_dict})
                         self.logger.debug(f"Sent model list response for provider '{provider_name or 'all'}'.")
                    except OSError as ose:
                         if ose.errno == 9: # Bad file descriptor
                             self.logger.warning(f"Socket closed before sending response for GET /api/models (OSError 9). Request likely interrupted.")
                         else:
                             self.logger.error(f"OSError sending response for GET /api/models: {ose}", exc_info=True)
                             # Avoid sending another error if socket is already broken
                    except Exception as send_e:
                         self.logger.error(f"Exception sending response for GET /api/models: {send_e}", exc_info=True)
                         # Avoid sending another error if socket is already broken

                except Exception as outer_e:
                    # Catch any unexpected error in the overall handler logic
                    self.logger.error(f"Unexpected outer error in _handle_get_models: {outer_e}", exc_info=True)
                    # Avoid sending response here if inner sending failed or socket is broken

            # --- End of Updated _handle_get_models ---


            def _handle_get_sessions(self):
                """Handle GET /api/sessions."""
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
                loop = self.ui_manager.get_event_loop() # Get the main loop
                if not loop or not loop.is_running():
                    self.logger.error("Event loop not available for POST handler.")
                    self.send_error_response(500, "Internal server error: Event loop unavailable")
                    return

                try:
                    parsed_url = urlparse(self.path)
                    path = parsed_url.path

                    content_length = int(self.headers.get("Content-Length", 0))
                    if content_length == 0:
                         body_raw = ""
                         body = {} # Assume empty dict if body is missing
                    else:
                        body_raw = self.rfile.read(content_length).decode("utf-8")
                        try:
                            body = json.loads(body_raw)
                        except json.JSONDecodeError:
                            self.send_error_response(400, "Invalid JSON format in request body")
                            return

                    if path == "/api/settings":
                        self._handle_post_settings(body) # Keep settings save synchronous
                    elif path == "/api/chat":
                        self._handle_post_chat(body) # This uses run_coroutine_threadsafe
                    elif path == "/api/discover_models":
                         # Run async handler in the main loop
                         asyncio.run_coroutine_threadsafe(self._handle_discover_models(), loop)
                    # Add other POST endpoints as needed
                    else:
                        self.send_error_response(404, "API endpoint not found")

                except (BrokenPipeError, ConnectionResetError) as client_err:
                    self.logger.warning(f"Client connection error during POST {self.path}: {client_err}")
                except Exception as e:
                    self.logger.error(f"POST request error for {self.path}: {e}", exc_info=True)
                    self.error_handler.handle_error(e, ErrorCategory.UI, ErrorSeverity.ERROR, {"path": self.path, "method": "POST"})
                    # Avoid sending error if socket is likely broken
                    # try:
                    #     self.send_error_response(500, "Internal Server Error", str(e))
                    # except:
                    #      self.logger.error("Failed to send error response for POST request.")


            def _handle_post_settings(self, body):
                """Handle POST /api/settings."""
                # Using previous version with detailed logging
                self.logger.info("Received POST /api/settings request.")
                try:
                    # Separate API keys from regular settings
                    api_keys_to_set = {}
                    regular_settings = {}
                    self.logger.debug(f"Processing settings body: {body}")

                    for key, value in body.items():
                        # Check for API key patterns (more robust check might be needed)
                        is_api_key_field = (
                            (isinstance(key, str) and "api_key" in key.lower()) or # Direct key name check
                            (key == "api.openrouter.api_key") or # Specific keys
                            (key == "api.anthropic.api_key") or
                            (key == "api.deepseek.api_key") or
                            (key == "api.litellm.api_key")
                        )

                        if is_api_key_field and isinstance(value, str):
                            # Extract provider name
                            provider_name = None
                            if key.startswith("api.") and key.endswith(".api_key"):
                                parts = key.split('.')
                                if len(parts) == 3:
                                     provider_name = parts[1]
                            elif "-api-key" in key: # Handle format like "openrouter-api-key"
                                 provider_name = key.replace("-api-key", "")

                            if provider_name:
                                api_keys_to_set[provider_name] = value
                            else:
                                self.logger.warning(f"Could not determine provider for API key field: {key}")
                                regular_settings[key] = value # Treat as regular if provider not clear
                        else:
                           regular_settings[key] = value

                    # Update regular settings
                    self.logger.debug(f"Updating regular settings: {regular_settings.keys()}")
                    for key, value in regular_settings.items():
                        if not self.config_manager.set_config(key, value): # set_config returns bool
                            self.logger.error(f"Failed to set regular setting: {key}")
                            # Decide how to handle partial failure
                    self.logger.debug("Finished updating regular settings.")

                    # Update API keys (secrets)
                    keys_saved = True
                    self.logger.debug(f"Updating API keys for providers: {api_keys_to_set.keys()}")
                    for provider, key_value in api_keys_to_set.items():
                        if not self.config_manager.set_api_key(provider, key_value):
                            keys_saved = False
                            self.logger.error(f"Failed to save API key for {provider}")
                    self.logger.debug("Finished updating API keys.")

                    if not keys_saved:
                         self.logger.warning("Failed to save one or more API keys.") # Log before sending error
                         self.send_error_response(500, "Failed to save one or more API keys")
                         return

                    # Publish a general settings updated event
                    self.ui_manager.event_bus.publish_event("settings.updated", {"keys_updated": list(regular_settings.keys()) + list(api_keys_to_set.keys())})

                    # *** Add log right before sending response ***
                    self.logger.info("Settings update processed successfully. Sending success response.")
                    self.send_json_response(200, {"status": "success", "message": "Settings updated"})
                    self.logger.info("Settings updated via API (response sent).") # Log after sending

                except Exception as e:
                    # Log the specific exception *before* sending error response
                    self.logger.error(f"Error processing POST /api/settings: {e}", exc_info=True)
                    try:
                         self.send_error_response(500, "Failed to update settings", str(e))
                    except:
                         self.logger.error("Failed to send error response for POST /api/settings.")


            def _handle_post_chat(self, body):
                """Handle POST /api/chat."""
                message_text = body.get("message")
                session_id = body.get("session_id") # Optional: UI might specify session

                if not message_text:
                    self.send_error_response(400, "Missing 'message' field in request body")
                    return

                # Get the event loop the UIManager is running in
                loop = self.ui_manager.get_event_loop()
                if not loop or not loop.is_running():
                    self.logger.error("Event loop not available for async chat processing.")
                    self.send_error_response(500, "Internal server error: Event loop unavailable")
                    return

                try:
                    # Run the async `process_message` coroutine in the main event loop
                    # Use run_coroutine_threadsafe because the HTTP server runs in a separate thread
                    future = asyncio.run_coroutine_threadsafe(
                        self.chat_interface.process_message(message_text, session_id),
                        loop
                    )
                    # Wait for the result (consider adding a timeout)
                    response_text = future.result(timeout=120) # 120 second timeout for LLM response

                    self.send_json_response(200, {"status": "success", "response": response_text})
                    self.logger.info(f"Chat message processed via API: {message_text[:50]}...")

                except asyncio.TimeoutError:
                     self.logger.error("Timeout waiting for chat response.")
                     self.send_error_response(504, "Request timed out while waiting for response.")
                except Exception as e:
                    self.logger.error(f"Error processing chat message via API: {e}", exc_info=True)
                    self.error_handler.handle_error(e, ErrorCategory.UI, ErrorSeverity.ERROR, {"action": "post_chat"})
                    self.send_error_response(500, "Error processing message", str(e))

            async def _handle_discover_models(self):
                 """Handle POST /api/discover_models (async)."""
                 try:
                     self.logger.info("Received request to discover models.")
                     await self.model_selection_interface.discover_models()
                     self.logger.info("Model discovery initiated successfully.")
                     # Send response back from the async handler
                     try:
                         self.send_json_response(200, {"status": "success", "message": "Model discovery initiated."})
                     except OSError as ose:
                         if ose.errno == 9: # Bad file descriptor
                             self.logger.warning(f"Socket closed before sending discovery response (OSError 9).")
                         else:
                             self.logger.error(f"OSError sending discovery response: {ose}", exc_info=True)
                     except Exception as send_e:
                         self.logger.error(f"Exception sending discovery response: {send_e}", exc_info=True)

                 except Exception as e:
                     self.logger.error(f"Error initiating model discovery: {e}", exc_info=True)
                     # Try sending error response, catching potential socket errors
                     try:
                         self.send_error_response(500, "Failed to start model discovery", str(e))
                     except OSError as ose_err:
                         if ose_err.errno == 9:
                              self.logger.warning(f"Socket closed before sending discovery error response (OSError 9).")
                         else:
                              self.logger.error(f"OSError sending discovery error response: {ose_err}", exc_info=True)
                     except Exception as send_err_e:
                          self.logger.error(f"Exception sending discovery error response: {send_err_e}", exc_info=True)


            # --- Static File Serving ---
            def _serve_static_file(self, path):
                """Serve a static file."""
                # Basic security: prevent directory traversal
                if ".." in path:
                    self.send_error_response(403, "Forbidden path")
                    return

                if path == "/" or path == "":
                    file_path = Path(self.ui_manager.static_dir) / "index.html"
                else:
                    # Remove leading slash
                    path = path[1:] if path.startswith("/") else path
                    file_path = Path(self.ui_manager.static_dir) / path

                # Check if file exists and is within the static directory
                if not file_path.is_file() or not str(file_path.resolve()).startswith(str(Path(self.ui_manager.static_dir).resolve())):
                    self.send_error_response(404, f"File not found: {path}")
                    return

                # Determine content type
                content_type = self._get_content_type(file_path)

                # Send the file
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(file_path.stat().st_size))
                    # Add Cache-Control header for better performance
                    if content_type in ["text/css", "application/javascript", "image/png", "image/jpeg", "image/gif", "image/svg+xml", "image/x-icon"]:
                        self.send_header("Cache-Control", "public, max-age=3600") # Cache for 1 hour
                    else:
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")

                    self.end_headers()
                    with file_path.open("rb") as f:
                        self.wfile.write(f.read())
                except (BrokenPipeError, ConnectionResetError) as client_err:
                     self.logger.warning(f"Client connection error serving static file {file_path}: {client_err}")
                except Exception as e:
                    self.logger.error(f"Error serving static file {file_path}: {e}", exc_info=True)
                    self.error_handler.handle_error(e, ErrorCategory.UI, ErrorSeverity.ERROR, {"path": path, "file_path": str(file_path)})
                    # Don't send another response if headers might already be sent


            def _get_content_type(self, file_path: Path) -> str:
                """Get the content type for a file."""
                content_types = {
                    ".html": "text/html", ".htm": "text/html",
                    ".css": "text/css",
                    ".js": "application/javascript",
                    ".json": "application/json",
                    ".png": "image/png",
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".svg": "image/svg+xml",
                    ".ico": "image/x-icon",
                    ".txt": "text/plain",
                    # Add more types as needed
                }
                return content_types.get(file_path.suffix.lower(), "application/octet-stream")
        # --- End of RequestHandler Class ---

        try:
            # Enable address reuse to prevent "Address already in use" errors on restart
            socketserver.TCPServer.allow_reuse_address = True

            # Create and start the server
            self.httpd = HTTPServer((self.host, self.port), CustomHandler)

            # Start the server in a separate thread
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, name="WebServerThread")
            self.server_thread.daemon = True # Allow program exit even if thread is running
            self.server_thread.start()

            self.logger.info(f"Web server starting on http://{self.host}:{self.port}")

        except OSError as e:
             self.logger.critical(f"Failed to start web server on {self.host}:{self.port}: {e}", exc_info=True)
             self.error_handler.handle_error(e, ErrorCategory.NETWORK, ErrorSeverity.FATAL, {"host": self.host, "port": self.port})
             # Consider shutting down the whole application if the UI server is critical
             # system_manager = SystemManager.get_instance() # Assuming you have access or pass it
             # system_manager.shutdown(force_exit=True)
             # Or raise the exception to be caught by the main starter
             raise
        except Exception as e:
             self.logger.critical(f"Unexpected error starting web server: {e}", exc_info=True)
             self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.FATAL, {"action": "start_webserver"})
             raise

    def _on_settings_updated(self, event: Event) -> None:
        """Handle settings updated event."""
        # Example: Reload config if needed, though ConfigManager handles this
        self.logger.debug(f"Settings updated event received: {event.data}")
        # Add specific actions if UI needs to react directly to setting changes

    # --- Public Methods to Control UI State (Called from other components) ---

    def set_theme(self, theme: UITheme) -> None:
        """
        Set the UI theme (sends event, UI needs to listen).

        Args:
            theme: Theme to set
        """
        # This method might just update config; the actual UI change happens client-side
        self.config_manager.set_config("ui.theme", theme.name.lower())
        self.event_bus.publish_event("ui.theme.changed", {"theme": theme.name.lower()})
        self.logger.debug(f"Set theme to {theme.name}")

    def set_font_size(self, font_size: int) -> None:
        """Set the UI font size (sends event)."""
        self.config_manager.set_config("ui.font_size", font_size)
        self.event_bus.publish_event("ui.fontsize.changed", {"size": font_size})
        self.logger.debug(f"Set font size to {font_size}")

    def toggle_animations(self, enabled: bool) -> None:
        """Toggle UI animations (sends event)."""
        self.config_manager.set_config("ui.animations_enabled", enabled)
        self.event_bus.publish_event("ui.animations.toggled", {"enabled": enabled})
        self.logger.debug(f"Set animations enabled to {enabled}")

    def toggle_compact_mode(self, enabled: bool) -> None:
        """Toggle UI compact mode (sends event)."""
        self.config_manager.set_config("ui.compact_mode", enabled)
        self.event_bus.publish_event("ui.compactmode.toggled", {"enabled": enabled})
        self.logger.debug(f"Set compact mode to {enabled}")

    def show_notification(self, message: str, type: str = "info", duration: int = 3000) -> None:
        """
        Show a notification in the UI (sends event).

        Args:
            message: Notification message
            type: Notification type (info, success, warning, error)
            duration: Duration in milliseconds
        """
        self.event_bus.publish_event("ui.notification.show", {
            "message": message,
            "type": type,
            "duration": duration
        })
        self.logger.debug(f"Sent notification: {message}")

    def show_error(self, message: str) -> None:
        """Show an error message in the UI (uses notification event)."""
        self.show_notification(message, type="error", duration=5000)
