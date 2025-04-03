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
        try:
            self._loop = asyncio.get_running_loop() # Capture the loop
        except RuntimeError:
            self.logger.error("Could not get running event loop during UIManager startup. Async operations might fail.")
            self._loop = None
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
                # Flag to prevent sending multiple error responses
                self._headers_sent_flag = False
                super().__init__(*args, **kwargs)

            def log_message(self, format, *args):
                # Redirect log messages to our logger
                self.logger.debug(format % args)

            def send_response(self, code, message=None):
                """Override to track if headers were sent."""
                super().send_response(code, message)
                self._headers_sent_flag = True

            def send_header(self, keyword, value):
                """Override to track if headers were sent."""
                super().send_header(keyword, value)
                self._headers_sent_flag = True

            def end_headers(self):
                """Override to track if headers were sent."""
                super().end_headers()
                self._headers_sent_flag = True

            def send_json_response(self, status_code: int, data: Dict[str, Any]):
                """Helper to send JSON responses."""
                if self._headers_sent_flag:
                     self.logger.warning("Attempted to send JSON response after headers were already sent.")
                     return
                try:
                    response_body = json.dumps(data).encode("utf-8")
                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(response_body)))
                    self.end_headers()
                    self.wfile.write(response_body)
                except (BrokenPipeError, ConnectionResetError, OSError) as sock_err:
                     self.logger.warning(f"Socket error sending JSON response (client disconnected?): {sock_err}")
                except Exception as e:
                     self.logger.error(f"Unexpected error sending JSON response: {e}", exc_info=True)

            def send_error_response(self, status_code: int, message: str, error_details: Optional[str] = None):
                """Helper to send JSON error responses."""
                if self._headers_sent_flag:
                    self.logger.warning(f"Attempted to send error response '{message}' after headers were already sent.")
                    return
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

                # Get coro name safely for logging
                coro_name = getattr(coro, '__name__', str(coro))
                self.logger.debug(f"Scheduling async task: {coro_name}")

                future = asyncio.run_coroutine_threadsafe(coro, loop)
                try:
                    # Wait for the result with a timeout
                    result = future.result(timeout=self.ui_manager.request_timeout)
                    self.logger.debug(f"Async task {coro_name} completed successfully.")
                    return result, None
                except asyncio.TimeoutError as e:
                    self.logger.error(f"Timeout waiting for async task result ({self.ui_manager.request_timeout}s). Task: {coro_name}")
                    # Attempt to cancel the future if it timed out
                    if not future.done():
                         future.cancel()
                    return None, e
                except Exception as e:
                    self.logger.error(f"Exception during async task execution: {e}. Task: {coro_name}", exc_info=True)
                    return None, e

            # --- GET Handlers ---
            def do_GET(self):
                """Handle GET requests."""
                # Reset headers sent flag for each request
                self._headers_sent_flag = False
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
                        # Send error only if headers haven't been sent
                        if not self._headers_sent_flag:
                             self.send_error_response(500, "Internal Server Error", str(e))
                    except Exception as e_resp:
                         self.logger.error(f"Failed to send error response for GET {self.path}: {e_resp}")

            def _handle_get_settings(self):
                """Handle GET /api/settings (Synchronous)."""
                try:
                    settings = self.config_manager.get_config() # Get a copy
                    # Ensure sensitive info is stripped before sending
                    if settings:
                        # Use .pop() with default to avoid errors if keys are missing
                        settings.pop('secrets', None)
                        api_settings = settings.get('api')
                        if isinstance(api_settings, dict):
                            api_settings.pop('api_keys', None)
                            # Also remove individual provider keys if they exist at this level (they shouldn't)
                            providers = list(api_settings.get("providers", {}).keys())
                            for p_key in list(api_settings.keys()):
                                if "api_key" in p_key:
                                    api_settings.pop(p_key, None)

                    self.send_json_response(200, settings or {}) # Send empty dict if settings is None
                except Exception as e:
                    self.logger.error(f"Error getting settings: {e}", exc_info=True)
                    self.send_error_response(500, "Failed to retrieve settings", str(e))

            def _handle_get_models(self, parsed_url):
                """Handle GET /api/models (Runs async task and waits)."""
                query = parse_qs(parsed_url.query)
                provider_name = query.get("provider", [None])[0]

                async def get_models_async():
                    if not provider_name:
                        models_info = await self.model_selection_interface.get_all_models()
                    else:
                        try:
                            provider_enum = APIProvider[provider_name.upper()]
                            models_info = await self.model_selection_interface.get_models_by_provider(provider_enum)
                        except KeyError:
                            raise ValueError(f"Invalid provider name: {provider_name}")
                        except Exception as e:
                             raise RuntimeError(f"Failed to retrieve models for {provider_name}") from e
                    return [m.to_dict() for m in models_info]

                result, error = self.run_async_task(get_models_async())

                if error:
                    if isinstance(error, ValueError):
                        self.send_error_response(400, str(error))
                    elif isinstance(error, asyncio.TimeoutError):
                        self.send_error_response(504, "Request timed out while fetching models.")
                    else:
                        self.send_error_response(500, "Internal server error processing models request.", str(error))
                elif result is not None:
                    self.send_json_response(200, {"models": result})
                else:
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
                # Reset headers sent flag
                self._headers_sent_flag = False
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
                        if not self._headers_sent_flag:
                             self.send_error_response(500, "Internal Server Error", str(e))
                    except Exception as e_resp:
                         self.logger.error(f"Failed to send error response for POST {self.path}: {e_resp}")


            def _handle_post_settings(self, body):
                """Handle POST /api/settings (Synchronous)."""
                self.logger.info("Received POST /api/settings request.")
                self.logger.debug(f"Raw settings body received: {json.dumps(body)}") # Log the received body

                api_keys_failed = []
                settings_failed = []
                keys_updated_count = 0
                secrets_updated_count = 0

                try:
                    # Process the combined payload
                    api_keys_to_set = {}
                    api_keys_to_remove = set()
                    regular_settings = {}

                    for key, value in body.items():
                        # --- Identify API Key Fields ---
                        # Use a more robust check based on expected patterns
                        is_api_key_field = False
                        provider_for_key = None
                        # Pattern 1: Direct keys like "provider-api-key" from JS
                        if isinstance(key, str) and key.endswith("-api-key"):
                            provider_for_key = key[:-len("-api-key")]
                            is_api_key_field = provider_for_key in [p.name.lower() for p in APIProvider]

                        # Pattern 2: Config keys like "api.providers.providername.api_key" (less likely from JS form)
                        elif isinstance(key, str) and key.startswith("api.providers.") and key.endswith(".api_key"):
                            parts = key.split('.')
                            if len(parts) == 4:
                                provider_for_key = parts[2]
                                is_api_key_field = provider_for_key in [p.name.lower() for p in APIProvider]

                        # --- Process API Keys vs Regular Settings ---
                        if is_api_key_field:
                            if isinstance(value, str):
                                if value: # If key has a value, plan to set it
                                    api_keys_to_set[provider_for_key] = value
                                else: # If key is present but value is empty, plan to remove it
                                    api_keys_to_remove.add(provider_for_key)
                            else:
                                self.logger.warning(f"Received non-string value for API key field '{key}'. Ignoring.")
                        else:
                           regular_settings[key] = value

                    # --- Update Regular Settings ---
                    self.logger.debug(f"Attempting to set regular config: {list(regular_settings.keys())}")
                    for key, value in regular_settings.items():
                        self.logger.debug(f"Calling set_config('{key}', value type: {type(value)})")
                        if not self.config_manager.set_config(key, value):
                            self.logger.error(f"Failed to set regular setting: {key}")
                            settings_failed.append(key)
                        else:
                            keys_updated_count += 1

                    # --- Update API Keys (Secrets) ---
                    self.logger.debug(f"Attempting to set API keys for: {list(api_keys_to_set.keys())}")
                    for provider, key_value in api_keys_to_set.items():
                        self.logger.debug(f"Calling set_api_key('{provider}', '***')")
                        if not self.config_manager.set_api_key(provider, key_value):
                            api_keys_failed.append(provider)
                            self.logger.error(f"Failed to save API key for {provider}")
                        else:
                            secrets_updated_count +=1

                    self.logger.debug(f"Attempting to remove API keys for: {list(api_keys_to_remove)}")
                    for provider in api_keys_to_remove:
                        # Only remove if it wasn't also set in the same request
                        if provider not in api_keys_to_set:
                            self.logger.debug(f"Calling remove_api_key('{provider}')")
                            if not self.config_manager.remove_api_key(provider):
                                # This might just mean it wasn't there, which is fine
                                self.logger.debug(f"Failed to remove API key for {provider} (may have already been removed or error occurred).")
                            else:
                                secrets_updated_count +=1 # Count removal as an update


                    if settings_failed or api_keys_failed:
                         error_msg = "Failed to update some settings."
                         if settings_failed: error_msg += f" Settings: {', '.join(settings_failed)}."
                         if api_keys_failed: error_msg += f" API Keys: {', '.join(api_keys_failed)}."
                         self.logger.warning(error_msg)
                         self.send_error_response(500, error_msg)
                         return

                    # Publish a general settings updated event
                    if keys_updated_count > 0 or secrets_updated_count > 0:
                        updated_keys_list = list(regular_settings.keys()) + [f"{p}_api_key" for p in api_keys_to_set] + [f"{p}_api_key" for p in api_keys_to_remove]
                        self.ui_manager.event_bus.publish_event("settings.updated", {"keys_updated": updated_keys_list})

                    self.logger.info(f"Settings update processed. Regular: {keys_updated_count}, Secrets: {secrets_updated_count}. Sending success response.")
                    self.send_json_response(200, {"status": "success", "message": "Settings updated"})

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

                async def process_chat_async():
                    return await self.chat_interface.process_message(message_text, session_id)

                result, error = self.run_async_task(process_chat_async())

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

                async def discover_models_async():
                    await self.model_selection_interface.discover_models()
                    all_models = await self.model_selection_interface.get_all_models()
                    return {"message": f"Model discovery complete. Found {len(all_models)} models."}

                result, error = self.run_async_task(discover_models_async())

                if error:
                    if isinstance(error, asyncio.TimeoutError):
                        self.send_error_response(504, "Request timed out while discovering models.")
                    else:
                        self.send_error_response(500, "Failed to run model discovery", str(error))
                elif result is not None:
                    self.send_json_response(200, {"status": "success", **result})
                else:
                    self.send_error_response(500, "Internal server error: No result from discovery task.")

            # --- Static File Serving ---
            def _serve_static_file(self, path):
                """Serve a static file."""
                # Prevent directory traversal
                if ".." in path:
                    self.send_error_response(403, "Forbidden path")
                    return

                # Determine the absolute file path
                requested_path = path[1:] if path.startswith("/") else path
                abs_static_dir = Path(self.ui_manager.static_dir).resolve()
                file_path = (abs_static_dir / requested_path).resolve()

                # Serve index.html for root path
                if path == "/" or path == "":
                    file_path = abs_static_dir / "index.html"

                # Security Check: Ensure the resolved path is still within the static directory
                if not file_path.is_file() or not str(file_path).startswith(str(abs_static_dir)):
                    self.logger.debug(f"Static file not found or outside base directory: {file_path}")
                    self.send_error_response(404, f"File not found: {path}")
                    return

                content_type = self._get_content_type(file_path)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(file_path.stat().st_size))
                    # Basic Caching Headers
                    if content_type in ["text/css", "application/javascript", "image/png", "image/jpeg", "image/gif", "image/svg+xml", "image/x-icon"]:
                        self.send_header("Cache-Control", "public, max-age=3600") # Cache assets for 1 hour
                    else: # HTML, etc.
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                        self.send_header("Pragma", "no-cache")
                        self.send_header("Expires", "0")
                    self.end_headers()
                    with file_path.open("rb") as f:
                        self.wfile.write(f.read())
                except (BrokenPipeError, ConnectionResetError) as client_err:
                     self.logger.warning(f"Client connection error serving static file {file_path}: {client_err}")
                except FileNotFoundError: # Should be caught by earlier check, but handle defensively
                    self.logger.error(f"Static file disappeared before serving: {file_path}")
                    if not self._headers_sent_flag: self.send_error_response(404, f"File not found: {path}")
                except Exception as e:
                    self.logger.error(f"Error serving static file {file_path}: {e}", exc_info=True)
                    if not self._headers_sent_flag: self.send_error_response(500, "Error serving file")

            def _get_content_type(self, file_path: Path) -> str:
                """Get the content type for a file."""
                content_types = { ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript", ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".svg": "image/svg+xml", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8"}
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
             self.error_handler.handle_error(e, ErrorCategory.NETWORK, ErrorSeverity.CRITICAL, {"host": self.host, "port": self.port})
             # Reraise to potentially stop the application startup
             raise RuntimeError(f"Could not start web server: {e}") from e
        except Exception as e:
             self.logger.critical(f"Unexpected error starting web server: {e}", exc_info=True)
             self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL, {"action": "start_webserver"})
             raise RuntimeError(f"Unexpected error during web server start: {e}") from e

    def _on_settings_updated(self, event: Event) -> None:
        """Handle settings updated event."""
        self.logger.debug(f"Settings updated event received: {event.data}")
        # Could potentially push updates to connected clients via websockets if implemented

    # --- Public Methods to Control UI State (Called from other backend components) ---
    # These methods primarily interact via ConfigManager and EventBus now

    def set_theme(self, theme: UITheme) -> None:
        """Sets the theme in configuration and publishes an event."""
        success = self.config_manager.set_config("ui.theme", theme.name.lower())
        if success:
            self.event_bus.publish_event("ui.theme.changed", {"theme": theme.name.lower()})
            self.logger.debug(f"Set theme to {theme.name}")
        else:
            self.logger.error("Failed to save theme setting.")

    def set_font_size(self, font_size: int) -> None:
        """Sets the font size in configuration and publishes an event."""
        success = self.config_manager.set_config("ui.font_size", font_size)
        if success:
            self.event_bus.publish_event("ui.fontsize.changed", {"size": font_size})
            self.logger.debug(f"Set font size to {font_size}")
        else:
            self.logger.error("Failed to save font size setting.")

    def toggle_animations(self, enabled: bool) -> None:
        """Sets animation state in configuration and publishes an event."""
        success = self.config_manager.set_config("ui.animations_enabled", enabled)
        if success:
            self.event_bus.publish_event("ui.animations.toggled", {"enabled": enabled})
            self.logger.debug(f"Set animations enabled to {enabled}")
        else:
             self.logger.error("Failed to save animation setting.")

    def toggle_compact_mode(self, enabled: bool) -> None:
        """Sets compact mode state in configuration and publishes an event."""
        success = self.config_manager.set_config("ui.compact_mode", enabled)
        if success:
            self.event_bus.publish_event("ui.compactmode.toggled", {"enabled": enabled})
            self.logger.debug(f"Set compact mode to {enabled}")
        else:
            self.logger.error("Failed to save compact mode setting.")

    def show_notification(self, message: str, type: str = "info", duration: int = 3000) -> None:
        """Publishes an event to show a notification in the UI (requires frontend handling)."""
        self.event_bus.publish_event("ui.notification.show", {"message": message, "type": type, "duration": duration})
        self.logger.debug(f"Sent notification event: {message}")

    def show_error(self, message: str) -> None:
        """Publishes an event to show an error notification in the UI."""
        self.show_notification(message, type="error", duration=5000)

# END OF FILE miniManus-main/minimanus/ui/ui_manager.py
