# miniManus Framework Changes

## Issues Fixed

### 1. Coroutine Not Awaited Warning
- **File**: `__main__.py`
- **Issue**: The async method `model_selection.discover_models()` was called without using `await`, causing the "coroutine was never awaited" warning.
- **Fix**: Added `await` keyword to properly handle the async call.

```python
# Before
model_selection.discover_models()

# After
await model_selection.discover_models()
```

### 2. Missing POST Handler for Chat API
- **File**: `ui_manager.py`
- **Issue**: The `CustomHandler` class didn't implement a `do_POST` method, causing 501 (Not Implemented) errors when trying to send messages through the UI.
- **Fix**: Added a complete `do_POST` method to handle POST requests to the `/api/chat` endpoint.

```python
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
```

### 3. Improper Port Release During Shutdown
- **File**: `ui_manager.py`
- **Issue**: The shutdown method only called `httpd.shutdown()` which stops the serve_forever() loop but doesn't close the socket, leaving the port in use after termination.
- **Fix**: Enhanced the shutdown method to properly release the port:

```python
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
```

- Also added `socketserver.TCPServer.allow_reuse_address = True` to prevent "Address already in use" errors on restart.

### 4. Incomplete Application Shutdown Process
- **Files**: `system_manager.py` and `__main__.py`
- **Issue**: The application wouldn't terminate properly when Ctrl+C was pressed, getting stuck with "Shutdown already in progress" messages.
- **Fix**: Implemented a comprehensive shutdown solution:

In `system_manager.py`:
```python
# Added a shutdown event to signal when shutdown is complete
_shutdown_event = threading.Event()

# Enhanced shutdown method with force_exit parameter
def shutdown(self, component_order: Optional[List[str]] = None, force_exit: bool = False) -> None:
    # ...
    # Signal that shutdown is complete
    SystemManager._shutdown_event.set()
    
    # If force_exit is True, exit the process
    if force_exit:
        self.logger.info("Exiting process...")
        sys.exit(0)

# Improved signal handler with timeout
def _handle_signal(self, signum: int, frame) -> None:
    # ...
    # Start shutdown in a separate thread
    shutdown_thread = threading.Thread(target=self.shutdown, kwargs={"force_exit": True})
    shutdown_thread.daemon = True
    shutdown_thread.start()
    
    # Wait for shutdown to complete with timeout
    if not SystemManager._shutdown_event.wait(self._shutdown_timeout):
        self.logger.warning(f"Shutdown timed out after {self._shutdown_timeout} seconds, forcing exit...")
        os._exit(1)  # Force exit if shutdown times out
```

In `__main__.py`:
```python
# Improved async loop handling
async def main_async():
    # ...
    # Keep the main thread alive until shutdown is requested
    while not shutdown_requested:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Main loop cancelled, initiating shutdown...")
            break

def main():
    # ...
    # Run the async main function with proper task cancellation
    loop = asyncio.get_event_loop()
    main_task = loop.create_task(main_async())
    
    try:
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        # Cancel the main task
        main_task.cancel()
        
        try:
            # Wait for the task to be cancelled
            loop.run_until_complete(main_task)
        except asyncio.CancelledError:
            pass
        
        # Ensure system manager shutdown is called
        if system_manager and not system_manager.is_shutting_down:
            system_manager.shutdown(force_exit=True)
```

## Testing Instructions

1. Replace your current miniManus code with the fixed version
2. Run the framework using `python -m minimanus`
3. Open the UI in your browser at `http://localhost:8080`
4. Send a message through the chat interface
5. Verify that you receive a response without any errors
6. Test the shutdown by pressing Ctrl+C and verify that the application terminates cleanly
7. Restart the application and verify that there are no "Address already in use" errors

## Future Improvements

1. Implement actual LLM API calls in the chat handler instead of just echoing messages
2. Add proper error handling for API calls
3. Implement streaming responses for better user experience
4. Add authentication for API endpoints
5. Improve the UI with more features and better styling
