# Function Index for miniManus

This document provides a comprehensive index of all functions in the miniManus framework, categorized by module. This index is designed to help LLM models accurately reference and use functions without typos or slight variations that could break function calls.

## Core Framework

### SystemManager (`src/core/system_manager.py`)

- `get_instance()` - Get or create the singleton instance of SystemManager
- `startup()` - Start the system manager and initialize all components
- `shutdown()` - Stop the system manager and clean up resources
- `get_system_info()` - Get information about the system
- `register_component()` - Register a system component
- `get_component()` - Get a registered system component
- `check_system_health()` - Check the health of the system
- `handle_system_event()` - Handle a system event

### ConfigurationManager (`src/core/config_manager.py`)

- `get_instance()` - Get or create the singleton instance of ConfigurationManager
- `get_config()` - Get a configuration value
- `set_config()` - Set a configuration value
- `delete_config()` - Delete a configuration value
- `get_all_configs()` - Get all configuration values
- `load_config()` - Load configuration from file
- `save_config()` - Save configuration to file
- `get_api_key()` - Get an API key
- `set_api_key()` - Set an API key
- `delete_api_key()` - Delete an API key

### EventBus (`src/core/event_bus.py`)

- `get_instance()` - Get or create the singleton instance of EventBus
- `startup()` - Start the event bus
- `shutdown()` - Stop the event bus
- `subscribe()` - Subscribe to an event
- `unsubscribe()` - Unsubscribe from an event
- `publish_event()` - Publish an event
- `wait_for_event()` - Wait for an event to occur

### ResourceMonitor (`src/core/resource_monitor.py`)

- `get_instance()` - Get or create the singleton instance of ResourceMonitor
- `startup()` - Start the resource monitor
- `shutdown()` - Stop the resource monitor
- `get_memory_usage()` - Get current memory usage
- `get_cpu_usage()` - Get current CPU usage
- `get_storage_usage()` - Get current storage usage
- `get_battery_status()` - Get current battery status
- `register_cleanup_callback()` - Register a callback for resource cleanup
- `unregister_cleanup_callback()` - Unregister a cleanup callback
- `check_resources()` - Check resource usage and trigger warnings if needed

### ErrorHandler (`src/core/error_handler.py`)

- `get_instance()` - Get or create the singleton instance of ErrorHandler
- `handle_error()` - Handle an error
- `get_error_history()` - Get error history
- `clear_error_history()` - Clear error history
- `set_error_callback()` - Set a callback for error handling
- `remove_error_callback()` - Remove an error callback

### PluginManager (`src/core/plugin_manager.py`)

- `get_instance()` - Get or create the singleton instance of PluginManager
- `startup()` - Start the plugin manager
- `shutdown()` - Stop the plugin manager
- `set_plugin_interface()` - Set the plugin interface class
- `add_plugin_directory()` - Add a directory to search for plugins
- `discover_plugins()` - Discover plugins in registered directories
- `load_plugin()` - Load a plugin module
- `enable_plugin()` - Enable a loaded plugin
- `disable_plugin()` - Disable an enabled plugin
- `unload_plugin()` - Unload a plugin
- `get_plugin_info()` - Get information about a plugin
- `get_all_plugins()` - Get information about all plugins

## API Integrations

### APIManager (`src/api/api_manager.py`)

- `get_instance()` - Get or create the singleton instance of APIManager
- `startup()` - Start the API manager
- `shutdown()` - Stop the API manager
- `register_adapter()` - Register an adapter for a provider
- `get_adapter()` - Get the adapter for a provider
- `check_provider_availability()` - Check if a provider is available
- `get_available_providers()` - Get a list of available providers
- `select_provider()` - Select a provider for a request
- `check_rate_limit()` - Check if a provider is rate limited
- `update_rate_limit()` - Update rate limit for a provider
- `record_request()` - Record a request to a provider for rate limiting
- `get_cache_key()` - Generate a cache key for a request
- `get_from_cache()` - Get a response from the cache
- `add_to_cache()` - Add a response to the cache
- `clear_cache()` - Clear the response cache
- `send_request()` - Send a request to an API provider

### OpenRouterAdapter (`src/api/openrouter_adapter.py`)

- `check_availability()` - Check if the OpenRouter API is available
- `get_available_models()` - Get list of available models from OpenRouter
- `send_chat_request()` - Send a chat completion request to OpenRouter
- `send_completion_request()` - Send a text completion request to OpenRouter
- `send_embedding_request()` - Send an embedding request to OpenRouter
- `send_image_request()` - Send an image generation request to OpenRouter
- `send_audio_request()` - Send an audio processing request to OpenRouter

### DeepSeekAdapter (`src/api/deepseek_adapter.py`)

- `check_availability()` - Check if the DeepSeek API is available
- `get_available_models()` - Get list of available models from DeepSeek
- `send_chat_request()` - Send a chat completion request to DeepSeek
- `send_completion_request()` - Send a text completion request to DeepSeek
- `send_embedding_request()` - Send an embedding request to DeepSeek
- `send_image_request()` - Send an image generation request to DeepSeek
- `send_audio_request()` - Send an audio processing request to DeepSeek

### AnthropicAdapter (`src/api/anthropic_adapter.py`)

- `check_availability()` - Check if the Anthropic API is available
- `get_available_models()` - Get list of available models from Anthropic
- `send_chat_request()` - Send a chat completion request to Anthropic
- `send_completion_request()` - Send a text completion request to Anthropic
- `send_embedding_request()` - Send an embedding request to Anthropic
- `send_image_request()` - Send an image generation request to Anthropic
- `send_audio_request()` - Send an audio processing request to Anthropic

### OllamaAdapter (`src/api/ollama_adapter.py`)

- `discover_ollama_servers()` - Discover Ollama servers on the local network
- `check_availability()` - Check if the Ollama API is available
- `get_available_models()` - Get list of available models from Ollama
- `send_chat_request()` - Send a chat completion request to Ollama
- `send_completion_request()` - Send a text completion request to Ollama
- `send_embedding_request()` - Send an embedding request to Ollama
- `send_image_request()` - Send an image generation request to Ollama
- `send_audio_request()` - Send an audio processing request to Ollama

### LiteLLMAdapter (`src/api/litellm_adapter.py`)

- `discover_litellm_servers()` - Discover LiteLLM servers on the local network
- `check_availability()` - Check if the LiteLLM API is available
- `get_available_models()` - Get list of available models from LiteLLM
- `send_chat_request()` - Send a chat completion request to LiteLLM
- `send_completion_request()` - Send a text completion request to LiteLLM
- `send_embedding_request()` - Send an embedding request to LiteLLM
- `send_image_request()` - Send an image generation request to LiteLLM
- `send_audio_request()` - Send an audio processing request to LiteLLM

## UI Components

### UIManager (`src/ui/ui_manager.py`)

- `get_instance()` - Get or create the singleton instance of UIManager
- `startup()` - Start the UI manager
- `shutdown()` - Stop the UI manager
- `register_component()` - Register a UI component
- `get_component()` - Get a UI component
- `set_theme()` - Set the UI theme
- `set_font_size()` - Set the UI font size
- `toggle_animations()` - Toggle UI animations
- `toggle_compact_mode()` - Toggle UI compact mode
- `change_view()` - Change the current UI view
- `show_notification()` - Show a notification
- `show_error()` - Show an error message
- `set_processing_state()` - Set the processing state

### ChatInterface (`src/ui/chat_interface.py`)

- `get_instance()` - Get or create the singleton instance of ChatInterface
- `startup()` - Start the chat interface
- `shutdown()` - Stop the chat interface
- `create_session()` - Create a new chat session
- `delete_session()` - Delete a chat session
- `select_session()` - Select a chat session
- `get_session()` - Get a chat session
- `get_all_sessions()` - Get all chat sessions
- `send_message()` - Send a user message
- `receive_message()` - Receive a message
- `update_message_status()` - Update message status
- `clear_session_history()` - Clear chat history for a session

### SettingsPanel (`src/ui/settings_panel.py`)

- `get_instance()` - Get or create the singleton instance of SettingsPanel
- `startup()` - Start the settings panel
- `shutdown()` - Stop the settings panel
- `register_setting()` - Register a setting
- `register_section()` - Register a settings section
- `get_setting()` - Get a setting
- `get_section()` - Get a settings section
- `get_all_settings()` - Get all settings
- `get_all_sections()` - Get all settings sections
- `get_section_settings()` - Get settings for a section
- `get_setting_value()` - Get the current value of a setting
- `set_setting_value()` - Set the value of a setting
- `reset_setting()` - Reset a setting to its default value
- `reset_all_settings()` - Reset all settings to their default values

### ModelSelectionInterface (`src/ui/model_selection.py`)

- `get_instance()` - Get or create the singleton instance of ModelSelectionInterface
- `startup()` - Start the model selection interface
- `shutdown()` - Stop the model selection interface
- `register_model()` - Register a model
- `register_model_parameter()` - Register a model parameter
- `get_model()` - Get a model
- `get_all_models()` - Get all models
- `get_models_by_provider()` - Get models by provider
- `get_models_by_category()` - Get models by category
- `get_model_parameters()` - Get parameters for a model
- `add_favorite_model()` - Add a model to favorites
- `remove_favorite_model()` - Remove a model from favorites
- `get_favorite_models()` - Get favorite models
- `add_recent_model()` - Add a model to recent models
- `get_recent_models()` - Get recent models
- `search_models()` - Search for models
- `select_model()` - Select a model
- `discover_models()` - Discover available models from all providers

## Utility Services

### Logger (`src/utils/logger.py`)

- `get_logger()` - Get a logger instance
- `set_log_level()` - Set the log level
- `get_log_level()` - Get the current log level
- `log_to_file()` - Enable logging to file
- `log_to_console()` - Enable logging to console
- `get_logs()` - Get recent logs

### NetworkUtils (`src/utils/network_utils.py`)

- `check_connectivity()` - Check internet connectivity
- `get_local_ip()` - Get local IP address
- `scan_network()` - Scan local network for services
- `check_port()` - Check if a port is open
- `download_file()` - Download a file from URL

### FileUtils (`src/utils/file_utils.py`)

- `read_file()` - Read file content
- `write_file()` - Write content to file
- `append_file()` - Append content to file
- `delete_file()` - Delete a file
- `file_exists()` - Check if a file exists
- `create_directory()` - Create a directory
- `list_directory()` - List directory contents
- `get_file_size()` - Get file size
- `get_file_modified_time()` - Get file modification time

### SecurityUtils (`src/utils/security_utils.py`)

- `hash_string()` - Hash a string
- `verify_hash()` - Verify a hash
- `encrypt_data()` - Encrypt data
- `decrypt_data()` - Decrypt data
- `generate_secure_token()` - Generate a secure token
- `sanitize_input()` - Sanitize user input
