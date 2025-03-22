# miniManus

A mobile-focused framework that runs on Linux in Termux for Android phones, leveraging external APIs for LLM inference.

## Overview

miniManus is a lightweight framework inspired by Manus, designed specifically to run on Android devices through the Termux application. It provides a mobile-optimized interface for interacting with various LLM providers including OpenRouter, DeepSeek, Anthropic, Ollama, and LiteLLM.

The framework is built with mobile constraints in mind, focusing on efficient resource usage, responsive UI, and flexible API integration options. miniManus can connect to external API providers or locally hosted models, making it versatile for various use cases.

## Features

- **Mobile-Optimized UI**: Responsive interface designed specifically for mobile devices
- **Multiple API Providers**: Support for OpenRouter, DeepSeek, Anthropic, Ollama, and LiteLLM
- **Local Model Discovery**: Automatic discovery of Ollama and LiteLLM servers on the local network
- **Resource Efficiency**: Designed to work within the constraints of mobile devices
- **Persistent Chat Sessions**: Save and restore chat conversations
- **Customizable Settings**: Comprehensive settings management
- **Model Selection**: Browse, search, and select from available models
- **Plugin System**: Extensible through plugins
- **Comprehensive Documentation**: Including a function index to help LLM models avoid typos

## Installation

### Prerequisites

- Android device with [Termux](https://termux.com/) installed
- Python 3.10+ installed in Termux

### Basic Installation

1. Update Termux packages:
```bash
pkg update && pkg upgrade
```

2. Install required packages:
```bash
pkg install python git
```

3. Clone the repository:
```bash
git clone https://github.com/yourusername/miniManus.git
cd miniManus
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Run miniManus:
```bash
python -m minimanus
```

## Configuration

miniManus can be configured through the settings interface or by directly editing the configuration files located in `~/.local/share/minimanus/config/`.

### API Keys

To use external API providers, you'll need to set up API keys:

1. Open miniManus
2. Navigate to Settings > API
3. Select your preferred provider
4. Enter your API key

Alternatively, you can add API keys to the configuration file:
```bash
mkdir -p ~/.local/share/minimanus/config
echo '{"api_keys": {"openrouter": "your-api-key-here"}}' > ~/.local/share/minimanus/config/api_keys.json
```

## Usage

### Starting miniManus

```bash
python -m minimanus
```

### Chat Interface

The chat interface allows you to:
- Create new chat sessions
- Send messages to LLM models
- View and manage chat history
- Switch between different models

### Model Selection

The model selection interface allows you to:
- Browse available models from different providers
- Search for specific models
- Mark favorite models
- Configure model parameters

### Settings

The settings panel allows you to customize:
- UI appearance (theme, font size, animations)
- Default models and system prompts
- API providers and caching behavior
- Advanced system settings

## Architecture

miniManus follows a modular architecture with the following components:

### Core Framework
- **SystemManager**: Coordinates system startup and shutdown
- **ConfigurationManager**: Manages configuration settings
- **EventBus**: Facilitates communication between components
- **ResourceMonitor**: Monitors system resources
- **ErrorHandler**: Handles and logs errors
- **PluginManager**: Manages plugins

### API Integrations
- **APIManager**: Coordinates API interactions
- **Provider Adapters**: Interfaces with specific API providers

### UI Components
- **UIManager**: Manages UI state and appearance
- **ChatInterface**: Handles chat sessions and messages
- **SettingsPanel**: Manages settings
- **ModelSelectionInterface**: Handles model discovery and selection

### Utility Services
- **Logger**: Handles logging
- **NetworkUtils**: Network-related utilities
- **FileUtils**: File operations
- **SecurityUtils**: Security-related functions

## Extending miniManus

### Creating Plugins

Plugins can extend miniManus functionality. To create a plugin:

1. Create a new directory in `~/.local/share/minimanus/plugins/`
2. Create a Python file with your plugin code
3. Implement the `PluginInterface` class
4. Add a `plugin.json` file with metadata

Example plugin structure:
```
my_plugin/
├── __init__.py
├── plugin.json
└── my_module.py
```

Example plugin.json:
```json
{
  "name": "My Plugin",
  "version": "1.0.0",
  "description": "An example plugin",
  "author": "Your Name",
  "dependencies": []
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the Manus project
- Built for the Termux community
- Special thanks to all the LLM API providers
