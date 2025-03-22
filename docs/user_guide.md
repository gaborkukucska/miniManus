# User Guide for miniManus

This guide provides detailed instructions for using miniManus on your Android device.

## Getting Started

After installing miniManus (see [Installation Guide](installation.md)), you can start using it to interact with various LLM models.

### Starting miniManus

Launch miniManus by running:

```bash
python -m minimanus
```

Or if you created a launcher script:

```bash
minimanus
```

### Initial Setup

When you first start miniManus, you'll be guided through an initial setup process:

1. Choose your preferred theme (Light, Dark, or System)
2. Select your default API provider
3. Enter any required API keys
4. Configure basic settings

## Main Interface

The miniManus interface consists of several key components:

### Navigation Bar

Located at the bottom of the screen, the navigation bar allows you to switch between:

- **Chat**: Main chat interface
- **Models**: Model selection interface
- **Settings**: Configuration options
- **Info**: About and help information

### Chat Interface

The chat interface is where you interact with LLM models:

#### Creating a New Chat

1. Tap the "+" button in the top-right corner
2. Enter a title for your chat (optional)
3. Select a model to use
4. Tap "Create"

#### Sending Messages

1. Type your message in the input field at the bottom
2. Tap the send button or press Enter

#### Managing Chat Sessions

- **Switch Sessions**: Tap the menu button in the top-left corner to see a list of your chat sessions
- **Rename Session**: Long-press on a session name and select "Rename"
- **Delete Session**: Long-press on a session name and select "Delete"
- **Clear History**: Tap the menu button in the top-right corner and select "Clear History"

### Model Selection Interface

The model selection interface allows you to browse and select from available models:

#### Browsing Models

Models are organized by:
- **Provider**: OpenRouter, DeepSeek, Anthropic, Ollama, LiteLLM
- **Category**: General, Chat, Code, Specialized, etc.

#### Searching for Models

Use the search bar at the top to find specific models by name, provider, or capability.

#### Favorite Models

- **Add to Favorites**: Tap the star icon next to a model
- **View Favorites**: Tap the "Favorites" tab

#### Model Parameters

Each model has configurable parameters:
1. Select a model
2. Tap "Parameters"
3. Adjust settings like temperature, max tokens, etc.
4. Tap "Save" to apply changes

### Settings Panel

The settings panel allows you to customize miniManus:

#### General Settings

- **Startup Action**: What happens when miniManus starts
- **Confirm Exit**: Whether to show a confirmation dialog when exiting

#### Appearance Settings

- **Theme**: Light, Dark, or System
- **Font Size**: Adjust text size
- **Enable Animations**: Toggle UI animations
- **Compact Mode**: More compact UI for smaller screens
- **Accent Color**: Customize the UI accent color

#### Chat Settings

- **Default Model**: Default model for new chats
- **Default System Prompt**: System prompt for new chats
- **Auto Save**: Automatically save chat history
- **Max History Length**: Maximum number of messages to keep

#### API Settings

- **Default Provider**: Default API provider
- **API Keys**: Manage API keys for different providers
- **Cache Settings**: Configure response caching

#### Advanced Settings

- **Debug Mode**: Enable debug information
- **Log Level**: Set logging verbosity
- **Max Memory**: Limit memory usage

## Working with Different API Providers

### OpenRouter

OpenRouter provides access to various models from different providers through a unified API:

1. Go to Settings > API > OpenRouter
2. Enter your API key
3. In the Model Selection interface, browse models under the OpenRouter provider

### DeepSeek

DeepSeek offers its own language models:

1. Go to Settings > API > DeepSeek
2. Enter your API key
3. In the Model Selection interface, browse models under the DeepSeek provider

### Anthropic

Anthropic provides Claude models:

1. Go to Settings > API > Anthropic
2. Enter your API key
3. In the Model Selection interface, browse models under the Anthropic provider

### Ollama

Ollama allows you to run models locally on your network:

1. Set up Ollama on a device in your network (see [Installation Guide](installation.md))
2. Go to Settings > API > Ollama
3. miniManus will automatically discover Ollama servers on your network
4. In the Model Selection interface, browse models under the Ollama provider

### LiteLLM

LiteLLM provides a unified interface to multiple LLM providers:

1. Set up LiteLLM on a device in your network (see [Installation Guide](installation.md))
2. Go to Settings > API > LiteLLM
3. miniManus will automatically discover LiteLLM servers on your network
4. In the Model Selection interface, browse models under the LiteLLM provider

## Tips and Tricks

### Optimizing for Mobile Performance

- Enable "Compact Mode" in Settings > Appearance for better performance on low-end devices
- Reduce "Max Memory" in Settings > Advanced if you experience slowdowns
- Use local models via Ollama when possible to reduce API costs

### Managing API Costs

- Enable caching in Settings > API to reuse responses for identical queries
- Monitor your usage in the Info section
- Set up usage alerts in Settings > API > Usage Alerts

### Keyboard Shortcuts

When using an external keyboard:
- **Ctrl+Enter**: Send message
- **Ctrl+N**: New chat
- **Ctrl+,**: Open settings
- **Ctrl+M**: Open model selection

### Sharing and Exporting

To export a chat session:
1. Open the chat session
2. Tap the menu button in the top-right corner
3. Select "Export"
4. Choose your preferred format (Text, Markdown, JSON)

## Troubleshooting

### Chat Issues

- **Slow Responses**: Try a different model or API provider
- **Error Messages**: Check your API key and internet connection
- **Message Not Sending**: Check if you've reached the context length limit

### UI Issues

- **UI Freezing**: Try disabling animations in Settings > Appearance
- **Text Too Small/Large**: Adjust font size in Settings > Appearance
- **High Battery Usage**: Enable "Low Power Mode" in Settings > Advanced

### API Issues

- **API Key Invalid**: Verify your API key in Settings > API
- **Rate Limit Exceeded**: Wait a few minutes or switch to a different provider
- **Network Error**: Check your internet connection

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/yourusername/miniManus/issues) page
2. Join our [Discord community](https://discord.gg/yourdiscord)
3. Submit a new issue with details about your problem
