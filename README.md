# miniManus 🧑‍💻🤖📱

A mobile-focused, agentic LLM framework designed to run on Linux (via Termux) on Android devices. It leverages local and external APIs for powerful, on-the-go AI interaction and task automation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add other badges as relevant: build status, version, etc. -->

## Overview

miniManus is a lightweight framework inspired by projects like AutoGen and CrewAI, but specifically optimized for the resource constraints of mobile devices running Termux. It provides a web-based UI for chat interactions and aims to enable complex task execution through an agentic system that can plan and utilize various capabilities (tools).

The framework supports multiple LLM providers (local and cloud-based) and includes features for dynamic model discovery and configuration.

**Goal:** To provide a powerful, extensible, and resource-conscious agentic AI framework accessible directly from your Android phone.

## Key Features

*   **Mobile-First:** Designed and optimized for Termux on Android.
*   **Web-Based UI:** Accessible interface for chat and settings via `http://localhost:8080`.
*   **Multiple LLM Providers:** Supports:
    *   Ollama (Local)
    *   LiteLLM (Local Proxy)
    *   OpenRouter
    *   Anthropic (Claude)
    *   DeepSeek
*   **Local Network Discovery:** Automatically finds running Ollama and LiteLLM instances on your network (configurable).
*   **Agentic System (Developing):** Core components in place for planning and tool usage.
    *   Planning using LLM reasoning.
    *   Capability registry for adding tools (e.g., web search, file access - *use with caution*).
*   **Configuration Management:** Easy setup via UI or JSON files (`~/.local/share/minimanus/config/`). Secure storage for API keys.
*   **Chat Interface:** Persistent chat sessions stored locally.
*   **Model Management:** Browse available models from configured providers.
*   **Resource Aware:** Core components monitor basic system resources (memory, CPU).
*   **(Planned) Plugin System:** Architecture supports future extensibility via plugins.

## Installation

### Prerequisites

*   Android device (Android 7.0+ recommended).
*   [Termux](https://f-droid.org/en/packages/com.termux/) installed from F-Droid (Google Play version is deprecated and won't work).
*   Python 3.10+ installed in Termux (`pkg install python`).
*   Git installed in Termux (`pkg install git`).
*   Internet connection (for initial setup, package downloads, and external API access).

### Steps

1.  **Update Termux Packages:**
    ```bash
    pkg update && pkg upgrade -y
    ```

2.  **Install Required System Packages:**
    ```bash
    pkg install -y python git openssl rust # Rust/openssl often needed for Python crypto libs
    ```

3.  **Clone the Repository:**
    ```bash
    git clone https://github.com/gaborkukucska/miniManus.git
    cd miniManus
    ```

4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Installation of some packages like `cryptography` or `numpy` might take time in Termux as they may need compilation.)*

5.  **Run miniManus:**
    ```bash
    python -m minimanus
    ```

6.  **Access the UI:** Open a web browser on your Android device (or another device on the same network if firewall allows) and navigate to `http://localhost:8080` (or the IP address of your phone and port 8080).

## Configuration

Configuration is managed primarily through the web UI's **Settings** tab, but can also be done by editing files directly.

**Configuration Files Location:** `~/.local/share/minimanus/config/`

*   `config.json`: General settings, UI preferences, provider URLs, etc.
*   `secrets.json`: Stores sensitive information like API keys. **Permissions are set to 600 (user read/write only).**

### API Keys & Providers

1.  **Navigate to Settings:** Open the miniManus UI (`http://localhost:8080`) and click the "Settings" tab.
2.  **Select Provider:** Use the "Default API Provider" dropdown to choose your primary provider.
3.  **Enter Details:** Find the section for your chosen provider(s) (e.g., "OpenRouter Settings", "Ollama Settings").
    *   **API Keys:** Enter your API keys into the masked password fields.
    *   **Host URLs:** For local providers like Ollama or LiteLLM, ensure the "Host URL" points to the correct address and port where the service is running (e.g., `http://192.168.1.100:11434`). Discovery attempts to find these automatically but manual configuration might be needed.
    *   **Default Model:** Select the default model you want to use *for that specific provider*.
4.  **Save Settings:** Click the "Save All Settings" button at the bottom. API keys will be saved securely to `secrets.json`.

*(Ensure any local services like Ollama or LiteLLM are running and accessible from your phone before configuring and using them.)*

## Usage

1.  **Start miniManus:** Run `python -m minimanus` in your Termux terminal within the `miniManus` directory.
2.  **Open UI:** Access `http://localhost:8080` in your browser.
3.  **Chat:** Use the main "Chat" tab to send messages. The backend will process the request using the configured default provider/model or potentially engage the agent system for complex tasks.
4.  **Settings:** Configure API providers, UI appearance, and other options in the "Settings" tab.

## Architecture Overview

miniManus uses a modular architecture:

*   **Core Framework (`minimanus/core/`)**
    *   `SystemManager`: Coordinates startup, shutdown, and component lifecycle. Handles signals.
    *   `ConfigManager`: Loads/saves `config.json` and `secrets.json`.
    *   `EventBus`: Decoupled communication between components.
    *   `ErrorHandler`: Centralized error logging and handling.
    *   `ResourceMonitor`: Basic monitoring of memory, CPU, storage (limited by Termux).
    *   `AgentSystem`: Orchestrates planning and tool/capability execution (under development).
    *   `PluginManager`: Manages loading/unloading of plugins (future feature).
*   **API Integrations (`minimanus/api/`)**
    *   `APIManager`: Central point for sending requests. Handles provider selection, caching (basic).
    *   `Provider Adapters` (e.g., `OllamaAdapter`, `OpenRouterAdapter`): Implement communication specifics for each LLM provider API using `asyncio` and `aiohttp`. Include discovery logic for local providers.
*   **UI Layer (`minimanus/ui/`)**
    *   `UIManager`: Manages the backend web server (Python `http.server`) serving the UI. Provides API endpoints (`/api/*`) for the frontend.
    *   `ChatInterface`: Manages chat sessions, message history persistence, and routes user messages to the `AgentSystem`.
    *   `SettingsPanel`: Defines available settings (interacts with `ConfigManager`).
    *   `ModelSelectionInterface`: Manages discovery and information about available models.
*   **Web UI (`minimanus/static/`)**
    *   `index.html`, `styles.css`, `script.js`: Frontend single-page application that interacts with the backend API endpoints.
*   **Utilities (`minimanus/utils/`)**
    *   (Placeholder for common helper functions - network, file ops, security).

## Contributing

Contributions are welcome! Please follow standard GitHub practices:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/your-amazing-feature`).
3.  Make your changes and commit them (`git commit -m 'Add some amazing feature'`).
4.  Push to your branch (`git push origin feature/your-amazing-feature`).
5.  Open a Pull Request against the `main` branch of the original repository.

Please ensure your code follows basic Python standards (PEP 8) and includes docstrings where appropriate.

## License

This project is licensed under the MIT License - see the `LICENSE` file (if available) or the standard MIT license text.

## Acknowledgments

*   Inspired by agentic frameworks like AutoGen, CrewAI, and the original Manus concept.
*   Built for the powerful Termux environment on Android, by Gemini2.5 🙌
*   Thanks to the developers of the LLM APIs and local inference tools (Ollama, LiteLLM).
