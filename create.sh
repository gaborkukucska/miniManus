#!/bin/bash

# Create static directory in the project
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
STATIC_DIR="$PROJECT_DIR/minimanus/static"

echo "Creating static directory: $STATIC_DIR"
mkdir -p "$STATIC_DIR"

# Create index.html
cat > "$STATIC_DIR/index.html" << 'EOL'
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>miniManus</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            background-color: #f5f5f5;
        }
        header {
            background-color: #4a90e2;
            color: white;
            padding: 1rem;
            text-align: center;
        }
        main {
            flex: 1;
            padding: 1rem;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 70vh;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background-color: white;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            background-color: #f9f9f9;
        }
        .message {
            margin-bottom: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 18px;
            max-width: 80%;
        }
        .user-message {
            background-color: #4a90e2;
            color: white;
            align-self: flex-end;
            margin-left: auto;
            text-align: right;
        }
        .assistant-message {
            background-color: #e5e5ea;
            color: black;
            align-self: flex-start;
        }
        .chat-input {
            display: flex;
            padding: 0.5rem;
            border-top: 1px solid #ddd;
            background-color: white;
        }
        .chat-input input {
            flex: 1;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 18px;
            margin-right: 0.5rem;
        }
        .chat-input button {
            padding: 0.5rem 1rem;
            background-color: #4a90e2;
            color: white;
            border: none;
            border-radius: 18px;
            cursor: pointer;
        }
        .chat-messages-container {
            display: flex;
            flex-direction: column;
        }
        .timestamp {
            font-size: 0.8em;
            color: #888;
            margin-top: 0.2rem;
        }
        .settings-panel {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .settings-panel h2 {
            margin-top: 0;
        }
        .settings-group {
            margin-bottom: 1rem;
        }
        .settings-group h3 {
            margin-bottom: 0.5rem;
        }
        .setting-item {
            margin-bottom: 0.5rem;
            display: flex;
            flex-direction: column;
        }
        .setting-item label {
            margin-bottom: 0.3rem;
        }
        .models-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
        }
        .model-card {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .model-card:hover {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .model-card h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        .model-card p {
            margin: 0;
            font-size: 0.9rem;
            color: #555;
        }
        .navigation {
            display: flex;
            justify-content: space-between;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 1rem;
        }
        .nav-item {
            flex: 1;
            text-align: center;
            padding: 0.8rem;
            cursor: pointer;
        }
        .nav-item.active {
            background-color: #4a90e2;
            color: white;
        }
        .notification {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.8rem 1rem;
            border-radius: 4px;
            background-color: #4a90e2;
            color: white;
            z-index: 1000;
            animation: fadeOut 0.3s ease-in-out 2.7s forwards;
        }
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }
    </style>
</head>
<body>
    <header>
        <h1>miniManus</h1>
    </header>
    <main>
        <div class="navigation">
            <div class="nav-item active" data-view="chat">Chat</div>
            <div class="nav-item" data-view="models">Models</div>
            <div class="nav-item" data-view="settings">Settings</div>
            <div class="nav-item" data-view="info">Info</div>
        </div>
        
        <div id="chat-view">
            <div class="chat-container">
                <div class="chat-messages" id="chat-messages">
                    <div class="chat-messages-container">
                        <div class="message assistant-message">
                            <p>Hello! I'm miniManus. How can I help you today?</p>
                            <div class="timestamp">Just now</div>
                        </div>
                    </div>
                </div>
                <div class="chat-input">
                    <input type="text" id="message-input" placeholder="Type your message...">
                    <button id="send-button">Send</button>
                </div>
            </div>
        </div>
        
        <div id="models-view" style="display: none;">
            <h2>Available Models</h2>
            <div class="models-list" id="models-list">
                <div class="model-card">
                    <h3>Loading models...</h3>
                    <p>Please wait while we fetch available models</p>
                </div>
            </div>
        </div>
        
        <div id="settings-view" style="display: none;">
            <div class="settings-panel">
                <h2>Settings</h2>
                
                <div class="settings-group">
                    <h3>Appearance</h3>
                    <div class="setting-item">
                        <label for="theme-select">Theme</label>
                        <select id="theme-select">
                            <option value="LIGHT">Light</option>
                            <option value="DARK" selected>Dark</option>
                            <option value="SYSTEM">System</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label for="font-size">Font Size</label>
                        <input type="range" id="font-size" min="12" max="24" value="16">
                        <span id="font-size-value">16px</span>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="animations-toggle" checked>
                            Enable Animations
                        </label>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="compact-mode-toggle">
                            Compact Mode
                        </label>
                    </div>
                </div>
                
                <div class="settings-group">
                    <h3>API Settings</h3>
                    <div class="setting-item">
                        <label for="api-provider">Default Provider</label>
                        <select id="api-provider">
                            <option value="openrouter" selected>OpenRouter</option>
                            <option value="deepseek">DeepSeek</option>
                            <option value="anthropic">Anthropic</option>
                            <option value="ollama">Ollama</option>
                            <option value="litellm">LiteLLM</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label for="api-key">API Key</label>
                        <input type="password" id="api-key" placeholder="Enter API key">
                    </div>
                </div>
            </div>
        </div>
        
        <div id="info-view" style="display: none;">
            <div class="settings-panel">
                <h2>About miniManus</h2>
                <p>miniManus is a mobile-focused framework that runs on Linux in Termux for Android phones.</p>
                <p>Version: 0.1.0</p>
                <p>Created by: miniManus Team</p>
                
                <h3>Support</h3>
                <p>For help and support, please visit:</p>
                <p><a href="https://github.com/yourusername/miniManus" target="_blank">GitHub Repository</a></p>
            </div>
        </div>
    </main>
    
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const messagesContainer = document.getElementById('chat-messages');
            const messageInput = document.getElementById('message-input');
            const sendButton = document.getElementById('send-button');
            const navItems = document.querySelectorAll('.nav-item');
            const views = {
                chat: document.getElementById('chat-view'),
                models: document.getElementById('models-view'),
