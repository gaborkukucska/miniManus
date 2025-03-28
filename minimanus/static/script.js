// script.js for miniManus

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatTab = document.getElementById('chat-tab');
    const settingsTab = document.getElementById('settings-tab');
    
    const chatSection = document.getElementById('chat-section');
    const settingsSection = document.getElementById('settings-section');
    
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    
    const defaultProvider = document.getElementById('default-provider');
    const saveSettings = document.getElementById('save-settings');
    
    // Tab Navigation
    chatTab.addEventListener('click', function(e) {
        e.preventDefault();
        setActiveTab(chatTab, chatSection);
    });
    
    settingsTab.addEventListener('click', function(e) {
        e.preventDefault();
        setActiveTab(settingsTab, settingsSection);
    });
    
    function setActiveTab(tab, section) {
        // Update tab classes
        chatTab.classList.remove('active');
        settingsTab.classList.remove('active');
        tab.classList.add('active');
        
        // Update section visibility
        chatSection.classList.remove('active-section');
        settingsSection.classList.remove('active-section');
        
        // Add active-section class to make the selected section visible
        section.classList.add('active-section');
        
        // Ensure proper display property is set
        if (section === chatSection) {
            chatSection.style.display = 'block';
            settingsSection.style.display = 'none';
        } else {
            chatSection.style.display = 'none';
            settingsSection.style.display = 'block';
        }
    }
    
    // Chat Functionality
    sendButton.addEventListener('click', sendMessage);
    
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Add user message to UI
        addMessage(message, 'user');
        
        // Clear input
        messageInput.value = '';
        messageInput.style.height = '40px';
        
        // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant-message';
        loadingDiv.innerHTML = '<div class="message-content">Thinking...</div>';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Send to backend API
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            // Remove loading indicator
            chatMessages.removeChild(loadingDiv);
            
            // Add assistant response
            addMessage(data.response, 'assistant');
        })
        .catch(error => {
            // Remove loading indicator
            chatMessages.removeChild(loadingDiv);
            
            // Add error message
            addMessage('Sorry, there was an error processing your request: ' + error.message, 'assistant');
            console.error('Error:', error);
        });
    }
    
    function addMessage(content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Settings Functionality
    defaultProvider.addEventListener('change', function() {
        // Hide all provider settings
        document.querySelectorAll('.provider-settings').forEach(el => {
            el.style.display = 'none';
        });
        
        // Show selected provider settings
        const selectedProvider = this.value;
        document.getElementById(`${selectedProvider}-settings`).style.display = 'block';
        
        // Fetch and populate models for the selected provider
        fetchModelsForProvider(selectedProvider);
    });
    
    // Function to fetch available models for a provider
    function fetchModelsForProvider(provider) {
        // Show loading indicator in the model dropdown
        const modelDropdown = document.getElementById(`${provider}-model`);
        modelDropdown.innerHTML = '<option value="">Loading models...</option>';
        
        fetch(`/api/models?provider=${provider}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to fetch models');
                }
                return response.json();
            })
            .then(data => {
                // Clear the dropdown
                modelDropdown.innerHTML = '';
                
                // Add each model to the dropdown
                if (data.models && data.models.length > 0) {
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.id;
                        option.textContent = model.name || model.id;
                        modelDropdown.appendChild(option);
                    });
                    
                    // Select the first model by default
                    modelDropdown.value = data.models[0].id;
                } else {
                    // No models available
                    const option = document.createElement('option');
                    option.value = "";
                    option.textContent = "No models available";
                    modelDropdown.appendChild(option);
                }
            })
            .catch(error => {
                console.error('Error fetching models:', error);
                modelDropdown.innerHTML = '<option value="">Error loading models</option>';
            });
    }
    
    saveSettings.addEventListener('click', function() {
        const settings = {
            defaultProvider: defaultProvider.value,
            theme: document.getElementById('theme').value,
            fontSize: document.getElementById('font-size').value,
            animations: document.getElementById('animations').checked,
            providers: {
                openrouter: {
                    apiKey: document.getElementById('openrouter-api-key').value,
                    model: document.getElementById('openrouter-model').value
                },
                anthropic: {
                    apiKey: document.getElementById('anthropic-api-key').value,
                    model: document.getElementById('anthropic-model').value
                },
                deepseek: {
                    apiKey: document.getElementById('deepseek-api-key').value,
                    model: document.getElementById('deepseek-model').value
                },
                ollama: {
                    host: document.getElementById('ollama-host').value,
                    model: document.getElementById('ollama-model').value
                },
                litellm: {
                    host: document.getElementById('litellm-host').value,
                    model: document.getElementById('litellm-model').value
                }
            }
        };
        
        // Save settings to backend
        fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            alert('Settings saved successfully!');
        })
        .catch(error => {
            alert('Error saving settings: ' + error.message);
        });
    });
    
    // Theme Handling
    const themeSelector = document.getElementById('theme');
    themeSelector.addEventListener('change', function() {
        applyTheme(this.value);
    });
    
    function applyTheme(theme) {
        if (theme === 'dark') {
            document.body.classList.add('dark-theme');
        } else if (theme === 'light') {
            document.body.classList.remove('dark-theme');
        } else {
            // System theme
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                document.body.classList.add('dark-theme');
            } else {
                document.body.classList.remove('dark-theme');
            }
        }
    }
    
    // Font Size Handling
    const fontSizeSlider = document.getElementById('font-size');
    const fontSizeValue = document.getElementById('font-size-value');
    
    fontSizeSlider.addEventListener('input', function() {
        const size = this.value;
        fontSizeValue.textContent = size + 'px';
        document.documentElement.style.setProperty('--font-size', size + 'px');
    });
    
    // Initialize UI
    function initializeUI() {
        // Load settings from backend
        fetch('/api/settings')
        .then(response => {
            if (!response.ok) {
                // If settings don't exist yet, that's ok
                if (response.status === 404) {
                    return null;
                }
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data) {
                // Apply settings to UI
                defaultProvider.value = data.defaultProvider || 'openrouter';
                document.getElementById('theme').value = data.theme || 'system';
                document.getElementById('font-size').value = data.fontSize || 14;
                document.getElementById('animations').checked = data.animations !== false;
                
                // Apply provider-specific settings
                if (data.providers) {
                    if (data.providers.openrouter) {
                        document.getElementById('openrouter-api-key').value = data.providers.openrouter.apiKey || '';
                    }
                    if (data.providers.anthropic) {
                        document.getElementById('anthropic-api-key').value = data.providers.anthropic.apiKey || '';
                    }
                    if (data.providers.deepseek) {
                        document.getElementById('deepseek-api-key').value = data.providers.deepseek.apiKey || '';
                    }
                    if (data.providers.ollama) {
                        document.getElementById('ollama-host').value = data.providers.ollama.host || 'http://localhost:11434';
                    }
                    if (data.providers.litellm) {
                        document.getElementById('litellm-host').value = data.providers.litellm.host || 'http://localhost:8000';
                    }
                }
                
                // Show the correct provider settings
                document.querySelectorAll('.provider-settings').forEach(el => {
                    el.style.display = 'none';
                });
                document.getElementById(`${defaultProvider.value}-settings`).style.display = 'block';
                
                // Apply theme
                applyTheme(data.theme || 'system');
                
                // Apply font size
                const fontSize = data.fontSize || 14;
                fontSizeValue.textContent = fontSize + 'px';
                document.documentElement.style.setProperty('--font-size', fontSize + 'px');
                
                // Fetch models for the selected provider
                fetchModelsForProvider(defaultProvider.value);
            } else {
                // If no settings exist, fetch models for the default provider
                fetchModelsForProvider(defaultProvider.value);
            }
        })
        .catch(error => {
            console.error('Error loading settings:', error);
            // Still try to fetch models for the default provider
            fetchModelsForProvider(defaultProvider.value);
        });
    }
    
    // Initialize the UI
    initializeUI();
});
