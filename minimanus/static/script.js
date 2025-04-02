// minimanus/static/script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const chatTab = document.getElementById('chat-tab');
    const settingsTab = document.getElementById('settings-tab');
    const chatSection = document.getElementById('chat-section');
    const settingsSection = document.getElementById('settings-section');
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');

    // Settings Elements
    const defaultProviderSelect = document.getElementById('default-provider');
    const providerSettingsDivs = document.querySelectorAll('.provider-settings');
    const themeSelect = document.getElementById('theme');
    const fontSizeSlider = document.getElementById('font-size');
    const fontSizeValue = document.getElementById('font-size-value');
    const animationsCheckbox = document.getElementById('animations');
    const saveSettingsButton = document.getElementById('save-settings');

    // --- State ---
    let currentSettings = {};
    let isLoading = false; // To prevent multiple chat submissions

    // --- Functions ---

    // Basic API Fetch Helper
    async function apiRequest(endpoint, method = 'GET', body = null) {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
        };
        if (body) {
            options.body = JSON.stringify(body);
        }

        try {
            const response = await fetch(endpoint, options);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ message: `HTTP error! status: ${response.status}` }));
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            alert(`API request failed: ${error.message}`); // Simple error feedback
            throw error; // Re-throw for calling function to handle if needed
        }
    }

    // Load Settings from Backend
    async function loadSettings() {
        try {
            const settings = await apiRequest('/api/settings');
            currentSettings = settings; // Store loaded settings
            applySettingsToUI(settings);
            console.log('Settings loaded:', settings);
        } catch (error) {
            console.error('Failed to load settings:', error);
            alert('Failed to load settings from the server.');
        }
    }

    // Apply loaded settings to the UI form
    function applySettingsToUI(settings) {
        // API Settings
        defaultProviderSelect.value = settings?.api?.default_provider || 'openrouter';
        updateVisibleProviderSettings(); // Show settings for the default provider

        // Populate API Keys and Default Models for each provider
        const providers = settings?.api?.providers || {};
        for (const providerKey in providers) {
            const providerSettings = providers[providerKey];
            const apiKeyInput = document.getElementById(`${providerKey}-api-key`);
            const modelSelect = document.getElementById(`${providerKey}-model`);
            const hostInput = document.getElementById(`${providerKey}-host`); // For Ollama/LiteLLM

            if (apiKeyInput) {
                 // Load API key from secrets (if backend provided it securely, otherwise it remains blank)
                 // NOTE: Backend /api/settings GET should NOT return secrets.
                 // API keys are typically write-only from UI or loaded client-side if needed, which isn't ideal.
                 // For now, we assume keys are set and saved, but not displayed back.
                 // apiKeyInput.value = settings?.secrets?.api_keys?.[providerKey] || ''; // Don't do this!
            }
            if (hostInput) {
                 hostInput.value = providerSettings?.base_url || ''; // Use base_url for host
            }
            if (modelSelect) {
                // We'll populate models dynamically later, just set the saved default for now
                const defaultModel = providerSettings?.default_model;
                // Check if the default model exists as an option before setting it
                let found = false;
                for(let i=0; i < modelSelect.options.length; i++) {
                    if (modelSelect.options[i].value === defaultModel) {
                        found = true;
                        break;
                    }
                }
                // If not found, maybe add it? Or just select first? For now, leave as is or select first.
                if (defaultModel && found) {
                    modelSelect.value = defaultModel;
                } else if (defaultModel) {
                     console.warn(`Default model '${defaultModel}' for ${providerKey} not found in static options. Fetching models...`);
                     // Fetch models to potentially add it, then set value
                     loadModelsForProvider(providerKey, defaultModel);
                } else {
                    // Select the first option if no default is set or found
                    modelSelect.selectedIndex = 0;
                }
            }
        }


        // UI Settings
        themeSelect.value = settings?.ui?.theme || 'dark';
        applyTheme(themeSelect.value);

        const savedFontSize = settings?.ui?.font_size || 14;
        fontSizeSlider.value = savedFontSize;
        fontSizeValue.textContent = `${savedFontSize}px`;
        applyFontSize(savedFontSize);

        animationsCheckbox.checked = settings?.ui?.animations_enabled !== false; // Default true
        // Apply animations setting (e.g., add/remove a class to body)
        document.body.classList.toggle('animations-disabled', !animationsCheckbox.checked);


        // Ensure correct provider settings are visible
        updateVisibleProviderSettings();
        // Load models for the initially selected provider
        loadModelsForProvider(defaultProviderSelect.value);
    }

    // Save Settings to Backend
    async function saveSettings() {
        const settingsToSave = {
            api: {
                default_provider: defaultProviderSelect.value,
                providers: {}
            },
            ui: {
                theme: themeSelect.value,
                font_size: parseInt(fontSizeSlider.value, 10),
                animations_enabled: animationsCheckbox.checked
            }
            // Include other top-level settings groups if necessary
        };

        // Collect provider-specific settings (API keys, default models, hosts)
        providerSettingsDivs.forEach(div => {
            const providerKey = div.id.replace('-settings', '');
            const apiKeyInput = document.getElementById(`${providerKey}-api-key`);
            const modelSelect = document.getElementById(`${providerKey}-model`);
            const hostInput = document.getElementById(`${providerKey}-host`);

             settingsToSave.api.providers[providerKey] = {
                  // Only include keys/hosts if the input exists
                  ...(apiKeyInput && { api_key: apiKeyInput.value }), // Send key ONLY if user entered/changed it
                  ...(hostInput && { base_url: hostInput.value }),
                  ...(modelSelect && { default_model: modelSelect.value }),
             };
             // IMPORTANT: We are sending the API key here. The backend MUST handle this securely
             // and store it in secrets.json, not config.json.
             // Clear the input after saving? Maybe not, user might want to see it's set (masked).
        });


        try {
            // Send ONLY the settings structure defined above
            const result = await apiRequest('/api/settings', 'POST', settingsToSave);
            if (result.status === 'success') {
                alert('Settings saved successfully!');
                currentSettings = await apiRequest('/api/settings'); // Reload effective settings
                applySettingsToUI(currentSettings); // Re-apply to ensure consistency
            } else {
                 throw new Error(result.message || 'Unknown error saving settings.');
            }
        } catch (error) {
            console.error('Failed to save settings:', error);
            alert(`Failed to save settings: ${error.message}`);
        }
    }

    // Update visibility of provider-specific settings blocks
    function updateVisibleProviderSettings() {
        const selectedProvider = defaultProviderSelect.value;
        providerSettingsDivs.forEach(div => {
            div.style.display = div.id === `${selectedProvider}-settings` ? 'block' : 'none';
        });
         // Fetch models for the newly selected provider
         loadModelsForProvider(selectedProvider);
    }

     // Load models for a specific provider and update its dropdown
     async function loadModelsForProvider(providerKey, selectedValue = null) {
        const modelSelect = document.getElementById(`${providerKey}-model`);
        if (!modelSelect) return; // No model dropdown for this provider (e.g., if settings HTML is incomplete)

        // Add a loading option
        modelSelect.innerHTML = '<option value="" disabled selected>Loading models...</option>';

        try {
            const data = await apiRequest(`/api/models?provider=${providerKey}`);
            const models = data.models || [];

            modelSelect.innerHTML = ''; // Clear loading/previous options

            if (models.length === 0) {
                modelSelect.innerHTML = '<option value="" disabled selected>No models found</option>';
                // Fallback: use the statically defined default from config if API fails?
                 const fallbackModel = currentSettings?.api?.providers?.[providerKey]?.default_model;
                 if(fallbackModel) {
                     const option = document.createElement('option');
                     option.value = fallbackModel;
                     option.textContent = `${fallbackModel} (Default/Offline)`;
                     modelSelect.appendChild(option);
                     modelSelect.value = fallbackModel; // Select it
                 }
                 console.warn(`No models loaded for ${providerKey}.`);

            } else {
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id; // Use the model ID as the value
                    option.textContent = model.name || model.id; // Display name or ID
                    modelSelect.appendChild(option);
                });

                // Try to set the previously selected value or the provider's default
                const valueToSelect = selectedValue || currentSettings?.api?.providers?.[providerKey]?.default_model;
                if (valueToSelect && modelSelect.querySelector(`option[value="${valueToSelect}"]`)) {
                     modelSelect.value = valueToSelect;
                 } else if (models.length > 0) {
                     modelSelect.selectedIndex = 0; // Select the first model if default/previous not found
                 }
            }
        } catch (error) {
            console.error(`Failed to load models for ${providerKey}:`, error);
            modelSelect.innerHTML = '<option value="" disabled selected>Error loading models</option>';
             // Fallback like above
             const fallbackModel = currentSettings?.api?.providers?.[providerKey]?.default_model;
             if(fallbackModel) {
                 const option = document.createElement('option');
                 option.value = fallbackModel;
                 option.textContent = `${fallbackModel} (Default/Offline)`;
                 modelSelect.appendChild(option);
                 modelSelect.value = fallbackModel;
             }
        }
    }

    // Add Message to Chat UI
    function addChatMessage(role, text) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${role}-message`); // e.g., 'user-message' or 'assistant-message'

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        // Basic sanitization (replace potential HTML tags) - consider a more robust library if needed
        contentDiv.textContent = text;
         // Simple Markdown rendering (bold, italics) - needs improvement for code blocks etc.
         // contentDiv.innerHTML = text
         //    .replace(/</g, "<").replace(/>/g, ">") // Basic HTML escape
         //    .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>') // Bold
         //    .replace(/\*(.*?)\*/g, '<i>$1</i>'); // Italics

        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Send Chat Message
    async function sendChatMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText || isLoading) {
            return;
        }

        isLoading = true;
        sendButton.disabled = true;
        sendButton.textContent = 'Sending...';
        addChatMessage('user', messageText);
        messageInput.value = ''; // Clear input after sending

        try {
            // Use current session ID (UI needs session management later)
            const response = await apiRequest('/api/chat', 'POST', { message: messageText /*, session_id: currentSessionId */ });
            addChatMessage('assistant', response.response);
        } catch (error) {
            addChatMessage('assistant', `Error: ${error.message}`); // Show error in chat
        } finally {
            isLoading = false;
            sendButton.disabled = false;
            sendButton.textContent = 'Send';
            messageInput.focus(); // Return focus to input
        }
    }

    // Apply Theme
    function applyTheme(themeName) {
        document.body.classList.remove('light-theme', 'dark-theme'); // Remove existing themes
        if (themeName === 'light') {
            document.body.classList.add('light-theme');
        } else if (themeName === 'dark') {
            document.body.classList.add('dark-theme');
        } else {
            // Handle 'system' theme - check preference and apply light/dark
            const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.body.classList.add(prefersDark ? 'dark-theme' : 'light-theme');
        }
        // Ensure dark theme variables apply correctly if needed by switching the class
         document.body.classList.toggle('dark-theme', document.body.classList.contains('dark-theme'));
    }

     // Apply Font Size
     function applyFontSize(size) {
          document.documentElement.style.setProperty('--font-size', `${size}px`);
          fontSizeValue.textContent = `${size}px`;
     }

    // --- Event Listeners ---

    // Tab Navigation
    chatTab.addEventListener('click', (e) => {
        e.preventDefault();
        chatSection.classList.add('active-section');
        settingsSection.classList.remove('active-section');
        chatTab.classList.add('active');
        settingsTab.classList.remove('active');
    });

    settingsTab.addEventListener('click', (e) => {
        e.preventDefault();
        settingsSection.classList.add('active-section');
        chatSection.classList.remove('active-section');
        settingsTab.classList.add('active');
        chatTab.classList.remove('active');
    });

    // Chat Input
    sendButton.addEventListener('click', sendChatMessage);
    messageInput.addEventListener('keypress', (e) => {
        // Send on Enter, allow Shift+Enter for newline
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent default newline insertion
            sendChatMessage();
        }
    });

     // Settings Listeners
     defaultProviderSelect.addEventListener('change', updateVisibleProviderSettings);
     saveSettingsButton.addEventListener('click', saveSettings);
     themeSelect.addEventListener('change', (e) => applyTheme(e.target.value));
     fontSizeSlider.addEventListener('input', (e) => applyFontSize(e.target.value));
     animationsCheckbox.addEventListener('change', (e) => {
          document.body.classList.toggle('animations-disabled', !e.target.checked);
     });

    // --- Initial Load ---
    loadSettings();

}); // End DOMContentLoaded

// Optional: Add listener for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    const themeSelect = document.getElementById('theme');
    if (themeSelect && themeSelect.value === 'system') {
         document.body.classList.remove('light-theme', 'dark-theme');
         document.body.classList.add(e.matches ? 'dark-theme' : 'light-theme');
         document.body.classList.toggle('dark-theme', e.matches); // Ensure dark-theme class applies correctly
    }
});
