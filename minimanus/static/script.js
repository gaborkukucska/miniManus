// miniManus-main/minimanus/static/script.js

document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const settingsSection = document.getElementById('settings-section');
    const chatSection = document.getElementById('chat-section');
    const settingsTab = document.getElementById('settings-tab');
    const chatTab = document.getElementById('chat-tab');
    const themeSelect = document.getElementById('theme');
    const fontSizeSlider = document.getElementById('font-size');
    const fontSizeValue = document.getElementById('font-size-value');
    const animationsCheckbox = document.getElementById('animations');
    const compactModeCheckbox = document.getElementById('compact-mode');
    const saveSettingsButton = document.getElementById('save-settings');
    const defaultProviderSelect = document.getElementById('default-provider');
    const providerSettingsDivs = document.querySelectorAll('.provider-settings'); // Select all provider setting divs

    let currentSessionId = null; // Store the current session ID

    // --- Utility Functions ---

    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${role}-message`);

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        // Basic sanitization (replace with a more robust library if needed)
        contentDiv.textContent = content; // Use textContent to prevent HTML injection

        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll
    }

    function setLoadingState(isLoading) {
        sendButton.disabled = isLoading;
        messageInput.disabled = isLoading;
        if (isLoading) {
            sendButton.textContent = '...';
        } else {
            sendButton.textContent = 'Send';
        }
    }

    function adjustTextareaHeight() {
        messageInput.style.height = 'auto'; // Reset height
        messageInput.style.height = (messageInput.scrollHeight > 120 ? 120 : messageInput.scrollHeight) + 'px';
    }

    // --- API Calls ---

    async function sendMessageToServer(message, sessionId) {
        setLoadingState(true);
        try {
            const payload = { message: message };
            if (sessionId) {
                payload.session_id = sessionId;
            }
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                addMessage('assistant', data.response);
            } else {
                console.error("Chat API Error:", data);
                addMessage('assistant', `Error: ${data.message || 'Failed to get response'}`);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            addMessage('assistant', `Error: ${error.message}`);
        } finally {
            setLoadingState(false);
            messageInput.focus();
        }
    }

    async function loadSettings() {
        console.log("Loading settings from backend...");
        try {
            const response = await fetch('/api/settings');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const settings = await response.json();
            console.log("Settings received:", settings);

            // --- Populate General UI Settings ---
            themeSelect.value = settings.ui?.theme || 'dark';
            applyTheme(settings.ui?.theme);

            const fontSize = settings.ui?.font_size || 14;
            fontSizeSlider.value = fontSize;
            fontSizeValue.textContent = `${fontSize}px`;
            document.body.style.setProperty('--font-size', `${fontSize}px`);

            animationsCheckbox.checked = settings.ui?.animations_enabled ?? true;
            compactModeCheckbox.checked = settings.ui?.compact_mode ?? false;

            // --- Populate API Settings ---
            defaultProviderSelect.value = settings.api?.default_provider || 'openrouter';
            toggleProviderSettingsVisibility(defaultProviderSelect.value); // Show correct section

            // --- Populate Provider Specific Settings ---
            const providers = ['openrouter', 'anthropic', 'deepseek', 'ollama', 'litellm'];
            const modelFetchPromises = []; // Store promises for fetching models

            providers.forEach(provider => {
                const providerConfig = settings.api?.providers?.[provider] || {};

                // API Key (don't load value, just check presence if needed)
                const apiKeyInput = document.getElementById(`${provider}-api-key`);
                if (apiKeyInput) {
                     apiKeyInput.value = ''; // Clear password fields on load for security
                     apiKeyInput.placeholder = `Enter ${provider} key...`;
                }

                // Base URL
                const baseUrlInput = document.getElementById(`${provider}-base_url`);
                if (baseUrlInput) {
                    baseUrlInput.value = providerConfig.base_url || '';
                }

                // Default Model (Load options later)
                const modelSelect = document.getElementById(`${provider}-model`);
                if (modelSelect) {
                    modelSelect.dataset.savedValue = providerConfig.default_model || ''; // Store saved value
                }

                // Embedding Model
                const embeddingInput = document.getElementById(`${provider}-embedding-model`);
                if (embeddingInput) {
                    embeddingInput.value = providerConfig.embedding_model || '';
                }

                // Discovery Enabled
                const discoveryCheckbox = document.getElementById(`${provider}-discovery_enabled`);
                 if (discoveryCheckbox) {
                    discoveryCheckbox.checked = providerConfig.discovery_enabled ?? true;
                 }

                // OpenRouter Specific
                if (provider === 'openrouter') {
                    document.getElementById('openrouter-referer').value = providerConfig.referer || '';
                    document.getElementById('openrouter-x_title').value = providerConfig.x_title || '';
                }

                // Fetch models for this provider
                if(modelSelect){
                    modelFetchPromises.push(loadModelsForProvider(provider, modelSelect, providerConfig.default_model));
                }
            });

            // Wait for all model fetches to complete
            await Promise.allSettled(modelFetchPromises);
            console.log("Finished fetching models for all providers.");

        } catch (error) {
            console.error('Error loading settings:', error);
            alert('Failed to load settings from the backend.');
        }
    }

     async function loadModelsForProvider(provider, selectElement, savedModelId) {
        console.log(`Fetching models for ${provider}...`);
        try {
            const response = await fetch(`/api/models?provider=${provider}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            const models = data.models || [];

            // Clear existing options except the placeholder
            selectElement.innerHTML = '<option value="" disabled>Select a model</option>';

            if (models.length === 0) {
                selectElement.innerHTML = '<option value="" disabled>No models found</option>';
                console.warn(`No models found for provider: ${provider}`);
                return;
            }

            // Populate options
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id; // Use the unique model ID
                // Use name if available, otherwise fallback to id
                option.textContent = model.name ? `${model.name} (${model.id})` : model.id;
                selectElement.appendChild(option);
            });

            // Set the selected value based on saved config or adapter's default
            if (savedModelId && selectElement.querySelector(`option[value="${savedModelId}"]`)) {
                 selectElement.value = savedModelId;
            } else if (models.length > 0) {
                 // If saved value is invalid or missing, select the first available model
                 selectElement.value = models[0].id;
                 console.log(`Default model for ${provider} ('${savedModelId}') not found or invalid, selecting first available: ${models[0].id}`);
            } else {
                 selectElement.value = ""; // No models available
            }
            console.log(`Models loaded for ${provider}. Selected: ${selectElement.value}`);


        } catch (error) {
            console.error(`Error fetching models for ${provider}:`, error);
            selectElement.innerHTML = '<option value="" disabled>Error loading models</option>';
        }
    }

    async function saveSettings(settingsPayload) {
        console.log("Saving settings to backend...");
        try {
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settingsPayload)
            });

            const result = await response.json();

            if (response.ok && result.status === 'success') {
                alert('Settings saved successfully!');
                // Optionally reload or provide feedback
                loadSettings(); // Reload settings after saving to reflect changes
            } else {
                console.error("Save settings failed:", result);
                alert(`Error saving settings: ${result.message || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error sending settings:', error);
            alert(`Failed to save settings: ${error.message}`);
        }
    }

    async function discoverModels() {
        alert("Starting model discovery... This might take a moment.");
        try {
            const response = await fetch('/api/discover_models', { method: 'POST' });
            const result = await response.json();
            if (response.ok && result.status === 'success') {
                 alert(`Model discovery finished! ${result.message || ''} Reloading model lists.`);
                 // Trigger a reload of models in the dropdowns
                 loadSettings(); // Reload settings will re-fetch models
            } else {
                console.error("Discover models failed:", result);
                alert(`Error during model discovery: ${result.message || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error triggering model discovery:', error);
            alert(`Failed to trigger model discovery: ${error.message}`);
        }
    }


    // --- UI Interaction ---

    function applyTheme(themeValue) {
        document.body.classList.remove('light-theme', 'dark-theme');
        if (themeValue === 'light') {
            document.body.classList.add('light-theme');
        } else if (themeValue === 'dark') {
            document.body.classList.add('dark-theme');
        } else {
            // Handle 'system' theme - check preference
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                document.body.classList.add('dark-theme');
            } else {
                // Default to light if system preference is light or unknown
                 document.body.classList.remove('dark-theme'); // Ensure dark is removed if switching from dark default
                 document.body.classList.add('light-theme'); // Or just remove dark and rely on base CSS
            }
        }
    }

    function toggleProviderSettingsVisibility(selectedProvider) {
        providerSettingsDivs.forEach(div => {
            if (div.id === `${selectedProvider}-settings`) {
                div.style.display = 'block';
            } else {
                div.style.display = 'none';
            }
        });
    }

    function switchTab(targetTab) {
        // Hide all sections
        document.querySelectorAll('main .section').forEach(section => {
            section.classList.remove('active-section');
        });
        // Deactivate all tabs
        document.querySelectorAll('header nav ul li a').forEach(tab => {
            tab.classList.remove('active');
        });

        // Show target section
        document.getElementById(`${targetTab}-section`).classList.add('active-section');
        // Activate target tab
        document.querySelector(`a[data-tab='${targetTab}']`).classList.add('active');

        // Reload settings if switching to settings tab
        if (targetTab === 'settings') {
            loadSettings();
        }
    }

    // --- Event Listeners ---

    sendButton.addEventListener('click', () => {
        const messageText = messageInput.value.trim();
        if (messageText) {
            addMessage('user', messageText);
            sendMessageToServer(messageText, currentSessionId);
            messageInput.value = '';
            adjustTextareaHeight(); // Reset height after sending
        }
    });

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent newline
            sendButton.click();
        }
    });

    messageInput.addEventListener('input', adjustTextareaHeight);

    // Tab switching
    chatTab.addEventListener('click', (e) => { e.preventDefault(); switchTab('chat'); });
    settingsTab.addEventListener('click', (e) => { e.preventDefault(); switchTab('settings'); });

    // Settings changes
    themeSelect.addEventListener('change', () => applyTheme(themeSelect.value));
    fontSizeSlider.addEventListener('input', () => {
        const size = fontSizeSlider.value;
        fontSizeValue.textContent = `${size}px`;
        document.body.style.setProperty('--font-size', `${size}px`);
    });

    defaultProviderSelect.addEventListener('change', () => {
        toggleProviderSettingsVisibility(defaultProviderSelect.value);
    });

    saveSettingsButton.addEventListener('click', async () => {
        const settingsToSave = {};
        const apiKeyFields = {}; // Keep API keys separate for potential backend handling

        // --- Collect General UI Settings ---
        settingsToSave['api.default_provider'] = document.getElementById('default-provider').value;
        settingsToSave['ui.theme'] = document.getElementById('theme').value;
        settingsToSave['ui.font_size'] = parseInt(document.getElementById('font-size').value, 10);
        settingsToSave['ui.animations_enabled'] = document.getElementById('animations').checked;
        settingsToSave['ui.compact_mode'] = document.getElementById('compact-mode').checked;


        // --- Collect Provider Specific Settings ---
        const providers = ['openrouter', 'anthropic', 'deepseek', 'ollama', 'litellm'];
        providers.forEach(provider => {
            const providerConfigPrefix = `api.providers.${provider}`;

            // API Key (sent separately for potential secret handling)
            const apiKeyInput = document.getElementById(`${provider}-api-key`);
            if (apiKeyInput) {
                // Send the key under a distinct name if non-empty, otherwise backend should handle removal
                if (apiKeyInput.value) {
                     apiKeyFields[`${provider}-api-key`] = apiKeyInput.value;
                 } else {
                      // Send empty value to signal potential removal on backend
                      apiKeyFields[`${provider}-api-key`] = "";
                 }
            }

            // Base URL / Host URL
            const baseUrlInput = document.getElementById(`${provider}-base_url`);
            if (baseUrlInput) {
                settingsToSave[`${providerConfigPrefix}.base_url`] = baseUrlInput.value;
            }

            // Default Model
            const modelSelect = document.getElementById(`${provider}-model`);
            if (modelSelect) {
                settingsToSave[`${providerConfigPrefix}.default_model`] = modelSelect.value;
            }

            // Embedding Model
            const embeddingModelInput = document.getElementById(`${provider}-embedding-model`);
            if (embeddingModelInput) {
                settingsToSave[`${providerConfigPrefix}.embedding_model`] = embeddingModelInput.value;
            }

            // Discovery Enabled
            const discoveryEnabledCheckbox = document.getElementById(`${provider}-discovery_enabled`);
            if (discoveryEnabledCheckbox) {
                settingsToSave[`${providerConfigPrefix}.discovery_enabled`] = discoveryEnabledCheckbox.checked;
            }

             // Discovery Ports (Handle potential future UI for this)
             // const discoveryPortsInput = document.getElementById(`${provider}-discovery_ports`);
             // if (discoveryPortsInput) {
             //     try {
             //        // Assume comma-separated list of numbers
             //        const ports = discoveryPortsInput.value.split(',').map(p => parseInt(p.trim())).filter(p => !isNaN(p));
             //        settingsToSave[`${providerConfigPrefix}.discovery_ports`] = ports;
             //     } catch (e) { console.error("Invalid format for discovery ports for", provider); }
             // }

            // OpenRouter Specific Headers
            if (provider === 'openrouter') {
                settingsToSave[`${providerConfigPrefix}.referer`] = document.getElementById('openrouter-referer').value;
                settingsToSave[`${providerConfigPrefix}.x_title`] = document.getElementById('openrouter-x_title').value;
            }
        });

        // Combine API keys with regular settings for the single POST request
        const combinedPayload = { ...settingsToSave, ...apiKeyFields };

        // *** LOGGING FOR DEBUGGING ***
        console.log("Settings Payload being sent:", JSON.stringify(combinedPayload, null, 2));

        // Call the save function
        await saveSettings(combinedPayload);
    });


    // --- Initial Load ---
    loadSettings(); // Load settings when the page loads
    adjustTextareaHeight(); // Initial adjustment for textarea
});
