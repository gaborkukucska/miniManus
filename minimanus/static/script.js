// miniManus-main/minimanus/static/script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const settingsSection = document.getElementById('settings-section');
    const chatSection = document.getElementById('chat-section');
    const settingsTab = document.getElementById('settings-tab');
    const chatTab = document.getElementById('chat-tab');
    // Settings Elements
    const themeSelect = document.getElementById('theme');
    const fontSizeSlider = document.getElementById('font-size');
    const fontSizeValue = document.getElementById('font-size-value');
    const animationsCheckbox = document.getElementById('animations');
    const compactModeCheckbox = document.getElementById('compact-mode');
    const saveSettingsButton = document.getElementById('save-settings');
    const defaultProviderSelect = document.getElementById('default-provider');
    const providerSettingsDivs = document.querySelectorAll('.provider-settings');
    // Add a placeholder for session list and current session info if needed later
    // const sessionList = document.getElementById('session-list');
    // const currentSessionInfo = document.getElementById('current-session-info');

    // --- State Variables ---
    let currentSessionId = null;
    let currentSessionModel = null; // Track the model selected for the current session

    // --- Utility Functions ---

    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${role}-message`);

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        // Basic sanitization (use textContent for safety)
        contentDiv.textContent = content;

        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        // Only scroll if the user is already near the bottom
        const scrollThreshold = 50; // Pixels from bottom
        const isScrolledNearBottom = chatMessages.scrollHeight - chatMessages.clientHeight <= chatMessages.scrollTop + scrollThreshold;
        if (isScrolledNearBottom) {
             chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    function setLoadingState(isLoading) {
        sendButton.disabled = isLoading;
        messageInput.disabled = isLoading;
        sendButton.textContent = isLoading ? '...' : 'Send';
    }

    function adjustTextareaHeight() {
        messageInput.style.height = 'auto'; // Reset height
        const scrollHeight = messageInput.scrollHeight;
        const maxHeight = 120;
        messageInput.style.height = (scrollHeight > maxHeight ? maxHeight : scrollHeight) + 'px';
    }

    // --- API Calls ---

    async function sendMessageToServer(message, sessionId, modelId) {
        setLoadingState(true);
        try {
            const payload = { message: message };
            // Include session_id if available (backend will use current if null)
            if (sessionId) {
                payload.session_id = sessionId;
            }
            // Note: Backend's process_message now reads model from the session object
            // We don't need to send modelId here anymore unless we want to override the session's model
            // if (modelId) {
            //    payload.model_id = modelId; // Or maybe just 'model'? Check backend API handler if needed
            // }
            console.log("Sending message payload:", payload); // Log payload

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
            const savedDefaultProvider = settings.api?.default_provider || 'openrouter';
            defaultProviderSelect.value = savedDefaultProvider;
            toggleProviderSettingsVisibility(savedDefaultProvider); // Show correct section

            // --- Populate Provider Specific Settings ---
            const providers = ['openrouter', 'anthropic', 'deepseek', 'ollama', 'litellm'];
            const modelFetchPromises = [];

            providers.forEach(provider => {
                const providerConfig = settings.api?.providers?.[provider] || {};
                const providerPrefix = `api.providers.${provider}`;

                // API Key (Clear password fields on load)
                const apiKeyInput = document.getElementById(`${provider}-api-key`);
                if (apiKeyInput) {
                     apiKeyInput.value = '';
                     apiKeyInput.placeholder = `Enter ${provider} key...`;
                }

                // Base URL
                const baseUrlInput = document.getElementById(`${provider}-base_url`);
                if (baseUrlInput) {
                    baseUrlInput.value = providerConfig.base_url || '';
                }

                // Default Model Select (Store saved value, load options async)
                const modelSelect = document.getElementById(`${provider}-model`);
                if (modelSelect) {
                    modelSelect.dataset.savedValue = providerConfig.default_model || ''; // Store for later selection
                    modelFetchPromises.push(loadModelsForProvider(provider, modelSelect));
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
            });

            await Promise.allSettled(modelFetchPromises);
            console.log("Finished fetching models for all providers.");

            // Load sessions to get the current session ID
            await loadSessions();

        } catch (error) {
            console.error('Error loading settings:', error);
            alert('Failed to load settings from the backend.');
        }
    }

    async function loadModelsForProvider(provider, selectElement) {
        console.log(`Fetching models for ${provider}...`);
        const savedModelId = selectElement.dataset.savedValue || ''; // Get the saved value
        try {
            const response = await fetch(`/api/models?provider=${provider}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            const models = data.models || [];

            selectElement.innerHTML = '<option value="" disabled>Select a model</option>';

            if (models.length === 0) {
                selectElement.innerHTML = '<option value="" disabled>No models found</option>';
                console.warn(`No models found for provider: ${provider}`);
                return;
            }

            models.sort((a, b) => (a.name || a.id).localeCompare(b.name || b.id)); // Sort alphabetically

            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name ? `${model.name} (${model.id})` : model.id;
                // Add context length if available
                if (model.context_length) {
                     option.textContent += ` [${(model.context_length / 1000).toFixed(0)}k]`;
                }
                selectElement.appendChild(option);
            });

            // Try to set the selected value based on saved config
            if (savedModelId && selectElement.querySelector(`option[value="${savedModelId}"]`)) {
                 selectElement.value = savedModelId;
            } else if (models.length > 0) {
                 // If saved value is invalid/missing, select first available
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
                loadSettings(); // Reload settings to reflect potential changes and ensure consistency
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
                 // Reload settings will re-fetch models and update dropdowns
                 loadSettings();
            } else {
                console.error("Discover models failed:", result);
                alert(`Error during model discovery: ${result.message || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error triggering model discovery:', error);
            alert(`Failed to trigger model discovery: ${error.message}`);
        }
    }

    async function loadSessions() {
        console.log("Loading session info...");
        try {
            const response = await fetch('/api/sessions');
            if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
            const data = await response.json();
            currentSessionId = data.current_session_id;
            console.log("Current session ID:", currentSessionId);
            // TODO: Populate session list UI if needed
            // TODO: Load messages for the current session
        } catch (error) {
            console.error("Failed to load sessions:", error);
            // Handle error, maybe create a new session?
        }
    }


    // --- UI Interaction ---

    function applyTheme(themeValue) {
        document.body.classList.remove('light-theme', 'dark-theme');
        if (themeValue === 'light') {
            document.body.classList.add('light-theme');
        } else if (themeValue === 'dark') {
            document.body.classList.add('dark-theme');
        } else { // System theme
            const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.body.classList.toggle('dark-theme', prefersDark);
            document.body.classList.toggle('light-theme', !prefersDark); // Apply light if not dark
        }
    }

    function toggleProviderSettingsVisibility(selectedProvider) {
        providerSettingsDivs.forEach(div => {
            div.style.display = div.id === `${selectedProvider}-settings` ? 'block' : 'none';
        });
    }

    function switchTab(targetTab) {
        document.querySelectorAll('main .section').forEach(section => section.classList.remove('active-section'));
        document.querySelectorAll('header nav ul li a').forEach(tab => tab.classList.remove('active'));
        document.getElementById(`${targetTab}-section`).classList.add('active-section');
        document.querySelector(`a[data-tab='${targetTab}']`).classList.add('active');
        if (targetTab === 'settings') {
            loadSettings(); // Reload settings each time the tab is viewed
        }
    }

    // --- Event Listeners ---

    sendButton.addEventListener('click', () => {
        const messageText = messageInput.value.trim();
        if (messageText) {
            addMessage('user', messageText);
            // Send using currentSessionId (which might be null initially, backend handles it)
            sendMessageToServer(messageText, currentSessionId);
            messageInput.value = '';
            adjustTextareaHeight();
        }
    });

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendButton.click();
        }
    });

    messageInput.addEventListener('input', adjustTextareaHeight);

    // Tab switching
    chatTab.addEventListener('click', (e) => { e.preventDefault(); switchTab('chat'); });
    settingsTab.addEventListener('click', (e) => { e.preventDefault(); switchTab('settings'); });

    // Settings changes listeners
    themeSelect.addEventListener('change', () => applyTheme(themeSelect.value));
    fontSizeSlider.addEventListener('input', () => {
        const size = fontSizeSlider.value;
        fontSizeValue.textContent = `${size}px`;
        document.body.style.setProperty('--font-size', `${size}px`);
    });
    defaultProviderSelect.addEventListener('change', () => {
        toggleProviderSettingsVisibility(defaultProviderSelect.value);
    });

    // Save Settings Button
    saveSettingsButton.addEventListener('click', async () => {
        const settingsToSave = {};
        const apiKeyFields = {}; // Separate API keys

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

            // API Key (use distinct key for backend identification)
            const apiKeyInput = document.getElementById(`${provider}-api-key`);
            if (apiKeyInput) {
                 // Send even if empty, backend handles removal/update logic
                apiKeyFields[`${provider}-api-key`] = apiKeyInput.value;
            }

            // Base URL / Host URL
            const baseUrlInput = document.getElementById(`${provider}-base_url`);
            if (baseUrlInput) {
                settingsToSave[`${providerConfigPrefix}.base_url`] = baseUrlInput.value.trim(); // Trim whitespace
            }

            // Default Model
            const modelSelect = document.getElementById(`${provider}-model`);
            if (modelSelect) {
                settingsToSave[`${providerConfigPrefix}.default_model`] = modelSelect.value;
            }

            // Embedding Model
            const embeddingModelInput = document.getElementById(`${provider}-embedding-model`);
            if (embeddingModelInput) {
                settingsToSave[`${providerConfigPrefix}.embedding_model`] = embeddingModelInput.value.trim();
            }

            // Discovery Enabled
            const discoveryEnabledCheckbox = document.getElementById(`${provider}-discovery_enabled`);
            if (discoveryEnabledCheckbox) {
                settingsToSave[`${providerConfigPrefix}.discovery_enabled`] = discoveryEnabledCheckbox.checked;
            }

            // OpenRouter Specific Headers
            if (provider === 'openrouter') {
                settingsToSave[`${providerConfigPrefix}.referer`] = document.getElementById('openrouter-referer').value.trim();
                settingsToSave[`${providerConfigPrefix}.x_title`] = document.getElementById('openrouter-x_title').value.trim();
            }
        });

        const combinedPayload = { ...settingsToSave, ...apiKeyFields };
        console.log("Settings Payload being sent:", JSON.stringify(combinedPayload, null, 2));
        await saveSettings(combinedPayload);
    });

    // Add listener for a hypothetical discover models button
    const discoverButton = document.getElementById('discover-models-button'); // Assuming you add this button
    if (discoverButton) {
        discoverButton.addEventListener('click', discoverModels);
    }


    // --- Initial Load ---
    loadSettings(); // Load settings and models on page load
    adjustTextareaHeight(); // Initial adjustment for textarea

    // Add listener for system theme changes
    const darkThemeMq = window.matchMedia("(prefers-color-scheme: dark)");
    darkThemeMq.addEventListener("change", (e) => {
        if (themeSelect.value === 'system') {
            applyTheme('system'); // Re-apply system theme logic
        }
    });

});
