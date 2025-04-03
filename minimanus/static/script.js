// START OF FILE miniManus-main/static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const settingsTab = document.getElementById('settings-tab');
    const chatTab = document.getElementById('chat-tab');
    const settingsSection = document.getElementById('settings-section');
    const chatSection = document.getElementById('chat-section');
    const saveSettingsButton = document.getElementById('save-settings');

    // Settings elements
    const defaultProviderSelect = document.getElementById('default-provider');
    const providerSettingsDivs = document.querySelectorAll('.provider-settings');
    const themeSelect = document.getElementById('theme');
    const fontSizeSlider = document.getElementById('font-size');
    const fontSizeValue = document.getElementById('font-size-value');
    const animationsCheckbox = document.getElementById('animations');
    const compactModeCheckbox = document.getElementById('compact-mode');

    // --- Global State (Simple) ---
    let currentSettings = {}; // Store loaded settings
    let availableModels = {}; // Store models fetched per provider: { providerName: [modelData] }

    // --- Theme Handling ---
    const applyTheme = (theme) => {
        document.body.classList.remove('light-theme', 'dark-theme');
        if (theme === 'light') {
            document.body.classList.add('light-theme');
        } else if (theme === 'dark') {
            document.body.classList.add('dark-theme');
        } else { // System theme
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            if (prefersDark) {
                document.body.classList.add('dark-theme');
            } else {
                document.body.classList.add('light-theme'); // Default to light if system pref unknown/light
            }
        }
    };

    const setInitialTheme = (savedTheme) => {
        applyTheme(savedTheme || 'dark'); // Default to dark if no setting
    };

    // --- Font Size Handling ---
    const applyFontSize = (size) => {
        const sizePx = size + 'px';
        document.body.style.fontSize = sizePx;
        if (fontSizeValue) {
            fontSizeValue.textContent = sizePx;
        }
    };

    // --- API Interaction ---
    const apiRequest = async (endpoint, method = 'GET', body = null) => {
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
            console.debug(`API Request: ${method} ${endpoint}`, body ? options.body : '');
            const response = await fetch(endpoint, options);
            if (!response.ok) {
                let errorData;
                try {
                    errorData = await response.json();
                } catch (e) {
                    errorData = { message: `HTTP error ${response.status}` };
                }
                console.error(`API Error (${response.status}):`, errorData);
                throw new Error(errorData.message || `HTTP error ${response.status}`);
            }
            // Handle empty response for methods like POST that might return 200 OK with no body
            if (response.status === 204 || response.headers.get('Content-Length') === '0') {
                return null; // Or return an empty object/success indicator as needed
            }
            const data = await response.json();
            console.debug(`API Response: ${method} ${endpoint}`, data);
            return data;
        } catch (error) {
            console.error(`API Request Failed: ${method} ${endpoint}`, error);
            // Optionally display error to user
            // addMessage('assistant', `Error communicating with backend: ${error.message}`);
            throw error; // Re-throw for caller handling if necessary
        }
    };

    // --- Settings ---

    const updateProviderSettingsVisibility = () => {
        const selectedProvider = defaultProviderSelect.value;
        providerSettingsDivs.forEach(div => {
            div.style.display = div.id === `${selectedProvider}-settings` ? 'block' : 'none';
        });
        // Fetch models for the newly selected provider if not already fetched
        if (selectedProvider && !availableModels[selectedProvider]) {
            fetchAndPopulateModels(selectedProvider);
        } else if (selectedProvider && availableModels[selectedProvider]) {
            // Models already fetched, just populate dropdown
             populateModelDropdown(selectedProvider, availableModels[selectedProvider]);
        }
    };

    // Function to fetch models for a specific provider
    const fetchAndPopulateModels = async (providerName) => {
        const modelSelect = document.getElementById(`${providerName}-model`);
        if (!modelSelect) return; // Element doesn't exist

        // Show loading state
        modelSelect.innerHTML = '<option value="" disabled selected>Loading models...</option>';
        modelSelect.disabled = true;

        try {
            console.log(`Fetching models for provider: ${providerName}...`);
            const data = await apiRequest(`/api/models?provider=${providerName}`);
            availableModels[providerName] = data.models || []; // Store fetched models
            populateModelDropdown(providerName, availableModels[providerName]);
            console.log(`Successfully fetched ${availableModels[providerName].length} models for ${providerName}.`);

             // Restore selected value after populating
            const savedModel = currentSettings?.api?.providers?.[providerName]?.default_model;
            if (savedModel) {
                modelSelect.value = savedModel;
                 // Fallback if saved value isn't in the list
                if (modelSelect.value !== savedModel && modelSelect.options.length > 1) {
                     modelSelect.selectedIndex = 1; // Select the first actual model
                     console.warn(`Saved default model '${savedModel}' not found for ${providerName}. Selecting first available.`);
                } else if (modelSelect.value !== savedModel) {
                     // No models available other than "loading..."
                     modelSelect.innerHTML = '<option value="" disabled selected>No models found</option>';
                }
            } else if(modelSelect.options.length > 1) {
                // If no saved model, select the first actual model in the list
                 modelSelect.selectedIndex = 1;
            } else if (modelSelect.options.length <= 1){
                 modelSelect.innerHTML = '<option value="" disabled selected>No models found</option>';
            }

        } catch (error) {
            console.error(`Failed to fetch models for ${providerName}:`, error);
            modelSelect.innerHTML = '<option value="" disabled selected>Error loading models</option>';
            // Optionally add a retry button or mechanism
        } finally {
            modelSelect.disabled = false; // Re-enable select even on error
        }
    };

    // Function to populate a model dropdown
    const populateModelDropdown = (providerName, models) => {
        const modelSelect = document.getElementById(`${providerName}-model`);
        if (!modelSelect) return;

        // Clear existing options except the placeholder
        modelSelect.innerHTML = '<option value="" disabled selected>Select a model...</option>';

        if (models && models.length > 0) {
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                // Use name for display, fallback to id
                option.textContent = model.name ? `${model.name} (${model.id})` : model.id;
                modelSelect.appendChild(option);
            });
        } else {
            modelSelect.innerHTML = '<option value="" disabled selected>No models found</option>';
        }
    };


    const loadSettings = async () => {
        try {
            console.log("Loading settings from backend...");
            currentSettings = await apiRequest('/api/settings');
            console.log("Settings loaded:", currentSettings);

            // Populate UI elements
            if (currentSettings.ui) {
                themeSelect.value = currentSettings.ui.theme || 'dark';
                fontSizeSlider.value = currentSettings.ui.font_size || 14;
                animationsCheckbox.checked = currentSettings.ui.animations_enabled !== false; // Default true
                compactModeCheckbox.checked = currentSettings.ui.compact_mode === true; // Default false
                applyFontSize(fontSizeSlider.value); // Apply loaded font size
                setInitialTheme(themeSelect.value); // Apply loaded theme
            } else {
                 // Apply defaults if ui section missing
                 applyFontSize(14);
                 setInitialTheme('dark');
            }


            if (currentSettings.api && currentSettings.api.default_provider) {
                defaultProviderSelect.value = currentSettings.api.default_provider;
            }

            // Populate provider-specific fields
            const providers = currentSettings?.api?.providers || {};
            for (const providerName in providers) {
                const providerConfig = providers[providerName];
                const apiKeyInput = document.getElementById(`${providerName}-api-key`);
                const modelSelect = document.getElementById(`${providerName}-model`);
                const baseUrlInput = document.getElementById(`${providerName}-base_url`);
                const discoveryCheckbox = document.getElementById(`${providerName}-discovery_enabled`);
                 const embeddingModelInput = document.getElementById(`${providerName}-embedding-model`); // Generic embedding model field name
                 const refererInput = document.getElementById(`${providerName}-referer`);
                 const xTitleInput = document.getElementById(`${providerName}-x_title`);


                // API Key (Secrets are not sent, so this field will be blank)
                if (apiKeyInput) {
                    apiKeyInput.value = ''; // Clear on load, user needs to re-enter if changed
                    apiKeyInput.placeholder = 'Enter API Key (if needed)';
                }
                // Base URL
                if (baseUrlInput && providerConfig.base_url) {
                    baseUrlInput.value = providerConfig.base_url;
                }
                 // Embedding Model
                 if (embeddingModelInput && providerConfig.embedding_model) {
                     embeddingModelInput.value = providerConfig.embedding_model;
                 }
                 // Discovery Enabled
                 if (discoveryCheckbox) {
                     discoveryCheckbox.checked = providerConfig.discovery_enabled !== false; // Default true if not present
                 }
                 // OpenRouter Specific Headers
                 if (refererInput && providerConfig.referer) {
                     refererInput.value = providerConfig.referer;
                 }
                  if (xTitleInput && providerConfig.x_title) {
                     xTitleInput.value = providerConfig.x_title;
                 }


                // Default Model (set value, but don't populate options yet)
                if (modelSelect && providerConfig.default_model) {
                    // We just set the value for now. fetchAndPopulateModels will handle options.
                     // Ensure the select has at least the saved option temporarily
                     let found = false;
                     for (let i = 0; i < modelSelect.options.length; i++) {
                         if (modelSelect.options[i].value === providerConfig.default_model) {
                             found = true;
                             break;
                         }
                     }
                     if (!found) {
                          const tempOption = document.createElement('option');
                          tempOption.value = providerConfig.default_model;
                          tempOption.textContent = `${providerConfig.default_model} (saved)`;
                          tempOption.selected = true;
                          modelSelect.appendChild(tempOption);
                     }
                     modelSelect.value = providerConfig.default_model;
                }
            }

            updateProviderSettingsVisibility(); // Show correct provider section initially
            console.log("Settings applied to UI.");


        } catch (error) {
            console.error("Failed to load settings:", error);
            // Handle error (e.g., show message to user)
        }
    };

    const saveSettings = async () => {
        const settingsToSave = {
            "ui.theme": themeSelect.value,
            "ui.font_size": parseInt(fontSizeSlider.value, 10),
            "ui.animations_enabled": animationsCheckbox.checked,
            "ui.compact_mode": compactModeCheckbox.checked,
            "api.default_provider": defaultProviderSelect.value,
        };

        // Get provider specific settings
        const providers = currentSettings?.api?.providers ? Object.keys(currentSettings.api.providers) : ['openrouter', 'anthropic', 'deepseek', 'ollama', 'litellm']; // Fallback list
        providers.forEach(providerName => {
            const apiKeyInput = document.getElementById(`${providerName}-api-key`);
            const modelSelect = document.getElementById(`${providerName}-model`);
            const baseUrlInput = document.getElementById(`${providerName}-base_url`);
            const discoveryCheckbox = document.getElementById(`${providerName}-discovery_enabled`);
            const embeddingModelInput = document.getElementById(`${providerName}-embedding-model`);
            const refererInput = document.getElementById(`${providerName}-referer`);
            const xTitleInput = document.getElementById(`${providerName}-x_title`);


            // Save API key ONLY if user entered something (it's not loaded from backend)
            if (apiKeyInput && apiKeyInput.value) {
                settingsToSave[`api.${providerName}.api_key`] = apiKeyInput.value;
            }
            if (modelSelect) {
                 // Ensure a value is selected before saving, don't save the placeholder ""
                 if(modelSelect.value) {
                    settingsToSave[`api.${providerName}.default_model`] = modelSelect.value;
                 } else {
                     // Optionally save null or remove the key if nothing is selected
                     console.warn(`No model selected for ${providerName}, not saving default_model.`);
                 }
            }
            if (baseUrlInput) {
                settingsToSave[`api.${providerName}.base_url`] = baseUrlInput.value;
            }
            if (discoveryCheckbox) {
                 settingsToSave[`api.${providerName}.discovery_enabled`] = discoveryCheckbox.checked;
            }
            if (embeddingModelInput) {
                 settingsToSave[`api.${providerName}.embedding_model`] = embeddingModelInput.value;
            }
            // OpenRouter Specific Headers
            if (refererInput) {
                settingsToSave[`api.${providerName}.referer`] = refererInput.value;
            }
            if (xTitleInput) {
                settingsToSave[`api.${providerName}.x_title`] = xTitleInput.value;
            }
        });

        console.log("Saving settings:", settingsToSave);

        try {
            await apiRequest('/api/settings', 'POST', settingsToSave);
            console.log("Settings saved successfully.");
            // Optionally show success message
            alert("Settings saved!");
             // Clear password fields after successful save
             providers.forEach(providerName => {
                 const apiKeyInput = document.getElementById(`${providerName}-api-key`);
                 if (apiKeyInput) apiKeyInput.value = '';
             });
             // Reload settings to confirm changes (optional, good practice)
             await loadSettings();
        } catch (error) {
            console.error("Failed to save settings:", error);
            // Show error message to user
            alert(`Error saving settings: ${error.message}`);
        }
    };


    // --- Chat ---
    const addMessage = (role, text) => {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${role}-message`);
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.textContent = text; // Use textContent for security
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to bottom
    };

    const sendMessage = async () => {
        const messageText = messageInput.value.trim();
        if (!messageText) return;

        addMessage('user', messageText);
        messageInput.value = '';
        messageInput.style.height = '40px'; // Reset height
        sendButton.disabled = true; // Disable button while waiting

        try {
            const response = await apiRequest('/api/chat', 'POST', { message: messageText });
            if (response && response.response) {
                addMessage('assistant', response.response);
            } else {
                addMessage('assistant', 'Received an empty or invalid response.');
            }
        } catch (error) {
            addMessage('assistant', `Error: ${error.message}`);
        } finally {
            sendButton.disabled = false; // Re-enable button
        }
    };

    // --- UI Navigation ---
    const switchTab = (targetTab) => {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active-section');
        });
        // Deactivate all nav links
        document.querySelectorAll('nav a').forEach(link => {
            link.classList.remove('active');
        });

        // Show target section
        const sectionToShow = document.getElementById(`${targetTab}-section`);
        if (sectionToShow) {
            sectionToShow.classList.add('active-section');
        }
        // Activate target nav link
        const linkToActivate = document.querySelector(`nav a[data-tab="${targetTab}"]`);
        if (linkToActivate) {
            linkToActivate.classList.add('active');
        }

        // Fetch models if switching to settings tab and provider models aren't loaded
        if(targetTab === 'settings') {
             const selectedProvider = defaultProviderSelect.value;
             if(selectedProvider && !availableModels[selectedProvider]) {
                 fetchAndPopulateModels(selectedProvider);
             }
        }
    };

    // --- Event Listeners ---
    sendButton.addEventListener('click', sendMessage);

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent newline
            sendMessage();
        }
    });

    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto'; // Temporarily shrink
        messageInput.style.height = (messageInput.scrollHeight) + 'px'; // Set to scroll height
    });

    // Tab switching
    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = e.target.getAttribute('data-tab');
            switchTab(tab);
        });
    });

    // Settings listeners
    saveSettingsButton.addEventListener('click', saveSettings);
    defaultProviderSelect.addEventListener('change', updateProviderSettingsVisibility);
    themeSelect.addEventListener('change', () => applyTheme(themeSelect.value));
    fontSizeSlider.addEventListener('input', () => applyFontSize(fontSizeSlider.value));

    // --- Initialization ---
    loadSettings(); // Load settings when the page loads
    switchTab('chat'); // Start on chat tab

});
// END OF FILE miniManus-main/static/script.js
