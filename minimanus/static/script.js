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
    let isFetchingModels = {}; // Track fetching state per provider

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
                    // If response is not JSON, use status text or default message
                    errorData = { message: response.statusText || `HTTP error ${response.status}` };
                }
                console.error(`API Error (${response.status}):`, errorData);
                throw new Error(errorData.message || `HTTP error ${response.status}`);
            }
            if (response.status === 204 || response.headers.get('Content-Length') === '0') {
                return null;
            }
            // Check content type before parsing JSON
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                 const data = await response.json();
                 console.debug(`API Response: ${method} ${endpoint}`, data);
                 return data;
            } else {
                 // Handle non-JSON success responses if necessary, or consider it an error
                 const textData = await response.text();
                 console.warn(`API Warning: Received non-JSON response for ${method} ${endpoint}`, textData);
                 // Decide how to handle: return text, null, or throw error?
                 // For settings save, a 200 OK might be enough, even if body is missing/not json
                 if (method === 'POST' && endpoint === '/api/settings') return { status: 'success', message: 'Settings updated (non-JSON response)' };
                 // Otherwise, treat as unexpected
                 throw new Error("Received unexpected non-JSON response from server.");
            }
        } catch (error) {
            console.error(`API Request Failed: ${method} ${endpoint}`, error);
            throw error;
        }
    };

    // --- Settings ---

    const updateProviderSettingsVisibility = () => {
        const selectedProvider = defaultProviderSelect.value;
        providerSettingsDivs.forEach(div => {
            div.style.display = div.id === `${selectedProvider}-settings` ? 'block' : 'none';
        });
        // Fetch models for the newly selected provider if not already fetched
        if (selectedProvider && !isFetchingModels[selectedProvider]) {
             // Fetch ONLY if not already fetched OR if the cache is considered stale (e.g., after save)
             // For simplicity now, only fetch if never fetched or if explicitly refreshed
            if (!availableModels[selectedProvider]) {
                fetchAndPopulateModels(selectedProvider);
            } else {
                // Models already fetched, ensure dropdown is populated correctly
                populateModelDropdown(selectedProvider, availableModels[selectedProvider]);
                 // Restore selected value after populating (important if switching back)
                restoreSelectedModel(selectedProvider);
            }
        } else if (selectedProvider && availableModels[selectedProvider]) {
             // If currently fetching, do nothing, let the fetch complete
             // If already fetched, ensure dropdown is correct
             populateModelDropdown(selectedProvider, availableModels[selectedProvider]);
             restoreSelectedModel(selectedProvider);
        }
    };

    // Function to fetch models for a specific provider
    const fetchAndPopulateModels = async (providerName, forceRefresh = false) => {
        const modelSelect = document.getElementById(`${providerName}-model`);
        const refreshButton = document.getElementById(`${providerName}-refresh-models`);
        if (!modelSelect) return;

        // Prevent multiple simultaneous fetches
        if (isFetchingModels[providerName] && !forceRefresh) {
             console.log(`Already fetching models for ${providerName}.`);
             return;
        }

        // Skip fetch if data exists and not forcing refresh
        if (availableModels[providerName] && !forceRefresh) {
            console.log(`Using cached models for ${providerName}.`);
            populateModelDropdown(providerName, availableModels[providerName]);
            restoreSelectedModel(providerName);
            return;
        }


        isFetchingModels[providerName] = true;
        // Show loading state
        modelSelect.innerHTML = '<option value="" disabled selected>Loading models...</option>';
        modelSelect.disabled = true;
        if (refreshButton) refreshButton.disabled = true;


        try {
            console.log(`Fetching models for provider: ${providerName}...`);
            // Clear previous cache if forcing refresh
            if (forceRefresh) {
                 delete availableModels[providerName];
            }
            const data = await apiRequest(`/api/models?provider=${providerName}`);
            availableModels[providerName] = data.models || []; // Store fetched models
            populateModelDropdown(providerName, availableModels[providerName]);
            console.log(`Successfully fetched ${availableModels[providerName].length} models for ${providerName}.`);

             // Restore selected value after populating
             restoreSelectedModel(providerName);

        } catch (error) {
            console.error(`Failed to fetch models for ${providerName}:`, error);
            modelSelect.innerHTML = `<option value="" disabled selected>Error: ${error.message}</option>`;
        } finally {
            modelSelect.disabled = false; // Re-enable select even on error
            if (refreshButton) refreshButton.disabled = false;
            isFetchingModels[providerName] = false;
        }
    };

    // Function to populate a model dropdown
    const populateModelDropdown = (providerName, models) => {
        const modelSelect = document.getElementById(`${providerName}-model`);
        if (!modelSelect) return;

        const currentVal = modelSelect.value; // Remember current selection if any
        modelSelect.innerHTML = '<option value="" disabled>Select a model...</option>'; // Placeholder

        if (models && models.length > 0) {
            models.sort((a, b) => a.id.localeCompare(b.id)); // Sort models alphabetically by ID
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name ? `${model.name} (${model.id})` : model.id;
                // Add description as title for hover effect
                if (model.description) {
                     option.title = model.description;
                }
                modelSelect.appendChild(option);
            });
            // Try to restore previous selection
            if (currentVal && modelSelect.querySelector(`option[value="${CSS.escape(currentVal)}"]`)) {
                 modelSelect.value = currentVal;
            } else if (modelSelect.options.length > 1) {
                 // If previous selection invalid or none, select the first actual model
                 // modelSelect.selectedIndex = 1; // Select first actual model
                 modelSelect.value = ""; // Keep placeholder selected initially
            } else {
                 // No models other than placeholder
                 modelSelect.innerHTML = '<option value="" disabled selected>No models available</option>';
            }

        } else {
            modelSelect.innerHTML = '<option value="" disabled selected>No models found</option>';
        }
    };

    // Function to restore the selected model based on currentSettings
    const restoreSelectedModel = (providerName) => {
         const modelSelect = document.getElementById(`${providerName}-model`);
         if (!modelSelect) return;

         const savedModel = currentSettings?.api?.providers?.[providerName]?.default_model;

         if (savedModel && modelSelect.querySelector(`option[value="${CSS.escape(savedModel)}"]`)) {
             modelSelect.value = savedModel;
             console.debug(`Restored selection for ${providerName} to ${savedModel}`);
         } else if (savedModel) {
              console.warn(`Saved default model '${savedModel}' not found in dropdown for ${providerName}.`);
              // Keep placeholder selected
              modelSelect.value = "";
         } else {
              // No saved model, keep placeholder selected
              modelSelect.value = "";
         }

         // Ensure placeholder is selected if value is empty and placeholder exists
         if (!modelSelect.value && modelSelect.options[0] && modelSelect.options[0].disabled) {
             modelSelect.selectedIndex = 0;
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
                const embeddingModelInput = document.getElementById(`${providerName}-embedding-model`);
                const refererInput = document.getElementById(`${providerName}-referer`);
                const xTitleInput = document.getElementById(`${providerName}-x_title`);

                if (apiKeyInput) {
                    apiKeyInput.value = '';
                    apiKeyInput.placeholder = 'Enter API Key (if needed/changed)';
                }
                if (baseUrlInput && providerConfig.base_url) {
                    baseUrlInput.value = providerConfig.base_url;
                }
                if (embeddingModelInput && providerConfig.embedding_model) {
                     embeddingModelInput.value = providerConfig.embedding_model;
                }
                if (discoveryCheckbox) {
                     discoveryCheckbox.checked = providerConfig.discovery_enabled !== false;
                }
                 if (refererInput && providerConfig.referer) {
                     refererInput.value = providerConfig.referer;
                 }
                  if (xTitleInput && providerConfig.x_title) {
                     xTitleInput.value = providerConfig.x_title;
                 }

                // Set the saved default model value (options populated later)
                if (modelSelect && providerConfig.default_model) {
                    // Set the value directly, restoreSelectedModel will handle selection later
                    modelSelect.value = providerConfig.default_model;
                }
            }

            updateProviderSettingsVisibility(); // Show correct section and trigger initial model fetch
            console.log("Settings applied to UI.");

        } catch (error) {
            console.error("Failed to load settings:", error);
            alert(`Error loading settings: ${error.message}`);
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
            if (apiKeyInput && apiKeyInput.value.trim()) { // Use trim()
                settingsToSave[`api.${providerName}.api_key`] = apiKeyInput.value.trim();
            }
            if (modelSelect) {
                 if(modelSelect.value) {
                    settingsToSave[`api.${providerName}.default_model`] = modelSelect.value;
                 }
            }
            if (baseUrlInput) {
                settingsToSave[`api.${providerName}.base_url`] = baseUrlInput.value.trim(); // Use trim()
            }
            if (discoveryCheckbox) {
                 settingsToSave[`api.${providerName}.discovery_enabled`] = discoveryCheckbox.checked;
            }
            if (embeddingModelInput) {
                 settingsToSave[`api.${providerName}.embedding_model`] = embeddingModelInput.value.trim(); // Use trim()
            }
            if (refererInput) {
                settingsToSave[`api.${providerName}.referer`] = refererInput.value.trim(); // Use trim()
            }
            if (xTitleInput) {
                settingsToSave[`api.${providerName}.x_title`] = xTitleInput.value.trim(); // Use trim()
            }
        });

        console.log("Saving settings:", settingsToSave); // Avoid logging full object if keys are present

        try {
            await apiRequest('/api/settings', 'POST', settingsToSave);
            console.log("Settings saved successfully.");
            alert("Settings saved!");
             // Clear password fields after successful save
             providers.forEach(providerName => {
                 const apiKeyInput = document.getElementById(`${providerName}-api-key`);
                 if (apiKeyInput) apiKeyInput.value = '';
             });

             // Refresh current provider's models after saving, as settings might affect availability
             const currentProvider = defaultProviderSelect.value;
             if (currentProvider) {
                  console.log(`Refreshing models for ${currentProvider} after saving settings...`);
                  await fetchAndPopulateModels(currentProvider, true); // Force refresh
             } else {
                 // Reload all settings to reflect changes accurately if no provider selected
                 await loadSettings();
             }


        } catch (error) {
            console.error("Failed to save settings:", error);
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
                // Handle case where response is null or response.response is missing
                 addMessage('assistant', response?.message || 'Received an empty or invalid response from server.');
            }
        } catch (error) {
            addMessage('assistant', `Error: ${error.message}`);
        } finally {
            sendButton.disabled = false; // Re-enable button
            messageInput.focus(); // Refocus input
        }
    };

    // --- UI Navigation ---
    const switchTab = (targetTab) => {
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active-section');
        });
        document.querySelectorAll('nav a').forEach(link => {
            link.classList.remove('active');
        });

        const sectionToShow = document.getElementById(`${targetTab}-section`);
        if (sectionToShow) {
            sectionToShow.classList.add('active-section');
        }
        const linkToActivate = document.querySelector(`nav a[data-tab="${targetTab}"]`);
        if (linkToActivate) {
            linkToActivate.classList.add('active');
        }

        // Fetch models if switching to settings tab and current provider models aren't loaded/cached
        if(targetTab === 'settings') {
             const selectedProvider = defaultProviderSelect.value;
             if(selectedProvider && !availableModels[selectedProvider] && !isFetchingModels[selectedProvider]) {
                 fetchAndPopulateModels(selectedProvider);
             }
        }
    };

     // --- Add Refresh Buttons ---
     const addRefreshButtons = () => {
         const providers = currentSettings?.api?.providers ? Object.keys(currentSettings.api.providers) : ['openrouter', 'anthropic', 'deepseek', 'ollama', 'litellm'];
         providers.forEach(providerName => {
             const modelSettingItem = document.getElementById(`${providerName}-model`)?.closest('.setting-item');
             if (modelSettingItem) {
                 let refreshButton = modelSettingItem.querySelector('.refresh-models-button');
                 if (!refreshButton) { // Only add if it doesn't exist
                     refreshButton = document.createElement('button');
                     refreshButton.textContent = '🔄'; // Refresh icon
                     refreshButton.title = `Refresh ${providerName} models`;
                     refreshButton.classList.add('refresh-models-button'); // Add class for styling
                     refreshButton.style.marginLeft = '10px'; // Add some spacing
                     refreshButton.id = `${providerName}-refresh-models`; // Add ID
                     refreshButton.addEventListener('click', (e) => {
                         e.preventDefault();
                         console.log(`Manual refresh triggered for ${providerName}`);
                         fetchAndPopulateModels(providerName, true); // Pass true to force refresh
                     });
                     // Insert after the select element
                     modelSettingItem.appendChild(refreshButton);
                 }
             }
         });
     };


    // --- Event Listeners ---
    sendButton.addEventListener('click', sendMessage);

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent newline
            sendMessage();
        }
    });

    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = (messageInput.scrollHeight) + 'px';
    });

    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = e.target.getAttribute('data-tab');
            switchTab(tab);
        });
    });

    saveSettingsButton.addEventListener('click', saveSettings);
    defaultProviderSelect.addEventListener('change', updateProviderSettingsVisibility);
    themeSelect.addEventListener('change', () => applyTheme(themeSelect.value));
    fontSizeSlider.addEventListener('input', () => applyFontSize(fontSizeSlider.value));

    // --- Initialization ---
    loadSettings().then(() => {
         // Add refresh buttons after settings are loaded and UI elements exist
         addRefreshButtons();
    });
    switchTab('chat'); // Start on chat tab

});
// END OF FILE miniManus-main/static/script.js
