# The plan

The key issues and a detailed plan to fix them.

## Key Issues Identified

1. **OpenRouter API Integration Problems**:
   - The `ui_manager.py` is accessing model configurations using an inconsistent path (`api.{provider_name}.default_model` instead of `api.providers.{provider_name}.default_model`)
   - The OpenRouter adapter needs enhancements to properly handle API keys and models loading
   - Settings aren't being properly saved when a model is selected

2. **Configuration Path Inconsistencies**:
   - There's inconsistent use of configuration paths across the codebase (sometimes `api.{provider}.default_model` and sometimes `api.providers.{provider}.default_model`)

3. **Models Not Loading**:
   - The system should load available models when an API key is added
   - Currently only showing hardcoded options

## Detailed Fix Plan

### 1. Fix OpenRouter Adapter

First, let's enhance the OpenRouter adapter to properly handle API keys and model loading:

```python
# In minimanus/api/openrouter_adapter.py
def _get_headers(self) -> Dict[str, str]:
    """
    Get headers for API requests.
    
    Returns:
        Dictionary of headers
    """
    api_key = self.config_manager.get_api_key("openrouter")
    if not api_key:
        self.logger.warning("No API key found for OpenRouter")
        return {}
    
    return {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://minimanus.app",
        "X-Title": "miniManus",
        "Content-Type": "application/json"
    }
```

### 2. Fix Configuration Path Inconsistencies

We need to standardize the configuration paths across the codebase. Let's modify key files:

#### a. Update UI Manager

```python
# In minimanus/ui/ui_manager.py (in the _process_chat_message method)
# Get the model for the provider - STANDARDIZED PATH
model = self.config_manager.get_config(f"api.providers.{provider_name.lower()}.default_model")
```

#### b. Update Agent System

```python
# In minimanus/core/agent_system.py (in the _handle_with_llm method)
# Get API key and model - STANDARDIZED PATH
provider_key = default_provider_name.lower()
api_key = self.config_manager.get_api_key(provider_key)
model = self.config_manager.get_config(f"api.providers.{provider_key}.default_model", "")
```

### 3. Fix Settings Storage and Retrieval

Update how settings are saved and retrieved:

```python
# In minimanus/ui/ui_manager.py (in the _handle_post_settings method)
# Update the settings in the config manager
if isinstance(body, dict):
    for key, value in body.items():
        if key == "providers":
            # Handle nested provider settings
            for provider, provider_settings in value.items():
                for setting_key, setting_value in provider_settings.items():
                    # Make sure we handle model selection properly
                    if setting_key == "model":
                        config_path = f"api.providers.{provider}.default_model"
                        ui_manager.config_manager.set_config(config_path, setting_value)
                        ui_manager.logger.info(f"Set {provider} model to {setting_value}")
                    elif setting_key == "apiKey":
                        ui_manager.config_manager.set_api_key(provider, setting_value)
                        ui_manager.logger.info(f"Set API key for {provider}")
                    else:
                        config_path = f"api.providers.{provider}.{setting_key}"
                        ui_manager.config_manager.set_config(config_path, setting_value)
        elif key == "defaultProvider":
            ui_manager.config_manager.set_config("api.default_provider", value)
        else:
            ui_manager.config_manager.set_config(key, value)
```

### 4. Fix API Key Handling

Ensure API keys are properly set and retrieved:

```python
# In minimanus/ui/ui_manager.py
def _handle_post_model(self, body):
    """Handle POST /api/model."""
    # Get the model from the request body
    model = body.get("model", "")
    provider = body.get("provider", "")
    
    if provider and model:
        # Update the model in the config manager using standardized path
        config_path = f"api.providers.{provider}.default_model"
        ui_manager.config_manager.set_config(config_path, model)
        ui_manager.logger.info(f"Updated model for {provider} to: {model}")
        
        # Also update the matching provider in adapters
        provider_map = {
            "openrouter": APIProvider.OPENROUTER,
            "anthropic": APIProvider.ANTHROPIC,
            "deepseek": APIProvider.DEEPSEEK,
            "ollama": APIProvider.OLLAMA,
            "litellm": APIProvider.LITELLM,
        }
        
        provider_enum = provider_map.get(provider.lower())
        if provider_enum:
            adapter = ui_manager.api_manager.get_adapter(provider_enum)
            if adapter:
                adapter.default_model = model
                ui_manager.logger.info(f"Updated {provider} adapter default model to: {model}")
    
    # Save the settings
    ui_manager.config_manager.save_config()
```

### 5. Fix Model Loading in the Frontend

Update the JavaScript to handle loading models properly:

```javascript
// In minimanus/static/script.js
function fetchModelsForProvider(provider) {
    // Show loading indicator in the model dropdown
    const modelDropdown = document.getElementById(`${provider}-model`);
    modelDropdown.innerHTML = '<option value="">Loading models...</option>';
    
    // Add the API key to the request
    const apiKeyInput = document.getElementById(`${provider}-api-key`);
    const apiKey = apiKeyInput ? apiKeyInput.value : '';
    
    fetch(`/api/models?provider=${provider}&apiKey=${encodeURIComponent(apiKey)}`)
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
                
                // Get the currently selected model from settings
                const currentModel = localStorage.getItem(`${provider}_model`);
                if (currentModel && data.models.some(m => m.id === currentModel)) {
                    modelDropdown.value = currentModel;
                } else {
                    // Select the first model by default
                    modelDropdown.value = data.models[0].id;
                }
            } else {
                // No models available
                const option = document.createElement('option');
                option.value = "";
                option.textContent = "No models available";
                modelDropdown.appendChild(option);
            }
            
            // Save the selected model to localStorage
            localStorage.setItem(`${provider}_model`, modelDropdown.value);
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            modelDropdown.innerHTML = '<option value="">Error loading models</option>';
        });
}
```

### 6. Update Backend to Handle API Key in Model Requests

```python
# In minimanus/ui/ui_manager.py
def _handle_get_models(self, parsed_url):
    """Handle GET /api/models."""
    # Parse query parameters
    query = parse_qs(parsed_url.query)
    provider = query.get("provider", ["openrouter"])[0]
    api_key = query.get("apiKey", [""])[0]
    
    # If API key is provided, update the config temporarily for this request
    if api_key:
        ui_manager.config_manager.set_api_key(provider, api_key)
    
    # Get models for the provider
    models = ui_manager._get_models_for_provider(provider)
    
    # Send the response
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.end_headers()
    self.wfile.write(json.dumps({"models": models}).encode("utf-8"))
```

### 7. Fix DeepSeek Adapter

Add the missing `generate_text` method to the DeepSeek adapter:

```python
# In minimanus/api/deepseek_adapter.py
def generate_text(self, messages: List[Dict[str, str]], model: str = None, 
                 temperature: float = 0.7, max_tokens: int = 1024, 
                 api_key: str = None) -> str:
    """
    Generate text using the DeepSeek API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model to use (defaults to configured default model)
        temperature: Temperature parameter for generation
        max_tokens: Maximum number of tokens to generate
        api_key: API key (optional, will use configured key if not provided)
        
    Returns:
        Generated text
    """
    try:
        # Use the provided API key or fall back to configured key
        key = api_key or self.config_manager.get_api_key("deepseek")
        if not key:
            return "I'm sorry, but the DeepSeek API key is not configured. Please check your API settings."
        
        # Use the provided model or fall back to default
        model_name = model or self.default_model
        
        # Create a simple message format
        payload = {
            "messages": messages,
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make the API request synchronously
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        # Import here to avoid circular imports
        import requests
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
            else:
                return "I'm sorry, but the API response did not contain any choices."
        else:
            error_message = f"DeepSeek API error: {response.status_code} - {response.text}"
            self.logger.error(error_message)
            return f"I'm sorry, but the DeepSeek API is not available. Please check your API settings."
            
    except Exception as e:
        error_message = f"Error generating text with DeepSeek: {str(e)}"
        self.logger.error(error_message)
        return f"I'm sorry, but there was an error communicating with the language model: {str(e)}"
```

### 8. Update Settings Panel Default Configuration

Update the settings panel default configurations to use the standardized paths:

```python
# In minimanus/ui/settings_panel.py
# Update all settings registrations to use standardized paths
self.register_setting(Setting(
    "api.providers.openrouter.default_model",
    "Default Model",
    "Default model to use with OpenRouter",
    SettingType.SELECT,
    "openai/gpt-3.5-turbo",
    [
        {"label": "GPT-3.5 Turbo", "value": "openai/gpt-3.5-turbo"},
        {"label": "GPT-4", "value": "openai/gpt-4"},
        {"label": "Claude 3 Opus", "value": "anthropic/claude-3-opus"},
        {"label": "Claude 3 Sonnet", "value": "anthropic/claude-3-sonnet"},
        {"label": "Llama 3 70B", "value": "meta-llama/llama-3-70b-instruct"},
        {"label": "Mistral Large", "value": "mistralai/mistral-large"}
    ],
    "api_openrouter",
    1
))
```

## Implementation Steps

1. Update the `openrouter_adapter.py` file to add the `_get_headers` method and ensure it properly handles API keys
2. Fix the configuration path inconsistencies in `ui_manager.py`, `agent_system.py`, and other relevant files
3. Update the settings storage and retrieval logic in `ui_manager.py`
4. Add the missing `generate_text` method to the `deepseek_adapter.py` file
5. Fix model loading in the frontend and backend
6. Update the settings panel to use standardized configuration paths

This comprehensive approach addresses all the issues identified and ensures that the OpenRouter adapter, configuration paths, and model loading are all fixed. The changes maintain consistency across the codebase and improve the user experience, allowing users to add their API keys and see available models.

## Files that need to be updated according to the plan:

1. `minimanus/api/openrouter_adapter.py`
2. `minimanus/api/deepseek_adapter.py`
3. `minimanus/ui/ui_manager.py`
4. `minimanus/core/agent_system.py`
5. `minimanus/ui/settings_panel.py`
6. `minimanus/static/script.js`

Now please provide the `openrouter_adapter.py` file first and nothing else. Thanks!
