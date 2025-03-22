# miniManus Requirements Analysis

## Overview
miniManus is a mobile-focused framework similar to Manus but designed to run on Linux in Termux on Android phones. It will use external APIs for inference while maintaining a mobile-optimized UI.

## Core Requirements

1. **Platform Compatibility**
   - Must run entirely on Linux within Termux app on Android phones
   - Must be optimized for mobile devices with limited resources

2. **Inference API Integration**
   - Support for multiple external API providers:
     - Openrouter
     - DeepSeek
     - Anthropic
     - Locally hosted Ollama
     - LiteLLM

3. **User Interface**
   - Mobile-focused UI similar to Manus
   - Must maintain state when navigating between different sections
   - Properly display conversations and user interactions

4. **Function Documentation**
   - Create a central index file listing ALL function names in the framework
   - Categorize functions by module or component
   - Reference this index in the README.md

## Technical Requirements

### API Handling
- Do not hard code API URLs or LLM models in code
- Dynamically determine API settings based on network search results
- Implement proper timeout handling, error catching, and logging for API calls
- Ensure graceful handling of API connection issues

### Model Selection
- Implement dynamic model selection logic
- Select the largest, most capable model available for primary tasks
- Allow flexibility in model selection based on available APIs

### Agent Communication
- Implement robust agent registration and message routing system
- Ensure proper verification before attempting inter-agent communication
- Include proper message type definitions for all communication types

### System Architecture
- Implement proper shutdown procedures with cleanup methods
- Handle exceptions during shutdown for graceful termination
- Include comprehensive system prompts for all agents
- Create a human operator identification system if needed

### Dependencies
- Document all required dependencies with specific version requirements
- Ensure compatibility with Termux environment

## Additional Considerations
- Open-source availability
- Support for helping smaller LLM models with clear function specifications
- Prevention of typos or slight variations that could break function calls
