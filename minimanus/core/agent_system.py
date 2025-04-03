# START OF FILE miniManus-main/minimanus/core/agent_system.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent System for miniManus

This module implements the Agent System component, providing agentic capabilities
to the miniManus framework, allowing it to perform complex tasks involving planning
and tool/capability usage.
"""

import os
import sys
import json
import logging
import asyncio
import re
import traceback
from typing import Dict, List, Optional, Any, Union, Callable, Coroutine, Tuple
from pathlib import Path # Added

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
    from ..api.api_manager import APIManager, APIProvider, APIRequestType
    # If capabilities call external utils, import them here
    # from ..utils.web_search import perform_web_search # Example
    # from ..utils.file_operations import manage_file # Example
except ImportError as e:
    # Handle potential import errors during early startup or testing
    logging.getLogger("miniManus.AgentSystem").critical(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(f"ImportError in agent_system.py: {e}. Ensure all components exist.")


logger = logging.getLogger("miniManus.AgentSystem")

# Type alias for capability handlers
CapabilityHandler = Callable[..., Coroutine[Any, Any, Any]] # Must be async

class AgentCapability:
    """Represents a capability (tool) that an agent can use."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Dict[str, Any]], handler: CapabilityHandler):
        """
        Initialize a capability.

        Args:
            name: Unique name of the capability (used in planning).
            description: Description for the LLM to understand when to use it.
            parameters: Dictionary defining the expected parameters for the handler.
                        Format: {"param_name": {"type": "string/number/boolean", "description": "...", "required": True/False}}
            handler: Async function that implements the capability logic.
                     It should accept parameters as keyword arguments based on the 'parameters' definition.
                     It should return a dictionary or string representing the result.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def get_tool_schema(self) -> Dict[str, Any]:
        """Generates an OpenAI-compatible tool schema for this capability."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: {
                            "type": details.get("type", "string"),
                            "description": details.get("description", ""),
                        }
                        for name, details in self.parameters.items()
                    },
                    "required": [name for name, details in self.parameters.items() if details.get("required", False)],
                },
            },
        }


class AgentSystem:
    """
    AgentSystem provides agentic capabilities to miniManus.

    Handles task planning, tool/capability usage, memory/context management (basic),
    and multi-step reasoning by interacting with LLMs and executing capabilities.
    """

    _instance = None  # Singleton instance

    @classmethod
    def get_instance(cls) -> 'AgentSystem':
        """Get or create the singleton instance of AgentSystem."""
        if cls._instance is None:
            cls._instance = AgentSystem()
        return cls._instance

    def __init__(self):
        """Initialize the AgentSystem."""
        if AgentSystem._instance is not None:
            raise RuntimeError("AgentSystem is a singleton. Use get_instance() instead.")

        self.logger = logger
        self.event_bus = EventBus.get_instance()
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        self.api_manager = APIManager.get_instance()

        # Agent capabilities registry
        self.capabilities: Dict[str, AgentCapability] = {}
        self._register_default_capabilities()

        # Agent configuration
        self.max_iterations = self.config_manager.get_config("agent.max_iterations", 5)
        # Store the name of the default provider, not the model directly
        self.default_provider_name = self.config_manager.get_config("agent.default_provider", "openrouter")
        # We get the actual default model from the provider config when needed

        # System prompt can be customized
        self.system_prompt = self._get_default_system_prompt()

        self.logger.info("AgentSystem initialized")

    def _register_default_capabilities(self):
        """Register built-in capabilities."""
        # --- Web Search ---
        self.register_capability(AgentCapability(
            name="web_search",
            description="Search the web for information using a query.",
            parameters={
                "query": {"type": "string", "description": "The search query.", "required": True}
            },
            handler=self._capability_web_search
        ))

        # --- File Read ---
        self.register_capability(AgentCapability(
            name="read_file",
            description="Read the content of a specified file within the allowed directory.",
            parameters={
                "filepath": {"type": "string", "description": "The relative path to the file within the allowed directory.", "required": True}
            },
            handler=self._capability_read_file
        ))

        # --- Add other capabilities here with strong security considerations ---

        self.logger.info(f"Registered {len(self.capabilities)} default capabilities.")

    def register_capability(self, capability: AgentCapability) -> None:
        """Register a new capability."""
        if capability.name in self.capabilities:
             self.logger.warning(f"Replacing existing capability: {capability.name}")
        self.capabilities[capability.name] = capability
        self.logger.debug(f"Registered capability: {capability.name}")

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent."""
        base_prompt = """You are miniManus, a helpful AI assistant running in a mobile environment (Termux on Android).
Your goal is to understand the user's request, plan the necessary steps, use available tools (capabilities) when needed, and provide a comprehensive final answer.
Be concise but complete. Explain your reasoning briefly if the task is complex.
Prioritize resource efficiency (network usage, computation) due to the mobile environment."""

        if self.capabilities:
            tools_description = "\n\nAvailable Tools:\n"
            for name, cap in self.capabilities.items():
                # Generate schema dynamically for the prompt
                params = cap.parameters
                param_desc = ", ".join([f"{p_name} ({details.get('type', 'string')})" for p_name, details in params.items()])
                tools_description += f"- {name}({param_desc}): {cap.description}\n"
            return base_prompt + tools_description
        else:
            return base_prompt + "\n\nNo specific tools are currently available."


    def set_system_prompt(self, prompt: str) -> None:
        """Set a custom system prompt."""
        self.system_prompt = prompt
        self.logger.info("Custom system prompt set for AgentSystem.")

    def _determine_request_complexity(self, user_message: str, history_length: int) -> bool:
        """
        Heuristic to determine if a request likely requires agentic capabilities.
        """
        agent_keywords = [
            "search", "find", "look up", "research", "google", "browse",
            "create", "make", "build", "develop", "write code", "program", "script",
            "analyze", "examine", "investigate", "calculate", "compute", "solve",
            "file", "save", "read", "list directory",
            "step by step", "plan", "organize", "how to",
            "latest", "current", "real-time", "update",
        ]
        message_lower = user_message.lower()
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', message_lower) for keyword in agent_keywords):
            return True

        if len(user_message.split()) > 25 or history_length > 4:
            return True

        if re.search(r'\b(what is|who is|when did|current|latest)\b', message_lower):
             if not re.search(r'\b(your name|you are|your purpose)\b', message_lower):
                 return True

        return False

    async def process_user_request(self, user_message: str, conversation_history: List[Dict[str, str]], requested_model_id: Optional[str] = None) -> str:
        """
        Processes a user request, deciding whether to use agentic capabilities or respond directly.

        Args:
            user_message: The latest message from the user.
            conversation_history: Formatted list of previous messages (including system prompt if any).
            requested_model_id: The model ID preferred for this request (e.g., from session).

        Returns:
            The final response string for the user.
        """
        if self._determine_request_complexity(user_message, len(conversation_history)):
            self.logger.info("Complex request detected, engaging agentic loop.")
            try:
                # Pass the requested model ID to the agentic loop
                return await self._agentic_loop(conversation_history + [{"role": "user", "content": user_message}], requested_model_id)
            except Exception as e:
                self.logger.error(f"Error during agentic processing: {e}", exc_info=True)
                self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.ERROR, {"action": "agentic_loop"})
                return f"I encountered an internal error trying to process your request agentically: {type(e).__name__}"
        else:
            self.logger.info("Simple request detected, responding directly.")
            try:
                # Determine model for direct call (use requested or default provider's default)
                model_to_use = requested_model_id
                if not model_to_use:
                     default_provider = self.config_manager.get_config("api.default_provider", "openrouter")
                     model_to_use = self.config_manager.get_config(f"api.providers.{default_provider}.default_model")
                     self.logger.debug(f"Using default provider's model '{model_to_use}' for direct call.")

                if not model_to_use: # Final fallback if config is missing
                    model_to_use = "openai/gpt-3.5-turbo" # A known common model as last resort
                    self.logger.warning(f"Using hardcoded fallback model '{model_to_use}' for direct call.")

                messages_for_llm = conversation_history + [{"role": "user", "content": user_message}]
                response_data = await self.api_manager.send_request(
                    request_type=APIRequestType.CHAT,
                    request_data={"model": model_to_use, "messages": messages_for_llm}
                    # Consider passing preferred_provider based on model_to_use if possible
                )

                if "error" in response_data:
                    return f"Sorry, I couldn't get a response: {response_data['error']}"
                elif response_data.get("choices") and response_data["choices"][0].get("message"):
                    return response_data["choices"][0]["message"].get("content", "Sorry, I received an empty response.")
                else:
                    self.logger.warning(f"Received unexpected direct LLM response format: {response_data}")
                    return "Sorry, I received an unexpected response format."

            except Exception as e:
                self.logger.error(f"Error during direct LLM call: {e}", exc_info=True)
                self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.ERROR, {"action": "direct_llm_call"})
                return f"Sorry, I encountered an error contacting the language model: {type(e).__name__}"


    async def _agentic_loop(self, messages: List[Dict[str, str]], requested_model_id: Optional[str] = None) -> str:
        """
        The core loop for handling complex requests using planning and tool execution.
        Uses the requested_model_id or falls back appropriately.
        """
        current_messages = list(messages) # Work on a copy
        iterations = 0
        last_used_provider: Optional[APIProvider] = None # Track the provider used
        last_used_model: Optional[str] = requested_model_id # Start with requested

        while iterations < self.max_iterations:
            iterations += 1
            self.logger.info(f"Agent Iteration: {iterations}/{self.max_iterations}")

            # --- Prepare Tools for LLM ---
            tools = [cap.get_tool_schema() for cap in self.capabilities.values()]

            # --- Determine Model for this Iteration ---
            # Priority: last_used_model (if valid) -> requested_model_id -> agent_default -> hardcoded fallback
            model_to_use_this_iteration = last_used_model if last_used_model else requested_model_id

            if not model_to_use_this_iteration:
                 # Fallback to agent's default provider's default model
                 fallback_provider_name = self.config_manager.get_config("agent.default_provider", self.default_provider_name)
                 model_to_use_this_iteration = self.config_manager.get_config(f"api.providers.{fallback_provider_name}.default_model")
                 self.logger.debug(f"Using agent's default model: {model_to_use_this_iteration} from provider {fallback_provider_name}")

            if not model_to_use_this_iteration:
                 # Absolute fallback if everything else fails
                 model_to_use_this_iteration = "openai/gpt-3.5-turbo" # Should be configured better
                 self.logger.error(f"CRITICAL FALLBACK: Using hardcoded model {model_to_use_this_iteration} as no other model could be determined.")

            self.logger.info(f"Agent loop using model: {model_to_use_this_iteration}")

            # --- Call LLM for Planning/Action ---
            request_data = {
                "model": model_to_use_this_iteration,
                "messages": current_messages,
                "tools": tools,
                "tool_choice": "auto"
            }

            # APIManager's send_request will select the provider based on availability and preferences.
            # It will use the effective_model_name determined within it (which prioritizes the one in request_data).
            llm_response_data = await self.api_manager.send_request(
                request_type=APIRequestType.CHAT,
                request_data=request_data
                # We could try to infer preferred_provider from model_to_use_this_iteration, but let APIManager handle it.
            )

            # --- Process Response ---
            if "error" in llm_response_data:
                self.logger.error(f"LLM call failed in agent loop (Model: {model_to_use_this_iteration}): {llm_response_data['error']}")
                # Pass the model that failed to the final response generator
                return await self._generate_final_response(current_messages, f"LLM Error: {llm_response_data['error']}", model_to_use_this_iteration)

            # Update last used model based on response (if available)
            last_used_model = llm_response_data.get("model", model_to_use_this_iteration)
            # TODO: Get provider used from APIManager response if possible

            assistant_message = llm_response_data.get("choices", [{}])[0].get("message")

            if not assistant_message:
                self.logger.warning("LLM response missing message content in agent loop.")
                return await self._generate_final_response(current_messages, "Received empty response from LLM.", last_used_model)

            current_messages.append(assistant_message)

            # --- Check for Tool Calls ---
            tool_calls = assistant_message.get("tool_calls")
            if not tool_calls:
                self.logger.info("LLM provided a response without tool calls. Ending loop.")
                return assistant_message.get("content", "I have completed the task.")

            # --- Execute Tool Calls ---
            self.logger.info(f"LLM requested {len(tool_calls)} tool call(s).")
            tool_results = await self._execute_tool_calls(tool_calls)

            for result in tool_results:
                 current_messages.append({
                      "tool_call_id": result["tool_call_id"],
                      "role": "tool",
                      "name": result["name"],
                      "content": result["content"],
                 })
            # Loop continues...

        # Max iterations reached
        self.logger.warning(f"Agent loop reached max iterations ({self.max_iterations}). Generating final response.")
        return await self._generate_final_response(current_messages, f"Reached maximum processing steps ({self.max_iterations}).", last_used_model)


    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Executes the tool calls requested by the LLM."""
        results = []
        tasks = []

        for tool_call in tool_calls:
            function_call = tool_call.get("function")
            if not function_call: continue

            tool_name = function_call.get("name")
            tool_call_id = tool_call.get("id")
            if not tool_name or not tool_call_id: continue

            if tool_name not in self.capabilities:
                self.logger.warning(f"LLM requested unknown tool: {tool_name}")
                results.append({
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": f"Error: Tool '{tool_name}' not found.",
                })
                continue

            capability = self.capabilities[tool_name]
            try:
                arguments_str = function_call.get("arguments", "{}")
                arguments = json.loads(arguments_str)
                if not isinstance(arguments, dict):
                     raise ValueError("Arguments must be a JSON object.")

                self.logger.info(f"Executing tool '{tool_name}' with args: {arguments}")
                tasks.append(self._run_capability(capability, arguments, tool_call_id))

            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse arguments for tool {tool_name}: {arguments_str}")
                results.append({ "tool_call_id": tool_call_id, "name": tool_name, "content": "Error: Invalid arguments format (not JSON)." })
            except ValueError as e:
                 self.logger.error(f"Invalid arguments for tool {tool_name}: {e}")
                 results.append({ "tool_call_id": tool_call_id, "name": tool_name, "content": f"Error: {e}" })
            except Exception as e:
                 self.logger.error(f"Error preparing tool call for {tool_name}: {e}", exc_info=True)
                 results.append({ "tool_call_id": tool_call_id, "name": tool_name, "content": f"Error: Unexpected error preparing tool call." })


        if tasks:
            task_results = await asyncio.gather(*tasks)
            results.extend(task_results)

        return results

    async def _run_capability(self, capability: AgentCapability, args: Dict[str, Any], tool_call_id: str) -> Dict[str, str]:
         """Safely runs a single capability handler."""
         try:
             required_params = [p for p, d in capability.parameters.items() if d.get("required")]
             missing = [p for p in required_params if p not in args]
             if missing:
                 return {"tool_call_id": tool_call_id, "name": capability.name, "content": f"Error: Missing required arguments: {', '.join(missing)}"}

             result = await capability.handler(**args)

             if isinstance(result, (dict, list)):
                 result_content = json.dumps(result, indent=2)
             else:
                 result_content = str(result)

             max_result_len = 2000 # Example limit
             if len(result_content) > max_result_len:
                  result_content = result_content[:max_result_len] + "\n... [Result Truncated]"

             return {"tool_call_id": tool_call_id, "name": capability.name, "content": result_content}

         except Exception as e:
             self.logger.error(f"Error executing tool '{capability.name}': {e}", exc_info=True)
             self.error_handler.handle_error(e, ErrorCategory.PLUGIN, ErrorSeverity.ERROR, {"capability": capability.name, "args": args})
             return {"tool_call_id": tool_call_id, "name": capability.name, "content": f"Error executing tool: {type(e).__name__} - {e}"}


    async def _generate_final_response(self, messages: List[Dict[str, str]], reason: str, model_id: Optional[str] = None) -> str:
        """
        Makes a final LLM call without tools to summarize results or explain failure.
        Uses the provided model_id or falls back to agent default.
        """
        self.logger.info(f"Generating final response. Reason: {reason}")

        final_instruction = f"Based on the preceding conversation and tool results (if any), please provide a comprehensive final answer to the initial user request. Reason for concluding: {reason}"
        messages.append({"role": "user", "content": final_instruction})

        # Determine model for final call
        model_to_use = model_id
        if not model_to_use:
             fallback_provider_name = self.config_manager.get_config("agent.default_provider", self.default_provider_name)
             model_to_use = self.config_manager.get_config(f"api.providers.{fallback_provider_name}.default_model")
             self.logger.warning(f"Falling back to agent's default model for final response: {model_to_use}")
        if not model_to_use:
            model_to_use = "openai/gpt-3.5-turbo" # Absolute fallback
            self.logger.error(f"CRITICAL FALLBACK: Using hardcoded model {model_to_use} for final response.")

        self.logger.info(f"Generating final response using model: {model_to_use}")

        try:
            response_data = await self.api_manager.send_request(
                request_type=APIRequestType.CHAT,
                request_data={
                    "model": model_to_use,
                    "messages": messages,
                    # No tools parameter here
                }
            )

            if "error" in response_data:
                self.logger.error(f"LLM call failed during final response generation: {response_data['error']}")
                return f"I tried to process your request but encountered an error during the final step: {response_data['error']}"
            elif response_data.get("choices") and response_data["choices"][0].get("message"):
                return response_data["choices"][0]["message"].get("content", "I have completed the steps, but couldn't generate a final summary.")
            else:
                 self.logger.warning(f"Received unexpected final LLM response format: {response_data}")
                 return "I have finished processing, but the final response format was unexpected."

        except Exception as e:
            self.logger.error(f"Error during final response generation: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.API, ErrorSeverity.ERROR, {"action": "generate_final_response"})
            return f"An error occurred while generating the final response: {type(e).__name__}"


    # --- Default Capability Handlers (Placeholders - NEED REAL IMPLEMENTATIONS) ---

    async def _capability_web_search(self, query: str) -> Dict[str, Any]:
        """Placeholder: Performs a web search."""
        self.logger.info(f"Executing placeholder web search for: {query}")
        # In a real implementation: Use libraries like 'requests'/'aiohttp' + 'beautifulsoup4'
        # or dedicated search APIs (SerpApi, Google Search API etc.)
        await asyncio.sleep(0.5) # Simulate network delay
        return {
            "status": "success",
            "results": [
                {"title": f"Simulated Result 1 for '{query}'", "snippet": "This is a placeholder snippet...", "url": "http://example.com/1"},
                {"title": f"Simulated Result 2 for '{query}'", "snippet": "Another placeholder snippet.", "url": "http://example.com/2"},
            ]
         }

    async def _capability_read_file(self, filepath: str) -> Dict[str, Any]:
        """Placeholder: Reads a file."""
        self.logger.info(f"Executing placeholder file read for: {filepath}")
        # SECURITY: VERY IMPORTANT - Sanitize filepath. Prevent directory traversal ('..').
        # Only allow access within a specific, restricted base directory.
        try:
            allowed_base_dir_str = self.config_manager.get_config("agent.files.allowed_read_dir")
            if not allowed_base_dir_str:
                return {"status": "error", "message": "File reading is disabled (no allowed directory configured)."}

            allowed_base_dir = Path(allowed_base_dir_str).resolve()
            # Resolve the combined path safely
            target_path = (allowed_base_dir / filepath).resolve()

            # Check if the resolved path is still within the allowed base directory
            if not str(target_path).startswith(str(allowed_base_dir)):
                self.logger.warning(f"Attempted file read outside allowed directory: {filepath} -> {target_path}")
                return {"status": "error", "message": "Access denied: File path is outside allowed directory."}

            if not target_path.is_file():
                return {"status": "error", "message": f"File not found: {filepath}"}

            # Limit file size read
            max_read_bytes = 10000
            content = target_path.read_text(encoding='utf-8', errors='ignore')[:max_read_bytes]
            if len(content) >= max_read_bytes:
                 content += "\n... [File Truncated]"

            return {"status": "success", "filepath": filepath, "content": content}
        except ValueError as ve: # Handle errors from Path construction if needed
             self.logger.error(f"Invalid path specified for file read: {filepath} - {ve}")
             return {"status": "error", "message": f"Invalid file path specified: {ve}"}
        except Exception as e:
            self.logger.error(f"Error reading file {filepath}: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to read file: {str(e)}"}

# END OF FILE miniManus-main/minimanus/core/agent_system.py
