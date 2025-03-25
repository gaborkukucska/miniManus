#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent System for miniManus

This module implements the Agent System component, which provides agentic capabilities
to the miniManus framework, allowing it to perform complex tasks autonomously.
"""

import os
import sys
import json
import logging
import asyncio
import re
from typing import Dict, List, Optional, Any, Union, Callable

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
    from ..api.api_manager import APIManager, APIProvider, APIRequestType
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager
    from api.api_manager import APIManager, APIProvider, APIRequestType

logger = logging.getLogger("miniManus.AgentSystem")

class AgentCapability:
    """Represents a capability that an agent can have."""
    
    def __init__(self, name: str, description: str, handler: Callable):
        """
        Initialize a capability.
        
        Args:
            name: Name of the capability
            description: Description of what the capability does
            handler: Function that implements the capability
        """
        self.name = name
        self.description = description
        self.handler = handler

class AgentSystem:
    """
    AgentSystem provides agentic capabilities to miniManus.
    
    It handles:
    - Task planning and execution
    - Tool usage and integration
    - Memory and context management
    - Multi-step reasoning
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
        
        # Agent capabilities
        self.capabilities = {}
        
        # Register default capabilities
        self._register_default_capabilities()
        
        # System prompt for agent
        self.system_prompt = self._get_default_system_prompt()
        
        self.logger.info("AgentSystem initialized")
    
    def _register_default_capabilities(self):
        """Register default capabilities."""
        self.register_capability(
            "web_search",
            "Search the web for information",
            self._web_search_handler
        )
        
        self.register_capability(
            "file_operations",
            "Read, write, and manage files",
            self._file_operations_handler
        )
        
        self.register_capability(
            "code_execution",
            "Execute code in various languages",
            self._code_execution_handler
        )
        
        self.register_capability(
            "task_planning",
            "Break down complex tasks into steps",
            self._task_planning_handler
        )
    
    def register_capability(self, name: str, description: str, handler: Callable) -> None:
        """
        Register a new capability.
        
        Args:
            name: Name of the capability
            description: Description of what the capability does
            handler: Function that implements the capability
        """
        self.capabilities[name] = AgentCapability(name, description, handler)
        self.logger.debug(f"Registered capability: {name}")
    
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for the agent.
        
        Returns:
            Default system prompt
        """
        return """You are miniManus, a mobile-focused AI assistant running on Android through Termux.
You have the following capabilities:
1. Web search for finding information
2. File operations for reading and writing files
3. Code execution for running scripts
4. Task planning for breaking down complex tasks

You should:
- Be helpful, accurate, and concise in your responses
- Use your capabilities when needed to solve problems
- Explain your reasoning and approach
- Adapt to the mobile environment and its constraints

When using external APIs, be mindful of:
- Data usage and costs
- Battery consumption
- Network reliability

Your goal is to assist the user effectively while being efficient with resources."""
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt for the agent.
        
        Args:
            prompt: System prompt to set
        """
        self.system_prompt = prompt
        self.logger.debug("Updated system prompt")
    
    async def process_user_request(self, user_message: str, 
                                 conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Process a user request using agentic capabilities.
        
        Args:
            user_message: User's message
            conversation_history: Previous conversation history
            
        Returns:
            Agent's response
        """
        if conversation_history is None:
            conversation_history = []
        
        # Detect if this is a request that requires agentic capabilities
        if self._requires_agent(user_message):
            return await self._handle_with_agent(user_message, conversation_history)
        else:
            # For simple requests, just use the LLM directly
            return await self._handle_with_llm(user_message, conversation_history)
    
    def _requires_agent(self, message: str) -> bool:
        """
        Determine if a message requires agentic capabilities.
        
        Args:
            message: User message
            
        Returns:
            True if agentic capabilities are needed, False otherwise
        """
        # Keywords that suggest agentic capabilities might be needed
        agent_keywords = [
            "search", "find", "look up", "research",
            "create", "make", "build", "develop",
            "analyze", "examine", "investigate",
            "calculate", "compute", "solve",
            "write code", "program", "script",
            "file", "save", "read", "write",
            "step by step", "plan", "organize",
            "help me with", "can you"
        ]
        
        # Check if any keywords are in the message
        for keyword in agent_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', message.lower()):
                return True
        
        # Check message length - longer messages often need more complex handling
        if len(message.split()) > 15:
            return True
        
        return False
    
    async def _handle_with_agent(self, user_message: str, 
                               conversation_history: List[Dict[str, str]]) -> str:
        """
        Handle a request using agentic capabilities.
        
        Args:
            user_message: User's message
            conversation_history: Previous conversation history
            
        Returns:
            Agent's response
        """
        self.logger.info("Handling request with agent capabilities")
        
        # Prepare messages for the LLM
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add conversation history
        for message in conversation_history:
            messages.append(message)
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # First, ask the LLM to analyze the request and determine needed capabilities
        planning_prompt = """
        Analyze the user's request and determine:
        1. What capabilities are needed to fulfill this request?
        2. What steps should be taken to complete the task?
        3. What information or resources are required?
        
        Format your response as JSON:
        {
            "capabilities": ["capability1", "capability2"],
            "steps": ["step1", "step2", "step3"],
            "resources": ["resource1", "resource2"]
        }
        """
        
        planning_messages = messages.copy()
        planning_messages.append({
            "role": "user",
            "content": planning_prompt
        })
        
        # Get the default provider
        default_provider_name = self.config_manager.get_config("api.default_provider", "openrouter")
        try:
            default_provider = APIProvider[default_provider_name.upper()]
        except (KeyError, ValueError):
            default_provider = APIProvider.OPENROUTER
        
        # Get the adapter
        adapter = self.api_manager.get_adapter(default_provider)
        if not adapter:
            return "I'm sorry, but I couldn't access the language model. Please check your API settings."
        
        # Get API key and model
        provider_key = default_provider_name.lower()
        api_key = self.config_manager.get_config(f"api.{provider_key}.api_key", "")
        model = self.config_manager.get_config(f"api.{provider_key}.default_model", "")
        
        try:
            # Get the plan
            plan_response = await adapter._generate_text_async(
                planning_messages,
                model=model,
                temperature=0.7,
                max_tokens=1024,
                api_key=api_key
            )
            
            # Extract JSON from response
            plan_data = self._extract_json(plan_response)
            
            if not plan_data:
                # If we couldn't parse the plan, fall back to regular LLM response
                return await self._handle_with_llm(user_message, conversation_history)
            
            # Execute the plan
            result = await self._execute_plan(plan_data, user_message, conversation_history)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in agent processing: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from text.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON as dict, or None if no valid JSON found
        """
        # Find JSON-like patterns
        json_pattern = r'({[\s\S]*?})'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    async def _execute_plan(self, plan: Dict, user_message: str, 
                          conversation_history: List[Dict[str, str]]) -> str:
        """
        Execute a plan created by the agent.
        
        Args:
            plan: Plan data
            user_message: Original user message
            conversation_history: Conversation history
            
        Returns:
            Result of plan execution
        """
        capabilities = plan.get("capabilities", [])
        steps = plan.get("steps", [])
        
        # Log the plan
        self.logger.info(f"Executing plan with capabilities: {capabilities}")
        self.logger.info(f"Steps: {steps}")
        
        # For now, we'll simulate execution and return a response
        # In a full implementation, this would actually execute the steps
        
        # Prepare a response about the plan
        response = f"I'll help you with that. Here's my plan:\n\n"
        
        for i, step in enumerate(steps):
            response += f"{i+1}. {step}\n"
        
        response += "\nLet me work on this for you..."
        
        # Get the default provider
        default_provider_name = self.config_manager.get_config("api.default_provider", "openrouter")
        try:
            default_provider = APIProvider[default_provider_name.upper()]
        except (KeyError, ValueError):
            default_provider = APIProvider.OPENROUTER
        
        # Get the adapter
        adapter = self.api_manager.get_adapter(default_provider)
        if not adapter:
            return response + "\n\nI'm sorry, but I couldn't access the language model to complete the task."
        
        # Get API key and model
        provider_key = default_provider_name.lower()
        api_key = self.config_manager.get_config(f"api.{provider_key}.api_key", "")
        model = self.config_manager.get_config(f"api.{provider_key}.default_model", "")
        
        # Prepare messages for final response
        final_messages = []
        
        # Add system prompt
        final_messages.append({
            "role": "system",
            "content": self.system_prompt + "\n\nYou have analyzed the user's request and created a plan. Now provide a helpful response that addresses their needs."
        })
        
        # Add conversation history
        for message in conversation_history:
            final_messages.append(message)
        
        # Add user message
        final_messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Add plan information
        final_messages.append({
            "role": "assistant",
            "content": f"I've analyzed your request and created a plan with these steps: {json.dumps(steps)}"
        })
        
        # Add final instruction
        final_messages.append({
            "role": "user",
            "content": "Please provide a helpful response that addresses my request. Be thorough but concise."
        })
        
        try:
            # Get the final response
            final_response = await adapter._generate_text_async(
                final_messages,
                model=model,
                temperature=0.7,
                max_tokens=1024,
                api_key=api_key
            )
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error generating final response: {str(e)}")
            return response + f"\n\nI encountered an error while completing the task: {str(e)}"
    
    async def _handle_with_llm(self, user_message: str, 
                             conversation_history: List[Dict[str, str]]) -> str:
        """
        Handle a request using just the LLM.
        
        Args:
            user_message: User's message
            conversation_history: Previous conversation history
            
        Returns:
            LLM's response
        """
        self.logger.info("Handling request with direct LLM")
        
        # Prepare messages for the LLM
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add conversation history
        for message in conversation_history:
            messages.append(message)
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Get the default provider
        default_provider_name = self.config_manager.get_config("api.default_provider", "openrouter")
        try:
            default_provider = APIProvider[default_provider_name.upper()]
        except (KeyError, ValueError):
            default_provider = APIProvider.OPENROUTER
        
        # Get the adapter
        adapter = self.api_manager.get_adapter(default_provider)
        if not adapter:
            return "I'm sorry, but I couldn't access the language model. Please check your API settings."
        
        # Get API key and model
        provider_key = default_provider_name.lower()
        api_key = self.config_manager.get_config(f"api.{provider_key}.api_key", "")
        model = self.config_manager.get_config(f"api.{provider_key}.default_model", "")
        
        try:
            # Get the response
            response = await adapter._generate_text_async(
                messages,
                model=model,
                temperature=0.7,
                max_tokens=1024,
                api_key=api_key
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in LLM processing: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    # Capability handlers
    
    async def _web_search_handler(self, query: str) -> str:
        """
        Handle web search capability.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        # This is a placeholder - in a real implementation, this would use a search API
        return f"Simulated web search results for: {query}"
    
    async def _file_operations_handler(self, operation: str, path: str, content: str = None) -> str:
        """
        Handle file operations capability.
        
        Args:
            operation: Operation to perform (read, write)
            path: File path
            content: Content to write (for write operation)
            
        Returns:
            Operation result
        """
        # This is a placeholder - in a real implementation, this would perform file operations
        return f"Simulated file {operation} on {path}"
    
    async def _code_execution_handler(self, code: str, language: str) -> str:
        """
        Handle code execution capability.
        
        Args:
            code: Code to execute
            language: Programming language
            
        Returns:
            Execution result
        """
        # This is a placeholder - in a real implementation, this would execute code
        return f"Simulated execution of {language} code"
    
    async def _task_planning_handler(self, task: str) -> List[str]:
        """
        Handle task planning capability.
        
        Args:
            task: Task to plan
            
        Returns:
            List of steps
        """
        # This is a placeholder - in a real implementation, this would create a task plan
        return [f"Step 1 for {task}", f"Step 2 for {task}", f"Step 3 for {task}"]
