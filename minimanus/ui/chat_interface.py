#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat Interface for miniManus

This module implements the Chat Interface component, which manages chat sessions
and interactions with the user.
"""

import os
import sys
import json
import logging
import uuid
import time
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
    from ..api.api_manager import APIManager, APIProvider, APIRequestType
    from ..core.agent_system import AgentSystem
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager
    from api.api_manager import APIManager, APIProvider, APIRequestType
    from core.agent_system import AgentSystem

logger = logging.getLogger("miniManus.ChatInterface")

class MessageRole(Enum):
    """Message roles in a chat session."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class ChatMessage:
    """A message in a chat session."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ChatSession:
    """A chat session with a history of messages."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Chat"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[ChatMessage] = field(default_factory=list)
    system_prompt: Optional[str] = None
    
    def add_message(self, message: ChatMessage) -> None:
        """
        Add a message to the session.
        
        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary.
        
        Returns:
            Session as dictionary
        """
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in self.messages
            ],
            "system_prompt": self.system_prompt
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """
        Create session from dictionary.
        
        Args:
            data: Session data
            
        Returns:
            Chat session
        """
        session = cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", "New Chat"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            system_prompt=data.get("system_prompt")
        )
        
        for msg_data in data.get("messages", []):
            try:
                role = MessageRole(msg_data.get("role", "user"))
            except ValueError:
                role = MessageRole.USER
            
            message = ChatMessage(
                id=msg_data.get("id", str(uuid.uuid4())),
                role=role,
                content=msg_data.get("content", ""),
                timestamp=msg_data.get("timestamp", time.time())
            )
            
            session.messages.append(message)
        
        return session

class ChatInterface:
    """
    ChatInterface manages chat sessions and interactions with the user.
    
    It handles:
    - Chat session creation and management
    - Message processing and routing
    - Integration with the agent system
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'ChatInterface':
        """Get or create the singleton instance of ChatInterface."""
        if cls._instance is None:
            cls._instance = ChatInterface()
        return cls._instance
    
    def __init__(self):
        """Initialize the ChatInterface."""
        if ChatInterface._instance is not None:
            raise RuntimeError("ChatInterface is a singleton. Use get_instance() instead.")
        
        self.logger = logger
        self.event_bus = EventBus.get_instance()
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        self.api_manager = APIManager.get_instance()
        
        # Initialize agent system
        self.agent_system = AgentSystem.get_instance()
        
        # Chat sessions
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        
        # Load sessions from disk
        self._load_sessions()
        
        # Create default session if none exists
        if not self.sessions:
            default_session = self.create_session("New Chat")
            self.current_session_id = default_session.id
        else:
            # Set current session to most recently updated
            self.current_session_id = max(
                self.sessions.items(),
                key=lambda x: x[1].updated_at
            )[0]
        
        self.logger.info(f"Loaded {len(self.sessions)} chat sessions")
        self.logger.info("ChatInterface initialized")
    
    def _load_sessions(self) -> None:
        """Load chat sessions from disk."""
        sessions_dir = os.path.join(os.path.dirname(__file__), "../data/sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        
        try:
            for filename in os.listdir(sessions_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(sessions_dir, filename)
                    try:
                        with open(file_path, "r") as f:
                            session_data = json.load(f)
                            session = ChatSession.from_dict(session_data)
                            self.sessions[session.id] = session
                    except Exception as e:
                        self.logger.error(f"Error loading session {filename}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading sessions: {str(e)}")
    
    def _save_session(self, session_id: str) -> None:
        """
        Save a chat session to disk.
        
        Args:
            session_id: ID of session to save
        """
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        sessions_dir = os.path.join(os.path.dirname(__file__), "../data/sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        
        file_path = os.path.join(sessions_dir, f"{session_id}.json")
        
        try:
            with open(file_path, "w") as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving session {session_id}: {str(e)}")
    
    def create_session(self, title: str = "New Chat", 
                      system_prompt: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.
        
        Args:
            title: Session title
            system_prompt: System prompt for the session
            
        Returns:
            New chat session
        """
        session = ChatSession(title=title, system_prompt=system_prompt)
        self.sessions[session.id] = session
        self._save_session(session.id)
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Chat session or None if not found
        """
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[ChatSession]:
        """
        Get all chat sessions.
        
        Returns:
            List of all chat sessions
        """
        return list(self.sessions.values())
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if deleted, False if not found
        """
        if session_id not in self.sessions:
            return False
        
        del self.sessions[session_id]
        
        # Delete session file
        sessions_dir = os.path.join(os.path.dirname(__file__), "../data/sessions")
        file_path = os.path.join(sessions_dir, f"{session_id}.json")
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            self.logger.error(f"Error deleting session file {session_id}: {str(e)}")
        
        # Update current session if needed
        if self.current_session_id == session_id:
            if self.sessions:
                self.current_session_id = next(iter(self.sessions.keys()))
            else:
                self.current_session_id = None
        
        return True
    
    def set_current_session(self, session_id: str) -> bool:
        """
        Set the current chat session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if set, False if not found
        """
        if session_id not in self.sessions:
            return False
        
        self.current_session_id = session_id
        return True
    
    async def process_message(self, message: str, session_id: Optional[str] = None) -> str:
        """
        Process a user message.
        
        Args:
            message: User message
            session_id: Session ID (uses current session if None)
            
        Returns:
            Assistant's response
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None or session_id not in self.sessions:
            # Create a new session if none exists
            session = self.create_session()
            session_id = session.id
            self.current_session_id = session_id
        else:
            session = self.sessions[session_id]
        
        # Add user message to session
        user_message = ChatMessage(MessageRole.USER, message)
        session.add_message(user_message)
        
        # Save session
        self._save_session(session_id)
        
        try:
            # Convert messages to format expected by agent system
            conversation_history = []
            
            # Add system message if available
            if session.system_prompt:
                conversation_history.append({
                    "role": "system",
                    "content": session.system_prompt
                })
            
            # Add conversation history (limited to last 10 messages)
            for msg in session.messages[:-1][-10:]:
                conversation_history.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            
            # Process with agent system
            response_text = await self.agent_system.process_user_request(
                message, conversation_history
            )
            
            # Add assistant message to session
            assistant_message = ChatMessage(MessageRole.ASSISTANT, response_text)
            session.add_message(assistant_message)
            
            # Save session
            self._save_session(session_id)
            
            return response_text
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error processing message: {error_msg}")
            
            # Add error message to session
            error_response = f"I'm sorry, but I encountered an error: {error_msg}"
            assistant_message = ChatMessage(MessageRole.ASSISTANT, error_response)
            session.add_message(assistant_message)
            
            # Save session
            self._save_session(session_id)
            
            return error_response
    
    def startup(self) -> None:
        """Start the chat interface."""
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), "../data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create sessions directory if it doesn't exist
        sessions_dir = os.path.join(data_dir, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        
        self.logger.info("ChatInterface started")
    
    def shutdown(self) -> None:
        """Shut down the chat interface."""
        # Save all sessions
        for session_id in self.sessions:
            self._save_session(session_id)
        
        self.logger.info("ChatInterface shut down")
