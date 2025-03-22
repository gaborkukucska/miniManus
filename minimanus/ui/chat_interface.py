#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat Interface for miniManus

This module implements the Chat Interface component, which provides a mobile-optimized
chat interface for interacting with LLM models.
"""

import os
import sys
import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum, auto
from pathlib import Path

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
    from ..ui.ui_manager import UIManager
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager
    from ui.ui_manager import UIManager

logger = logging.getLogger("miniManus.ChatInterface")

class MessageRole(Enum):
    """Message role types."""
    USER = auto()
    ASSISTANT = auto()
    SYSTEM = auto()
    FUNCTION = auto()
    TOOL = auto()

class MessageStatus(Enum):
    """Message status types."""
    PENDING = auto()
    SENDING = auto()
    DELIVERED = auto()
    ERROR = auto()
    TYPING = auto()

class ChatMessage:
    """Represents a chat message."""
    
    def __init__(self, role: MessageRole, content: str, timestamp: Optional[float] = None,
                status: MessageStatus = MessageStatus.DELIVERED, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a chat message.
        
        Args:
            role: Role of the message sender
            content: Message content
            timestamp: Message timestamp (defaults to current time)
            status: Message status
            metadata: Additional message metadata
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.status = status
        self.metadata = metadata or {}
        self.id = f"{int(self.timestamp * 1000)}-{id(self)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "id": self.id,
            "role": self.role.name.lower(),
            "content": self.content,
            "timestamp": self.timestamp,
            "status": self.status.name.lower(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """
        Create message from dictionary.
        
        Args:
            data: Dictionary representation of the message
            
        Returns:
            ChatMessage instance
        """
        role = MessageRole[data["role"].upper()]
        content = data["content"]
        timestamp = data.get("timestamp")
        status = MessageStatus[data.get("status", "DELIVERED").upper()]
        metadata = data.get("metadata", {})
        
        msg = cls(role, content, timestamp, status, metadata)
        if "id" in data:
            msg.id = data["id"]
        
        return msg

class ChatSession:
    """Represents a chat session."""
    
    def __init__(self, id: str, title: str, model: str, 
                system_prompt: Optional[str] = None, messages: Optional[List[ChatMessage]] = None):
        """
        Initialize a chat session.
        
        Args:
            id: Session ID
            title: Session title
            model: Model used for the session
            system_prompt: System prompt for the session
            messages: List of messages in the session
        """
        self.id = id
        self.title = title
        self.model = model
        self.system_prompt = system_prompt
        self.messages = messages or []
        self.created_at = time.time()
        self.updated_at = time.time()
    
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
            Dictionary representation of the session
        """
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """
        Create session from dictionary.
        
        Args:
            data: Dictionary representation of the session
            
        Returns:
            ChatSession instance
        """
        id = data["id"]
        title = data["title"]
        model = data["model"]
        system_prompt = data.get("system_prompt")
        messages = [ChatMessage.from_dict(msg) for msg in data.get("messages", [])]
        
        session = cls(id, title, model, system_prompt, messages)
        session.created_at = data.get("created_at", session.created_at)
        session.updated_at = data.get("updated_at", session.updated_at)
        
        return session

class ChatInterface:
    """
    ChatInterface provides a mobile-optimized chat interface for miniManus.
    
    It handles:
    - Chat session management
    - Message sending and receiving
    - Chat history persistence
    - Mobile-optimized UI rendering
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
        self.ui_manager = UIManager.get_instance()
        
        # Chat sessions
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        
        # Chat settings
        self.default_model = self.config_manager.get_config(
            "chat.default_model", 
            "gpt-3.5-turbo"
        )
        self.default_system_prompt = self.config_manager.get_config(
            "chat.default_system_prompt", 
            "You are a helpful assistant."
        )
        self.max_history_length = self.config_manager.get_config(
            "chat.max_history_length", 
            100
        )
        self.auto_save = self.config_manager.get_config(
            "chat.auto_save", 
            True
        )
        
        # Register event handlers
        self.event_bus.subscribe("chat.message_sent", self._handle_message_sent)
        self.event_bus.subscribe("chat.message_received", self._handle_message_received)
        self.event_bus.subscribe("chat.session_created", self._handle_session_created)
        self.event_bus.subscribe("chat.session_deleted", self._handle_session_deleted)
        self.event_bus.subscribe("chat.session_selected", self._handle_session_selected)
        
        self.logger.info("ChatInterface initialized")
    
    def create_session(self, title: Optional[str] = None, model: Optional[str] = None,
                      system_prompt: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.
        
        Args:
            title: Session title (defaults to timestamp)
            model: Model to use (defaults to default model)
            system_prompt: System prompt (defaults to default system prompt)
            
        Returns:
            Created chat session
        """
        # Generate session ID
        session_id = f"session_{int(time.time() * 1000)}"
        
        # Set defaults
        if title is None:
            title = f"Chat {time.strftime('%Y-%m-%d %H:%M')}"
        
        if model is None:
            model = self.default_model
        
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        
        # Create session
        session = ChatSession(session_id, title, model, system_prompt)
        
        # Add system message if system prompt is provided
        if system_prompt:
            system_message = ChatMessage(MessageRole.SYSTEM, system_prompt)
            session.add_message(system_message)
        
        # Store session
        self.sessions[session_id] = session
        
        # Set as current session if no current session
        if self.current_session_id is None:
            self.current_session_id = session_id
        
        # Publish event
        self.event_bus.publish_event("chat.session_created", {
            "session_id": session_id,
            "title": title,
            "model": model
        })
        
        # Save sessions if auto-save is enabled
        if self.auto_save:
            self._save_sessions()
        
        self.logger.info(f"Created chat session: {session_id}")
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if deleted, False if not found
        """
        if session_id not in self.sessions:
            return False
        
        # Remove session
        del self.sessions[session_id]
        
        # Update current session if deleted
        if self.current_session_id == session_id:
            if self.sessions:
                # Set to most recent session
                self.current_session_id = max(
                    self.sessions.keys(),
                    key=lambda sid: self.sessions[sid].updated_at
                )
            else:
                self.current_session_id = None
        
        # Publish event
        self.event_bus.publish_event("chat.session_deleted", {
            "session_id": session_id
        })
        
        # Save sessions if auto-save is enabled
        if self.auto_save:
            self._save_sessions()
        
        self.logger.info(f"Deleted chat session: {session_id}")
        return True
    
    def select_session(self, session_id: str) -> bool:
        """
        Select a chat session.
        
        Args:
            session_id: ID of the session to select
            
        Returns:
            True if selected, False if not found
        """
        if session_id not in self.sessions:
            return False
        
        self.current_session_id = session_id
        
        # Publish event
        self.event_bus.publish_event("chat.session_selected", {
            "session_id": session_id
        })
        
        self.logger.info(f"Selected chat session: {session_id}")
        return True
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[ChatSession]:
        """
        Get a chat session.
        
        Args:
            session_id: ID of the session to get (defaults to current session)
            
        Returns:
            Chat session or None if not found
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            return None
        
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[ChatSession]:
        """
        Get all chat sessions.
        
        Returns:
            List of all chat sessions
        """
        return list(self.sessions.values())
    
    def send_message(self, content: str, session_id: Optional[str] = None) -> ChatMessage:
        """
        Send a user message.
        
        Args:
            content: Message content
            session_id: ID of the session to send to (defaults to current session)
            
        Returns:
            Created message
        """
        # Get session
        session = self.get_session(session_id)
        if session is None:
            # Create new session if none exists
            session = self.create_session()
        
        # Create message
        message = ChatMessage(MessageRole.USER, content)
        
        # Add to session
        session.add_message(message)
        
        # Publish event
        self.event_bus.publish_event("chat.message_sent", {
            "session_id": session.id,
            "message": message.to_dict()
        })
        
        # Save sessions if auto-save is enabled
        if self.auto_save:
            self._save_sessions()
        
        self.logger.debug(f"Sent message in session {session.id}")
        return message
    
    def receive_message(self, content: str, session_id: Optional[str] = None,
                       role: MessageRole = MessageRole.ASSISTANT,
                       metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """
        Receive a message.
        
        Args:
            content: Message content
            session_id: ID of the session to receive in (defaults to current session)
            role: Role of the message sender
            metadata: Additional message metadata
            
        Returns:
            Created message
        """
        # Get session
        session = self.get_session(session_id)
        if session is None:
            # Create new session if none exists
            session = self.create_session()
        
        # Create message
        message = ChatMessage(role, content, metadata=metadata)
        
        # Add to session
        session.add_message(message)
        
        # Publish event
        self.event_bus.publish_event("chat.message_received", {
            "session_id": session.id,
            "message": message.to_dict()
        })
        
        # Save sessions if auto-save is enabled
        if self.auto_save:
            self._save_sessions()
        
        self.logger.debug(f"Received message in session {session.id}")
        return message
    
    def update_message_status(self, message_id: str, status: MessageStatus,
                            session_id: Optional[str] = None) -> bool:
        """
        Update message status.
        
        Args:
            message_id: ID of the message to update
            status: New status
            session_id: ID of the session containing the message (defaults to current session)
            
        Returns:
            True if updated, False if not found
        """
        # Get session
        session = self.get_session(session_id)
        if session is None:
            return False
        
        # Find message
        for message in session.messages:
            if message.id == message_id:
                message.status = status
                
                # Publish event
                self.event_bus.publish_event("chat.message_status_updated", {
                    "session_id": session.id,
                    "message_id": message_id,
                    "status": status.name.lower()
                })
                
                # Save sessions if auto-save is enabled
                if self.auto_save:
                    self._save_sessions()
                
                return True
        
        return False
    
    def clear_session_history(self, session_id: Optional[str] = None) -> bool:
        """
        Clear chat history for a session.
        
        Args:
            session_id: ID of the session to clear (defaults to current session)
            
        Returns:
            True if cleared, False if not found
        """
        # Get session
        session = self.get_session(session_id)
        if session is None:
            return False
        
        # Keep system message if present
        system_messages = [msg for msg in session.messages if msg.role == MessageRole.SYSTEM]
        session.messages = system_messages
        
        # Publish event
        self.event_bus.publish_event("chat.session_cleared", {
            "session_id": session.id
        })
        
        # Save sessions if auto-save is enabled
        if self.auto_save:
            self._save_sessions()
        
        self.logger.info(f"Cleared history for session {session.id}")
        return True
    
    def _save_sessions(self) -> None:
        """Save chat sessions to storage."""
        try:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus', 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Save sessions
            sessions_file = os.path.join(data_dir, 'chat_sessions.json')
            
            # Convert sessions to dict
            sessions_data = {
                "sessions": {sid: session.to_dict() for sid, session in self.sessions.items()},
                "current_session_id": self.current_session_id
            }
            
            # Write to file
            with open(sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
            
            self.logger.debug("Saved chat sessions")
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.STORAGE, ErrorSeverity.WARNING,
                {"action": "save_chat_sessions"}
            )
    
    def _load_sessions(self) -> None:
        """Load chat sessions from storage."""
        try:
            # Check if sessions file exists
            data_dir = os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus', 'data')
            sessions_file = os.path.join(data_dir, 'chat_sessions.json')
            
            if not os.path.exists(sessions_file):
                return
            
            # Read sessions
            with open(sessions_file, 'r') as f:
                sessions_data = json.load(f)
            
            # Convert to session objects
            self.sessions = {
                sid: ChatSession.from_dict(session_data)
                for sid, session_data in sessions_data.get("sessions", {}).items()
            }
            
            # Set current session
            self.current_session_id = sessions_data.get("current_session_id")
            
            # Create new session if none loaded
            if not self.sessions:
                self.create_session()
            
            self.logger.info(f"Loaded {len(self.sessions)} chat sessions")
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.STORAGE, ErrorSeverity.WARNING,
                {"action": "load_chat_sessions"}
            )
            
            # Create default session
            self.create_session()
    
    def _handle_message_sent(self, event: Dict[str, Any]) -> None:
        """
        Handle message sent event.
        
        Args:
            event: Event data
        """
        session_id = event.get("session_id")
        message_data = event.get("message")
        
        if session_id and message_data:
            self.logger.debug(f"Message sent in session {session_id}")
    
    def _handle_message_received(self, event: Dict[str, Any]) -> None:
        """
        Handle message received event.
        
        Args:
            event: Event data
        """
        session_id = event.get("session_id")
        message_data = event.get("message")
        
        if session_id and message_data:
            self.logger.debug(f"Message received in session {session_id}")
    
    def _handle_session_created(self, event: Dict[str, Any]) -> None:
        """
        Handle session created event.
        
        Args:
            event: Event data
        """
        session_id = event.get("session_id")
        
        if session_id:
            self.logger.debug(f"Session created: {session_id}")
    
    def _handle_session_deleted(self, event: Dict[str, Any]) -> None:
        """
        Handle session deleted event.
        
        Args:
            event: Event data
        """
        session_id = event.get("session_id")
        
        if session_id:
            self.logger.debug(f"Session deleted: {session_id}")
    
    def _handle_session_selected(self, event: Dict[str, Any]) -> None:
        """
        Handle session selected event.
        
        Args:
            event: Event data
        """
        session_id = event.get("session_id")
        
        if session_id:
            self.logger.debug(f"Session selected: {session_id}")
    
    def startup(self) -> None:
        """Start the chat interface."""
        # Load saved sessions
        self._load_sessions()
        
        self.logger.info("ChatInterface started")
    
    def shutdown(self) -> None:
        """Stop the chat interface."""
        # Save sessions
        self._save_sessions()
        
        self.logger.info("ChatInterface stopped")

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    error_handler = ErrorHandler.get_instance()
    
    config_manager = ConfigurationManager.get_instance()
    
    ui_manager = UIManager.get_instance()
    ui_manager.startup()
    
    # Initialize ChatInterface
    chat_interface = ChatInterface.get_instance()
    chat_interface.startup()
    
    # Example usage
    session = chat_interface.create_session("Test Chat", "gpt-3.5-turbo")
    chat_interface.send_message("Hello, how are you?")
    chat_interface.receive_message("I'm doing well, thank you for asking!")
    
    # Shutdown
    chat_interface.shutdown()
    ui_manager.shutdown()
    event_bus.shutdown()
