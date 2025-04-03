# START OF FILE miniManus-main/minimanus/ui/chat_interface.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat Interface for miniManus

This module implements the Chat Interface component, which manages chat sessions
and interactions with the user, including routing requests to the AgentSystem.
"""

import os
import sys
import json
import logging
import uuid
import time
import asyncio
import threading # <<<--- LINE ADDED HERE
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
    # No direct API Manager needed here, AgentSystem will handle it
    # from ..api.api_manager import APIManager, APIProvider, APIRequestType
    from ..core.agent_system import AgentSystem # Important: Need AgentSystem
except ImportError as e:
    logging.getLogger("miniManus.ChatInterface").critical(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(f"ImportError in chat_interface.py: {e}. Ensure all components exist.")

logger = logging.getLogger("miniManus.ChatInterface")

class MessageRole(Enum):
    """Message roles in a chat session."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool" # Added for potential agent tool interactions

@dataclass
class ChatMessage:
    """A message in a chat session."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Add optional fields for tool calls if needed later
    # tool_calls: Optional[List[Dict]] = None
    # tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary.

        Returns:
            Session as dictionary
        """
        data = {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp
        }
        # Add tool fields if present
        # if hasattr(self, 'tool_calls') and self.tool_calls:
        #     data['tool_calls'] = self.tool_calls
        # if hasattr(self, 'tool_call_id') and self.tool_call_id:
        #     data['tool_call_id'] = self.tool_call_id
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """
        Create session from dictionary.

        Args:
            data: Session data

        Returns:
            Chat session
        """
        try:
            role = MessageRole(data.get("role", "user"))
        except ValueError:
            role = MessageRole.USER # Default to USER if role is invalid

        message = cls(
            id=data.get("id", str(uuid.uuid4())),
            role=role,
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time())
        )
        # Add tool fields if present in data
        # message.tool_calls = data.get('tool_calls')
        # message.tool_call_id = data.get('tool_call_id')
        return message

@dataclass
class ChatSession:
    """A chat session with a history of messages."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Chat"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[ChatMessage] = field(default_factory=list)
    system_prompt: Optional[str] = None
    # Add other session metadata if needed, e.g., selected model
    # selected_model: Optional[str] = None

    def add_message(self, message: ChatMessage) -> None:
        """
        Add a message to the session.

        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.updated_at = time.time()

    def get_history_for_llm(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Formats message history for sending to an LLM (OpenAI format).

        Args:
            max_messages: Maximum number of recent messages to include.

        Returns:
            List of message dictionaries.
        """
        history = []
        if self.system_prompt:
            history.append({"role": "system", "content": self.system_prompt})

        messages_to_include = self.messages
        if max_messages is not None and len(messages_to_include) > max_messages:
             # Take the last N messages, but preserve the first system message if any
             system_msg_present = self.messages and self.messages[0].role == MessageRole.SYSTEM
             if system_msg_present:
                 messages_to_include = [self.messages[0]] + self.messages[-(max_messages-1):]
             else:
                  messages_to_include = self.messages[-max_messages:]


        for msg in messages_to_include:
            # Convert ChatMessage to the dict format expected by LLMs
            # Skip system message if already added via system_prompt attribute
            if msg.role == MessageRole.SYSTEM and self.system_prompt:
                continue
            msg_dict = {"role": msg.role.value, "content": msg.content}
            # Add tool fields if they exist (for future agent use)
            # if hasattr(msg, 'tool_calls') and msg.tool_calls:
            #     msg_dict['tool_calls'] = msg.tool_calls
            # if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
            #     msg_dict['tool_call_id'] = msg.tool_call_id
            history.append(msg_dict)
        return history


    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary for saving.

        Returns:
            Session as dictionary
        """
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [msg.to_dict() for msg in self.messages],
            "system_prompt": self.system_prompt,
            # "selected_model": self.selected_model # Save model if needed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """
        Create session from dictionary loaded from storage.

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
            # selected_model=data.get("selected_model") # Load model if saved
        )

        session.messages = [ChatMessage.from_dict(msg_data) for msg_data in data.get("messages", [])]

        return session

class ChatInterface:
    """
    ChatInterface manages chat sessions and interactions with the user.

    It handles:
    - Chat session creation and management
    - Message processing and routing via the AgentSystem
    - Persistence of chat sessions
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
        self.agent_system = AgentSystem.get_instance() # Get the AgentSystem instance

        # Chat sessions
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        self._sessions_lock = threading.Lock() # Lock for session operations

        # Session storage directory - needs to be set by caller (e.g., __main__.py)
        self.sessions_dir: Optional[Path] = None

        # Settings read from config manager
        self.max_history_for_llm = self.config_manager.get_config("chat.max_history_for_llm", 20) # Number of messages to send to LLM
        self.auto_save = self.config_manager.get_config("chat.auto_save", True)

        self.logger.info("ChatInterface initialized")

    def startup(self) -> None:
        """Start the chat interface."""
        self.logger.info("ChatInterface startup: STARTING") # ADDED LOG
        if self.sessions_dir is None:
             self.logger.error("Session directory not set before startup. Sessions will not be loaded/saved.")
             # Optionally set a default path here, but it's better if set explicitly
             # self.sessions_dir = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'minimanus' / 'data' / 'sessions'
        else:
             self.logger.info(f"ChatInterface startup: Ensuring sessions dir exists: {self.sessions_dir}") # ADDED LOG
             self.sessions_dir.mkdir(parents=True, exist_ok=True)
             self.logger.info("ChatInterface startup: Loading sessions...") # ADDED LOG
             self._load_sessions() # Load sessions after directory is ensured
             self.logger.info("ChatInterface startup: Sessions loaded.") # ADDED LOG

        # Create default session if none exists after loading
        self.logger.info("ChatInterface startup: Checking if session needs creation...") # ADDED LOG
        if not self.sessions:
            self.logger.info("No existing sessions found or loaded. Creating a new default session.")
            default_session = self.create_session("New Chat")
            self.current_session_id = default_session.id
            self.logger.info(f"ChatInterface startup: Default session created: {self.current_session_id}") # ADDED LOG
        elif not self.current_session_id or self.current_session_id not in self.sessions:
            # Set current session to most recently updated if current ID is invalid
            self.logger.info("ChatInterface startup: Finding most recent session...") # ADDED LOG
            most_recent_id = max(self.sessions, key=lambda sid: self.sessions[sid].updated_at, default=None)
            if most_recent_id:
                 self.current_session_id = most_recent_id
                 self.logger.info(f"Set current session to most recent: {self.current_session_id}")
            else:
                 # If somehow sessions exist but none are valid, create a new one
                 self.logger.warning("No valid sessions found after load. Creating a new default session.")
                 default_session = self.create_session("New Chat")
                 self.current_session_id = default_session.id
                 self.logger.warning(f"ChatInterface startup: Created new default session as fallback: {self.current_session_id}") # ADDED LOG

        self.logger.info(f"ChatInterface started. Current session: {self.current_session_id}")
        self.logger.info("ChatInterface startup: FINISHED") # ADDED LOG


    def shutdown(self) -> None:
        """Shut down the chat interface."""
        if self.auto_save:
             self._save_all_sessions()
        self.logger.info("ChatInterface shut down")

    def _load_sessions(self) -> None:
        """Load chat sessions from disk."""
        if not self.sessions_dir or not self.sessions_dir.exists():
            self.logger.warning(f"Session directory not available for loading: {self.sessions_dir}")
            return

        loaded_count = 0
        with self._sessions_lock:
            self.sessions.clear() # Clear existing in-memory sessions before loading
            self.logger.debug(f"ChatInterface _load_sessions: Scanning {self.sessions_dir}...") # ADDED LOG
            for file_path in self.sessions_dir.glob("*.json"):
                self.logger.debug(f"ChatInterface _load_sessions: Attempting to load {file_path.name}") # ADDED LOG
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        session_data = json.load(f)
                        session = ChatSession.from_dict(session_data)
                        if session.id == file_path.stem: # Basic sanity check
                             self.sessions[session.id] = session
                             loaded_count += 1
                        else:
                             self.logger.warning(f"Session ID mismatch in file {file_path.name}. Expected {file_path.stem}, got {session.id}. Skipping.")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error decoding JSON from session file {file_path.name}: {e}")
                    self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"file": str(file_path), "action": "load_session"})
                except Exception as e:
                    self.logger.error(f"Error loading session file {file_path.name}: {e}", exc_info=True)
                    self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"file": str(file_path), "action": "load_session"})

            # Load the last used session ID if it exists
            current_session_file = self.sessions_dir / ".current_session"
            if current_session_file.exists():
                 self.logger.debug(f"ChatInterface _load_sessions: Loading current session ID from {current_session_file}") # ADDED LOG
                 try:
                     last_id = current_session_file.read_text(encoding="utf-8").strip()
                     if last_id in self.sessions:
                         self.current_session_id = last_id
                         self.logger.debug(f"ChatInterface _load_sessions: Current session set to {last_id}") # ADDED LOG
                     else:
                         self.logger.warning(f"ChatInterface _load_sessions: Saved current session ID '{last_id}' not found in loaded sessions.") # ADDED LOG
                 except Exception as e:
                      self.logger.error(f"Error reading current session file: {e}")

        self.logger.info(f"Loaded {loaded_count} chat sessions from {self.sessions_dir}")


    def _save_session(self, session_id: str) -> None:
        """
        Save a single chat session to disk.

        Args:
            session_id: ID of session to save
        """
        self.logger.debug(f"ChatInterface _save_session: Saving session {session_id}...") # ADDED LOG
        if not self.sessions_dir:
            self.logger.warning("Session directory not set. Cannot save session.")
            return

        with self._sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                self.logger.warning(f"Attempted to save non-existent session: {session_id}")
                return

            file_path = self.sessions_dir / f"{session_id}.json"
            try:
                self.logger.debug(f"ChatInterface _save_session: Converting session {session_id} to dict...") # ADDED LOG
                session_dict = session.to_dict()
                self.logger.debug(f"ChatInterface _save_session: Writing session {session_id} to {file_path}...") # ADDED LOG
                with file_path.open("w", encoding="utf-8") as f:
                    json.dump(session_dict, f, indent=2, ensure_ascii=False)
                self.logger.debug(f"Saved session {session_id} to {file_path}")
            except Exception as e:
                self.logger.error(f"Error saving session {session_id} to {file_path}: {e}", exc_info=True)
                self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"file": str(file_path), "action": "save_session"})
        self.logger.debug(f"ChatInterface _save_session: Finished saving session {session_id}.") # ADDED LOG


    def _save_all_sessions(self) -> None:
         """Saves all current sessions and the current session ID."""
         if not self.sessions_dir:
             self.logger.warning("Session directory not set. Cannot save sessions.")
             return

         self.logger.info("Saving all chat sessions...")
         with self._sessions_lock:
             session_ids = list(self.sessions.keys()) # Get keys before iterating

         for session_id in session_ids:
             self._save_session(session_id) # _save_session handles its own locking

         # Save the current session ID
         if self.current_session_id:
              current_session_file = self.sessions_dir / ".current_session"
              try:
                  current_session_file.write_text(self.current_session_id, encoding="utf-8")
                  self.logger.debug(f"Saved current session ID: {self.current_session_id}")
              except Exception as e:
                   self.logger.error(f"Error saving current session ID: {e}")

    def create_session(self, title: str = "New Chat", system_prompt: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.

        Args:
            title: Session title
            system_prompt: System prompt for the session

        Returns:
            New chat session
        """
        self.logger.debug(f"ChatInterface create_session: STARTING creation for title '{title}'") # ADDED LOG
        with self._sessions_lock:
            session = ChatSession(title=title, system_prompt=system_prompt)
            self.sessions[session.id] = session
            self.logger.info(f"Created new chat session: {session.id} ('{title}')")

            # Optionally set as current if none is set
            if self.current_session_id is None:
                 self.logger.debug(f"ChatInterface create_session: Setting new session {session.id} as current.") # ADDED LOG
                 self.set_current_session(session.id)

            if self.auto_save:
                self.logger.debug(f"ChatInterface create_session: Auto-saving session {session.id}...") # ADDED LOG
                self._save_session(session.id)
                self.logger.debug(f"ChatInterface create_session: Auto-save complete for {session.id}.") # ADDED LOG

            self.logger.debug(f"ChatInterface create_session: Publishing event for {session.id}...") # ADDED LOG
            self.event_bus.publish_event("chat.session.created", {"session_id": session.id})
            self.logger.debug(f"ChatInterface create_session: FINISHED creation for {session.id}.") # ADDED LOG
            return session

    # --- Rest of the methods (get_session, get_all_sessions, delete_session, etc.) remain the same ---

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by ID.

        Args:
            session_id: Session ID

        Returns:
            Chat session or None if not found
        """
        with self._sessions_lock:
            return self.sessions.get(session_id)

    def get_all_sessions(self) -> List[ChatSession]:
        """
        Get all chat sessions, sorted by last updated time (newest first).

        Returns:
            List of all chat sessions
        """
        with self._sessions_lock:
            # Sort sessions by updated_at timestamp, descending
            return sorted(list(self.sessions.values()), key=lambda s: s.updated_at, reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        with self._sessions_lock:
            if session_id not in self.sessions:
                self.logger.warning(f"Attempted to delete non-existent session: {session_id}")
                return False

            del self.sessions[session_id]
            self.logger.info(f"Deleted chat session: {session_id}")

            # Delete session file
            if self.sessions_dir:
                file_path = self.sessions_dir / f"{session_id}.json"
                try:
                    file_path.unlink(missing_ok=True) # Delete file if it exists
                    self.logger.debug(f"Deleted session file: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error deleting session file {file_path}: {e}", exc_info=True)
                    self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.WARNING, {"file": str(file_path), "action": "delete_session_file"})

            # Update current session if needed
            if self.current_session_id == session_id:
                most_recent_id = max(self.sessions, key=lambda sid: self.sessions[sid].updated_at, default=None)
                self.current_session_id = most_recent_id
                if self.current_session_id:
                    self.logger.info(f"Current session changed to {self.current_session_id} after deletion.")
                    if self.auto_save and self.sessions_dir: # Save the new current session ID
                        current_session_file = self.sessions_dir / ".current_session"
                        try:
                             current_session_file.write_text(self.current_session_id, encoding="utf-8")
                        except Exception as e:
                             self.logger.error(f"Error saving new current session ID after deletion: {e}")
                else:
                    self.logger.info("No sessions left after deletion.")
                    # Optionally create a new default session here if desired

            self.event_bus.publish_event("chat.session.deleted", {"session_id": session_id})
            return True


    def set_current_session(self, session_id: str) -> bool:
        """
        Set the current chat session.

        Args:
            session_id: Session ID

        Returns:
            True if set, False if not found
        """
        with self._sessions_lock:
            if session_id not in self.sessions:
                self.logger.warning(f"Attempted to set current session to non-existent ID: {session_id}")
                return False

            if self.current_session_id != session_id:
                 self.current_session_id = session_id
                 self.logger.info(f"Current session set to: {session_id}")
                 self.event_bus.publish_event("chat.session.selected", {"session_id": session_id})
                 # Save the new current session ID if auto-saving
                 if self.auto_save and self.sessions_dir:
                     current_session_file = self.sessions_dir / ".current_session"
                     try:
                         current_session_file.write_text(self.current_session_id, encoding="utf-8")
                     except Exception as e:
                          self.logger.error(f"Error saving current session ID: {e}")
            return True

    async def process_message(self, message_text: str, session_id: Optional[str] = None) -> str:
        """
        Process a user message using the AgentSystem.

        Args:
            message_text: User message content.
            session_id: Optional session ID (uses current if None).

        Returns:
            Assistant's response content.
        """
        target_session_id = session_id if session_id is not None else self.current_session_id

        # Ensure a session exists
        if target_session_id is None or target_session_id not in self.sessions:
            self.logger.warning(f"No valid session ID provided or found ({target_session_id}). Creating new session.")
            session = self.create_session()
            target_session_id = session.id
            self.set_current_session(target_session_id) # Set the new one as current
        else:
            session = self.sessions[target_session_id]

        # Add user message to the session
        user_message = ChatMessage(role=MessageRole.USER, content=message_text)
        session.add_message(user_message)
        self.event_bus.publish_event("chat.message.added", {"session_id": session.id, "message": user_message.to_dict()})

        if self.auto_save:
            self._save_session(session.id)

        self.logger.info(f"Processing message for session {session.id}: '{message_text[:50]}...'")

        try:
            # Prepare history for the agent system
            history = session.get_history_for_llm(max_messages=self.max_history_for_llm)

            # Call the agent system to get the response
            response_text = await self.agent_system.process_user_request(
                user_message=message_text, # Pass only the latest user message here
                conversation_history=history[:-1] # Pass history *excluding* the latest user message
            )

            # Add assistant's response to the session
            assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
            session.add_message(assistant_message)
            self.event_bus.publish_event("chat.message.added", {"session_id": session.id, "message": assistant_message.to_dict()})

            if self.auto_save:
                self._save_session(session.id)

            self.logger.info(f"Generated response for session {session.id}: '{response_text[:50]}...'")
            return response_text

        except Exception as e:
            self.logger.error(f"Error processing message in session {session.id}: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.ERROR, {"session_id": session.id, "action": "process_message"})

            error_response = f"I encountered an error while processing your request. Please try again later. (Error: {type(e).__name__})"
            # Add error message as assistant response
            error_message = ChatMessage(role=MessageRole.ASSISTANT, content=error_response)
            session.add_message(error_message)
            self.event_bus.publish_event("chat.message.added", {"session_id": session.id, "message": error_message.to_dict()})

            if self.auto_save:
                self._save_session(session.id)

            return error_response


    def clear_session_history(self, session_id: Optional[str] = None) -> bool:
        """
        Clear chat history for a session, keeping the system prompt.

        Args:
            session_id: ID of the session to clear (defaults to current session)

        Returns:
            True if cleared, False if not found
        """
        target_session_id = session_id if session_id is not None else self.current_session_id
        if not target_session_id:
             self.logger.warning("Cannot clear history: No session specified or current.")
             return False

        with self._sessions_lock:
            session = self.sessions.get(target_session_id)
            if session is None:
                 self.logger.warning(f"Cannot clear history: Session not found: {target_session_id}")
                 return False

            # Keep only the system message(s) if they exist
            original_messages = session.messages
            session.messages = [msg for msg in original_messages if msg.role == MessageRole.SYSTEM]
            session.updated_at = time.time() # Update timestamp

            # Publish event
            self.event_bus.publish_event("chat.session.cleared", {"session_id": session.id})

            # Save sessions if auto-save is enabled
            if self.auto_save:
                self._save_session(session.id)

            self.logger.info(f"Cleared history for session {session.id}")
            return True
# END OF FILE miniManus-main/minimanus/ui/chat_interface.py
