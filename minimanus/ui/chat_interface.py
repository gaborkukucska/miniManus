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
import threading # Needed for locks
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
        self.logger.info("ChatInterface startup: STARTING")
        if self.sessions_dir is None:
             self.logger.error("Session directory not set before startup. Sessions will not be loaded/saved.")
        else:
             self.logger.info(f"ChatInterface startup: Ensuring sessions dir exists: {self.sessions_dir}")
             try:
                 self.sessions_dir.mkdir(parents=True, exist_ok=True)
             except OSError as e:
                  self.logger.error(f"ChatInterface startup: Failed to create sessions directory {self.sessions_dir}: {e}", exc_info=True)
                  self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"action": "ensure_sessions_dir"})
                  # Decide if we should continue without session saving or exit
                  self.logger.warning("ChatInterface startup: Continuing without session persistence due to directory error.")
                  self.sessions_dir = None # Disable saving/loading if dir fails
                  self.auto_save = False
             else:
                 self.logger.info("ChatInterface startup: Loading sessions...")
                 self._load_sessions()
                 self.logger.info("ChatInterface startup: Sessions loaded.")

        # Create default session if none exists after loading
        self.logger.info("ChatInterface startup: Checking if session needs creation...")
        if not self.sessions:
            self.logger.info("No existing sessions found or loaded. Creating a new default session.")
            try:
                # Wrap session creation in try/except during startup
                default_session = self.create_session("New Chat")
                if default_session:
                    self.current_session_id = default_session.id
                    self.logger.info(f"ChatInterface startup: Default session created: {self.current_session_id}")
                else:
                    self.logger.error("ChatInterface startup: Failed to create default session.")
            except Exception as e:
                self.logger.critical("ChatInterface startup: CRITICAL ERROR during default session creation.", exc_info=True)
                # Depending on severity, might need to exit
                # sys.exit(1)
        elif not self.current_session_id or self.current_session_id not in self.sessions:
            self.logger.info("ChatInterface startup: Finding most recent session...")
            most_recent_id = max(self.sessions, key=lambda sid: self.sessions[sid].updated_at, default=None)
            if most_recent_id:
                 self.current_session_id = most_recent_id
                 self.logger.info(f"Set current session to most recent: {self.current_session_id}")
                 self._save_current_session_id() # Save the determined current session ID
            else:
                 self.logger.warning("No valid sessions found after load. Creating a new default session.")
                 try:
                     default_session = self.create_session("New Chat")
                     if default_session:
                         self.current_session_id = default_session.id
                         self.logger.warning(f"ChatInterface startup: Created new default session as fallback: {self.current_session_id}")
                     else:
                         self.logger.error("ChatInterface startup: Failed to create fallback default session.")
                 except Exception as e:
                     self.logger.critical("ChatInterface startup: CRITICAL ERROR during fallback session creation.", exc_info=True)

        self.logger.info(f"ChatInterface started. Current session: {self.current_session_id}")
        self.logger.info("ChatInterface startup: FINISHED")


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
            self.logger.debug(f"ChatInterface _load_sessions: Scanning {self.sessions_dir}...")
            for file_path in self.sessions_dir.glob("*.json"):
                self.logger.debug(f"ChatInterface _load_sessions: Attempting to load {file_path.name}")
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        session_data = json.load(f)
                        session = ChatSession.from_dict(session_data)
                        if session and session.id == file_path.stem: # Check if session creation succeeded
                             self.sessions[session.id] = session
                             loaded_count += 1
                        elif session: # ID mismatch
                             self.logger.warning(f"Session ID mismatch in file {file_path.name}. Expected {file_path.stem}, got {session.id}. Skipping.")
                        # else: from_dict already logged the error
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error decoding JSON from session file {file_path.name}: {e}")
                    self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"file": str(file_path), "action": "load_session"})
                except Exception as e:
                    self.logger.error(f"Error loading session file {file_path.name}: {e}", exc_info=True)
                    self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"file": str(file_path), "action": "load_session"})

            # Load the last used session ID if it exists
            self._load_current_session_id()

        self.logger.info(f"Loaded {loaded_count} chat sessions from {self.sessions_dir}")

    def _load_current_session_id(self) -> None:
        """Loads the current session ID from .current_session file."""
        if not self.sessions_dir: return

        current_session_file = self.sessions_dir / ".current_session"
        if current_session_file.exists():
             self.logger.debug(f"ChatInterface: Loading current session ID from {current_session_file}")
             try:
                 last_id = current_session_file.read_text(encoding="utf-8").strip()
                 if last_id in self.sessions:
                     self.current_session_id = last_id
                     self.logger.debug(f"ChatInterface: Current session set to {last_id}")
                 else:
                     self.logger.warning(f"ChatInterface: Saved current session ID '{last_id}' not found in loaded sessions.")
             except Exception as e:
                  self.logger.error(f"Error reading current session file: {e}", exc_info=True)


    def _save_current_session_id(self) -> bool:
        """Saves the current session ID to the .current_session file."""
        if not self.sessions_dir:
            self.logger.warning("Cannot save current session ID: sessions_dir not set.")
            return False
        if not self.current_session_id:
             self.logger.warning("Cannot save current session ID: current_session_id is None.")
             return False

        current_session_file = self.sessions_dir / ".current_session"
        try:
            current_session_file.write_text(self.current_session_id, encoding="utf-8")
            self.logger.debug(f"Saved current session ID: {self.current_session_id} to {current_session_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving current session ID to {current_session_file}: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"file": str(current_session_file), "action": "save_current_id"})
            return False


    def _save_session(self, session_id: str) -> bool:
        """
        Save a single chat session to disk. Returns True on success, False on failure.
        """
        self.logger.debug(f"ChatInterface _save_session: Saving session {session_id}...")
        if not self.sessions_dir:
            self.logger.warning("Session directory not set. Cannot save session.")
            return False

        session_to_save = None # Define outside the lock
        with self._sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                self.logger.warning(f"Attempted to save non-existent session: {session_id}")
                return False
            # Create a copy of the data to save outside the lock if needed
            # For simplicity, convert to dict inside the lock for now.
            try:
                session_dict = session.to_dict()
                session_to_save = session_dict # Assign the dict to save
            except Exception as e:
                 self.logger.error(f"Error converting session {session_id} to dict: {e}", exc_info=True)
                 return False # Don't proceed if conversion fails


        # Perform file I/O outside the lock if possible, using the prepared dict
        if session_to_save:
            file_path = self.sessions_dir / f"{session_id}.json"
            temp_file_path = file_path.with_suffix(".tmp")
            try:
                self.logger.debug(f"ChatInterface _save_session: Writing session {session_id} to temp file {temp_file_path}...")
                with temp_file_path.open("w", encoding="utf-8") as f:
                    json.dump(session_to_save, f, indent=2, ensure_ascii=False)

                # Atomic rename
                os.replace(temp_file_path, file_path)
                self.logger.debug(f"Saved session {session_id} to {file_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error saving session {session_id} to {file_path}: {e}", exc_info=True)
                self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"file": str(file_path), "action": "save_session"})
                # Clean up temp file if rename failed
                if temp_file_path.exists():
                    try: temp_file_path.unlink()
                    except OSError: pass
                return False
        else:
             # This case means converting to dict failed inside the lock
             return False


    def _save_all_sessions(self) -> None:
         """Saves all current sessions and the current session ID."""
         if not self.sessions_dir:
             self.logger.warning("Session directory not set. Cannot save sessions.")
             return

         self.logger.info("Saving all chat sessions...")
         session_ids_to_save = []
         with self._sessions_lock:
             session_ids_to_save = list(self.sessions.keys()) # Get keys inside lock

         saved_count = 0
         failed_count = 0
         for session_id in session_ids_to_save:
             if self._save_session(session_id): # _save_session now returns bool
                 saved_count += 1
             else:
                  failed_count += 1

         self.logger.info(f"Finished saving sessions. Success: {saved_count}, Failed: {failed_count}")

         # Save the current session ID
         self._save_current_session_id()

    def create_session(self, title: str = "New Chat", system_prompt: Optional[str] = None) -> Optional[ChatSession]:
        """
        Create a new chat session. Returns the session on success, None on failure.
        """
        self.logger.debug(f"ChatInterface create_session: STARTING creation for title '{title}'")
        session_saved = False
        session = None # Initialize session variable

        try:
            with self._sessions_lock:
                session = ChatSession(title=title, system_prompt=system_prompt)
                self.sessions[session.id] = session
                self.logger.info(f"Created new chat session object: {session.id} ('{title}')")

                # Set as current immediately if needed (inside lock for consistency)
                if self.current_session_id is None:
                     self.current_session_id = session.id
                     self.logger.debug(f"ChatInterface create_session: Set new session {session.id} as current (in memory).")

            # Perform saving operations outside the main lock if possible
            if self.auto_save:
                self.logger.debug(f"ChatInterface create_session: Auto-saving session {session.id}...")
                session_saved = self._save_session(session.id)
                if not session_saved:
                     self.logger.error(f"ChatInterface create_session: Failed to auto-save session {session.id}.")
                     # Decide if we should rollback the creation? For now, log error and continue.
                else:
                     self.logger.debug(f"ChatInterface create_session: Auto-save complete for {session.id}.")

            # Save current session ID if it was just set
            if self.current_session_id == session.id:
                self.logger.debug(f"ChatInterface create_session: Saving current session ID {session.id}...")
                if not self._save_current_session_id():
                     self.logger.error(f"ChatInterface create_session: Failed to save current session ID {session.id}.")

            self.logger.debug(f"ChatInterface create_session: Publishing event for {session.id}...")
            self.event_bus.publish_event("chat.session.created", {"session_id": session.id})
            self.logger.debug(f"ChatInterface create_session: FINISHED creation for {session.id}. Save success: {session_saved}")
            return session

        except Exception as e:
             self.logger.error(f"ChatInterface create_session: CRITICAL ERROR during session creation for title '{title}'.", exc_info=True)
             self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL, {"action": "create_session"})
             # Attempt to clean up partially created session if it exists in memory
             if session and session.id in self.sessions:
                 with self._sessions_lock:
                     if session.id in self.sessions:
                         del self.sessions[session.id]
             return None # Indicate failure

    # --- Rest of the methods (get_session, get_all_sessions, delete_session, etc.) ---

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        with self._sessions_lock:
            return self.sessions.get(session_id)

    def get_all_sessions(self) -> List[ChatSession]:
        """Get all chat sessions, sorted by last updated time (newest first)."""
        with self._sessions_lock:
            return sorted(list(self.sessions.values()), key=lambda s: s.updated_at, reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        session_file_deleted = False
        with self._sessions_lock:
            if session_id not in self.sessions:
                self.logger.warning(f"Attempted to delete non-existent session: {session_id}")
                return False

            del self.sessions[session_id]
            self.logger.info(f"Deleted chat session from memory: {session_id}")

            # Delete session file (perform file I/O outside lock if safe, but might be simpler here)
            if self.sessions_dir:
                file_path = self.sessions_dir / f"{session_id}.json"
                try:
                    file_path.unlink(missing_ok=True)
                    session_file_deleted = True
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
                    self._save_current_session_id() # Save the new current session ID
                else:
                    self.logger.info("No sessions left after deletion.")

            self.event_bus.publish_event("chat.session.deleted", {"session_id": session_id})
            return True # Return True even if file deletion failed, as memory object is gone


    def set_current_session(self, session_id: str) -> bool:
        """Set the current chat session."""
        session_exists = False
        with self._sessions_lock:
            session_exists = session_id in self.sessions

        if not session_exists:
            self.logger.warning(f"Attempted to set current session to non-existent ID: {session_id}")
            return False

        if self.current_session_id != session_id:
             self.current_session_id = session_id
             self.logger.info(f"Current session set to: {session_id}")
             self.event_bus.publish_event("chat.session.selected", {"session_id": session_id})
             # Save the new current session ID
             if not self._save_current_session_id():
                 self.logger.error("Failed to save updated current session ID.")
        return True

    async def process_message(self, message_text: str, session_id: Optional[str] = None) -> str:
        """Process a user message using the AgentSystem."""
        target_session_id = session_id if session_id is not None else self.current_session_id

        session = None # Initialize session
        if target_session_id:
             session = self.get_session(target_session_id) # Use thread-safe getter

        # Ensure a session exists
        if session is None:
            self.logger.warning(f"No valid session ID ({target_session_id}). Creating new session.")
            try:
                session = self.create_session() # This returns session or None
                if not session:
                     # Handle critical failure during session creation
                     return "Error: Could not create a new chat session."
                target_session_id = session.id
                self.set_current_session(target_session_id) # Set the new one as current
            except Exception as e:
                 self.logger.error(f"Failed to create session in process_message: {e}", exc_info=True)
                 return "Error: Failed to establish a chat session."

        # Add user message to the session
        user_message = ChatMessage(role=MessageRole.USER, content=message_text)
        # Add message requires lock internally if modifying session state directly,
        # but better to work on a copy or pass data if high concurrency needed.
        # For now, assume session.add_message is simple append + timestamp update.
        session.add_message(user_message) # Assume this is safe for now
        self.event_bus.publish_event("chat.message.added", {"session_id": session.id, "message": user_message.to_dict()})

        if self.auto_save:
            if not self._save_session(session.id):
                 self.logger.warning(f"Failed to auto-save session {session.id} after adding user message.")


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
            session.add_message(assistant_message) # Assume safe for now
            self.event_bus.publish_event("chat.message.added", {"session_id": session.id, "message": assistant_message.to_dict()})

            if self.auto_save:
                 if not self._save_session(session.id):
                     self.logger.warning(f"Failed to auto-save session {session.id} after adding assistant message.")

            self.logger.info(f"Generated response for session {session.id}: '{response_text[:50]}...'")
            return response_text

        except Exception as e:
            self.logger.error(f"Error processing message in session {session.id}: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.ERROR, {"session_id": session.id, "action": "process_message"})

            error_response = f"I encountered an error while processing your request. Please try again later. (Error: {type(e).__name__})"
            # Add error message as assistant response
            error_message = ChatMessage(role=MessageRole.ASSISTANT, content=error_response)
            session.add_message(error_message) # Assume safe for now
            self.event_bus.publish_event("chat.message.added", {"session_id": session.id, "message": error_message.to_dict()})

            if self.auto_save:
                 if not self._save_session(session.id):
                     self.logger.warning(f"Failed to auto-save session {session.id} after adding error message.")

            return error_response


    def clear_session_history(self, session_id: Optional[str] = None) -> bool:
        """Clear chat history for a session, keeping the system prompt."""
        target_session_id = session_id if session_id is not None else self.current_session_id
        if not target_session_id:
             self.logger.warning("Cannot clear history: No session specified or current.")
             return False

        session_cleared = False
        with self._sessions_lock:
            session = self.sessions.get(target_session_id)
            if session is None:
                 self.logger.warning(f"Cannot clear history: Session not found: {target_session_id}")
                 return False

            # Keep only the system message(s) if they exist
            original_messages = session.messages
            session.messages = [msg for msg in original_messages if msg.role == MessageRole.SYSTEM]
            session.updated_at = time.time() # Update timestamp
            session_cleared = True

        if session_cleared:
            # Publish event
            self.event_bus.publish_event("chat.session.cleared", {"session_id": target_session_id})

            # Save sessions if auto-save is enabled
            if self.auto_save:
                if not self._save_session(target_session_id):
                     self.logger.warning(f"Failed to save session {target_session_id} after clearing history.")


            self.logger.info(f"Cleared history for session {target_session_id}")
            return True
        else:
            return False # Should not happen if lock logic is correct
# END OF FILE miniManus-main/minimanus/ui/chat_interface.py
