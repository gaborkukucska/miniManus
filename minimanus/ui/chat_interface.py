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
    # Optional fields for tool calls
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp
        }
        if self.tool_calls:
            data['tool_calls'] = self.tool_calls
        if self.tool_call_id:
            data['tool_call_id'] = self.tool_call_id
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create message from dictionary."""
        try:
            role = MessageRole(data.get("role", "user"))
        except ValueError:
            logger.warning(f"Invalid message role '{data.get('role')}' found. Defaulting to USER.")
            role = MessageRole.USER

        message = cls(
            id=data.get("id", str(uuid.uuid4())),
            role=role,
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time())
        )
        message.tool_calls = data.get('tool_calls')
        message.tool_call_id = data.get('tool_call_id')
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
    selected_model: Optional[str] = None # Added field for selected model

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.updated_at = time.time()

    def get_history_for_llm(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Formats message history for sending to an LLM (OpenAI format)."""
        history = []
        if self.system_prompt:
            history.append({"role": "system", "content": self.system_prompt})

        messages_to_include = self.messages
        if max_messages is not None and len(messages_to_include) > max_messages:
             # Simple truncation for now, preserving system prompt is complex
             messages_to_include = self.messages[-max_messages:] # Keep last N messages

        for msg in messages_to_include:
            if msg.role == MessageRole.SYSTEM and self.system_prompt:
                continue # Skip if system prompt is handled separately
            msg_dict = {"role": msg.role.value, "content": msg.content}
            if msg.tool_calls:
                msg_dict['tool_calls'] = msg.tool_calls
            if msg.tool_call_id:
                msg_dict['tool_call_id'] = msg.tool_call_id
            history.append(msg_dict)
        return history

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for saving."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [msg.to_dict() for msg in self.messages],
            "system_prompt": self.system_prompt,
            "selected_model": self.selected_model
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create session from dictionary loaded from storage."""
        session = cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", "New Chat"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            system_prompt=data.get("system_prompt"),
            selected_model=data.get("selected_model") # Load may be None initially
        )
        messages = []
        for msg_data in data.get("messages", []):
            try:
                messages.append(ChatMessage.from_dict(msg_data))
            except Exception as e:
                logger.error(f"Failed to load message {msg_data.get('id','unknown')} in session {session.id}: {e}", exc_info=False)
        session.messages = messages
        return session

class ChatInterface:
    """
    ChatInterface manages chat sessions and interactions with the user.
    """
    _instance = None

    @classmethod
    def get_instance(cls) -> 'ChatInterface':
        if cls._instance is None:
            cls._instance = ChatInterface()
        return cls._instance

    def __init__(self):
        if ChatInterface._instance is not None:
            raise RuntimeError("ChatInterface is a singleton.")
        self.logger = logger
        self.event_bus = EventBus.get_instance()
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        self.agent_system = AgentSystem.get_instance()
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        self._sessions_lock = threading.Lock()
        self.sessions_dir: Optional[Path] = None
        self.max_history_for_llm = self.config_manager.get_config("chat.max_history_for_llm", 20)
        self.auto_save = self.config_manager.get_config("chat.auto_save", True)
        self.logger.info("ChatInterface initialized")

    def _get_default_model_from_config(self) -> Optional[str]:
        """Gets the default model ID based on the default provider configuration."""
        try:
            default_provider = self.config_manager.get_config("api.default_provider", "openrouter")
            model_id = self.config_manager.get_config(f"api.providers.{default_provider}.default_model")
            if model_id:
                 self.logger.debug(f"Determined default model from config: '{model_id}' (Provider: {default_provider})")
                 return model_id
            else:
                 self.logger.warning(f"Default model not configured for default provider '{default_provider}'.")
                 return None
        except Exception as e:
            self.logger.error(f"Error getting default model from config: {e}", exc_info=True)
            return None


    def startup(self) -> None:
        """Start the chat interface, load sessions, ensure current session and model exist."""
        self.logger.info("ChatInterface startup: STARTING")
        # --- Ensure sessions dir exists ---
        if self.sessions_dir:
            self.logger.info(f"ChatInterface startup: Ensuring sessions dir exists: {self.sessions_dir}")
            try:
                self.sessions_dir.mkdir(parents=True, exist_ok=True)
                self._load_sessions() # Load existing sessions
                self.logger.info("ChatInterface startup: Sessions loaded.")
            except OSError as e:
                self.logger.error(f"ChatInterface startup: Failed to create/access sessions directory {self.sessions_dir}: {e}", exc_info=True)
                self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"action": "ensure_sessions_dir"})
                self.logger.warning("ChatInterface startup: Continuing without session persistence.")
                self.sessions_dir = None
                self.auto_save = False
        else:
            self.logger.error("Session directory not set before startup. Sessions will not be loaded/saved.")
            self.auto_save = False

        self.logger.info("ChatInterface startup: Ensuring a valid session exists...")
        # --- Ensure a session exists and is selected ---
        if not self.sessions:
            self.logger.info("No sessions loaded. Creating a new default session.")
            default_session = self.create_session("New Chat") # Creates with default model logic
            if default_session:
                self.current_session_id = default_session.id
                self.logger.info(f"ChatInterface startup: Default session created: {self.current_session_id}")
            else:
                self.logger.error("ChatInterface startup: CRITICAL - Failed to create initial default session.")
                # Handle critical failure? For now, log and continue, chat might not work.
                self.current_session_id = None
        elif not self.current_session_id or self.current_session_id not in self.sessions:
            self.logger.info("ChatInterface startup: No current session ID set or invalid. Selecting most recent.")
            most_recent_id = max(self.sessions, key=lambda sid: self.sessions[sid].updated_at, default=None)
            if most_recent_id:
                self.current_session_id = most_recent_id
                self.logger.info(f"Set current session to most recent: {self.current_session_id}")
                self._save_current_session_id()
            else: # Should not happen if sessions exist, but handle defensively
                self.logger.warning("No valid sessions found. Creating a new default session as fallback.")
                fallback_session = self.create_session("New Chat")
                if fallback_session:
                    self.current_session_id = fallback_session.id
                else:
                    self.logger.error("ChatInterface startup: CRITICAL - Failed to create fallback default session.")
                    self.current_session_id = None

        # --- Ensure the current session has a selected model ---
        if self.current_session_id:
            current_session = self.get_session(self.current_session_id)
            if current_session and current_session.selected_model is None:
                self.logger.warning(f"Current session {self.current_session_id} has no selected model. Assigning default.")
                default_model = self._get_default_model_from_config()
                if default_model:
                     self.update_session_model(self.current_session_id, default_model)
                else:
                     self.logger.error(f"Could not assign default model to session {self.current_session_id}.")

        self.logger.info(f"ChatInterface started. Current session: {self.current_session_id}")
        self.logger.info("ChatInterface startup: FINISHED")

    # ... (shutdown, _load_sessions, _load_current_session_id, _save_current_session_id, _save_session, _save_all_sessions) ...
    # These remain the same as the previous correct version

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
            self.sessions.clear()
            self.logger.debug(f"ChatInterface _load_sessions: Scanning {self.sessions_dir}...")
            for file_path in self.sessions_dir.glob("*.json"):
                self.logger.debug(f"ChatInterface _load_sessions: Attempting to load {file_path.name}")
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        session_data = json.load(f)
                    session = ChatSession.from_dict(session_data)
                    if session and session.id == file_path.stem:
                         self.sessions[session.id] = session
                         loaded_count += 1
                    elif session:
                         self.logger.warning(f"Session ID mismatch in file {file_path.name}. Expected {file_path.stem}, got {session.id}. Skipping.")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error decoding JSON from session file {file_path.name}: {e}")
                    self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"file": str(file_path), "action": "load_session"})
                except Exception as e:
                    self.logger.error(f"Error loading session file {file_path.name}: {e}", exc_info=True)
                    self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"file": str(file_path), "action": "load_session"})

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
                 # Verify the loaded ID actually exists in the sessions dict before setting
                 if last_id in self.sessions:
                     self.current_session_id = last_id
                     self.logger.debug(f"ChatInterface: Current session set to {last_id}")
                 else:
                     self.logger.warning(f"ChatInterface: Saved current session ID '{last_id}' not found in loaded sessions. Will select most recent.")
                     self.current_session_id = None # Force selection of most recent later
             except Exception as e:
                  self.logger.error(f"Error reading current session file: {e}", exc_info=True)


    def _save_current_session_id(self) -> bool:
        """Saves the current session ID to the .current_session file."""
        if not self.sessions_dir:
            self.logger.warning("Cannot save current session ID: sessions_dir not set.")
            return False
        if not self.current_session_id:
             self.logger.warning("Cannot save current session ID: current_session_id is None.")
             # Optionally delete the file if current session is None
             current_session_file = self.sessions_dir / ".current_session"
             if current_session_file.exists():
                 try:
                     current_session_file.unlink()
                     self.logger.debug("Removed .current_session file as current session ID is None.")
                 except OSError as e:
                     self.logger.error(f"Error removing .current_session file: {e}")
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
        """Save a single chat session to disk. Returns True on success, False on failure."""
        self.logger.debug(f"ChatInterface _save_session: Saving session {session_id}...")
        if not self.sessions_dir:
            self.logger.warning("Session directory not set. Cannot save session.")
            return False

        session_to_save = None
        with self._sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                self.logger.warning(f"Attempted to save non-existent session: {session_id}")
                return False
            try:
                session_dict = session.to_dict()
                session_to_save = session_dict
            except Exception as e:
                 self.logger.error(f"Error converting session {session_id} to dict: {e}", exc_info=True)
                 return False

        if session_to_save:
            file_path = self.sessions_dir / f"{session_id}.json"
            temp_file_path = file_path.with_suffix(".tmp")
            try:
                self.logger.debug(f"ChatInterface _save_session: Writing session {session_id} to temp file {temp_file_path}...")
                with temp_file_path.open("w", encoding="utf-8") as f:
                    json.dump(session_to_save, f, indent=2, ensure_ascii=False)
                os.replace(temp_file_path, file_path)
                self.logger.debug(f"Saved session {session_id} to {file_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error saving session {session_id} to {file_path}: {e}", exc_info=True)
                self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.ERROR, {"file": str(file_path), "action": "save_session"})
                if temp_file_path.exists():
                    try: temp_file_path.unlink()
                    except OSError: pass
                return False
        else:
             return False


    def _save_all_sessions(self) -> None:
         """Saves all current sessions and the current session ID."""
         if not self.sessions_dir:
             self.logger.warning("Session directory not set. Cannot save sessions.")
             return

         self.logger.info("Saving all chat sessions...")
         session_ids_to_save = []
         with self._sessions_lock:
             session_ids_to_save = list(self.sessions.keys())

         saved_count = 0
         failed_count = 0
         for session_id in session_ids_to_save:
             if self._save_session(session_id):
                 saved_count += 1
             else:
                  failed_count += 1

         self.logger.info(f"Finished saving sessions. Success: {saved_count}, Failed: {failed_count}")
         self._save_current_session_id()

    def create_session(self, title: str = "New Chat", system_prompt: Optional[str] = None, model_id: Optional[str] = None) -> Optional[ChatSession]:
        """
        Create a new chat session. Assigns a default model if not provided.
        Returns the session on success, None on failure.
        """
        self.logger.debug(f"ChatInterface create_session: STARTING creation for title '{title}'")
        session_saved = False
        session = None

        # Determine model if not provided
        if not model_id:
             model_id = self._get_default_model_from_config()
             # Handle case where even the default couldn't be determined
             if not model_id:
                  self.logger.error("Failed to determine a default model for new session. Cannot create session.")
                  return None # Cannot create without a model

        try:
            with self._sessions_lock:
                session = ChatSession(title=title, system_prompt=system_prompt, selected_model=model_id)
                self.sessions[session.id] = session
                self.logger.info(f"Created new chat session object: {session.id} ('{title}') with model {model_id}")

                if self.current_session_id is None:
                     self.current_session_id = session.id
                     self.logger.debug(f"ChatInterface create_session: Set new session {session.id} as current (in memory).")

            # Perform saving operations outside the main lock
            if self.auto_save:
                self.logger.debug(f"ChatInterface create_session: Auto-saving session {session.id}...")
                session_saved = self._save_session(session.id)
                if not session_saved:
                     self.logger.error(f"ChatInterface create_session: Failed to auto-save session {session.id}.")
                else:
                     self.logger.debug(f"ChatInterface create_session: Auto-save complete for {session.id}.")

            if self.current_session_id == session.id:
                self.logger.debug(f"ChatInterface create_session: Saving current session ID {session.id}...")
                if not self._save_current_session_id():
                     self.logger.error(f"ChatInterface create_session: Failed to save current session ID {session.id}.")

            self.logger.debug(f"ChatInterface create_session: Publishing event for {session.id}...")
            self.event_bus.publish_event("chat.session.created", {"session_id": session.id, "title": title, "model_id": model_id})
            self.logger.debug(f"ChatInterface create_session: FINISHED creation for {session.id}. Save success: {session_saved}")
            return session

        except Exception as e:
             self.logger.error(f"ChatInterface create_session: CRITICAL ERROR during session creation for title '{title}'.", exc_info=True)
             self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL, {"action": "create_session"})
             if session and session.id in self.sessions:
                 with self._sessions_lock:
                     if session.id in self.sessions:
                         del self.sessions[session.id]
             return None

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by ID. Ensures the session has a selected_model assigned.
        """
        with self._sessions_lock:
            session = self.sessions.get(session_id)
            # Ensure session has a model assigned if accessed
            if session and session.selected_model is None:
                 self.logger.warning(f"Session {session_id} retrieved without a selected model. Assigning default.")
                 default_model = self._get_default_model_from_config()
                 if default_model:
                     session.selected_model = default_model
                     # Mark as updated? Might trigger unnecessary saves if not careful
                     # session.updated_at = time.time()
                 else:
                     self.logger.error(f"Could not assign default model to session {session_id} during get.")
            return session

    def get_all_sessions(self) -> List[ChatSession]:
        """Get all chat sessions, sorted by last updated time (newest first)."""
        with self._sessions_lock:
            sessions_list = list(self.sessions.values())
        # Ensure all sessions have a model before returning (could be done in get_session too)
        default_model = self._get_default_model_from_config()
        for session in sessions_list:
            if session.selected_model is None:
                 session.selected_model = default_model
        return sorted(sessions_list, key=lambda s: s.updated_at, reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        session_file_deleted = False
        session_existed = False
        new_current_id = None

        with self._sessions_lock:
            if session_id in self.sessions:
                session_existed = True
                del self.sessions[session_id]
                self.logger.info(f"Deleted chat session from memory: {session_id}")

                if self.sessions_dir:
                    file_path = self.sessions_dir / f"{session_id}.json"
                    try:
                        file_path.unlink(missing_ok=True)
                        session_file_deleted = True
                        self.logger.debug(f"Deleted session file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error deleting session file {file_path}: {e}", exc_info=True)
                        self.error_handler.handle_error(e, ErrorCategory.STORAGE, ErrorSeverity.WARNING, {"file": str(file_path), "action": "delete_session_file"})

                if self.current_session_id == session_id:
                    most_recent_id = max(self.sessions, key=lambda sid: self.sessions[sid].updated_at, default=None)
                    self.current_session_id = most_recent_id
                    new_current_id = most_recent_id
                    if self.current_session_id:
                        self.logger.info(f"Current session changed to {self.current_session_id} after deletion.")
                    else:
                        self.logger.info("No sessions left after deletion. Will create new one on next message.")
            else:
                self.logger.warning(f"Attempted to delete non-existent session: {session_id}")
                return False

        if session_existed:
             self._save_current_session_id() # Save potentially changed current ID
             self.event_bus.publish_event("chat.session.deleted", {"session_id": session_id})

        return True


    def set_current_session(self, session_id: str) -> bool:
        """Set the current chat session."""
        session = self.get_session(session_id) # Use getter which ensures model is set
        if not session:
            self.logger.warning(f"Attempted to set current session to non-existent ID: {session_id}")
            return False

        if self.current_session_id != session_id:
             self.current_session_id = session_id
             self.logger.info(f"Current session set to: {session_id}")
             if not self._save_current_session_id():
                 self.logger.error("Failed to save updated current session ID.")
             self.event_bus.publish_event("chat.session.selected", {"session_id": session_id})
        return True

    async def process_message(self, message_text: str, session_id: Optional[str] = None) -> str:
        """Process a user message using the AgentSystem and the session's selected model."""
        target_session_id = session_id if session_id is not None else self.current_session_id
        session = self.get_session(target_session_id) # Use getter to ensure session exists and has model

        if session is None:
            self.logger.warning(f"No valid session ID ({target_session_id}). Creating new session.")
            try:
                session = self.create_session() # Creates with default model logic
                if not session:
                     return "Error: Could not create a new chat session."
                target_session_id = session.id
                self.set_current_session(target_session_id) # This sets current_session_id
            except Exception as e:
                 self.logger.error(f"Failed to create session in process_message: {e}", exc_info=True)
                 return "Error: Failed to establish a chat session."

        # Add User Message
        user_message = ChatMessage(role=MessageRole.USER, content=message_text)
        session.add_message(user_message)
        self.event_bus.publish_event("chat.message.added", {"session_id": session.id, "message": user_message.to_dict()})
        if self.auto_save: self._save_session(session.id)

        # Log with model info
        self.logger.info(f"Processing message for session {session.id} (Model: {session.selected_model}): '{message_text[:50]}...'")

        try:
            history = session.get_history_for_llm(max_messages=self.max_history_for_llm)
            session_model_id = session.selected_model # Get model guaranteed by get_session

            if not session_model_id: # Should not happen if startup/get logic is correct
                self.logger.error(f"Session {session.id} still has no model ID before calling agent!")
                return "Error: Internal configuration error - session has no model assigned."

            response_text = await self.agent_system.process_user_request(
                user_message=message_text,
                conversation_history=history[:-1],
                requested_model_id=session_model_id # Pass the session's model
            )

            # Add Assistant Response
            assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
            session.add_message(assistant_message)
            self.event_bus.publish_event("chat.message.added", {"session_id": session.id, "message": assistant_message.to_dict()})
            if self.auto_save: self._save_session(session.id)

            self.logger.info(f"Generated response for session {session.id}: '{response_text[:50]}...'")
            return response_text

        except Exception as e:
            self.logger.error(f"Error processing message in session {session.id}: {e}", exc_info=True)
            self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.ERROR, {"session_id": session.id, "action": "process_message"})
            error_response = f"I encountered an error processing your request: {type(e).__name__}"
            # Add Error Response
            error_message = ChatMessage(role=MessageRole.ASSISTANT, content=error_response)
            session.add_message(error_message)
            self.event_bus.publish_event("chat.message.added", {"session_id": session.id, "message": error_message.to_dict()})
            if self.auto_save: self._save_session(session.id)
            return error_response

    def clear_session_history(self, session_id: Optional[str] = None) -> bool:
        """Clear chat history for a session, keeping system prompt and model."""
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

            # Keep only system messages if system_prompt isn't explicitly set on session
            if session.system_prompt:
                 session.messages = []
            else:
                 session.messages = [msg for msg in session.messages if msg.role == MessageRole.SYSTEM]

            session.updated_at = time.time()
            session_cleared = True

        if session_cleared:
            self.event_bus.publish_event("chat.session.cleared", {"session_id": target_session_id})
            if self.auto_save:
                if not self._save_session(target_session_id):
                     self.logger.warning(f"Failed to save session {target_session_id} after clearing history.")
            self.logger.info(f"Cleared history for session {target_session_id}")
            return True
        else:
            return False

    def update_session_model(self, session_id: str, model_id: str) -> bool:
        """Updates the selected model for a given session."""
        # Validate model_id format briefly?
        if not isinstance(model_id, str) or not model_id:
            self.logger.error(f"Invalid model_id provided for update: {model_id}")
            return False

        session = self.get_session(session_id) # Uses internal lock
        if not session:
             self.logger.warning(f"Cannot update model for non-existent session: {session_id}")
             return False

        # Check if model actually changed
        if session.selected_model == model_id:
            self.logger.debug(f"Model for session {session_id} is already {model_id}. No update needed.")
            return True

        # Update session object
        with self._sessions_lock: # Lock needed for modification
            session.selected_model = model_id
            session.updated_at = time.time()

        self.logger.info(f"Updated model for session {session_id} to {model_id}")

        # Save outside lock
        if self.auto_save:
            if not self._save_session(session_id):
                 self.logger.error(f"Failed to save session {session_id} after model update.")
                 # Should we revert the change in memory? For now, log and return False
                 return False

        self.event_bus.publish_event("chat.session.model.updated", {"session_id": session_id, "model_id": model_id})
        return True
# END OF FILE miniManus-main/minimanus/ui/chat_interface.py
