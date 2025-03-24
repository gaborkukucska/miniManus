#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Error Handler for miniManus

This module implements the Error Handler component, which provides centralized
error handling and logging capabilities for the miniManus framework.
"""

import os
import sys
import logging
import traceback
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum, auto
from pathlib import Path

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority

logger = logging.getLogger("miniManus.ErrorHandler")

class ErrorCategory(Enum):
    """Error categories."""
    SYSTEM = auto()
    API = auto()
    UI = auto()
    NETWORK = auto()
    STORAGE = auto()
    RESOURCE = auto()
    PLUGIN = auto()
    SECURITY = auto()
    UNKNOWN = auto()

class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class ErrorHandler:
    """
    ErrorHandler provides centralized error handling and logging capabilities.
    
    It handles:
    - Error logging
    - Error categorization
    - Error notification
    - Error recovery
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'ErrorHandler':
        """Get or create the singleton instance of ErrorHandler."""
        if cls._instance is None:
            cls._instance = ErrorHandler()
        return cls._instance
    
    def __init__(self):
        """Initialize the ErrorHandler."""
        if ErrorHandler._instance is not None:
            raise RuntimeError("ErrorHandler is a singleton. Use get_instance() instead.")
        
        self.logger = logger
        self.event_bus = EventBus.get_instance()
        
        # Error history
        self.error_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
        # Error callbacks
        self.error_callbacks: Dict[ErrorCategory, List[Callable[[Exception, ErrorSeverity, Dict[str, Any]], None]]] = {
            category: [] for category in ErrorCategory
        }
        
        # Register event handlers
        self.event_bus.subscribe("error.occurred", self._handle_error_event)
        
        self.logger.info("ErrorHandler initialized")
    
    def handle_error(self, error: Exception, category: ErrorCategory = ErrorCategory.UNKNOWN,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    context: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle an error.
        
        Args:
            error: Exception to handle
            category: Error category
            severity: Error severity
            context: Additional context information
        """
        # Create error record
        error_record = {
            "error": str(error),
            "type": type(error).__name__,
            "category": category.name,
            "severity": severity.name,
            "context": context or {},
            "traceback": traceback.format_exc(),
            "timestamp": time.time(),  # Current time
        }
        
        # Add to history
        self._add_to_history(error_record)
        
        # Log error
        self._log_error(error_record)
        
        # Publish event
        self.event_bus.publish_event("error.occurred", error_record)
        
        # Call callbacks
        self._call_callbacks(error, category, severity, context or {})
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """
        Get error history.
        
        Returns:
            List of error records
        """
        return self.error_history.copy()
    
    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
        self.logger.debug("Error history cleared")
    
    def set_error_callback(self, category: ErrorCategory,
                          callback: Callable[[Exception, ErrorSeverity, Dict[str, Any]], None]) -> None:
        """
        Set a callback for error handling.
        
        Args:
            category: Error category
            callback: Callback function
        """
        self.error_callbacks[category].append(callback)
        self.logger.debug(f"Error callback set for category {category.name}")
    
    def remove_error_callback(self, category: ErrorCategory,
                             callback: Callable[[Exception, ErrorSeverity, Dict[str, Any]], None]) -> bool:
        """
        Remove an error callback.
        
        Args:
            category: Error category
            callback: Callback function
            
        Returns:
            True if removed, False if not found
        """
        if callback in self.error_callbacks[category]:
            self.error_callbacks[category].remove(callback)
            self.logger.debug(f"Error callback removed for category {category.name}")
            return True
        return False
    
    def _add_to_history(self, error_record: Dict[str, Any]) -> None:
        """
        Add error record to history.
        
        Args:
            error_record: Error record to add
        """
        self.error_history.append(error_record)
        
        # Trim history if needed
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _log_error(self, error_record: Dict[str, Any]) -> None:
        """
        Log error.
        
        Args:
            error_record: Error record to log
        """
        severity = error_record["severity"]
        message = f"{error_record['category']} {severity}: {error_record['error']}"
        
        if severity == ErrorSeverity.CRITICAL.name:
            self.logger.critical(message)
        elif severity == ErrorSeverity.ERROR.name:
            self.logger.error(message)
        elif severity == ErrorSeverity.WARNING.name:
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        # Log traceback for non-info errors
        if severity != ErrorSeverity.INFO.name:
            self.logger.debug(f"Traceback: {error_record['traceback']}")
    
    def _call_callbacks(self, error: Exception, category: ErrorCategory,
                       severity: ErrorSeverity, context: Dict[str, Any]) -> None:
        """
        Call error callbacks.
        
        Args:
            error: Exception
            category: Error category
            severity: Error severity
            context: Additional context information
        """
        for callback in self.error_callbacks[category]:
            try:
                callback(error, severity, context)
            except Exception as e:
                # Log but don't recurse
                self.logger.error(f"Error in error callback: {str(e)}")
    
    def _handle_error_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle error event.
        
        Args:
            event_data: Event data
        """
        # This is just to handle events from other components
        # We don't need to do anything here since we already handled the error
        pass

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    # Initialize ErrorHandler
    error_handler = ErrorHandler.get_instance()
    
    # Example usage
    try:
        # Simulate an error
        raise ValueError("This is a test error")
    except Exception as e:
        error_handler.handle_error(
            e, ErrorCategory.SYSTEM, ErrorSeverity.WARNING,
            {"action": "test"}
        )
    
    # Get error history
    history = error_handler.get_error_history()
    print(f"Error history: {history}")
    
    # Shutdown
    event_bus.shutdown()
