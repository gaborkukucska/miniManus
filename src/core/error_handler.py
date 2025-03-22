#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Error Handler for miniManus

This module implements the Error Handler component, which centralizes error management,
implements graceful recovery strategies, provides user-friendly error messages,
and logs errors for troubleshooting.
"""

import os
import sys
import logging
import traceback
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum, auto

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority

logger = logging.getLogger("miniManus.ErrorHandler")

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    INFO = auto()      # Informational, not an error
    WARNING = auto()   # Warning, operation can continue
    ERROR = auto()     # Error, operation failed but system can continue
    CRITICAL = auto()  # Critical error, system stability affected
    FATAL = auto()     # Fatal error, system cannot continue

class ErrorCategory(Enum):
    """Categories of errors."""
    SYSTEM = auto()    # System-level errors
    API = auto()       # API-related errors
    UI = auto()        # UI-related errors
    CONFIG = auto()    # Configuration errors
    RESOURCE = auto()  # Resource-related errors
    PLUGIN = auto()    # Plugin-related errors
    NETWORK = auto()   # Network-related errors
    UNKNOWN = auto()   # Unknown errors

class ErrorHandler:
    """
    ErrorHandler centralizes error management across the system.
    
    It handles:
    - Error logging and categorization
    - Error event publishing
    - Recovery strategy selection
    - User-friendly error message generation
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
        
        # Error history (limited size)
        self.max_history_size = 100
        self.error_history = []
        self.history_lock = threading.RLock()
        
        # Recovery strategies
        self.recovery_strategies = {}
        
        # Error message templates
        self.message_templates = {
            ErrorCategory.SYSTEM: {
                ErrorSeverity.INFO: "System information: {message}",
                ErrorSeverity.WARNING: "System warning: {message}",
                ErrorSeverity.ERROR: "System error: {message}. Please try again later.",
                ErrorSeverity.CRITICAL: "Critical system error: {message}. Some features may be unavailable.",
                ErrorSeverity.FATAL: "Fatal system error: {message}. Please restart the application.",
            },
            ErrorCategory.API: {
                ErrorSeverity.INFO: "API information: {message}",
                ErrorSeverity.WARNING: "API warning: {message}",
                ErrorSeverity.ERROR: "API error: {message}. Please check your connection and try again.",
                ErrorSeverity.CRITICAL: "Critical API error: {message}. Switching to fallback provider.",
                ErrorSeverity.FATAL: "Fatal API error: {message}. No providers available.",
            },
            ErrorCategory.UI: {
                ErrorSeverity.INFO: "UI information: {message}",
                ErrorSeverity.WARNING: "UI warning: {message}",
                ErrorSeverity.ERROR: "UI error: {message}. Please try again.",
                ErrorSeverity.CRITICAL: "Critical UI error: {message}. Resetting UI state.",
                ErrorSeverity.FATAL: "Fatal UI error: {message}. Please restart the application.",
            },
            ErrorCategory.CONFIG: {
                ErrorSeverity.INFO: "Configuration information: {message}",
                ErrorSeverity.WARNING: "Configuration warning: {message}",
                ErrorSeverity.ERROR: "Configuration error: {message}. Using default settings.",
                ErrorSeverity.CRITICAL: "Critical configuration error: {message}. Some features may be unavailable.",
                ErrorSeverity.FATAL: "Fatal configuration error: {message}. Please check your configuration file.",
            },
            ErrorCategory.RESOURCE: {
                ErrorSeverity.INFO: "Resource information: {message}",
                ErrorSeverity.WARNING: "Resource warning: {message}",
                ErrorSeverity.ERROR: "Resource error: {message}. Some operations may be slower.",
                ErrorSeverity.CRITICAL: "Critical resource error: {message}. Reducing functionality to conserve resources.",
                ErrorSeverity.FATAL: "Fatal resource error: {message}. Cannot continue operation.",
            },
            ErrorCategory.PLUGIN: {
                ErrorSeverity.INFO: "Plugin information: {message}",
                ErrorSeverity.WARNING: "Plugin warning: {message}",
                ErrorSeverity.ERROR: "Plugin error: {message}. Plugin functionality may be limited.",
                ErrorSeverity.CRITICAL: "Critical plugin error: {message}. Disabling plugin.",
                ErrorSeverity.FATAL: "Fatal plugin error: {message}. Please remove or update the plugin.",
            },
            ErrorCategory.NETWORK: {
                ErrorSeverity.INFO: "Network information: {message}",
                ErrorSeverity.WARNING: "Network warning: {message}",
                ErrorSeverity.ERROR: "Network error: {message}. Please check your connection.",
                ErrorSeverity.CRITICAL: "Critical network error: {message}. Switching to offline mode.",
                ErrorSeverity.FATAL: "Fatal network error: {message}. Network functionality unavailable.",
            },
            ErrorCategory.UNKNOWN: {
                ErrorSeverity.INFO: "Information: {message}",
                ErrorSeverity.WARNING: "Warning: {message}",
                ErrorSeverity.ERROR: "Error: {message}. Please try again.",
                ErrorSeverity.CRITICAL: "Critical error: {message}. Please restart the application.",
                ErrorSeverity.FATAL: "Fatal error: {message}. Please contact support.",
            },
        }
        
        self.logger.info("ErrorHandler initialized")
    
    def register_recovery_strategy(self, category: ErrorCategory, 
                                  severity: ErrorSeverity,
                                  strategy: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Register a recovery strategy for a specific error category and severity.
        
        Args:
            category: Error category
            severity: Error severity
            strategy: Function to call for recovery, returns True if recovery successful
        """
        key = (category, severity)
        if key not in self.recovery_strategies:
            self.recovery_strategies[key] = []
        
        self.recovery_strategies[key].append(strategy)
        self.logger.debug(f"Registered recovery strategy for {category.name}/{severity.name}")
    
    def handle_error(self, error: Exception, category: ErrorCategory = ErrorCategory.UNKNOWN,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    context: Optional[Dict[str, Any]] = None) -> str:
        """
        Handle an error by logging it, publishing an event, and attempting recovery.
        
        Args:
            error: Exception that occurred
            category: Category of the error
            severity: Severity of the error
            context: Additional context information
            
        Returns:
            User-friendly error message
        """
        context = context or {}
        error_info = {
            "error": error,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "category": category,
            "severity": severity,
            "context": context,
            "timestamp": threading.Event().time(),  # Current time
        }
        
        # Log the error
        log_message = f"{category.name} {severity.name}: {str(error)}"
        if severity == ErrorSeverity.FATAL:
            self.logger.critical(log_message, exc_info=True)
        elif severity == ErrorSeverity.CRITICAL:
            self.logger.error(log_message, exc_info=True)
        elif severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Add to error history
        with self.history_lock:
            self.error_history.append(error_info)
            # Trim history if needed
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size:]
        
        # Publish error event
        event_priority = EventPriority.NORMAL
        if severity == ErrorSeverity.FATAL:
            event_priority = EventPriority.CRITICAL
        elif severity == ErrorSeverity.CRITICAL:
            event_priority = EventPriority.HIGH
        
        self.event_bus.publish_event(
            f"error.{category.name.lower()}.{severity.name.lower()}",
            error_info,
            event_priority
        )
        
        # Attempt recovery
        self._attempt_recovery(error_info)
        
        # Generate user-friendly message
        return self.get_user_message(error, category, severity)
    
    def _attempt_recovery(self, error_info: Dict[str, Any]) -> bool:
        """
        Attempt to recover from an error using registered strategies.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        category = error_info["category"]
        severity = error_info["severity"]
        key = (category, severity)
        
        if key in self.recovery_strategies:
            for strategy in self.recovery_strategies[key]:
                try:
                    if strategy(error_info):
                        self.logger.info(f"Recovery successful for {category.name}/{severity.name}")
                        return True
                except Exception as e:
                    self.logger.error(f"Error in recovery strategy: {str(e)}")
        
        return False
    
    def get_user_message(self, error: Exception, category: ErrorCategory,
                        severity: ErrorSeverity) -> str:
        """
        Generate a user-friendly error message.
        
        Args:
            error: Exception that occurred
            category: Category of the error
            severity: Severity of the error
            
        Returns:
            User-friendly error message
        """
        template = self.message_templates.get(category, {}).get(
            severity, "Error: {message}"
        )
        
        return template.format(message=str(error))
    
    def get_error_history(self, limit: Optional[int] = None,
                         category: Optional[ErrorCategory] = None,
                         min_severity: Optional[ErrorSeverity] = None) -> List[Dict[str, Any]]:
        """
        Get error history, optionally filtered.
        
        Args:
            limit: Maximum number of errors to return
            category: Filter by error category
            min_severity: Filter by minimum severity
            
        Returns:
            List of error information dictionaries
        """
        with self.history_lock:
            filtered_history = self.error_history
            
            # Apply filters
            if category:
                filtered_history = [e for e in filtered_history if e["category"] == category]
            
            if min_severity:
                filtered_history = [e for e in filtered_history if e["severity"].value >= min_severity.value]
            
            # Apply limit
            if limit is not None:
                filtered_history = filtered_history[-limit:]
            
            return filtered_history
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        with self.history_lock:
            self.error_history = []
        self.logger.debug("Error history cleared")

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize EventBus first
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    # Subscribe to error events
    def handle_error_event(event):
        print(f"Error event: {event.event_type}")
        print(f"  Error: {event.data['error_type']}: {event.data['message']}")
    
    event_bus.subscribe("error.api.error", handle_error_event)
    
    # Initialize ErrorHandler
    error_handler = ErrorHandler.get_instance()
    
    # Register a recovery strategy
    def api_error_recovery(error_info):
        print("Attempting API error recovery...")
        return True  # Simulate successful recovery
    
    error_handler.register_recovery_strategy(
        ErrorCategory.API, ErrorSeverity.ERROR, api_error_recovery
    )
    
    # Handle some errors
    try:
        # Simulate an API error
        raise ConnectionError("Failed to connect to API server")
    except Exception as e:
        user_message = error_handler.handle_error(
            e, ErrorCategory.API, ErrorSeverity.ERROR,
            {"api": "openrouter", "endpoint": "/completions"}
        )
        print(f"User message: {user_message}")
    
    # Print error history
    print("\nError history:")
    for error in error_handler.get_error_history():
        print(f"- {error['category'].name} {error['severity'].name}: {error['message']}")
    
    # Shutdown
    event_bus.shutdown()
