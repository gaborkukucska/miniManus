#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
System Manager for miniManus

This module implements the System Manager component, which is responsible for
initializing and coordinating all system components, managing the application
lifecycle, and handling system-level events.
"""

import os
import sys
import signal
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("miniManus.SystemManager")

class SystemManager:
    """
    SystemManager is the central coordinator for the miniManus framework.
    
    It handles:
    - System initialization and shutdown
    - Component lifecycle management
    - Signal handling
    - System-level event coordination
    """
    
    _instance = None  # Singleton instance
    _shutdown_event = threading.Event()  # Event to signal shutdown completion
    
    @classmethod
    def get_instance(cls) -> 'SystemManager':
        """Get or create the singleton instance of SystemManager."""
        if cls._instance is None:
            cls._instance = SystemManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the SystemManager."""
        if SystemManager._instance is not None:
            raise RuntimeError("SystemManager is a singleton. Use get_instance() instead.")
        
        self.logger = logger
        self.components = {}
        self.is_running = False
        self.is_shutting_down = False
        self.startup_complete = False
        self._shutdown_lock = threading.Lock()
        self._startup_errors = []
        self._shutdown_timeout = 5  # Timeout in seconds for shutdown
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        self.logger.info("SystemManager initialized")
    
    def register_component(self, name: str, component: Any) -> None:
        """
        Register a component with the SystemManager.
        
        Args:
            name: Unique name for the component
            component: The component instance to register
        """
        if name in self.components:
            self.logger.warning(f"Component {name} already registered, replacing")
        
        self.components[name] = component
        self.logger.debug(f"Registered component: {name}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a registered component by name.
        
        Args:
            name: Name of the component to retrieve
            
        Returns:
            The component instance or None if not found
        """
        return self.components.get(name)
    
    def startup(self, component_order: Optional[List[str]] = None) -> bool:
        """
        Start all registered components in the specified order.
        
        Args:
            component_order: Optional list specifying the order in which to start components
            
        Returns:
            True if startup was successful, False otherwise
        """
        self.logger.info("Starting miniManus system...")
        self.is_running = True
        self.is_shutting_down = False
        SystemManager._shutdown_event.clear()
        
        # Determine component startup order
        if component_order is None:
            # Default order: start core components first
            component_order = list(self.components.keys())
        
        # Start each component
        for name in component_order:
            if name not in self.components:
                self.logger.warning(f"Component {name} not found, skipping startup")
                continue
            
            component = self.components[name]
            try:
                if hasattr(component, 'startup') and callable(getattr(component, 'startup')):
                    self.logger.info(f"Starting component: {name}")
                    component.startup()
                    self.logger.debug(f"Component {name} started successfully")
                else:
                    self.logger.debug(f"Component {name} has no startup method")
            except Exception as e:
                self.logger.error(f"Error starting component {name}: {str(e)}")
                self._startup_errors.append((name, str(e)))
        
        if self._startup_errors:
            self.logger.error(f"System startup completed with {len(self._startup_errors)} errors")
            return False
        
        self.startup_complete = True
        self.logger.info("miniManus system startup complete")
        return True
    
    def shutdown(self, component_order: Optional[List[str]] = None, force_exit: bool = False) -> None:
        """
        Shutdown all registered components in the specified order.
        
        Args:
            component_order: Optional list specifying the order in which to shutdown components
            force_exit: Whether to force exit the process after shutdown
        """
        # Use lock to prevent multiple simultaneous shutdown attempts
        with self._shutdown_lock:
            if self.is_shutting_down:
                self.logger.info("Shutdown already in progress")
                
                # If force_exit is True, exit immediately even if shutdown is in progress
                if force_exit:
                    self.logger.info("Forcing exit...")
                    os._exit(0)  # Use os._exit to force immediate termination
                
                return
            
            self.logger.info("Shutting down miniManus system...")
            self.is_shutting_down = True
            self.is_running = False
            
            # Determine component shutdown order (reverse of startup by default)
            if component_order is None:
                # Default order: shutdown in reverse order of startup
                component_order = list(reversed(list(self.components.keys())))
            
            # Shutdown each component
            for name in component_order:
                if name not in self.components:
                    continue
                
                component = self.components[name]
                try:
                    if hasattr(component, 'shutdown') and callable(getattr(component, 'shutdown')):
                        self.logger.info(f"Shutting down component: {name}")
                        component.shutdown()
                        self.logger.debug(f"Component {name} shutdown complete")
                    else:
                        self.logger.debug(f"Component {name} has no shutdown method")
                except Exception as e:
                    self.logger.error(f"Error shutting down component {name}: {str(e)}")
            
            self.logger.info("miniManus system shutdown complete")
            
            # Signal that shutdown is complete
            SystemManager._shutdown_event.set()
            
            # If force_exit is True, exit the process
            if force_exit:
                self.logger.info("Exiting process...")
                sys.exit(0)
    
    def _handle_signal(self, signum: int, frame) -> None:
        """
        Handle system signals (SIGINT, SIGTERM).
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        sig_name = signal.Signals(signum).name
        self.logger.info(f"Received signal {sig_name}")
        
        if signum in (signal.SIGINT, signal.SIGTERM):
            self.logger.info("Initiating graceful shutdown...")
            
            # If this is the first signal, try graceful shutdown
            if not self.is_shutting_down:
                # Start shutdown in a separate thread
                shutdown_thread = threading.Thread(target=self.shutdown, kwargs={"force_exit": True})
                shutdown_thread.daemon = True  # Make thread daemon so it doesn't block process exit
                shutdown_thread.start()
                
                # Wait for shutdown to complete with timeout
                if not SystemManager._shutdown_event.wait(self._shutdown_timeout):
                    self.logger.warning(f"Shutdown timed out after {self._shutdown_timeout} seconds, forcing exit...")
                    os._exit(1)  # Force exit if shutdown times out
            else:
                # If shutdown is already in progress and we get another signal, force exit
                self.logger.warning("Received second interrupt signal, forcing immediate exit...")
                os._exit(1)  # Force immediate exit
    
    def run_forever(self) -> None:
        """Run the system until shutdown is requested."""
        if not self.startup_complete:
            self.startup()
        
        self.logger.info("miniManus system running...")
        
        try:
            # Keep the main thread alive until shutdown is requested
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received")
        finally:
            if not self.is_shutting_down:
                self.shutdown(force_exit=True)

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    system = SystemManager.get_instance()
    
    class DummyComponent:
        def startup(self):
            print("DummyComponent started")
        
        def shutdown(self):
            print("DummyComponent shutdown")
    
    system.register_component("dummy", DummyComponent())
    system.startup()
    
    # Simulate running for a short time
    time.sleep(2)
    
    system.shutdown()
