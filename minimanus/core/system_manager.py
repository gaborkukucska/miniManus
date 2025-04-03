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
# Logging is likely configured in __main__.py, but setup a basic one here
# in case SystemManager is used or tested standalone.
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
        # Use a lock for thread-safe singleton creation if needed in complex scenarios
        # but typically __main__ ensures it's called sequentially first.
        if cls._instance is None:
            cls._instance = SystemManager()
        return cls._instance

    def __init__(self):
        """Initialize the SystemManager."""
        if SystemManager._instance is not None:
            raise RuntimeError("SystemManager is a singleton. Use get_instance() instead.")

        self.logger = logger
        self.components: Dict[str, Any] = {}
        self.is_running: bool = False
        self.is_shutting_down: bool = False
        self.startup_complete: bool = False
        self._shutdown_lock = threading.Lock()
        self._startup_errors: List[Tuple[str, str]] = []
        self._shutdown_timeout: float = 10.0  # Increased timeout in seconds for shutdown

        # Register signal handlers only once
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            self.logger.debug("Signal handlers registered for SIGINT and SIGTERM.")
        except ValueError as e:
            # This can happen if run in a thread where signals can't be set
            self.logger.warning(f"Could not set signal handlers: {e}. Graceful shutdown via Ctrl+C might not work.")
        except Exception as e:
            self.logger.error(f"Unexpected error setting signal handlers: {e}", exc_info=True)


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
        if self.is_running:
            self.logger.warning("Startup called but system is already running.")
            return True # Or False depending on desired behavior

        self.logger.info("Starting miniManus system...")
        self.is_running = True
        self.is_shutting_down = False
        SystemManager._shutdown_event.clear()
        self._startup_errors = [] # Reset errors on new startup attempt

        # Determine component startup order
        startup_order: List[str]
        if component_order is None:
            # Default order: Use the order they were registered (often reflects dependencies)
            startup_order = list(self.components.keys())
        else:
            # Validate provided order contains known components
            startup_order = [name for name in component_order if name in self.components]
            missing = [name for name in component_order if name not in self.components]
            if missing:
                 self.logger.warning(f"Startup order specified unknown components: {missing}")
            # Add any missing registered components to the end
            for name in self.components:
                 if name not in startup_order:
                     startup_order.append(name)

        self.logger.debug(f"Startup order: {startup_order}")

        # Start each component
        for name in startup_order:
            # Skip if already validated above, but double-check just in case
            if name not in self.components:
                # This shouldn't happen with the logic above, but good safeguard
                self.logger.warning(f"Component {name} suddenly not found, skipping startup")
                continue

            component = self.components[name]
            try:
                if hasattr(component, 'startup') and callable(getattr(component, 'startup')):
                    self.logger.info(f"Starting component: {name}...")
                    # Consider adding timeout to component startup?
                    component.startup()
                    self.logger.debug(f"Component {name} started successfully")
                else:
                    self.logger.debug(f"Component {name} has no startup method")
            except Exception as e:
                self.logger.error(f"Error starting component {name}: {e}", exc_info=True) # Log full traceback for startup errors
                self._startup_errors.append((name, str(e)))
                # Decide: Stop startup on first error, or try starting others?
                # For now, continue trying to start other components.

        if self._startup_errors:
            self.logger.error(f"System startup completed with {len(self._startup_errors)} errors:")
            for name, err in self._startup_errors:
                 self.logger.error(f"  - {name}: {err}")
            self.is_running = False # Mark as not running if startup failed
            return False

        self.startup_complete = True
        self.logger.info("miniManus system startup complete")
        return True

    def shutdown(self, component_order: Optional[List[str]] = None, force_exit: bool = False) -> None:
        """
        Shutdown all registered components in the specified order.

        Args:
            component_order: Optional list specifying the order in which to shutdown components
            force_exit: Whether to force exit the process after shutdown (use with caution)
        """
        # Use lock to prevent multiple simultaneous shutdown attempts
        with self._shutdown_lock:
            if self.is_shutting_down:
                self.logger.info("Shutdown already in progress")
                if force_exit:
                    self.logger.info("Forcing exit during ongoing shutdown...")
                    os._exit(0) # Use os._exit for immediate termination, bypassing further cleanup
                return

            self.logger.info("Shutting down miniManus system...")
            self.is_shutting_down = True
            self.is_running = False
            shutdown_errors = [] # Track errors during this shutdown attempt

            # Determine component shutdown order (reverse of registration by default)
            shutdown_order: List[str]
            if component_order is None:
                shutdown_order = list(reversed(list(self.components.keys())))
            else:
                # Validate provided order
                shutdown_order = [name for name in component_order if name in self.components]
                missing = [name for name in component_order if name not in self.components]
                if missing:
                    self.logger.warning(f"Shutdown order specified unknown components: {missing}")
                 # Add any missing registered components to the end (to ensure they are shut down)
                for name in reversed(list(self.components.keys())):
                     if name not in shutdown_order:
                         shutdown_order.append(name)

            self.logger.debug(f"Shutdown order: {shutdown_order}") # Log the order

            # Shutdown each component
            for name in shutdown_order:
                # Shouldn't happen with logic above, but safeguard
                if name not in self.components:
                    self.logger.warning(f"Component {name} not found during shutdown.")
                    continue

                component = self.components[name]
                try:
                    if hasattr(component, 'shutdown') and callable(getattr(component, 'shutdown')):
                        # --- DETAILED SHUTDOWN LOGGING ---
                        self.logger.info(f"--> Shutting down component: {name}...")
                        start_time = time.time()
                        component.shutdown() # Call the component's shutdown
                        end_time = time.time()
                        self.logger.info(f"<-- Component {name} shutdown complete ({end_time - start_time:.2f}s)")
                        # --- END DETAILED LOGGING ---
                    else:
                        self.logger.debug(f"Component {name} has no shutdown method, skipping.")
                except Exception as e:
                    # Log error but continue shutting down others
                    self.logger.error(f"Error shutting down component {name}: {e}", exc_info=True) # Log traceback for shutdown errors
                    shutdown_errors.append((name, str(e)))

            if shutdown_errors:
                self.logger.error(f"System shutdown completed with {len(shutdown_errors)} errors:")
                for name, err in shutdown_errors:
                     self.logger.error(f"  - {name}: {err}")
            else:
                 self.logger.info("miniManus system components shutdown complete.")

            # Signal that shutdown is complete (even if there were errors)
            SystemManager._shutdown_event.set()
            self.logger.debug("Shutdown event set.")

            # If force_exit is True, exit the process
            if force_exit:
                exit_code = 1 if shutdown_errors else 0 # Exit with error code if shutdown had issues
                self.logger.info(f"Exiting process with code {exit_code}...")
                sys.exit(exit_code) # Use sys.exit for cleaner exit than os._exit if possible

    def _handle_signal(self, signum: int, frame) -> None:
        """
        Handle system signals (SIGINT, SIGTERM).

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        # Log signal name safely
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = f"Signal {signum}"
        self.logger.info(f"Received signal {sig_name}")

        if signum in (signal.SIGINT, signal.SIGTERM):
            self.logger.info("Initiating graceful shutdown from signal handler...")

            # If shutdown already running, give it a moment, then force exit if still needed
            if self.is_shutting_down:
                 self.logger.warning("Shutdown already in progress. Forcing exit.")
                 os._exit(1) # Force immediate exit if interrupted during shutdown

            # Start shutdown in a separate thread to avoid blocking signal handler
            # Pass force_exit=False initially to allow timeout check
            shutdown_thread = threading.Thread(target=self.shutdown, kwargs={"force_exit": False}, name="ShutdownThread")
            shutdown_thread.daemon = True
            shutdown_thread.start()

            # Wait for shutdown completion *or* timeout within the signal handler's context
            # This waiting is crucial for graceful exit but can be tricky
            if not SystemManager._shutdown_event.wait(self._shutdown_timeout):
                self.logger.warning(f"Graceful shutdown timed out after {self._shutdown_timeout} seconds. Forcing exit.")
                os._exit(1) # Force exit if shutdown doesn't complete within timeout
            else:
                 self.logger.info("Graceful shutdown completed successfully.")
                 sys.exit(0) # Clean exit after successful shutdown

    def run_forever(self) -> None:
        """Run the system until shutdown is requested."""
        if not self.startup_complete:
            if not self.startup():
                 self.logger.critical("System startup failed. Aborting run_forever.")
                 # Ensure shutdown is called even if startup partially failed
                 if not self.is_shutting_down:
                      self.shutdown(force_exit=True)
                 return # Exit if startup failed

        self.logger.info("miniManus system running... Press Ctrl+C to exit.")

        try:
            # Keep the main thread alive by waiting on the shutdown event
            # This is better than time.sleep() as it responds immediately to shutdown
            SystemManager._shutdown_event.wait()
            self.logger.info("Shutdown event received, run_forever loop ending.")

        except KeyboardInterrupt:
            # This might still happen if the signal handler setup failed
            self.logger.info("KeyboardInterrupt caught in run_forever (Signal handler might have failed). Initiating shutdown.")
            if not self.is_shutting_down:
                # Initiate shutdown directly if signal handler didn't catch it
                self.shutdown(force_exit=True) # Force exit might be appropriate here
        finally:
            # Final check to ensure shutdown is called if loop exits unexpectedly
            if not self.is_shutting_down:
                 self.logger.warning("run_forever loop exited unexpectedly. Ensuring shutdown...")
                 self.shutdown(force_exit=True)

# Example usage (can be run standalone for basic testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for testing SystemManager

    system = SystemManager.get_instance()

    class DummyComponent:
        def __init__(self, name, sleep_time=0.1):
             self.name = name
             self.sleep_time = sleep_time
             self._logger = logging.getLogger(f"Dummy.{name}")

        def startup(self):
            self._logger.info(f"{self.name} started")

        def shutdown(self):
            self._logger.info(f"{self.name} shutting down...")
            time.sleep(self.sleep_time) # Simulate shutdown work
            self._logger.info(f"{self.name} shutdown complete.")

    class HangingComponent(DummyComponent):
         def shutdown(self):
             self._logger.info(f"{self.name} shutting down... and hanging!")
             time.sleep(15) # Simulate a long hang during shutdown
             self._logger.info(f"{self.name} finally finished hanging shutdown.")


    # Register components
    system.register_component("CompA", DummyComponent("A"))
    system.register_component("CompB", DummyComponent("B", sleep_time=0.5))
    # system.register_component("Hanging", HangingComponent("Hanging")) # Uncomment to test timeout
    system.register_component("CompC", DummyComponent("C"))


    # Start and run
    system.run_forever()

    print("Main thread finished.") # Should only print after shutdown
