# START OF FILE miniManus-main/minimanus/__main__.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for miniManus

This module serves as the main entry point for the miniManus framework,
initializing core components and managing the application lifecycle.
"""

import os
import sys
import logging
import asyncio
import signal
import time
from pathlib import Path
from typing import Optional, Tuple, List # Added List for Tuple hint fix

# Define the base directory for miniManus data, logs, and config
BASE_DIR = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'minimanus'
LOG_DIR = BASE_DIR / 'logs'
CONFIG_DIR = BASE_DIR / 'config'
DATA_DIR = BASE_DIR / 'data'
PLUGINS_DIR = BASE_DIR / 'plugins'

# Create necessary directories early
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure secure permissions on config dir
    try: os.chmod(CONFIG_DIR, 0o700)
    except OSError as e: print(f"Warning: Could not set permissions on {CONFIG_DIR}: {e}", file=sys.stderr)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    print(f"Error creating necessary directories: {e}", file=sys.stderr)
    # Depending on severity, you might want to exit here
    sys.exit(1)

# Configure logging
# Note: Log level might be adjusted later based on config
logging.basicConfig(
    level=logging.INFO, # Default level, can be changed by config
    format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s', # Added level name width
    handlers=[
        logging.StreamHandler(sys.stdout), # Log to console
        logging.FileHandler(LOG_DIR / 'minimanus.log', mode='a') # Log to file
    ]
)

logger = logging.getLogger("miniManus")

# Import local modules (handle potential import errors during early startup)
try:
    from minimanus.core.system_manager import SystemManager
    from minimanus.core.event_bus import EventBus
    from minimanus.core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from minimanus.core.config_manager import ConfigurationManager
    from minimanus.core.resource_monitor import ResourceMonitor
    from minimanus.core.plugin_manager import PluginManager
    from minimanus.api.api_manager import APIManager
    from minimanus.ui.ui_manager import UIManager
    from minimanus.ui.chat_interface import ChatInterface
    from minimanus.ui.settings_panel import SettingsPanel
    from minimanus.ui.model_selection import ModelSelectionInterface
    from minimanus.core.agent_system import AgentSystem # Ensure AgentSystem is imported
except ImportError as e:
    logger.critical(f"Failed to import core modules: {e}", exc_info=True)
    sys.exit(f"ImportError: {e}. Please ensure miniManus is installed correctly or run from the correct directory.")

# Global variables for signal handling and main loop control
system_manager: Optional[SystemManager] = None
shutdown_requested: bool = False
main_loop_task: Optional[asyncio.Task] = None

async def main_async():
    """Main asynchronous function to initialize and start components."""
    global system_manager, shutdown_requested # Declare intent to modify globals here

    logger.info(f"Starting miniManus...")
    logger.info(f"Using base directory: {BASE_DIR}")

    # Order of initialization matters for dependencies
    try:
        # --- DETAILED STARTUP LOGGING ---
        logger.debug("Initializing EventBus...")
        event_bus = EventBus.get_instance()
        event_bus.startup()
        logger.debug("EventBus initialized and started.")

        logger.debug("Initializing ErrorHandler...")
        error_handler = ErrorHandler.get_instance()
        logger.debug("ErrorHandler initialized.")

        logger.debug("Initializing ConfigManager...")
        config_manager = ConfigurationManager.get_instance()
        config_manager.config_dir = CONFIG_DIR
        config_manager.secrets_file = CONFIG_DIR / 'secrets.json'
        logger.debug("ConfigManager initialized.")

        # Adjust log level based on loaded config
        log_level_str = config_manager.get_config("general.log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        # Configure root logger and miniManus logger specifically
        logging.getLogger().setLevel(log_level) # Set root logger level
        logging.getLogger("miniManus").setLevel(log_level) # Ensure our logger has the right level
        for handler in logging.getLogger().handlers: # Apply to all handlers
             handler.setLevel(log_level)
        logger.info(f"Logging level set to {log_level_str}")

        logger.debug("Initializing ResourceMonitor...")
        resource_monitor = ResourceMonitor.get_instance()
        resource_monitor.startup()
        logger.debug("ResourceMonitor initialized and started.")

        logger.debug("Initializing PluginManager...")
        plugin_manager = PluginManager.get_instance()
        plugin_manager.plugin_dirs = []
        plugin_manager.add_plugin_directory(str(PLUGINS_DIR))
        plugin_manager.startup()
        logger.debug("PluginManager initialized and started.")

        # API Layer
        logger.debug("Initializing APIManager...")
        api_manager = APIManager.get_instance()
        api_manager.startup() # Initializes adapters
        logger.debug("APIManager initialized and started.")

        # Agent System
        logger.debug("Initializing AgentSystem...")
        agent_system = AgentSystem.get_instance() # Initialize AgentSystem
        logger.debug("AgentSystem initialized.")

        # UI Layer
        logger.info("--- Initializing ChatInterface ---")
        chat_interface = ChatInterface.get_instance()
        chat_interface.sessions_dir = DATA_DIR / 'sessions'
        chat_interface.startup() # Loads sessions, potentially creates default
        logger.info("--- ChatInterface startup() completed ---")

        # --- ADDED LOGGING ---
        logger.info("--- Initializing ModelSelectionInterface ---")
        model_selection = ModelSelectionInterface.get_instance()
        logger.info("--- ModelSelectionInterface instance obtained ---")
        model_selection.startup() # Loads prefs
        logger.info("--- ModelSelectionInterface startup() completed ---")

        # --- ADDED LOGGING ---
        logger.info("--- Initializing SettingsPanel ---")
        settings_panel = SettingsPanel.get_instance()
        logger.info("--- SettingsPanel instance obtained ---")
        settings_panel.startup() # Registers settings definitions
        logger.info("--- SettingsPanel startup() completed ---")

        # --- ADDED LOGGING ---
        logger.info("--- Initializing UIManager ---")
        ui_manager = UIManager.get_instance()
        logger.info("--- UIManager instance obtained ---")
        main_file_dir = Path(__file__).parent
        potential_static_dir = main_file_dir / 'static'
        if potential_static_dir.is_dir():
             ui_manager.static_dir = str(potential_static_dir)
             logger.debug(f"Using static directory: {ui_manager.static_dir}")
        else:
             logger.warning(f"Static directory not found at {potential_static_dir}, using default: {ui_manager.static_dir}")
        ui_manager.startup() # Starts web server thread
        logger.info("--- UIManager startup() completed (Web server thread started) ---")

        # System Manager Registration
        logger.debug("Registering components with SystemManager...")
        if system_manager:
            system_manager.register_component("event_bus", event_bus)
            system_manager.register_component("config_manager", config_manager)
            system_manager.register_component("resource_monitor", resource_monitor)
            system_manager.register_component("plugin_manager", plugin_manager)
            system_manager.register_component("api_manager", api_manager)
            system_manager.register_component("agent_system", agent_system)
            system_manager.register_component("chat_interface", chat_interface)
            system_manager.register_component("model_selection", model_selection)
            system_manager.register_component("settings_panel", settings_panel)
            system_manager.register_component("ui_manager", ui_manager)
            system_manager.startup_complete = True
            system_manager.is_running = True
            logger.debug("Components registered.")
        else:
             logger.error("SystemManager instance is None during component registration!")
             raise RuntimeError("SystemManager instance is unexpectedly None in main_async.")
        # --- END DETAILED STARTUP LOGGING ---

        # Post-startup actions that require the event loop
        logger.info("Performing post-startup actions...")
        await model_selection.discover_models()
        logger.debug("Model discovery complete.")

        logger.info("miniManus started successfully.")
        logger.info(f"Access the UI at http://{ui_manager.host}:{ui_manager.port}")

        # Keep the main async task alive until shutdown is requested
        while not shutdown_requested:
            await asyncio.sleep(0.5) # Check shutdown flag periodically

        logger.info("Shutdown requested, exiting main_async loop.")

    except Exception as e:
        # Log with full traceback for any error during startup
        logger.critical(f"Critical error during async startup: {e}", exc_info=True)
        # Ensure shutdown is attempted even if startup fails
        # No need for 'global system_manager' here
        if system_manager and not system_manager.is_shutting_down:
             logger.error("Attempting emergency shutdown due to startup error...")
             system_manager.shutdown(force_exit=True) # Force exit if startup failed critically
        else:
            # If system_manager wasn't even assigned yet or already shutting down, just exit
            logger.error("SystemManager not available for emergency shutdown or already shutting down.")
            sys.exit(1)

def main():
    """Main synchronous entry point."""
    global system_manager, shutdown_requested, main_loop_task # Declare globals used in this function

    # SystemManager now handles signal registration in its __init__
    # Ensure SystemManager is instantiated early to catch signals
    try:
        system_manager = SystemManager.get_instance()
    except Exception as e:
        logger.critical(f"Failed to initialize SystemManager: {e}", exc_info=True)
        sys.exit(1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    main_task_cancelled = False
    exit_code = 0

    try:
        logger.debug("Creating main async task...")
        main_loop_task = loop.create_task(main_async())
        logger.debug("Running event loop until main task completes...")
        loop.run_until_complete(main_loop_task)
        logger.debug("Main async task completed normally.")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main. Requesting shutdown...")
        shutdown_requested = True # Signal async loops to stop
        if main_loop_task and not main_loop_task.done():
            logger.info("Cancelling main async task...")
            main_loop_task.cancel()
            try:
                # Allow cancellation to propagate - run loop briefly
                # Use gather to wait for the task and handle CancelledError
                logger.debug("Running loop briefly to process cancellation...")
                # Increased timeout for cancellation processing
                loop.run_until_complete(asyncio.wait_for(asyncio.gather(main_loop_task, return_exceptions=True), timeout=5.0))
                logger.info("Main async task successfully cancelled or finished.")
                main_task_cancelled = True
            except asyncio.CancelledError:
                 logger.info("Main async task explicitly cancelled.")
                 main_task_cancelled = True
            except asyncio.TimeoutError:
                 logger.warning("Timeout waiting for main task cancellation to complete.")
                 # Proceed with shutdown anyway
            except Exception as e:
                logger.error(f"Error occurred while awaiting cancelled main task: {e}", exc_info=True)
                exit_code = 1 # Indicate error during cancellation handling
        else:
             logger.info("Main async task already completed or does not exist.")

    except asyncio.CancelledError:
         # This might happen if cancellation occurs elsewhere
         logger.info("Main loop task was cancelled externally.")
         main_task_cancelled = True
    except Exception as e:
        # Log critical error with traceback
        logger.critical(f"Unhandled error during main asyncio execution: {e}", exc_info=True)
        exit_code = 1 # Critical error before shutdown initiated
    finally:
        logger.info("Entering final shutdown phase...")
        # Ensure system manager shutdown is called *after* the main loop is stopped/cancelled
        if system_manager:
            if not system_manager.is_shutting_down:
                logger.info("Initiating SystemManager shutdown...")
                try:
                    # Run synchronous shutdown (handles its own logging/timeout)
                    # Pass force_exit=False as we handle exit below
                    system_manager.shutdown(force_exit=False)
                    logger.info("SystemManager shutdown sequence completed.")
                except Exception as shutdown_e:
                     logger.error(f"Error during SystemManager shutdown call: {shutdown_e}", exc_info=True)
                     exit_code = 1 # Error during shutdown
            else:
                logger.info("SystemManager shutdown was already in progress or completed.")
        else:
             logger.warning("SystemManager not available for final shutdown call.")

        # Close the loop gracefully
        logger.debug("Closing asyncio event loop...")
        try:
            # Cancel any remaining tasks
            tasks = asyncio.all_tasks(loop)
            # Filter out the main loop task if it's already finished/cancelled
            tasks = {task for task in tasks if task is not main_loop_task}

            if tasks:
                 logger.debug(f"Cancelling {len(tasks)} remaining asyncio tasks...")
                 for task in tasks:
                     task.cancel()
                 # Give cancelled tasks a moment to finish
                 # Use return_exceptions=True to prevent gather from stopping on first CancelledError
                 loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                 logger.debug("Remaining tasks cancelled.")

            # Stop the loop if running (needed before closing)
            if loop.is_running():
                logger.debug("Stopping event loop...")
                loop.stop()

            # Close the loop if not already closed
            if not loop.is_closed():
                 logger.debug("Closing event loop...")
                 loop.close()
                 logger.info("Asyncio event loop closed.")
            else:
                 logger.debug("Event loop already closed.")
        except Exception as loop_close_e:
            logger.error(f"Error closing asyncio loop: {loop_close_e}", exc_info=True)
            if exit_code == 0: exit_code = 1 # Indicate error if none previously set

        logger.info(f"miniManus finished with exit code {exit_code}.")
        # Use sys.exit here instead of letting SystemManager handle it
        sys.exit(exit_code)

# Entry point
if __name__ == "__main__":
    main()
# END OF FILE miniManus-main/minimanus/__main__.py
