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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    print(f"Error creating necessary directories: {e}", file=sys.stderr)
    # Depending on severity, you might want to exit here
    # sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
    sys.exit(f"ImportError: {e}. Please ensure miniManus is installed correctly.")

# Global variables for signal handling and main loop control
system_manager: Optional[SystemManager] = None
shutdown_requested: bool = False
main_loop_task: Optional[asyncio.Task] = None

async def main_async():
    """Main asynchronous function to initialize and start components."""
    global system_manager, shutdown_requested

    logger.info(f"Starting miniManus...")
    logger.info(f"Using base directory: {BASE_DIR}")

    # Order of initialization matters for dependencies
    try:
        # Core services
        event_bus = EventBus.get_instance()
        event_bus.startup()

        error_handler = ErrorHandler.get_instance() # ErrorHandler often needs EventBus

        config_manager = ConfigurationManager.get_instance() # ConfigManager might use ErrorHandler
        config_manager.config_dir = CONFIG_DIR # Ensure correct path is used
        config_manager.secrets_file = CONFIG_DIR / 'secrets.json'
        config_manager._load_config() # Explicitly load after path is set
        config_manager._load_secrets()

        resource_monitor = ResourceMonitor.get_instance()
        resource_monitor.startup()

        plugin_manager = PluginManager.get_instance()
        # Set plugin directories explicitly
        plugin_manager.plugin_dirs = [] # Clear defaults if any
        plugin_manager.add_plugin_directory(str(PLUGINS_DIR))
        # Add other potential plugin locations if needed
        plugin_manager.startup()

        # API Layer
        api_manager = APIManager.get_instance()
        api_manager.startup()

        # Agent System
        agent_system = AgentSystem.get_instance() # Initialize AgentSystem

        # UI Layer
        ui_manager = UIManager.get_instance()
        ui_manager.static_dir = str(BASE_DIR.parent.parent / 'minimanus' / 'static') # Adjust if needed based on install location
        ui_manager.startup()

        chat_interface = ChatInterface.get_instance()
        chat_interface.sessions_dir = DATA_DIR / 'sessions' # Ensure correct path
        chat_interface.startup()

        settings_panel = SettingsPanel.get_instance()
        settings_panel.startup()

        model_selection = ModelSelectionInterface.get_instance()
        model_selection.startup()

        # System Manager (Last, as it coordinates others)
        system_manager = SystemManager.get_instance()
        # Register components with SystemManager (optional, depends on its design)
        # system_manager.register_component("event_bus", event_bus)
        # ... register other components ...
        system_manager.startup() # Starts components if they have a startup method and are registered

        # Post-startup actions
        logger.info("Discovering models...")
        await model_selection.discover_models()

        logger.info("miniManus started successfully.")
        logger.info(f"Access the UI at http://{ui_manager.host}:{ui_manager.port}")

        # Keep the main async task alive until shutdown is requested
        while not shutdown_requested:
            await asyncio.sleep(1)

    except Exception as e:
        logger.critical(f"Critical error during async startup: {e}", exc_info=True)
        # Attempt graceful shutdown if system_manager is available
        if system_manager:
            system_manager.shutdown(force_exit=True)
        else:
            sys.exit(1) # Exit if system manager wasn't even initialized

def main():
    """Main synchronous entry point."""
    global system_manager, shutdown_requested, main_loop_task

    # Note: Signal handlers are now managed by SystemManager's _handle_signal

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        main_loop_task = loop.create_task(main_async())
        loop.run_until_complete(main_loop_task)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main. Requesting shutdown...")
        shutdown_requested = True
        if main_loop_task:
            main_loop_task.cancel() # Request cancellation of the main async loop
            try:
                # Give the loop a chance to process the cancellation
                loop.run_until_complete(main_loop_task)
            except asyncio.CancelledError:
                logger.debug("Main async task cancelled.")
            except Exception as e:
                logger.error(f"Error during main task cancellation processing: {e}", exc_info=True)

        # Ensure system manager shutdown is called *after* the main loop is stopped/cancelled
        if system_manager and not system_manager.is_shutting_down:
            logger.info("Initiating shutdown from main KeyboardInterrupt handler...")
            # SystemManager's signal handler should take over, but this is a fallback
            # We don't call force_exit=True here directly, let the signal handler manage timeouts
            system_manager.shutdown()
            # Wait for the shutdown event managed by SystemManager
            SystemManager._shutdown_event.wait() # Wait for graceful shutdown
        elif not system_manager:
             logger.warning("SystemManager not available for shutdown.")

    except asyncio.CancelledError:
         logger.info("Main loop was cancelled.")
    except Exception as e:
        logger.critical(f"Unhandled error in main execution: {e}", exc_info=True)
        if system_manager and not system_manager.is_shutting_down:
            system_manager.shutdown(force_exit=True)
        else:
            sys.exit(1) # Exit if critical error happened before/during shutdown
    finally:
        if loop.is_running():
            loop.close()
            logger.info("Asyncio event loop closed.")
        logger.info("miniManus finished.")

if __name__ == "__main__":
    main()
