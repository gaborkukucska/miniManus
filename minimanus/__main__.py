#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for miniManus

This module serves as the main entry point for the miniManus framework.
"""

import os
import sys
import logging
import asyncio
import signal
import time
from pathlib import Path

# Configure logging
log_dir = os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus', 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'minimanus.log'), mode='a')
    ]
)

logger = logging.getLogger("miniManus")

# Import local modules
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

# Global variables for signal handling
system_manager = None
shutdown_requested = False

async def main_async():
    """Main asynchronous function."""
    global system_manager, shutdown_requested
    
    logger.info("Starting miniManus...")
    
    # Initialize event bus
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    # Initialize error handler
    error_handler = ErrorHandler.get_instance()
    
    # Initialize configuration manager
    config_manager = ConfigurationManager.get_instance()
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor.get_instance()
    resource_monitor.startup()
    
    # Initialize plugin manager
    plugin_manager = PluginManager.get_instance()
    plugin_manager.startup()
    
    # Initialize API manager
    api_manager = APIManager.get_instance()
    api_manager.startup()
    
    # Initialize UI manager
    ui_manager = UIManager.get_instance()
    ui_manager.startup()
    
    # Initialize chat interface
    chat_interface = ChatInterface.get_instance()
    chat_interface.startup()
    
    # Initialize settings panel
    settings_panel = SettingsPanel.get_instance()
    settings_panel.startup()
    
    # Initialize model selection interface
    model_selection = ModelSelectionInterface.get_instance()
    model_selection.startup()
    
    # Initialize system manager
    system_manager = SystemManager.get_instance()
    system_manager.startup()
    
    # Discover models - Fixed: properly await the coroutine
    await model_selection.discover_models()
    
    logger.info("miniManus started successfully")
    
    # Keep the main thread alive until shutdown is requested
    while not shutdown_requested:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Main loop cancelled, initiating shutdown...")
            break

def main():
    """Main function."""
    global system_manager, shutdown_requested
    
    # We don't register signal handlers here anymore as they're handled by SystemManager
    
    try:
        # Create necessary directories
        os.makedirs(os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus', 'config'), exist_ok=True)
        os.makedirs(os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus', 'data'), exist_ok=True)
        os.makedirs(os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus', 'plugins'), exist_ok=True)
        
        # Run the async main function
        loop = asyncio.get_event_loop()
        main_task = loop.create_task(main_async())
        
        try:
            loop.run_until_complete(main_task)
        except KeyboardInterrupt:
            # This should be handled by SystemManager's signal handler,
            # but we add this as a fallback
            logger.info("KeyboardInterrupt received in main loop")
            shutdown_requested = True
            
            # Cancel the main task
            main_task.cancel()
            
            try:
                # Wait for the task to be cancelled
                loop.run_until_complete(main_task)
            except asyncio.CancelledError:
                pass
            
            # Ensure system manager shutdown is called
            if system_manager and not system_manager.is_shutting_down:
                logger.info("Initiating shutdown from main...")
                system_manager.shutdown(force_exit=True)
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        if 'error_handler' in locals():
            error_handler.handle_error(
                e, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL,
                {"action": "main"}
            )
        else:
            logger.exception("Unhandled exception")
        sys.exit(1)

if __name__ == "__main__":
    main()
