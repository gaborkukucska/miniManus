#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for miniManus.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add parent directory to path to allow running as script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Import miniManus modules
from minimanus.core.system_manager import SystemManager
from minimanus.core.config_manager import ConfigurationManager
from minimanus.core.event_bus import EventBus
from minimanus.core.resource_monitor import ResourceMonitor
from minimanus.core.error_handler import ErrorHandler
from minimanus.core.plugin_manager import PluginManager
from minimanus.api.api_manager import APIManager
from minimanus.ui.ui_manager import UIManager
from minimanus.ui.chat_interface import ChatInterface
from minimanus.ui.settings_panel import SettingsPanel
from minimanus.ui.model_selection import ModelSelectionInterface

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus', 'logs', 'minimanus.log'), mode='a')
    ]
)

logger = logging.getLogger("miniManus")

async def main_async():
    """Main async function to start miniManus."""
    logger.info("Starting miniManus...")
    
    try:
        # Create necessary directories
        data_dir = os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus')
        os.makedirs(os.path.join(data_dir, 'config'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'plugins'), exist_ok=True)
        
        # Initialize core components
        event_bus = EventBus.get_instance()
        event_bus.startup()
        
        error_handler = ErrorHandler.get_instance()
        
        config_manager = ConfigurationManager.get_instance()
        
        resource_monitor = ResourceMonitor.get_instance()
        resource_monitor.startup()
        
        plugin_manager = PluginManager.get_instance()
        plugin_manager.startup()
        
        # Initialize API components
        api_manager = APIManager.get_instance()
        api_manager.startup()
        
        # Initialize UI components
        ui_manager = UIManager.get_instance()
        ui_manager.startup()
        
        chat_interface = ChatInterface.get_instance()
        chat_interface.startup()
        
        settings_panel = SettingsPanel.get_instance()
        settings_panel.startup()
        
        model_selection = ModelSelectionInterface.get_instance()
        model_selection.startup()
        
        # Initialize system manager (must be last)
        system_manager = SystemManager.get_instance()
        system_manager.startup()
        
        # Discover models
        await model_selection.discover_models()
        
        # Start UI loop
        logger.info("miniManus started successfully")
        
        # Keep running until shutdown
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}", exc_info=True)
    finally:
        # Shutdown in reverse order
        system_manager.shutdown()
        
        model_selection.shutdown()
        settings_panel.shutdown()
        chat_interface.shutdown()
        ui_manager.shutdown()
        
        api_manager.shutdown()
        
        plugin_manager.shutdown()
        resource_monitor.shutdown()
        event_bus.shutdown()
        
        logger.info("miniManus shutdown complete")

def main():
    """Main entry point for miniManus."""
    try:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.environ.get('HOME', '.'), '.local', 'share', 'minimanus', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Run the async main function
        asyncio.run(main_async())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
