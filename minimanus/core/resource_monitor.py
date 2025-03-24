#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resource Monitor for miniManus

This module implements the Resource Monitor component, which monitors system resources
and provides resource management capabilities for the miniManus framework.
"""

import os
import sys
import logging
import time
import threading
import psutil
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum, auto

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
    from ..core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from ..core.config_manager import ConfigurationManager
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
    from core.config_manager import ConfigurationManager

logger = logging.getLogger("miniManus.ResourceMonitor")

class ResourceType(Enum):
    """Resource types."""
    MEMORY = auto()
    CPU = auto()
    STORAGE = auto()
    BATTERY = auto()

class ResourceWarningLevel(Enum):
    """Resource warning severity levels."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()

class ResourceMonitor:
    """
    ResourceMonitor monitors system resources and provides resource management capabilities.
    
    It handles:
    - Memory usage monitoring
    - CPU usage monitoring
    - Storage usage monitoring
    - Battery status monitoring
    - Resource cleanup
    - Resource warning notifications
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'ResourceMonitor':
        """Get or create the singleton instance of ResourceMonitor."""
        if cls._instance is None:
            cls._instance = ResourceMonitor()
        return cls._instance
    
    def __init__(self):
        """Initialize the ResourceMonitor."""
        if ResourceMonitor._instance is not None:
            raise RuntimeError("ResourceMonitor is a singleton. Use get_instance() instead.")
        
        self.logger = logger
        self.event_bus = EventBus.get_instance()
        self.error_handler = ErrorHandler.get_instance()
        self.config_manager = ConfigurationManager.get_instance()
        
        # Resource thresholds
        self.memory_warning_threshold = self.config_manager.get_config(
            "resources.memory_warning_threshold", 
            80
        )  # Percentage
        self.memory_critical_threshold = self.config_manager.get_config(
            "resources.memory_critical_threshold", 
            90
        )  # Percentage
        self.cpu_warning_threshold = self.config_manager.get_config(
            "resources.cpu_warning_threshold", 
            80
        )  # Percentage
        self.cpu_critical_threshold = self.config_manager.get_config(
            "resources.cpu_critical_threshold", 
            90
        )  # Percentage
        self.storage_warning_threshold = self.config_manager.get_config(
            "resources.storage_warning_threshold", 
            80
        )  # Percentage
        self.storage_critical_threshold = self.config_manager.get_config(
            "resources.storage_critical_threshold", 
            90
        )  # Percentage
        self.battery_warning_threshold = self.config_manager.get_config(
            "resources.battery_warning_threshold", 
            20
        )  # Percentage
        self.battery_critical_threshold = self.config_manager.get_config(
            "resources.battery_critical_threshold", 
            10
        )  # Percentage
        
        # Monitoring intervals
        self.monitoring_interval = self.config_manager.get_config(
            "resources.monitoring_interval", 
            60
        )  # Seconds
        
        # Cleanup callbacks
        self.cleanup_callbacks = []
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Feature availability
        self.battery_monitoring_available = self._check_battery_monitoring()
        self.cpu_monitoring_available = self._check_cpu_monitoring()
        
        self.logger.info("ResourceMonitor initialized")
    
    def _check_battery_monitoring(self) -> bool:
        """
        Check if battery monitoring is available.
        
        Returns:
            True if battery monitoring is available, False otherwise
        """
        try:
            battery = psutil.sensors_battery()
            return battery is not None
        except (AttributeError, PermissionError, FileNotFoundError):
            self.logger.warning("Battery monitoring not available due to permissions or missing hardware")
            return False
    
    def _check_cpu_monitoring(self) -> bool:
        """
        Check if CPU monitoring is available.
        
        Returns:
            True if CPU monitoring is available, False otherwise
        """
        try:
            # Just try to access /proc/stat directly to check permissions
            with open('/proc/stat', 'rb') as f:
                pass
            return True
        except (PermissionError, FileNotFoundError):
            self.logger.warning("CPU monitoring not available due to permissions or missing hardware")
            return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
        except (AttributeError, PermissionError, FileNotFoundError):
            self.logger.warning("Memory usage monitoring not available due to permissions")
            return {
                "total": 0,
                "available": 0,
                "used": 0,
                "percent": 0
            }
    
    def get_cpu_usage(self) -> Dict[str, Any]:
        """
        Get current CPU usage.
        
        Returns:
            Dictionary with CPU usage information
        """
        try:
            if not self.cpu_monitoring_available:
                return {
                    "percent": 0,
                    "count": 1
                }
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            return {
                "percent": cpu_percent,
                "count": cpu_count
            }
        except (AttributeError, PermissionError, FileNotFoundError):
            self.logger.warning("CPU usage monitoring not available due to permissions")
            self.cpu_monitoring_available = False
            return {
                "percent": 0,
                "count": 1
            }
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get current storage usage.
        
        Returns:
            Dictionary with storage usage information
        """
        try:
            storage = psutil.disk_usage(os.path.expanduser("~"))
            return {
                "total": storage.total,
                "used": storage.used,
                "free": storage.free,
                "percent": storage.percent
            }
        except (AttributeError, PermissionError, FileNotFoundError):
            self.logger.warning("Storage usage monitoring not available due to permissions")
            return {
                "total": 0,
                "used": 0,
                "free": 0,
                "percent": 0
            }
    
    def get_battery_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current battery status.
        
        Returns:
            Dictionary with battery status information or None if not available
        """
        if not self.battery_monitoring_available:
            return None
        
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return None
            
            return {
                "percent": battery.percent,
                "power_plugged": battery.power_plugged,
                "seconds_left": battery.secsleft if battery.secsleft != -1 else None
            }
        except (AttributeError, PermissionError, FileNotFoundError):
            self.battery_monitoring_available = False
            self.logger.warning("Battery monitoring became unavailable")
            return None
    
    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback for resource cleanup.
        
        Args:
            callback: Callback function to register
        """
        self.cleanup_callbacks.append(callback)
        self.logger.debug(f"Registered cleanup callback: {callback.__name__}")
    
    def unregister_cleanup_callback(self, callback: Callable[[], None]) -> bool:
        """
        Unregister a cleanup callback.
        
        Args:
            callback: Callback function to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if callback in self.cleanup_callbacks:
            self.cleanup_callbacks.remove(callback)
            self.logger.debug(f"Unregistered cleanup callback: {callback.__name__}")
            return True
        return False
    
    def check_resources(self) -> List[Dict[str, Any]]:
        """
        Check resource usage and trigger warnings if needed.
        
        Returns:
            List of resource warnings
        """
        warnings = []
        
        # Check memory
        memory = self.get_memory_usage()
        if memory["percent"] >= self.memory_critical_threshold:
            level = ResourceWarningLevel.CRITICAL
        elif memory["percent"] >= self.memory_warning_threshold:
            level = ResourceWarningLevel.WARNING
        else:
            level = ResourceWarningLevel.INFO
        
        if level != ResourceWarningLevel.INFO:
            warning = {
                "type": ResourceType.MEMORY,
                "level": level,
                "value": memory["percent"],
                "threshold": self.memory_critical_threshold if level == ResourceWarningLevel.CRITICAL else self.memory_warning_threshold,
                "message": f"Memory usage is {memory['percent']}%"
            }
            warnings.append(warning)
            self._publish_resource_warning(warning)
        
        # Check CPU
        if self.cpu_monitoring_available:
            cpu = self.get_cpu_usage()
            if cpu["percent"] >= self.cpu_critical_threshold:
                level = ResourceWarningLevel.CRITICAL
            elif cpu["percent"] >= self.cpu_warning_threshold:
                level = ResourceWarningLevel.WARNING
            else:
                level = ResourceWarningLevel.INFO
            
            if level != ResourceWarningLevel.INFO:
                warning = {
                    "type": ResourceType.CPU,
                    "level": level,
                    "value": cpu["percent"],
                    "threshold": self.cpu_critical_threshold if level == ResourceWarningLevel.CRITICAL else self.cpu_warning_threshold,
                    "message": f"CPU usage is {cpu['percent']}%"
                }
                warnings.append(warning)
                self._publish_resource_warning(warning)
        
        # Check storage
        storage = self.get_storage_usage()
        if storage["percent"] >= self.storage_critical_threshold:
            level = ResourceWarningLevel.CRITICAL
        elif storage["percent"] >= self.storage_warning_threshold:
            level = ResourceWarningLevel.WARNING
        else:
            level = ResourceWarningLevel.INFO
        
        if level != ResourceWarningLevel.INFO:
            warning = {
                "type": ResourceType.STORAGE,
                "level": level,
                "value": storage["percent"],
                "threshold": self.storage_critical_threshold if level == ResourceWarningLevel.CRITICAL else self.storage_warning_threshold,
                "message": f"Storage usage is {storage['percent']}%"
            }
            warnings.append(warning)
            self._publish_resource_warning(warning)
        
        # Check battery
        if self.battery_monitoring_available:
            battery = self.get_battery_status()
            if battery and not battery["power_plugged"]:
                if battery["percent"] <= self.battery_critical_threshold:
                    level = ResourceWarningLevel.CRITICAL
                elif battery["percent"] <= self.battery_warning_threshold:
                    level = ResourceWarningLevel.WARNING
                else:
                    level = ResourceWarningLevel.INFO
                
                if level != ResourceWarningLevel.INFO:
                    warning = {
                        "type": ResourceType.BATTERY,
                        "level": level,
                        "value": battery["percent"],
                        "threshold": self.battery_critical_threshold if level == ResourceWarningLevel.CRITICAL else self.battery_warning_threshold,
                        "message": f"Battery level is {battery['percent']}%"
                    }
                    warnings.append(warning)
                    self._publish_resource_warning(warning)
        
        return warnings
    
    def _publish_resource_warning(self, warning: Dict[str, Any]) -> None:
        """
        Publish a resource warning event.
        
        Args:
            warning: Warning information
        """
        self.event_bus.publish_event("resources.warning", warning)
        
        level_name = warning["level"].name.lower()
        resource_name = warning["type"].name.lower()
        self.logger.warning(f"Resource {level_name} for {resource_name}: {warning['message']}")
        
        # Trigger cleanup if critical
        if warning["level"] == ResourceWarningLevel.CRITICAL:
            self._trigger_cleanup(warning["type"])
    
    def _trigger_cleanup(self, resource_type: ResourceType) -> None:
        """
        Trigger resource cleanup.
        
        Args:
            resource_type: Type of resource to clean up
        """
        self.logger.info(f"Triggering cleanup for {resource_type.name}")
        
        # Call cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.error_handler.handle_error(
                    e, ErrorCategory.RESOURCE, ErrorSeverity.WARNING,
                    {"action": "cleanup", "resource_type": resource_type.name}
                )
        
        # Publish cleanup event
        self.event_bus.publish_event("resources.cleanup", {
            "type": resource_type.name
        })
    
    def _monitoring_loop(self) -> None:
        """Resource monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                self.check_resources()
            except Exception as e:
                self.error_handler.handle_error(
                    e, ErrorCategory.RESOURCE, ErrorSeverity.WARNING,
                    {"action": "monitoring_loop"}
                )
            
            # Wait for next check
            self.stop_monitoring.wait(self.monitoring_interval)
    
    def startup(self) -> None:
        """Start the resource monitor."""
        # Start monitoring thread
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitorThread"
        )
        self.monitoring_thread.start()
        
        self.logger.info("ResourceMonitor started")
    
    def shutdown(self) -> None:
        """Stop the resource monitor."""
        # Stop monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=1.0)
        
        self.logger.info("ResourceMonitor stopped")

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize required components
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    error_handler = ErrorHandler.get_instance()
    
    config_manager = ConfigurationManager.get_instance()
    
    # Initialize ResourceMonitor
    resource_monitor = ResourceMonitor.get_instance()
    resource_monitor.startup()
    
    # Example usage
    print(f"Memory usage: {resource_monitor.get_memory_usage()}")
    print(f"CPU usage: {resource_monitor.get_cpu_usage()}")
    print(f"Storage usage: {resource_monitor.get_storage_usage()}")
    print(f"Battery status: {resource_monitor.get_battery_status()}")
    
    # Register cleanup callback
    def cleanup_example():
        print("Cleaning up resources...")
    
    resource_monitor.register_cleanup_callback(cleanup_example)
    
    # Check resources
    warnings = resource_monitor.check_resources()
    print(f"Resource warnings: {warnings}")
    
    # Wait a bit
    time.sleep(5)
    
    # Shutdown
    resource_monitor.shutdown()
    event_bus.shutdown()
