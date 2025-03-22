#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resource Monitor for miniManus

This module implements the Resource Monitor component, which tracks memory and CPU usage,
implements adaptive resource allocation, provides early warnings before resource exhaustion,
and triggers cleanup procedures when needed.
"""

import os
import sys
import time
import logging
import threading
import psutil
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum, auto

# Import local modules
try:
    from ..core.event_bus import EventBus, Event, EventPriority
except ImportError:
    # For standalone testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.event_bus import EventBus, Event, EventPriority

logger = logging.getLogger("miniManus.ResourceMonitor")

class ResourceWarningLevel(Enum):
    """Warning levels for resource usage."""
    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()

class ResourceType(Enum):
    """Types of resources to monitor."""
    MEMORY = auto()
    CPU = auto()
    STORAGE = auto()
    BATTERY = auto()

class ResourceMonitor:
    """
    ResourceMonitor tracks system resource usage and provides warnings when thresholds are exceeded.
    
    It handles:
    - Memory usage monitoring
    - CPU usage monitoring
    - Storage space monitoring
    - Battery level monitoring (if available)
    - Warning notifications when thresholds are exceeded
    - Triggering cleanup procedures when needed
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
        
        # Default thresholds (will be updated from config)
        self.thresholds = {
            ResourceType.MEMORY: {
                ResourceWarningLevel.WARNING: 30,    # 30% usage
                ResourceWarningLevel.CRITICAL: 40,   # 40% usage
                ResourceWarningLevel.EMERGENCY: 45,  # 45% usage (Android OOM killer at ~50%)
            },
            ResourceType.CPU: {
                ResourceWarningLevel.WARNING: 70,    # 70% usage
                ResourceWarningLevel.CRITICAL: 85,   # 85% usage
                ResourceWarningLevel.EMERGENCY: 95,  # 95% usage
            },
            ResourceType.STORAGE: {
                ResourceWarningLevel.WARNING: 80,    # 80% usage
                ResourceWarningLevel.CRITICAL: 90,   # 90% usage
                ResourceWarningLevel.EMERGENCY: 95,  # 95% usage
            },
            ResourceType.BATTERY: {
                ResourceWarningLevel.WARNING: 30,    # 30% remaining
                ResourceWarningLevel.CRITICAL: 15,   # 15% remaining
                ResourceWarningLevel.EMERGENCY: 5,   # 5% remaining
            },
        }
        
        # Current warning levels
        self.current_levels = {
            ResourceType.MEMORY: ResourceWarningLevel.NORMAL,
            ResourceType.CPU: ResourceWarningLevel.NORMAL,
            ResourceType.STORAGE: ResourceWarningLevel.NORMAL,
            ResourceType.BATTERY: ResourceWarningLevel.NORMAL,
        }
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread = None
        self.monitor_interval = 5.0  # seconds
        self.cleanup_callbacks = {
            ResourceType.MEMORY: [],
            ResourceType.CPU: [],
            ResourceType.STORAGE: [],
            ResourceType.BATTERY: [],
        }
        
        # Battery monitoring capability
        self.battery_monitoring_available = self._check_battery_monitoring()
        
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
        except (AttributeError, NotImplementedError):
            return False
    
    def update_thresholds(self, resource_type: ResourceType, 
                         warning: Optional[float] = None, 
                         critical: Optional[float] = None,
                         emergency: Optional[float] = None) -> None:
        """
        Update thresholds for a specific resource type.
        
        Args:
            resource_type: Type of resource to update thresholds for
            warning: New warning threshold (percentage)
            critical: New critical threshold (percentage)
            emergency: New emergency threshold (percentage)
        """
        if warning is not None:
            self.thresholds[resource_type][ResourceWarningLevel.WARNING] = warning
        
        if critical is not None:
            self.thresholds[resource_type][ResourceWarningLevel.CRITICAL] = critical
        
        if emergency is not None:
            self.thresholds[resource_type][ResourceWarningLevel.EMERGENCY] = emergency
        
        self.logger.debug(f"Updated thresholds for {resource_type.name}")
    
    def register_cleanup_callback(self, resource_type: ResourceType, 
                                 callback: Callable[[ResourceWarningLevel], None]) -> None:
        """
        Register a callback to be called when cleanup is needed for a resource.
        
        Args:
            resource_type: Type of resource to register callback for
            callback: Function to call when cleanup is needed
        """
        self.cleanup_callbacks[resource_type].append(callback)
        self.logger.debug(f"Registered cleanup callback for {resource_type.name}")
    
    def unregister_cleanup_callback(self, resource_type: ResourceType, 
                                   callback: Callable[[ResourceWarningLevel], None]) -> bool:
        """
        Unregister a cleanup callback.
        
        Args:
            resource_type: Type of resource to unregister callback for
            callback: Function to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        if callback in self.cleanup_callbacks[resource_type]:
            self.cleanup_callbacks[resource_type].remove(callback)
            self.logger.debug(f"Unregistered cleanup callback for {resource_type.name}")
            return True
        return False
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """
        Get current memory usage.
        
        Returns:
            Tuple of (used_percent, available_bytes)
        """
        mem = psutil.virtual_memory()
        return mem.percent, mem.available
    
    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage.
        
        Returns:
            CPU usage percentage
        """
        return psutil.cpu_percent(interval=0.1)
    
    def get_storage_usage(self) -> Tuple[float, float]:
        """
        Get current storage usage for the Termux data directory.
        
        Returns:
            Tuple of (used_percent, free_bytes)
        """
        # Get storage info for the directory containing user data
        home_dir = os.environ.get('HOME', '.')
        disk = psutil.disk_usage(home_dir)
        return disk.percent, disk.free
    
    def get_battery_status(self) -> Tuple[Optional[float], Optional[bool]]:
        """
        Get current battery status.
        
        Returns:
            Tuple of (percent, is_charging) or (None, None) if not available
        """
        if not self.battery_monitoring_available:
            return None, None
        
        try:
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent, battery.power_plugged
            return None, None
        except (AttributeError, NotImplementedError):
            self.battery_monitoring_available = False
            return None, None
    
    def get_resource_status(self) -> Dict[ResourceType, Dict[str, Any]]:
        """
        Get status of all monitored resources.
        
        Returns:
            Dictionary with status of all resources
        """
        status = {}
        
        # Memory status
        mem_percent, mem_available = self.get_memory_usage()
        status[ResourceType.MEMORY] = {
            "percent": mem_percent,
            "available": mem_available,
            "warning_level": self.current_levels[ResourceType.MEMORY].name,
        }
        
        # CPU status
        cpu_percent = self.get_cpu_usage()
        status[ResourceType.CPU] = {
            "percent": cpu_percent,
            "warning_level": self.current_levels[ResourceType.CPU].name,
        }
        
        # Storage status
        storage_percent, storage_free = self.get_storage_usage()
        status[ResourceType.STORAGE] = {
            "percent": storage_percent,
            "free": storage_free,
            "warning_level": self.current_levels[ResourceType.STORAGE].name,
        }
        
        # Battery status
        if self.battery_monitoring_available:
            battery_percent, is_charging = self.get_battery_status()
            status[ResourceType.BATTERY] = {
                "percent": battery_percent,
                "is_charging": is_charging,
                "warning_level": self.current_levels[ResourceType.BATTERY].name,
            }
        
        return status
    
    def check_resource_levels(self) -> Dict[ResourceType, ResourceWarningLevel]:
        """
        Check current resource levels and determine warning levels.
        
        Returns:
            Dictionary mapping resource types to warning levels
        """
        new_levels = {}
        
        # Check memory
        mem_percent, _ = self.get_memory_usage()
        if mem_percent >= self.thresholds[ResourceType.MEMORY][ResourceWarningLevel.EMERGENCY]:
            new_levels[ResourceType.MEMORY] = ResourceWarningLevel.EMERGENCY
        elif mem_percent >= self.thresholds[ResourceType.MEMORY][ResourceWarningLevel.CRITICAL]:
            new_levels[ResourceType.MEMORY] = ResourceWarningLevel.CRITICAL
        elif mem_percent >= self.thresholds[ResourceType.MEMORY][ResourceWarningLevel.WARNING]:
            new_levels[ResourceType.MEMORY] = ResourceWarningLevel.WARNING
        else:
            new_levels[ResourceType.MEMORY] = ResourceWarningLevel.NORMAL
        
        # Check CPU
        cpu_percent = self.get_cpu_usage()
        if cpu_percent >= self.thresholds[ResourceType.CPU][ResourceWarningLevel.EMERGENCY]:
            new_levels[ResourceType.CPU] = ResourceWarningLevel.EMERGENCY
        elif cpu_percent >= self.thresholds[ResourceType.CPU][ResourceWarningLevel.CRITICAL]:
            new_levels[ResourceType.CPU] = ResourceWarningLevel.CRITICAL
        elif cpu_percent >= self.thresholds[ResourceType.CPU][ResourceWarningLevel.WARNING]:
            new_levels[ResourceType.CPU] = ResourceWarningLevel.WARNING
        else:
            new_levels[ResourceType.CPU] = ResourceWarningLevel.NORMAL
        
        # Check storage
        storage_percent, _ = self.get_storage_usage()
        if storage_percent >= self.thresholds[ResourceType.STORAGE][ResourceWarningLevel.EMERGENCY]:
            new_levels[ResourceType.STORAGE] = ResourceWarningLevel.EMERGENCY
        elif storage_percent >= self.thresholds[ResourceType.STORAGE][ResourceWarningLevel.CRITICAL]:
            new_levels[ResourceType.STORAGE] = ResourceWarningLevel.CRITICAL
        elif storage_percent >= self.thresholds[ResourceType.STORAGE][ResourceWarningLevel.WARNING]:
            new_levels[ResourceType.STORAGE] = ResourceWarningLevel.WARNING
        else:
            new_levels[ResourceType.STORAGE] = ResourceWarningLevel.NORMAL
        
        # Check battery
        if self.battery_monitoring_available:
            battery_percent, is_charging = self.get_battery_status()
            if battery_percent is not None:
                # Only check battery level if not charging
                if not is_charging:
                    if battery_percent <= self.thresholds[ResourceType.BATTERY][ResourceWarningLevel.EMERGENCY]:
                        new_levels[ResourceType.BATTERY] = ResourceWarningLevel.EMERGENCY
                    elif battery_percent <= self.thresholds[ResourceType.BATTERY][ResourceWarningLevel.CRITICAL]:
                        new_levels[ResourceType.BATTERY] = ResourceWarningLevel.CRITICAL
                    elif battery_percent <= self.thresholds[ResourceType.BATTERY][ResourceWarningLevel.WARNING]:
                        new_levels[ResourceType.BATTERY] = ResourceWarningLevel.WARNING
                    else:
                        new_levels[ResourceType.BATTERY] = ResourceWarningLevel.NORMAL
                else:
                    new_levels[ResourceType.BATTERY] = ResourceWarningLevel.NORMAL
        
        return new_levels
    
    def _handle_warning_level_change(self, resource_type: ResourceType, 
                                    old_level: ResourceWarningLevel, 
                                    new_level: ResourceWarningLevel) -> None:
        """
        Handle a change in warning level for a resource.
        
        Args:
            resource_type: Type of resource with changed warning level
            old_level: Previous warning level
            new_level: New warning level
        """
        # Only handle increases in warning level or return to normal
        if new_level.value > old_level.value or new_level == ResourceWarningLevel.NORMAL:
            # Publish event
            self.event_bus.publish_event(
                f"resource.{resource_type.name.lower()}.{new_level.name.lower()}",
                {
                    "resource_type": resource_type,
                    "warning_level": new_level,
                    "previous_level": old_level,
                },
                EventPriority.HIGH if new_level == ResourceWarningLevel.EMERGENCY else EventPriority.NORMAL
            )
            
            # Call cleanup callbacks if warning level is not NORMAL
            if new_level != ResourceWarningLevel.NORMAL:
                for callback in self.cleanup_callbacks[resource_type]:
                    try:
                        callback(new_level)
                    except Exception as e:
                        self.logger.error(f"Error in cleanup callback for {resource_type.name}: {str(e)}")
    
    def _monitor_resources(self) -> None:
        """Monitor resources periodically."""
        while self.is_running:
            try:
                # Check resource levels
                new_levels = self.check_resource_levels()
                
                # Handle changes in warning levels
                for resource_type, new_level in new_levels.items():
                    old_level = self.current_levels[resource_type]
                    if new_level != old_level:
                        self._handle_warning_level_change(resource_type, old_level, new_level)
                        self.current_levels[resource_type] = new_level
                
                # Adjust monitoring interval based on resource pressure
                max_level = max(level.value for level in new_levels.values())
                if max_level >= ResourceWarningLevel.CRITICAL.value:
                    # More frequent monitoring under pressure
                    interval = 1.0
                elif max_level >= ResourceWarningLevel.WARNING.value:
                    interval = 2.0
                else:
                    interval = self.monitor_interval
                
                # Sleep until next check
                time.sleep(interval)
            
            except Exception as e:
                self.logger.error(f"Error monitoring resources: {str(e)}")
                time.sleep(5.0)  # Sleep longer on error
    
    def startup(self) -> None:
        """Start resource monitoring."""
        if self.is_running:
            self.logger.warning("ResourceMonitor already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        self.logger.info("ResourceMonitor started")
    
    def shutdown(self) -> None:
        """Stop resource monitoring."""
        self.is_running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        self.logger.info("ResourceMonitor stopped")

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    # Initialize EventBus first
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    # Subscribe to resource events
    def handle_resource_event(event):
        print(f"Resource event: {event.event_type} - {event.data}")
    
    event_bus.subscribe("resource.memory.warning", handle_resource_event)
    event_bus.subscribe("resource.memory.critical", handle_resource_event)
    event_bus.subscribe("resource.memory.emergency", handle_resource_event)
    
    # Initialize and start ResourceMonitor
    monitor = ResourceMonitor.get_instance()
    
    # Register a cleanup callback
    def memory_cleanup(warning_level):
        print(f"Memory cleanup triggered at level: {warning_level.name}")
    
    monitor.register_cleanup_callback(ResourceType.MEMORY, memory_cleanup)
    
    # Start monitoring
    monitor.startup()
    
    # Print current resource status
    status = monitor.get_resource_status()
    for resource_type, info in status.items():
        print(f"{resource_type.name}: {info}")
    
    # Run for a while
    try:
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    # Shutdown
    monitor.shutdown()
    event_bus.shutdown()
