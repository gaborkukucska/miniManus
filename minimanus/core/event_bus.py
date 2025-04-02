#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event Bus for miniManus

This module implements the Event Bus component, which provides a publish-subscribe
pattern for system events, decouples components for better modularity, and enables
asynchronous communication between modules.
"""

import logging
import threading
import queue
import time
from typing import Dict, List, Callable, Any, Optional, Set, Tuple

from enum import Enum, auto

logger = logging.getLogger("miniManus.EventBus")

class EventPriority(Enum):
    """Priority levels for events."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

class Event:
    """
    Base class for all events in the system.
    """
    
    def __init__(self, event_type: str, data: Any = None, priority: EventPriority = EventPriority.NORMAL):
        """
        Initialize a new event.
        
        Args:
            event_type: Type identifier for the event
            data: Optional data associated with the event
            priority: Priority level for the event
        """
        self.event_type = event_type
        self.data = data
        self.priority = priority
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        """String representation of the event."""
        return f"Event(type={self.event_type}, priority={self.priority.name}, data={self.data})"
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the event data.
        
        Args:
            key: Key to get from data
            default: Default value to return if key not found
            
        Returns:
            Value for the key if found, otherwise default
        """
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default

class EventBus:
    """
    EventBus implements a publish-subscribe pattern for system events.
    
    It handles:
    - Event subscription and unsubscription
    - Event publishing
    - Asynchronous event processing
    - Priority-based event handling
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'EventBus':
        """Get or create the singleton instance of EventBus."""
        if cls._instance is None:
            cls._instance = EventBus()
        return cls._instance
    
    def __init__(self):
        """Initialize the EventBus."""
        if EventBus._instance is not None:
            raise RuntimeError("EventBus is a singleton. Use get_instance() instead.")
        
        self.logger = logger
        self.subscribers: Dict[str, List[Tuple[Callable[[Event], None], bool]]] = {}
        self.event_queue = queue.PriorityQueue()
        self.is_running = False
        self.worker_thread = None
        self._subscribers_lock = threading.RLock()
        self._event_counter = 0  # Add counter for tiebreaking
        self._counter_lock = threading.Lock()  # Lock for the counter
        
        self.logger.info("EventBus initialized")
    
    def subscribe(self, event_type: str, callback: Callable[[Event], None], synchronous: bool = False) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Function to call when an event of this type is published
            synchronous: If True, callback will be executed synchronously in the publishing thread
        """
        with self._subscribers_lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            
            # Check if already subscribed
            for existing_callback, _ in self.subscribers[event_type]:
                if existing_callback == callback:
                    self.logger.warning(f"Callback already subscribed to {event_type}")
                    return
            
            self.subscribers[event_type].append((callback, synchronous))
            self.logger.debug(f"Subscribed to {event_type} events")
    
    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> bool:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            callback: Function to unsubscribe
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        with self._subscribers_lock:
            if event_type not in self.subscribers:
                return False
            
            for i, (cb, _) in enumerate(self.subscribers[event_type]):
                if cb == callback:
                    self.subscribers[event_type].pop(i)
                    self.logger.debug(f"Unsubscribed from {event_type} events")
                    
                    # Clean up empty subscriber lists
                    if not self.subscribers[event_type]:
                        del self.subscribers[event_type]
                    
                    return True
            
            return False
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        event_type = event.event_type
        
        # Handle synchronous subscribers immediately
        with self._subscribers_lock:
            if event_type in self.subscribers:
                for callback, synchronous in self.subscribers[event_type]:
                    if synchronous:
                        try:
                            callback(event)
                        except Exception as e:
                            self.logger.error(f"Error in synchronous event handler: {str(e)}")
        
        # Queue event for asynchronous processing
        priority_value = 100 - event.priority.value  # Invert priority for queue (lower value = higher priority)
        
        # Use a counter as tiebreaker for events with the same priority
        with self._counter_lock:
            counter = self._event_counter
            self._event_counter += 1
        
        self.event_queue.put((priority_value, counter, event))
        
        self.logger.debug(f"Published event: {event}")
    
    def publish_event(self, event_type: str, data: Any = None, 
                     priority: EventPriority = EventPriority.NORMAL) -> None:
        """
        Create and publish an event.
        
        Args:
            event_type: Type of event to publish
            data: Data to include with the event
            priority: Priority level for the event
        """
        event = Event(event_type, data, priority)
        self.publish(event)
    
    def startup(self) -> None:
        """Start the event processing thread."""
        if self.is_running:
            self.logger.warning("EventBus already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self.worker_thread.start()
        self.logger.info("EventBus started")
    
    def shutdown(self) -> None:
        """Stop the event processing thread."""
        self.is_running = False
        
        # Add a sentinel event to unblock the queue
        self.event_queue.put((0, 0, None))
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        self.logger.info("EventBus stopped")
    
    def _process_events(self) -> None:
        """Process events from the queue."""
        while self.is_running:
            try:
                # Get the next event from the queue
                # Update to unpack three values instead of two
                _, _, event = self.event_queue.get(timeout=0.1)
                
                # Check for sentinel event
                if event is None:
                    continue
                
                event_type = event.event_type
                
                # Process event for asynchronous subscribers
                with self._subscribers_lock:
                    if event_type in self.subscribers:
                        for callback, synchronous in self.subscribers[event_type]:
                            if not synchronous:
                                try:
                                    callback(event)
                                except Exception as e:
                                    self.logger.error(f"Error in asynchronous event handler: {str(e)}")
                
                self.event_queue.task_done()
            
            except queue.Empty:
                # No events in queue, continue waiting
                continue
            except Exception as e:
                self.logger.error(f"Error processing events: {str(e)}")
    
    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """
        Get the number of subscribers for a specific event type or all event types.
        
        Args:
            event_type: Type of events to count subscribers for, or None for all
            
        Returns:
            Number of subscribers
        """
        with self._subscribers_lock:
            if event_type:
                return len(self.subscribers.get(event_type, []))
            else:
                return sum(len(subscribers) for subscribers in self.subscribers.values())

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    logging.basicConfig(level=logging.INFO)
    
    event_bus = EventBus.get_instance()
    event_bus.startup()
    
    # Define event handlers
    def handle_system_event(event):
        print(f"System event received: {event.data}")
    
    def handle_user_event(event):
        print(f"User event received: {event.data}")
    
    # Subscribe to events
    event_bus.subscribe("system", handle_system_event)
    event_bus.subscribe("user", handle_user_event, synchronous=True)
    
    # Publish events
    event_bus.publish_event("system", "System starting up", EventPriority.HIGH)
    event_bus.publish_event("user", "User logged in")
    
    # Wait for events to be processed
    time.sleep(1)
    
    # Unsubscribe and shutdown
    event_bus.unsubscribe("system", handle_system_event)
    event_bus.shutdown()
