"""
Event System Module for Client
==============================

This module provides a comprehensive event system for the MegaPythonLibrary,
enabling event-driven architecture throughout the application.

Features:
- Event callback registration and management
- Thread-safe event triggering
- Event namespace support
- Wildcard event listeners
- Event history and debugging
- Performance monitoring

Author: Extracted and enhanced for modular architecture
"""

import logging
import threading
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import weakref

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================
# === EVENT SYSTEM CLASSES ===
# ==============================================

@dataclass
class EventInfo:
    """Information about a triggered event."""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event info to dictionary."""
        return {
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source
        }


class EventCallback:
    """Wrapper for event callbacks with metadata."""
    
    def __init__(self, callback: Callable, once: bool = False, priority: int = 0):
        self.callback = callback
        self.once = once
        self.priority = priority
        self.call_count = 0
        self.created_at = datetime.now()
        
    def __call__(self, data: Dict[str, Any]) -> Any:
        """Execute the callback."""
        self.call_count += 1
        return self.callback(data)
    
    def __repr__(self) -> str:
        return f"EventCallback({self.callback.__name__}, calls={self.call_count})"


class EventManager:
    """Thread-safe event management system."""
    
    def __init__(self, enable_history: bool = True, max_history: int = 1000):
        self._callbacks: Dict[str, List[EventCallback]] = {}
        self._lock = threading.RLock()
        self._enable_history = enable_history
        self._max_history = max_history
        self._event_history: List[EventInfo] = []
        self._stats = {
            'events_triggered': 0,
            'callbacks_executed': 0,
            'errors_encountered': 0
        }
        
    def on(self, event: str, callback: Callable, once: bool = False, priority: int = 0) -> None:
        """
        Register an event callback.
        
        Args:
            event: Event name (supports wildcards with *)
            callback: Function to call when event occurs
            once: If True, callback is removed after first execution
            priority: Higher priority callbacks execute first
        """
        with self._lock:
            if event not in self._callbacks:
                self._callbacks[event] = []
            
            event_callback = EventCallback(callback, once, priority)
            self._callbacks[event].append(event_callback)
            
            # Sort by priority (highest first)
            self._callbacks[event].sort(key=lambda x: x.priority, reverse=True)
            
            logger.debug(f"Registered callback for event '{event}': {callback.__name__}")
    
    def once(self, event: str, callback: Callable, priority: int = 0) -> None:
        """Register a one-time event callback."""
        self.on(event, callback, once=True, priority=priority)
    
    def off(self, event: str, callback: Callable = None) -> int:
        """
        Remove event callback(s).
        
        Args:
            event: Event name
            callback: Specific callback to remove (None to remove all)
            
        Returns:
            Number of callbacks removed
        """
        with self._lock:
            if event not in self._callbacks:
                return 0
            
            removed_count = 0
            if callback is None:
                # Remove all callbacks for this event
                removed_count = len(self._callbacks[event])
                self._callbacks[event].clear()
            else:
                # Remove specific callback
                original_count = len(self._callbacks[event])
                self._callbacks[event] = [
                    cb for cb in self._callbacks[event] 
                    if cb.callback != callback
                ]
                removed_count = original_count - len(self._callbacks[event])
            
            logger.debug(f"Removed {removed_count} callback(s) for event '{event}'")
            return removed_count
    
    def trigger(self, event: str, data: Dict[str, Any] = None, source: str = None) -> int:
        """
        Trigger an event and execute all registered callbacks.
        
        Args:
            event: Event name
            data: Event data to pass to callbacks
            source: Source identifier for debugging
            
        Returns:
            Number of callbacks executed
        """
        if data is None:
            data = {}
        
        event_info = EventInfo(event, data, source=source)
        
        with self._lock:
            self._stats['events_triggered'] += 1
            
            # Add to history
            if self._enable_history:
                self._event_history.append(event_info)
                if len(self._event_history) > self._max_history:
                    self._event_history.pop(0)
            
            # Find matching event patterns
            matching_events = []
            for event_pattern in self._callbacks:
                if self._event_matches(event, event_pattern):
                    matching_events.append(event_pattern)
            
            callbacks_executed = 0
            callbacks_to_remove = []
            
            # Execute callbacks for matching events
            for event_pattern in matching_events:
                for callback_wrapper in self._callbacks[event_pattern][:]:  # Copy to avoid modification during iteration
                    try:
                        callback_wrapper(data)
                        callbacks_executed += 1
                        self._stats['callbacks_executed'] += 1
                        
                        # Remove one-time callbacks
                        if callback_wrapper.once:
                            callbacks_to_remove.append((event_pattern, callback_wrapper))
                            
                    except Exception as e:
                        self._stats['errors_encountered'] += 1
                        logger.error(f"Event callback failed for '{event}': {e}")
            
            # Remove one-time callbacks
            for event_pattern, callback_wrapper in callbacks_to_remove:
                if callback_wrapper in self._callbacks[event_pattern]:
                    self._callbacks[event_pattern].remove(callback_wrapper)
            
            logger.debug(f"Event '{event}' triggered: {callbacks_executed} callbacks executed")
            return callbacks_executed
    
    def _event_matches(self, event: str, pattern: str) -> bool:
        """Check if an event matches a pattern (supports wildcards)."""
        if pattern == "*":
            return True
        if "*" not in pattern:
            return event == pattern
        
        # Simple wildcard matching
        parts = pattern.split("*")
        if not event.startswith(parts[0]):
            return False
        
        current_pos = len(parts[0])
        for part in parts[1:-1]:
            if part not in event[current_pos:]:
                return False
            current_pos = event.find(part, current_pos) + len(part)
        
        return event[current_pos:].endswith(parts[-1]) if parts[-1] else True
    
    def list_events(self) -> List[str]:
        """Get list of registered event patterns."""
        with self._lock:
            return list(self._callbacks.keys())
    
    def get_callback_count(self, event: str = None) -> int:
        """Get number of registered callbacks for an event or total."""
        with self._lock:
            if event:
                return len(self._callbacks.get(event, []))
            return sum(len(callbacks) for callbacks in self._callbacks.values())
    
    def get_event_history(self, event_type: str = None, limit: int = 100) -> List[EventInfo]:
        """Get recent event history."""
        with self._lock:
            history = self._event_history[-limit:] if limit else self._event_history
            if event_type:
                return [event for event in history if event.event_type == event_type]
            return history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics."""
        with self._lock:
            return {
                **self._stats,
                'registered_events': len(self._callbacks),
                'total_callbacks': self.get_callback_count(),
                'history_size': len(self._event_history)
            }
    
    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()
    
    def clear_all(self) -> None:
        """Clear all callbacks and history."""
        with self._lock:
            self._callbacks.clear()
            self._event_history.clear()
            logger.info("Event system cleared")


# ==============================================
# === GLOBAL EVENT MANAGER ===
# ==============================================

# Global event manager instance
_global_event_manager = EventManager()


# ==============================================
# === CONVENIENCE FUNCTIONS ===
# ==============================================

def on(event: str, callback: Callable, once: bool = False, priority: int = 0) -> None:
    """Register an event callback using global event manager."""
    _global_event_manager.on(event, callback, once, priority)


def once(event: str, callback: Callable, priority: int = 0) -> None:
    """Register a one-time event callback using global event manager."""
    _global_event_manager.once(event, callback, priority)


def off(event: str, callback: Callable = None) -> int:
    """Remove event callback using global event manager."""
    return _global_event_manager.off(event, callback)


def trigger(event: str, data: Dict[str, Any] = None, source: str = None) -> int:
    """Trigger an event using global event manager."""
    return _global_event_manager.trigger(event, data, source)


def get_event_manager() -> EventManager:
    """Get the global event manager instance."""
    return _global_event_manager


# ==============================================
# === CLIENT INTEGRATION METHODS ===
# ==============================================

def add_event_methods(client_class):
    """Add event system methods to the MPLClient class."""
    
    def on_method(self, event: str, callback: Callable) -> None:
        """Register event callback."""
        if not hasattr(self, '_event_manager'):
            self._event_manager = EventManager()
        self._event_manager.on(event, callback)
    
    def off_method(self, event: str, callback: Callable = None) -> None:
        """Remove event callback."""
        if hasattr(self, '_event_manager'):
            self._event_manager.off(event, callback)
    
    def _trigger_event_method(self, event: str, data: Dict[str, Any]) -> None:
        """Trigger event callbacks."""
        if hasattr(self, '_event_manager'):
            self._event_manager.trigger(event, data, source='MPLClient')
    
    def get_event_stats_method(self) -> Dict[str, Any]:
        """Get event system statistics."""
        if hasattr(self, '_event_manager'):
            return self._event_manager.get_stats()
        return {'events_triggered': 0, 'callbacks_executed': 0, 'errors_encountered': 0}
    
    def clear_event_history_method(self) -> None:
        """Clear event history."""
        if hasattr(self, '_event_manager'):
            self._event_manager.clear_history()
    
    # Add methods to client class
    setattr(client_class, 'on', on_method)
    setattr(client_class, 'off', off_method)
    setattr(client_class, '_trigger_event', _trigger_event_method)
    setattr(client_class, 'get_event_stats', get_event_stats_method)
    setattr(client_class, 'clear_event_history', clear_event_history_method)


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Classes
    'EventManager',
    'EventInfo',
    'EventCallback',
    
    # Global functions
    'on',
    'once', 
    'off',
    'trigger',
    'get_event_manager',
    
    # Integration
    'add_event_methods',
]
