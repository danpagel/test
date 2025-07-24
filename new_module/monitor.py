"""
Monitoring and System Management for MegaPythonLibrary.

This module contains:
- Event system and event management
- Logging configuration and management  
- Error handling and monitoring
- Performance monitoring and diagnostics
- System health checks
"""

import logging
import threading
import time
import traceback
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque

from .utils import RequestError

# ==============================================
# === LOGGING CONFIGURATION ===
# ==============================================

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO, 
                 format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s') -> None:
    """
    Set up logging configuration for MPL.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Log message format
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up new handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    # Configure MPL logger
    mpl_logger = logging.getLogger('mpl_merged')
    mpl_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"mpl.{name}")


# ==============================================
# === EVENT SYSTEM ===
# ==============================================

@dataclass
class EventInfo:
    """Information about a triggered event."""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None


class EventManager:
    """Simple event management system."""
    
    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._event_history: deque = deque(maxlen=1000)  # Keep last 1000 events
        self._stats = defaultdict(int)
        self.logger = get_logger("events")
    
    def on(self, event: str, callback: Callable) -> None:
        """Register an event callback."""
        with self._lock:
            if event not in self._callbacks:
                self._callbacks[event] = []
            self._callbacks[event].append(callback)
            self.logger.debug(f"Registered callback for event: {event}")
    
    def off(self, event: str, callback: Callable = None) -> None:
        """Remove event callback(s)."""
        with self._lock:
            if event in self._callbacks:
                if callback:
                    if callback in self._callbacks[event]:
                        self._callbacks[event].remove(callback)
                        self.logger.debug(f"Removed specific callback for event: {event}")
                else:
                    self._callbacks[event].clear()
                    self.logger.debug(f"Cleared all callbacks for event: {event}")
    
    def trigger(self, event: str, data: Dict[str, Any], source: str = None) -> None:
        """Trigger an event."""
        event_info = EventInfo(event, data, source=source)
        
        with self._lock:
            # Add to history
            self._event_history.append(event_info)
            self._stats[event] += 1
            
            # Trigger callbacks
            if event in self._callbacks:
                for callback in self._callbacks[event]:
                    try:
                        callback(event_info.data)
                    except Exception as e:
                        self.logger.warning(f"Event callback failed for {event}: {e}")
            
            self.logger.debug(f"Triggered event: {event} with data: {data}")
    
    def get_event_history(self, limit: int = 100) -> List[EventInfo]:
        """Get recent event history."""
        with self._lock:
            return list(self._event_history)[-limit:]
    
    def get_event_stats(self) -> Dict[str, int]:
        """Get event statistics."""
        with self._lock:
            return dict(self._stats)
    
    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()
            self._stats.clear()
            self.logger.info("Event history cleared")


# Global event manager
_event_manager = EventManager()


def on_event(event: str, callback: Callable) -> None:
    """Register a global event callback."""
    _event_manager.on(event, callback)


def off_event(event: str, callback: Callable = None) -> None:
    """Remove a global event callback."""
    _event_manager.off(event, callback)


def trigger_event(event: str, data: Dict[str, Any], source: str = None) -> None:
    """Trigger a global event."""
    _event_manager.trigger(event, data, source)


# ==============================================
# === PERFORMANCE MONITORING ===
# ==============================================

class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._lock = threading.RLock()
        self.logger = get_logger("performance")
    
    def record_metric(self, name: str, value: float, metadata: Dict[str, Any] = None) -> None:
        """Record a performance metric."""
        with self._lock:
            metric_data = {
                'value': value,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            self._metrics[name].append(metric_data)
            
            # Keep only last 1000 measurements per metric
            if len(self._metrics[name]) > 1000:
                self._metrics[name] = self._metrics[name][-1000:]
            
            self.logger.debug(f"Recorded metric {name}: {value}")
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            if name not in self._metrics or not self._metrics[name]:
                return {}
            
            values = [m['value'] for m in self._metrics[name]]
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': values[-1]
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        with self._lock:
            return {name: self.get_metric_stats(name) for name in self._metrics}
    
    def clear_metrics(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            self.logger.info("Performance metrics cleared")


# Global performance monitor
_performance_monitor = PerformanceMonitor()


def record_performance(name: str, value: float, metadata: Dict[str, Any] = None) -> None:
    """Record a performance metric."""
    _performance_monitor.record_metric(name, value, metadata)


def get_performance_stats(name: str = None) -> Dict[str, Any]:
    """Get performance statistics."""
    if name:
        return _performance_monitor.get_metric_stats(name)
    else:
        return _performance_monitor.get_all_metrics()


# ==============================================
# === ERROR MONITORING ===
# ==============================================

class ErrorMonitor:
    """Monitor and track errors."""
    
    def __init__(self):
        self._errors = deque(maxlen=1000)
        self._error_counts = defaultdict(int)
        self._lock = threading.RLock()
        self.logger = get_logger("errors")
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Record an error occurrence."""
        with self._lock:
            error_info = {
                'type': type(error).__name__,
                'message': str(error),
                'timestamp': time.time(),
                'context': context or {},
                'traceback': traceback.format_exc()
            }
            self._errors.append(error_info)
            self._error_counts[error_info['type']] += 1
            
            self.logger.error(f"Recorded error {error_info['type']}: {error_info['message']}")
            trigger_event('error_occurred', error_info)
    
    def get_recent_errors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent errors."""
        with self._lock:
            return list(self._errors)[-limit:]
    
    def get_error_counts(self) -> Dict[str, int]:
        """Get error counts by type."""
        with self._lock:
            return dict(self._error_counts)
    
    def clear_errors(self) -> None:
        """Clear error history."""
        with self._lock:
            self._errors.clear()
            self._error_counts.clear()
            self.logger.info("Error history cleared")


# Global error monitor
_error_monitor = ErrorMonitor()


def record_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Record an error for monitoring."""
    _error_monitor.record_error(error, context)


def get_error_history(limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent error history."""
    return _error_monitor.get_recent_errors(limit)


def get_error_counts() -> Dict[str, int]:
    """Get error counts by type."""
    return _error_monitor.get_error_counts()


# ==============================================
# === DIAGNOSTIC FUNCTIONS ===
# ==============================================

def get_system_health() -> Dict[str, Any]:
    """Get overall system health status."""
    return {
        'timestamp': time.time(),
        'event_stats': _event_manager.get_event_stats(),
        'performance_stats': _performance_monitor.get_all_metrics(),
        'error_counts': _error_monitor.get_error_counts(),
        'recent_events_count': len(_event_manager.get_event_history()),
        'total_errors': sum(_error_monitor.get_error_counts().values())
    }


def clear_all_monitoring_data() -> None:
    """Clear all monitoring data."""
    _event_manager.clear_history()
    _performance_monitor.clear_metrics()
    _error_monitor.clear_errors()
    logger.info("All monitoring data cleared")


# ==============================================
# === CONTEXT MANAGERS FOR MONITORING ===
# ==============================================

class MonitoredOperation:
    """Context manager for monitoring operations."""
    
    def __init__(self, operation_name: str, emit_events: bool = True):
        self.operation_name = operation_name
        self.emit_events = emit_events
        self.start_time = None
        self.logger = get_logger("operations")
    
    def __enter__(self):
        self.start_time = time.time()
        if self.emit_events:
            trigger_event(f"{self.operation_name}_started", {'operation': self.operation_name})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        record_performance(f"{self.operation_name}_duration", duration)
        
        if exc_type is None:
            # Success
            if self.emit_events:
                trigger_event(f"{self.operation_name}_completed", {
                    'operation': self.operation_name,
                    'duration': duration
                })
            self.logger.info(f"{self.operation_name} completed in {duration:.2f}s")
        else:
            # Error occurred
            if exc_val:
                record_error(exc_val, {'operation': self.operation_name})
            if self.emit_events:
                trigger_event(f"{self.operation_name}_failed", {
                    'operation': self.operation_name,
                    'error': str(exc_val) if exc_val else 'Unknown error',
                    'duration': duration
                })
            self.logger.error(f"{self.operation_name} failed after {duration:.2f}s: {exc_val}")
        
        return False  # Don't suppress exceptions


# ==============================================
# === FINALIZATION ===
# ==============================================

# Set up final logging configuration if needed
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info("MPL Monitoring system initialized")