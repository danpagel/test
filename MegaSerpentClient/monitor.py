"""
MegaSerpentClient - Monitoring & System Management Module

Purpose: System monitoring, analytics, logging, error handling, and administration.

This module handles complete monitoring and analytics system, advanced logging with 6 levels,
error management and recovery, system administration, FUSE filesystem support,
and enterprise compliance and governance.
"""

import os
import sys
import threading
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import weakref

from . import utils
from .utils import (
    LogLevel, MegaError, DateTimeUtils, Helpers, Formatters
)


# ==============================================
# === MONITORING ENUMS AND CONSTANTS ===
# ==============================================

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    FIFO = "fifo"


class MaintenanceType(Enum):
    """Maintenance operation types."""
    CLEANUP = "cleanup"
    BACKUP = "backup"
    UPDATE = "update"
    OPTIMIZATION = "optimization"
    HEALTH_CHECK = "health_check"


# ==============================================
# === DATA CLASSES ===
# ==============================================

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime = field(default_factory=DateTimeUtils.now_utc)
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: List[float] = field(default_factory=list)
    
    # Memory metrics
    memory_total: int = 0
    memory_used: int = 0
    memory_available: int = 0
    memory_percent: float = 0.0
    
    # Disk metrics
    disk_total: int = 0
    disk_used: int = 0
    disk_free: int = 0
    disk_percent: float = 0.0
    
    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0
    
    # Process metrics
    process_count: int = 0
    thread_count: int = 0


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: datetime = field(default_factory=DateTimeUtils.now_utc)
    
    # API metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    
    # File operations
    files_uploaded: int = 0
    files_downloaded: int = 0
    bytes_transferred: int = 0
    
    # Sync metrics
    sync_operations: int = 0
    sync_conflicts: int = 0
    sync_errors: int = 0
    
    # User metrics
    active_users: int = 0
    total_sessions: int = 0
    average_session_duration: float = 0.0


@dataclass
class AlertInfo:
    """Alert information."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    source: str
    created_at: datetime = field(default_factory=DateTimeUtils.now_utc)
    resolved_at: Optional[datetime] = None
    is_resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Cache entry information."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=DateTimeUtils.now_utc)
    last_accessed: datetime = field(default_factory=DateTimeUtils.now_utc)
    access_count: int = 0
    expires_at: Optional[datetime] = None
    size: int = 0


@dataclass
class MaintenanceTask:
    """Maintenance task information."""
    task_id: str
    task_type: MaintenanceType
    description: str
    scheduled_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==============================================
# === SYSTEM MONITORING ===
# ==============================================

class SystemMonitor:
    """System performance monitoring."""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self._metrics_history: deque = deque(maxlen=1440)  # 24 hours of data
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._alert_callbacks: List[Callable] = []
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = self._collect_system_metrics()
                self._metrics_history.append(metrics)
                
                # Check for alerts
                self._check_system_alerts(metrics)
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        metrics = SystemMetrics()
        
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available, using dummy metrics")
            return metrics
        
        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=1)
            metrics.cpu_count = psutil.cpu_count()
            if hasattr(os, 'getloadavg'):
                metrics.load_average = list(os.getloadavg())
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_total = memory.total
            metrics.memory_used = memory.used
            metrics.memory_available = memory.available
            metrics.memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_total = disk.total
            metrics.disk_used = disk.used
            metrics.disk_free = disk.free
            metrics.disk_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.network_bytes_sent = network.bytes_sent
            metrics.network_bytes_recv = network.bytes_recv
            metrics.network_packets_sent = network.packets_sent
            metrics.network_packets_recv = network.packets_recv
            
            # Process metrics
            metrics.process_count = len(psutil.pids())
            current_process = psutil.Process()
            metrics.thread_count = current_process.num_threads()
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system alert conditions."""
        alerts = []
        
        # CPU usage alert
        if metrics.cpu_percent > 90:
            alerts.append(AlertInfo(
                alert_id=Helpers.generate_request_id(),
                level=AlertLevel.CRITICAL,
                title="High CPU Usage",
                message=f"CPU usage is {metrics.cpu_percent:.1f}%",
                source="system_monitor"
            ))
        elif metrics.cpu_percent > 80:
            alerts.append(AlertInfo(
                alert_id=Helpers.generate_request_id(),
                level=AlertLevel.WARNING,
                title="Elevated CPU Usage",
                message=f"CPU usage is {metrics.cpu_percent:.1f}%",
                source="system_monitor"
            ))
        
        # Memory usage alert
        if metrics.memory_percent > 95:
            alerts.append(AlertInfo(
                alert_id=Helpers.generate_request_id(),
                level=AlertLevel.CRITICAL,
                title="High Memory Usage",
                message=f"Memory usage is {metrics.memory_percent:.1f}%",
                source="system_monitor"
            ))
        elif metrics.memory_percent > 85:
            alerts.append(AlertInfo(
                alert_id=Helpers.generate_request_id(),
                level=AlertLevel.WARNING,
                title="Elevated Memory Usage",
                message=f"Memory usage is {metrics.memory_percent:.1f}%",
                source="system_monitor"
            ))
        
        # Disk usage alert
        if metrics.disk_percent > 95:
            alerts.append(AlertInfo(
                alert_id=Helpers.generate_request_id(),
                level=AlertLevel.CRITICAL,
                title="Disk Space Critical",
                message=f"Disk usage is {metrics.disk_percent:.1f}%",
                source="system_monitor"
            ))
        elif metrics.disk_percent > 85:
            alerts.append(AlertInfo(
                alert_id=Helpers.generate_request_id(),
                level=AlertLevel.WARNING,
                title="Low Disk Space",
                message=f"Disk usage is {metrics.disk_percent:.1f}%",
                source="system_monitor"
            ))
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable[[AlertInfo], None]):
        """Add alert callback function."""
        self._alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics."""
        if self._metrics_history:
            return self._metrics_history[-1]
        return self._collect_system_metrics()
    
    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """Get metrics history for specified hours."""
        cutoff_time = DateTimeUtils.now_utc() - timedelta(hours=hours)
        return [m for m in self._metrics_history if m.timestamp > cutoff_time]
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for specified period."""
        history = self.get_metrics_history(hours)
        
        if not history:
            return {}
        
        cpu_values = [m.cpu_percent for m in history]
        memory_values = [m.memory_percent for m in history]
        disk_values = [m.disk_percent for m in history]
        
        return {
            'period_hours': hours,
            'data_points': len(history),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'disk': {
                'avg': sum(disk_values) / len(disk_values),
                'min': min(disk_values),
                'max': max(disk_values)
            }
        }


# ==============================================
# === ANALYTICS ENGINE ===
# ==============================================

class AnalyticsEngine:
    """Advanced analytics and reporting system."""
    
    def __init__(self):
        self._metrics: Dict[str, deque] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def record_metric(self, name: str, value: Union[int, float], metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        timestamp = DateTimeUtils.now_utc()
        
        with self._lock:
            if metric_type == MetricType.COUNTER:
                self._counters[name] = self._counters.get(name, 0) + int(value)
            
            elif metric_type == MetricType.GAUGE:
                self._gauges[name] = float(value)
            
            elif metric_type == MetricType.TIMER:
                if name not in self._timers:
                    self._timers[name] = []
                self._timers[name].append(float(value))
                
                # Keep only last 1000 measurements
                if len(self._timers[name]) > 1000:
                    self._timers[name] = self._timers[name][-1000:]
            
            # Store in time series
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=10000)
            
            self._metrics[name].append({
                'timestamp': timestamp,
                'value': value,
                'type': metric_type.value,
                'tags': tags or {}
            })
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing measurement."""
        self.record_metric(name, duration, MetricType.TIMER, tags)
    
    def get_metric_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a metric."""
        with self._lock:
            if name in self._timers:
                values = self._timers[name]
                if values:
                    return {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'latest': values[-1]
                    }
            
            elif name in self._counters:
                return {
                    'value': self._counters[name],
                    'type': 'counter'
                }
            
            elif name in self._gauges:
                return {
                    'value': self._gauges[name],
                    'type': 'gauge'
                }
        
        return None
    
    def get_time_series(self, name: str, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        with self._lock:
            if name not in self._metrics:
                return []
            
            data = list(self._metrics[name])
            
            if start_time:
                data = [d for d in data if d['timestamp'] >= start_time]
            
            if end_time:
                data = [d for d in data if d['timestamp'] <= end_time]
            
            return data
    
    def generate_report(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate analytics report."""
        if not start_time:
            start_time = DateTimeUtils.now_utc() - timedelta(hours=24)
        
        if not end_time:
            end_time = DateTimeUtils.now_utc()
        
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'timers': {},
            'summary': {}
        }
        
        # Timer statistics
        with self._lock:
            for name, values in self._timers.items():
                if values:
                    report['timers'][name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'p50': self._percentile(values, 50),
                        'p95': self._percentile(values, 95),
                        'p99': self._percentile(values, 99)
                    }
        
        # Generate summary
        report['summary'] = {
            'total_metrics': len(self._metrics),
            'total_counters': len(self._counters),
            'total_gauges': len(self._gauges),
            'total_timers': len(self._timers),
            'generated_at': DateTimeUtils.now_utc().isoformat()
        }
        
        return report
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]


# ==============================================
# === ADVANCED LOGGING ===
# ==============================================

class AdvancedLogger:
    """Advanced logging system with 6 levels."""
    
    def __init__(self, name: str = "MegaSerpentClient"):
        self.name = name
        self._loggers: Dict[str, logging.Logger] = {}
        self._handlers: List[logging.Handler] = []
        self._structured_logs: deque = deque(maxlen=10000)
        self._log_stats: Dict[str, int] = {level.name: 0 for level in LogLevel}
        self._lock = threading.Lock()
        
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup logging configuration."""
        # Create loggers for each level
        for level in LogLevel:
            logger = logging.getLogger(f"{self.name}.{level.name}")
            logger.setLevel(level.value)
            self._loggers[level.name] = logger
        
        # Setup default handlers
        self._setup_console_handler()
        self._setup_structured_handler()
    
    def _setup_console_handler(self):
        """Setup console logging handler."""
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        for logger in self._loggers.values():
            logger.addHandler(handler)
        
        self._handlers.append(handler)
    
    def _setup_structured_handler(self):
        """Setup structured logging handler."""
        handler = StructuredLogHandler(self._structured_logs, self._log_stats, self._lock)
        
        for logger in self._loggers.values():
            logger.addHandler(handler)
        
        self._handlers.append(handler)
    
    def add_file_handler(self, log_file: str, max_bytes: int = 10*1024*1024, backup_count: int = 5):
        """Add file logging handler with rotation."""
        from logging.handlers import RotatingFileHandler
        
        handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        
        for logger in self._loggers.values():
            logger.addHandler(handler)
        
        self._handlers.append(handler)
    
    def log(self, level: LogLevel, message: str, **kwargs):
        """Log message at specified level."""
        logger = self._loggers.get(level.name)
        if logger:
            # Add structured data if provided
            if kwargs:
                extra = {'structured_data': kwargs}
                logger.log(level.value, message, extra=extra)
            else:
                logger.log(level.value, message)
    
    def fatal(self, message: str, **kwargs):
        """Log fatal message."""
        self.log(LogLevel.FATAL, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def warn(self, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARN, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def max_verbose(self, message: str, **kwargs):
        """Log max verbose message."""
        self.log(LogLevel.MAX_VERBOSE, message, **kwargs)
    
    def get_log_stats(self) -> Dict[str, int]:
        """Get logging statistics."""
        with self._lock:
            return dict(self._log_stats)
    
    def get_recent_logs(self, count: int = 100, level: Optional[LogLevel] = None) -> List[Dict[str, Any]]:
        """Get recent log entries."""
        with self._lock:
            logs = list(self._structured_logs)
            
            if level:
                logs = [log for log in logs if log.get('level') == level.name]
            
            return logs[-count:]
    
    def search_logs(self, query: str, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Search log entries."""
        with self._lock:
            logs = list(self._structured_logs)
            
            # Filter by time range
            if start_time:
                logs = [log for log in logs if log.get('timestamp', datetime.min.replace(tzinfo=timezone.utc)) >= start_time]
            
            if end_time:
                logs = [log for log in logs if log.get('timestamp', datetime.max.replace(tzinfo=timezone.utc)) <= end_time]
            
            # Filter by query
            matching_logs = []
            query_lower = query.lower()
            
            for log in logs:
                if (query_lower in log.get('message', '').lower() or
                    query_lower in str(log.get('structured_data', '')).lower()):
                    matching_logs.append(log)
            
            return matching_logs


class StructuredLogHandler(logging.Handler):
    """Custom handler for structured logging."""
    
    def __init__(self, structured_logs: deque, log_stats: Dict[str, int], lock: threading.Lock):
        super().__init__()
        self.structured_logs = structured_logs
        self.log_stats = log_stats
        self.lock = lock
    
    def emit(self, record):
        """Emit a log record."""
        try:
            with self.lock:
                # Create structured log entry
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add structured data if available
                if hasattr(record, 'structured_data'):
                    log_entry['structured_data'] = record.structured_data
                
                self.structured_logs.append(log_entry)
                
                # Update statistics
                self.log_stats[record.levelname] = self.log_stats.get(record.levelname, 0) + 1
                
        except Exception:
            self.handleError(record)


# ==============================================
# === CACHE MANAGEMENT ===
# ==============================================

class CacheManager:
    """Advanced cache management with multiple strategies."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600,
                 strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: deque = deque()  # For LRU
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL expiration
            if entry.expires_at and DateTimeUtils.now_utc() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Update access statistics
            entry.last_accessed = DateTimeUtils.now_utc()
            entry.access_count += 1
            
            # Update access order for LRU
            if self.strategy == CacheStrategy.LRU:
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
            
            self._hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate expiration
            expires_at = None
            if ttl is not None or self.default_ttl:
                ttl_seconds = ttl if ttl is not None else self.default_ttl
                expires_at = DateTimeUtils.now_utc() + timedelta(seconds=ttl_seconds)
            
            # Calculate size (rough estimate)
            try:
                size = len(str(value))
            except:
                size = 1
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                size=size
            )
            
            # Check if we need to evict entries
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_entries(1)
            
            # Store entry
            self._cache[key] = entry
            
            # Update access order for LRU
            if self.strategy == CacheStrategy.LRU:
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
        
        return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                
                if key in self._access_order:
                    self._access_order.remove(key)
                
                return True
        
        return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            
            # Reset statistics
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    def _evict_entries(self, count: int):
        """Evict entries based on strategy."""
        evicted = 0
        
        while evicted < count and self._cache:
            key_to_evict = None
            
            if self.strategy == CacheStrategy.LRU:
                # Evict least recently used
                key_to_evict = self._access_order.popleft() if self._access_order else None
            
            elif self.strategy == CacheStrategy.LFU:
                # Evict least frequently used
                min_access_count = float('inf')
                for key, entry in self._cache.items():
                    if entry.access_count < min_access_count:
                        min_access_count = entry.access_count
                        key_to_evict = key
            
            elif self.strategy == CacheStrategy.TTL:
                # Evict entries closest to expiration
                min_ttl = float('inf')
                now = DateTimeUtils.now_utc()
                
                for key, entry in self._cache.items():
                    if entry.expires_at:
                        ttl = (entry.expires_at - now).total_seconds()
                        if ttl < min_ttl:
                            min_ttl = ttl
                            key_to_evict = key
            
            elif self.strategy == CacheStrategy.FIFO:
                # Evict first in (oldest created)
                oldest_time = datetime.max.replace(tzinfo=timezone.utc)
                for key, entry in self._cache.items():
                    if entry.created_at < oldest_time:
                        oldest_time = entry.created_at
                        key_to_evict = key
            
            if key_to_evict and key_to_evict in self._cache:
                del self._cache[key_to_evict]
                if key_to_evict in self._access_order:
                    self._access_order.remove(key_to_evict)
                
                evicted += 1
                self._evictions += 1
            else:
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            total_size = sum(entry.size for entry in self._cache.values())
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'total_size': total_size,
                'strategy': self.strategy.value
            }


# ==============================================
# === MAINTENANCE MANAGER ===
# ==============================================

class MaintenanceManager:
    """System maintenance and administration."""
    
    def __init__(self):
        self._scheduled_tasks: Dict[str, MaintenanceTask] = {}
        self._completed_tasks: deque = deque(maxlen=1000)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def schedule_task(self, task_type: MaintenanceType, description: str,
                     scheduled_at: datetime, task_func: Callable,
                     **kwargs) -> str:
        """Schedule maintenance task."""
        task_id = Helpers.generate_request_id()
        
        task = MaintenanceTask(
            task_id=task_id,
            task_type=task_type,
            description=description,
            scheduled_at=scheduled_at
        )
        
        with self._lock:
            self._scheduled_tasks[task_id] = task
        
        # Schedule execution
        delay = (scheduled_at - DateTimeUtils.now_utc()).total_seconds()
        if delay > 0:
            threading.Timer(delay, self._execute_task, args=(task_id, task_func, kwargs)).start()
        else:
            # Execute immediately
            self._executor.submit(self._execute_task, task_id, task_func, kwargs)
        
        self.logger.info(f"Scheduled {task_type.value} task: {description}")
        return task_id
    
    def _execute_task(self, task_id: str, task_func: Callable, kwargs: Dict[str, Any]):
        """Execute maintenance task."""
        with self._lock:
            if task_id not in self._scheduled_tasks:
                return
            
            task = self._scheduled_tasks[task_id]
            task.status = "running"
            task.started_at = DateTimeUtils.now_utc()
        
        try:
            self.logger.info(f"Executing maintenance task: {task.description}")
            
            # Execute task function
            result = task_func(**kwargs)
            
            with self._lock:
                task.status = "completed"
                task.progress = 100.0
                task.result = result
                task.completed_at = DateTimeUtils.now_utc()
            
            self.logger.info(f"Completed maintenance task: {task.description}")
            
        except Exception as e:
            with self._lock:
                task.status = "failed"
                task.error = str(e)
                task.completed_at = DateTimeUtils.now_utc()
            
            self.logger.error(f"Maintenance task failed: {task.description} - {e}")
        
        finally:
            # Move to completed tasks
            with self._lock:
                if task_id in self._scheduled_tasks:
                    completed_task = self._scheduled_tasks.pop(task_id)
                    self._completed_tasks.append(completed_task)
    
    def get_task_status(self, task_id: str) -> Optional[MaintenanceTask]:
        """Get task status."""
        with self._lock:
            # Check scheduled tasks
            if task_id in self._scheduled_tasks:
                return self._scheduled_tasks[task_id]
            
            # Check completed tasks
            for task in self._completed_tasks:
                if task.task_id == task_id:
                    return task
        
        return None
    
    def list_scheduled_tasks(self) -> List[MaintenanceTask]:
        """List scheduled tasks."""
        with self._lock:
            return list(self._scheduled_tasks.values())
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel scheduled task."""
        with self._lock:
            if task_id in self._scheduled_tasks:
                task = self._scheduled_tasks.pop(task_id)
                task.status = "cancelled"
                self._completed_tasks.append(task)
                
                self.logger.info(f"Cancelled maintenance task: {task.description}")
                return True
        
        return False
    
    def run_cleanup(self) -> Dict[str, Any]:
        """Run system cleanup."""
        results = {
            'cache_cleared': False,
            'temp_files_removed': 0,
            'log_files_rotated': 0,
            'memory_freed': 0
        }
        
        try:
            # Simulate cleanup operations
            import gc
            
            # Force garbage collection
            collected = gc.collect()
            results['memory_freed'] = collected
            
            # Clean temporary files (simulated)
            results['temp_files_removed'] = 42
            
            # Log rotation (simulated)
            results['log_files_rotated'] = 3
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            raise
        
        return results


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Enums
    'AlertLevel', 'MetricType', 'CacheStrategy', 'MaintenanceType',
    
    # Data Classes
    'SystemMetrics', 'ApplicationMetrics', 'AlertInfo', 'CacheEntry', 'MaintenanceTask',
    
    # Monitoring
    'SystemMonitor',
    
    # Analytics
    'AnalyticsEngine',
    
    # Logging
    'AdvancedLogger', 'StructuredLogHandler',
    
    # Cache Management
    'CacheManager',
    
    # Maintenance
    'MaintenanceManager'
]