"""
API Enhancements Module
=======================

This module provides advanced API enhancement features for the MegaPythonLibrary:
1. Rate limiting management with configurable limits
2. Bandwidth throttling with upload/download speed control
3. Connection pooling for improved performance
4. Async/await support for non-blocking operations

These enhancements maintain compatibility with the existing sync API while providing
opt-in performance and control features for production deployments.

Author: MegaPythonLibrary Team
Date: July 2025
"""

import asyncio
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union, Awaitable
import weakref
import logging

from .dependencies import *
from .exceptions import RequestError, ValidationError


# ==============================================
# === RATE LIMITING MANAGEMENT ===
# ==============================================

@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting."""
    max_requests_per_second: float = 10.0
    max_requests_per_minute: float = 600.0
    max_requests_per_hour: float = 10000.0
    burst_limit: int = 20  # Allow brief bursts above the per-second limit
    adaptive: bool = True  # Automatically adjust based on API responses


class RateLimiter:
    """
    Advanced rate limiter with multiple time windows and burst handling.
    
    Features:
    - Multiple time windows (second, minute, hour)
    - Burst allowance for brief spikes
    - Adaptive rate adjustment based on API responses
    - Thread-safe operation
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.lock = threading.RLock()
        
        # Request tracking
        self.requests_per_second = deque()
        self.requests_per_minute = deque()
        self.requests_per_hour = deque()
        
        # Adaptive rate limiting
        self.current_rate_multiplier = 1.0
        self.consecutive_errors = 0
        self.last_rate_adjustment = time.time()
        
        self.logger = logging.getLogger(__name__)
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make an API request.
        
        Args:
            timeout: Maximum time to wait for permission (None = wait indefinitely)
            
        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                current_time = time.time()
                
                # Clean old entries
                self._clean_old_entries(current_time)
                
                # Check rate limits
                if self._can_make_request(current_time):
                    self._record_request(current_time)
                    return True
                
                # Check timeout
                if timeout is not None and (current_time - start_time) >= timeout:
                    return False
                
                # Calculate wait time
                wait_time = self._calculate_wait_time(current_time)
            
            # Wait before retrying
            time.sleep(min(wait_time, 0.1))  # Max 100ms sleep
    
    def record_response(self, success: bool, response_time: float = None):
        """
        Record API response for adaptive rate limiting.
        
        Args:
            success: Whether the request was successful
            response_time: Response time in seconds
        """
        with self.lock:
            if not self.config.adaptive:
                return
            
            current_time = time.time()
            
            if success:
                self.consecutive_errors = 0
                # Gradually increase rate if we're being too conservative
                if (current_time - self.last_rate_adjustment) > 60:  # 1 minute
                    self.current_rate_multiplier = min(1.0, self.current_rate_multiplier * 1.05)
                    self.last_rate_adjustment = current_time
            else:
                self.consecutive_errors += 1
                # Reduce rate after consecutive errors
                if self.consecutive_errors >= 3:
                    self.current_rate_multiplier *= 0.7  # Reduce by 30%
                    self.consecutive_errors = 0
                    self.last_rate_adjustment = current_time
                    self.logger.warning(f"Rate limit reduced due to errors: {self.current_rate_multiplier:.2f}")
    
    def _clean_old_entries(self, current_time: float):
        """Remove expired entries from tracking queues."""
        # Clean second entries (older than 1 second)
        while self.requests_per_second and (current_time - self.requests_per_second[0]) > 1.0:
            self.requests_per_second.popleft()
        
        # Clean minute entries (older than 60 seconds)
        while self.requests_per_minute and (current_time - self.requests_per_minute[0]) > 60.0:
            self.requests_per_minute.popleft()
        
        # Clean hour entries (older than 3600 seconds)
        while self.requests_per_hour and (current_time - self.requests_per_hour[0]) > 3600.0:
            self.requests_per_hour.popleft()
    
    def _can_make_request(self, current_time: float) -> bool:
        """Check if we can make a request without exceeding limits."""
        effective_per_second = self.config.max_requests_per_second * self.current_rate_multiplier
        
        # Check per-second limit (but allow burst initially)
        requests_in_last_second = len(self.requests_per_second)
        
        # If we haven't reached burst limit, allow the request
        if requests_in_last_second < self.config.burst_limit:
            return True
        
        # If we've used our burst, check against the per-second rate
        if requests_in_last_second >= effective_per_second:
            return False
        
        # Check per-minute limit
        if len(self.requests_per_minute) >= self.config.max_requests_per_minute:
            return False
        
        # Check per-hour limit
        if len(self.requests_per_hour) >= self.config.max_requests_per_hour:
            return False
        
        return True
    
    def _record_request(self, current_time: float):
        """Record a new request in all tracking queues."""
        self.requests_per_second.append(current_time)
        self.requests_per_minute.append(current_time)
        self.requests_per_hour.append(current_time)
    
    def _calculate_wait_time(self, current_time: float) -> float:
        """Calculate how long to wait before next request."""
        wait_times = []
        
        # Wait time based on per-second limit
        if self.requests_per_second:
            oldest_in_second = self.requests_per_second[0]
            wait_times.append(1.0 - (current_time - oldest_in_second))
        
        # Wait time based on per-minute limit
        if len(self.requests_per_minute) >= self.config.max_requests_per_minute:
            oldest_in_minute = self.requests_per_minute[0]
            wait_times.append(60.0 - (current_time - oldest_in_minute))
        
        # Return the maximum wait time needed
        return max(wait_times) if wait_times else 0.01


# ==============================================
# === BANDWIDTH THROTTLING ===
# ==============================================

@dataclass
class BandwidthConfig:
    """Configuration for bandwidth throttling."""
    max_upload_speed: Optional[int] = None  # Bytes per second
    max_download_speed: Optional[int] = None  # Bytes per second
    burst_allowance: float = 2.0  # Allow brief bursts up to 2x the limit
    measurement_window: float = 1.0  # Time window for measuring speed


class BandwidthThrottler:
    """
    Bandwidth throttling for upload and download operations.
    
    Features:
    - Separate upload and download limits
    - Burst allowance for better user experience
    - Adaptive throttling based on actual transfer rates
    - Thread-safe operation
    """
    
    def __init__(self, config: BandwidthConfig = None):
        self.config = config or BandwidthConfig()
        self.lock = threading.RLock()
        
        # Transfer tracking
        self.upload_history = deque()  # (timestamp, bytes) tuples
        self.download_history = deque()  # (timestamp, bytes) tuples
        
        self.logger = logging.getLogger(__name__)
    
    def throttle_upload(self, bytes_to_send: int) -> float:
        """
        Calculate delay needed before sending data to respect upload limits.
        
        Args:
            bytes_to_send: Number of bytes about to be sent
            
        Returns:
            Seconds to wait before sending
        """
        if not self.config.max_upload_speed:
            return 0.0
        
        with self.lock:
            current_time = time.time()
            self._clean_old_entries(current_time, self.upload_history)
            
            # Calculate current upload rate
            current_rate = self._calculate_current_rate(self.upload_history, current_time)
            max_allowed = self.config.max_upload_speed * self.config.burst_allowance
            
            # If adding this transfer would exceed limits, calculate delay
            if current_rate + (bytes_to_send / self.config.measurement_window) > max_allowed:
                # Calculate required delay to stay within limits
                excess_rate = current_rate + (bytes_to_send / self.config.measurement_window) - max_allowed
                delay = excess_rate / self.config.max_upload_speed
                return max(0.0, delay)
            
            return 0.0
    
    def throttle_download(self, bytes_received: int) -> float:
        """
        Calculate delay needed after receiving data to respect download limits.
        
        Args:
            bytes_received: Number of bytes just received
            
        Returns:
            Seconds to wait after receiving
        """
        if not self.config.max_download_speed:
            return 0.0
        
        with self.lock:
            current_time = time.time()
            self._clean_old_entries(current_time, self.download_history)
            
            # Calculate current download rate
            current_rate = self._calculate_current_rate(self.download_history, current_time)
            max_allowed = self.config.max_download_speed * self.config.burst_allowance
            
            # If this transfer exceeded limits, calculate delay
            if current_rate + (bytes_received / self.config.measurement_window) > max_allowed:
                excess_rate = current_rate + (bytes_received / self.config.measurement_window) - max_allowed
                delay = excess_rate / self.config.max_download_speed
                return max(0.0, delay)
            
            return 0.0
    
    def record_upload(self, bytes_sent: int):
        """Record bytes sent for bandwidth tracking."""
        with self.lock:
            current_time = time.time()
            self.upload_history.append((current_time, bytes_sent))
    
    def record_download(self, bytes_received: int):
        """Record bytes received for bandwidth tracking."""
        with self.lock:
            current_time = time.time()
            self.download_history.append((current_time, bytes_received))
    
    def _clean_old_entries(self, current_time: float, history: deque):
        """Remove entries older than the measurement window."""
        cutoff_time = current_time - self.config.measurement_window
        while history and history[0][0] < cutoff_time:
            history.popleft()
    
    def _calculate_current_rate(self, history: deque, current_time: float) -> float:
        """Calculate current transfer rate in bytes per second."""
        if not history:
            return 0.0
        
        # Sum bytes in the current measurement window
        total_bytes = sum(entry[1] for entry in history)
        return total_bytes / self.config.measurement_window


# ==============================================
# === CONNECTION POOLING ===
# ==============================================

@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling."""
    max_connections: int = 10
    max_connections_per_host: int = 5
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    pool_timeout: float = 10.0
    retry_on_timeout: bool = True


class ConnectionPoolManager:
    """
    Enhanced connection pooling for improved performance.
    
    Features:
    - Configurable connection limits
    - Connection reuse and keep-alive
    - Automatic retry on connection failures
    - Health monitoring and cleanup
    """
    
    def __init__(self, config: ConnectionPoolConfig = None):
        self.config = config or ConnectionPoolConfig()
        self.session = None
        self.lock = threading.RLock()
        self._initialize_session()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_session(self):
        """Initialize the requests session with connection pooling."""
        import requests.adapters
        from urllib3.util.retry import Retry
        
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        # Configure HTTP adapter with connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.config.max_connections,
            pool_maxsize=self.config.max_connections_per_host,
            max_retries=retry_strategy,
            pool_block=True
        )
        
        # Mount adapters for HTTP and HTTPS
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default timeouts
        self.session.timeout = (self.config.connection_timeout, self.config.read_timeout)
    
    def make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make HTTP request using the connection pool.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
        """
        with self.lock:
            if not self.session:
                self._initialize_session()
            
            return self.session.request(method, url, **kwargs)
    
    def close(self):
        """Close all connections and cleanup."""
        with self.lock:
            if self.session:
                self.session.close()
                self.session = None


# ==============================================
# === ASYNC/AWAIT SUPPORT ===
# ==============================================

class AsyncAPIClient:
    """
    Async/await wrapper for the MegaPythonLibrary API.
    
    Provides non-blocking access to all API operations while maintaining
    compatibility with the existing synchronous API.
    """
    
    def __init__(self, sync_client, executor: Optional[ThreadPoolExecutor] = None):
        """
        Initialize async API client.
        
        Args:
            sync_client: The synchronous MPLClient instance
            executor: Thread pool executor (created if not provided)
        """
        self.sync_client = sync_client
        self.executor = executor or ThreadPoolExecutor(max_workers=4, thread_name_prefix="MPL-Async")
        self.loop = None
        self.logger = logging.getLogger(__name__)
    
    async def login(self, email: str, password: str, save_session: bool = True) -> bool:
        """Async login to Mega."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_client.login,
            email, password, save_session
        )
    
    async def upload_file(self, local_path: str, remote_path: str = None, 
                         progress_callback: Optional[Callable] = None) -> bool:
        """Async file upload."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_client.upload_file,
            local_path, remote_path, progress_callback
        )
    
    async def download_file(self, remote_path: str, local_path: str = None,
                           progress_callback: Optional[Callable] = None) -> str:
        """Async file download."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_client.download_file,
            remote_path, local_path, progress_callback
        )
    
    async def list_folder(self, remote_path: str = "/") -> List[Dict[str, Any]]:
        """Async folder listing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_client.list_folder,
            remote_path
        )
    
    async def create_folder(self, remote_path: str) -> bool:
        """Async folder creation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_client.create_folder,
            remote_path
        )
    
    async def delete_node(self, remote_path: str) -> bool:
        """Async node deletion."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_client.delete_node,
            remote_path
        )
    
    async def batch_operation(self, operations: List[Callable]) -> List[Any]:
        """
        Execute multiple operations concurrently.
        
        Args:
            operations: List of callable operations
            
        Returns:
            List of results in the same order as operations
        """
        loop = asyncio.get_event_loop()
        
        # Create tasks for all operations
        tasks = []
        for operation in operations:
            if asyncio.iscoroutinefunction(operation):
                tasks.append(operation())
            else:
                tasks.append(loop.run_in_executor(self.executor, operation))
        
        # Wait for all tasks to complete
        return await asyncio.gather(*tasks)
    
    def close(self):
        """Cleanup async resources."""
        if self.executor:
            self.executor.shutdown(wait=True)


# ==============================================
# === ENHANCED API SESSION ===
# ==============================================

class EnhancedAPISession:
    """
    Enhanced API session with all advanced features integrated.
    """
    
    def __init__(self, 
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 bandwidth_config: Optional[BandwidthConfig] = None,
                 connection_config: Optional[ConnectionPoolConfig] = None):
        """
        Initialize enhanced API session.
        
        Args:
            rate_limit_config: Rate limiting configuration
            bandwidth_config: Bandwidth throttling configuration
            connection_config: Connection pooling configuration
        """
        self.rate_limiter = RateLimiter(rate_limit_config)
        self.bandwidth_throttler = BandwidthThrottler(bandwidth_config)
        self.connection_pool = ConnectionPoolManager(connection_config)
        
        # Integration with existing network module
        self._original_make_request = None
        self.logger = logging.getLogger(__name__)
    
    def install(self):
        """Install enhancements into the network module."""
        from . import network
        
        # Store original function
        self._original_make_request = network.make_request
        
        # Replace with enhanced version
        network.make_request = self._enhanced_make_request
        
        self.logger.info("API enhancements installed successfully")
    
    def uninstall(self):
        """Remove enhancements and restore original functionality."""
        if self._original_make_request:
            from . import network
            network.make_request = self._original_make_request
            self.logger.info("API enhancements uninstalled")
    
    def _enhanced_make_request(self, url: str, data: Any = None, method: str = 'POST', 
                              timeout: float = 30, **kwargs) -> requests.Response:
        """
        Enhanced request function with all features integrated.
        """
        start_time = time.time()
        
        # Apply rate limiting
        if not self.rate_limiter.acquire(timeout=timeout):
            raise RequestError("Rate limit timeout - too many requests")
        
        try:
            # Apply bandwidth throttling for uploads
            if method in ['POST', 'PUT', 'PATCH'] and data:
                if isinstance(data, (str, bytes)):
                    upload_size = len(data)
                elif hasattr(data, 'read'):  # File-like object
                    current_pos = data.tell()
                    data.seek(0, 2)  # Seek to end
                    upload_size = data.tell()
                    data.seek(current_pos)  # Restore position
                else:
                    upload_size = len(str(data).encode('utf-8'))
                
                upload_delay = self.bandwidth_throttler.throttle_upload(upload_size)
                if upload_delay > 0:
                    time.sleep(upload_delay)
            
            # Make request using connection pool
            response = self.connection_pool.make_request(method, url, data=data, timeout=timeout, **kwargs)
            
            # Apply bandwidth throttling for downloads
            if response.content:
                download_size = len(response.content)
                self.bandwidth_throttler.record_download(download_size)
                download_delay = self.bandwidth_throttler.throttle_download(download_size)
                if download_delay > 0:
                    time.sleep(download_delay)
            
            # Record successful request for rate limiting
            response_time = time.time() - start_time
            self.rate_limiter.record_response(True, response_time)
            
            return response
            
        except Exception as e:
            # Record failed request for adaptive rate limiting
            response_time = time.time() - start_time
            self.rate_limiter.record_response(False, response_time)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about API enhancement usage."""
        with self.rate_limiter.lock:
            current_time = time.time()
            self.rate_limiter._clean_old_entries(current_time)
            
            return {
                'rate_limiting': {
                    'requests_last_second': len(self.rate_limiter.requests_per_second),
                    'requests_last_minute': len(self.rate_limiter.requests_per_minute),
                    'requests_last_hour': len(self.rate_limiter.requests_per_hour),
                    'current_rate_multiplier': self.rate_limiter.current_rate_multiplier,
                    'consecutive_errors': self.rate_limiter.consecutive_errors
                },
                'bandwidth': {
                    'upload_history_entries': len(self.bandwidth_throttler.upload_history),
                    'download_history_entries': len(self.bandwidth_throttler.download_history),
                    'max_upload_speed': self.bandwidth_throttler.config.max_upload_speed,
                    'max_download_speed': self.bandwidth_throttler.config.max_download_speed
                },
                'connection_pool': {
                    'max_connections': self.connection_pool.config.max_connections,
                    'max_connections_per_host': self.connection_pool.config.max_connections_per_host,
                    'session_active': self.connection_pool.session is not None
                }
            }
    
    def close(self):
        """Cleanup all resources."""
        self.uninstall()
        self.connection_pool.close()


# ==============================================
# === CONVENIENCE FUNCTIONS ===
# ==============================================

def create_enhanced_session(
    max_requests_per_second: float = 10.0,
    max_upload_speed: Optional[int] = None,
    max_download_speed: Optional[int] = None,
    max_connections: int = 10
) -> EnhancedAPISession:
    """
    Create an enhanced API session with common configuration.
    
    Args:
        max_requests_per_second: Maximum API requests per second
        max_upload_speed: Maximum upload speed in bytes/second (None = unlimited)
        max_download_speed: Maximum download speed in bytes/second (None = unlimited)
        max_connections: Maximum concurrent connections
        
    Returns:
        Enhanced API session ready for use
    """
    rate_config = RateLimitConfig(max_requests_per_second=max_requests_per_second)
    bandwidth_config = BandwidthConfig(
        max_upload_speed=max_upload_speed,
        max_download_speed=max_download_speed
    )
    connection_config = ConnectionPoolConfig(max_connections=max_connections)
    
    return EnhancedAPISession(rate_config, bandwidth_config, connection_config)


def enable_api_enhancements(
    max_requests_per_second: float = 10.0,
    max_upload_speed: Optional[int] = None,
    max_download_speed: Optional[int] = None,
    max_connections: int = 10
) -> EnhancedAPISession:
    """
    Enable API enhancements with default configuration.
    
    Args:
        max_requests_per_second: Maximum API requests per second
        max_upload_speed: Maximum upload speed in bytes/second (None = unlimited)
        max_download_speed: Maximum download speed in bytes/second (None = unlimited)
        max_connections: Maximum concurrent connections
        
    Returns:
        Installed enhanced API session
    """
    session = create_enhanced_session(
        max_requests_per_second, max_upload_speed, max_download_speed, max_connections
    )
    session.install()
    return session


# ==============================================
# === CLIENT METHOD INJECTION ===
# ==============================================

class AsyncAPIClient:
    """Async wrapper for API operations."""
    
    def __init__(self, enhanced_session, client):
        self.session = enhanced_session
        self.client = client
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async client statistics."""
        return {
            'session_active': bool(self.session),
            'rate_limits': self.session.rate_limiter.config.__dict__ if self.session else {},
            'bandwidth_limits': self.session.bandwidth_throttler.config.__dict__ if self.session else {}
        }


def add_api_enhancements_methods_with_events(client_class):
    """Add API enhancements methods with event support to the MPLClient class."""
    
    def enable_api_enhancements_method(self, config: Dict[str, Any] = None) -> bool:
        """Enable advanced API enhancements."""
        try:
            config = config or {}
            session = enable_api_enhancements(
                config.get('max_requests_per_second', 10.0),
                config.get('max_upload_speed'),
                config.get('max_download_speed'),
                config.get('max_connections', 10)
            )
            self._api_enhancements = session
            if hasattr(self, '_trigger_event'):
                self._trigger_event('api_enhancements_enabled', {'config': config})
            return True
        except Exception as e:
            if hasattr(self, '_trigger_event'):
                self._trigger_event('api_enhancements_error', {'error': str(e)})
            raise RequestError(f"API enhancements initialization failed: {e}")
    
    def disable_api_enhancements_method(self) -> bool:
        """Disable API enhancements and restore default behavior."""
        if hasattr(self, '_api_enhancements') and self._api_enhancements:
            self._api_enhancements.uninstall()
            self._api_enhancements = None
            if hasattr(self, '_trigger_event'):
                self._trigger_event('api_enhancements_disabled', {})
            return True
        return False
    
    def get_api_enhancement_stats_method(self) -> Optional[Dict[str, Any]]:
        """Get statistics about API enhancement usage."""
        if hasattr(self, '_api_enhancements') and self._api_enhancements:
            return self._api_enhancements.get_stats()
        return None
    
    def create_async_client_method(self):
        """Create an async/await client for non-blocking operations."""
        if not hasattr(self, '_api_enhancements') or not self._api_enhancements:
            raise RequestError("API enhancements must be enabled first")
        
        if not hasattr(self, '_async_client') or not self._async_client:
            # Create async client wrapper around enhanced session
            self._async_client = AsyncAPIClient(self._api_enhancements, self)
        
        return self._async_client
    
    def configure_rate_limiting_method(self, 
                                     max_requests_per_second: float = None,
                                     max_requests_per_minute: float = None,
                                     max_requests_per_hour: float = None,
                                     adaptive: bool = None) -> bool:
        """Configure rate limiting parameters."""
        if not hasattr(self, '_api_enhancements') or not self._api_enhancements:
            return False
        
        return self._api_enhancements.configure_rate_limiting(
            max_requests_per_second, max_requests_per_minute,
            max_requests_per_hour, adaptive
        )
    
    def configure_bandwidth_throttling_method(self,
                                             max_upload_speed: int = None,
                                             max_download_speed: int = None,
                                             burst_allowance: float = None) -> bool:
        """Configure bandwidth throttling parameters."""
        if not hasattr(self, '_api_enhancements') or not self._api_enhancements:
            return False
        
        return self._api_enhancements.configure_bandwidth_throttling(
            max_upload_speed, max_download_speed, burst_allowance
        )
    
    # Add methods to client class
    setattr(client_class, 'enable_api_enhancements', enable_api_enhancements_method)
    setattr(client_class, 'disable_api_enhancements', disable_api_enhancements_method)
    setattr(client_class, 'get_api_enhancement_stats', get_api_enhancement_stats_method)
    setattr(client_class, 'create_async_client', create_async_client_method)
    setattr(client_class, 'configure_rate_limiting', configure_rate_limiting_method)
    setattr(client_class, 'configure_bandwidth_throttling', configure_bandwidth_throttling_method)


# ==============================================
# === ENHANCED API FUNCTIONS WITH EVENTS ===
# ==============================================

def enable_api_enhancements_with_events(config: Dict[str, Any] = None, 
                                       event_callback=None) -> Dict[str, Any]:
    """
    Enhanced API enhancements enabler with event callbacks and logging.
    
    Args:
        config: Enhancement configuration dictionary
        event_callback: Function to call for events (optional)
        
    Returns:
        Enhancement status and configuration
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        config = config or {}
        
        # Create enhanced session with configuration
        enhanced_session = create_enhanced_session(
            max_requests_per_second=config.get('max_requests_per_second', 10.0),
            max_upload_speed=config.get('max_upload_speed'),
            max_download_speed=config.get('max_download_speed'),
            max_connections=config.get('max_connections', 10)
        )
        
        # Install the enhancements
        enhanced_session.install()
        
        result = {
            'status': 'enabled',
            'config': config,
            'session': enhanced_session
        }
        
        logger.info("API enhancements enabled successfully")
        trigger_event('api_enhancements_enabled', result)
        
        return result
        
    except Exception as e:
        error_result = {'status': 'error', 'error': str(e)}
        logger.error(f"Failed to enable API enhancements: {e}")
        trigger_event('api_enhancements_failed', error_result)
        raise RequestError(f"API enhancements initialization failed: {e}")


def disable_api_enhancements_with_events(enhanced_session, async_client=None,
                                        event_callback=None) -> bool:
    """
    Enhanced API enhancements disabler with event callbacks and logging.
    
    Args:
        enhanced_session: The enhanced session to disable
        async_client: Optional async client to close
        event_callback: Function to call for events (optional)
        
    Returns:
        True if enhancements disabled successfully
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        if enhanced_session:
            enhanced_session.uninstall()
            enhanced_session.close()
        
        if async_client:
            async_client.close()
        
        logger.info("API enhancements disabled")
        trigger_event('api_enhancements_disabled', {'status': 'disabled'})
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to disable API enhancements: {e}")
        trigger_event('api_enhancements_disable_failed', {'error': str(e)})
        return False


def get_api_enhancement_stats_with_events(enhanced_session, event_callback=None) -> Optional[Dict[str, Any]]:
    """
    Enhanced API enhancement statistics getter with event callbacks and logging.
    
    Args:
        enhanced_session: The enhanced session to get stats from
        event_callback: Function to call for events (optional)
        
    Returns:
        Dictionary with rate limiting, bandwidth, and connection stats
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    if not enhanced_session:
        return None
    
    try:
        stats = enhanced_session.get_stats()
        logger.debug("API enhancement stats retrieved")
        trigger_event('api_stats_retrieved', stats)
        return stats
    except Exception as e:
        logger.error(f"Failed to get API enhancement stats: {e}")
        trigger_event('api_stats_failed', {'error': str(e)})
        return None


def create_async_client_with_events(enhanced_session, client_instance, 
                                   event_callback=None):
    """
    Enhanced async client creator with event callbacks and logging.
    
    Args:
        enhanced_session: The enhanced session for async support
        client_instance: The client instance to create async client for
        event_callback: Function to call for events (optional)
        
    Returns:
        AsyncAPIClient instance
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    if not enhanced_session:
        error_msg = "API enhancements must be enabled for async support"
        trigger_event('async_client_creation_failed', {'error': error_msg})
        raise RequestError(error_msg)
    
    try:
        async_client = AsyncAPIClient(client_instance)
        
        logger.info("Async client created successfully")
        trigger_event('async_client_created', {'status': 'created'})
        
        return async_client
        
    except Exception as e:
        logger.error(f"Failed to create async client: {e}")
        trigger_event('async_client_creation_failed', {'error': str(e)})
        raise


def configure_rate_limiting_with_events(enhanced_session,
                                       max_requests_per_second: float = None,
                                       max_requests_per_minute: float = None,
                                       max_requests_per_hour: float = None,
                                       adaptive: bool = None,
                                       event_callback=None) -> bool:
    """
    Enhanced rate limiting configuration with event callbacks and logging.
    
    Args:
        enhanced_session: The enhanced session to configure
        max_requests_per_second: Maximum requests per second
        max_requests_per_minute: Maximum requests per minute
        max_requests_per_hour: Maximum requests per hour
        adaptive: Enable adaptive rate limiting
        event_callback: Function to call for events (optional)
        
    Returns:
        True if configuration applied successfully
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    if not enhanced_session:
        error_msg = "API enhancements must be enabled first"
        trigger_event('rate_limiting_config_failed', {'error': error_msg})
        raise RequestError(error_msg)
    
    try:
        rate_limiter = enhanced_session.rate_limiter
        config = rate_limiter.config
        
        config_changes = {}
        if max_requests_per_second is not None:
            config.max_requests_per_second = max_requests_per_second
            config_changes['max_requests_per_second'] = max_requests_per_second
        if max_requests_per_minute is not None:
            config.max_requests_per_minute = max_requests_per_minute
            config_changes['max_requests_per_minute'] = max_requests_per_minute
        if max_requests_per_hour is not None:
            config.max_requests_per_hour = max_requests_per_hour
            config_changes['max_requests_per_hour'] = max_requests_per_hour
        if adaptive is not None:
            config.adaptive = adaptive
            config_changes['adaptive'] = adaptive
        
        logger.info("Rate limiting configuration updated")
        trigger_event('rate_limiting_configured', config_changes)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure rate limiting: {e}")
        trigger_event('rate_limiting_config_failed', {'error': str(e)})
        return False


def configure_bandwidth_throttling_with_events(enhanced_session,
                                              max_upload_speed: int = None,
                                              max_download_speed: int = None,
                                              burst_allowance: float = None,
                                              event_callback=None) -> bool:
    """
    Enhanced bandwidth throttling configuration with event callbacks and logging.
    
    Args:
        enhanced_session: The enhanced session to configure
        max_upload_speed: Maximum upload speed in bytes/second (None = unlimited)
        max_download_speed: Maximum download speed in bytes/second (None = unlimited)
        burst_allowance: Burst multiplier (e.g., 2.0 allows 2x speed bursts)
        event_callback: Function to call for events (optional)
        
    Returns:
        True if configuration applied successfully
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    if not enhanced_session:
        error_msg = "API enhancements must be enabled first"
        trigger_event('bandwidth_throttling_config_failed', {'error': error_msg})
        raise RequestError(error_msg)
    
    try:
        throttler = enhanced_session.bandwidth_throttler
        config = throttler.config
        
        config_changes = {}
        if max_upload_speed is not None:
            config.max_upload_speed = max_upload_speed
            config_changes['max_upload_speed'] = max_upload_speed
        if max_download_speed is not None:
            config.max_download_speed = max_download_speed
            config_changes['max_download_speed'] = max_download_speed
        if burst_allowance is not None:
            config.burst_allowance = burst_allowance
            config_changes['burst_allowance'] = burst_allowance
        
        logger.info("Bandwidth throttling configuration updated")
        trigger_event('bandwidth_throttling_configured', config_changes)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure bandwidth throttling: {e}")
        trigger_event('bandwidth_throttling_config_failed', {'error': str(e)})
        return False


def make_request_with_events(url: str, data: Dict[str, Any] = None,
                           session_id: str = None, timeout: int = 30,
                           event_callback=None) -> Any:
    """
    Enhanced HTTP request function with event callbacks and logging.
    
    Args:
        url: URL to make request to
        data: Request data (optional)
        session_id: Session ID for authenticated requests (optional)
        timeout: Request timeout in seconds
        event_callback: Function to call for events (optional)
        
    Returns:
        Response data
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        from . import network
        
        request_info = {
            'url': url,
            'has_data': data is not None,
            'has_session': session_id is not None,
            'timeout': timeout
        }
        
        trigger_event('request_started', request_info)
        
        response = network.make_request(url, data, session_id, timeout)
        
        logger.debug(f"Request successful to {url}")
        trigger_event('request_completed', {'url': url, 'success': True})
        
        return response
        
    except Exception as e:
        logger.error(f"Request failed to {url}: {e}")
        trigger_event('request_failed', {'url': url, 'error': str(e)})
        raise


def api_request_with_events(command: Dict[str, Any], session_id: str = None,
                          event_callback=None) -> Any:
    """
    Enhanced API request function with event callbacks and logging.
    
    Args:
        command: API command dictionary
        session_id: Session ID for authenticated requests (optional)
        event_callback: Function to call for events (optional)
        
    Returns:
        API response
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        from . import network
        
        api_info = {
            'command': command.get('a', 'unknown'),
            'has_session': session_id is not None
        }
        
        trigger_event('api_request_started', api_info)
        
        response = network.api_request(command, session_id)
        
        logger.debug(f"API request successful: {command.get('a', 'unknown')}")
        trigger_event('api_request_completed', {'command': command.get('a'), 'success': True})
        
        return response
        
    except Exception as e:
        logger.error(f"API request failed for {command.get('a', 'unknown')}: {e}")
        trigger_event('api_request_failed', {'command': command.get('a'), 'error': str(e)})
        raise


# Configure logging for this module
logger = logging.getLogger(__name__)

