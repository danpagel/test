"""
MegaSerpentClient - Network & Communication Module

Purpose: All network communications, API interactions, and real-time messaging.

This module handles the complete API layer (REST, WebSocket, GraphQL, MEGAcmd compatibility),
network protocols, chat features, optimization, and real-time event-driven communication.
"""

import json
import time
import asyncio
import threading
import ssl
import socket
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, urlparse
import logging
from concurrent.futures import ThreadPoolExecutor
import weakref

from . import utils
from .utils import (
    Constants, NetworkError, ValidationError, MegaError,
    NetworkUtils, Helpers, DateTimeUtils, Decorators
)

# Try to import requests with fallback
try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    # Basic HTTP client fallback would be implemented here

# Try to import websocket libraries
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


# ==============================================
# === NETWORK ENUMS AND CONSTANTS ===
# ==============================================

class RequestMethod(Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class APIEndpoint(Enum):
    """API endpoint types."""
    AUTH = "/auth"
    FILES = "/files"
    FOLDERS = "/folders"
    SHARES = "/shares"
    USERS = "/users"
    SYNC = "/sync"
    UPLOAD = "/upload"
    DOWNLOAD = "/download"


class EventType(Enum):
    """Real-time event types."""
    FILE_UPLOAD = "file_upload"
    FILE_DOWNLOAD = "file_download"
    FILE_DELETED = "file_deleted"
    FOLDER_CREATED = "folder_created"
    SHARE_CREATED = "share_created"
    USER_ACTIVITY = "user_activity"
    SYNC_STATUS = "sync_status"


class NetworkCondition(Enum):
    """Network condition states."""
    EXCELLENT = "excellent"
    GOOD = "good"
    POOR = "poor"
    OFFLINE = "offline"


# ==============================================
# === DATA CLASSES ===
# ==============================================

@dataclass
class APIRequest:
    """API request container."""
    method: RequestMethod
    endpoint: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data: Optional[Union[Dict, bytes, str]] = None
    files: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None
    retries: int = 0


@dataclass
class APIResponse:
    """API response container."""
    status_code: int
    headers: Dict[str, str]
    data: Any
    raw_response: Optional[bytes] = None
    request_id: Optional[str] = None
    elapsed_time: float = 0.0
    cached: bool = False


@dataclass
class NetworkMetrics:
    """Network performance metrics."""
    latency: float = 0.0
    bandwidth_up: float = 0.0
    bandwidth_down: float = 0.0
    packet_loss: float = 0.0
    connection_count: int = 0
    requests_per_second: float = 0.0
    error_rate: float = 0.0


@dataclass
class ProxyConfig:
    """Proxy configuration."""
    enabled: bool = False
    host: str = ""
    port: int = 0
    username: Optional[str] = None
    password: Optional[str] = None
    proxy_type: str = "http"  # http, https, socks4, socks5


# ==============================================
# === API CLIENT CLASSES ===
# ==============================================

class HTTPClient:
    """Advanced HTTP/HTTPS client with middleware support."""
    
    def __init__(self, base_url: str = Constants.API_BASE_URL, timeout: int = Constants.DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
        self._session = None
        self._middleware_stack: List[Callable] = []
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self._request_count = 0
        self._error_count = 0
        self._total_time = 0.0
        
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize HTTP session with proper configuration."""
        if not REQUESTS_AVAILABLE:
            raise NetworkError("Requests library not available")
        
        self._session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=Constants.MAX_RETRIES,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        
        # Set default headers
        self._session.headers.update({
            'User-Agent': Constants.USER_AGENT,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to the request pipeline."""
        self._middleware_stack.append(middleware)
    
    def request(self, method: RequestMethod, endpoint: str, **kwargs) -> APIResponse:
        """Make HTTP request with middleware processing."""
        request = APIRequest(
            method=method,
            endpoint=endpoint,
            **kwargs
        )
        
        # Apply middleware (pre-request)
        for middleware in self._middleware_stack:
            request = middleware(request, None) or request
        
        # Make the actual request
        response = self._make_request(request)
        
        # Apply middleware (post-request)
        for middleware in reversed(self._middleware_stack):
            response = middleware(request, response) or response
        
        return response
    
    def _make_request(self, request: APIRequest) -> APIResponse:
        """Make the actual HTTP request."""
        start_time = time.time()
        request_id = Helpers.generate_request_id()
        
        try:
            url = urljoin(self.base_url, request.endpoint)
            
            # Prepare request arguments
            request_kwargs = {
                'method': request.method.value,
                'url': url,
                'headers': request.headers,
                'params': request.params,
                'timeout': request.timeout or self.timeout
            }
            
            # Add data/files if present
            if request.data:
                if isinstance(request.data, dict):
                    request_kwargs['json'] = request.data
                else:
                    request_kwargs['data'] = request.data
            
            if request.files:
                request_kwargs['files'] = request.files
            
            # Make request
            response = self._session.request(**request_kwargs)
            elapsed_time = time.time() - start_time
            
            # Update metrics
            self._request_count += 1
            self._total_time += elapsed_time
            
            # Parse response data
            try:
                if response.headers.get('content-type', '').startswith('application/json'):
                    data = response.json()
                else:
                    data = response.text
            except Exception:
                data = response.content
            
            api_response = APIResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                data=data,
                raw_response=response.content,
                request_id=request_id,
                elapsed_time=elapsed_time
            )
            
            # Check for HTTP errors
            if not response.ok:
                self._error_count += 1
                raise NetworkError(
                    f"HTTP {response.status_code}: {response.reason}",
                    error_code=response.status_code
                )
            
            return api_response
            
        except requests.exceptions.Timeout:
            self._error_count += 1
            raise NetworkError("Request timeout")
        except requests.exceptions.ConnectionError:
            self._error_count += 1
            raise NetworkError("Connection error")
        except requests.exceptions.RequestException as e:
            self._error_count += 1
            raise NetworkError(f"Request failed: {e}")
    
    def get_metrics(self) -> NetworkMetrics:
        """Get network performance metrics."""
        avg_latency = self._total_time / max(self._request_count, 1)
        error_rate = self._error_count / max(self._request_count, 1)
        
        return NetworkMetrics(
            latency=avg_latency,
            requests_per_second=self._request_count / max(self._total_time, 1),
            error_rate=error_rate,
            connection_count=len(self._session.adapters) if self._session else 0
        )
    
    def close(self):
        """Close the HTTP session."""
        if self._session:
            self._session.close()


class APIClient:
    """Main REST API client."""
    
    def __init__(self, http_client: HTTPClient):
        self.http_client = http_client
        self.logger = logging.getLogger(__name__)
        self._auth_token: Optional[str] = None
    
    def set_auth_token(self, token: str):
        """Set authentication token."""
        self._auth_token = token
        self.http_client._session.headers['Authorization'] = f'Bearer {token}'
    
    def clear_auth_token(self):
        """Clear authentication token."""
        self._auth_token = None
        if 'Authorization' in self.http_client._session.headers:
            del self.http_client._session.headers['Authorization']
    
    # Authentication endpoints
    def login(self, email: str, password: str) -> APIResponse:
        """Login user."""
        return self.http_client.request(
            RequestMethod.POST,
            APIEndpoint.AUTH.value + "/login",
            data={'email': email, 'password': password}
        )
    
    def logout(self) -> APIResponse:
        """Logout user."""
        return self.http_client.request(
            RequestMethod.POST,
            APIEndpoint.AUTH.value + "/logout"
        )
    
    def refresh_token(self, refresh_token: str) -> APIResponse:
        """Refresh authentication token."""
        return self.http_client.request(
            RequestMethod.POST,
            APIEndpoint.AUTH.value + "/refresh",
            data={'refresh_token': refresh_token}
        )
    
    # File operations
    def list_files(self, folder_id: Optional[str] = None) -> APIResponse:
        """List files in folder."""
        params = {'folder_id': folder_id} if folder_id else {}
        return self.http_client.request(
            RequestMethod.GET,
            APIEndpoint.FILES.value,
            params=params
        )
    
    def upload_file(self, file_data: bytes, filename: str, folder_id: Optional[str] = None) -> APIResponse:
        """Upload file."""
        files = {'file': (filename, file_data)}
        data = {'folder_id': folder_id} if folder_id else {}
        
        return self.http_client.request(
            RequestMethod.POST,
            APIEndpoint.UPLOAD.value,
            files=files,
            data=data
        )
    
    def download_file(self, file_id: str) -> APIResponse:
        """Download file."""
        return self.http_client.request(
            RequestMethod.GET,
            APIEndpoint.DOWNLOAD.value + f"/{file_id}"
        )
    
    def delete_file(self, file_id: str) -> APIResponse:
        """Delete file."""
        return self.http_client.request(
            RequestMethod.DELETE,
            APIEndpoint.FILES.value + f"/{file_id}"
        )
    
    # Folder operations
    def create_folder(self, name: str, parent_id: Optional[str] = None) -> APIResponse:
        """Create folder."""
        data = {'name': name}
        if parent_id:
            data['parent_id'] = parent_id
        
        return self.http_client.request(
            RequestMethod.POST,
            APIEndpoint.FOLDERS.value,
            data=data
        )
    
    def delete_folder(self, folder_id: str) -> APIResponse:
        """Delete folder."""
        return self.http_client.request(
            RequestMethod.DELETE,
            APIEndpoint.FOLDERS.value + f"/{folder_id}"
        )


class WebSocketClient:
    """Real-time updates via WebSocket."""
    
    def __init__(self, url: str):
        self.url = url
        self._ws = None
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self.logger = logging.getLogger(__name__)
        
        if not WEBSOCKET_AVAILABLE:
            self.logger.warning("WebSocket library not available")
    
    def connect(self):
        """Connect to WebSocket server."""
        if not WEBSOCKET_AVAILABLE:
            raise NetworkError("WebSocket library not available")
        
        try:
            self._ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            self._ws.run_forever()
            
        except Exception as e:
            raise NetworkError(f"WebSocket connection failed: {e}")
    
    def disconnect(self):
        """Disconnect from WebSocket server."""
        if self._ws:
            self._ws.close()
            self._connected = False
    
    def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket server."""
        if not self._connected or not self._ws:
            raise NetworkError("WebSocket not connected")
        
        try:
            self._ws.send(json.dumps(message))
        except Exception as e:
            raise NetworkError(f"Failed to send WebSocket message: {e}")
    
    def subscribe_to_event(self, event_type: str, handler: Callable):
        """Subscribe to event type."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        self.logger.info(f"Subscribed to event: {event_type}")
    
    def unsubscribe_from_event(self, event_type: str, handler: Callable):
        """Unsubscribe from event type."""
        if event_type in self._event_handlers and handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)
    
    def _on_open(self, ws):
        """WebSocket connection opened."""
        self._connected = True
        self._reconnect_attempts = 0
        self.logger.info("WebSocket connected")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            event_type = data.get('type')
            
            if event_type and event_type in self._event_handlers:
                for handler in self._event_handlers[event_type]:
                    try:
                        handler(data)
                    except Exception as e:
                        self.logger.error(f"Event handler error: {e}")
        
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON in WebSocket message")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        self.logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        self._connected = False
        self.logger.info("WebSocket disconnected")
        
        # Attempt reconnection
        if self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            self.logger.info(f"Attempting reconnection ({self._reconnect_attempts}/{self._max_reconnect_attempts})")
            time.sleep(2 ** self._reconnect_attempts)  # Exponential backoff
            self.connect()


class GraphQLClient:
    """GraphQL query support."""
    
    def __init__(self, http_client: HTTPClient, endpoint: str = "/graphql"):
        self.http_client = http_client
        self.endpoint = endpoint
        self.logger = logging.getLogger(__name__)
    
    def query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Execute GraphQL query."""
        data = {'query': query}
        if variables:
            data['variables'] = variables
        
        return self.http_client.request(
            RequestMethod.POST,
            self.endpoint,
            data=data
        )
    
    def mutation(self, mutation: str, variables: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Execute GraphQL mutation."""
        return self.query(mutation, variables)
    
    def subscription(self, subscription: str, variables: Optional[Dict[str, Any]] = None):
        """Execute GraphQL subscription (requires WebSocket)."""
        # GraphQL subscriptions typically require WebSocket
        raise NotImplementedError("GraphQL subscriptions require WebSocket implementation")


# ==============================================
# === MIDDLEWARE CLASSES ===
# ==============================================

class RetryMiddleware:
    """Intelligent retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, request: APIRequest, response: Optional[APIResponse]) -> Optional[APIResponse]:
        """Process request/response."""
        if response is None:
            # Pre-request processing
            return None
        
        # Post-request processing
        if response.status_code >= 500 and request.retries < self.max_retries:
            # Retry on server errors
            wait_time = self.backoff_factor * (2 ** request.retries)
            time.sleep(wait_time)
            
            request.retries += 1
            self.logger.info(f"Retrying request (attempt {request.retries})")
            
            # Return None to indicate retry needed
            return None
        
        return response


class CacheMiddleware:
    """Response caching middleware."""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[APIResponse, float]] = {}
        self._cache_order: List[str] = []
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, request: APIRequest, response: Optional[APIResponse]) -> Optional[APIResponse]:
        """Process request/response."""
        cache_key = self._generate_cache_key(request)
        
        if response is None:
            # Pre-request: check cache
            if request.method == RequestMethod.GET:
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    cached_response.cached = True
                    self.logger.debug(f"Cache hit for {cache_key}")
                    return cached_response
            return None
        
        # Post-request: store in cache
        if request.method == RequestMethod.GET and response.status_code == 200:
            self._store_response(cache_key, response)
        
        return response
    
    def _generate_cache_key(self, request: APIRequest) -> str:
        """Generate cache key for request."""
        key_parts = [
            request.method.value,
            request.endpoint,
            json.dumps(request.params, sort_keys=True) if request.params else "",
            json.dumps(request.data, sort_keys=True) if isinstance(request.data, dict) else ""
        ]
        return "|".join(key_parts)
    
    def _get_cached_response(self, cache_key: str) -> Optional[APIResponse]:
        """Get cached response if valid."""
        if cache_key not in self._cache:
            return None
        
        response, timestamp = self._cache[cache_key]
        
        # Check if cache entry is still valid
        if time.time() - timestamp > self.ttl:
            self._remove_from_cache(cache_key)
            return None
        
        return response
    
    def _store_response(self, cache_key: str, response: APIResponse):
        """Store response in cache."""
        # Remove old entry if exists
        if cache_key in self._cache:
            self._remove_from_cache(cache_key)
        
        # Check cache size limit
        while len(self._cache) >= self.max_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
        
        # Store new entry
        self._cache[cache_key] = (response, time.time())
        self._cache_order.append(cache_key)
    
    def _remove_from_cache(self, cache_key: str):
        """Remove entry from cache."""
        if cache_key in self._cache:
            del self._cache[cache_key]
        if cache_key in self._cache_order:
            self._cache_order.remove(cache_key)
    
    def clear_cache(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._cache_order.clear()


class AuthMiddleware:
    """Automatic authentication injection."""
    
    def __init__(self, get_token_func: Callable[[], Optional[str]]):
        self.get_token_func = get_token_func
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, request: APIRequest, response: Optional[APIResponse]) -> Optional[APIRequest]:
        """Process request/response."""
        if response is not None:
            # Post-request: handle auth errors
            if response.status_code == 401:
                self.logger.warning("Authentication failed - token may be expired")
            return None
        
        # Pre-request: inject auth token
        token = self.get_token_func()
        if token:
            request.headers['Authorization'] = f'Bearer {token}'
        
        return request


class LoggingMiddleware:
    """Request/response logging."""
    
    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, request: APIRequest, response: Optional[APIResponse]) -> None:
        """Process request/response."""
        if response is None:
            # Pre-request logging
            self.logger.log(
                self.log_level,
                f"Request: {request.method.value} {request.endpoint}"
            )
        else:
            # Post-request logging
            self.logger.log(
                self.log_level,
                f"Response: {response.status_code} ({response.elapsed_time:.3f}s)"
            )


class CompressionMiddleware:
    """Data compression middleware."""
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, request: APIRequest, response: Optional[APIResponse]) -> Optional[APIRequest]:
        """Process request/response."""
        if response is not None:
            return None
        
        # Pre-request: add compression headers
        request.headers['Accept-Encoding'] = 'gzip, deflate'
        
        return request


# ==============================================
# === NETWORK OPTIMIZATION ===
# ==============================================

class ConnectionPool:
    """HTTP connection pooling."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._connections: List[HTTPClient] = []
        self._available: List[HTTPClient] = []
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def get_connection(self) -> HTTPClient:
        """Get available connection from pool."""
        with self._lock:
            if self._available:
                return self._available.pop()
            
            if len(self._connections) < self.max_connections:
                connection = HTTPClient()
                self._connections.append(connection)
                return connection
            
            # Wait for available connection
            # In a real implementation, this would use a queue with timeout
            return self._connections[0]  # Fallback
    
    def return_connection(self, connection: HTTPClient):
        """Return connection to pool."""
        with self._lock:
            if connection in self._connections and connection not in self._available:
                self._available.append(connection)
    
    def close_all(self):
        """Close all connections in pool."""
        with self._lock:
            for connection in self._connections:
                connection.close()
            self._connections.clear()
            self._available.clear()


class BandwidthMonitor:
    """Monitor bandwidth usage."""
    
    def __init__(self):
        self._bytes_sent = 0
        self._bytes_received = 0
        self._start_time = time.time()
        self._lock = threading.Lock()
    
    def record_sent(self, bytes_count: int):
        """Record bytes sent."""
        with self._lock:
            self._bytes_sent += bytes_count
    
    def record_received(self, bytes_count: int):
        """Record bytes received."""
        with self._lock:
            self._bytes_received += bytes_count
    
    def get_upload_speed(self) -> float:
        """Get upload speed in bytes per second."""
        with self._lock:
            elapsed = time.time() - self._start_time
            return self._bytes_sent / max(elapsed, 1)
    
    def get_download_speed(self) -> float:
        """Get download speed in bytes per second."""
        with self._lock:
            elapsed = time.time() - self._start_time
            return self._bytes_received / max(elapsed, 1)
    
    def reset_stats(self):
        """Reset bandwidth statistics."""
        with self._lock:
            self._bytes_sent = 0
            self._bytes_received = 0
            self._start_time = time.time()


class NetworkConditionAdapter:
    """Adapt to network conditions."""
    
    def __init__(self):
        self._current_condition = NetworkCondition.GOOD
        self._metrics_history: List[NetworkMetrics] = []
        self.logger = logging.getLogger(__name__)
    
    def update_metrics(self, metrics: NetworkMetrics):
        """Update network metrics and adapt settings."""
        self._metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self._metrics_history) > 10:
            self._metrics_history.pop(0)
        
        # Determine network condition
        old_condition = self._current_condition
        self._current_condition = self._determine_condition(metrics)
        
        if old_condition != self._current_condition:
            self.logger.info(f"Network condition changed: {old_condition.value} -> {self._current_condition.value}")
            self._adapt_to_condition()
    
    def _determine_condition(self, metrics: NetworkMetrics) -> NetworkCondition:
        """Determine network condition based on metrics."""
        if metrics.error_rate > 0.2:  # 20% error rate
            return NetworkCondition.OFFLINE
        elif metrics.latency > 2.0 or metrics.error_rate > 0.1:  # 2s latency or 10% errors
            return NetworkCondition.POOR
        elif metrics.latency > 0.5:  # 500ms latency
            return NetworkCondition.GOOD
        else:
            return NetworkCondition.EXCELLENT
    
    def _adapt_to_condition(self):
        """Adapt settings based on network condition."""
        if self._current_condition == NetworkCondition.POOR:
            # Reduce timeout, increase retries
            self.logger.info("Adapting to poor network: reducing timeout, increasing retries")
        elif self._current_condition == NetworkCondition.EXCELLENT:
            # Increase timeouts, enable optimizations
            self.logger.info("Adapting to excellent network: enabling optimizations")
    
    def get_current_condition(self) -> NetworkCondition:
        """Get current network condition."""
        return self._current_condition


# ==============================================
# === PROXY AND VPN MANAGEMENT ===
# ==============================================

class ProxyManager:
    """Proxy configuration and management."""
    
    def __init__(self):
        self._proxy_config: Optional[ProxyConfig] = None
        self.logger = logging.getLogger(__name__)
    
    def set_proxy(self, host: str, port: int, proxy_type: str = "http",
                  username: Optional[str] = None, password: Optional[str] = None):
        """Set proxy configuration."""
        self._proxy_config = ProxyConfig(
            enabled=True,
            host=host,
            port=port,
            proxy_type=proxy_type,
            username=username,
            password=password
        )
        
        self.logger.info(f"Proxy configured: {proxy_type}://{host}:{port}")
    
    def disable_proxy(self):
        """Disable proxy."""
        if self._proxy_config:
            self._proxy_config.enabled = False
            self.logger.info("Proxy disabled")
    
    def get_proxy_dict(self) -> Optional[Dict[str, str]]:
        """Get proxy configuration for requests."""
        if not self._proxy_config or not self._proxy_config.enabled:
            return None
        
        proxy_url = f"{self._proxy_config.proxy_type}://"
        
        if self._proxy_config.username and self._proxy_config.password:
            proxy_url += f"{self._proxy_config.username}:{self._proxy_config.password}@"
        
        proxy_url += f"{self._proxy_config.host}:{self._proxy_config.port}"
        
        return {
            'http': proxy_url,
            'https': proxy_url
        }


# ==============================================
# === EVENT SYSTEM ===
# ==============================================

class EventBus:
    """Central event bus/dispatcher."""
    
    def __init__(self):
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type."""
        with self._lock:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            
            self._event_handlers[event_type].append(handler)
        
        self.logger.debug(f"Subscribed to event: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from event type."""
        with self._lock:
            if event_type in self._event_handlers and handler in self._event_handlers[event_type]:
                self._event_handlers[event_type].remove(handler)
    
    def publish(self, event_type: str, data: Any):
        """Publish event to subscribers."""
        with self._lock:
            handlers = self._event_handlers.get(event_type, []).copy()
        
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"Event handler error for {event_type}: {e}")
    
    def clear_handlers(self, event_type: Optional[str] = None):
        """Clear event handlers."""
        with self._lock:
            if event_type:
                self._event_handlers[event_type] = []
            else:
                self._event_handlers.clear()


class EventEmitter:
    """Event publishing interface."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
    
    def emit(self, event_type: str, data: Any):
        """Emit event."""
        self.event_bus.publish(event_type, data)
        self.logger.debug(f"Emitted event: {event_type}")


class EventListener:
    """Event subscription interface."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._subscriptions: List[Tuple[str, Callable]] = []
        self.logger = logging.getLogger(__name__)
    
    def on(self, event_type: str, handler: Callable):
        """Subscribe to event."""
        self.event_bus.subscribe(event_type, handler)
        self._subscriptions.append((event_type, handler))
    
    def off(self, event_type: str, handler: Callable):
        """Unsubscribe from event."""
        self.event_bus.unsubscribe(event_type, handler)
        if (event_type, handler) in self._subscriptions:
            self._subscriptions.remove((event_type, handler))
    
    def remove_all_listeners(self):
        """Remove all event listeners."""
        for event_type, handler in self._subscriptions:
            self.event_bus.unsubscribe(event_type, handler)
        self._subscriptions.clear()


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Enums
    'RequestMethod', 'APIEndpoint', 'EventType', 'NetworkCondition',
    
    # Data Classes
    'APIRequest', 'APIResponse', 'NetworkMetrics', 'ProxyConfig',
    
    # Core Clients
    'HTTPClient', 'APIClient', 'WebSocketClient', 'GraphQLClient',
    
    # Middleware
    'RetryMiddleware', 'CacheMiddleware', 'AuthMiddleware', 'LoggingMiddleware', 'CompressionMiddleware',
    
    # Optimization
    'ConnectionPool', 'BandwidthMonitor', 'NetworkConditionAdapter', 'ProxyManager',
    
    # Event System
    'EventBus', 'EventEmitter', 'EventListener'
]