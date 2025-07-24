"""
Network and Communication module for MegaPythonLibrary.

This module contains:
- HTTP session management and connection pooling
- API request handling with retry logic
- Rate limiting and throttling
- Upload and download URL management
- Network performance monitoring
- Request caching
"""

import json
import time
import random
import logging
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter

from .utils import RequestError, ValidationError, raise_mega_error
from .monitor import record_performance, get_logger

# ==============================================
# === CONFIGURATION AND CONSTANTS ===
# ==============================================

# Mega API endpoints
MEGA_API_URL = "https://g.api.mega.co.nz/cs"
MEGA_UPLOAD_URL = "https://eu.api.mega.co.nz/ufa"

# Request configuration
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 16.0

# Request headers
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}


# ==============================================
# === RATE LIMITING ===
# ==============================================

class RateLimiter:
    """
    Simple rate limiter for API requests.
    """
    
    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.last_request_time = 0.0
        self.logger = get_logger("rate_limiter")
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def set_rate(self, requests_per_second: float) -> None:
        """Update the rate limit."""
        self.requests_per_second = requests_per_second
        self.logger.info(f"Rate limit updated to {requests_per_second} requests/second")


# ==============================================
# === HTTP SESSION MANAGEMENT ===
# ==============================================

class APISession:
    """
    Manages HTTP session for API requests with automatic retry and error handling.
    Enhanced with connection pooling and performance optimizations.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.logger = get_logger("api_session")
        
        # Configure connection pooling for better performance
        adapter = HTTPAdapter(
            pool_connections=10,    # Number of connection pools
            pool_maxsize=20,       # Maximum connections per pool
            max_retries=0,         # We handle retries manually
            pool_block=False       # Don't block on pool exhaustion
        )
        
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        
        # Set headers
        self.session.headers.update(DEFAULT_HEADERS)
        
        # Session state
        self.sequence_number = random.randint(0, 0x7FFFFFFF)
        self.sid = None  # Session ID after login
        
        # Request caching
        self._request_cache = {}
        self._cache_timeout = 30  # seconds
        
        # Performance tracking
        self.request_count = 0
        self.total_request_time = 0.0
        
        self.logger.info("API session initialized")
    
    def get_sequence_number(self) -> int:
        """Get next sequence number for API requests."""
        self.sequence_number += 1
        return self.sequence_number
    
    def set_session_id(self, sid: str) -> None:
        """Set session ID after successful login."""
        self.sid = sid
        self.logger.info("Session ID set")
    
    def clear_session_id(self) -> None:
        """Clear session ID on logout."""
        self.sid = None
        self.logger.info("Session ID cleared")
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics for this session."""
        avg_time = self.total_request_time / max(1, self.request_count)
        return {
            'total_requests': self.request_count,
            'total_time': self.total_request_time,
            'average_request_time': avg_time,
            'cache_entries': len(self._request_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the request cache."""
        self._request_cache.clear()
        self.logger.info("Request cache cleared")
    
    def close(self) -> None:
        """Close the session."""
        self.session.close()
        self._request_cache.clear()
        self.logger.info("API session closed")
    
    def _check_cache(self, cache_key: str) -> Optional[requests.Response]:
        """Check if we have a cached response for this key."""
        if cache_key in self._request_cache:
            cached_data, timestamp = self._request_cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                self.logger.debug(f"Cache hit for {cache_key}")
                return cached_data
            else:
                # Remove expired cache entry
                del self._request_cache[cache_key]
                self.logger.debug(f"Cache expired for {cache_key}")
        return None
    
    def _cache_response(self, cache_key: str, response: requests.Response) -> None:
        """Cache a response for future use."""
        self._request_cache[cache_key] = (response, time.time())
        self.logger.debug(f"Cached response for {cache_key}")


# Global session instance
_api_session = APISession()

# Global rate limiter
_rate_limiter = RateLimiter(requests_per_second=0.5)  # Conservative rate


# ==============================================
# === HTTP REQUEST FUNCTIONS ===
# ==============================================

def make_request(url: str, data: Any = None, method: str = 'POST', 
                timeout: float = DEFAULT_TIMEOUT, cache_key: str = None, **kwargs) -> requests.Response:
    """
    Make HTTP request with retry logic, error handling, and performance optimizations.
    
    Args:
        url: The URL to request
        data: Request data (will be JSON encoded if dict)
        method: HTTP method
        timeout: Request timeout
        cache_key: Optional cache key for GET requests
        **kwargs: Additional arguments for requests
        
    Returns:
        Response object
        
    Raises:
        RequestError: If request fails after retries
    """
    logger = get_logger("requests")
    start_time = time.time()
    
    # Check cache for GET requests
    if method == 'GET' and cache_key:
        cached_response = _api_session._check_cache(cache_key)
        if cached_response:
            return cached_response
    
    retries = 0
    delay = RETRY_DELAY
    
    while retries <= MAX_RETRIES:
        try:
            # Prepare request data
            if isinstance(data, (dict, list)):
                kwargs['json'] = data
            elif data is not None:
                kwargs['data'] = data
            
            # Make request
            response = _api_session.session.request(
                method=method,
                url=url,
                timeout=timeout,
                **kwargs
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Cache successful GET responses
            if method == 'GET' and cache_key and response.status_code == 200:
                _api_session._cache_response(cache_key, response)
            
            # Track performance
            request_time = time.time() - start_time
            _api_session.request_count += 1
            _api_session.total_request_time += request_time
            record_performance('network_request_time', request_time)
            
            if request_time > 1.0:  # Log slow requests
                logger.warning(f"Slow request: {method} {url} took {request_time:.2f}s")
            
            return response
            
        except requests.exceptions.Timeout:
            if retries >= MAX_RETRIES:
                raise RequestError(f"Request timeout after {MAX_RETRIES} retries")
            
        except requests.exceptions.ConnectionError as e:
            if retries >= MAX_RETRIES:
                raise RequestError(f"Connection error: {e}")
            
        except requests.exceptions.HTTPError as e:
            # Don't retry client errors (4xx)
            if 400 <= e.response.status_code < 500:
                raise RequestError(f"HTTP {e.response.status_code}: {e.response.text}")
            
            # Retry server errors (5xx)
            if retries >= MAX_RETRIES:
                raise RequestError(f"HTTP {e.response.status_code}: {e.response.text}")
        
        except Exception as e:
            if retries >= MAX_RETRIES:
                raise RequestError(f"Request failed: {e}")
        
        # Wait before retry
        retries += 1
        if retries <= MAX_RETRIES:
            logger.warning(f"Request failed, retrying in {delay}s (attempt {retries}/{MAX_RETRIES})")
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)  # Exponential backoff
    
    raise RequestError("Maximum retries exceeded")


# ==============================================
# === MEGA API FUNCTIONS ===
# ==============================================

def api_request(commands: List[Dict[str, Any]], sid: Optional[str] = None) -> List[Any]:
    """
    Make Mega API request with command list.
    
    Args:
        commands: List of API command dictionaries
        sid: Session ID (if authenticated)
        
    Returns:
        List of API responses
        
    Raises:
        RequestError: If API request fails
        ValidationError: If response is invalid
    """
    logger = get_logger("api")
    
    # Build request URL
    url = MEGA_API_URL
    params = {'id': _api_session.get_sequence_number()}
    
    if sid:
        params['sid'] = sid
    elif _api_session.sid:
        params['sid'] = _api_session.sid
    
    logger.debug(f"Making API request with {len(commands)} commands")
    
    # Make request
    response = make_request(url, data=commands, params=params)
    
    # Parse response
    try:
        result = response.json()
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON response: {e}")
    
    # Check for API errors
    if isinstance(result, list):
        for i, item in enumerate(result):
            if isinstance(item, int) and item < 0:
                logger.error(f"API error in command {i}: {item}")
                raise_mega_error(item)
    elif isinstance(result, int) and result < 0:
        logger.error(f"API error: {result}")
        raise_mega_error(result)
    
    logger.debug(f"API request completed successfully")
    return result


def single_api_request(command: Dict[str, Any], sid: Optional[str] = None) -> Any:
    """
    Make single Mega API request.
    
    Args:
        command: API command dictionary
        sid: Session ID (if authenticated)
        
    Returns:
        API response
    """
    result = api_request([command], sid)
    return result[0] if result else None


def rate_limited_request(*args, **kwargs):
    """
    Make rate-limited API request.
    
    Args/kwargs passed to api_request()
    """
    _rate_limiter.wait_if_needed()
    return api_request(*args, **kwargs)


# ==============================================
# === UPLOAD/DOWNLOAD URL MANAGEMENT ===
# ==============================================

def get_upload_url(size: int) -> str:
    """
    Get upload URL for file of given size.
    
    Args:
        size: File size in bytes
        
    Returns:
        Upload URL
        
    Raises:
        RequestError: If unable to get upload URL
    """
    command = {'a': 'u', 's': size}
    result = single_api_request(command)
    
    if not isinstance(result, dict) or 'p' not in result:
        raise RequestError("Invalid upload URL response")
    
    return result['p']


def get_download_url(node_id: str) -> str:
    """
    Get download URL for node.
    
    Args:
        node_id: The node ID to download
        
    Returns:
        Download URL
        
    Raises:
        RequestError: If unable to get download URL
    """
    command = {'a': 'g', 'g': 1, 'n': node_id}
    result = single_api_request(command)
    
    if not isinstance(result, dict) or 'g' not in result:
        raise RequestError("Invalid download URL response")
    
    return result['g']


# ==============================================
# === CHUNK TRANSFER FUNCTIONS ===
# ==============================================

def upload_chunk(url: str, chunk_data: bytes, chunk_start: int) -> Dict[str, Any]:
    """
    Upload file chunk to Mega.
    
    Args:
        url: Upload URL
        chunk_data: Chunk data to upload
        chunk_start: Starting position of chunk
        
    Returns:
        Upload response
        
    Raises:
        RequestError: If upload fails
    """
    headers = {
        'Content-Type': 'application/octet-stream',
        'Content-Length': str(len(chunk_data)),
    }
    
    # Add chunk position to URL
    chunk_url = f"{url}/{chunk_start}"
    
    response = make_request(chunk_url, data=chunk_data, method='POST', headers=headers)
    
    try:
        return response.json()
    except json.JSONDecodeError:
        # Some responses may not be JSON
        return {'status': 'ok'}


def download_chunk(url: str, start: int, end: int) -> bytes:
    """
    Download file chunk from Mega.
    
    Args:
        url: Download URL
        start: Start byte position
        end: End byte position
        
    Returns:
        Chunk data
        
    Raises:
        RequestError: If download fails
    """
    headers = {'Range': f'bytes={start}-{end}'}
    
    response = make_request(url, method='GET', headers=headers)
    return response.content


# ==============================================
# === SESSION AND CONFIGURATION MANAGEMENT ===
# ==============================================

def set_session_id(sid: str) -> None:
    """Set the global session ID."""
    _api_session.set_session_id(sid)


def clear_session_id() -> None:
    """Clear the global session ID."""
    _api_session.clear_session_id()


def get_session_id() -> Optional[str]:
    """Get the current session ID."""
    return _api_session.sid


def set_rate_limit(requests_per_second: float) -> None:
    """Set the global rate limit."""
    _rate_limiter.set_rate(requests_per_second)


def get_network_performance_stats() -> dict:
    """Get network performance statistics."""
    return _api_session.get_performance_stats()


def clear_network_cache() -> None:
    """Clear the network request cache."""
    _api_session.clear_cache()


def close_network_session() -> None:
    """Close the network session."""
    _api_session.close()


# ==============================================
# === BANDWIDTH THROTTLING ===
# ==============================================

class BandwidthThrottler:
    """Throttle bandwidth usage for uploads/downloads."""
    
    def __init__(self, max_bytes_per_second: Optional[int] = None):
        self.max_bytes_per_second = max_bytes_per_second
        self.last_update_time = time.time()
        self.bytes_transferred = 0
        self.logger = get_logger("bandwidth")
    
    def throttle(self, bytes_transferred: int) -> None:
        """Apply bandwidth throttling if configured."""
        if not self.max_bytes_per_second:
            return
        
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        
        self.bytes_transferred += bytes_transferred
        
        # Calculate expected time for bytes transferred
        expected_time = self.bytes_transferred / self.max_bytes_per_second
        
        # Sleep if we're going too fast
        if time_elapsed < expected_time:
            sleep_time = expected_time - time_elapsed
            self.logger.debug(f"Throttling: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        # Reset counters periodically
        if time_elapsed > 1.0:
            self.last_update_time = current_time
            self.bytes_transferred = 0


# Global bandwidth throttler (disabled by default)
_bandwidth_throttler = BandwidthThrottler()


def set_bandwidth_limit(max_bytes_per_second: Optional[int]) -> None:
    """Set global bandwidth limit."""
    global _bandwidth_throttler
    _bandwidth_throttler = BandwidthThrottler(max_bytes_per_second)
    logger = get_logger("bandwidth")
    if max_bytes_per_second:
        logger.info(f"Bandwidth limit set to {max_bytes_per_second} bytes/second")
    else:
        logger.info("Bandwidth limit disabled")


def apply_bandwidth_throttling(bytes_transferred: int) -> None:
    """Apply bandwidth throttling for the given number of bytes."""
    _bandwidth_throttler.throttle(bytes_transferred)