"""
MegaSerpentClient - Shared Utilities Module

Purpose: Common utilities, helpers, and shared functionality used across all modules.

This module provides the foundational utilities that all other modules depend on,
including constants, exceptions, validators, converters, and helper functions.
"""

import logging
import re
import hashlib
import base64
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import asyncio
from functools import wraps
import json


# ==============================================
# === CONSTANTS AND ENUMS ===
# ==============================================

class Constants:
    """Application constants and configuration values."""
    
    # API Configuration
    API_BASE_URL = "https://g.api.mega.co.nz"
    USER_AGENT = "MegaSerpentClient/1.0"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # File Limits
    MAX_FILE_SIZE = 50 * 1024 * 1024 * 1024  # 50GB
    CHUNK_SIZE = 1024 * 1024  # 1MB
    
    # Encryption
    AES_BLOCK_SIZE = 16
    KEY_SIZE = 32  # 256-bit keys
    
    # Logging Levels (6 levels as specified)
    LOG_LEVELS = {
        'FATAL': 50,
        'ERROR': 40,
        'WARN': 30,
        'INFO': 20,
        'DEBUG': 10,
        'MAX_VERBOSE': 5
    }


class LogLevel(Enum):
    """Logging levels with 6 levels as specified in architecture."""
    FATAL = 50
    ERROR = 40
    WARN = 30
    INFO = 20
    DEBUG = 10
    MAX_VERBOSE = 5


class FileType(Enum):
    """File type classifications."""
    FILE = 0
    FOLDER = 1
    ROOT = 2
    INBOX = 3
    TRASH = 4


class ShareType(Enum):
    """Share type classifications."""
    PRIVATE = 0
    PUBLIC = 1
    TEAM = 2
    ENTERPRISE = 3


class SyncDirection(Enum):
    """Synchronization direction options."""
    UP = "up"
    DOWN = "down"
    BIDIRECTIONAL = "bidirectional"


# ==============================================
# === CUSTOM EXCEPTIONS ===
# ==============================================

class MegaError(Exception):
    """Base exception class for all MEGA-related errors."""
    
    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class AuthenticationError(MegaError):
    """Authentication-related errors."""
    pass


class NetworkError(MegaError):
    """Network and communication errors."""
    pass


class StorageError(MegaError):
    """Storage and file operation errors."""
    pass


class SyncError(MegaError):
    """Synchronization errors."""
    pass


class ValidationError(MegaError):
    """Data validation errors."""
    pass


class ConfigurationError(MegaError):
    """Configuration and setup errors."""
    pass


class SecurityError(MegaError):
    """Security and cryptography errors."""
    pass


# ==============================================
# === VALIDATORS ===
# ==============================================

class Validators:
    """Common validation functions."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, bool]:
        """Validate password strength."""
        return {
            'length': len(password) >= 8,
            'uppercase': any(c.isupper() for c in password),
            'lowercase': any(c.islower() for c in password),
            'digit': any(c.isdigit() for c in password),
            'special': any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        }
    
    @staticmethod
    def validate_file_path(path: str) -> bool:
        """Validate file path format."""
        if not path or not isinstance(path, str):
            return False
        # Basic path validation - starts with / and doesn't contain invalid chars
        return path.startswith('/') and not any(c in path for c in '<>:"|?*')
    
    @staticmethod
    def validate_node_handle(handle: str) -> bool:
        """Validate MEGA node handle format."""
        if not handle or not isinstance(handle, str):
            return False
        # MEGA handles are typically 8 characters, base64-like
        return len(handle) == 8 and handle.replace('-', '').replace('_', '').isalnum()


# ==============================================
# === CONVERTERS ===
# ==============================================

class Converters:
    """Data type conversion utilities."""
    
    @staticmethod
    def bytes_to_base64url(data: bytes) -> str:
        """Convert bytes to base64url string."""
        return base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')
    
    @staticmethod
    def base64url_to_bytes(data: str) -> bytes:
        """Convert base64url string to bytes."""
        # Add padding if necessary
        padding = 4 - (len(data) % 4)
        if padding != 4:
            data += '=' * padding
        return base64.urlsafe_b64decode(data.encode('ascii'))
    
    @staticmethod
    def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
        """Convert timestamp to datetime object."""
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    
    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> int:
        """Convert datetime to timestamp."""
        return int(dt.timestamp())


# ==============================================
# === FORMATTERS ===
# ==============================================

class Formatters:
    """Data formatting utilities."""
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = 0
        size = float(size_bytes)
        
        while size >= 1024.0 and i < len(size_names) - 1:
            size /= 1024.0
            i += 1
        
        return f"{size:.1f} {size_names[i]}"
    
    @staticmethod
    def format_duration(seconds: Union[int, float]) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    @staticmethod
    def format_percentage(current: Union[int, float], total: Union[int, float]) -> str:
        """Format percentage progress."""
        if total == 0:
            return "0.0%"
        return f"{(current / total) * 100:.1f}%"


# ==============================================
# === HELPER FUNCTIONS ===
# ==============================================

class Helpers:
    """General helper functions."""
    
    @staticmethod
    def generate_request_id() -> str:
        """Generate unique request ID."""
        return secrets.token_hex(8)
    
    @staticmethod
    def safe_json_loads(data: str) -> Optional[Dict]:
        """Safely parse JSON data."""
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    
    @staticmethod
    def safe_json_dumps(data: Any) -> str:
        """Safely serialize data to JSON."""
        try:
            return json.dumps(data, default=str)
        except (TypeError, ValueError):
            return "{}"
    
    @staticmethod
    def chunks(lst: List[Any], n: int) -> List[List[Any]]:
        """Split list into chunks of size n."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    @staticmethod
    def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(Helpers.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# ==============================================
# === DECORATORS ===
# ==============================================

class Decorators:
    """Common decorators for retry, cache, etc."""
    
    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Retry decorator with exponential backoff."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                attempt = 0
                current_delay = delay
                
                while attempt < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempt += 1
                        if attempt >= max_attempts:
                            raise e
                        
                        time.sleep(current_delay)
                        current_delay *= backoff
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Async retry decorator with exponential backoff."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                attempt = 0
                current_delay = delay
                
                while attempt < max_attempts:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        attempt += 1
                        if attempt >= max_attempts:
                            raise e
                        
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def timing(func: Callable) -> Callable:
        """Decorator to measure function execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            logger = logging.getLogger(__name__)
            logger.debug(f"{func.__name__} executed in {end_time - start_time:.3f}s")
            return result
        return wrapper


# ==============================================
# === ASYNC HELPERS ===
# ==============================================

class AsyncHelpers:
    """Async/await helper functions."""
    
    @staticmethod
    async def run_in_executor(func: Callable, *args, **kwargs):
        """Run synchronous function in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    @staticmethod
    async def gather_with_limit(limit: int, *tasks):
        """Run tasks with concurrency limit."""
        semaphore = asyncio.Semaphore(limit)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[limited_task(task) for task in tasks])


# ==============================================
# === DATE/TIME UTILITIES ===
# ==============================================

class DateTimeUtils:
    """Date and time utility functions."""
    
    @staticmethod
    def now_utc() -> datetime:
        """Get current UTC datetime."""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def iso_format(dt: datetime) -> str:
        """Format datetime as ISO string."""
        return dt.isoformat()
    
    @staticmethod
    def parse_iso(iso_string: str) -> datetime:
        """Parse ISO datetime string."""
        return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    
    @staticmethod
    def age_in_seconds(dt: datetime) -> float:
        """Get age of datetime in seconds."""
        return (DateTimeUtils.now_utc() - dt).total_seconds()


# ==============================================
# === FILE UTILITIES ===
# ==============================================

class FileUtils:
    """File system utility functions."""
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension."""
        return filename.lower().split('.')[-1] if '.' in filename else ''
    
    @staticmethod
    def is_image_file(filename: str) -> bool:
        """Check if file is an image."""
        image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg'}
        return FileUtils.get_file_extension(filename) in image_extensions
    
    @staticmethod
    def is_video_file(filename: str) -> bool:
        """Check if file is a video."""
        video_extensions = {'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm'}
        return FileUtils.get_file_extension(filename) in video_extensions
    
    @staticmethod
    def is_audio_file(filename: str) -> bool:
        """Check if file is audio."""
        audio_extensions = {'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a'}
        return FileUtils.get_file_extension(filename) in audio_extensions
    
    @staticmethod
    def is_document_file(filename: str) -> bool:
        """Check if file is a document."""
        doc_extensions = {'pdf', 'doc', 'docx', 'txt', 'rtf', 'odt'}
        return FileUtils.get_file_extension(filename) in doc_extensions


# ==============================================
# === STRING UTILITIES ===
# ==============================================

class StringUtils:
    """String manipulation utilities."""
    
    @staticmethod
    def truncate(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate string to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for filesystem."""
        # Remove/replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename.strip()
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize file path."""
        if not path.startswith('/'):
            path = '/' + path
        
        # Remove double slashes
        while '//' in path:
            path = path.replace('//', '/')
        
        # Remove trailing slash unless root
        if len(path) > 1 and path.endswith('/'):
            path = path[:-1]
        
        return path


# ==============================================
# === CRYPTO UTILITIES ===
# ==============================================

class CryptoUtils:
    """Cryptographic utility functions."""
    
    @staticmethod
    def generate_random_bytes(length: int) -> bytes:
        """Generate cryptographically secure random bytes."""
        return secrets.token_bytes(length)
    
    @staticmethod
    def generate_random_key(length: int = 32) -> bytes:
        """Generate random encryption key."""
        return CryptoUtils.generate_random_bytes(length)
    
    @staticmethod
    def hash_sha256(data: Union[str, bytes]) -> str:
        """Calculate SHA-256 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, bytes]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Simple PBKDF2 implementation
        import hashlib
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return key.hex(), salt


# ==============================================
# === NETWORK UTILITIES ===
# ==============================================

class NetworkUtils:
    """Network utility functions."""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate URL format."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(url))
    
    @staticmethod
    def extract_domain(url: str) -> Optional[str]:
        """Extract domain from URL."""
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc
        except Exception:
            return None


# ==============================================
# === SERIALIZATION UTILITIES ===
# ==============================================

class SerializationUtils:
    """Serialization/deserialization utilities."""
    
    @staticmethod
    def serialize_object(obj: Any) -> str:
        """Serialize object to JSON string."""
        def default_serializer(o):
            if isinstance(o, datetime):
                return o.isoformat()
            elif hasattr(o, '__dict__'):
                return o.__dict__
            return str(o)
        
        return json.dumps(obj, default=default_serializer, indent=2)
    
    @staticmethod
    def deserialize_object(data: str) -> Any:
        """Deserialize JSON string to object."""
        return json.loads(data)


# ==============================================
# === MODULE EXPORT ===
# ==============================================

__all__ = [
    # Constants and Enums
    'Constants', 'LogLevel', 'FileType', 'ShareType', 'SyncDirection',
    
    # Exceptions
    'MegaError', 'AuthenticationError', 'NetworkError', 'StorageError',
    'SyncError', 'ValidationError', 'ConfigurationError', 'SecurityError',
    
    # Utility Classes
    'Validators', 'Converters', 'Formatters', 'Helpers', 'Decorators',
    'AsyncHelpers', 'DateTimeUtils', 'FileUtils', 'StringUtils',
    'CryptoUtils', 'NetworkUtils', 'SerializationUtils'
]