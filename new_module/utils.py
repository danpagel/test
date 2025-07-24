"""
Shared utilities and helpers for MegaPythonLibrary.

This module contains:
- Version information
- Exception classes
- Validation utilities  
- Basic utility functions
- Constants and configuration
- Crypto utility functions
- File type detection
"""

import sys
import subprocess
import math
import re
import json
import hashlib
import time
import os
import random
import binascii
import tempfile
import shutil
import fnmatch
import mimetypes
import uuid
import codecs
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Auto-install missing dependencies
def _install_and_import(package: str, pip_name: Optional[str] = None) -> None:
    """
    Automatically install missing packages and import them.
    
    Args:
        package: The package name to import
        pip_name: The pip package name (if different from import name)
    """
    try:
        __import__(package)
    except ImportError:
        pip_name = pip_name or package
        print(f"Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
        __import__(package)

# Install required packages if not available
try:
    import Crypto
except ImportError:
    _install_and_import("pycryptodome", "pycryptodome")
    # Try alternative import path
    try:
        import Crypto
    except ImportError:
        # Some systems need this path
        import sys
        import os
        site_packages = next(p for p in sys.path if 'site-packages' in p)
        sys.path.insert(0, os.path.join(site_packages, 'Crypto'))

try:
    import requests
except ImportError:
    _install_and_import("requests", "requests")

# Import crypto modules after ensuring they're available
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Util import Counter
import base64
import struct

# ==============================================
# === VERSION INFORMATION ===
# ==============================================

__version__ = "2.5.0-modular"
__author__ = "MegaPythonLibrary Team"
__email__ = "contact@megapythonlibrary.dev"
__license__ = "MIT"
__status__ = "Production"

# ==============================================
# === VERSION COMPATIBILITY ===
# ==============================================

# Python 2/3 compatibility functions for byte handling
if sys.version_info < (3,):
    def makebyte(x: str) -> str:
        return x
    
    def makestring(x: str) -> str:
        return x
else:
    def makebyte(x: str) -> bytes:
        return codecs.latin_1_encode(x)[0]
    
    def makestring(x: bytes) -> str:
        return codecs.latin_1_decode(x)[0]

# ==============================================
# === EXCEPTION CLASSES ===
# ==============================================

class ValidationError(Exception):
    """
    Raised when input validation fails.
    
    This exception is raised when user-provided data doesn't meet
    the required format or constraints (e.g., invalid email format,
    weak password, etc.).
    """
    pass


class RequestError(Exception):
    """
    Raised when an API request fails.
    
    This exception is raised when a request to MEGA's API fails,
    either due to network issues, invalid parameters, or server errors.
    """
    pass


class MPLError(Exception):
    """Base exception class for all MegaPythonLibrary-specific errors."""
    pass


class AuthenticationError(MPLError):
    """Raised when authentication fails or user is not logged in."""
    pass


class CryptoError(MPLError):
    """Raised when cryptographic operations fail."""
    pass


class NetworkError(MPLError):
    """Raised when network operations fail."""
    pass


class BusinessError(MPLError):
    """Raised for business logic violations (quota exceeded, etc.)."""
    pass


class PaymentError(MPLError):
    """Raised for payment-related errors."""
    pass


class FUSEError(MPLError):
    """Raised for FUSE filesystem errors."""
    pass


class LocalError(MPLError):
    """Raised for local filesystem errors."""
    pass


# ==============================================
# === MEGA API ERROR CODES ===
# ==============================================

# Complete mapping of Mega API error codes to human-readable descriptions
MEGA_ERROR_CODES = {
    # Success codes
    0: "Success",
    
    # General errors
    -1: "An internal error has occurred. Please submit a bug report, detailing the exact circumstances in which this error occurred.",
    -2: "You have passed invalid arguments to this command.",
    -3: "A temporary congestion or server malfunction prevented your request from being processed. No data was altered. Retry your request.",
    -4: "You have exceeded your command weight per time quota. Please wait a few seconds, then try again.",
    -5: "An unknown error has occurred.",
    -6: "Access denied (insufficient permissions).",
    -7: "Bad session ID.",
    -8: "You have been blocked.",
    -9: "Folder link unavailable.",
    -10: "Already exists.",
    -11: "Access denied.",
    -12: "Trying to create an object that already exists.",
    -13: "Trying to access an incomplete resource.",
    -14: "A decryption operation failed.",
    -15: "Invalid or expired user session, please log in again.",
    -16: "User blocked.",
    -17: "Request over quota.",
    -18: "Resource temporarily not available, please try again later.",
    -19: "Too many connections from this IP address.",
    -20: "Write failed.",
    -21: "Read failed.",
    -22: "Invalid application key; request not processed.",
    
    # File/folder specific errors
    -23: "SSL verification failed.",
    -24: "Not enough quota.",
    -25: "Terms of Service not accepted.",
    -26: "Upload produces recursivity.",
    
    # Authentication errors  
    -101: "Invalid email.",
    -102: "Already registered.",
    -103: "Not registered.",
    -104: "Email not confirmed.",
    -105: "Invalid credentials.",
    -106: "Email already taken.",
    -107: "Too many login attempts.",
    -108: "SMS verification required.",
    -109: "SMS verification failed.",
    -110: "Password incorrect.",
    -111: "Too many requests. Please wait.",
    -112: "Invalid verification code.",
    -113: "Invalid phone number.",
    -114: "SMS not allowed.",
    -115: "Invalid state.",
    -116: "SMS already sent.",
    -117: "SMS send failed.",
    -118: "Invalid country calling code.",
    -119: "SMS phone number already verified.",
    -120: "SMS phone number not verified.",
    
    # Payment/subscription errors
    -201: "Purchase failed.",
    -202: "Balance insufficient.",
    -203: "Payment failed.",
    -204: "Voucher invalid.",
    -205: "Voucher used.",
    -206: "Voucher expired.",
    
    # File transfer errors
    -300: "Transfer over quota.",
    -301: "Transfer failed.",
    -302: "Transfer temporarily unavailable.",
    -303: "Too many concurrent transfers.",
    -304: "Upload target URL expired.",
    -305: "Upload too big for this account.",
    -306: "Upload cancelled.",
    
    # Folder/node errors
    -400: "Folder link unavailable.",
    -401: "File removed.",
    -402: "File not found.",
    -403: "Circular linkage attempted.",
    -404: "Access denied.",
    -405: "Upload target does not exist.",
    
    # Cryptographic errors
    -500: "Decryption failed.",
    -501: "Invalid key.",
    -502: "Invalid MAC.",
    -503: "Key not found.",
    -504: "Invalid signature.",
    
    # Multi-factor authentication errors
    -26: "Multi-factor authentication required.",
    -27: "Access denied for sub-users (only for business accounts).",
    -28: "Business account expired.",
    -29: "Over Disk Quota Paywall.",
    -30: "Subuser key missing - business account encryption required.",
    
    # Payment system errors
    -101: "Invalid credit card.",
    -102: "Billing failed.",
    -103: "Fraud detected.",
    -104: "Too many payment attempts.",
    -105: "Insufficient balance.",
    -106: "Payment failed.",
    
    # Local/Client errors
    -1000: "Insufficient local storage space.",
    -1001: "Request timeout.",
    -1002: "Request abandoned due to logout.",
    -1003: "Local network error (DNS resolution failure).",
    
    # FUSE filesystem errors  
    -2000: "Bad file descriptor.",
    -2001: "Is a directory.",
    -2002: "Filename too long.",
    -2003: "Not a directory.",
    -2004: "Directory not empty.",
    -2005: "File not found.",
    -2006: "Permission denied.",
    -2007: "Read-only filesystem.",
}


# ==============================================
# === VALIDATION UTILITIES ===
# ==============================================

def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
        
    # Basic email pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def validate_password(password: str) -> bool:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not password or not isinstance(password, str):
        return False
        
    # Mega requires at least 8 characters
    if len(password) < 8:
        return False
        
    # Should contain at least one letter and one number
    has_letter = bool(re.search(r'[a-zA-Z]', password))
    has_number = bool(re.search(r'[0-9]', password))
    
    return has_letter and has_number


def get_error_message(error_code: int) -> str:
    """
    Get human-readable error message for Mega API error code.
    
    Args:
        error_code: The error code returned by Mega API
        
    Returns:
        Human-readable error message
    """
    return MEGA_ERROR_CODES.get(error_code, f"Unknown error code: {error_code}")


def is_retryable_error(error_code: int) -> bool:
    """
    Determine if an error code indicates a retryable condition.
    
    Args:
        error_code: The error code returned by Mega API
        
    Returns:
        True if the error is retryable, False otherwise
    """
    retryable_codes = {
        -3,   # Temporary congestion
        -4,   # Rate limited
        -18,  # Resource temporarily unavailable
        -19,  # Too many connections
        -111, # Too many requests
        -302, # Transfer temporarily unavailable
        -1001, # Request timeout
        -1003, # Local network error
    }
    return error_code in retryable_codes


def is_authentication_error(error_code: int) -> bool:
    """
    Determine if an error code indicates an authentication problem.
    
    Args:
        error_code: The error code returned by Mega API
        
    Returns:
        True if the error is authentication-related, False otherwise
    """
    auth_error_codes = {
        -7,   # Bad session ID
        -15,  # Invalid or expired user session
        -16,  # User blocked
        -26,  # Multi-factor authentication required
        -105, # Invalid credentials
        -110, # Password incorrect
    }
    return error_code in auth_error_codes


def raise_mega_error(error_code: int) -> None:
    """
    Raise appropriate exception for Mega API error code.
    
    Args:
        error_code: The error code returned by Mega API
        
    Raises:
        RequestError: With appropriate error message
    """
    error_message = get_error_message(error_code)
    raise RequestError(f"Mega API Error {error_code}: {error_message}")


# ==============================================
# === UTILITY FUNCTIONS ===
# ==============================================

def format_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def detect_file_type(file_path: str) -> Optional[str]:
    """
    Detect MIME type from file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type string or None if not detectable
    """
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type
    except Exception:
        return None


def get_file_extension(file_path: str) -> str:
    """
    Get file extension from path.
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension (including dot) or empty string
    """
    return os.path.splitext(file_path)[1].lower()


def is_image_file(file_path: str) -> bool:
    """
    Check if file is an image based on MIME type.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is an image
    """
    mime_type = detect_file_type(file_path)
    return mime_type is not None and mime_type.startswith('image/')


def is_video_file(file_path: str) -> bool:
    """
    Check if file is a video based on MIME type.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is a video
    """
    mime_type = detect_file_type(file_path)
    return mime_type is not None and mime_type.startswith('video/')


def is_audio_file(file_path: str) -> bool:
    """
    Check if file is audio based on MIME type.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is audio
    """
    mime_type = detect_file_type(file_path)
    return mime_type is not None and mime_type.startswith('audio/')


def get_version_info():
    """Get detailed version information."""
    return {
        'version': __version__,
        'author': __author__,
        'status': __status__,
        'features': {
            'authentication': True,
            'filesystem': True,
            'events': True,
            'utilities': True,
        }
    }


# ==============================================
# === CRYPTOGRAPHIC UTILITIES ===
# ==============================================

def aes_cbc_encrypt(data: bytes, key: bytes, use_zero_iv: bool = False) -> bytes:
    """
    Encrypt data using AES in CBC mode.
    For Mega compatibility, use zero IV and manual padding.
    
    Args:
        data: The data to encrypt
        key: The AES key (16 bytes for AES-128)
        use_zero_iv: If True, use zero IV (for Mega compatibility)
        
    Returns:
        Encrypted data with IV prepended (unless using zero IV)
        
    Raises:
        ValidationError: If key length is invalid
    """
    if len(key) != 16:
        raise ValidationError("AES key must be 16 bytes")
    
    if use_zero_iv:
        # Use zero IV for Mega compatibility - add padding
        iv = b'\x00' * 16
        # Pad data to AES block size (16 bytes)
        padding_length = 16 - (len(data) % 16)
        if padding_length == 16:
            padding_length = 0
        if padding_length > 0:
            data = data + bytes([padding_length] * padding_length)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.encrypt(data)
    else:
        # Generate random IV
        import secrets
        iv = secrets.token_bytes(16)
        
        # Pad data to AES block size (16 bytes)
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        # Encrypt
        cipher = AES.new(key, AES.MODE_CBC, iv)
        encrypted = cipher.encrypt(padded_data)
        
        return iv + encrypted


def aes_cbc_decrypt(data: bytes, key: bytes, use_zero_iv: bool = False) -> bytes:
    """
    Decrypt data using AES in CBC mode.
    For Mega compatibility, use zero IV and no automatic padding removal.
    
    Args:
        data: The encrypted data (with IV prepended unless using zero IV)
        key: The AES key (16 bytes for AES-128)
        use_zero_iv: If True, use zero IV (for Mega compatibility)
        
    Returns:
        Decrypted data with padding removed (unless using zero IV)
        
    Raises:
        ValidationError: If key length is invalid or decryption fails
    """
    if len(key) != 16:
        raise ValidationError("AES key must be 16 bytes")
    
    if use_zero_iv:
        # Use zero IV for Mega compatibility - handle padding removal
        iv = b'\x00' * 16
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(data)
        # Remove padding if present
        if len(decrypted) > 0:
            padding_length = decrypted[-1]
            if padding_length <= 16 and padding_length > 0:
                # Verify padding
                valid_padding = all(decrypted[-i] == padding_length for i in range(1, padding_length + 1))
                if valid_padding:
                    decrypted = decrypted[:-padding_length]
        return decrypted
    else:
        if len(data) < 16:
            raise ValidationError("Encrypted data too short")
        
        # Extract IV and encrypted data
        iv = data[:16]
        encrypted = data[16:]
    
        # Decrypt
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(encrypted)
        
        # Remove padding
        padding_length = decrypted[-1]
        if padding_length > 16 or padding_length == 0:
            raise ValidationError("Invalid padding")
        
        return decrypted[:-padding_length]


def string_to_a32(s: str) -> List[int]:
    """
    Convert string to array of 32-bit integers (Mega's format).
    
    Args:
        s: Input string
        
    Returns:
        List of 32-bit integers
    """
    # Pad string to multiple of 4 bytes
    while len(s) % 4:
        s += '\0'
    
    # Convert to array of 32-bit integers
    return [struct.unpack('>I', makebyte(s[i:i+4]))[0] for i in range(0, len(s), 4)]


def a32_to_string(a: List[int]) -> str:
    """
    Convert array of 32-bit integers to string.
    
    Args:
        a: List of 32-bit integers
        
    Returns:
        Converted string
    """
    return ''.join([makestring(struct.pack('>I', i)) for i in a])


def base64_url_encode(data: bytes) -> str:
    """
    Encode bytes using Mega's base64 URL-safe encoding.
    
    Args:
        data: Bytes to encode
        
    Returns:
        Base64 URL-safe encoded string
    """
    encoded = base64.b64encode(data).decode('ascii')
    # Replace standard base64 chars with URL-safe equivalents
    encoded = encoded.replace('+', '-').replace('/', '_')
    # Remove padding
    return encoded.rstrip('=')


def base64_url_decode(s: str) -> bytes:
    """
    Decode Mega's base64 URL-safe encoding.
    
    Args:
        s: Base64 URL-safe encoded string
        
    Returns:
        Decoded bytes
        
    Raises:
        ValidationError: If decoding fails
    """
    try:
        # Replace URL-safe chars with standard base64 chars
        s = s.replace('-', '+').replace('_', '/')
        
        # Add padding if needed
        padding = 4 - (len(s) % 4)
        if padding != 4:
            s += '=' * padding
        
        return base64.b64decode(s)
    except Exception as e:
        raise ValidationError(f"Base64 decode failed: {e}")


def derive_key(password: str, salt: bytes = b'') -> bytes:
    """
    Derive encryption key from password using Mega's key derivation.
    
    This uses a simplified PBKDF2-like approach as used by Mega.
    
    Args:
        password: The user's password
        salt: Optional salt bytes
        
    Returns:
        Derived key (16 bytes)
    """
    # Convert password to bytes
    password_bytes = password.encode('utf-8')
    
    # Mega uses a specific key derivation approach
    key_material = password_bytes + salt
    
    # Hash multiple times for key strengthening
    for _ in range(65536):
        key_material = hashlib.sha256(key_material).digest()
    
    # Return first 16 bytes as AES key
    return key_material[:16]


def generate_random_key() -> bytes:
    """
    Generate a random AES key.
    
    Returns:
        16 bytes of random key material
    """
    import secrets
    return secrets.token_bytes(16)


def hash_password(password: str, email: str) -> str:
    """
    Hash password for Mega authentication.
    
    Args:
        password: User's password
        email: User's email address
        
    Returns:
        Hashed password for authentication
    """
    # Normalize email to lowercase
    email = email.lower()
    
    # Derive key from password
    key = derive_key(password)
    
    # Hash email with derived key
    email_hash = hashlib.sha256(makebyte(email)).digest()
    
    # Encrypt email hash with derived key
    encrypted = aes_cbc_encrypt(email_hash, key)
    
    # Return as base64
    return base64_url_encode(encrypted)