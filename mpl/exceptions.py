"""
Exception Definitions Module
=============================

This module defines all exception classes and error code mappings used by the Mega.nz client.

This follows the exact methodology from the reference implementation:
1. Custom exception classes for validation and request errors
2. Complete Mega API error code mappings
3. Error handling utilities

Author: Modernized from reference implementation
Date: July 2025
"""

from .dependencies import *

# ==============================================
# === CUSTOM EXCEPTION CLASSES ===
# ==============================================

class ValidationError(Exception):
    """
    Exception raised when data validation fails.
    
    This is used throughout the client for validating:
    - Email addresses
    - Passwords
    - File paths
    - API responses
    - Cryptographic data
    """
    pass


class RequestError(Exception):
    """
    Exception raised when API requests fail.
    
    This encompasses:
    - Network connection errors
    - HTTP errors
    - Mega API errors
    - Timeout errors
    - Authentication failures
    """
    pass


class MPLError(Exception):
    """Base exception class for all MegaPythonLibrary errors."""
    pass


class AuthenticationError(MPLError):
    """Exception raised for authentication-related errors."""
    pass


class CryptoError(MPLError):
    """Exception raised for cryptographic operation errors."""
    pass


class NetworkError(MPLError):
    """Exception raised for network-related errors."""
    pass


class BusinessError(MPLError):
    """Exception raised for business account specific errors."""
    pass


class PaymentError(MPLError):
    """Exception raised for payment and billing errors."""
    pass


class FUSEError(MPLError):
    """Exception raised for FUSE filesystem errors."""
    pass


class LocalError(MPLError):
    """Exception raised for local client-side errors."""
    pass


# ==============================================
# === MEGA API ERROR CODES ===
# ==============================================

# Complete mapping of Mega API error codes to human-readable descriptions
# These codes are returned by the Mega API and must be handled appropriately
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
    import re
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
    import re
    has_letter = bool(re.search(r'[a-zA-Z]', password))
    has_number = bool(re.search(r'[0-9]', password))
    
    return has_letter and has_number


# ==============================================
# === ERROR HANDLING UTILITIES ===
# ==============================================

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


def get_error_category(error_code: int) -> str:
    """
    Get the category of error based on error code.
    
    Args:
        error_code: The error code returned by Mega API
        
    Returns:
        Error category string
    """
    if error_code >= 0:
        return "success"
    elif is_authentication_error(error_code):
        return "authentication"
    elif error_code in [-26, -27, -28, -29, -30]:
        return "business"
    elif error_code in [-101, -102, -103, -104, -105, -106]:
        return "payment"
    elif error_code in [-500, -501, -502, -503, -504]:
        return "crypto"
    elif error_code in [-1000, -1001, -1002, -1003]:
        return "local"
    elif error_code in [-2000, -2001, -2002, -2003, -2004, -2005, -2006, -2007]:
        return "fuse"
    elif error_code in [-300, -301, -302, -303, -304, -305, -306]:
        return "transfer"
    elif error_code in [-400, -401, -402, -403, -404, -405]:
        return "filesystem"
    else:
        return "general"


def is_permanent_error(error_code: int) -> bool:
    """
    Determine if an error code indicates a permanent failure.
    
    Args:
        error_code: The error code returned by Mega API
        
    Returns:
        True if the error is permanent (should not be retried), False otherwise
    """
    permanent_codes = {
        -2,   # Bad arguments
        -5,   # Request failed permanently
        -6,   # Access denied (insufficient permissions)
        -8,   # User blocked
        -9,   # Folder link unavailable
        -10,  # Already exists
        -11,  # Access denied
        -12,  # Trying to create an object that already exists
        -14,  # A decryption operation failed
        -15,  # Invalid or expired user session
        -16,  # User blocked
        -22,  # Invalid application key
        -25,  # Terms of Service not accepted
        -101, # Invalid email
        -102, # Already registered
        -103, # Not registered
        -105, # Invalid credentials
        -110, # Password incorrect
        -401, # File removed
        -402, # File not found
        -500, # Decryption failed
        -501, # Invalid key
        -502, # Invalid MAC
        -503, # Key not found
        -504, # Invalid signature
    }
    return error_code in permanent_codes


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Exception classes
    'MPLError',
    'ValidationError',
    'RequestError',
    'AuthenticationError',
    'CryptoError',
    'NetworkError',
    'BusinessError',
    'PaymentError',
    'FUSEError',
    'LocalError',
    
    # Validation utilities
    'validate_email',
    'validate_password',
    
    # Error code mappings
    'MEGA_ERROR_CODES',
    
    # Error handling utilities
    'get_error_message',
    'get_error_category',
    'is_retryable_error', 
    'is_authentication_error',
    'is_permanent_error',
    'raise_mega_error',
]

# Configure logging
logger = logging.getLogger(__name__)

