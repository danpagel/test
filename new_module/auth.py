"""
Authentication and Security module for MegaPythonLibrary.

This module contains:
- User session management
- Authentication functions (login, logout, register)
- Password and user management
- Cryptographic operations specific to authentication
- Session persistence
- Advanced crypto functions for Mega protocol
"""

import json
import math
import hashlib
import secrets
import binascii
import struct
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Util import Counter

from .utils import (
    ValidationError, RequestError, AuthenticationError,
    validate_email, validate_password, makebyte, makestring,
    base64_url_encode, base64_url_decode, string_to_a32, a32_to_string,
    is_authentication_error
)
from .network import single_api_request, set_session_id, clear_session_id
from .monitor import get_logger, trigger_event

# ==============================================
# === USER SESSION MANAGEMENT ===
# ==============================================

class UserSession:
    """
    Represents an authenticated user session with Mega.
    """
    
    def __init__(self):
        self.email: Optional[str] = None
        self.session_id: Optional[str] = None
        self.master_key: Optional[bytes] = None
        self.rsa_private_key: Optional[bytes] = None
        self.user_handle: Optional[str] = None
        self.is_authenticated = False
        self.session_data: Dict[str, Any] = {}
        self.logger = get_logger("session")
    
    def clear(self) -> None:
        """Clear all session data."""
        self.email = None
        self.session_id = None
        self.master_key = None
        self.rsa_private_key = None
        self.user_handle = None
        self.is_authenticated = False
        self.session_data.clear()
        self.logger.info("Session cleared")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for persistence."""
        return {
            'email': self.email,
            'session_id': self.session_id,
            'master_key': base64_url_encode(self.master_key) if self.master_key else None,
            'rsa_private_key': base64_url_encode(self.rsa_private_key) if self.rsa_private_key else None,
            'user_handle': self.user_handle,
            'is_authenticated': self.is_authenticated,
            'session_data': self.session_data,
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore session from dictionary."""
        self.email = data.get('email')
        self.session_id = data.get('session_id')
        self.master_key = base64_url_decode(data['master_key']) if data.get('master_key') else None
        self.rsa_private_key = base64_url_decode(data['rsa_private_key']) if data.get('rsa_private_key') else None
        self.user_handle = data.get('user_handle')
        self.is_authenticated = data.get('is_authenticated', False)
        self.session_data = data.get('session_data', {})
        self.logger.info("Session restored from data")


# Global user session
current_session = UserSession()


# ==============================================
# === ADVANCED CRYPTOGRAPHIC FUNCTIONS ===
# ==============================================

def aes_cbc_encrypt_mega(data: bytes, key: bytes) -> bytes:
    """
    Encrypts data using AES in CBC mode with zero IV (Mega style).
    No automatic padding - data must be pre-padded.
    
    Args:
        data: Data to encrypt (must be multiple of 16 bytes)
        key: AES key (16 bytes)
        
    Returns:
        Encrypted data
    """
    if len(key) != 16:
        raise ValidationError("AES key must be 16 bytes")
    
    cipher = AES.new(key, AES.MODE_CBC, makebyte('\0' * 16))
    return cipher.encrypt(data)


def aes_cbc_decrypt_mega(data: bytes, key: bytes) -> bytes:
    """
    Decrypts data using AES in CBC mode with zero IV (Mega style).
    No automatic padding removal.
    
    Args:
        data: Data to decrypt (must be multiple of 16 bytes)
        key: AES key (16 bytes)
        
    Returns:
        Decrypted data
    """
    if len(key) != 16:
        raise ValidationError("AES key must be 16 bytes")
    
    cipher = AES.new(key, AES.MODE_CBC, makebyte('\0' * 16))
    return cipher.decrypt(data)


def aes_cbc_encrypt_a32(data: List[int], key: List[int]) -> List[int]:
    """
    Encrypts an array of 32-bit integers using AES CBC with zero IV.
    This is the core function used by Mega's authentication system.
    
    Args:
        data: Array of 32-bit integers to encrypt
        key: Array of 32-bit integers as encryption key
        
    Returns:
        Encrypted data as array of 32-bit integers
    """
    data_bytes = makebyte(a32_to_string(data))
    key_bytes = makebyte(a32_to_string(key))
    
    encrypted = aes_cbc_encrypt_mega(data_bytes, key_bytes)
    
    return string_to_a32(makestring(encrypted))


def aes_cbc_decrypt_a32(data: List[int], key: List[int]) -> List[int]:
    """
    Decrypts an array of 32-bit integers using AES CBC with zero IV.
    
    Args:
        data: Array of 32-bit integers to decrypt
        key: Array of 32-bit integers as decryption key
        
    Returns:
        Decrypted data as array of 32-bit integers
    """
    data_bytes = makebyte(a32_to_string(data))
    key_bytes = makebyte(a32_to_string(key))
    
    decrypted = aes_cbc_decrypt_mega(data_bytes, key_bytes)
    
    return string_to_a32(makestring(decrypted))


def prepare_key(arr: List[int]) -> List[int]:
    """
    Derives a key from the input array using repeated AES CBC encryption.
    This is Mega's password-based key derivation for v1 accounts.
    
    Args:
        arr: Input array of 32-bit integers
        
    Returns:
        Derived key as array of 32-bit integers
    """
    pkey = [0x93C467E3, 0x7DB0C7A4, 0xD1BE3F81, 0x0152CB56]
    
    for r in range(0x10000):  # 65536 iterations
        for j in range(0, len(arr), 4):
            key = [0, 0, 0, 0]
            for i in range(4):
                if i + j < len(arr):
                    key[i] = arr[i + j]
            pkey = aes_cbc_encrypt_a32(pkey, key)
    
    return pkey


def stringhash(s: str, aeskey: List[int]) -> str:
    """
    Generates a hash of the input string using the provided AES key.
    This is used for user authentication in Mega.
    
    Args:
        s: Input string to hash
        aeskey: AES key as array of 32-bit integers
        
    Returns:
        Base64-encoded hash string
    """
    s32 = string_to_a32(s)
    h32 = [0, 0, 0, 0]
    
    # XOR string into hash
    for i in range(len(s32)):
        h32[i % 4] ^= s32[i]
    
    # Encrypt hash 16384 times
    for r in range(0x4000):
        h32 = aes_cbc_encrypt_a32(h32, aeskey)
    
    # Return specific elements as base64
    return a32_to_base64([h32[0], h32[2]])


def encrypt_key(a: List[int], key: List[int]) -> List[int]:
    """
    Encrypts a key (array of 32-bit ints) with another key.
    
    Args:
        a: Key array to encrypt
        key: Encryption key array
        
    Returns:
        Encrypted key array
    """
    result = []
    for i in range(0, len(a), 4):
        chunk = a[i:i + 4]
        # Pad chunk to 4 elements if needed
        while len(chunk) < 4:
            chunk.append(0)
        encrypted_chunk = aes_cbc_encrypt_a32(chunk, key)
        result.extend(encrypted_chunk)
    
    return result


def decrypt_key(a: List[int], key: List[int]) -> List[int]:
    """
    Decrypts a key (array of 32-bit ints) with another key.
    
    Args:
        a: Key array to decrypt
        key: Decryption key array
        
    Returns:
        Decrypted key array
    """
    result = []
    for i in range(0, len(a), 4):
        chunk = a[i:i + 4]
        # Pad chunk to 4 elements if needed
        while len(chunk) < 4:
            chunk.append(0)
        decrypted_chunk = aes_cbc_decrypt_a32(chunk, key)
        result.extend(decrypted_chunk)
    
    return result


def base64_to_a32(s: str) -> List[int]:
    """
    Convert base64 string to array of 32-bit integers.
    
    Args:
        s: Base64 encoded string
        
    Returns:
        List of 32-bit integers
    """
    decoded = base64_url_decode(s)
    return string_to_a32(makestring(decoded))


def a32_to_base64(a: List[int]) -> str:
    """
    Convert array of 32-bit integers to base64 string.
    
    Args:
        a: List of 32-bit integers
        
    Returns:
        Base64 encoded string
    """
    s = a32_to_string(a)
    return base64_url_encode(makebyte(s))


def mpi_to_int(s: bytes) -> int:
    """
    Convert MPI (Multi-Precision Integer) format to integer.
    
    Args:
        s: MPI format bytes
        
    Returns:
        Integer value
    """
    if len(s) < 2:
        return 0
    
    # First 2 bytes contain bit length
    bit_length = (s[0] << 8) + s[1]
    byte_length = math.ceil(bit_length / 8)
    
    if len(s) < 2 + byte_length:
        return 0
    
    # Convert bytes to integer
    return int.from_bytes(s[2:2 + byte_length], 'big')


def modular_inverse(a: int, m: int) -> int:
    """
    Calculate modular inverse using extended Euclidean algorithm.
    
    Args:
        a: Number to find inverse for
        m: Modulus
        
    Returns:
        Modular inverse of a mod m
    """
    if m == 0:
        return 1
    
    # Extended Euclidean Algorithm
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(a % m, m)
    if gcd != 1:
        raise ValueError("Modular inverse does not exist")
    return (x % m + m) % m


# ==============================================
# === SESSION PERSISTENCE ===
# ==============================================

def get_session_file_path() -> Path:
    """Get path to session file."""
    return Path.home() / '.mpl_session.json'


def save_user_session() -> None:
    """Save current user session to file."""
    if not current_session.is_authenticated:
        return
    
    logger = get_logger("session")
    try:
        session_file = get_session_file_path()
        session_data = current_session.to_dict()
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
        
        # Set restrictive permissions
        os.chmod(session_file, 0o600)
        logger.info("Session saved to file")
        
    except Exception as e:
        logger.warning(f"Failed to save session: {e}")


def load_user_session() -> bool:
    """
    Load saved user session from file.
    
    Returns:
        True if session was loaded successfully, False otherwise
    """
    logger = get_logger("session")
    try:
        session_file = get_session_file_path()
        if not session_file.exists():
            return False
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        current_session.from_dict(session_data)
        
        # Set session ID in network module
        if current_session.session_id:
            set_session_id(current_session.session_id)
        
        logger.info("Session loaded from file")
        return current_session.is_authenticated
        
    except Exception as e:
        logger.warning(f"Failed to load session: {e}")
        return False


def clear_saved_session() -> None:
    """Clear saved session file."""
    logger = get_logger("session")
    try:
        session_file = get_session_file_path()
        if session_file.exists():
            session_file.unlink()
            logger.info("Saved session file cleared")
    except Exception as e:
        logger.warning(f"Failed to clear saved session: {e}")


# ==============================================
# === AUTHENTICATION FUNCTIONS ===
# ==============================================

def login(email: str, password: str, save_session: bool = True) -> UserSession:
    """
    Authenticate user with email and password using Mega's exact login sequence.
    
    Args:
        email: User's email address
        password: User's password
        save_session: Whether to save session for persistence
        
    Returns:
        Authenticated user session
        
    Raises:
        ValidationError: If email/password format is invalid
        RequestError: If authentication fails
    """
    logger = get_logger("auth")
    
    # Validate inputs
    if not validate_email(email):
        raise ValidationError("Invalid email address format")
    
    if not validate_password(password):
        raise ValidationError("Password must be at least 8 characters")
    
    # Normalize email
    email = email.lower().strip()
    
    logger.info(f'Logging in user: {email}')
    trigger_event('login_started', {'email': email})
    
    try:
        # Step 1: Check if user has salt (v2 account) or is v1 account
        get_user_salt_resp = single_api_request({'a': 'us0', 'user': email})
        user_salt = None
        
        try:
            user_salt = base64_to_a32(get_user_salt_resp['s'])
            logger.info('Detected v2 user account (with salt)')
        except (KeyError, TypeError):
            logger.info('Detected v1 user account (no salt)')
            user_salt = None
        
        # Step 2: Derive password AES key and user hash
        if user_salt is None:
            # v1 user account: derive key directly from password
            password_a32 = string_to_a32(password)
            password_aes = prepare_key(password_a32)
            user_hash = stringhash(email, password_aes)
        else:
            # v2 user account: use PBKDF2 for key derivation
            pbkdf2_key = hashlib.pbkdf2_hmac(
                hash_name='sha512',
                password=password.encode(),
                salt=makebyte(a32_to_string(user_salt)),
                iterations=100000,
                dklen=32
            )
            password_aes = string_to_a32(makestring(pbkdf2_key[:16]))
            user_hash = base64_url_encode(pbkdf2_key[-16:])
        
        # Step 3: Send login request
        login_command = {
            'a': 'us',  # User session
            'user': email,
            'uh': user_hash,
        }
        
        resp = single_api_request(login_command)
        
        if isinstance(resp, int):
            raise RequestError(resp)
        
        if not isinstance(resp, dict):
            raise RequestError("Invalid login response format")
        
        # Step 4: Process login response and decrypt master key
        encrypted_master_key = base64_to_a32(resp['k'])
        master_key = decrypt_key(encrypted_master_key, password_aes)
        
        # Step 5: Handle session ID (tsid or csid)
        session_id = None
        
        if 'tsid' in resp:
            # Temporary session ID
            tsid = base64_url_decode(resp['tsid'])
            key_encrypted = makebyte(a32_to_string(
                encrypt_key(string_to_a32(makestring(tsid[:16])), master_key)
            ))
            if key_encrypted == tsid[-16:]:
                session_id = resp['tsid']
        elif 'csid' in resp:
            # CSE session ID - requires RSA decryption
            encrypted_rsa_private_key = base64_to_a32(resp['privk'])
            rsa_private_key = decrypt_key(encrypted_rsa_private_key, master_key)
            
            private_key = makebyte(a32_to_string(rsa_private_key))
            
            # Parse MPI integers from private key
            rsa_private_key_components = [0, 0, 0, 0]
            key_pos = 0
            for i in range(4):
                # MPI integer has 2-byte header describing bit length
                if key_pos + 2 > len(private_key):
                    break
                bitlength = (private_key[key_pos] << 8) + private_key[key_pos + 1]
                bytelength = math.ceil(bitlength / 8)
                # Add 2 bytes for MPI header
                total_length = bytelength + 2
                
                if key_pos + total_length > len(private_key):
                    break
                    
                rsa_private_key_components[i] = mpi_to_int(private_key[key_pos:key_pos + total_length])
                key_pos += total_length
            
            first_factor_p = rsa_private_key_components[0]
            second_factor_q = rsa_private_key_components[1]
            private_exponent_d = rsa_private_key_components[2]
            
            rsa_modulus_n = first_factor_p * second_factor_q
            phi = (first_factor_p - 1) * (second_factor_q - 1)
            public_exponent_e = modular_inverse(private_exponent_d, phi)
            
            rsa_components = (
                rsa_modulus_n,
                public_exponent_e,
                private_exponent_d,
                first_factor_p,
                second_factor_q,
            )
            rsa_decrypter = RSA.construct(rsa_components)
            
            encrypted_sid = mpi_to_int(base64_url_decode(resp['csid']))
            sid = '%x' % rsa_decrypter._decrypt(encrypted_sid)
            sid = binascii.unhexlify('0' + sid if len(sid) % 2 else sid)
            session_id = base64_url_encode(sid[:43])
        
        if not session_id:
            raise RequestError("No valid session ID received")
        
        # Extract user data
        user_handle = resp.get('u')
        
        # Set up session
        current_session.clear()
        current_session.email = email
        current_session.session_id = session_id
        current_session.master_key = makebyte(a32_to_string(master_key))
        current_session.user_handle = user_handle
        current_session.is_authenticated = True
        
        # Set session ID in network module
        set_session_id(session_id)
        
        # Save session if requested
        if save_session:
            save_user_session()
        
        logger.info(f"Successfully logged in as {email}")
        trigger_event('login_completed', {'email': email, 'success': True})
        return current_session
        
    except Exception as e:
        logger.error(f"Login failed: {e}")
        trigger_event('login_failed', {'email': email, 'error': str(e)})
        if is_authentication_error(getattr(e, 'args', [None])[0] if hasattr(e, 'args') else -1):
            raise AuthenticationError("Invalid email or password")
        raise


def logout() -> None:
    """
    Log out current user and clear session.
    """
    logger = get_logger("auth")
    email = current_session.email
    
    trigger_event('logout_started', {'email': email})
    
    if current_session.is_authenticated and current_session.session_id:
        try:
            # Send logout command
            logout_command = {'a': 'sml'}  # Session logout
            single_api_request(logout_command, current_session.session_id)
        except Exception as e:
            logger.warning(f"Logout request failed: {e}")
    
    # Clear session data
    current_session.clear()
    clear_session_id()
    
    # Clear saved session
    clear_saved_session()
    
    logger.info("Successfully logged out")
    trigger_event('logout_completed', {'email': email})


def register(email: str, password: str, first_name: str = "", last_name: str = "") -> bool:
    """
    Register new user account.
    
    Args:
        email: User's email address
        password: User's password
        first_name: User's first name (optional)
        last_name: User's last name (optional)
        
    Returns:
        True if registration initiated successfully
    """
    logger = get_logger("auth")
    
    if not validate_email(email):
        raise ValidationError("Invalid email address format")
    
    if not validate_password(password):
        raise ValidationError("Password must be at least 8 characters")
    
    trigger_event('registration_started', {'email': email})
    
    try:
        # Derive user key
        password_a32 = string_to_a32(password)
        password_aes = prepare_key(password_a32)
        
        # Create user
        command = {
            'a': 'up',
            'k': a32_to_base64(password_aes),
            'ts': base64_url_encode(secrets.token_bytes(16)),
        }
        
        result = single_api_request(command)
        logger.info(f"Registration initiated for {email}")
        trigger_event('registration_completed', {'email': email, 'success': True})
        return True
        
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        trigger_event('registration_failed', {'email': email, 'error': str(e)})
        return False


def verify_email(email: str, verification_code: str) -> bool:
    """
    Verify email address with verification code.
    
    Args:
        email: Email address to verify
        verification_code: Verification code
        
    Returns:
        True if verification successful
    """
    logger = get_logger("auth")
    command = {
        'a': 'uv',
        'c': verification_code
    }
    
    try:
        result = single_api_request(command)
        logger.info(f"Email verification successful for {email}")
        return True
    except Exception as e:
        logger.error(f"Email verification failed: {e}")
        return False


def change_password(old_password: str, new_password: str) -> bool:
    """
    Change user password.
    
    Args:
        old_password: Current password
        new_password: New password
        
    Returns:
        True if password changed successfully
    """
    logger = get_logger("auth")
    
    if not current_session.is_authenticated:
        raise RequestError("Not logged in")
    
    if not validate_password(new_password):
        raise ValidationError("New password must be at least 8 characters")
    
    try:
        # Get user salt
        get_salt_resp = single_api_request({'a': 'us0', 'user': current_session.email})
        user_salt = base64_to_a32(get_salt_resp['s']) if 's' in get_salt_resp else None
        
        # Derive new key
        if user_salt is None:
            new_password_a32 = string_to_a32(new_password)
            new_password_aes = prepare_key(new_password_a32)
        else:
            pbkdf2_key = hashlib.pbkdf2_hmac(
                hash_name='sha512',
                password=new_password.encode(),
                salt=makebyte(a32_to_string(user_salt)),
                iterations=100000,
                dklen=32
            )
            new_password_aes = string_to_a32(makestring(pbkdf2_key[:16]))
        
        # Encrypt master key with new password
        encrypted_master_key = encrypt_key(
            string_to_a32(makestring(current_session.master_key)),
            new_password_aes
        )
        
        # Send password change request
        command = {
            'a': 'up',
            'k': a32_to_base64(encrypted_master_key),
        }
        
        result = single_api_request(command, current_session.session_id)
        logger.info("Password changed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        return False


# ==============================================
# === UTILITY FUNCTIONS ===
# ==============================================

def is_logged_in() -> bool:
    """Check if user is currently logged in."""
    return current_session.is_authenticated


def get_current_user() -> Optional[str]:
    """Get current user's email address."""
    return current_session.email if current_session.is_authenticated else None


def get_user_info() -> Dict[str, Any]:
    """
    Get current user information.
    
    Returns:
        Dictionary with user information
    """
    if not current_session.is_authenticated:
        raise RequestError("Not logged in")
    
    command = {'a': 'ug'}
    result = single_api_request(command, current_session.session_id)
    
    return {
        'email': current_session.email,
        'user_handle': current_session.user_handle,
        'name': result.get('name', ''),
        'firstname': result.get('firstname', ''),
        'lastname': result.get('lastname', ''),
        'birthday': result.get('birthday'),
        'birthmonth': result.get('birthmonth'),
        'birthyear': result.get('birthyear'),
        'country': result.get('country', ''),
    }


def get_user_quota() -> Dict[str, int]:
    """
    Get current user storage quota information.
    
    Returns:
        Dictionary with quota information (bytes)
    """
    if not current_session.is_authenticated:
        raise RequestError("Not logged in")
    
    command = {'a': 'uq', 'xfer': 1, 'strg': 1}
    result = single_api_request(command, current_session.session_id)
    
    return {
        'total_storage': result.get('mstrg', 0),
        'used_storage': result.get('cstrg', 0),
        'available_storage': result.get('mstrg', 0) - result.get('cstrg', 0),
        'total_transfer': result.get('mxfer', 0),
        'used_transfer': result.get('caxfer', 0),
        'available_transfer': result.get('mxfer', 0) - result.get('caxfer', 0),
    }


def require_authentication(func):
    """Decorator to require authentication for function calls."""
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            raise AuthenticationError("Authentication required")
        return func(*args, **kwargs)
    return wrapper


# ==============================================
# === AUTO-LOGIN SUPPORT ===
# ==============================================

def try_auto_login() -> bool:
    """
    Try to automatically log in using saved session.
    
    Returns:
        True if auto-login was successful
    """
    return load_user_session()