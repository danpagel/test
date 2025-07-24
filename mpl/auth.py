"""
Authentication and User Management Module
=========================================

This module handles all user authentication and account management for the Mega.nz client.

This follows the exact methodology from the reference implementation:
1. Login/logout functionality with session management
2. User registration and email verification
3. Master key derivation and secure storage
4. Session persistence and restoration
5. Multi-factor authentication support

Author: Modernized from reference implementation
Date: July 2025
"""

from .dependencies import *
import math
import re
import hashlib
import binascii
try:
    from Crypto.PublicKey import RSA
except ImportError:
    RSA = None
from .exceptions import RequestError, ValidationError, is_authentication_error, validate_email, validate_password
from .crypto import (
    derive_key, hash_password, string_to_a32, a32_to_string, 
    base64_url_encode, base64_url_decode, base64_to_a32, a32_to_base64,
    aes_cbc_encrypt, aes_cbc_decrypt, generate_random_key,
    prepare_key, stringhash, encrypt_key, decrypt_key,
    aes_cbc_encrypt_a32, aes_cbc_decrypt_a32, mpi_to_int, modular_inverse
)
from .network import (
    single_api_request, _api_session, APISession
)

# ==============================================
# === USER SESSION CLASS ===
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
    
    def clear(self) -> None:
        """Clear all session data."""
        self.email = None
        self.session_id = None
        self.master_key = None
        self.rsa_private_key = None
        self.user_handle = None
        self.is_authenticated = False
        self.session_data.clear()
    
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


# Global user session
current_session = UserSession()


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
    # Validate inputs
    if not validate_email(email):
        raise ValidationError("Invalid email address format")
    
    if not validate_password(password):
        raise ValidationError("Password must be at least 8 characters")
    
    # Normalize email
    email = email.lower().strip()
    
    logger.info(f'Logging in user: {email}')
    
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
    
    try:
        # Make login request
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
            
            if RSA is None:
                raise RequestError("RSA library not available for CSE login")
                
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
        
        # Set session ID in API session
        _api_session.set_session_id(session_id)
        
        # Save session if requested
        if save_session:
            save_user_session()
        
        logger.info(f"Successfully logged in as {email}")
        return current_session
        
    except Exception as e:
        if is_authentication_error(getattr(e, 'args', [None])[0] if hasattr(e, 'args') else -1):
            raise RequestError("Invalid email or password")
        raise


def logout() -> None:
    """
    Log out current user and clear session.
    """
    if current_session.is_authenticated and current_session.session_id:
        try:
            # Send logout command
            logout_command = {'a': 'sml'}  # Session logout
            single_api_request(logout_command, current_session.session_id)
        except Exception as e:
            logger.warning(f"Logout request failed: {e}")
    
    # Clear session data
    current_session.clear()
    _api_session.set_session_id(None)
    
    # Clear saved session
    clear_saved_session()
    
    logger.info("Successfully logged out")


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
        
    Raises:
        ValidationError: If email/password format is invalid
        RequestError: If registration fails
    """
    # Validate inputs
    if not validate_email(email):
        raise ValidationError("Invalid email address format")
    
    if not validate_password(password):
        raise ValidationError("Password must be at least 8 characters")
    
    # Normalize email
    email = email.lower().strip()
    
    # Derive master key from password
    master_key = derive_key(password)
    
    # Generate RSA key pair for encryption
    rsa_key = RSA.generate(2048)
    private_key_data = rsa_key.export_key('DER')
    public_key_data = rsa_key.publickey().export_key('DER')
    
    # Encrypt private key with master key
    encrypted_private_key = aes_cbc_encrypt(private_key_data, master_key)
    
    # Hash password for authentication
    password_hash = hash_password(password, email)
    
    # Prepare user attributes
    user_attrs = {
        'n': f"{first_name} {last_name}".strip() or email.split('@')[0]
    }
    user_attrs_json = json.dumps(user_attrs)
    encrypted_attrs = aes_cbc_encrypt(user_attrs_json.encode('utf-8'), master_key)
    
    # Prepare registration command
    register_command = {
        'a': 'up',  # User registration
        'k': base64_url_encode(encrypted_private_key),
        'ts': base64_url_encode(public_key_data),
        'uh': password_hash,
        'c': base64_url_encode(encrypted_attrs),
        'name': email,
    }
    
    try:
        result = single_api_request(register_command)
        
        if isinstance(result, int) and result < 0:
            if result == -2:
                raise RequestError("Email address already registered")
            elif result == -9:
                raise RequestError("Invalid email address")
            else:
                raise RequestError(f"Registration failed with error code: {result}")
        
        logger.info(f"Registration initiated for {email}")
        return True
        
    except Exception as e:
        if "already registered" in str(e).lower():
            raise RequestError("Email address already registered")
        raise


def verify_email(email: str, verification_code: str) -> bool:
    """
    Verify email address with confirmation code.
    
    Args:
        email: User's email address
        verification_code: Verification code from email
        
    Returns:
        True if verification successful
        
    Raises:
        RequestError: If verification fails
    """
    command = {
        'a': 'ud',  # User verification
        'c': verification_code,
    }
    
    try:
        result = single_api_request(command)
        
        if isinstance(result, int) and result < 0:
            if result == -12:
                raise RequestError("Invalid or expired verification code")
            else:
                raise RequestError(f"Verification failed with error code: {result}")
        
        logger.info(f"Email verification successful for {email}")
        return True
        
    except Exception:
        raise RequestError("Email verification failed")


# ==============================================
# === SESSION PERSISTENCE ===
# ==============================================

def get_session_file_path() -> Path:
    """Get path to session file."""
    return Path.home() / '.mega_session.json'


def save_user_session() -> None:
    """Save current session to file."""
    if not current_session.is_authenticated:
        return
    
    try:
        session_file = get_session_file_path()
        session_data = current_session.to_dict()
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Make file readable only by owner
        if hasattr(os, 'chmod'):
            os.chmod(session_file, 0o600)
        
        logger.debug("Session saved successfully")
        
    except Exception as e:
        logger.warning(f"Failed to save session: {e}")


def load_user_session() -> bool:
    """
    Load saved session from file.
    
    Returns:
        True if session loaded successfully, False otherwise
    """
    try:
        session_file = get_session_file_path()
        
        if not session_file.exists():
            return False
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Restore session
        current_session.from_dict(session_data)
        
        # Validate session by making a test request
        if current_session.session_id:
            try:
                # Test session validity
                test_command = {'a': 'uq'}  # User quota
                single_api_request(test_command, current_session.session_id)
                
                # Session is valid
                _api_session.set_session_id(current_session.session_id)
                logger.info(f"Session restored for {current_session.email}")
                return True
                
            except Exception:
                # Session expired or invalid
                current_session.clear()
                clear_saved_session()
                return False
        
        return False
        
    except Exception as e:
        logger.warning(f"Failed to load session: {e}")
        return False


def clear_saved_session() -> None:
    """Clear saved session file."""
    try:
        session_file = get_session_file_path()
        if session_file.exists():
            session_file.unlink()
        logger.debug("Saved session cleared")
    except Exception as e:
        logger.warning(f"Failed to clear saved session: {e}")


# ==============================================
# === USER INFORMATION ===
# ==============================================

def get_user_info() -> Dict[str, Any]:
    """
    Get current user information.
    
    Returns:
        Dictionary with user information
        
    Raises:
        RequestError: If not authenticated or request fails
    """
    if not current_session.is_authenticated:
        raise RequestError("Not authenticated")
    
    command = {'a': 'ug'}  # User info
    result = single_api_request(command)
    
    if not isinstance(result, dict):
        raise RequestError("Invalid user info response")
    
    return {
        'email': current_session.email,
        'handle': current_session.user_handle,
        'name': result.get('name', ''),
        'storage_used': result.get('cstrg', 0),
        'storage_quota': result.get('mstrg', 0),
        'transfer_used': result.get('cstrg', 0),
        'transfer_quota': result.get('mstrg', 0),
        'account_type': result.get('utype', 0),
    }


def get_user_quota() -> Dict[str, int]:
    """
    Get user storage and transfer quota information.
    
    Returns:
        Dictionary with quota information
        
    Raises:
        RequestError: If not authenticated or request fails
    """
    if not current_session.is_authenticated:
        raise RequestError("Not authenticated")
    
    command = {'a': 'uq'}  # User quota
    result = single_api_request(command)
    
    if not isinstance(result, dict):
        raise RequestError("Invalid quota response")
    
    return {
        'storage_used': result.get('cstrg', 0),
        'storage_max': result.get('mstrg', 0),
        'transfer_used': result.get('caxfer', 0),
        'transfer_max': result.get('maxfer', 0),
    }


# ==============================================
# === PASSWORD MANAGEMENT ===
# ==============================================

def change_password(old_password: str, new_password: str) -> bool:
    """
    Change user password.
    
    Args:
        old_password: Current password
        new_password: New password
        
    Returns:
        True if password changed successfully
        
    Raises:
        ValidationError: If password format is invalid
        RequestError: If not authenticated or change fails
    """
    if not current_session.is_authenticated:
        raise RequestError("Not authenticated")
    
    if not validate_password(new_password):
        raise ValidationError("New password must be at least 8 characters")
    
    # Verify old password
    old_hash = hash_password(old_password, current_session.email)
    
    # Generate new master key and hash
    new_master_key = derive_key(new_password)
    new_hash = hash_password(new_password, current_session.email)
    
    # Re-encrypt RSA private key with new master key
    if current_session.rsa_private_key:
        encrypted_key = aes_cbc_encrypt(current_session.rsa_private_key, new_master_key)
        key_data = base64_url_encode(encrypted_key)
    else:
        key_data = None
    
    # Prepare change password command
    command = {
        'a': 'up',  # Update user
        'currk': old_hash,
        'k': key_data,
        'uh': new_hash,
    }
    
    try:
        result = single_api_request(command)
        
        if isinstance(result, int) and result < 0:
            if result == -2:
                raise RequestError("Current password is incorrect")
            else:
                raise RequestError(f"Password change failed with error code: {result}")
        
        # Update session with new master key
        current_session.master_key = new_master_key
        save_user_session()
        
        logger.info("Password changed successfully")
        return True
        
    except Exception:
        raise RequestError("Password change failed")


# ==============================================
# === UTILITY FUNCTIONS ===
# ==============================================

def is_logged_in() -> bool:
    """Check if user is currently logged in."""
    return current_session.is_authenticated and current_session.session_id is not None


def get_current_user() -> Optional[str]:
    """Get current user's email address."""
    return current_session.email if current_session.is_authenticated else None


def require_authentication(func):
    """Decorator to require authentication for functions."""
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            raise RequestError("Authentication required")
        return func(*args, **kwargs)
    return wrapper


# ==============================================
# === ENHANCED AUTHENTICATION WITH EVENTS ===
# ==============================================

def login_with_events(email: str, password: str, save_session: bool = True, 
                     event_callback=None) -> UserSession:
    """
    Enhanced login function with event callbacks and logging.
    
    Args:
        email: User's email address
        password: User's password
        save_session: Whether to save session for persistence
        event_callback: Function to call for events (optional)
        
    Returns:
        Authenticated user session
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        session = login(email, password, save_session)
        logger.info(f"Successfully logged in as {email}")
        trigger_event('login', {'email': email, 'session': session})
        return session
        
    except Exception as e:
        logger.error(f"Login failed: {e}")
        trigger_event('login_failed', {'email': email, 'error': str(e)})
        raise


def logout_with_events(event_callback=None) -> None:
    """
    Enhanced logout function with event callbacks and logging.
    
    Args:
        event_callback: Function to call for events (optional)
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    if current_session.is_authenticated:
        current_email = current_session.email
        logout()
        logger.info(f"Logged out user: {current_email}")
        trigger_event('logout', {'email': current_email})
    else:
        logger.warning("No user logged in")


def register_with_events(email: str, password: str, first_name: str = "", 
                        last_name: str = "", event_callback=None) -> bool:
    """
    Enhanced register function with event callbacks and logging.
    
    Args:
        email: User's email address
        password: User's password  
        first_name: User's first name (optional)
        last_name: User's last name (optional)
        event_callback: Function to call for events (optional)
        
    Returns:
        True if registration initiated successfully
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        result = register(email, password, first_name, last_name)
        logger.info(f"Registration initiated for {email}")
        trigger_event('registration', {'email': email, 'success': result})
        return result
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        trigger_event('registration_failed', {'email': email, 'error': str(e)})
        raise


def verify_email_with_events(email: str, verification_code: str, 
                            event_callback=None) -> bool:
    """
    Enhanced email verification function with event callbacks and logging.
    
    Args:
        email: User's email address
        verification_code: Verification code from email
        event_callback: Function to call for events (optional)
        
    Returns:
        True if verification successful
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        result = verify_email(email, verification_code)
        logger.info(f"Email verified for {email}")
        trigger_event('email_verified', {'email': email})
        return result
    except Exception as e:
        logger.error(f"Email verification failed: {e}")
        raise


def change_password_with_events(old_password: str, new_password: str, 
                               event_callback=None) -> bool:
    """
    Enhanced change password function with event callbacks and logging.
    
    Args:
        old_password: Current password
        new_password: New password
        event_callback: Function to call for events (optional)
        
    Returns:
        True if password changed successfully
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    if not current_session.is_authenticated:
        raise RequestError("Not logged in")
    
    try:
        result = change_password(old_password, new_password)
        logger.info("Password changed successfully")
        trigger_event('password_changed', {})
        return result
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        raise


# ==============================================
# === CLIENT METHOD INJECTION ===
# ==============================================

def add_authentication_methods(client_class):
    """Add authentication methods to the MPLClient class."""
    
    def login_method(self, email: str, password: str, save_session: bool = True) -> bool:
        """
        Log in to Mega with email and password.
        
        Args:
            email: User's email address
            password: User's password
            save_session: Whether to save session for automatic restoration
            
        Returns:
            True if login successful
            
        Raises:
            ValidationError: If email/password format is invalid
            RequestError: If authentication fails
        """
        from .filesystem import fs_tree, get_nodes
        
        session = login_with_events(email, password, save_session, getattr(self, '_trigger_event', None))
        
        # Load filesystem
        if hasattr(self, '_refresh_filesystem_if_needed'):
            self._refresh_filesystem_if_needed()
        else:
            # Fallback direct filesystem refresh
            if not fs_tree.nodes:
                get_nodes()
        
        return True
    
    def logout_method(self) -> None:
        """Log out current user and clear all data."""
        from .filesystem import fs_tree
        
        logout_with_events(getattr(self, '_trigger_event', None))
        fs_tree.clear()
    
    def register_method(self, email: str, password: str, first_name: str = "", 
                       last_name: str = "") -> bool:
        """
        Register new user account.
        
        Args:
            email: User's email address
            password: User's password
            first_name: User's first name (optional)
            last_name: User's last name (optional)
            
        Returns:
            True if registration initiated successfully
            
        Raises:
            ValidationError: If email/password format is invalid
            RequestError: If registration fails
        """
        return register_with_events(email, password, first_name, last_name, getattr(self, '_trigger_event', None))
    
    def verify_email_method(self, email: str, verification_code: str) -> bool:
        """
        Verify email address with confirmation code.
        
        Args:
            email: User's email address
            verification_code: Verification code from email
            
        Returns:
            True if verification successful
        """
        return verify_email_with_events(email, verification_code, getattr(self, '_trigger_event', None))
    
    def change_password_method(self, old_password: str, new_password: str) -> bool:
        """
        Change current user's password.
        
        Args:
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully
        """
        return change_password_with_events(old_password, new_password, getattr(self, '_trigger_event', None))
    
    def get_user_info_method(self) -> Dict[str, Any]:
        """Get current user information."""
        return get_user_info()
    
    def get_quota_method(self) -> Dict[str, int]:
        """Get user storage and transfer quota."""
        return get_user_quota()
    
    def is_logged_in_method(self) -> bool:
        """Check if user is currently logged in."""
        return is_logged_in()
    
    def get_current_user_method(self) -> Optional[str]:
        """Get current user's email address."""
        return get_current_user()
    
    # Add methods to client class
    setattr(client_class, 'login', login_method)
    setattr(client_class, 'logout', logout_method)
    setattr(client_class, 'register', register_method)
    setattr(client_class, 'verify_email', verify_email_method)
    setattr(client_class, 'change_password', change_password_method)
    setattr(client_class, 'get_user_info', get_user_info_method)
    setattr(client_class, 'get_quota', get_quota_method)
    setattr(client_class, 'is_logged_in', is_logged_in_method)
    setattr(client_class, 'get_current_user', get_current_user_method)


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Session class
    'UserSession',
    'current_session',
    
    # Core authentication functions
    'login',
    'logout',
    'register',
    'verify_email',
    
    # Enhanced authentication with events
    'login_with_events',
    'logout_with_events', 
    'register_with_events',
    'verify_email_with_events',
    'change_password_with_events',
    
    # Session persistence
    'save_user_session',
    'load_user_session',
    'clear_saved_session',
    
    # User information
    'get_user_info',
    'get_user_quota',
    
    # Password management
    'change_password',
    
    # Utilities
    'is_logged_in',
    'get_current_user',
    'require_authentication',
    
    # Client integration
    'add_authentication_methods',
]

# Configure logging
logger = logging.getLogger(__name__)

