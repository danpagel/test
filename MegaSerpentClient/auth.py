"""
MegaSerpentClient - Authentication & Security Module

Purpose: Authentication, authorization, cryptography, and all security operations.

This module handles all aspects of security including login/logout, session management,
multi-factor authentication, cryptographic operations, key management, and security policies.
"""

import hashlib
import hmac
import secrets
import base64
import time
import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from . import utils
from .utils import (
    Constants, MegaError, AuthenticationError, SecurityError, ValidationError,
    Validators, CryptoUtils, DateTimeUtils, Helpers
)

# Import crypto libraries with fallbacks
try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import PBKDF2
    from Crypto.Hash import SHA256, HMAC
    from Crypto.PublicKey import RSA, ECC
    from Crypto.Signature import pss, DSS
    from Crypto.Util.Padding import pad, unpad
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# ==============================================
# === AUTHENTICATION ENUMS AND CONSTANTS ===
# ==============================================

class AuthMethod(Enum):
    """Authentication methods."""
    EMAIL_PASSWORD = "email_password"
    OAUTH = "oauth"
    SAML = "saml"
    ENTERPRISE = "enterprise"
    API_KEY = "api_key"
    MFA_TOTP = "mfa_totp"


class SessionState(Enum):
    """Session state enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


class MFAType(Enum):
    """Multi-factor authentication types."""
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BACKUP_CODES = "backup_codes"


class KeyType(Enum):
    """Cryptographic key types."""
    AES = "aes"
    RSA = "rsa"
    ECC = "ecc"
    MASTER = "master"


# ==============================================
# === DATA CLASSES ===
# ==============================================

@dataclass
class UserCredentials:
    """User credentials container."""
    email: str
    password_hash: str
    salt: bytes
    created_at: datetime = field(default_factory=DateTimeUtils.now_utc)
    last_modified: datetime = field(default_factory=DateTimeUtils.now_utc)


@dataclass
class SessionInfo:
    """Session information."""
    session_id: str
    user_id: str
    email: str
    created_at: datetime
    expires_at: datetime
    last_accessed: datetime
    state: SessionState = SessionState.ACTIVE
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    refresh_token: Optional[str] = None


@dataclass
class MFAConfig:
    """Multi-factor authentication configuration."""
    enabled: bool = False
    primary_method: Optional[MFAType] = None
    backup_methods: List[MFAType] = field(default_factory=list)
    totp_secret: Optional[str] = None
    backup_codes: List[str] = field(default_factory=list)
    phone_number: Optional[str] = None


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    min_password_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    session_timeout: int = 3600  # 1 hour
    require_mfa: bool = False
    password_expiry_days: Optional[int] = None


# ==============================================
# === CREDENTIAL MANAGER ===
# ==============================================

class CredentialManager:
    """Secure credential storage and management."""
    
    def __init__(self):
        self._credentials: Dict[str, UserCredentials] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def store_credentials(self, email: str, password: str) -> UserCredentials:
        """Store user credentials securely."""
        if not Validators.validate_email(email):
            raise ValidationError("Invalid email format")
        
        # Generate salt and hash password
        salt = secrets.token_bytes(32)
        password_hash = self._hash_password(password, salt)
        
        credentials = UserCredentials(
            email=email,
            password_hash=password_hash,
            salt=salt
        )
        
        with self._lock:
            self._credentials[email] = credentials
        
        self.logger.info(f"Stored credentials for user: {email}")
        return credentials
    
    def verify_credentials(self, email: str, password: str) -> bool:
        """Verify user credentials."""
        with self._lock:
            if email not in self._credentials:
                return False
            
            credentials = self._credentials[email]
        
        # Verify password
        password_hash = self._hash_password(password, credentials.salt)
        return hmac.compare_digest(password_hash, credentials.password_hash)
    
    def get_credentials(self, email: str) -> Optional[UserCredentials]:
        """Get stored credentials."""
        with self._lock:
            return self._credentials.get(email)
    
    def update_password(self, email: str, new_password: str) -> bool:
        """Update user password."""
        with self._lock:
            if email not in self._credentials:
                return False
            
            # Generate new salt and hash
            salt = secrets.token_bytes(32)
            password_hash = self._hash_password(new_password, salt)
            
            credentials = self._credentials[email]
            credentials.password_hash = password_hash
            credentials.salt = salt
            credentials.last_modified = DateTimeUtils.now_utc()
        
        self.logger.info(f"Updated password for user: {email}")
        return True
    
    def delete_credentials(self, email: str) -> bool:
        """Delete stored credentials."""
        with self._lock:
            if email in self._credentials:
                del self._credentials[email]
                self.logger.info(f"Deleted credentials for user: {email}")
                return True
        return False
    
    def _hash_password(self, password: str, salt: bytes) -> str:
        """Hash password with salt using PBKDF2."""
        if not CRYPTO_AVAILABLE:
            # Fallback to basic hashing
            return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000).hex()
        
        key = PBKDF2(password, salt, 32, count=100000, hmac_hash_module=SHA256)
        return key.hex()


# ==============================================
# === SESSION MANAGER ===
# ==============================================

class SessionManager:
    """Session lifecycle management."""
    
    def __init__(self, security_policy: SecurityPolicy):
        self.security_policy = security_policy
        self._sessions: Dict[str, SessionInfo] = {}
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def create_session(self, user_id: str, email: str, ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None) -> SessionInfo:
        """Create new user session."""
        session_id = self._generate_session_id()
        refresh_token = self._generate_refresh_token()
        
        now = DateTimeUtils.now_utc()
        expires_at = now + timedelta(seconds=self.security_policy.session_timeout)
        
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            email=email,
            created_at=now,
            expires_at=expires_at,
            last_accessed=now,
            ip_address=ip_address,
            user_agent=user_agent,
            refresh_token=refresh_token
        )
        
        with self._lock:
            self._sessions[session_id] = session_info
            
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)
        
        self.logger.info(f"Created session for user {email}: {session_id}")
        return session_info
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            
            if session and self._is_session_valid(session):
                # Update last accessed time
                session.last_accessed = DateTimeUtils.now_utc()
                return session
            elif session:
                # Mark expired session
                session.state = SessionState.EXPIRED
        
        return None
    
    def validate_session(self, session_id: str) -> bool:
        """Validate session and update access time."""
        session = self.get_session(session_id)
        return session is not None and session.state == SessionState.ACTIVE
    
    def refresh_session(self, session_id: str, refresh_token: str) -> Optional[SessionInfo]:
        """Refresh session using refresh token."""
        with self._lock:
            session = self._sessions.get(session_id)
            
            if not session or session.refresh_token != refresh_token:
                return None
            
            # Extend session
            now = DateTimeUtils.now_utc()
            session.expires_at = now + timedelta(seconds=self.security_policy.session_timeout)
            session.last_accessed = now
            session.refresh_token = self._generate_refresh_token()
        
        self.logger.info(f"Refreshed session: {session_id}")
        return session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a specific session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].state = SessionState.REVOKED
                self.logger.info(f"Revoked session: {session_id}")
                return True
        return False
    
    def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user."""
        revoked_count = 0
        
        with self._lock:
            session_ids = self._user_sessions.get(user_id, [])
            
            for session_id in session_ids:
                if session_id in self._sessions:
                    self._sessions[session_id].state = SessionState.REVOKED
                    revoked_count += 1
        
        self.logger.info(f"Revoked {revoked_count} sessions for user: {user_id}")
        return revoked_count
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        cleaned_count = 0
        now = DateTimeUtils.now_utc()
        
        with self._lock:
            expired_sessions = []
            
            for session_id, session in self._sessions.items():
                if session.expires_at < now or session.state != SessionState.ACTIVE:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                session = self._sessions[session_id]
                
                # Remove from user sessions
                if session.user_id in self._user_sessions:
                    user_sessions = self._user_sessions[session.user_id]
                    if session_id in user_sessions:
                        user_sessions.remove(session_id)
                
                # Remove session
                del self._sessions[session_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} expired sessions")
        
        return cleaned_count
    
    def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all active sessions for a user."""
        sessions = []
        
        with self._lock:
            session_ids = self._user_sessions.get(user_id, [])
            
            for session_id in session_ids:
                session = self._sessions.get(session_id)
                if session and self._is_session_valid(session):
                    sessions.append(session)
        
        return sessions
    
    def _is_session_valid(self, session: SessionInfo) -> bool:
        """Check if session is valid."""
        if session.state != SessionState.ACTIVE:
            return False
        
        return session.expires_at > DateTimeUtils.now_utc()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return secrets.token_urlsafe(32)
    
    def _generate_refresh_token(self) -> str:
        """Generate refresh token."""
        return secrets.token_urlsafe(32)


# ==============================================
# === LOGIN MANAGER ===
# ==============================================

class LoginManager:
    """Login/logout operations management."""
    
    def __init__(self, credential_manager: CredentialManager, 
                 session_manager: SessionManager, security_policy: SecurityPolicy):
        self.credential_manager = credential_manager
        self.session_manager = session_manager
        self.security_policy = security_policy
        self._login_attempts: Dict[str, List[datetime]] = {}
        self._locked_accounts: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def login(self, email: str, password: str, ip_address: Optional[str] = None,
              user_agent: Optional[str] = None) -> Optional[SessionInfo]:
        """Perform user login."""
        # Validate input
        if not Validators.validate_email(email):
            raise ValidationError("Invalid email format")
        
        # Check if account is locked
        if self._is_account_locked(email):
            raise AuthenticationError("Account is temporarily locked due to too many failed attempts")
        
        # Verify credentials
        if not self.credential_manager.verify_credentials(email, password):
            self._record_failed_attempt(email)
            raise AuthenticationError("Invalid email or password")
        
        # Clear failed attempts on successful login
        self._clear_failed_attempts(email)
        
        # Create session
        user_id = self._get_user_id(email)
        session = self.session_manager.create_session(
            user_id=user_id,
            email=email,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.logger.info(f"User logged in successfully: {email}")
        return session
    
    def logout(self, session_id: str) -> bool:
        """Perform user logout."""
        if self.session_manager.revoke_session(session_id):
            self.logger.info(f"User logged out: {session_id}")
            return True
        return False
    
    def logout_all_sessions(self, user_id: str) -> int:
        """Logout user from all sessions."""
        count = self.session_manager.revoke_all_user_sessions(user_id)
        self.logger.info(f"Logged out user from {count} sessions: {user_id}")
        return count
    
    def is_logged_in(self, session_id: str) -> bool:
        """Check if user is logged in."""
        return self.session_manager.validate_session(session_id)
    
    def _is_account_locked(self, email: str) -> bool:
        """Check if account is locked."""
        with self._lock:
            if email in self._locked_accounts:
                lock_time = self._locked_accounts[email]
                if DateTimeUtils.now_utc() < lock_time + timedelta(seconds=self.security_policy.lockout_duration):
                    return True
                else:
                    # Lock expired, remove it
                    del self._locked_accounts[email]
        return False
    
    def _record_failed_attempt(self, email: str):
        """Record failed login attempt."""
        now = DateTimeUtils.now_utc()
        
        with self._lock:
            if email not in self._login_attempts:
                self._login_attempts[email] = []
            
            self._login_attempts[email].append(now)
            
            # Remove attempts older than 1 hour
            cutoff = now - timedelta(hours=1)
            self._login_attempts[email] = [
                attempt for attempt in self._login_attempts[email]
                if attempt > cutoff
            ]
            
            # Check if account should be locked
            if len(self._login_attempts[email]) >= self.security_policy.max_login_attempts:
                self._locked_accounts[email] = now
                self.logger.warning(f"Account locked due to too many failed attempts: {email}")
    
    def _clear_failed_attempts(self, email: str):
        """Clear failed login attempts."""
        with self._lock:
            if email in self._login_attempts:
                del self._login_attempts[email]
            if email in self._locked_accounts:
                del self._locked_accounts[email]
    
    def _get_user_id(self, email: str) -> str:
        """Get or generate user ID for email."""
        # In a real implementation, this would query a user database
        return hashlib.sha256(email.encode()).hexdigest()[:16]


# ==============================================
# === SIGNUP MANAGER ===
# ==============================================

class SignupManager:
    """Account creation and registration management."""
    
    def __init__(self, credential_manager: CredentialManager, security_policy: SecurityPolicy):
        self.credential_manager = credential_manager
        self.security_policy = security_policy
        self._pending_confirmations: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def signup(self, email: str, password: str, confirm_password: str) -> Dict[str, Any]:
        """Create new user account."""
        # Validate email
        if not Validators.validate_email(email):
            raise ValidationError("Invalid email format")
        
        # Check if user already exists
        if self.credential_manager.get_credentials(email):
            raise ValidationError("Account with this email already exists")
        
        # Validate password match
        if password != confirm_password:
            raise ValidationError("Passwords do not match")
        
        # Validate password strength
        password_checks = Validators.validate_password(password)
        if not self._meets_password_policy(password_checks):
            raise ValidationError("Password does not meet security requirements")
        
        # Store credentials
        credentials = self.credential_manager.store_credentials(email, password)
        
        # Generate confirmation token
        confirmation_token = self._generate_confirmation_token()
        
        with self._lock:
            self._pending_confirmations[confirmation_token] = {
                'email': email,
                'created_at': DateTimeUtils.now_utc(),
                'expires_at': DateTimeUtils.now_utc() + timedelta(hours=24)
            }
        
        self.logger.info(f"User account created (pending confirmation): {email}")
        
        return {
            'email': email,
            'confirmation_token': confirmation_token,
            'status': 'pending_confirmation'
        }
    
    def confirm_signup(self, confirmation_token: str) -> bool:
        """Confirm user account creation."""
        with self._lock:
            if confirmation_token not in self._pending_confirmations:
                return False
            
            confirmation_data = self._pending_confirmations[confirmation_token]
            
            # Check if token expired
            if DateTimeUtils.now_utc() > confirmation_data['expires_at']:
                del self._pending_confirmations[confirmation_token]
                return False
            
            email = confirmation_data['email']
            del self._pending_confirmations[confirmation_token]
        
        self.logger.info(f"User account confirmed: {email}")
        return True
    
    def resend_confirmation(self, email: str) -> Optional[str]:
        """Resend confirmation email."""
        # Check if user exists and is pending confirmation
        if not self.credential_manager.get_credentials(email):
            return None
        
        # Generate new confirmation token
        confirmation_token = self._generate_confirmation_token()
        
        with self._lock:
            # Remove old token if exists
            old_tokens = [
                token for token, data in self._pending_confirmations.items()
                if data['email'] == email
            ]
            for token in old_tokens:
                del self._pending_confirmations[token]
            
            # Add new token
            self._pending_confirmations[confirmation_token] = {
                'email': email,
                'created_at': DateTimeUtils.now_utc(),
                'expires_at': DateTimeUtils.now_utc() + timedelta(hours=24)
            }
        
        self.logger.info(f"Resent confirmation for user: {email}")
        return confirmation_token
    
    def _meets_password_policy(self, checks: Dict[str, bool]) -> bool:
        """Check if password meets security policy."""
        if self.security_policy.min_password_length > 8:
            # Additional length check
            pass
        
        required_checks = []
        if self.security_policy.require_uppercase:
            required_checks.append('uppercase')
        if self.security_policy.require_lowercase:
            required_checks.append('lowercase')
        if self.security_policy.require_digits:
            required_checks.append('digit')
        if self.security_policy.require_special_chars:
            required_checks.append('special')
        
        return all(checks.get(check, False) for check in required_checks)
    
    def _generate_confirmation_token(self) -> str:
        """Generate confirmation token."""
        return secrets.token_urlsafe(32)


# ==============================================
# === MFA MANAGER ===
# ==============================================

class MFAManager:
    """Multi-factor authentication management."""
    
    def __init__(self):
        self._user_mfa_configs: Dict[str, MFAConfig] = {}
        self._pending_mfa: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def setup_totp(self, user_id: str) -> Dict[str, Any]:
        """Setup TOTP for user."""
        secret = self._generate_totp_secret()
        
        with self._lock:
            if user_id not in self._user_mfa_configs:
                self._user_mfa_configs[user_id] = MFAConfig()
            
            config = self._user_mfa_configs[user_id]
            config.totp_secret = secret
            config.primary_method = MFAType.TOTP
            config.enabled = True
        
        # Generate QR code data
        qr_data = f"otpauth://totp/MegaSerpentClient:{user_id}?secret={secret}&issuer=MegaSerpentClient"
        
        self.logger.info(f"TOTP setup for user: {user_id}")
        
        return {
            'secret': secret,
            'qr_code_data': qr_data,
            'backup_codes': self._generate_backup_codes(user_id)
        }
    
    def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token."""
        with self._lock:
            config = self._user_mfa_configs.get(user_id)
            if not config or not config.totp_secret:
                return False
        
        # In a real implementation, this would use a TOTP library
        # For now, we'll implement a basic version
        return self._verify_totp_token(config.totp_secret, token)
    
    def generate_backup_codes(self, user_id: str) -> List[str]:
        """Generate backup codes for user."""
        codes = self._generate_backup_codes(user_id)
        
        with self._lock:
            if user_id not in self._user_mfa_configs:
                self._user_mfa_configs[user_id] = MFAConfig()
            
            self._user_mfa_configs[user_id].backup_codes = codes
        
        self.logger.info(f"Generated backup codes for user: {user_id}")
        return codes
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify and consume backup code."""
        with self._lock:
            config = self._user_mfa_configs.get(user_id)
            if not config or code not in config.backup_codes:
                return False
            
            # Remove used backup code
            config.backup_codes.remove(code)
        
        self.logger.info(f"Backup code used for user: {user_id}")
        return True
    
    def is_mfa_enabled(self, user_id: str) -> bool:
        """Check if MFA is enabled for user."""
        with self._lock:
            config = self._user_mfa_configs.get(user_id)
            return config is not None and config.enabled
    
    def disable_mfa(self, user_id: str) -> bool:
        """Disable MFA for user."""
        with self._lock:
            if user_id in self._user_mfa_configs:
                self._user_mfa_configs[user_id].enabled = False
                self.logger.info(f"MFA disabled for user: {user_id}")
                return True
        return False
    
    def get_mfa_config(self, user_id: str) -> Optional[MFAConfig]:
        """Get MFA configuration for user."""
        with self._lock:
            return self._user_mfa_configs.get(user_id)
    
    def _generate_totp_secret(self) -> str:
        """Generate TOTP secret."""
        return secrets.token_urlsafe(20)
    
    def _generate_backup_codes(self, user_id: str) -> List[str]:
        """Generate backup codes."""
        return [secrets.token_hex(4).upper() for _ in range(10)]
    
    def _verify_totp_token(self, secret: str, token: str) -> bool:
        """Verify TOTP token (simplified implementation)."""
        # This is a simplified implementation
        # In production, use a proper TOTP library like pyotp
        try:
            # Basic time-based validation
            current_time = int(time.time() // 30)
            
            # Allow for time drift (±1 window)
            for time_window in [current_time - 1, current_time, current_time + 1]:
                expected_token = self._generate_totp_token(secret, time_window)
                if hmac.compare_digest(token, expected_token):
                    return True
            
            return False
        except Exception:
            return False
    
    def _generate_totp_token(self, secret: str, time_window: int) -> str:
        """Generate TOTP token for given time window."""
        # Simplified TOTP implementation
        key = base64.b32decode(secret.upper() + '=' * (-len(secret) % 8))
        msg = time_window.to_bytes(8, byteorder='big')
        
        if CRYPTO_AVAILABLE:
            h = HMAC.new(key, msg, SHA256)
            hash_value = h.digest()
        else:
            hash_value = hmac.new(key, msg, hashlib.sha256).digest()
        
        offset = hash_value[-1] & 0xf
        code = hash_value[offset:offset+4]
        code = int.from_bytes(code, byteorder='big') & 0x7fffffff
        
        return f"{code % 1000000:06d}"


# ==============================================
# === CRYPTOGRAPHY CLASSES ===
# ==============================================

class SymmetricCrypto:
    """AES encryption for file data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def encrypt(self, data: bytes, key: bytes, iv: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Encrypt data using AES-CBC."""
        if not CRYPTO_AVAILABLE:
            raise SecurityError("Cryptography library not available")
        
        if len(key) not in [16, 24, 32]:
            raise ValidationError("Key must be 16, 24, or 32 bytes")
        
        if iv is None:
            iv = get_random_bytes(AES.block_size)
        
        cipher = AES.new(key, AES.MODE_CBC, iv)
        padded_data = pad(data, AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return encrypted_data, iv
    
    def decrypt(self, encrypted_data: bytes, key: bytes, iv: bytes) -> bytes:
        """Decrypt data using AES-CBC."""
        if not CRYPTO_AVAILABLE:
            raise SecurityError("Cryptography library not available")
        
        cipher = AES.new(key, AES.MODE_CBC, iv)
        padded_data = cipher.decrypt(encrypted_data)
        data = unpad(padded_data, AES.block_size)
        
        return data
    
    def encrypt_stream(self, data_stream, key: bytes, chunk_size: int = 8192):
        """Encrypt data stream."""
        if not CRYPTO_AVAILABLE:
            raise SecurityError("Cryptography library not available")
        
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        yield iv  # First yield the IV
        
        for chunk in data_stream:
            if len(chunk) % AES.block_size != 0:
                chunk = pad(chunk, AES.block_size)
            yield cipher.encrypt(chunk)


class AsymmetricCrypto:
    """RSA/ECC for key exchange."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_rsa_key_pair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate RSA key pair."""
        if not CRYPTO_AVAILABLE:
            raise SecurityError("Cryptography library not available")
        
        key = RSA.generate(key_size)
        private_key = key.export_key()
        public_key = key.publickey().export_key()
        
        return private_key, public_key
    
    def generate_ecc_key_pair(self, curve: str = 'P-256') -> Tuple[bytes, bytes]:
        """Generate ECC key pair."""
        if not CRYPTO_AVAILABLE:
            raise SecurityError("Cryptography library not available")
        
        key = ECC.generate(curve=curve)
        private_key = key.export_key(format='PEM')
        public_key = key.public_key().export_key(format='PEM')
        
        return private_key.encode(), public_key.encode()


class HybridCrypto:
    """Combined encryption strategies."""
    
    def __init__(self):
        self.symmetric = SymmetricCrypto()
        self.asymmetric = AsymmetricCrypto()
        self.logger = logging.getLogger(__name__)
    
    def encrypt_hybrid(self, data: bytes, public_key: bytes) -> Dict[str, bytes]:
        """Encrypt using hybrid approach (RSA + AES)."""
        # Generate symmetric key
        symmetric_key = secrets.token_bytes(32)  # 256-bit key
        
        # Encrypt data with symmetric key
        encrypted_data, iv = self.symmetric.encrypt(data, symmetric_key)
        
        # Encrypt symmetric key with public key
        if not CRYPTO_AVAILABLE:
            raise SecurityError("Cryptography library not available")
        
        rsa_key = RSA.import_key(public_key)
        cipher_rsa = pss.new(rsa_key)
        encrypted_key = cipher_rsa.encrypt(symmetric_key)
        
        return {
            'encrypted_data': encrypted_data,
            'encrypted_key': encrypted_key,
            'iv': iv
        }


class KeyGenerator:
    """Generate cryptographic keys."""
    
    @staticmethod
    def generate_aes_key(key_size: int = 256) -> bytes:
        """Generate AES key."""
        if key_size not in [128, 192, 256]:
            raise ValidationError("Key size must be 128, 192, or 256 bits")
        
        return secrets.token_bytes(key_size // 8)
    
    @staticmethod
    def generate_master_key() -> bytes:
        """Generate master key."""
        return secrets.token_bytes(32)  # 256-bit master key
    
    @staticmethod
    def derive_key(password: str, salt: bytes, iterations: int = 100000) -> bytes:
        """Derive key from password using PBKDF2."""
        if not CRYPTO_AVAILABLE:
            return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations, 32)
        
        return PBKDF2(password, salt, 32, count=iterations, hmac_hash_module=SHA256)


class MasterKey:
    """Master key operations."""
    
    def __init__(self):
        self._master_key: Optional[bytes] = None
        self._derived_keys: Dict[str, bytes] = {}
        self.logger = logging.getLogger(__name__)
    
    def set_master_key(self, key: bytes):
        """Set master key."""
        if len(key) != 32:
            raise ValidationError("Master key must be 32 bytes")
        self._master_key = key
        self.logger.info("Master key set")
    
    def derive_key(self, purpose: str, info: str = "") -> bytes:
        """Derive key for specific purpose."""
        if not self._master_key:
            raise SecurityError("Master key not set")
        
        key_id = f"{purpose}:{info}"
        if key_id in self._derived_keys:
            return self._derived_keys[key_id]
        
        # Simple key derivation using HMAC
        derived_key = hmac.new(
            self._master_key,
            f"{purpose}:{info}".encode(),
            hashlib.sha256
        ).digest()
        
        self._derived_keys[key_id] = derived_key
        return derived_key
    
    def rotate_master_key(self) -> bytes:
        """Rotate master key."""
        new_key = KeyGenerator.generate_master_key()
        self._master_key = new_key
        self._derived_keys.clear()  # Clear derived keys
        self.logger.info("Master key rotated")
        return new_key


# ==============================================
# === MAIN MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Enums
    'AuthMethod', 'SessionState', 'MFAType', 'KeyType',
    
    # Data Classes
    'UserCredentials', 'SessionInfo', 'MFAConfig', 'SecurityPolicy',
    
    # Authentication Managers
    'CredentialManager', 'SessionManager', 'LoginManager', 'SignupManager', 'MFAManager',
    
    # Cryptography Classes
    'SymmetricCrypto', 'AsymmetricCrypto', 'HybridCrypto', 'KeyGenerator', 'MasterKey'
]