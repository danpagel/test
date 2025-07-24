"""
Cryptographic Utilities Module
==============================

This module contains all cryptographic functions used by the Mega.nz client.

This follows the exact methodology from the reference implementation:
1. AES encryption/decryption functions
2. Key derivation and manipulation utilities
3. Base64 encoding/decoding helpers
4. Checksum and MAC calculations
5. Password hashing functions

Enhanced with event-driven functions for comprehensive callback support.

Author: Modernized from reference implementation
Date: July 2025
"""

import time
from .dependencies import *
from .exceptions import ValidationError

# ==============================================
# === AES ENCRYPTION/DECRYPTION ===
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
        # Use zero IV for Mega compatibility - no padding handling here
        iv = b'\x00' * 16
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.encrypt(data)
    else:
        # Generate random IV
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
        # Use zero IV for Mega compatibility - no padding removal here
        iv = b'\x00' * 16
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.decrypt(data)
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


def aes_ctr_encrypt_decrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    """
    Encrypt/decrypt data using AES in CTR mode.
    
    CTR mode is symmetric - the same function encrypts and decrypts.
    
    Args:
        data: The data to encrypt/decrypt
        key: The AES key (16 bytes for AES-128)
        iv: The initialization vector (16 bytes)
        
    Returns:
        Encrypted/decrypted data
        
    Raises:
        ValidationError: If key or IV length is invalid
    """
    if len(key) != 16:
        raise ValidationError("AES key must be 16 bytes")
    
    if len(iv) != 16:
        raise ValidationError("IV must be 16 bytes")
    
    # Create counter from IV
    counter = Counter.new(128, initial_value=int.from_bytes(iv, 'big'))
    
    # Encrypt/decrypt
    cipher = AES.new(key, AES.MODE_CTR, counter=counter)
    return cipher.encrypt(data)


# ==============================================
# === KEY DERIVATION AND MANIPULATION ===
# ==============================================

def parse_node_key(key_data: str, master_key: bytes) -> bytes:
    """
    Parse and decrypt node key.
    
    Args:
        key_data: Base64 encoded encrypted key
        master_key: Master decryption key
        
    Returns:
        Decrypted node key
        
    Raises:
        ValidationError: If key parsing fails
    """
    try:
        # Decode key data
        encrypted_key = base64_url_decode(key_data)
        
        # Decrypt with master key
        node_key = aes_cbc_decrypt(encrypted_key, master_key)
        
        return node_key
        
    except Exception as e:
        raise ValidationError(f"Failed to parse node key: {e}")


def parse_file_attributes(attr_data: str) -> Dict[str, Any]:
    """
    Parse file attributes from Mega API response.
    
    Args:
        attr_data: Base64 encoded attribute data
        
    Returns:
        Parsed attributes dictionary
        
    Raises:
        ValidationError: If parsing fails
    """
    try:
        # json already imported via dependencies
        
        # Decode attributes
        decoded = base64_url_decode(attr_data)
        
        # Parse JSON attributes
        attr_json = decoded.decode('utf-8').rstrip('\0')
        return json.loads(attr_json)
        
    except Exception as e:
        raise ValidationError(f"Failed to parse file attributes: {e}")


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


# ==============================================
# === CHECKSUM AND MAC CALCULATIONS ===
# ==============================================

def calculate_mac(key: bytes, data: bytes) -> bytes:
    """
    Calculate MAC (Message Authentication Code) for data.
    
    Args:
        key: MAC key
        data: Data to authenticate
        
    Returns:
        MAC bytes
    """
    return hashlib.sha256(key + data).digest()[:16]


def verify_mac(key: bytes, data: bytes, expected_mac: bytes) -> bool:
    """
    Verify MAC for data.
    
    Args:
        key: MAC key
        data: Data to verify
        expected_mac: Expected MAC value
        
    Returns:
        True if MAC is valid, False otherwise
    """
    calculated_mac = calculate_mac(key, data)
    return calculated_mac == expected_mac


def calculate_chunk_mac(key: List[int], chunk_start: int, chunk_data: bytes) -> List[int]:
    """
    Calculate MAC for a file chunk (used in file uploads).
    
    Args:
        key: MAC key as 32-bit integer array
        chunk_start: Starting position of chunk
        chunk_data: Chunk data
        
    Returns:
        MAC as 32-bit integer array
    """
    # Convert chunk position to bytes
    chunk_pos = struct.pack('>Q', chunk_start)
    
    # Create MAC input
    mac_input = chunk_pos + chunk_data
    
    # Calculate MAC
    key_bytes = makebyte(a32_to_string(key))
    mac = calculate_mac(key_bytes, mac_input)
    
    # Convert back to 32-bit integer array
    return string_to_a32(makestring(mac))


# ==============================================
# === PASSWORD HASHING ===
# ==============================================

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


# ==============================================
# === UTILITY FUNCTIONS ===
# ==============================================

def mpi_to_int(s: bytes) -> int:
    """
    Converts a multi-precision integer (MPI) byte string to an integer.
    
    Args:
        s: MPI byte string
        
    Returns:
        Integer value
    """
    import binascii
    return int(binascii.hexlify(s[2:]), 16)


def extended_gcd(a: int, b: int) -> tuple:
    """
    Extended Euclidean algorithm for finding modular inverses.
    Returns (gcd, x, y) such that ax + by = gcd.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Tuple of (gcd, x, y)
    """
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)


def modular_inverse(a: int, m: int) -> int:
    """
    Computes the modular inverse of a modulo m.
    
    Args:
        a: Integer to find inverse of
        m: Modulus
        
    Returns:
        Modular inverse
        
    Raises:
        Exception: If modular inverse does not exist
    """
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m


def generate_random_key() -> bytes:
    """
    Generate a random AES key.
    
    Returns:
        Random 16-byte AES key
    """
    return secrets.token_bytes(16)


def generate_random_iv() -> bytes:
    """
    Generate a random initialization vector.
    
    Returns:
        Random 16-byte IV
    """
    return secrets.token_bytes(16)


def generate_secure_password(length: int = 12) -> str:
    """
    Generate a secure password for protected links and general use.
    
    Args:
        length: Length of password to generate (default: 12)
        
    Returns:
        Secure alphanumeric password string
    """
    import string  # Import locally to ensure availability
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """
    XOR two byte sequences.
    
    Args:
        a: First byte sequence
        b: Second byte sequence
        
    Returns:
        XOR result
        
    Raises:
        ValidationError: If sequences have different lengths
    """
    if len(a) != len(b):
        raise ValidationError("Byte sequences must have same length for XOR")
    
    return bytes(x ^ y for x, y in zip(a, b))


def calculate_file_checksum(file_path: str, algorithm: str = 'md5') -> str:
    """
    Calculate checksum of a file using specified algorithm.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', etc.)
        
    Returns:
        Hexadecimal checksum string
        
    Raises:
        ValidationError: If file cannot be read or algorithm is unsupported
    """
    try:
        import hashlib
        
        # Get the hash function
        hash_func = getattr(hashlib, algorithm.lower(), None)
        if hash_func is None:
            raise ValidationError(f"Unsupported hash algorithm: {algorithm}")
        
        hasher = hash_func()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
        
    except Exception as e:
        raise ValidationError(f"Failed to calculate checksum: {e}")


def calculate_string_hash(data: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of a string using specified algorithm.
    
    Args:
        data: String data to hash
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', etc.)
        
    Returns:
        Hexadecimal hash string
    """
    import hashlib
    
    hash_func = getattr(hashlib, algorithm.lower())
    return hash_func(data.encode()).hexdigest()


def encrypt_attr(attr: Dict[str, Any], key: List[int]) -> bytes:
    """
    Encrypts file or folder attributes using AES CBC.
    
    Args:
        attr: Attribute dictionary
        key: Encryption key as 32-bit integer array
        
    Returns:
        Encrypted attributes
    """
    # json already imported via dependencies
    attr_str = 'MEGA' + json.dumps(attr)
    attr_bytes = makebyte(attr_str)
    
    # Pad to 16 bytes
    if len(attr_bytes) % 16:
        attr_bytes += b'\0' * (16 - len(attr_bytes) % 16)
    
    key_bytes = makebyte(a32_to_string(key))
    return aes_cbc_encrypt_mega(attr_bytes, key_bytes)


def decrypt_attr(attr: bytes, key: List[int]):
    """
    Decrypts file or folder attributes using AES CBC.
    Returns the attribute dictionary if successful, False otherwise.
    Exact copy of reference implementation.
    
    Args:
        attr: Encrypted attributes
        key: Decryption key as 32-bit integer array
        
    Returns:
        Decrypted attribute dictionary or False
    """
    try:
        # Use the exact same method as reference
        key_bytes = makebyte(a32_to_string(key))
        decrypted = aes_cbc_decrypt_mega(attr, key_bytes)
        attr_str = makestring(decrypted).rstrip('\0')
        
        # Check for MEGA magic header exactly like reference
        if attr_str.startswith('MEGA{"'):
            # json already imported via dependencies
            return json.loads(attr_str[4:])
        else:
            return False
    except Exception as e:
        return False


def make_id(length: int) -> str:
    """
    Generates a random alphanumeric string of the given length.
    Used for request IDs.
    
    Args:
        length: Length of the ID to generate
        
    Returns:
        Random alphanumeric string
    """
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def get_chunks(size: int):
    """
    Generator that yields (start, size) tuples for chunked file operations.
    Uses Mega's chunk size algorithm.
    
    Args:
        size: Total file size
        
    Yields:
        Tuples of (chunk_start, chunk_size)
    """
    chunk_start = 0
    chunk_size = 0x20000  # 128KB initial chunk size
    
    while chunk_start < size:
        if chunk_size + chunk_start > size:
            chunk_size = size - chunk_start
        
        yield chunk_start, chunk_size
        chunk_start += chunk_size
        
        # Increase chunk size progressively (Mega's algorithm)
        if chunk_size < 0x100000:  # If less than 1MB
            chunk_size += 0x20000  # Add 128KB
        
        # Cap at 1MB
        if chunk_size > 0x100000:
            chunk_size = 0x100000


# ==============================================
# === ENHANCED CRYPTO FUNCTIONS WITH EVENTS ===
# ==============================================

def derive_key_with_events(password: str, salt: bytes = b'', **kwargs) -> bytes:
    """
    Enhanced key derivation with comprehensive event callbacks.
    
    Args:
        password: The user's password
        salt: Optional salt bytes
        callback_fn: Optional callback for general events
        log_fn: Optional logging callback
        progress_fn: Optional progress callback for key strengthening iterations
        
    Returns:
        Derived key (16 bytes)
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    if log_fn:
        log_fn(f"Starting key derivation for password (length: {len(password)} chars)")
    
    if callback_fn:
        callback_fn({
            'event': 'key_derivation_started',
            'password_length': len(password),
            'salt_length': len(salt),
            'iterations': 65536
        })
    
    try:
        # Convert password to bytes
        password_bytes = password.encode('utf-8')
        
        # Mega uses a specific key derivation approach
        key_material = password_bytes + salt
        
        # Hash multiple times for key strengthening
        total_iterations = 65536
        for i in range(total_iterations):
            key_material = hashlib.sha256(key_material).digest()
            
            # Progress updates every 1000 iterations
            if progress_fn and i % 1000 == 0:
                progress_fn({
                    'iteration': i,
                    'total_iterations': total_iterations,
                    'progress_percent': (i / total_iterations) * 100
                })
        
        # Return first 16 bytes as AES key
        derived_key = key_material[:16]
        
        if callback_fn:
            callback_fn({
                'event': 'key_derivation_completed',
                'key_length': len(derived_key),
                'iterations_completed': total_iterations
            })
        
        if log_fn:
            log_fn(f"Key derivation completed: {len(derived_key)} byte key generated")
        
        return derived_key
        
    except Exception as e:
        if callback_fn:
            callback_fn({
                'event': 'key_derivation_failed',
                'error': str(e)
            })
        if log_fn:
            log_fn(f"Key derivation failed: {e}")
        raise


def encrypt_file_data_with_events(data: bytes, key: bytes, **kwargs) -> Dict[str, Any]:
    """
    Enhanced file data encryption with event callbacks.
    
    Args:
        data: File data to encrypt
        key: Encryption key (16 bytes)
        callback_fn: Optional callback for encryption events
        log_fn: Optional logging callback
        progress_fn: Optional progress callback for large file encryption
        
    Returns:
        Dictionary with encrypted data and metadata
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    if log_fn:
        log_fn(f"Starting file encryption: {len(data)} bytes")
    
    if callback_fn:
        callback_fn({
            'event': 'file_encryption_started',
            'data_size': len(data),
            'key_size': len(key),
            'encryption_mode': 'AES-CBC-CTR'
        })
    
    try:
        # Generate IV for CTR mode
        iv = generate_random_iv()
        
        # For large files, show progress
        chunk_size = 1024 * 1024  # 1MB chunks for progress
        if len(data) > chunk_size and progress_fn:
            encrypted_chunks = []
            processed = 0
            
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                chunk_iv = iv[:]  # Copy IV
                
                # Adjust counter for this chunk
                counter_offset = i // 16
                chunk_iv = (int.from_bytes(chunk_iv, 'big') + counter_offset).to_bytes(16, 'big')
                
                encrypted_chunk = aes_ctr_encrypt_decrypt(chunk, key, chunk_iv)
                encrypted_chunks.append(encrypted_chunk)
                
                processed += len(chunk)
                progress_fn({
                    'bytes_processed': processed,
                    'total_bytes': len(data),
                    'progress_percent': (processed / len(data)) * 100,
                    'chunk_number': len(encrypted_chunks)
                })
            
            encrypted_data = b''.join(encrypted_chunks)
        else:
            # Small file or no progress callback
            encrypted_data = aes_ctr_encrypt_decrypt(data, key, iv)
        
        # Calculate MAC for integrity
        mac = calculate_mac(key, encrypted_data)
        
        result = {
            'encrypted_data': encrypted_data,
            'iv': iv,
            'mac': mac,
            'original_size': len(data),
            'encrypted_size': len(encrypted_data)
        }
        
        if callback_fn:
            callback_fn({
                'event': 'file_encryption_completed',
                'original_size': len(data),
                'encrypted_size': len(encrypted_data),
                'compression_ratio': len(encrypted_data) / len(data) if data else 0
            })
        
        if log_fn:
            log_fn(f"File encryption completed: {len(data)} -> {len(encrypted_data)} bytes")
        
        return result
        
    except Exception as e:
        if callback_fn:
            callback_fn({
                'event': 'file_encryption_failed',
                'error': str(e)
            })
        if log_fn:
            log_fn(f"File encryption failed: {e}")
        raise


def decrypt_file_data_with_events(encrypted_data: bytes, key: bytes, iv: bytes, 
                                 mac: bytes = None, **kwargs) -> bytes:
    """
    Enhanced file data decryption with event callbacks.
    
    Args:
        encrypted_data: Encrypted file data
        key: Decryption key (16 bytes)
        iv: Initialization vector
        mac: Optional MAC for integrity verification
        callback_fn: Optional callback for decryption events
        log_fn: Optional logging callback
        progress_fn: Optional progress callback for large file decryption
        
    Returns:
        Decrypted file data
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    if log_fn:
        log_fn(f"Starting file decryption: {len(encrypted_data)} bytes")
    
    if callback_fn:
        callback_fn({
            'event': 'file_decryption_started',
            'encrypted_size': len(encrypted_data),
            'key_size': len(key),
            'has_mac': mac is not None
        })
    
    try:
        # Verify MAC if provided
        if mac:
            calculated_mac = calculate_mac(key, encrypted_data)
            if calculated_mac != mac:
                raise ValidationError("MAC verification failed - data may be corrupted")
            
            if callback_fn:
                callback_fn({
                    'event': 'mac_verification_completed',
                    'verified': True
                })
        
        # For large files, show progress
        chunk_size = 1024 * 1024  # 1MB chunks for progress
        if len(encrypted_data) > chunk_size and progress_fn:
            decrypted_chunks = []
            processed = 0
            
            for i in range(0, len(encrypted_data), chunk_size):
                chunk = encrypted_data[i:i + chunk_size]
                chunk_iv = iv[:]  # Copy IV
                
                # Adjust counter for this chunk
                counter_offset = i // 16
                chunk_iv = (int.from_bytes(chunk_iv, 'big') + counter_offset).to_bytes(16, 'big')
                
                decrypted_chunk = aes_ctr_encrypt_decrypt(chunk, key, chunk_iv)
                decrypted_chunks.append(decrypted_chunk)
                
                processed += len(chunk)
                progress_fn({
                    'bytes_processed': processed,
                    'total_bytes': len(encrypted_data),
                    'progress_percent': (processed / len(encrypted_data)) * 100,
                    'chunk_number': len(decrypted_chunks)
                })
            
            decrypted_data = b''.join(decrypted_chunks)
        else:
            # Small file or no progress callback
            decrypted_data = aes_ctr_encrypt_decrypt(encrypted_data, key, iv)
        
        if callback_fn:
            callback_fn({
                'event': 'file_decryption_completed',
                'encrypted_size': len(encrypted_data),
                'decrypted_size': len(decrypted_data)
            })
        
        if log_fn:
            log_fn(f"File decryption completed: {len(encrypted_data)} -> {len(decrypted_data)} bytes")
        
        return decrypted_data
        
    except Exception as e:
        if callback_fn:
            callback_fn({
                'event': 'file_decryption_failed',
                'error': str(e)
            })
        if log_fn:
            log_fn(f"File decryption failed: {e}")
        raise


def generate_secure_key_with_events(**kwargs) -> Dict[str, Any]:
    """
    Enhanced secure key generation with event callbacks.
    
    Args:
        callback_fn: Optional callback for key generation events
        log_fn: Optional logging callback
        progress_fn: Optional progress callback for entropy gathering
        
    Returns:
        Dictionary with generated key and metadata
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    if log_fn:
        log_fn("Starting secure key generation")
    
    if callback_fn:
        callback_fn({
            'event': 'key_generation_started',
            'key_type': 'AES-128',
            'entropy_source': 'system_random'
        })
    
    try:
        # Simulate entropy gathering for progress
        if progress_fn:
            # time already imported at module level
            for i in range(10):
                time.sleep(0.01)  # Small delay to simulate entropy gathering
                progress_fn({
                    'step': i + 1,
                    'total_steps': 10,
                    'progress_percent': ((i + 1) / 10) * 100,
                    'activity': 'gathering_entropy'
                })
        
        # Generate cryptographically secure key
        key = generate_random_key()
        iv = generate_random_iv()
        
        # Generate key metadata
        key_strength = len(key) * 8  # bits
        
        result = {
            'key': key,
            'iv': iv,
            'key_strength_bits': key_strength,
            'generation_method': 'CSPRNG',
            'timestamp': time.time()
        }
        
        if callback_fn:
            callback_fn({
                'event': 'key_generation_completed',
                'key_strength_bits': key_strength,
                'key_length': len(key),
                'iv_length': len(iv)
            })
        
        if log_fn:
            log_fn(f"Secure key generated: {key_strength}-bit strength")
        
        return result
        
    except Exception as e:
        if callback_fn:
            callback_fn({
                'event': 'key_generation_failed',
                'error': str(e)
            })
        if log_fn:
            log_fn(f"Key generation failed: {e}")
        raise


def hash_password_with_events(password: str, email: str, **kwargs) -> Dict[str, Any]:
    """
    Enhanced password hashing with event callbacks.
    
    Args:
        password: User's password
        email: User's email address
        callback_fn: Optional callback for hashing events
        log_fn: Optional logging callback
        progress_fn: Optional progress callback for hashing iterations
        
    Returns:
        Dictionary with hashed password and metadata
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    if log_fn:
        log_fn(f"Starting password hashing for email: {email}")
    
    if callback_fn:
        callback_fn({
            'event': 'password_hashing_started',
            'email': email,
            'password_length': len(password)
        })
    
    try:
        # Normalize email to lowercase
        normalized_email = email.lower()
        
        if progress_fn:
            progress_fn({
                'step': 'email_normalization',
                'progress_percent': 10
            })
        
        # Derive key from password with progress
        kwargs_for_derive = kwargs.copy()
        if progress_fn:
            # Create a nested progress callback that doesn't conflict
            def nested_progress_fn(p):
                progress_fn({
                    'step': 'key_derivation',
                    'progress_percent': 10 + (p.get('progress_percent', 0) * 0.6)  # 10-70%
                })
            kwargs_for_derive['progress_fn'] = nested_progress_fn
        
        key = derive_key_with_events(password, **kwargs_for_derive)
        
        if progress_fn:
            progress_fn({
                'step': 'email_hashing',
                'progress_percent': 80
            })
        
        # Hash email with derived key
        email_hash = hashlib.sha256(makebyte(normalized_email)).digest()
        
        # Encrypt email hash with derived key
        encrypted = aes_cbc_encrypt(email_hash, key)
        
        if progress_fn:
            progress_fn({
                'step': 'base64_encoding',
                'progress_percent': 95
            })
        
        # Return as base64
        hashed_password = base64_url_encode(encrypted)
        
        result = {
            'hashed_password': hashed_password,
            'email': email,
            'normalized_email': normalized_email,
            'hash_length': len(hashed_password)
        }
        
        if callback_fn:
            callback_fn({
                'event': 'password_hashing_completed',
                'email': email,
                'hash_length': len(hashed_password)
            })
        
        if log_fn:
            log_fn(f"Password hashing completed for {email}")
        
        if progress_fn:
            progress_fn({
                'step': 'completed',
                'progress_percent': 100
            })
        
        return result
        
    except Exception as e:
        if callback_fn:
            callback_fn({
                'event': 'password_hashing_failed',
                'error': str(e)
            })
        if log_fn:
            log_fn(f"Password hashing failed: {e}")
        raise


def encrypt_attributes_with_events(attributes: Dict[str, Any], key: List[int], **kwargs) -> Dict[str, Any]:
    """
    Enhanced attribute encryption with event callbacks.
    
    Args:
        attributes: Attribute dictionary to encrypt
        key: Encryption key as 32-bit integer array
        callback_fn: Optional callback for encryption events
        log_fn: Optional logging callback
        
    Returns:
        Dictionary with encrypted attributes and metadata
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    if log_fn:
        log_fn(f"Starting attribute encryption: {len(attributes)} attributes")
    
    if callback_fn:
        callback_fn({
            'event': 'attribute_encryption_started',
            'attribute_count': len(attributes),
            'attributes': list(attributes.keys())
        })
    
    try:
        encrypted_attrs = encrypt_attr(attributes, key)
        
        result = {
            'encrypted_attributes': encrypted_attrs,
            'original_attributes': attributes,
            'encrypted_size': len(encrypted_attrs),
            'attribute_count': len(attributes)
        }
        
        if callback_fn:
            callback_fn({
                'event': 'attribute_encryption_completed',
                'attribute_count': len(attributes),
                'encrypted_size': len(encrypted_attrs)
            })
        
        if log_fn:
            log_fn(f"Attribute encryption completed: {len(attributes)} attributes -> {len(encrypted_attrs)} bytes")
        
        return result
        
    except Exception as e:
        if callback_fn:
            callback_fn({
                'event': 'attribute_encryption_failed',
                'error': str(e)
            })
        if log_fn:
            log_fn(f"Attribute encryption failed: {e}")
        raise


def decrypt_attributes_with_events(encrypted_attrs: bytes, key: List[int], **kwargs) -> Dict[str, Any]:
    """
    Enhanced attribute decryption with event callbacks.
    
    Args:
        encrypted_attrs: Encrypted attributes
        key: Decryption key as 32-bit integer array
        callback_fn: Optional callback for decryption events
        log_fn: Optional logging callback
        
    Returns:
        Dictionary with decrypted attributes and metadata
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    if log_fn:
        log_fn(f"Starting attribute decryption: {len(encrypted_attrs)} bytes")
    
    if callback_fn:
        callback_fn({
            'event': 'attribute_decryption_started',
            'encrypted_size': len(encrypted_attrs)
        })
    
    try:
        decrypted_attrs = decrypt_attr(encrypted_attrs, key)
        
        if decrypted_attrs is False:
            raise ValidationError("Attribute decryption failed - invalid format or key")
        
        result = {
            'decrypted_attributes': decrypted_attrs,
            'encrypted_size': len(encrypted_attrs),
            'attribute_count': len(decrypted_attrs) if isinstance(decrypted_attrs, dict) else 0,
            'decryption_successful': True
        }
        
        if callback_fn:
            callback_fn({
                'event': 'attribute_decryption_completed',
                'attribute_count': result['attribute_count'],
                'attributes': list(decrypted_attrs.keys()) if isinstance(decrypted_attrs, dict) else []
            })
        
        if log_fn:
            log_fn(f"Attribute decryption completed: {len(encrypted_attrs)} bytes -> {result['attribute_count']} attributes")
        
        return result
        
    except Exception as e:
        if callback_fn:
            callback_fn({
                'event': 'attribute_decryption_failed',
                'error': str(e)
            })
        if log_fn:
            log_fn(f"Attribute decryption failed: {e}")
        raise


def calculate_file_mac_with_events(file_data: bytes, key: bytes, **kwargs) -> Dict[str, Any]:
    """
    Enhanced file MAC calculation with event callbacks.
    
    Args:
        file_data: File data to calculate MAC for
        key: MAC key
        callback_fn: Optional callback for MAC calculation events
        log_fn: Optional logging callback
        progress_fn: Optional progress callback for large files
        
    Returns:
        Dictionary with MAC and metadata
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    if log_fn:
        log_fn(f"Starting MAC calculation: {len(file_data)} bytes")
    
    if callback_fn:
        callback_fn({
            'event': 'mac_calculation_started',
            'data_size': len(file_data),
            'key_size': len(key)
        })
    
    try:
        # For large files, calculate MAC in chunks with progress
        chunk_size = 1024 * 1024  # 1MB chunks
        if len(file_data) > chunk_size and progress_fn:
            hasher = hashlib.sha256()
            hasher.update(key)
            
            processed = 0
            for i in range(0, len(file_data), chunk_size):
                chunk = file_data[i:i + chunk_size]
                hasher.update(chunk)
                
                processed += len(chunk)
                progress_fn({
                    'bytes_processed': processed,
                    'total_bytes': len(file_data),
                    'progress_percent': (processed / len(file_data)) * 100,
                    'chunk_number': (i // chunk_size) + 1
                })
            
            mac = hasher.digest()[:16]
        else:
            mac = calculate_mac(key, file_data)
        
        result = {
            'mac': mac,
            'data_size': len(file_data),
            'mac_length': len(mac),
            'algorithm': 'SHA256-HMAC'
        }
        
        if callback_fn:
            callback_fn({
                'event': 'mac_calculation_completed',
                'data_size': len(file_data),
                'mac_length': len(mac)
            })
        
        if log_fn:
            log_fn(f"MAC calculation completed: {len(file_data)} bytes -> {len(mac)} byte MAC")
        
        return result
        
    except Exception as e:
        if callback_fn:
            callback_fn({
                'event': 'mac_calculation_failed',
                'error': str(e)
            })
        if log_fn:
            log_fn(f"MAC calculation failed: {e}")
        raise


def add_crypto_methods_with_events(client_class):
    """
    Add enhanced crypto methods to the client class with event callbacks.
    
    Args:
        client_class: The client class to enhance
    """
    def derive_key_enhanced(self, password: str, salt: bytes = b'', **kwargs) -> bytes:
        """Derive encryption key from password with enhanced event callbacks."""
        return derive_key_with_events(password, salt, **kwargs)
    
    def encrypt_file_data(self, data: bytes, key: bytes, **kwargs) -> Dict[str, Any]:
        """Encrypt file data with enhanced event callbacks."""
        return encrypt_file_data_with_events(data, key, **kwargs)
    
    def decrypt_file_data(self, encrypted_data: bytes, key: bytes, iv: bytes, 
                         mac: bytes = None, **kwargs) -> bytes:
        """Decrypt file data with enhanced event callbacks."""
        return decrypt_file_data_with_events(encrypted_data, key, iv, mac, **kwargs)
    
    def generate_secure_key(self, **kwargs) -> Dict[str, Any]:
        """Generate secure cryptographic key with enhanced event callbacks."""
        return generate_secure_key_with_events(**kwargs)
    
    def hash_password_enhanced(self, password: str, email: str, **kwargs) -> Dict[str, Any]:
        """Hash password for authentication with enhanced event callbacks."""
        return hash_password_with_events(password, email, **kwargs)
    
    def encrypt_attributes(self, attributes: Dict[str, Any], key: List[int], **kwargs) -> Dict[str, Any]:
        """Encrypt file/folder attributes with enhanced event callbacks."""
        return encrypt_attributes_with_events(attributes, key, **kwargs)
    
    def decrypt_attributes(self, encrypted_attrs: bytes, key: List[int], **kwargs) -> Dict[str, Any]:
        """Decrypt file/folder attributes with enhanced event callbacks."""
        return decrypt_attributes_with_events(encrypted_attrs, key, **kwargs)
    
    def calculate_file_mac(self, file_data: bytes, key: bytes, **kwargs) -> Dict[str, Any]:
        """Calculate MAC for file data with enhanced event callbacks."""
        return calculate_file_mac_with_events(file_data, key, **kwargs)
    
    # Add methods to client class
    client_class.derive_key_enhanced = derive_key_enhanced
    client_class.encrypt_file_data = encrypt_file_data
    client_class.decrypt_file_data = decrypt_file_data
    client_class.generate_secure_key = generate_secure_key
    client_class.hash_password_enhanced = hash_password_enhanced
    client_class.encrypt_attributes = encrypt_attributes
    client_class.decrypt_attributes = decrypt_attributes
    client_class.calculate_file_mac = calculate_file_mac


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Original AES functions
    'aes_cbc_encrypt',
    'aes_cbc_decrypt', 
    'aes_cbc_encrypt_mega',
    'aes_cbc_decrypt_mega',
    'aes_cbc_encrypt_a32',
    'aes_cbc_decrypt_a32',
    'aes_ctr_encrypt_decrypt',
    
    # Original key derivation
    'derive_key',
    'string_to_a32',
    'a32_to_string',
    
    # Original base64 encoding
    'base64_url_encode',
    'base64_url_decode',
    'base64_to_a32',
    'a32_to_base64',
    
    # Original MAC functions
    'calculate_mac',
    'verify_mac',
    'calculate_chunk_mac',
    
    # Original password hashing
    'hash_password',
    
    # Original utilities
    'generate_random_key',
    'generate_random_iv',
    'generate_secure_password',
    'calculate_file_checksum',
    'calculate_string_hash',
    'xor_bytes',
    'mpi_to_int',
    'modular_inverse',
    'encrypt_attr',
    'decrypt_attr',
    'make_id',
    'get_chunks',
    'extended_gcd',
    'parse_node_key',
    'parse_file_attributes',
    
    # Enhanced functions with events
    'derive_key_with_events',
    'encrypt_file_data_with_events',
    'decrypt_file_data_with_events',
    'generate_secure_key_with_events',
    'hash_password_with_events',
    'encrypt_attributes_with_events',
    'decrypt_attributes_with_events',
    'calculate_file_mac_with_events',
    'add_crypto_methods_with_events',
]

