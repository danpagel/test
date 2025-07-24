"""
MegaPythonLibrary (MPL) - Modular Architecture
==============================================

A complete, secure, and professional Python client for MEGA.nz cloud storage 
with advanced features, comprehensive exception handling, real-time synchronization, 
and enterprise-ready capabilities.

This modular version splits the functionality across 8 specialized modules
while maintaining full API compatibility with the original merged implementation.

Version: 2.5.0 Professional Edition (Modular)
Author: MegaPythonLibrary Team
Date: January 2025

Quick Start:
    >>> from new_module import MPLClient
    >>> client = MPLClient()
    >>> client.login("your_email@example.com", "your_password")
    >>> client.upload("local_file.txt", "/")
    >>> files = client.list("/")
    >>> client.logout()

Enhanced Usage:
    >>> from new_module import create_enhanced_client
    >>> client = create_enhanced_client(
    ...     max_requests_per_second=10.0,
    ...     max_upload_speed=1024*1024,  # 1MB/s
    ... )
"""

# Import main client classes
from .client import MPLClient, create_client, create_enhanced_client

# Import utilities and exceptions
from .utils import (
    # Version info
    __version__, __author__, __license__,
    
    # Exception classes
    ValidationError, RequestError, MPLError, AuthenticationError,
    CryptoError, NetworkError, BusinessError, PaymentError, 
    FUSEError, LocalError,
    
    # Validation functions
    validate_email, validate_password,
    
    # Utility functions
    format_size, detect_file_type, is_image_file, 
    is_video_file, is_audio_file,
    
    # Crypto utilities
    aes_cbc_encrypt, aes_cbc_decrypt, derive_key, 
    generate_random_key, base64_url_encode, base64_url_decode,
    hash_password,
    
    # Other utilities
    get_version_info,
)

# Import authentication functions
from .auth import (
    login, logout, register, verify_email, change_password,
    is_logged_in, get_current_user, get_user_info, get_user_quota,
)

# Import storage functions
from .storage import (
    refresh_filesystem, list_folder, create_folder, delete_node,
    move_node, rename_node, upload_file, download_file,
    get_node_by_path, search_nodes_by_name, MegaNode
)

# Import sharing functions
from .sharing import (
    create_public_link, remove_public_link, parse_mega_url,
    is_valid_mega_url, get_share_type
)

# Import content functions
from .content import (
    detect_file_type, categorize_file, is_image_file,
    is_video_file, is_audio_file, is_document_file,
    get_file_info
)

# Define what gets imported with "from new_module import *"
__all__ = [
    # Version info
    '__version__',
    '__author__', 
    '__license__',
    
    # Main classes
    'MPLClient',
    'MegaNode',
    'create_client',
    'create_enhanced_client',
    
    # Core functions
    'login',
    'logout',
    'register',
    'verify_email',
    'change_password',
    'is_logged_in',
    'get_current_user',
    'get_user_info',
    'get_user_quota',
    
    # Filesystem functions
    'refresh_filesystem',
    'list_folder',
    'create_folder',
    'delete_node',
    'move_node',
    'rename_node',
    'upload_file',
    'download_file',
    'get_node_by_path',
    'search_nodes_by_name',
    
    # Utility functions
    'format_size',
    'detect_file_type',
    'is_image_file',
    'is_video_file',
    'is_audio_file',
    
    # Errors and validation
    'RequestError',
    'ValidationError',
    'validate_email',
    'validate_password',
    
    # Cryptographic utilities
    'aes_cbc_encrypt',
    'aes_cbc_decrypt',
    'derive_key',
    'generate_random_key',
    'base64_url_encode',
    'base64_url_decode',
    'hash_password',
    
    # Sharing functions
    'create_public_link',
    'remove_public_link',
    'parse_mega_url',
    'is_valid_mega_url',
    
    # Content functions
    'categorize_file',
    'is_document_file',
    'get_file_info',
    
    # Utilities
    'get_version_info',
]