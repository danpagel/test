"""
MegaPythonLibrary (MPL) - Professional MEGA.nz Python Client
==========================================================

A complete, secure, and professional Python client for MEGA.nz cloud storage 
with advanced features, comprehensive exception handling, real-time synchronization, 
and enterprise-ready capabilities.

Version: 2.5.0 Professional Edition
Author: MegaPythonLibrary Team
Date: January 2025

Features:
- 🔐 Complete authentication system with session management
- 📁 Full filesystem operations (upload, download, move, copy, delete)
- 🔍 Advanced search with filters, regex, and saved queries  
- 🔗 Enhanced public sharing with parameters, analytics, and bulk operations
- 🔄 Real-time bidirectional synchronization with conflict resolution
- 🚀 Advanced transfer management with queue control and monitoring
- 🖼️ Media processing with thumbnail and preview generation
- ⚡ API enhancements with rate limiting and bandwidth throttling
- �️ Professional exception handling with 50+ official MEGA error codes
- � Real-time event system with progress tracking and monitoring
- 🌍 Cross-platform compatibility with encoding support

Quick Start:
    >>> from mpl import MPLClient
    >>> client = MPLClient()
    >>> client.login("your_email@example.com", "your_password")
    >>> client.upload("local_file.txt", "/")
    >>> files = client.list("/")
    >>> client.logout()

Enhanced Usage:
    >>> from mpl import create_enhanced_client
    >>> client = create_enhanced_client(
    ...     max_requests_per_second=10.0,
    ...     max_upload_speed=1024*1024,  # 1MB/s
    ... )

Package Structure:
- client: Main MPLClient class with comprehensive functionality
- auth: Authentication and session management
- filesystem: File and folder operations with versioning support
- network: HTTP session and request handling
- sync: Real-time synchronization system with monitoring
- search: Advanced search engine with filters and analytics
- public_sharing: Enhanced sharing with analytics and parameters
- transfer_management: Advanced transfer queue management
- media_thumbnails: Professional media processing
- api_enhancements: Rate limiting and bandwidth control
- events: Real-time event system with callbacks
- utilities: Helper functions and utilities
- crypto: Cryptographic utilities and security
- exceptions: Professional exception handling with MEGA error codes
- dependencies: Common imports and utilities
"""

# Version information
__version__ = "2.5.0"
__author__ = "MegaPythonLibrary Team"
__email__ = "contact@megapythonlibrary.dev"
__license__ = "MIT"
__status__ = "Production"

# Main client imports
from .client import MPLClient, create_client, create_enhanced_client

# Core functionality imports
from .auth import (
    login, logout, register, verify_email, change_password,
    is_logged_in, get_current_user, get_user_info, get_user_quota,
    login_with_events, logout_with_events, register_with_events,
    verify_email_with_events, change_password_with_events
)

from .filesystem import (
    MegaNode, refresh_filesystem, list_folder, create_folder,
    delete_node, move_node, rename_node, upload_file, download_file,
    create_public_link, remove_public_link, copy_file, copy_folder,
    refresh_filesystem_with_events, create_folder_with_events, delete_node_with_events,
    move_node_with_events, rename_node_with_events, upload_file_with_events,
    download_file_with_events, create_public_link_with_events, remove_public_link_with_events
)

from .api_enhancements import (
    enable_api_enhancements_with_events, disable_api_enhancements_with_events,
    get_api_enhancement_stats_with_events, create_async_client_with_events,
    configure_rate_limiting_with_events, configure_bandwidth_throttling_with_events,
    make_request_with_events, api_request_with_events
)

from .public_sharing import (
    create_enhanced_share_with_events, get_share_info_with_events,
    list_shares_with_events, revoke_share_with_events,
    get_share_analytics_with_events, bulk_share_with_events,
    cleanup_expired_shares_with_events, get_sharing_manager
)

from .exceptions import RequestError, ValidationError, validate_email, validate_password

# Cryptographic utilities imports
from .crypto import (
    # Basic crypto functions
    aes_cbc_encrypt, aes_cbc_decrypt, derive_key, generate_random_key,
    base64_url_encode, base64_url_decode, hash_password,
    # Enhanced crypto functions with events
    derive_key_with_events, encrypt_file_data_with_events,
    decrypt_file_data_with_events, generate_secure_key_with_events,
    hash_password_with_events, encrypt_attributes_with_events,
    decrypt_attributes_with_events, calculate_file_mac_with_events,
    add_crypto_methods_with_events
)

# Advanced features imports
try:
    from .search import (
        AdvancedSearchEngine, SearchFilter, SearchResult, SavedSearch,
        SizeOperator, DateOperator, SearchQueryBuilder,
        advanced_search, search_by_size, search_by_type, search_by_extension,
        search_with_regex, search_images, search_documents, search_videos,
        search_audio, create_search_query, save_search, load_saved_search,
        list_saved_searches, delete_saved_search, get_search_statistics,
        search_recent_files,
        # Enhanced search functions with events
        advanced_search_with_events, search_by_size_with_events, 
        search_by_type_with_events, search_by_extension_with_events,
        search_with_regex_with_events, search_images_with_events, 
        search_documents_with_events, search_videos_with_events,
        search_audio_with_events, create_search_query_with_events,
        save_search_with_events, load_saved_search_with_events,
        list_saved_searches_with_events, delete_saved_search_with_events,
        get_search_statistics_with_events
    )
    ADVANCED_SEARCH_AVAILABLE = True
except ImportError:
    ADVANCED_SEARCH_AVAILABLE = False

# Note: Sync module available but not imported at package level to avoid circular imports
SYNC_AVAILABLE = True

# Try to import sync enhanced functions (but expect failure due to circular imports)
try:
    from .sync import (
        create_sync_config_with_events,
        stop_real_time_sync_with_events,
        get_sync_status_with_events,
        list_sync_instances_with_events,
        cleanup_sync_databases_with_events
    )
    SYNC_ENHANCED_AVAILABLE = True
except ImportError:
    SYNC_ENHANCED_AVAILABLE = False

try:
    from .api_enhancements import (
        create_enhanced_session, EnhancedSession, RateLimiter,
        BandwidthThrottler, AsyncAPIClient
    )
    API_ENHANCEMENTS_AVAILABLE = True
except ImportError:
    API_ENHANCEMENTS_AVAILABLE = False

# Package-level convenience functions
def get_version_info():
    """Get detailed version information."""
    return {
        'version': __version__,
        'author': __author__,
        'status': __status__,
        'features': {
            'advanced_search': ADVANCED_SEARCH_AVAILABLE,
            'sync': SYNC_AVAILABLE,
            'api_enhancements': API_ENHANCEMENTS_AVAILABLE,
        }
    }

def create_client_with_features(**kwargs):
    """
    Create a client with all available features enabled.
    
    Args:
        **kwargs: Configuration options for client creation
        
    Returns:
        MPLClient with all available features enabled
    """
    # Enable API enhancements if available
    if API_ENHANCEMENTS_AVAILABLE and 'enable_api_enhancements' not in kwargs:
        kwargs['enable_api_enhancements'] = True
    
    return MPLClient(**kwargs)

# Define what gets imported with "from mpl import *"
__all__ = [
    # Version info
    '__version__',
    '__author__', 
    '__license__',
    
    # Main classes
    'MPLClient',
    'create_client',
    'create_enhanced_client', 
    'create_client_with_features',
    
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
    
    # Enhanced authentication with events
    'login_with_events',
    'logout_with_events',
    'register_with_events', 
    'verify_email_with_events',
    'change_password_with_events',
    'get_user_info',
    'get_user_quota',
    
    # Filesystem
    'MegaNode',
    'refresh_filesystem',
    'list_folder',
    'create_folder',
    'delete_node',
    'move_node',
    'rename_node',
    'upload_file',
    'download_file',
    'create_public_link',
    'remove_public_link',
    'copy_file',
    'copy_folder',
    
    # Enhanced filesystem with events
    'refresh_filesystem_with_events',
    'create_folder_with_events',
    'delete_node_with_events',
    'move_node_with_events',
    'rename_node_with_events',
    'upload_file_with_events',
    'download_file_with_events',
    'create_public_link_with_events',
    'remove_public_link_with_events',
    
    # Network and API Enhancement functions
    'enable_api_enhancements_with_events',
    'disable_api_enhancements_with_events',
    'get_api_enhancement_stats_with_events',
    'create_async_client_with_events',
    'configure_rate_limiting_with_events',
    'configure_bandwidth_throttling_with_events',
    'make_request_with_events',
    'api_request_with_events',
    
    # Public Sharing functions
    'create_enhanced_share_with_events',
    'get_share_info_with_events',
    'list_shares_with_events',
    'revoke_share_with_events',
    'get_share_analytics_with_events',
    'bulk_share_with_events',
    'cleanup_expired_shares_with_events',
    'get_sharing_manager',
    
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
    
    # Enhanced crypto functions with events
    'derive_key_with_events',
    'encrypt_file_data_with_events',
    'decrypt_file_data_with_events',
    'generate_secure_key_with_events',
    'hash_password_with_events',
    'encrypt_attributes_with_events',
    'decrypt_attributes_with_events',
    'calculate_file_mac_with_events',
    'add_crypto_methods_with_events',
    
    # Utilities
    'get_version_info',
]

# Add advanced features to __all__ if available
if ADVANCED_SEARCH_AVAILABLE:
    __all__.extend([
        'AdvancedSearchEngine',
        'SearchFilter', 
        'SearchResult',
        'SavedSearch',
        'SizeOperator',
        'DateOperator',
        'SearchQueryBuilder',
        'advanced_search',
        'search_by_size',
        'search_by_type', 
        'search_by_extension',
        'search_with_regex',
        'search_images',
        'search_documents',
        'search_videos',
        'search_audio',
        'create_search_query',
        'save_search',
        'load_saved_search',
        'list_saved_searches',
        'delete_saved_search',
        'get_search_statistics',
        'search_recent_files',
        # Enhanced search functions with events
        'advanced_search_with_events',
        'search_by_size_with_events', 
        'search_by_type_with_events', 
        'search_by_extension_with_events',
        'search_with_regex_with_events',
        'search_images_with_events', 
        'search_documents_with_events',
        'search_videos_with_events',
        'search_audio_with_events',
        'create_search_query_with_events',
        'save_search_with_events',
        'load_saved_search_with_events',
        'list_saved_searches_with_events',
        'delete_saved_search_with_events',
        'get_search_statistics_with_events'
    ])

if SYNC_AVAILABLE:
    __all__.extend([
        'SyncManager',
        'SyncDatabase',
        # Enhanced sync functions (available through client) - some removed due to AdvancedSynchronizer removal
        'create_sync_config_with_events',
        'stop_real_time_sync_with_events',
        'get_sync_status_with_events',
        'list_sync_instances_with_events',
        'cleanup_sync_databases_with_events'
    ])

if API_ENHANCEMENTS_AVAILABLE:
    __all__.extend([
        'create_enhanced_session',
        'EnhancedSession',
        'RateLimiter',
        'BandwidthThrottler',
        'AsyncAPIClient'
    ])

# Package initialization
import logging

# Set up package-level logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Prevent "No handlers" warnings

# Optional: Set up basic logging configuration if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info(f"MegaPythonLibrary v{__version__} initialized")

# Feature availability logging
features_status = []
if ADVANCED_SEARCH_AVAILABLE:
    features_status.append("✅ Advanced Search")
if SYNC_AVAILABLE:
    features_status.append("✅ Synchronization")  
if API_ENHANCEMENTS_AVAILABLE:
    features_status.append("✅ API Enhancements")

if features_status:
    logger.info(f"Available features: {', '.join(features_status)}")
