"""
Client - Optimized MEGA.nz Python Client
========================================

This module provides the optimized MPLClient class that integrates all functionality
into a single, high-performance interface for interacting with Mega.nz.

Key optimizations implemented:
1. Streamlined imports - removed unused legacy functions
2. Consolidated search methods - eliminated redundant local imports  
3. Removed duplicate functionality (copy_folder merged with copy)
4. Clean module integration - optimized sync and crypto loading
5. Enhanced error handling and event callbacks
6. Professional API with comprehensive functionality

Features:
- Authentication with session management
- File/folder operations with progress tracking
- Advanced search with multiple filter types
- Public sharing with enhanced features
- Sync functionality with real-time monitoring  
- Cryptographic operations with event callbacks
- API enhancements (rate limiting, bandwidth throttling)
- Event-driven architecture throughout

Author: Modernized and optimized for July 2025
"""

# Core dependencies
from .dependencies import *
from .exceptions import RequestError, ValidationError
from typing import Optional, List, Dict, Any, Callable

# Enterprise Optimization Manager
try:
    from .optimization_manager import OptimizedDownloadManager, OptimizationMode
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Search functionality
from .search import (
    SearchFilter, SearchResult, SavedSearch,
    SizeOperator, DateOperator, SearchQueryBuilder,
    advanced_search_with_events, load_saved_search_with_events,
    list_saved_searches_with_events, delete_saved_search_with_events,
    get_search_statistics_with_events
)

# Authentication modules
from .auth import (
    load_user_session, get_user_info, get_user_quota, is_logged_in, 
    get_current_user, login_with_events, logout_with_events, 
    register_with_events, verify_email_with_events, change_password_with_events,
    add_authentication_methods
)

# Filesystem operations
from .filesystem import (
    list_folder, get_node_by_path, MegaNode, fs_tree, get_nodes,
    copy_file, copy_folder, refresh_filesystem_with_events, create_folder_with_events, 
    delete_node_with_events, move_node_with_events, rename_node_with_events, 
    upload_file_with_events, download_file_with_events, create_public_link_with_events, 
    remove_public_link_with_events, add_filesystem_methods_with_events
)

# Network and API enhancements
from .network import _api_session
from .api_enhancements import (
    enable_api_enhancements_with_events, disable_api_enhancements_with_events,
    get_api_enhancement_stats_with_events, create_async_client_with_events,
    configure_rate_limiting_with_events, configure_bandwidth_throttling_with_events
)

# Public sharing management
# (Methods injected via add_public_sharing_methods_with_events)

# Advanced transfer management
from .transfer_management import (
    TransferPriority, TransferState, TransferType, TransferSettings,
    add_upload_transfer, add_download_transfer, pause_transfer_with_events,
    resume_transfer_with_events, cancel_transfer_with_events,
    get_transfer_queue_status_with_events, get_transfer_statistics_with_events,
    list_transfers_with_events, retry_failed_transfers_with_events,
    configure_transfer_settings_with_events, get_transfer_manager, shutdown_transfer_manager
)

# Media & Thumbnails management
from .media_thumbnails import (
    MediaType, MediaProcessor, MegaMediaManager,
    create_thumbnail, create_preview
)

# Advanced module integrations
FILESYSTEM_AVAILABLE = True  # Always available as core functionality

# NOTE: Sync methods integration removed - functions were dependent on deleted AdvancedSynchronizer
SYNC_AVAILABLE = False

try:
    from .crypto import add_crypto_methods_with_events
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    from .search import add_search_methods_with_events
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False

try:
    from .public_sharing import add_public_sharing_methods_with_events
    PUBLIC_SHARING_AVAILABLE = True
except ImportError:
    PUBLIC_SHARING_AVAILABLE = False

try:
    from .transfer_management import add_transfer_management_methods_with_events
    TRANSFER_MANAGEMENT_AVAILABLE = True
except ImportError:
    TRANSFER_MANAGEMENT_AVAILABLE = False

try:
    from .media_thumbnails import add_media_thumbnails_methods_with_events
    MEDIA_THUMBNAILS_AVAILABLE = True
except ImportError:
    MEDIA_THUMBNAILS_AVAILABLE = False

try:
    from .api_enhancements import add_api_enhancements_methods_with_events
    API_ENHANCEMENTS_AVAILABLE = True
except ImportError:
    API_ENHANCEMENTS_AVAILABLE = False

try:
    from .events import add_event_methods, EventManager
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False

try:
    from .utilities import add_utilities_methods
    UTILITIES_AVAILABLE = True
except ImportError:
    UTILITIES_AVAILABLE = False

try:
    from .auth import add_authentication_methods
    AUTHENTICATION_AVAILABLE = True
except ImportError:
    AUTHENTICATION_AVAILABLE = False


# ==============================================
# === MAIN MEGA CLIENT CLASS ===
# ==============================================

class MPLClient:
    """
    Main Mega.nz client providing unified access to all functionality.
    
    This class integrates authentication, file operations, and user management
    into a single, easy-to-use interface following the reference implementation.
    
    Enhanced with advanced API features:
    - Rate limiting management
    - Bandwidth throttling
    - Connection pooling
    - Async/await support
    """
    
    def __init__(self, auto_login: bool = True, enable_api_enhancements: bool = False,
                 api_enhancement_config: Dict[str, Any] = None,
                 enable_optimizations: bool = True,
                 optimization_mode: str = "balanced",
                 optimization_config: Dict[str, Any] = None):
        """
        Initialize Mega client with enterprise optimization support.
        
        Args:
            auto_login: If True, attempt to restore saved session on startup
            enable_api_enhancements: Enable advanced API features
            api_enhancement_config: Configuration for API enhancements
            enable_optimizations: Enable enterprise optimization system
            optimization_mode: Optimization mode (conservative, balanced, aggressive, legacy_only, optimized_only)
            optimization_config: Configuration for optimization systems
        """
        self.auto_login = auto_login
        self._progress_callbacks = {}
        self._api_enhancements = None
        self._async_client = None
        
        # Initialize enterprise optimization system
        self._optimization_enabled = enable_optimizations and OPTIMIZATION_AVAILABLE
        
        # Configure logging first
        self.logger = logging.getLogger(__name__)
        
        if self._optimization_enabled:
            try:
                optimization_mode_enum = OptimizationMode(optimization_mode)
                self._optimization_manager = OptimizedDownloadManager(
                    optimization_mode=optimization_mode_enum,
                    fallback_enabled=True,
                    optimization_config=optimization_config or {}
                )
                # Set legacy download method for fallback
                self._optimization_manager.set_legacy_download_method(self._legacy_download)
                self.logger.info(f"Enterprise optimization system initialized in {optimization_mode} mode")
            except Exception as e:
                self.logger.error(f"Failed to initialize optimization system: {e}")
                self._optimization_enabled = False
                self._optimization_manager = None
        else:
            self._optimization_manager = None
            if not OPTIMIZATION_AVAILABLE:
                self.logger.warning("Optimization system not available")
        
        # Initialize event system
        if EVENTS_AVAILABLE:
            self._event_manager = EventManager()
        else:
            self._event_callbacks = {}
        
        # Initialize media manager
        self._media_manager = MegaMediaManager(self)
        
        # Additional logging configuration
        self.logger.info("MPLClient initialized")
        
        # Initialize API enhancements if requested
        if enable_api_enhancements:
            self.enable_api_enhancements(api_enhancement_config or {})
        
        # Attempt to restore session if requested
        if auto_login:
            try:
                if load_user_session():
                    self.logger.info(f"Session restored for {get_current_user()}")
                    self._refresh_filesystem_if_needed()
            except Exception as e:
                self.logger.warning(f"Failed to restore session: {e}")
    
    # ==============================================
    # === EVENT SYSTEM INTEGRATION ===
    # ==============================================
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Internal method to trigger events if event system is available."""
        if EVENTS_AVAILABLE and hasattr(self, '_event_manager'):
            try:
                self._event_manager.trigger(event_type, data)
            except Exception as e:
                self.logger.warning(f"Event trigger failed: {e}")
    
    # ==============================================
    # === CORE METHODS ===
    # ==============================================
    
    def list(self, path: str = "/") -> List[MegaNode]:
        """
        List contents of a folder.
        
        Args:
            path: Folder path to list (default: root)
            
        Returns:
            List of nodes in the folder
        """
        if not is_logged_in():
            raise RequestError("Not logged in")
        
        self._refresh_filesystem_if_needed()
        
        # Handle path-based lookup
        if path != "/":
            node = get_node_by_path(path)
            if not node:
                raise RequestError(f"Path not found: {path}")
            folder_handle = node.handle
        else:
            folder_handle = None
        
        return list_folder(folder_handle)

    # ==============================================
    # === UTILITY AND CONVENIENCE METHODS ===
    # ==============================================
    
    def close(self) -> None:
        """Close client and clean up resources."""
        if is_logged_in():
            logout_with_events(self._trigger_event)
            fs_tree.clear()
        
        _api_session.close()
        self.logger.info("MPLClient closed")

    # ==============================================
    # === ENTERPRISE OPTIMIZATION METHODS ===
    # ==============================================

    def _legacy_download(self, url: str, local_path: str, 
                        progress_callback: Optional[Callable] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Legacy download method for fallback purposes.
        
        This method uses the original filesystem download functionality.
        """
        try:
            # Import the legacy download function
            from .filesystem import download_file
            
            # Extract handle from URL if needed
            handle = kwargs.get('handle', url)
            
            # Use the original download_file function
            result = download_file(
                handle=handle,
                output_path=local_path,
                progress_callback=progress_callback,
                **kwargs
            )
            
            return {
                'success': True,
                'method': 'legacy',
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f"Legacy download failed: {e}")
            raise

    def download(self, remote_path: str, local_path: str,
                progress_callback: Optional[Callable] = None,
                priority: str = "NORMAL",
                **kwargs) -> Dict[str, Any]:
        """
        Enterprise download with optimization and smart fallback.
        
        Args:
            remote_path: Remote file path or handle
            local_path: Local path to save the file
            progress_callback: Optional progress callback function
            priority: Download priority (LOW, NORMAL, HIGH, URGENT)
            **kwargs: Additional download parameters
            
        Returns:
            Dictionary with download results and optimization metrics
        """
        if self._optimization_enabled and self._optimization_manager:
            # Use enterprise optimization system
            return self._optimization_manager.download(
                url=remote_path,
                local_path=local_path,
                progress_callback=progress_callback,
                priority=priority,
                **kwargs
            )
        else:
            # Fall back to legacy download
            return self._legacy_download(
                url=remote_path,
                local_path=local_path,
                progress_callback=progress_callback,
                **kwargs
            )

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get enterprise optimization system status."""
        if self._optimization_enabled and self._optimization_manager:
            return self._optimization_manager.get_status()
        else:
            return {
                'optimization_enabled': False,
                'reason': 'Optimization system not available or disabled'
            }

    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics."""
        if self._optimization_enabled and self._optimization_manager:
            metrics = self._optimization_manager.get_metrics()
            return {
                'success_rate': metrics.success_rate,
                'fallback_rate': metrics.fallback_rate,
                'total_attempts': metrics.optimization_attempts,
                'total_successes': metrics.optimization_successes,
                'total_fallbacks': metrics.fallback_uses,
                'average_speed': metrics.average_speed,
                'optimization_score': self._optimization_manager.get_optimization_score()
            }
        else:
            return {
                'optimization_enabled': False,
                'message': 'No metrics available - optimization system disabled'
            }

    def reset_optimization_metrics(self):
        """Reset optimization performance metrics."""
        if self._optimization_enabled and self._optimization_manager:
            self._optimization_manager.reset_metrics()
            self.logger.info("Optimization metrics reset")
        else:
            self.logger.warning("Cannot reset metrics - optimization system not available")

    def set_optimization_mode(self, mode: str):
        """
        Change optimization mode at runtime.
        
        Args:
            mode: Optimization mode (conservative, balanced, aggressive, legacy_only, optimized_only)
        """
        if self._optimization_enabled and self._optimization_manager:
            try:
                optimization_mode = OptimizationMode(mode)
                self._optimization_manager.optimization_mode = optimization_mode
                self.logger.info(f"Optimization mode changed to: {mode}")
            except ValueError:
                self.logger.error(f"Invalid optimization mode: {mode}")
                raise ValueError(f"Invalid optimization mode: {mode}")
        else:
            self.logger.warning("Cannot change optimization mode - system not available")

    # ==============================================
    # === ADVANCED SEARCH METHODS ===
    # ==============================================
    # Note: All search methods are now available via method injection from search module


# ==============================================
# === CONVENIENCE FUNCTIONS ===
# ==============================================

def create_client(auto_login: bool = True, enhanced: bool = False, 
                 enhancement_config: Dict[str, Any] = None) -> MPLClient:
    """
    Create a new Mega client instance.
    
    Args:
        auto_login: If True, attempt to restore saved session
        enhanced: Enable API enhancements (rate limiting, bandwidth throttling, etc.)
        enhancement_config: Configuration for API enhancements
        
    Returns:
        Configured MPLClient instance
    """
    return MPLClient(auto_login=auto_login, 
                    enable_api_enhancements=enhanced,
                    api_enhancement_config=enhancement_config)


def create_enterprise_client(auto_login: bool = True,
                           optimization_mode: str = "balanced",
                           optimization_config: Dict[str, Any] = None) -> MPLClient:
    """
    Create an enterprise MEGA client with full optimizations enabled.
    
    Args:
        auto_login: Whether to automatically attempt login on initialization
        optimization_mode: Optimization mode (conservative, balanced, aggressive, legacy_only, optimized_only)
        optimization_config: Configuration for optimization systems
        
    Returns:
        Configured MPLClient instance with enterprise optimizations
    """
    return MPLClient(
        auto_login=auto_login,
        enable_optimizations=True,
        optimization_mode=optimization_mode,
        optimization_config=optimization_config or {}
    )


def create_enhanced_client(auto_login: bool = True,
                          max_requests_per_second: float = 10.0,
                          max_upload_speed: int = None,
                          max_download_speed: int = None,
                          max_connections: int = 10) -> MPLClient:
    """
    Create a Mega client with API enhancements pre-configured.
    
    Args:
        auto_login: Whether to attempt automatic login
        max_requests_per_second: Rate limiting for API requests
        max_upload_speed: Upload speed limit in bytes/second (None = unlimited)
        max_download_speed: Download speed limit in bytes/second (None = unlimited)
        max_connections: Maximum concurrent connections
        
    Returns:
        Enhanced MPLClient instance
    """
    config = {
        'max_requests_per_second': max_requests_per_second,
        'max_upload_speed': max_upload_speed,
        'max_download_speed': max_download_speed,
        'max_connections': max_connections
    }
    
    return MPLClient(auto_login=auto_login,
                    enable_api_enhancements=True,
                    api_enhancement_config=config)


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    'MPLClient',
    'create_client',
    'create_enterprise_client',
    'create_enhanced_client',
]

# Configure logging
logger = logging.getLogger(__name__)

# Integrate advanced functionality
# NOTE: Sync functionality integration was removed due to AdvancedSynchronizer deletion
# The sync methods that were dependent on the removed class are no longer available

if CRYPTO_AVAILABLE:
    try:
        add_crypto_methods_with_events(MPLClient)
        logger.info("✅ Enhanced Crypto functionality integrated into MPLClient")
    except Exception as e:
        logger.warning(f"⚠️ Failed to integrate crypto functionality: {e}")

if SEARCH_AVAILABLE:
    try:
        add_search_methods_with_events(MPLClient)
        logger.info("✅ Advanced Search functionality integrated into MPLClient")
    except Exception as e:
        logger.warning(f"⚠️ Failed to integrate search functionality: {e}")

if PUBLIC_SHARING_AVAILABLE:
    try:
        add_public_sharing_methods_with_events(MPLClient)
        logger.info("✅ Advanced Public Sharing functionality integrated into MPLClient")
    except Exception as e:
        logger.warning(f"⚠️ Failed to integrate public sharing functionality: {e}")

# Add file versioning methods from filesystem module
# Always integrate core filesystem functionality
try:
    add_filesystem_methods_with_events(MPLClient)
    logger.info("✅ Core Filesystem functionality integrated into MPLClient")
except Exception as e:
    logger.warning(f"⚠️ Failed to integrate core filesystem functionality: {e}")

# Always integrate file versioning functionality
try:
    from .filesystem import add_file_versioning_methods_with_events
    add_file_versioning_methods_with_events(MPLClient)
    logger.info("✅ File Versioning functionality integrated into MPLClient")
except Exception as e:
    logger.warning(f"⚠️ Failed to integrate file versioning functionality: {e}")

if TRANSFER_MANAGEMENT_AVAILABLE:
    try:
        add_transfer_management_methods_with_events(MPLClient)
        logger.info("✅ Advanced Transfer Management functionality integrated into MPLClient")
    except Exception as e:
        logger.warning(f"⚠️ Failed to integrate transfer management functionality: {e}")

if MEDIA_THUMBNAILS_AVAILABLE:
    try:
        add_media_thumbnails_methods_with_events(MPLClient)
        logger.info("✅ Media & Thumbnails functionality integrated into MPLClient")
    except Exception as e:
        logger.warning(f"⚠️ Failed to integrate media thumbnails functionality: {e}")

if API_ENHANCEMENTS_AVAILABLE:
    try:
        add_api_enhancements_methods_with_events(MPLClient)
        logger.info("✅ API Enhancements functionality integrated into MPLClient")
    except Exception as e:
        logger.warning(f"⚠️ Failed to integrate API enhancements functionality: {e}")

# Integrate new modules
if EVENTS_AVAILABLE:
    try:
        add_event_methods(MPLClient)
        logger.info("✅ Event System functionality integrated into MPLClient")
    except Exception as e:
        logger.warning(f"⚠️ Failed to integrate event system functionality: {e}")

if UTILITIES_AVAILABLE:
    try:
        add_utilities_methods(MPLClient)
        logger.info("✅ Utilities functionality integrated into MPLClient")
    except Exception as e:
        logger.warning(f"⚠️ Failed to integrate utilities functionality: {e}")

if AUTHENTICATION_AVAILABLE:
    try:
        add_authentication_methods(MPLClient)
        logger.info("✅ Authentication functionality integrated into MPLClient")
    except Exception as e:
        logger.warning(f"⚠️ Failed to integrate authentication functionality: {e}")

# Configure basic logging if needed
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

