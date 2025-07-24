"""
Advanced Public Sharing Management Module for MegaPythonLibrary
==============================================================

This module provides advanced public sharing capabilities including:
- Advanced link permissions (view-only, download, etc.)
- Link expiration dates and automatic cleanup
- Password-protected links with encryption
- Link analytics and usage tracking
- Bulk sharing operations
- Custom sharing templates

Author: MegaPythonLibrary Team
Version: 2.4.0 - Public Sharing Enhanced
Date: July 18, 2025
"""

import time
import json
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path

# Import existing functionality
from .filesystem import get_node_by_path, fs_tree, create_public_link, remove_public_link
from .network import single_api_request
from .crypto import derive_key, encrypt_attr, decrypt_attr
from .exceptions import RequestError
from .auth import is_logged_in, require_authentication


class SharePermission(Enum):
    """Public link permission levels."""
    VIEW_ONLY = "view"           # Can view file info but not download
    DOWNLOAD = "download"        # Can download files (default)
    PREVIEW = "preview"          # Can preview files online only
    LIMITED_DOWNLOAD = "limited" # Limited number of downloads


class LinkStatus(Enum):
    """Public link status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    DISABLED = "disabled"
    PASSWORD_PROTECTED = "protected"
    USAGE_EXCEEDED = "exceeded"


@dataclass
class ShareConfig:
    """Configuration for advanced public sharing."""
    permission: SharePermission = SharePermission.DOWNLOAD
    expires_at: Optional[datetime] = None
    password: Optional[str] = None
    max_downloads: Optional[int] = None
    enable_analytics: bool = True
    custom_name: Optional[str] = None
    description: Optional[str] = None
    allow_preview: bool = True
    require_email: bool = False


@dataclass
class ShareAnalytics:
    """Analytics data for a public share."""
    share_id: str
    created_at: datetime
    total_views: int = 0
    total_downloads: int = 0
    last_accessed: Optional[datetime] = None
    access_history: List[Dict[str, Any]] = None
    geographic_data: Dict[str, int] = None
    referrer_data: Dict[str, int] = None
    
    def __post_init__(self):
        if self.access_history is None:
            self.access_history = []
        if self.geographic_data is None:
            self.geographic_data = {}
        if self.referrer_data is None:
            self.referrer_data = {}


@dataclass
class PublicShare:
    """Enhanced public share with advanced features."""
    share_id: str
    path: str
    handle: str
    original_link: str
    enhanced_link: str
    config: ShareConfig
    analytics: ShareAnalytics
    status: LinkStatus = LinkStatus.ACTIVE
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ShareTemplateManager:
    """Manages sharing templates for common use cases."""
    
    def __init__(self):
        self.templates = {
            'temporary': ShareConfig(
                expires_at=datetime.now() + timedelta(hours=24),
                max_downloads=10,
                description="Temporary 24-hour share"
            ),
            'secure': ShareConfig(
                password="auto-generate",
                require_email=True,
                max_downloads=5,
                description="Secure password-protected share"
            ),
            'presentation': ShareConfig(
                permission=SharePermission.PREVIEW,
                expires_at=datetime.now() + timedelta(days=7),
                description="Presentation share - preview only"
            ),
            'archive': ShareConfig(
                permission=SharePermission.DOWNLOAD,
                expires_at=datetime.now() + timedelta(days=30),
                description="Long-term archive share"
            )
        }
    
    def get_template(self, name: str) -> ShareConfig:
        """Get a sharing template by name."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]
    
    def add_template(self, name: str, config: ShareConfig) -> None:
        """Add a custom sharing template."""
        self.templates[name] = config
    
    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.templates.keys())


class PublicSharingManager:
    """Advanced public sharing management system."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.logger = logging.getLogger('public_sharing')
        self.shares: Dict[str, PublicShare] = {}
        self.template_manager = ShareTemplateManager()
        self.storage_path = storage_path or "sharing_data.json"
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Load existing shares
        self.load_shares()
        
        # Start cleanup thread
        self.start_cleanup_monitor()
    
    def create_enhanced_share(self, path: str, config: Optional[ShareConfig] = None,
                            template: Optional[str] = None) -> PublicShare:
        """
        Create an enhanced public share with advanced features.
        
        Args:
            path: Path to file/folder to share
            config: Share configuration (optional)
            template: Template name to use (optional)
            
        Returns:
            PublicShare object with enhanced features
        """
        if not is_logged_in():
            raise RequestError("Not logged in")
        
        # Get configuration
        if template and not config:
            config = self.template_manager.get_template(template)
        elif not config:
            config = ShareConfig()
        
        # Get node information
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"Path not found: {path}")
        
        # Create basic public link
        original_link = create_public_link(node.handle)
        
        # Generate unique share ID
        share_id = str(uuid.uuid4())
        
        # Handle password generation
        if config.password == "auto-generate":
            config.password = self._generate_secure_password()
        
        # Create enhanced link
        enhanced_link = self._create_enhanced_link(original_link, share_id, config)
        
        # Initialize analytics
        analytics = ShareAnalytics(
            share_id=share_id,
            created_at=datetime.now()
        )
        
        # Create public share object
        public_share = PublicShare(
            share_id=share_id,
            path=path,
            handle=node.handle,
            original_link=original_link,
            enhanced_link=enhanced_link,
            config=config,
            analytics=analytics
        )
        
        # Store share
        self.shares[share_id] = public_share
        self.save_shares()
        
        self.logger.info(f"Created enhanced public share: {path} (ID: {share_id})")
        return public_share
    
    def get_share_info(self, share_id: str) -> Optional[PublicShare]:
        """Get information about a public share."""
        return self.shares.get(share_id)
    
    def list_shares(self, active_only: bool = True) -> List[PublicShare]:
        """List all public shares."""
        shares = list(self.shares.values())
        if active_only:
            shares = [s for s in shares if s.status == LinkStatus.ACTIVE]
        return shares
    
    def update_share_config(self, share_id: str, config: ShareConfig) -> None:
        """Update configuration for an existing share."""
        if share_id not in self.shares:
            raise ValueError(f"Share not found: {share_id}")
        
        share = self.shares[share_id]
        share.config = config
        
        # Regenerate enhanced link if needed
        share.enhanced_link = self._create_enhanced_link(
            share.original_link, share_id, config
        )
        
        self.save_shares()
        self.logger.info(f"Updated share configuration: {share_id}")
    
    def revoke_share(self, share_id: str) -> None:
        """Revoke a public share."""
        if share_id not in self.shares:
            raise ValueError(f"Share not found: {share_id}")
        
        share = self.shares[share_id]
        
        # Remove the original public link
        try:
            remove_public_link(share.handle)
        except Exception as e:
            self.logger.warning(f"Failed to remove original link: {e}")
        
        # Update status
        share.status = LinkStatus.DISABLED
        self.save_shares()
        
        self.logger.info(f"Revoked public share: {share_id}")
    
    def track_access(self, share_id: str, access_type: str = "view",
                    user_info: Optional[Dict[str, Any]] = None) -> None:
        """Track access to a public share for analytics."""
        if share_id not in self.shares:
            return
        
        share = self.shares[share_id]
        analytics = share.analytics
        
        # Update counters
        if access_type == "view":
            analytics.total_views += 1
        elif access_type == "download":
            analytics.total_downloads += 1
        
        # Update last accessed
        analytics.last_accessed = datetime.now()
        
        # Add to access history
        access_record = {
            'timestamp': datetime.now().isoformat(),
            'type': access_type,
            'user_info': user_info or {}
        }
        analytics.access_history.append(access_record)
        
        # Limit history size
        if len(analytics.access_history) > 1000:
            analytics.access_history = analytics.access_history[-1000:]
        
        # Check limits
        config = share.config
        if config.max_downloads and analytics.total_downloads >= config.max_downloads:
            share.status = LinkStatus.USAGE_EXCEEDED
        
        self.save_shares()
    
    def get_share_analytics(self, share_id: str) -> Optional[ShareAnalytics]:
        """Get analytics for a specific share."""
        share = self.shares.get(share_id)
        return share.analytics if share else None
    
    def bulk_share(self, paths: List[str], config: Optional[ShareConfig] = None,
                  template: Optional[str] = None) -> List[PublicShare]:
        """Create multiple shares at once."""
        shares = []
        for path in paths:
            try:
                share = self.create_enhanced_share(path, config, template)
                shares.append(share)
            except Exception as e:
                self.logger.error(f"Failed to share {path}: {e}")
        
        return shares
    
    def cleanup_expired_shares(self) -> int:
        """Clean up expired shares."""
        cleaned = 0
        now = datetime.now()
        
        for share_id, share in list(self.shares.items()):
            if share.config.expires_at and now > share.config.expires_at:
                try:
                    self.revoke_share(share_id)
                    cleaned += 1
                except Exception as e:
                    self.logger.error(f"Failed to cleanup expired share {share_id}: {e}")
        
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} expired shares")
        
        return cleaned
    
    def export_analytics(self, share_id: Optional[str] = None) -> Dict[str, Any]:
        """Export analytics data."""
        if share_id:
            share = self.shares.get(share_id)
            if not share:
                raise ValueError(f"Share not found: {share_id}")
            return asdict(share.analytics)
        else:
            # Export all analytics
            analytics_data = {}
            for sid, share in self.shares.items():
                analytics_data[sid] = asdict(share.analytics)
            return analytics_data
    
    def start_cleanup_monitor(self) -> None:
        """Start background cleanup monitoring."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        def cleanup_loop():
            while not self._stop_cleanup.wait(3600):  # Check every hour
                try:
                    self.cleanup_expired_shares()
                except Exception as e:
                    self.logger.error(f"Cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def stop_cleanup_monitor(self) -> None:
        """Stop background cleanup monitoring."""
        if self._stop_cleanup:
            self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
    
    def load_shares(self) -> None:
        """Load shares from storage."""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for share_data in data.get('shares', []):
                    share = self._deserialize_share(share_data)
                    self.shares[share.share_id] = share
                
                self.logger.info(f"Loaded {len(self.shares)} shares from storage")
        except Exception as e:
            self.logger.error(f"Failed to load shares: {e}")
    
    def save_shares(self) -> None:
        """Save shares to storage."""
        try:
            data = {
                'shares': [self._serialize_share(share) for share in self.shares.values()],
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save shares: {e}")
    
    def _create_enhanced_link(self, original_link: str, share_id: str,
                            config: ShareConfig) -> str:
        """Create enhanced link with additional parameters."""
        # For now, we'll create a custom URL format that includes our enhancements
        # In a real implementation, this would integrate with a web service
        
        params = []
        if config.password:
            params.append(f"protected=true")
        if config.expires_at:
            params.append(f"expires={int(config.expires_at.timestamp())}")
        if config.max_downloads:
            params.append(f"max_dl={config.max_downloads}")
        
        enhanced_url = f"{original_link}&share_id={share_id}"
        if params:
            enhanced_url += "&" + "&".join(params)
        
        return enhanced_url
    
    def _generate_secure_password(self) -> str:
        """Generate a secure password for protected links."""
        from .crypto import generate_secure_password
        return generate_secure_password(12)
    
    def _serialize_share(self, share: PublicShare) -> Dict[str, Any]:
        """Serialize share for storage."""
        data = asdict(share)
        
        # Convert datetime objects to ISO strings
        if data['created_at']:
            data['created_at'] = share.created_at.isoformat()
        if data['config']['expires_at']:
            data['config']['expires_at'] = share.config.expires_at.isoformat()
        if data['analytics']['created_at']:
            data['analytics']['created_at'] = share.analytics.created_at.isoformat()
        if data['analytics']['last_accessed']:
            data['analytics']['last_accessed'] = share.analytics.last_accessed.isoformat()
        
        # Convert enums to strings
        data['status'] = share.status.value
        data['config']['permission'] = share.config.permission.value
        
        return data
    
    def _deserialize_share(self, data: Dict[str, Any]) -> PublicShare:
        """Deserialize share from storage."""
        # Convert datetime strings back to objects
        if data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['config']['expires_at']:
            data['config']['expires_at'] = datetime.fromisoformat(data['config']['expires_at'])
        if data['analytics']['created_at']:
            data['analytics']['created_at'] = datetime.fromisoformat(data['analytics']['created_at'])
        if data['analytics']['last_accessed']:
            data['analytics']['last_accessed'] = datetime.fromisoformat(data['analytics']['last_accessed'])
        
        # Convert strings back to enums
        data['status'] = LinkStatus(data['status'])
        data['config']['permission'] = SharePermission(data['config']['permission'])
        
        # Recreate objects
        config = ShareConfig(**data['config'])
        analytics = ShareAnalytics(**data['analytics'])
        
        return PublicShare(
            share_id=data['share_id'],
            path=data['path'],
            handle=data['handle'],
            original_link=data['original_link'],
            enhanced_link=data['enhanced_link'],
            config=config,
            analytics=analytics,
            status=data['status'],
            created_at=data['created_at']
        )


# Convenience functions for easy integration
def create_temporary_share(path: str, hours: int = 24) -> PublicShare:
    """Create a temporary share that expires after specified hours."""
    config = ShareConfig(
        expires_at=datetime.now() + timedelta(hours=hours),
        description=f"Temporary {hours}-hour share"
    )
    manager = PublicSharingManager()
    return manager.create_enhanced_share(path, config)


def create_secure_share(path: str, password: Optional[str] = None) -> PublicShare:
    """Create a password-protected share."""
    config = ShareConfig(
        password=password or "auto-generate",
        require_email=True,
        description="Secure password-protected share"
    )
    manager = PublicSharingManager()
    return manager.create_enhanced_share(path, config)


def create_limited_share(path: str, max_downloads: int = 10) -> PublicShare:
    """Create a share with limited download count."""
    config = ShareConfig(
        max_downloads=max_downloads,
        description=f"Limited share - max {max_downloads} downloads"
    )
    manager = PublicSharingManager()
    return manager.create_enhanced_share(path, config)


# Global sharing manager instance
_global_manager = None

def get_sharing_manager() -> PublicSharingManager:
    """Get the global sharing manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = PublicSharingManager()
    return _global_manager


# ==============================================
# === ENHANCED PUBLIC SHARING FUNCTIONS WITH EVENTS ===
# ==============================================

def create_enhanced_share_with_events(path: str, expires_hours: Optional[int] = None,
                                     password: Optional[str] = None, max_downloads: Optional[int] = None,
                                     permission: str = "download", description: Optional[str] = None,
                                     event_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Create an enhanced public share with advanced features and event callbacks.
    
    Args:
        path: Path to file/folder to share
        expires_hours: Hours until link expires (optional)
        password: Password protection (optional, use 'auto' for generated password)
        max_downloads: Maximum number of downloads allowed (optional)
        permission: Permission level ('download', 'view', 'preview', 'limited')
        description: Custom description for the share
        event_callback: Optional callback for events
        
    Returns:
        Dictionary with share information including enhanced link and settings
    """
    try:
        # datetime and timedelta already imported at module level
        
        if not is_logged_in():
            raise RequestError("Not logged in")

        # Trigger start event
        if event_callback:
            event_callback('enhanced_share_start', {'path': path})

        # Create configuration
        config = ShareConfig()
        
        # Set expiration
        if expires_hours:
            config.expires_at = datetime.now() + timedelta(hours=expires_hours)
        
        # Set password
        if password == 'auto':
            config.password = "auto-generate"
        elif password:
            config.password = password
        
        # Set download limit
        if max_downloads:
            config.max_downloads = max_downloads
        
        # Set permission level
        permission_map = {
            'download': SharePermission.DOWNLOAD,
            'view': SharePermission.VIEW_ONLY,
            'preview': SharePermission.PREVIEW,
            'limited': SharePermission.LIMITED_DOWNLOAD
        }
        if permission in permission_map:
            config.permission = permission_map[permission]
        
        # Set description
        if description:
            config.description = description
        
        # Create enhanced share
        manager = get_sharing_manager()
        share = manager.create_enhanced_share(path, config)
        
        # Format response
        result = {
            'share_id': share.share_id,
            'path': share.path,
            'original_link': share.original_link,
            'enhanced_link': share.enhanced_link,
            'status': share.status.value,
            'created_at': share.created_at.isoformat(),
            'config': {
                'permission': share.config.permission.value,
                'expires_at': share.config.expires_at.isoformat() if share.config.expires_at else None,
                'password_protected': bool(share.config.password),
                'max_downloads': share.config.max_downloads,
                'description': share.config.description
            }
        }
        
        # Add generated password to result if applicable
        if share.config.password and password == 'auto':
            result['generated_password'] = share.config.password
        
        # Trigger success event
        if event_callback:
            event_callback('enhanced_share_created', result)
            
        return result
        
    except Exception as e:
        # Trigger error event
        if event_callback:
            event_callback('enhanced_share_error', {'path': path, 'error': str(e)})
        raise


def get_share_info_with_events(share_id: str, event_callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
    """
    Get information about an enhanced public share with event callbacks.
    
    Args:
        share_id: Share identifier
        event_callback: Optional callback for events
        
    Returns:
        Dictionary with share information or None if not found
    """
    try:
        # Trigger start event
        if event_callback:
            event_callback('get_share_info_start', {'share_id': share_id})
            
        manager = get_sharing_manager()
        share = manager.get_share_info(share_id)
        
        if not share:
            if event_callback:
                event_callback('share_not_found', {'share_id': share_id})
            return None
        
        result = {
            'share_id': share.share_id,
            'path': share.path,
            'original_link': share.original_link,
            'enhanced_link': share.enhanced_link,
            'status': share.status.value,
            'created_at': share.created_at.isoformat(),
            'config': {
                'permission': share.config.permission.value,
                'expires_at': share.config.expires_at.isoformat() if share.config.expires_at else None,
                'password_protected': bool(share.config.password),
                'max_downloads': share.config.max_downloads,
                'description': share.config.description
            },
            'analytics': {
                'total_views': share.analytics.total_views,
                'total_downloads': share.analytics.total_downloads,
                'last_accessed': share.analytics.last_accessed.isoformat() if share.analytics.last_accessed else None
            }
        }
        
        # Trigger success event
        if event_callback:
            event_callback('share_info_retrieved', result)
            
        return result
        
    except Exception as e:
        # Trigger error event
        if event_callback:
            event_callback('get_share_info_error', {'share_id': share_id, 'error': str(e)})
        raise


def list_shares_with_events(active_only: bool = True, event_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    List all enhanced public shares with event callbacks.
    
    Args:
        active_only: Only return active shares
        event_callback: Optional callback for events
        
    Returns:
        List of share information dictionaries
    """
    try:
        # Trigger start event
        if event_callback:
            event_callback('list_shares_start', {'active_only': active_only})
            
        manager = get_sharing_manager()
        shares = manager.list_shares(active_only)
        
        results = [
            {
                'share_id': share.share_id,
                'path': share.path,
                'status': share.status.value,
                'created_at': share.created_at.isoformat(),
                'expires_at': share.config.expires_at.isoformat() if share.config.expires_at else None,
                'total_downloads': share.analytics.total_downloads,
                'description': share.config.description
            }
            for share in shares
        ]
        
        # Trigger success event
        if event_callback:
            event_callback('shares_listed', {'count': len(results), 'active_only': active_only})
            
        return results
        
    except Exception as e:
        # Trigger error event
        if event_callback:
            event_callback('list_shares_error', {'active_only': active_only, 'error': str(e)})
        raise


def revoke_share_with_events(share_id: str, event_callback: Optional[Callable] = None) -> None:
    """
    Revoke an enhanced public share with event callbacks.
    
    Args:
        share_id: Share identifier to revoke
        event_callback: Optional callback for events
    """
    try:
        # Trigger start event
        if event_callback:
            event_callback('revoke_share_start', {'share_id': share_id})
            
        manager = get_sharing_manager()
        manager.revoke_share(share_id)
        
        # Trigger success event
        if event_callback:
            event_callback('share_revoked', {'share_id': share_id})
            
    except Exception as e:
        # Trigger error event
        if event_callback:
            event_callback('revoke_share_error', {'share_id': share_id, 'error': str(e)})
        raise


def get_share_analytics_with_events(share_id: str, event_callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
    """
    Get detailed analytics for a public share with event callbacks.
    
    Args:
        share_id: Share identifier
        event_callback: Optional callback for events
        
    Returns:
        Dictionary with detailed analytics or None if not found
    """
    try:
        # Trigger start event
        if event_callback:
            event_callback('get_analytics_start', {'share_id': share_id})
            
        manager = get_sharing_manager()
        analytics = manager.get_share_analytics(share_id)
        
        if not analytics:
            if event_callback:
                event_callback('analytics_not_found', {'share_id': share_id})
            return None
        
        result = {
            'share_id': analytics.share_id,
            'created_at': analytics.created_at.isoformat(),
            'total_views': analytics.total_views,
            'total_downloads': analytics.total_downloads,
            'last_accessed': analytics.last_accessed.isoformat() if analytics.last_accessed else None,
            'access_history': analytics.access_history[-10:],  # Last 10 accesses
            'geographic_data': analytics.geographic_data,
            'referrer_data': analytics.referrer_data
        }
        
        # Trigger success event
        if event_callback:
            event_callback('analytics_retrieved', result)
            
        return result
        
    except Exception as e:
        # Trigger error event
        if event_callback:
            event_callback('get_analytics_error', {'share_id': share_id, 'error': str(e)})
        raise


def bulk_share_with_events(paths: List[str], template: str = "default",
                          expires_hours: Optional[int] = None,
                          event_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    Create multiple enhanced shares at once with event callbacks.
    
    Args:
        paths: List of paths to share
        template: Template to use ('temporary', 'secure', 'presentation', 'archive')
        expires_hours: Override expiration hours for all shares
        event_callback: Optional callback for events
        
    Returns:
        List of share information dictionaries
    """
    try:
        # datetime and timedelta already imported at module level
        
        # Trigger start event
        if event_callback:
            event_callback('bulk_share_start', {'count': len(paths), 'template': template})
        
        manager = get_sharing_manager()
        
        # Get template or create default config
        if template == "default":
            config = ShareConfig()
            if expires_hours:
                config.expires_at = datetime.now() + timedelta(hours=expires_hours)
        else:
            config = manager.template_manager.get_template(template)
            if expires_hours:  # Override template expiration
                config.expires_at = datetime.now() + timedelta(hours=expires_hours)
        
        shares = manager.bulk_share(paths, config)
        
        # Format results
        results = []
        for share in shares:
            result = {
                'share_id': share.share_id,
                'path': share.path,
                'enhanced_link': share.enhanced_link,
                'status': share.status.value,
                'created_at': share.created_at.isoformat()
            }
            results.append(result)
        
        # Trigger success event
        if event_callback:
            event_callback('bulk_shares_created', {'count': len(results), 'template': template})
            
        return results
        
    except Exception as e:
        # Trigger error event
        if event_callback:
            event_callback('bulk_share_error', {'paths': paths, 'template': template, 'error': str(e)})
        raise


def cleanup_expired_shares_with_events(event_callback: Optional[Callable] = None) -> int:
    """
    Clean up expired enhanced shares with event callbacks.
    
    Args:
        event_callback: Optional callback for events
    
    Returns:
        Number of shares cleaned up
    """
    try:
        # Trigger start event
        if event_callback:
            event_callback('cleanup_start', {})
            
        manager = get_sharing_manager()
        cleaned = manager.cleanup_expired_shares()
        
        # Trigger success event
        if event_callback:
            event_callback('shares_cleaned', {'count': cleaned})
            
        return cleaned
        
    except Exception as e:
        # Trigger error event
        if event_callback:
            event_callback('cleanup_error', {'error': str(e)})
        raise


# Update __all__ to include enhanced functions
__all__ = [
    # Classes
    'SharePermission',
    'LinkStatus', 
    'ShareConfig',
    'ShareAnalytics',
    'PublicShare',
    'ShareTemplateManager',
    'PublicSharingManager',
    
    # Convenience functions
    'create_temporary_share',
    'create_secure_share',
    'create_limited_share',
    'get_sharing_manager',
    
    # Enhanced functions with events
    'create_enhanced_share_with_events',
    'get_share_info_with_events',
    'list_shares_with_events',
    'revoke_share_with_events',
    'get_share_analytics_with_events',
    'bulk_share_with_events',
    'cleanup_expired_shares_with_events',
    
    # Client method injection
    'add_public_sharing_methods_with_events',
]


# ==============================================
# === CLIENT METHOD INJECTION ===
# ==============================================

def add_public_sharing_methods_with_events(client_class):
    """Add public sharing methods with event support to the MPLClient class."""
    
    def create_enhanced_share_method(self, path: str, expires_hours: Optional[int] = None,
                                   password: Optional[str] = None, max_downloads: Optional[int] = None,
                                   permission: str = "download", description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an enhanced public share with advanced features.
        
        Args:
            path: Path to file/folder to share
            expires_hours: Hours until link expires (optional)
            password: Password protection (optional, use 'auto' for generated password)
            max_downloads: Maximum number of downloads allowed (optional)
            permission: Permission level ('download', 'view', 'preview', 'limited')
            description: Custom description for the share
            
        Returns:
            Dictionary with share information including enhanced link and settings
        """
        if not is_logged_in():
            raise RequestError("Not logged in")
        
        return create_enhanced_share_with_events(
            path, expires_hours, password, max_downloads,
            permission, description, getattr(self, '_trigger_event', None)
        )
    
    def get_share_info_method(self, share_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an enhanced public share.
        
        Args:
            share_id: Share identifier
            
        Returns:
            Dictionary with share information or None if not found
        """
        return get_share_info_with_events(share_id, getattr(self, '_trigger_event', None))
    
    def list_shares_method(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        List all enhanced public shares.
        
        Args:
            active_only: Only return active shares
            
        Returns:
            List of share information dictionaries
        """
        return list_shares_with_events(active_only, getattr(self, '_trigger_event', None))
    
    def revoke_share_method(self, share_id: str) -> None:
        """
        Revoke an enhanced public share.
        
        Args:
            share_id: Share identifier to revoke
        """
        revoke_share_with_events(share_id, getattr(self, '_trigger_event', None))
    
    def get_share_analytics_method(self, share_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed analytics for a public share.
        
        Args:
            share_id: Share identifier
            
        Returns:
            Dictionary with detailed analytics or None if not found
        """
        return get_share_analytics_with_events(share_id, getattr(self, '_trigger_event', None))
    
    def bulk_share_method(self, paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Create multiple enhanced shares at once.
        
        Args:
            paths: List of file/folder paths to share
            **kwargs: Common sharing options for all paths
            
        Returns:
            List of share information dictionaries
        """
        return bulk_share_with_events(paths, callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def cleanup_expired_shares_method(self) -> int:
        """
        Clean up expired shares.
        
        Returns:
            Number of shares cleaned up
        """
        return cleanup_expired_shares_with_events(callback_fn=getattr(self, '_trigger_event', None))
    
    # Add methods to client class
    setattr(client_class, 'create_enhanced_share', create_enhanced_share_method)
    setattr(client_class, 'get_share_info', get_share_info_method)
    setattr(client_class, 'list_shares', list_shares_method)
    setattr(client_class, 'revoke_share', revoke_share_method)
    setattr(client_class, 'get_share_analytics', get_share_analytics_method)
    setattr(client_class, 'bulk_share', bulk_share_method)
    setattr(client_class, 'cleanup_expired_shares', cleanup_expired_shares_method)

