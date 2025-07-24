"""
Main Client module for MegaPythonLibrary.

This module contains:
- MPLClient class that orchestrates all functionality
- High-level API that integrates all modules
- Convenience functions for client creation
- Client lifecycle management
"""

import logging
from typing import Dict, List, Optional, Any, Callable

# Import all modules
from .utils import RequestError, ValidationError
from .monitor import get_logger, trigger_event
from .auth import (
    login, logout, register, verify_email, change_password,
    is_logged_in, get_current_user, get_user_info, get_user_quota,
    load_user_session, try_auto_login
)
from .storage import (
    refresh_filesystem, list_folder, create_folder, upload_file, download_file,
    delete_node, move_node, rename_node, get_node_by_path, search_nodes_by_name,
    get_nodes, fs_tree, MegaNode
)
from .network import RateLimiter, get_network_performance_stats, clear_network_cache, close_network_session
from .content import detect_file_type, is_image_file, is_video_file, is_audio_file
from .sharing import create_public_link, remove_public_link, parse_mega_url, get_sharing_stats
from .sync import get_transfer_queue_status, clear_completed_transfers


# ==============================================
# === MAIN MPL CLIENT CLASS ===
# ==============================================

class MPLClient:
    """
    Main Mega.nz client providing unified access to all functionality.
    
    This class integrates authentication, file operations, and user management
    into a single, easy-to-use interface.
    """
    
    def __init__(self, auto_login: bool = True):
        """
        Initialize Mega client.
        
        Args:
            auto_login: If True, attempt to restore saved session on startup
        """
        self.auto_login = auto_login
        self._event_callbacks = {}
        
        # Set up logging
        self.logger = get_logger("client")
        self.logger.info("MPLClient initialized")
        
        # Attempt to restore session if requested
        if auto_login:
            try:
                if try_auto_login():
                    self.logger.info(f"Session restored for {get_current_user()}")
                    self._refresh_filesystem_if_needed()
            except Exception as e:
                self.logger.warning(f"Failed to restore session: {e}")
    
    def _refresh_filesystem_if_needed(self) -> None:
        """Refresh filesystem if needed."""
        if fs_tree.needs_refresh() or not fs_tree.nodes:
            try:
                refresh_filesystem()
            except Exception as e:
                self.logger.warning(f"Failed to refresh filesystem: {e}")
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Internal method to trigger events."""
        try:
            trigger_event(event_type, data)
        except Exception as e:
            self.logger.warning(f"Event trigger failed: {e}")
    
    # ==============================================
    # === AUTHENTICATION METHODS ===
    # ==============================================
    
    def login(self, email: str, password: str, save_session: bool = True) -> bool:
        """
        Log in to Mega.
        
        Args:
            email: User email address
            password: User password
            save_session: Whether to save session for auto-login
            
        Returns:
            True if login successful
        """
        try:
            session = login(email, password, save_session)
            self._refresh_filesystem_if_needed()
            self.logger.info(f"Logged in as {email}")
            return True
        except Exception as e:
            self.logger.error(f"Login failed: {e}")
            return False
    
    def logout(self) -> bool:
        """
        Log out from Mega.
        
        Returns:
            True if logout successful
        """
        try:
            logout()
            fs_tree.clear()
            self.logger.info("Logged out successfully")
            return True
        except Exception as e:
            self.logger.error(f"Logout failed: {e}")
            return False
    
    def register(self, email: str, password: str, first_name: str = "", last_name: str = "") -> bool:
        """
        Register a new user account.
        
        Args:
            email: User email address
            password: User password
            first_name: User's first name
            last_name: User's last name
            
        Returns:
            True if registration successful
        """
        try:
            result = register(email, password, first_name, last_name)
            self.logger.info(f"Registration initiated for {email}")
            return result
        except Exception as e:
            self.logger.error(f"Registration failed: {e}")
            return False
    
    def verify_email(self, email: str, verification_code: str) -> bool:
        """
        Verify email address.
        
        Args:
            email: Email address
            verification_code: Verification code from email
            
        Returns:
            True if verification successful
        """
        try:
            result = verify_email(email, verification_code)
            self.logger.info(f"Email verification completed for {email}")
            return result
        except Exception as e:
            self.logger.error(f"Email verification failed: {e}")
            return False
    
    def change_password(self, old_password: str, new_password: str) -> bool:
        """
        Change user password.
        
        Args:
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password change successful
        """
        try:
            result = change_password(old_password, new_password)
            self.logger.info("Password changed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Password change failed: {e}")
            return False
    
    def is_logged_in(self) -> bool:
        """Check if user is logged in."""
        return is_logged_in()
    
    def get_current_user(self) -> Optional[str]:
        """Get current user email."""
        return get_current_user()
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get current user information."""
        try:
            return get_user_info()
        except Exception as e:
            self.logger.error(f"Failed to get user info: {e}")
            return {'error': str(e)}
    
    def get_user_quota(self) -> Dict[str, int]:
        """Get user storage quota information."""
        try:
            return get_user_quota()
        except Exception as e:
            self.logger.error(f"Failed to get user quota: {e}")
            return {}
    
    # ==============================================
    # === FILESYSTEM METHODS ===
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
        
        try:
            return list_folder(path)
        except Exception as e:
            self.logger.error(f"Failed to list folder {path}: {e}")
            raise
    
    def create_folder(self, name: str, parent_path: str = "/") -> MegaNode:
        """
        Create a new folder.
        
        Args:
            name: Name of the folder
            parent_path: Parent folder path
            
        Returns:
            Created folder node
        """
        try:
            result = create_folder(name, parent_path)
            self.logger.info(f"Created folder: {name} in {parent_path}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to create folder: {e}")
            raise
    
    def upload(self, local_path: str, remote_path: str = "/") -> MegaNode:
        """
        Upload a file.
        
        Args:
            local_path: Local file path
            remote_path: Remote folder path
            
        Returns:
            Uploaded file node
        """
        try:
            result = upload_file(local_path, remote_path)
            self.logger.info(f"Uploaded file: {local_path} -> {remote_path}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to upload file: {e}")
            raise
    
    def download(self, remote_path: str, local_path: str) -> bool:
        """
        Download a file.
        
        Args:
            remote_path: Remote file path or handle
            local_path: Local path to save file
            
        Returns:
            True if download successful
        """
        try:
            # Check if remote_path is a handle or path
            if remote_path.startswith('/') or remote_path == '':
                node = get_node_by_path(remote_path)
                if not node:
                    raise RequestError(f"File not found: {remote_path}")
                handle = node.handle
            else:
                handle = remote_path
            
            result = download_file(handle, local_path)
            self.logger.info(f"Downloaded file: {remote_path} -> {local_path}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to download file: {e}")
            raise
    
    def delete(self, path: str) -> bool:
        """
        Delete a file or folder.
        
        Args:
            path: Path to delete
            
        Returns:
            True if deletion successful
        """
        try:
            node = get_node_by_path(path)
            if not node:
                raise RequestError(f"Path not found: {path}")
            
            result = delete_node(node.handle)
            self.logger.info(f"Deleted: {path}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to delete: {e}")
            raise
    
    def move(self, source_path: str, destination_path: str) -> bool:
        """
        Move a file or folder.
        
        Args:
            source_path: Source path
            destination_path: Destination folder path
            
        Returns:
            True if move successful
        """
        try:
            source_node = get_node_by_path(source_path)
            dest_node = get_node_by_path(destination_path)
            
            if not source_node:
                raise RequestError(f"Source not found: {source_path}")
            if not dest_node:
                raise RequestError(f"Destination not found: {destination_path}")
            
            result = move_node(source_node.handle, dest_node.handle)
            self.logger.info(f"Moved: {source_path} -> {destination_path}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to move: {e}")
            raise
    
    def rename(self, path: str, new_name: str) -> bool:
        """
        Rename a file or folder.
        
        Args:
            path: Path to rename
            new_name: New name
            
        Returns:
            True if rename successful
        """
        try:
            node = get_node_by_path(path)
            if not node:
                raise RequestError(f"Path not found: {path}")
            
            result = rename_node(node.handle, new_name)
            self.logger.info(f"Renamed: {path} -> {new_name}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to rename: {e}")
            raise
    
    def search(self, pattern: str, folder_path: str = None) -> List[MegaNode]:
        """
        Search for files and folders by name.
        
        Args:
            pattern: Search pattern (supports wildcards)
            folder_path: Folder to search in (None for all)
            
        Returns:
            List of matching nodes
        """
        try:
            folder_handle = None
            if folder_path:
                folder_node = get_node_by_path(folder_path)
                if folder_node:
                    folder_handle = folder_node.handle
            
            results = search_nodes_by_name(pattern, folder_handle)
            self.logger.info(f"Search for '{pattern}' returned {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_node_info(self, path: str) -> Optional[MegaNode]:
        """
        Get information about a node.
        
        Args:
            path: Path to the node
            
        Returns:
            Node information or None if not found
        """
        try:
            return get_node_by_path(path)
        except Exception as e:
            self.logger.error(f"Failed to get node info: {e}")
            return None
    
    def refresh_filesystem(self) -> None:
        """Refresh filesystem data."""
        try:
            refresh_filesystem()
            self.logger.info("Filesystem refreshed")
        except Exception as e:
            self.logger.error(f"Failed to refresh filesystem: {e}")
            raise
    
    # ==============================================
    # === SHARING METHODS ===
    # ==============================================
    
    def create_public_link(self, path: str) -> str:
        """
        Create a public sharing link for a file or folder.
        
        Args:
            path: Path to the file or folder
            
        Returns:
            Public sharing URL
        """
        try:
            node = get_node_by_path(path)
            if not node:
                raise RequestError(f"Path not found: {path}")
            
            link = create_public_link(node.handle)
            self.logger.info(f"Created public link for: {path}")
            return link
        except Exception as e:
            self.logger.error(f"Failed to create public link: {e}")
            raise
    
    def remove_public_link(self, path: str) -> bool:
        """
        Remove public sharing link for a file or folder.
        
        Args:
            path: Path to the file or folder
            
        Returns:
            True if link was removed successfully
        """
        try:
            node = get_node_by_path(path)
            if not node:
                raise RequestError(f"Path not found: {path}")
            
            result = remove_public_link(node.handle)
            self.logger.info(f"Removed public link for: {path}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to remove public link: {e}")
            return False
    
    # ==============================================
    # === EVENT METHODS ===
    # ==============================================
    
    def on(self, event: str, callback: Callable) -> None:
        """
        Register an event callback.
        
        Args:
            event: Event name
            callback: Callback function
        """
        from .monitor import on_event
        on_event(event, callback)
        self.logger.debug(f"Registered event callback for: {event}")
    
    def off(self, event: str, callback: Callable = None) -> None:
        """
        Remove an event callback.
        
        Args:
            event: Event name
            callback: Callback function (None to remove all)
        """
        from .monitor import off_event
        off_event(event, callback)
        self.logger.debug(f"Removed event callback for: {event}")
    
    # ==============================================
    # === UTILITY METHODS ===
    # ==============================================
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get comprehensive storage information.
        
        Returns:
            Dictionary with storage details
        """
        if not is_logged_in():
            return {'error': 'Not logged in'}
        
        try:
            quota = get_user_quota()
            nodes = get_nodes()
            
            file_count = sum(1 for node in nodes.values() if node.is_file())
            folder_count = sum(1 for node in nodes.values() if node.is_folder())
            
            return {
                'quota': quota,
                'file_count': file_count,
                'folder_count': folder_count,
                'total_nodes': len(nodes),
            }
        except Exception as e:
            self.logger.error(f"Failed to get storage info: {e}")
            return {'error': str(e)}
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network performance statistics."""
        try:
            return get_network_performance_stats()
        except Exception as e:
            self.logger.error(f"Failed to get network stats: {e}")
            return {}
    
    def get_transfer_stats(self) -> Dict[str, Any]:
        """Get transfer queue statistics."""
        try:
            return get_transfer_queue_status()
        except Exception as e:
            self.logger.error(f"Failed to get transfer stats: {e}")
            return {}
    
    def get_sharing_stats(self) -> Dict[str, Any]:
        """Get sharing statistics."""
        try:
            return get_sharing_stats()
        except Exception as e:
            self.logger.error(f"Failed to get sharing stats: {e}")
            return {}
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        try:
            clear_network_cache()
            clear_completed_transfers()
            fs_tree.clear()
            self.logger.info("All caches cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def close(self) -> None:
        """Close client and clean up resources."""
        try:
            if is_logged_in():
                self.logout()
            
            close_network_session()
            self.logger.info("MPLClient closed")
        except Exception as e:
            self.logger.error(f"Error closing client: {e}")


# ==============================================
# === CONVENIENCE FUNCTIONS ===
# ==============================================

def create_client(auto_login: bool = True) -> MPLClient:
    """
    Create a new Mega client instance.
    
    Args:
        auto_login: If True, attempt to restore saved session
        
    Returns:
        Configured MPLClient instance
    """
    return MPLClient(auto_login=auto_login)


def create_enhanced_client(auto_login: bool = True,
                          max_requests_per_second: float = 10.0) -> MPLClient:
    """
    Create a Mega client with enhanced features.
    
    Args:
        auto_login: Whether to attempt automatic login
        max_requests_per_second: Rate limiting for API requests
        
    Returns:
        Enhanced MPLClient instance
    """
    from .network import set_rate_limit
    
    client = MPLClient(auto_login=auto_login)
    
    # Configure rate limiting
    set_rate_limit(max_requests_per_second)
    
    return client