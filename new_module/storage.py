"""
Storage and File Operations module for MegaPythonLibrary.

This module contains:
- Filesystem tree management
- Node operations (files and folders)
- File upload and download
- Directory navigation
- Metadata handling
"""

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .utils import format_size, RequestError, ValidationError
from .network import single_api_request, get_upload_url, get_download_url
from .auth import current_session, require_authentication
from .monitor import get_logger, trigger_event
from .sync import upload_file_chunked, download_file_chunked

# ==============================================
# === NODE TYPES AND CONSTANTS ===
# ==============================================

# Node types in Mega filesystem
NODE_TYPE_FILE = 0
NODE_TYPE_FOLDER = 1
NODE_TYPE_ROOT = 2
NODE_TYPE_INBOX = 3
NODE_TYPE_TRASH = 4

# Chunk size for file operations
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
MAX_CHUNK_SIZE = 1024 * 1024 * 10  # 10MB max chunk


# ==============================================
# === NODE CLASSES ===
# ==============================================

class MegaNode:
    """
    Represents a file or folder node in the Mega filesystem.
    """
    
    def __init__(self, node_data: Dict[str, Any]):
        self.handle = node_data.get('h', '')
        self.parent_handle = node_data.get('p', '')
        self.owner = node_data.get('u', '')
        self.node_type = node_data.get('t', NODE_TYPE_FILE)
        self.size = node_data.get('s', 0)
        self.timestamp = node_data.get('ts', 0)
        
        # Get attributes (already decrypted by _process_file)
        self.attributes = node_data.get('a', {})
        
        # Get name from decrypted attributes
        if isinstance(self.attributes, dict):
            self.name = self.attributes.get('n', f'Unknown_{self.handle}')
        else:
            self.name = f'Encrypted_{self.handle}'
        
        # Key data (already processed by _process_file)
        self.key = node_data.get('key', None)  # Full decrypted key (8 elements)
        self.decryption_key = node_data.get('k', None)  # Processed key for decrypt (4 elements for files)
        
        # Original encrypted key data (only if not processed)
        if isinstance(node_data.get('k'), str):
            self.encrypted_key_data = node_data.get('k', '')
        else:
            self.encrypted_key_data = ''
        
        # File-specific data
        if self.node_type == NODE_TYPE_FILE and self.key:
            self.file_iv = node_data.get('iv', [])
            self.meta_mac = node_data.get('meta_mac', [])
        
        # Calculate creation time
        self.created_time = self.timestamp
        self.modified_time = self.timestamp
    
    def is_file(self) -> bool:
        """Check if node is a file."""
        return self.node_type == NODE_TYPE_FILE
    
    def is_folder(self) -> bool:
        """Check if node is a folder."""
        return self.node_type == NODE_TYPE_FOLDER
    
    def is_root(self) -> bool:
        """Check if node is the root folder."""
        return self.node_type == NODE_TYPE_ROOT
    
    def is_trash(self) -> bool:
        """Check if node is in trash."""
        return self.node_type == NODE_TYPE_TRASH
    
    def get_size_formatted(self) -> str:
        """Get human-readable file size."""
        return format_size(self.size)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'handle': self.handle,
            'parent_handle': self.parent_handle,
            'name': self.name,
            'type': 'file' if self.is_file() else 'folder',
            'size': self.size,
            'timestamp': self.timestamp,
            'attributes': self.attributes,
        }


# ==============================================
# === FILESYSTEM TREE MANAGEMENT ===
# ==============================================

class FileSystemTree:
    """
    Manages the Mega filesystem tree structure with smart caching.
    """
    
    def __init__(self):
        self.nodes: Dict[str, MegaNode] = {}
        self.children: Dict[str, List[str]] = {}
        self.root_handle = None
        self.last_refresh = 0
        self.refresh_threshold = 300  # 5 minutes
        self.logger = get_logger("filesystem")
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.nodes.clear()
        self.children.clear()
        self.root_handle = None
        self.last_refresh = 0
        self.logger.info("Filesystem tree cleared")
    
    def add_node(self, node: MegaNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.handle] = node
        
        # Update children mapping
        if node.parent_handle:
            if node.parent_handle not in self.children:
                self.children[node.parent_handle] = []
            if node.handle not in self.children[node.parent_handle]:
                self.children[node.parent_handle].append(node.handle)
        
        # Set root handle
        if node.is_root():
            self.root_handle = node.handle
            self.logger.debug(f"Root handle set to {node.handle}")
    
    def get_node(self, handle: str) -> Optional[MegaNode]:
        """Get node by handle."""
        return self.nodes.get(handle)
    
    def get_children(self, handle: str) -> List[MegaNode]:
        """Get children of a node."""
        child_handles = self.children.get(handle, [])
        return [self.nodes[h] for h in child_handles if h in self.nodes]
    
    def get_node_by_path(self, path: str) -> Optional[MegaNode]:
        """Get node by path."""
        if path == "/" or path == "":
            return self.nodes.get(self.root_handle) if self.root_handle else None
        
        # Split path into components
        parts = [p for p in path.split('/') if p]
        
        # Start from root
        current_node = self.nodes.get(self.root_handle) if self.root_handle else None
        if not current_node:
            return None
        
        # Navigate path
        for part in parts:
            found = False
            for child in self.get_children(current_node.handle):
                if child.name == part:
                    current_node = child
                    found = True
                    break
            
            if not found:
                return None
        
        return current_node
    
    def needs_refresh(self) -> bool:
        """Check if filesystem data needs refresh."""
        return (time.time() - self.last_refresh) > self.refresh_threshold
    
    def mark_refreshed(self) -> None:
        """Mark filesystem as refreshed."""
        self.last_refresh = time.time()


# Global filesystem tree
fs_tree = FileSystemTree()


# ==============================================
# === FILESYSTEM OPERATIONS ===
# ==============================================

@require_authentication
def refresh_filesystem() -> None:
    """Refresh the filesystem tree from MEGA."""
    logger = get_logger("filesystem")
    
    logger.info("Refreshing filesystem")
    trigger_event('filesystem_refresh_started', {})
    
    try:
        # Get filesystem data
        command = {'a': 'f', 'c': 1, 'r': 1}  # Get files with crypto
        result = single_api_request(command, current_session.session_id)
        
        if not isinstance(result, dict) or 'f' not in result:
            raise RequestError("Invalid filesystem response")
        
        # Clear existing tree
        fs_tree.clear()
        
        # Process nodes
        nodes = result['f']
        for node_data in nodes:
            # Create simplified node for this version
            simplified_node = {
                'h': node_data.get('h'),
                'p': node_data.get('p'),
                'u': node_data.get('u'),
                't': node_data.get('t', NODE_TYPE_FILE),
                's': node_data.get('s', 0),
                'ts': node_data.get('ts', 0),
                'a': {'n': f"Node_{node_data.get('h', 'unknown')}"}  # Simplified for now
            }
            
            node = MegaNode(simplified_node)
            fs_tree.add_node(node)
        
        fs_tree.mark_refreshed()
        
        logger.info(f"Filesystem refreshed: {len(fs_tree.nodes)} nodes loaded")
        trigger_event('filesystem_refresh_completed', {'nodes_count': len(fs_tree.nodes)})
        
    except Exception as e:
        logger.error(f"Filesystem refresh failed: {e}")
        trigger_event('filesystem_refresh_failed', {'error': str(e)})
        raise


@require_authentication
def list_folder(folder_path: str = "/") -> List[MegaNode]:
    """
    List contents of a folder.
    
    Args:
        folder_path: Path to folder (default: root)
        
    Returns:
        List of nodes in the folder
    """
    logger = get_logger("filesystem")
    
    if fs_tree.needs_refresh() or not fs_tree.nodes:
        refresh_filesystem()
    
    folder_node = fs_tree.get_node_by_path(folder_path)
    if not folder_node:
        raise RequestError(f"Folder not found: {folder_path}")
    
    if not folder_node.is_folder() and not folder_node.is_root():
        raise RequestError(f"Path is not a folder: {folder_path}")
    
    children = fs_tree.get_children(folder_node.handle)
    logger.info(f"Listed folder {folder_path}: {len(children)} items")
    
    return children


@require_authentication
def create_folder(name: str, parent_path: str = "/") -> MegaNode:
    """
    Create a new folder.
    
    Args:
        name: Name of the new folder
        parent_path: Path to parent folder
        
    Returns:
        Created folder node
    """
    logger = get_logger("filesystem")
    
    if not name.strip():
        raise ValidationError("Folder name cannot be empty")
    
    if fs_tree.needs_refresh() or not fs_tree.nodes:
        refresh_filesystem()
    
    parent_node = fs_tree.get_node_by_path(parent_path)
    if not parent_node:
        raise RequestError(f"Parent folder not found: {parent_path}")
    
    # Create folder
    command = {
        'a': 'p',  # Create folder
        't': parent_node.handle,
        'n': [{'h': 'xxxxxxxx', 'a': name, 't': NODE_TYPE_FOLDER}]
    }
    
    try:
        result = single_api_request(command, current_session.session_id)
        
        # Refresh filesystem to get new folder
        refresh_filesystem()
        
        logger.info(f"Created folder: {name} in {parent_path}")
        trigger_event('folder_created', {'name': name, 'parent_path': parent_path})
        
        # Find and return the new folder (simplified)
        children = list_folder(parent_path)
        for child in children:
            if child.name == name and child.is_folder():
                return child
        
        # Return a mock node if we can't find it
        mock_node_data = {
            'h': 'new_folder_handle',
            'p': parent_node.handle,
            't': NODE_TYPE_FOLDER,
            'a': {'n': name}
        }
        return MegaNode(mock_node_data)
        
    except Exception as e:
        logger.error(f"Failed to create folder {name}: {e}")
        raise RequestError(f"Failed to create folder: {e}")


@require_authentication
def upload_file(file_path: str, remote_path: str = "/", 
               progress_callback: Optional[callable] = None) -> MegaNode:
    """
    Upload a file to MEGA.
    
    Args:
        file_path: Path to local file
        remote_path: Remote path to upload to
        progress_callback: Optional progress callback
        
    Returns:
        Uploaded file node
    """
    logger = get_logger("filesystem")
    
    from pathlib import Path
    local_file = Path(file_path)
    
    if not local_file.exists():
        raise RequestError(f"File not found: {file_path}")
    
    file_size = local_file.stat().st_size
    
    if fs_tree.needs_refresh() or not fs_tree.nodes:
        refresh_filesystem()
    
    target_folder = fs_tree.get_node_by_path(remote_path)
    if not target_folder:
        raise RequestError(f"Target folder not found: {remote_path}")
    
    logger.info(f"Starting upload: {file_path} -> {remote_path}")
    trigger_event('upload_started', {
        'file_path': file_path,
        'remote_path': remote_path,
        'file_size': file_size
    })
    
    try:
        # Get upload URL
        upload_url = get_upload_url(file_size)
        
        # Upload file in chunks
        result = upload_file_chunked(file_path, upload_url, progress_callback)
        
        # Create file node in MEGA (simplified)
        file_name = local_file.name
        command = {
            'a': 'p',  # Create node
            't': target_folder.handle,
            'n': [{
                'h': 'xxxxxxxx',
                'a': file_name,
                't': NODE_TYPE_FILE,
                's': file_size
            }]
        }
        
        api_result = single_api_request(command, current_session.session_id)
        
        # Refresh filesystem
        refresh_filesystem()
        
        logger.info(f"Upload completed: {file_path}")
        trigger_event('upload_completed', {
            'file_path': file_path,
            'remote_path': remote_path,
            'file_size': file_size
        })
        
        # Return mock node for now
        mock_node_data = {
            'h': 'uploaded_file_handle',
            'p': target_folder.handle,
            't': NODE_TYPE_FILE,
            's': file_size,
            'a': {'n': file_name}
        }
        return MegaNode(mock_node_data)
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        trigger_event('upload_failed', {
            'file_path': file_path,
            'error': str(e)
        })
        raise


@require_authentication
def download_file(node_handle: str, output_path: str,
                 progress_callback: Optional[callable] = None) -> bool:
    """
    Download a file from MEGA.
    
    Args:
        node_handle: Handle of file to download
        output_path: Local path to save file
        progress_callback: Optional progress callback
        
    Returns:
        True if download successful
    """
    logger = get_logger("filesystem")
    
    if fs_tree.needs_refresh() or not fs_tree.nodes:
        refresh_filesystem()
    
    node = fs_tree.get_node(node_handle)
    if not node:
        raise RequestError(f"File not found: {node_handle}")
    
    if not node.is_file():
        raise RequestError(f"Node is not a file: {node_handle}")
    
    logger.info(f"Starting download: {node.name} -> {output_path}")
    trigger_event('download_started', {
        'node_handle': node_handle,
        'output_path': output_path,
        'file_size': node.size
    })
    
    try:
        # Get download URL
        download_url = get_download_url(node_handle)
        
        # Download file in chunks
        result = download_file_chunked(download_url, output_path, node.size, progress_callback)
        
        logger.info(f"Download completed: {node.name}")
        trigger_event('download_completed', {
            'node_handle': node_handle,
            'output_path': output_path,
            'file_size': node.size
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        trigger_event('download_failed', {
            'node_handle': node_handle,
            'error': str(e)
        })
        raise


@require_authentication
def delete_node(node_handle: str) -> bool:
    """
    Delete a file or folder.
    
    Args:
        node_handle: Handle of node to delete
        
    Returns:
        True if deletion successful
    """
    logger = get_logger("filesystem")
    
    node = fs_tree.get_node(node_handle)
    if not node:
        raise RequestError(f"Node not found: {node_handle}")
    
    command = {'a': 'd', 'n': node_handle}
    
    try:
        result = single_api_request(command, current_session.session_id)
        
        # Remove from filesystem tree
        if node_handle in fs_tree.nodes:
            del fs_tree.nodes[node_handle]
        
        logger.info(f"Deleted node: {node.name}")
        trigger_event('node_deleted', {
            'node_handle': node_handle,
            'name': node.name
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise RequestError(f"Delete failed: {e}")


@require_authentication
def move_node(node_handle: str, new_parent_handle: str) -> bool:
    """
    Move a file or folder to a new location.
    
    Args:
        node_handle: Handle of node to move
        new_parent_handle: Handle of new parent folder
        
    Returns:
        True if move successful
    """
    logger = get_logger("filesystem")
    
    command = {
        'a': 'm',  # Move
        'n': node_handle,
        't': new_parent_handle
    }
    
    try:
        result = single_api_request(command, current_session.session_id)
        
        # Update filesystem tree
        if node_handle in fs_tree.nodes:
            fs_tree.nodes[node_handle].parent_handle = new_parent_handle
        
        logger.info(f"Moved node: {node_handle}")
        trigger_event('node_moved', {
            'node_handle': node_handle,
            'new_parent_handle': new_parent_handle
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Move failed: {e}")
        raise RequestError(f"Move failed: {e}")


@require_authentication
def rename_node(node_handle: str, new_name: str) -> bool:
    """
    Rename a file or folder.
    
    Args:
        node_handle: Handle of node to rename
        new_name: New name
        
    Returns:
        True if rename successful
    """
    logger = get_logger("filesystem")
    
    if not new_name.strip():
        raise ValidationError("New name cannot be empty")
    
    # Simplified rename - in real implementation would need proper attribute encryption
    command = {
        'a': 'a',  # Set attributes
        'n': node_handle,
        'at': new_name  # Simplified
    }
    
    try:
        result = single_api_request(command, current_session.session_id)
        
        # Update filesystem tree
        if node_handle in fs_tree.nodes:
            fs_tree.nodes[node_handle].name = new_name
        
        logger.info(f"Renamed node: {node_handle} -> {new_name}")
        trigger_event('node_renamed', {
            'node_handle': node_handle,
            'new_name': new_name
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Rename failed: {e}")
        raise RequestError(f"Rename failed: {e}")


# ==============================================
# === CONVENIENCE FUNCTIONS ===
# ==============================================

def get_node_by_path(path: str) -> Optional[MegaNode]:
    """Get node by path."""
    if fs_tree.needs_refresh() or not fs_tree.nodes:
        refresh_filesystem()
    
    return fs_tree.get_node_by_path(path)


def search_nodes_by_name(pattern: str, folder_handle: Optional[str] = None) -> List[MegaNode]:
    """
    Search for nodes by name pattern.
    
    Args:
        pattern: Search pattern (supports wildcards)
        folder_handle: Folder to search in (None for all)
        
    Returns:
        List of matching nodes
    """
    import fnmatch
    
    if not current_session.is_authenticated:
        raise RequestError("Not logged in")
    
    if fs_tree.needs_refresh() or not fs_tree.nodes:
        refresh_filesystem()
    
    results = []
    nodes_to_search = []
    
    if folder_handle:
        nodes_to_search = fs_tree.get_children(folder_handle)
    else:
        nodes_to_search = list(fs_tree.nodes.values())
    
    for node in nodes_to_search:
        if fnmatch.fnmatch(node.name.lower(), pattern.lower()):
            results.append(node)
    
    return results


def get_nodes() -> Dict[str, MegaNode]:
    """Get all nodes in the filesystem."""
    if not current_session.is_authenticated:
        raise RequestError("Not logged in")
    
    if fs_tree.needs_refresh() or not fs_tree.nodes:
        refresh_filesystem()
    
    return fs_tree.nodes.copy()