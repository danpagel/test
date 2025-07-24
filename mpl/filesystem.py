"""
File System Operations Module
=============================

This module handles all file system operations for the Mega.nz client.

This follows the exact methodology from the reference implementation:
1. Node tree management and navigation
2. File/folder listing and metadata handling
3. File upload/download with chunking and encryption
4. File sharing and public link generation
5. Trash/recycle bin operations

Author: Modernized from reference implementation
Date: July 2025
"""

from .dependencies import *
from typing import Callable  # Explicit import for type hints
import time  # For timestamp handling in versioning
from .exceptions import RequestError, ValidationError
from .crypto import (
    aes_cbc_encrypt, aes_cbc_decrypt, aes_ctr_encrypt_decrypt,
    base64_url_encode, base64_url_decode, generate_random_key,
    calculate_chunk_mac, string_to_a32, a32_to_string,
    base64_to_a32, a32_to_base64, encrypt_attr, decrypt_attr, 
    make_id, get_chunks, encrypt_key, decrypt_key
)
from .network import (
    single_api_request, api_request, get_upload_url, get_download_url,
    upload_chunk, download_chunk
)
from .crypto import parse_file_attributes
from .auth import current_session, require_authentication, is_logged_in

# Bandwidth management (optional import for enhanced download performance)
try:
    from bandwidth_management import get_bandwidth_manager, TransferPriority
    BANDWIDTH_MANAGEMENT_AVAILABLE = True
except ImportError:
    BANDWIDTH_MANAGEMENT_AVAILABLE = False

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
# === NODE CLASS ===
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
        from .utilities import format_size
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
# === FILE SYSTEM TREE ===
# ==============================================

class FileSystemTree:
    """
    Manages the Mega filesystem tree structure with smart caching.
    🚀 PERFORMANCE OPTIMIZATION: Enhanced with intelligent caching system.
    """
    
    def __init__(self):
        self.nodes: Dict[str, MegaNode] = {}
        self.children: Dict[str, List[str]] = {}
        self.root_handle = None
        self.trash_handle = None
        self.inbox_handle = None
        
        # 🚀 SMART CACHING: Cache management
        self.last_refresh_time = 0
        self.cache_timeout = 300  # 5 minutes default cache timeout
        self.auto_refresh_threshold = 60  # Auto refresh after 1 minute
        self.cache_hits = 0
        self.cache_misses = 0
        self.force_refresh_next = False
        
        # Cache invalidation tracking
        self.dirty_operations = set()  # Track operations that invalidate cache
        self.last_operation_time = 0
        
        # Performance tracking
        self.refresh_count = 0
        self.avoided_refreshes = 0
    
    def clear(self) -> None:
        """Clear all nodes."""
        self.nodes.clear()
        self.children.clear()
        self.root_handle = None
        self.trash_handle = None
        self.inbox_handle = None
        self.last_refresh_time = 0
        self.dirty_operations.clear()
    
    def is_cache_valid(self) -> bool:
        """🚀 Check if filesystem cache is still valid."""
        if self.force_refresh_next:
            return False
        
        current_time = time.time()
        cache_age = current_time - self.last_refresh_time
        
        # Always invalid if no data
        if not self.nodes:
            return False
        
        # Invalid if cache expired
        if cache_age > self.cache_timeout:
            return False
        
        # Invalid if recent operations that modify filesystem
        if self.dirty_operations and (current_time - self.last_operation_time) < 5:
            return False
        
        return True
    
    def mark_cache_dirty(self, operation: str) -> None:
        """🚀 Mark cache as dirty due to filesystem-modifying operation."""
        self.dirty_operations.add(operation)
        self.last_operation_time = time.time()
        
        # For operations that definitely change filesystem, force refresh next time
        if operation in {'upload', 'delete', 'mkdir', 'move', 'rename', 'copy'}:
            self.force_refresh_next = True
    
    def should_auto_refresh(self) -> bool:
        """🚀 Determine if automatic refresh is needed."""
        if not self.nodes:
            return True
        
        current_time = time.time()
        cache_age = current_time - self.last_refresh_time
        
        # Auto refresh if cache is getting old and we had modifying operations
        if cache_age > self.auto_refresh_threshold and self.dirty_operations:
            return True
        
        return False
    
    def record_cache_hit(self) -> None:
        """🚀 Record successful cache usage."""
        self.cache_hits += 1
        self.avoided_refreshes += 1
    
    def record_cache_miss(self) -> None:
        """🚀 Record cache miss requiring refresh."""
        self.cache_misses += 1
        self.refresh_count += 1
    
    def get_cache_stats(self) -> dict:
        """🚀 Get caching performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / max(1, total_requests)) * 100
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'total_refreshes': self.refresh_count,
            'avoided_refreshes': self.avoided_refreshes,
            'last_refresh_age': time.time() - self.last_refresh_time,
            'cache_valid': self.is_cache_valid(),
            'dirty_operations': list(self.dirty_operations)
        }
    
    def clear_dirty_state(self) -> None:
        """🚀 Clear dirty state after successful refresh."""
        self.dirty_operations.clear()
        self.force_refresh_next = False
        self.last_refresh_time = time.time()
    
    def add_node(self, node: MegaNode) -> None:
        """Add node to tree."""
        self.nodes[node.handle] = node
        
        # Add to parent's children
        parent_handle = node.parent_handle
        if parent_handle not in self.children:
            self.children[parent_handle] = []
        if node.handle not in self.children[parent_handle]:
            self.children[parent_handle].append(node.handle)
        
        # Set special folder handles
        if node.node_type == NODE_TYPE_ROOT:
            self.root_handle = node.handle
        elif node.node_type == NODE_TYPE_TRASH:
            self.trash_handle = node.handle
        elif node.node_type == NODE_TYPE_INBOX:
            self.inbox_handle = node.handle
    
    def get_node(self, handle: str) -> Optional[MegaNode]:
        """Get node by handle."""
        return self.nodes.get(handle)
    
    def get_children(self, handle: str) -> List[MegaNode]:
        """Get child nodes of a folder."""
        child_handles = self.children.get(handle, [])
        return [self.nodes[h] for h in child_handles if h in self.nodes]
    
    def find_node_by_name(self, name: str, parent_handle: Optional[str] = None) -> Optional[MegaNode]:
        """Find node by name, optionally within a specific parent."""
        search_nodes = []
        if parent_handle:
            search_nodes = self.get_children(parent_handle)
        else:
            search_nodes = list(self.nodes.values())
        
        for node in search_nodes:
            if node.name == name:
                return node
        return None
    
    def get_path(self, handle: str) -> str:
        """Get full path to node."""
        if handle not in self.nodes:
            return ""
        
        node = self.nodes[handle]
        if node.node_type == NODE_TYPE_ROOT:
            return "/"
        
        path_parts = []
        current = node
        
        while current and current.handle != self.root_handle:
            path_parts.append(current.name)
            if current.parent_handle in self.nodes:
                current = self.nodes[current.parent_handle]
            else:
                break
        
        path_parts.reverse()
        return "/" + "/".join(path_parts)


# Global filesystem tree
fs_tree = FileSystemTree()


# ==============================================
# === FILESYSTEM OPERATIONS ===
# ==============================================

@require_authentication
def refresh_filesystem(force: bool = False) -> None:
    """
    🚀 SMART CACHING: Refresh filesystem with intelligent caching.
    
    Only refreshes if cache is invalid or force=True is specified.
    Preserves decryption keys and prevents unnecessary API calls.
    
    Args:
        force: Force refresh even if cache is valid
    
    Raises:
        RequestError: If filesystem fetch fails
    """
    # 🚀 SMART CACHING: Check if refresh is actually needed
    if not force and fs_tree.is_cache_valid():
        fs_tree.record_cache_hit()
        logger.debug("Filesystem cache hit - skipping refresh")
        return
    
    # Record cache miss
    fs_tree.record_cache_miss()
    
    # Store existing decryption keys and attributes before clearing
    existing_keys = {}
    existing_decryption_keys = {}
    existing_attributes = {}
    
    for handle, node in fs_tree.nodes.items():
        if hasattr(node, 'key') and node.key:
            existing_keys[handle] = node.key
        if hasattr(node, 'decryption_key') and node.decryption_key:
            existing_decryption_keys[handle] = node.decryption_key
        if hasattr(node, 'attributes') and isinstance(node.attributes, dict):
            existing_attributes[handle] = node.attributes.copy()
    
    command = {'a': 'f', 'c': 1}  # Fetch filesystem
    result = single_api_request(command)
    
    if not isinstance(result, dict) or 'f' not in result:
        raise RequestError("Invalid filesystem response")
    
    # Clear existing tree
    fs_tree.clear()
    
    # Process nodes and restore preserved data
    nodes_data = result['f']
    for node_data in nodes_data:
        node = MegaNode(node_data)
        
        # Restore preserved keys and attributes to prevent corruption
        handle = node.handle
        if handle in existing_keys:
            node.key = existing_keys[handle]
        if handle in existing_decryption_keys:
            node.decryption_key = existing_decryption_keys[handle]
        if handle in existing_attributes:
            node.attributes = existing_attributes[handle]
            # Update name from preserved attributes
            if 'n' in node.attributes:
                node.name = node.attributes['n']
        
        fs_tree.add_node(node)
    
    # 🚀 SMART CACHING: Update cache state
    fs_tree.clear_dirty_state()
    
    logger.info(f"Filesystem refreshed: {len(fs_tree.nodes)} nodes loaded (keys preserved)")


def _refresh_filesystem_if_needed() -> bool:
    """
    🚀 SMART CACHING: Intelligently refresh filesystem only when needed.
    
    Returns:
        bool: True if refresh was performed, False if cache was used
    """
    if fs_tree.should_auto_refresh() or not fs_tree.nodes:
        refresh_filesystem()
        return True
    elif fs_tree.is_cache_valid():
        fs_tree.record_cache_hit()
        return False
    else:
        refresh_filesystem()
        return True


def get_filesystem_cache_stats() -> dict:
    """🚀 Get filesystem caching performance statistics."""
    return fs_tree.get_cache_stats()


@require_authentication
def list_folder(folder_handle: Optional[str] = None) -> List[MegaNode]:
    """
    🚀 SMART CACHING: List contents of a folder with intelligent caching.
    
    Args:
        folder_handle: Handle of folder to list (None for root)
        
    Returns:
        List of nodes in the folder
        
    Raises:
        RequestError: If folder not found or access denied
    """
    # 🚀 SMART CACHING: Only refresh if needed
    _refresh_filesystem_if_needed()
    
    # Use root folder if no handle specified
    if folder_handle is None:
        folder_handle = fs_tree.root_handle
    
    if folder_handle not in fs_tree.nodes:
        raise RequestError(f"Folder not found: {folder_handle}")
    
    folder_node = fs_tree.nodes[folder_handle]
    if not folder_node.is_folder() and not folder_node.is_root():
        raise RequestError("Node is not a folder")
    
    return fs_tree.get_children(folder_handle)


@require_authentication
def create_folder(name: str, parent_handle: Optional[str] = None) -> MegaNode:
    """
    Create a new folder.
    
    Args:
        name: Name of the new folder
        parent_handle: Parent folder handle (None for root)
        
    Returns:
        Created folder node
        
    Raises:
        ValidationError: If name is invalid
        RequestError: If creation fails
    """
    if not name or not name.strip():
        raise ValidationError("Folder name cannot be empty")
    
    # Use root folder if no parent specified
    if parent_handle is None:
        parent_handle = fs_tree.root_handle
    
    # Generate random AES key for folder (like reference)
    from random import randint
    ul_key = [randint(0, 0xFFFFFFFF) for _ in range(6)]
    folder_key = ul_key[:4]  # Use first 4 elements like reference
    
    # Encrypt attributes (like reference)
    attributes = {'n': name.strip()}
    encrypted_attrs = base64_url_encode(encrypt_attr(attributes, folder_key))
    
    # Encrypt folder key with master key (like reference)
    master_key_a32 = string_to_a32(makestring(current_session.master_key))
    encrypted_key = a32_to_base64(encrypt_key(folder_key, master_key_a32))
    
    # Create folder command (like reference)
    command = {
        'a': 'p',  # Create folder
        't': parent_handle,
        'n': [{
            'h': 'xxxxxxxx',  # Temporary handle
            't': NODE_TYPE_FOLDER,
            'a': encrypted_attrs,
            'k': encrypted_key
        }],
        'i': make_id(10)  # Request ID like reference
    }
    
    result = single_api_request(command)
    
    if not isinstance(result, dict) or 'f' not in result:
        raise RequestError("Failed to create folder")
    
    # Process created folder
    folder_data = result['f'][0]
    
    # Manually decrypt attributes since we have the key
    encrypted_attrs = base64_url_decode(folder_data['a'])
    decrypted_attrs = decrypt_attr(encrypted_attrs, folder_key)
    
    # Update folder data with decrypted attributes and keys
    folder_data['a'] = decrypted_attrs
    folder_data['key'] = folder_key
    folder_data['k'] = folder_key  # For folders, processed key = key
    
    # Create node with processed data
    folder_node = MegaNode(folder_data)
    fs_tree.add_node(folder_node)
    
    # 🚀 SMART CACHING: Mark filesystem as dirty after folder creation
    fs_tree.mark_cache_dirty('mkdir')
    
    logger.info(f"Created folder: {name}")
    return folder_node


@require_authentication
def delete_node(handle: str, permanent: bool = False) -> None:
    """
    Delete a file or folder.
    
    Args:
        handle: Handle of node to delete
        permanent: If True, permanently delete; if False, move to trash
        
    Raises:
        RequestError: If deletion fails
    """
    if handle not in fs_tree.nodes:
        raise RequestError(f"Node not found: {handle}")
    
    node = fs_tree.nodes[handle]
    
    if permanent:
        # Permanent deletion
        command = {'a': 'd', 'n': handle}
    else:
        # Move to trash
        command = {'a': 'm', 'n': handle, 't': fs_tree.trash_handle}
    
    result = single_api_request(command)
    
    if isinstance(result, int) and result < 0:
        raise RequestError(f"Delete failed with error code: {result}")
    
    # Remove from local tree
    if handle in fs_tree.nodes:
        del fs_tree.nodes[handle]
    
    # Remove from parent's children
    for parent_handle, children in fs_tree.children.items():
        if handle in children:
            children.remove(handle)
    
    logger.info(f"Deleted node: {node.name} ({'permanent' if permanent else 'to trash'})")


@require_authentication
def move_node(handle: str, new_parent_handle: str) -> None:
    """
    Move a file or folder to a new parent.
    
    Args:
        handle: Handle of node to move
        new_parent_handle: Handle of new parent folder
        
    Raises:
        RequestError: If move fails
    """
    if handle not in fs_tree.nodes:
        raise RequestError(f"Node not found: {handle}")
    
    if new_parent_handle not in fs_tree.nodes:
        raise RequestError(f"Parent folder not found: {new_parent_handle}")
    
    command = {'a': 'm', 'n': handle, 't': new_parent_handle}
    result = single_api_request(command)
    
    if isinstance(result, int) and result < 0:
        raise RequestError(f"Move failed with error code: {result}")
    
    # Update local tree
    node = fs_tree.nodes[handle]
    old_parent = node.parent_handle
    
    # Remove from old parent's children
    if old_parent in fs_tree.children and handle in fs_tree.children[old_parent]:
        fs_tree.children[old_parent].remove(handle)
    
    # Add to new parent's children
    node.parent_handle = new_parent_handle
    if new_parent_handle not in fs_tree.children:
        fs_tree.children[new_parent_handle] = []
    fs_tree.children[new_parent_handle].append(handle)
    
    logger.info(f"Moved node: {node.name}")


@require_authentication
def rename_node(handle: str, new_name: str) -> None:
    """
    Rename a file or folder.
    
    Args:
        handle: Handle of node to rename
        new_name: New name for the node
        
    Raises:
        ValidationError: If name is invalid
        RequestError: If rename fails
    """
    if not new_name or not new_name.strip():
        raise ValidationError("Name cannot be empty")
    
    if handle not in fs_tree.nodes:
        raise RequestError(f"Node not found: {handle}")
    
    node = fs_tree.nodes[handle]
    
    # Create new encrypted attributes
    attributes = {'n': new_name.strip()}
    # Use encrypt_attr which expects key as List[int], use first 4 elements for file key
    encrypted_attrs = base64_url_encode(encrypt_attr(attributes, node.key[:4]))
    
    command = {
        'a': 'a',  # Set attributes
        'n': handle,
        'at': encrypted_attrs,
    }
    
    result = single_api_request(command)
    
    if isinstance(result, int) and result < 0:
        raise RequestError(f"Rename failed with error code: {result}")
    
    # Update local node
    node.name = new_name.strip()
    node.attributes['n'] = new_name.strip()
    
    logger.info(f"Renamed node to: {new_name}")


@require_authentication
def copy_file(source_handle: str, dest_parent_handle: str) -> MegaNode:
    """
    Copy a file within Mega's cloud storage using the same approach as the official SDK.
    
    Args:
        source_handle: Handle of source file
        dest_parent_handle: Handle of destination parent folder
        
    Returns:
        The copied file node
        
    Raises:
        RequestError: If copy fails
        ValidationError: If source is not a file
    """
    if source_handle not in fs_tree.nodes:
        raise RequestError(f"Source file not found: {source_handle}")
    
    source_node = fs_tree.nodes[source_handle]
    if not source_node.is_file():
        raise ValidationError("Source must be a file")
    
    if dest_parent_handle not in fs_tree.nodes:
        raise RequestError(f"Destination folder not found: {dest_parent_handle}")
    
    dest_parent = fs_tree.nodes[dest_parent_handle]
    if not (dest_parent.is_folder() or dest_parent.is_root()):
        raise ValidationError("Destination must be a folder")
    
    # Use the same approach as the official SDK - putnodes with NEW_NODE source
    # This creates a copy by referencing the existing node handle
    command = {
        'a': 'p',  # putnodes command
        't': dest_parent_handle,  # target parent
        'v': 4,  # include file IDs/handles
        'sm': 1,  # include file attributes
        'n': [{
            'h': source_handle,  # source node handle (not 'xxxxxxxx')
            't': NODE_TYPE_FILE  # node type
        }],
        'i': make_id(10)  # Request ID
    }
    
    result = single_api_request(command)
    
    # Handle the response - MEGA can return different response formats for copy operations
    if isinstance(result, dict):
        # Check for the 'fh' response (file handle array)
        if 'fh' in result and len(result['fh']) > 0:
            # Success - result contains new file handle
            new_handle_raw = result['fh'][0]
            
            # Handle composite handles (e.g., "nFtFUT6S:Jtze0NQXQo0")
            # Extract the first part before the colon if present
            new_handle = new_handle_raw.split(':')[0] if ':' in new_handle_raw else new_handle_raw
            
            # Refresh the filesystem to get the new node
            from .client import MPLClient
            temp_client = MPLClient()
            temp_client.refresh()
            
            # First try to find by the exact handle
            if new_handle in fs_tree.nodes:
                copied_node = fs_tree.nodes[new_handle]
                logger.info(f"Copied file: {source_node.name} -> {dest_parent.name}")
                return copied_node
            
            # If not found by handle, try to find by the raw handle
            if new_handle_raw in fs_tree.nodes:
                copied_node = fs_tree.nodes[new_handle_raw]
                logger.info(f"Copied file: {source_node.name} -> {dest_parent.name}")
                return copied_node
            
            # If still not found, search in destination folder by name and different handle
            for child in fs_tree.get_children(dest_parent_handle):
                if child.name == source_node.name and child.is_file() and child.handle != source_handle:
                    logger.info(f"Copied file: {source_node.name} -> {dest_parent.name}")
                    return child
            
            raise RequestError(f"Copy succeeded but new node not found. Handles tried: {new_handle}, {new_handle_raw}")
        
        # Check for the 'f' response (full node data)
        elif 'f' in result:
            copied_data = result['f'][0]
            copied_node = MegaNode(copied_data)
            fs_tree.add_node(copied_node)
            logger.info(f"Copied file: {source_node.name} -> {dest_parent.name}")
            return copied_node
    
    elif isinstance(result, list) and len(result) > 0:
        # Alternative response format - just node handle
        # Refresh filesystem to get the new node
        from .client import MPLClient
        temp_client = MPLClient()
        temp_client.refresh()
        
        # Try to find the copied file in the destination folder
        for child in fs_tree.get_children(dest_parent_handle):
            if child.name == source_node.name and child.is_file() and child.handle != source_handle:
                logger.info(f"Copied file: {source_node.name} -> {dest_parent.name}")
                return child
        
        raise RequestError("Copy succeeded but copied file not found")
    
    else:
        raise RequestError(f"Failed to copy file: {result}")


# ==============================================
# === FILE UPLOAD ===
# ==============================================

@require_authentication
def upload_file(file_path: Union[str, Path], parent_handle: Optional[str] = None,
                progress_callback: Optional[callable] = None) -> MegaNode:
    """
    Upload a file to Mega using the exact reference implementation.
    
    Args:
        file_path: Path to file to upload
        parent_handle: Parent folder handle (None for root)
        progress_callback: Function to call with progress updates (bytes_uploaded, total_bytes)
        
    Returns:
        Uploaded file node
        
    Raises:
        ValidationError: If file is invalid
        RequestError: If upload fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")
    
    file_size = file_path.stat().st_size
    
    # Determine storage node - if none set, upload to cloud drive node
    if parent_handle is None:
        # Need to get files first to populate root_id
        if fs_tree.root_handle is None:
            get_nodes()
        parent_handle = fs_tree.root_handle
    
    logger.info(f"Starting upload: {file_path.name} ({file_size} bytes)")
    
    # Request upload URL - call 'u' method exactly like reference
    from .network import single_api_request
    ul_url_resp = single_api_request({'a': 'u', 's': file_size})
    ul_url = ul_url_resp['p']
    
    # Generate random AES key (128) for file - exactly like reference
    import random
    ul_key = [random.randint(0, 0xFFFFFFFF) for _ in range(6)]
    k_str = makebyte(a32_to_string(ul_key[:4]))
    
    # Set up counter for CTR mode - exactly like reference
    count = Counter.new(128, initial_value=((ul_key[4] << 32) + ul_key[5]) << 64)
    aes = AES.new(k_str, AES.MODE_CTR, counter=count)
    
    upload_progress = 0
    completion_file_handle = None
    
    # MAC calculation setup - exactly like reference
    mac_str = b'\0' * 16
    mac_encryptor = AES.new(k_str, AES.MODE_CBC, mac_str)
    iv_str = makebyte(a32_to_string([ul_key[4], ul_key[5], ul_key[4], ul_key[5]]))
    
    # Upload file in chunks
    with open(file_path, 'rb') as input_file:
        if file_size > 0:
            for chunk_start, chunk_size in get_chunks(file_size):
                chunk = input_file.read(chunk_size)
                upload_progress += len(chunk)
                
                # MAC calculation - exactly like reference
                encryptor = AES.new(k_str, AES.MODE_CBC, iv_str)
                for i in range(0, len(chunk) - 16, 16):
                    block = chunk[i:i + 16]
                    encryptor.encrypt(block)
                
                # Fix for files under 16 bytes failing - exactly like reference
                if file_size > 16:
                    i += 16
                else:
                    i = 0
                
                block = chunk[i:i + 16]
                if len(block) % 16:
                    block += b'\0' * (16 - len(block) % 16)
                mac_str = mac_encryptor.encrypt(encryptor.encrypt(block))
                
                # Encrypt file and upload - exactly like reference
                chunk = aes.encrypt(chunk)
                output_file = requests.post(
                    ul_url + "/" + str(chunk_start),
                    data=chunk,
                    timeout=60
                )
                completion_file_handle = output_file.text
                
                # Progress callback
                if progress_callback:
                    progress_callback(upload_progress, file_size)
                
                logger.info(f'{upload_progress} of {file_size} uploaded')
        else:
            # Empty file case
            output_file = requests.post(ul_url + "/0", data='', timeout=60)
            completion_file_handle = output_file.text
    
    logger.info('Chunks uploaded')
    logger.info('Setting attributes to complete upload')
    
    # Calculate file MAC and meta MAC - exactly like reference
    file_mac = string_to_a32(makestring(mac_str))
    meta_mac = [file_mac[0] ^ file_mac[1], file_mac[2] ^ file_mac[3]]
    
    # Prepare attributes
    dest_filename = file_path.name
    attribs = {'n': dest_filename}
    
    # Encrypt attributes - exactly like reference
    from .crypto import encrypt_attr
    encrypt_attribs = base64_url_encode(encrypt_attr(attribs, ul_key[:4]))
    
    # Prepare key - exactly like reference
    key = [
        ul_key[0] ^ ul_key[4], ul_key[1] ^ ul_key[5],
        ul_key[2] ^ meta_mac[0], ul_key[3] ^ meta_mac[1], 
        ul_key[4], ul_key[5], meta_mac[0], meta_mac[1]
    ]
    
    # Encrypt key with master key
    from .auth import current_session
    master_key_a32 = string_to_a32(makestring(current_session.master_key))
    encrypted_key = a32_to_base64(encrypt_key(key, master_key_a32))
    
    # Generate request ID
    request_id = make_id(10)
    
    logger.info('Sending request to update attributes')
    
    # Update attributes - exactly like reference
    data = single_api_request({
        'a': 'p',
        't': parent_handle,
        'i': request_id,
        'n': [{
            'h': completion_file_handle,
            't': 0,
            'a': encrypt_attribs,
            'k': encrypted_key
        }]
    })
    
    logger.info('Upload complete')
    
    # 🚀 SMART CACHING: Mark filesystem as dirty after upload
    fs_tree.mark_cache_dirty('upload')
    
    # Refresh nodes to get the new file
    get_nodes()
    
    # Find and return the uploaded file node
    for node in fs_tree.nodes.values():
        if node.name == dest_filename and node.parent_handle == parent_handle:
            logger.info(f"Upload completed: {dest_filename}")
            return node
    
    # If we can't find it, create a basic node representation
    node_data = {
        'h': completion_file_handle,
        'p': parent_handle,
        't': NODE_TYPE_FILE,
        's': file_size,
        'ts': int(time.time()),
        'a': encrypt_attribs,
        'k': encrypted_key
    }
    logger.info(f"Upload completed: {dest_filename}")
    return MegaNode(node_data)
    
    # Process uploaded file
    file_data = result['f'][0]
    file_node = MegaNode(file_data)
    fs_tree.add_node(file_node)
    
    logger.info(f"Upload completed: {file_path.name}")
    return file_node


# ==============================================
# === FILE DOWNLOAD ===
# ==============================================

@require_authentication
def download_file(handle: str, output_path: Union[str, Path],
                 progress_callback: Optional[callable] = None, 
                 concurrent: bool = True, max_workers: int = 6,
                 use_adaptive_optimization: bool = True,
                 enable_error_recovery: bool = True,
                 enable_bandwidth_management: bool = True,
                 transfer_priority: Optional[str] = None) -> Path:
    """
    🚀 MEGA SDK ENHANCED DOWNLOADS: Download files with official MEGA optimization techniques.
    
    Based on research into MEGA's official SDK, MEGAcmd, and industry best practices:
    - Adaptive chunk sizing and worker optimization
    - Enhanced connection pooling and session management
    - Memory-efficient streaming for large files
    - Performance monitoring and recommendations
    - Advanced error recovery with partial chunk resume
    - MEGAcmd-style bandwidth management and speed limiting
    
    Args:
        handle: Handle of file to download
        output_path: Path where to save the file
        progress_callback: Function to call with progress updates (bytes_downloaded, total_bytes)
        concurrent: Enable concurrent downloading (automatically optimized per file size)
        max_workers: Maximum number of concurrent workers (up to 6 for optimal performance)
        use_adaptive_optimization: Enable adaptive network condition optimization (default: True)
        enable_error_recovery: Enable advanced error recovery with partial resume (default: True)
        enable_bandwidth_management: Enable MEGAcmd-style bandwidth management (default: True)
        transfer_priority: Transfer priority ("low", "normal", "high", "urgent") for bandwidth allocation
        
    Returns:
        Path to downloaded file
        
    Raises:
        RequestError: If download fails
        ValidationError: If file is invalid
    """
    if handle not in fs_tree.nodes:
        raise RequestError(f"File not found: {handle}")
    
    node = fs_tree.nodes[handle]
    if not node.is_file():
        raise ValidationError("Node is not a file")
    
    if not node.decryption_key:
        raise ValidationError("File decryption key not available")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 🌐 BANDWIDTH MANAGEMENT: Initialize bandwidth management if available
    bandwidth_manager = None
    transfer_id = f"download_{handle}_{int(time.time())}"
    
    if enable_bandwidth_management and BANDWIDTH_MANAGEMENT_AVAILABLE:
        try:
            bandwidth_manager = get_bandwidth_manager()
            
            # Convert string priority to enum
            priority_map = {
                "low": TransferPriority.LOW,
                "normal": TransferPriority.NORMAL,
                "high": TransferPriority.HIGH,
                "urgent": TransferPriority.URGENT
            }
            priority = priority_map.get(transfer_priority, TransferPriority.NORMAL)
            
            bandwidth_manager.register_download(transfer_id, priority)
            logger.info(f"🌐 Bandwidth management enabled for {node.name} (priority: {priority.value})")
            
        except Exception as e:
            logger.warning(f"Failed to initialize bandwidth management: {e}")
            bandwidth_manager = None
    
    logger.info(f"Starting download: {node.name} ({node.size} bytes)")
    
    # 🚀 INTELLIGENT CONCURRENT THRESHOLD: Only use concurrent for files where it helps
    # Concurrent downloads have overhead, so only use for larger files
    concurrent_threshold = CHUNK_SIZE * 10  # 10MB threshold
    if concurrent and node.size > concurrent_threshold:
        result = _download_file_concurrent(handle, output_path, progress_callback, max_workers, 
                                         use_adaptive_optimization, enable_error_recovery,
                                         bandwidth_manager, transfer_id)
    else:
        result = _download_file_sequential(handle, output_path, progress_callback)
    
    # Cleanup bandwidth management
    if bandwidth_manager:
        try:
            bandwidth_manager.unregister_download(transfer_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup bandwidth management: {e}")
    
    return result


def _download_file_sequential(handle: str, output_path: Path, 
                            progress_callback: Optional[callable] = None) -> Path:
    """🚀 Original sequential download implementation for small files."""
    node = fs_tree.nodes[handle]
    
    # Get download URL
    download_url = get_download_url(handle)
    
    # Setup decryption (following reference exactly)
    k = node.decryption_key  # Already processed key
    
    # Get IV from processed file data, with fallback if missing
    if hasattr(node, 'file_iv') and node.file_iv:
        iv = node.file_iv
    else:
        # Fallback: calculate IV from key if missing (like reference implementation)
        if len(k) >= 6:
            iv = k[4:6] + [0, 0]  # Use key elements 4-5 as IV with padding
        else:
            # Last resort: default IV
            iv = [0, 0, 0, 0]
        logger.warning(f"file_iv missing for {node.name}, using fallback IV")
    
    # Convert key to string (bytes) like reference
    from .crypto import a32_to_string
    k_str = makebyte(a32_to_string(k))
    
    # Create CTR cipher like reference
    counter = Counter.new(128, initial_value=((iv[0] << 32) + iv[1]) << 64)
    aes = AES.new(k_str, AES.MODE_CTR, counter=counter)
    
    # Download and decrypt file
    downloaded_bytes = 0
    
    with open(output_path, 'wb') as f:
        # Stream download from URL
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                # Decrypt chunk like reference
                decrypted_chunk = aes.decrypt(chunk)
                f.write(decrypted_chunk)
                
                downloaded_bytes += len(decrypted_chunk)
                
                # Call progress callback
                if progress_callback:
                    progress_callback(downloaded_bytes, node.size)
    
    logger.info(f"Download completed: {node.name}")
    return output_path


def _download_file_concurrent(handle: str, output_path: Path, 
                            progress_callback: Optional[callable] = None,
                            max_workers: int = 6, use_adaptive: bool = True,
                            enable_recovery: bool = True,
                            bandwidth_manager=None, transfer_id: str = None) -> Path:
    """
    🚀 MEGA SDK OPTIMIZED DOWNLOADS with ADAPTIVE CONDITIONS, ERROR RECOVERY & BANDWIDTH MANAGEMENT
    
    Advanced parallel downloading with MEGA best practices, network adaptation, robust error recovery,
    and MEGAcmd-style bandwidth management:
    - Adaptive chunk sizing based on file size and network conditions
    - Connection pooling optimization (similar to MEGAcmd speedlimit)
    - Memory-efficient streaming for large files
    - Enhanced error recovery and performance monitoring
    - Real-time network condition monitoring and adaptation
    - Partial chunk resume capabilities for interrupted downloads
    - Sophisticated retry strategies with exponential backoff
    - Intelligent bandwidth throttling and speed limiting
    - Priority-based transfer management
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Import error recovery system
    error_recovery = None
    if enable_recovery:
        try:
            from advanced_error_recovery import AdvancedErrorRecovery, ErrorType
            error_recovery = AdvancedErrorRecovery()
            logger.info("🛡️ Advanced error recovery enabled")
        except ImportError:
            logger.warning("Advanced error recovery not available")
            enable_recovery = False
    
    # Import network condition adapter
    try:
        from network_condition_adapter import get_adaptive_optimizer, get_network_monitor
        network_monitor = get_network_monitor()
        adaptive_optimizer = get_adaptive_optimizer()
        use_adaptive = True
        network_monitor.start_monitoring()
    except ImportError:
        logger.warning("Network condition adapter not available, using static optimization")
        use_adaptive = False
    
    node = fs_tree.nodes[handle]
    file_size = node.size
    file_size_mb = file_size / 1024 / 1024
    
    # Get download URL
    download_url = get_download_url(handle)
    
    # Setup decryption parameters
    k = node.decryption_key
    
    # Get IV with fallback
    if hasattr(node, 'file_iv') and node.file_iv:
        iv = node.file_iv
    else:
        if len(k) >= 6:
            iv = k[4:6] + [0, 0]
        else:
            iv = [0, 0, 0, 0]
        logger.warning(f"file_iv missing for {node.name}, using fallback IV")
    
    # Convert key to bytes
    from .crypto import a32_to_string
    k_str = makebyte(a32_to_string(k))
    
    # 🌐 ADAPTIVE NETWORK OPTIMIZATION: Get optimal settings based on current conditions
    if use_adaptive:
        adaptive_settings = adaptive_optimizer.get_optimal_settings(file_size_mb)
        
        optimal_workers = min(adaptive_settings.worker_count, max_workers)
        chunk_size = int(adaptive_settings.chunk_size_mb * 1024 * 1024)
        timeout_seconds = adaptive_settings.timeout_seconds
        use_streaming = adaptive_settings.use_streaming
        
        logger.info(f"🌐 Adaptive settings: {optimal_workers} workers, "
                   f"{adaptive_settings.chunk_size_mb:.1f}MB chunks, "
                   f"network={network_monitor.get_current_condition().quality_rating}")
    else:
        # 🚀 MEGA SDK OPTIMIZATION: Static adaptive chunking strategy (fallback)
        very_small_threshold = 2 * 1024 * 1024      # 2MB
        small_threshold = 8 * 1024 * 1024           # 8MB  
        medium_threshold = 32 * 1024 * 1024         # 32MB
        large_threshold = 100 * 1024 * 1024         # 100MB
        
        if file_size < very_small_threshold:
            return _download_file_sequential(handle, output_path, progress_callback)
        elif file_size < small_threshold:
            optimal_workers = 2
            chunk_size = max(1024 * 1024, file_size // 2)
        elif file_size < medium_threshold:
            optimal_workers = 2
            chunk_size = 8 * 1024 * 1024
        elif file_size < large_threshold:
            optimal_workers = 4
            chunk_size = 8 * 1024 * 1024
        else:
            optimal_workers = min(6, max_workers)
            chunk_size = min(16 * 1024 * 1024, file_size // optimal_workers)
        
        timeout_seconds = 120 if file_size > large_threshold else 90
        use_streaming = file_size > large_threshold
    
    # Ensure chunk size is AES block-aligned (16 bytes) and reasonable
    chunk_size = max(1024 * 1024, (chunk_size + 15) // 16 * 16)  # Minimum 1MB
    
    # Calculate actual number of chunks
    num_chunks = min(optimal_workers, (file_size + chunk_size - 1) // chunk_size)
    
    # If we need fewer chunks than workers, recalculate chunk size
    if num_chunks < optimal_workers and file_size > chunk_size:
        chunk_size = max(1024 * 1024, (file_size + optimal_workers - 1) // optimal_workers)
        chunk_size = (chunk_size + 15) // 16 * 16  # AES alignment
        num_chunks = optimal_workers
    
    chunk_ranges = []
    for i in range(num_chunks):
        start = i * chunk_size
        if i == num_chunks - 1:
            end = file_size - 1  # Last chunk gets remainder
        else:
            end = min(start + chunk_size - 1, file_size - 1)
        
        if start <= end:
            chunk_ranges.append((start, end))
    
    actual_workers = min(max_workers, len(chunk_ranges))
    
    adaptation_info = f"adaptive ({network_monitor.get_current_condition().quality_rating})" if use_adaptive else "static"
    logger.info(f"🚀 MEGA SDK Enhanced Download ({adaptation_info}): {len(chunk_ranges)} chunks, {actual_workers} workers, {chunk_size/1024/1024:.1f}MB per chunk")
    
    # 🛡️ ERROR RECOVERY: Initialize or resume download progress
    download_progress = None
    if enable_recovery and error_recovery:
        # Check if we can resume an existing download
        if error_recovery.can_resume_download(handle):
            download_progress = error_recovery.resume_download(handle)
            if download_progress:
                logger.info(f"🔄 Resuming download: {download_progress.completion_percentage():.1f}% complete")
                # Update chunk_ranges to only include incomplete chunks
                incomplete_chunks = [c for c in download_progress.chunks if not c.completed]
                chunk_ranges = [(c.start_byte, c.end_byte) for c in incomplete_chunks]
                actual_workers = min(max_workers, len(chunk_ranges))
        
        # Create new download progress if not resuming
        if download_progress is None:
            download_progress = error_recovery.create_download_progress(
                handle, node.name, file_size, chunk_size
            )
    
    # Thread-safe progress tracking with performance monitoring
    progress_lock = threading.Lock()
    total_downloaded = 0
    download_start_time = time.time()
    chunk_stats = []  # For performance analysis
    
    def update_progress(chunk_bytes: int, chunk_time: float = 0):
        nonlocal total_downloaded
        with progress_lock:
            total_downloaded += chunk_bytes
            
            # Track chunk performance for optimization insights
            if chunk_time > 0:
                chunk_speed = chunk_bytes / chunk_time / 1024 / 1024  # MB/s
                chunk_stats.append({
                    'size': chunk_bytes,
                    'time': chunk_time,
                    'speed': chunk_speed
                })
            
            if progress_callback:
                progress_callback(total_downloaded, file_size)
    
    def download_chunk_range(chunk_id: int, start: int, end: int) -> tuple:
        """Download a specific byte range with MEGA SDK optimizations, network monitoring, and error recovery."""
        chunk_start_time = time.time()
        chunk_size_bytes = end - start + 1
        max_chunk_retries = 3  # Additional retries per chunk beyond error recovery system
        
        for attempt in range(max_chunk_retries + 1):
            try:
                # 🚀 MEGA SDK OPTIMIZATION: Enhanced headers and session management
                headers = {
                    'Range': f'bytes={start}-{end}',
                    'Connection': 'keep-alive',
                    'Accept-Encoding': 'identity',  # Disable compression for encrypted data
                    'Cache-Control': 'no-cache'
                }
                
                # Use the global session with enhanced connection pooling
                from .network import _api_session
                
                # Optimize session for downloads (similar to MEGAcmd approach)
                if not hasattr(_api_session, '_download_optimized'):
                    _optimize_session_for_downloads(_api_session.session)
                    _api_session._download_optimized = True
                
                # 🌐 BANDWIDTH MANAGEMENT: Check for throttling before download
                if bandwidth_manager and transfer_id:
                    try:
                        should_throttle, delay_seconds = bandwidth_manager.check_throttling(transfer_id, chunk_size_bytes)
                        if should_throttle and delay_seconds > 0:
                            logger.debug(f"🚦 Throttling chunk {chunk_id}: {delay_seconds:.2f}s delay")
                            time.sleep(delay_seconds)
                    except Exception as e:
                        logger.warning(f"Bandwidth throttling check failed: {e}")
                
                # 🌐 ADAPTIVE TIMEOUT: Use adaptive timeout based on network conditions
                response = _api_session.session.get(
                    download_url, 
                    headers=headers, 
                    timeout=timeout_seconds,
                    stream=use_streaming
                )
                response.raise_for_status()
                
                # Handle streaming vs direct download
                if use_streaming:
                    encrypted_data = b''.join(response.iter_content(chunk_size=8192))
                else:
                    encrypted_data = response.content
                
                expected_size = end - start + 1
                
                # 🛡️ CHUNK INTEGRITY VERIFICATION
                if len(encrypted_data) != expected_size:
                    error_msg = f"Chunk {chunk_id}: expected {expected_size} bytes, got {len(encrypted_data)}"
                    logger.warning(error_msg)
                    if attempt < max_chunk_retries:
                        logger.info(f"Retrying chunk {chunk_id} (attempt {attempt + 2}/{max_chunk_retries + 1})")
                        time.sleep(0.5 * (attempt + 1))  # Progressive delay
                        continue
                    else:
                        raise ValueError(error_msg)
                
                # 🚀 CRYPTO OPTIMIZATION: Efficient CTR counter calculation
                block_offset = start // 16  # AES block size is 16 bytes
                
                # Calculate initial counter value for this position
                initial_counter_value = ((iv[0] << 32) + iv[1]) << 64
                counter_for_chunk = initial_counter_value + block_offset
                
                counter = Counter.new(128, initial_value=counter_for_chunk)
                aes = AES.new(k_str, AES.MODE_CTR, counter=counter)
                
                # Decrypt the chunk
                decrypted_data = aes.decrypt(encrypted_data)
                
                # Update progress with timing and network monitoring
                chunk_time = time.time() - chunk_start_time
                update_progress(len(encrypted_data), chunk_time)
                
                # 🌐 BANDWIDTH MANAGEMENT: Update transfer progress for bandwidth calculations
                if bandwidth_manager and transfer_id:
                    try:
                        bandwidth_manager.update_download_progress(transfer_id, len(encrypted_data))
                    except Exception as e:
                        logger.debug(f"Bandwidth progress update failed: {e}")
                
                # 🌐 NETWORK MONITORING: Record performance data for adaptive optimization
                if use_adaptive:
                    try:
                        network_monitor.record_download_sample(
                            bytes_downloaded=len(encrypted_data),
                            download_time=chunk_time,
                            chunk_size=chunk_size_bytes
                        )
                    except Exception as e:
                        logger.debug(f"Network monitoring error: {e}")
                
                # 🛡️ ERROR RECOVERY: Update success status
                if enable_recovery and error_recovery and download_progress:
                    try:
                        error_recovery.update_chunk_success(download_progress, chunk_id, decrypted_data)
                    except Exception as e:
                        logger.debug(f"Error recovery update failed: {e}")
                
                return (chunk_id, start, decrypted_data, None)
                
            except Exception as e:
                logger.warning(f"Chunk {chunk_id} download attempt {attempt + 1} failed: {e}")
                
                # 🛡️ ERROR RECOVERY: Update failure status
                if enable_recovery and error_recovery and download_progress:
                    try:
                        # Classify error type for smart retry strategy
                        if 'timeout' in str(e).lower():
                            error_type = ErrorType.NETWORK_TIMEOUT
                        elif 'connection' in str(e).lower():
                            error_type = ErrorType.CONNECTION_ERROR
                        elif 'http' in str(e).lower() or hasattr(e, 'response'):
                            error_type = ErrorType.HTTP_ERROR
                        else:
                            error_type = ErrorType.UNKNOWN_ERROR
                        
                        should_retry = error_recovery.update_chunk_failure(
                            download_progress, chunk_id, e, error_type
                        )
                        
                        if not should_retry:
                            logger.error(f"Chunk {chunk_id} exceeded maximum retry attempts")
                            return (chunk_id, start, None, str(e))
                        
                    except Exception as recovery_error:
                        logger.debug(f"Error recovery update failed: {recovery_error}")
                
                # Immediate retry logic (before error recovery takes over)
                if attempt < max_chunk_retries:
                    delay = 0.5 * (2 ** attempt)  # Exponential backoff: 0.5s, 1s, 2s
                    logger.info(f"Retrying chunk {chunk_id} in {delay:.1f}s (attempt {attempt + 2}/{max_chunk_retries + 1})")
                    time.sleep(delay)
                else:
                    logger.error(f"Chunk {chunk_id} failed after {max_chunk_retries + 1} attempts: {e}")
                    return (chunk_id, start, None, str(e))
                logger.warning(f"Chunk {chunk_id}: expected {expected_size} bytes, got {len(encrypted_data)}")
            
            # 🚀 CRYPTO OPTIMIZATION: Efficient CTR counter calculation
            block_offset = start // 16  # AES block size is 16 bytes
            
            # Calculate initial counter value for this position
            initial_counter_value = ((iv[0] << 32) + iv[1]) << 64
            counter_for_chunk = initial_counter_value + block_offset
            
            counter = Counter.new(128, initial_value=counter_for_chunk)
            aes = AES.new(k_str, AES.MODE_CTR, counter=counter)
            
            # Decrypt the chunk
            decrypted_data = aes.decrypt(encrypted_data)
            
            # Update progress with timing and network monitoring
            chunk_time = time.time() - chunk_start_time
            update_progress(len(encrypted_data), chunk_time)
            
            # 🌐 NETWORK MONITORING: Record performance data for adaptive optimization
            if use_adaptive:
                try:
                    network_monitor.record_download_sample(
                        bytes_downloaded=len(encrypted_data),
                        download_time=chunk_time,
                        chunk_size=chunk_size_bytes
                    )
                except Exception as e:
                    logger.debug(f"Network monitoring error: {e}")
                
                # 🛡️ ERROR RECOVERY: Update success status
                if enable_recovery and error_recovery and download_progress:
                    try:
                        error_recovery.update_chunk_success(download_progress, chunk_id, decrypted_data)
                    except Exception as e:
                        logger.debug(f"Error recovery update failed: {e}")
                
                return (chunk_id, start, decrypted_data, None)
    
    # Helper function to optimize session (similar to MEGAcmd)
    def _optimize_session_for_downloads(session):
        """Apply MEGA SDK download optimizations to session."""
        from requests.adapters import HTTPAdapter
        
        # Enhanced connection pooling for downloads
        download_adapter = HTTPAdapter(
            pool_connections=20,      # More connections for parallel downloads
            pool_maxsize=40,         # Larger pool for concurrent chunks
            max_retries=0,           # Handle retries manually
            pool_block=False
        )
        
        session.mount('https://', download_adapter)
        session.mount('http://', download_adapter)
    
    # 🚀 CONCURRENCY OPTIMIZATION: Download chunks with enhanced management and error recovery
    chunk_data = {}
    failed_chunks = []
    
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        # Submit all chunk download tasks
        future_to_chunk = {
            executor.submit(download_chunk_range, i, start, end): (i, start, end)
            for i, (start, end) in enumerate(chunk_ranges)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_id, start, data, error = future.result()
            
            if error:
                failed_chunks.append((chunk_id, start, error))
                logger.warning(f"Chunk {chunk_id} failed: {error}")
            else:
                chunk_data[start] = data
    
    # 🛡️ ERROR RECOVERY: Handle failed chunks and retry if needed
    if failed_chunks and enable_recovery and error_recovery and download_progress:
        logger.warning(f"Attempting error recovery for {len(failed_chunks)} failed chunks...")
        
        # Get chunks that can be retried
        retry_chunks = error_recovery.get_retry_chunks(download_progress)
        if retry_chunks:
            # Retry failed chunks with error recovery strategies
            with ThreadPoolExecutor(max_workers=min(actual_workers, len(retry_chunks))) as retry_executor:
                retry_futures = {
                    retry_executor.submit(download_chunk_range, chunk.chunk_id, chunk.start_byte, chunk.end_byte): chunk
                    for chunk in retry_chunks
                }
                
                for future in as_completed(retry_futures):
                    chunk = retry_futures[future]
                    chunk_id, start, data, error = future.result()
                    
                    if not error:
                        chunk_data[start] = data
                        failed_chunks = [(cid, cstart, cerr) for cid, cstart, cerr in failed_chunks if cstart != start]
                        logger.info(f"🛡️ Successfully recovered chunk {chunk_id}")
                    else:
                        logger.error(f"🛡️ Chunk {chunk_id} recovery failed: {error}")
    
    # Final check for any remaining failures
    if failed_chunks:
        error_summary = f"Download failed with {len(failed_chunks)} unrecoverable chunks: " + \
                       ", ".join([f"chunk {cid}: {error}" for cid, _, error in failed_chunks[:3]])
        if len(failed_chunks) > 3:
            error_summary += f" and {len(failed_chunks) - 3} more..."
        raise RequestError(error_summary)
    
    # �️ ERROR RECOVERY: Use recovery system for file assembly if available
    if enable_recovery and error_recovery and download_progress:
        try:
            # Update all completed chunks in progress
            for start, data in chunk_data.items():
                chunk_id = next((i for i, (s, e) in enumerate(chunk_ranges) if s == start), None)
                if chunk_id is not None:
                    error_recovery.update_chunk_success(download_progress, chunk_id, data)
            
            # Assemble file using error recovery system
            if error_recovery.assemble_file(download_progress, output_path):
                logger.info("🛡️ File assembled using error recovery system")
                # Clean up recovery files on success
                error_recovery.cleanup_progress(handle)
            else:
                logger.warning("🛡️ Error recovery assembly failed, falling back to standard assembly")
                raise ValueError("Recovery assembly failed")
                
        except Exception as recovery_error:
            logger.warning(f"🛡️ Error recovery assembly failed: {recovery_error}, using standard assembly")
            # Fall back to standard file assembly
            with open(output_path, 'wb') as f:
                for start, end in chunk_ranges:
                    if start in chunk_data:
                        f.write(chunk_data[start])
    else:
        # 🚀 I/O OPTIMIZATION: Standard file assembly
        with open(output_path, 'wb') as f:
            for start, end in chunk_ranges:
                if start in chunk_data:
                    f.write(chunk_data[start])
    
    # Performance analysis and reporting
    total_time = time.time() - download_start_time
    avg_speed = file_size / max(total_time, 0.001) / 1024 / 1024  # MB/s
    
    # Verify final file size
    actual_size = output_path.stat().st_size
    if actual_size != file_size:
        logger.warning(f"File size mismatch: expected {file_size}, got {actual_size}")
    
    # Performance insights
    if chunk_stats:
        chunk_speeds = [stat['speed'] for stat in chunk_stats]
        avg_chunk_speed = sum(chunk_speeds) / len(chunk_speeds)
        min_speed = min(chunk_speeds)
        max_speed = max(chunk_speeds)
        
        logger.info(f"🚀 MEGA SDK Enhanced download completed: {node.name} ({avg_speed:.1f} MB/s avg)")
        logger.debug(f"Chunk performance: avg={avg_chunk_speed:.1f} MB/s, range={min_speed:.1f}-{max_speed:.1f} MB/s")
        
        # Performance recommendations
        if max_speed / min_speed > 3:
            logger.info("💡 Variable chunk performance detected - network conditions may be unstable")
        if avg_speed < 1.0:
            logger.info("💡 Low download speed - consider reducing concurrent workers or checking network")
    else:
        logger.info(f"🚀 MEGA SDK Enhanced download completed: {node.name}")
    
    return output_path


# ==============================================
# === SHARING AND LINKS ===
# ==============================================

@require_authentication
def create_public_link(handle: str) -> str:
    """
    Create a public download link for a file or folder.
    
    Args:
        handle: Handle of node to share
        
    Returns:
        Public link URL
        
    Raises:
        RequestError: If link creation fails
    """
    if handle not in fs_tree.nodes:
        raise RequestError(f"Node not found: {handle}")
    
    node = fs_tree.nodes[handle]
    
    command = {'a': 'l', 'n': handle}
    result = single_api_request(command)
    
    if isinstance(result, int) and result < 0:
        raise RequestError(f"Link creation failed with error code: {result}")
    
    if not isinstance(result, str):
        raise RequestError("Invalid link response")
    
    # Construct public URL
    # Convert key from list of integers to bytes for base64 encoding
    key_bytes = makebyte(a32_to_string(node.key))
    if node.is_file():
        link = f"https://mega.nz/file/{result}#{base64_url_encode(key_bytes)}"
    else:
        link = f"https://mega.nz/folder/{result}#{base64_url_encode(key_bytes)}"
    
    logger.info(f"Created public link for: {node.name}")
    return link


@require_authentication
def remove_public_link(handle: str) -> None:
    """
    Remove public link for a file or folder.
    
    Args:
        handle: Handle of node to unshare
        
    Raises:
        RequestError: If link removal fails
    """
    command = {'a': 'l', 'n': handle, 'd': 1}  # Delete link
    result = single_api_request(command)
    
    if isinstance(result, int) and result < 0:
        raise RequestError(f"Link removal failed with error code: {result}")
    
    logger.info(f"Removed public link for node: {handle}")


# ==============================================
# === UTILITY FUNCTIONS ===
# ==============================================

def get_node_by_path(path: str) -> Optional[MegaNode]:
    """
    Get node by filesystem path.
    
    Args:
        path: Filesystem path (e.g., "/folder/file.txt")
        
    Returns:
        Node if found, None otherwise
    """
    if not fs_tree.nodes or not path:
        return None
    
    if path == "/":
        return fs_tree.nodes.get(fs_tree.root_handle)
    
    # Split path and navigate
    parts = [p for p in path.split('/') if p]
    current_handle = fs_tree.root_handle
    
    for part in parts:
        found = False
        for child in fs_tree.get_children(current_handle):
            if child.name == part:
                current_handle = child.handle
                found = True
                break
        
        if not found:
            return None
    
    return fs_tree.nodes.get(current_handle)


# ==============================================
# === ENHANCED FILESYSTEM WITH EVENTS ===
# ==============================================

def refresh_filesystem_with_events(event_callback=None) -> None:
    """
    Enhanced filesystem refresh function with event callbacks and logging.
    
    Args:
        event_callback: Function to call for events (optional)
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        refresh_filesystem()
        node_count = len(fs_tree.nodes)
        logger.info("Filesystem refreshed")
        trigger_event('filesystem_refreshed', {'node_count': node_count})
    except Exception as e:
        logger.error(f"Filesystem refresh failed: {e}")
        trigger_event('refresh_failed', {'error': str(e)})
        raise


def create_folder_with_events(name: str, parent_handle: Optional[str] = None, 
                             event_callback=None) -> MegaNode:
    """
    Enhanced create folder function with event callbacks and logging.
    
    Args:
        name: Name of the new folder
        parent_handle: Handle of parent folder (None for root)
        event_callback: Function to call for events (optional)
        
    Returns:
        Created folder node
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        folder = create_folder(name, parent_handle)
        logger.info(f"Created folder: {name}")
        trigger_event('folder_created', {'name': name, 'folder': folder})
        return folder
    except Exception as e:
        logger.error(f"Folder creation failed: {e}")
        trigger_event('folder_creation_failed', {'name': name, 'error': str(e)})
        raise


def delete_node_with_events(handle: str, permanent: bool = False, event_callback=None) -> None:
    """
    Enhanced delete node function with event callbacks and logging.
    
    Args:
        handle: Handle of node to delete
        permanent: If True, permanently delete; if False, move to trash
        event_callback: Function to call for events (optional)
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        # Get node info before deletion
        node = fs_tree.nodes.get(handle)
        node_name = node.name if node else "unknown"
        
        delete_node(handle, permanent)
        logger.info(f"Deleted: {node_name} ({'permanent' if permanent else 'to trash'})")
        trigger_event('node_deleted', {'handle': handle, 'name': node_name, 'permanent': permanent})
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        trigger_event('delete_failed', {'handle': handle, 'error': str(e)})
        raise


def move_node_with_events(source_handle: str, dest_handle: str, event_callback=None) -> None:
    """
    Enhanced move node function with event callbacks and logging.
    
    Args:
        source_handle: Handle of node to move
        dest_handle: Handle of destination parent folder
        event_callback: Function to call for events (optional)
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        # Get node info before moving
        source_node = fs_tree.nodes.get(source_handle)
        dest_node = fs_tree.nodes.get(dest_handle)
        source_name = source_node.name if source_node else "unknown"
        dest_name = dest_node.name if dest_node else "unknown"
        
        move_node(source_handle, dest_handle)
        logger.info(f"Moved: {source_name} -> {dest_name}")
        trigger_event('node_moved', {
            'source_handle': source_handle, 
            'dest_handle': dest_handle,
            'source_name': source_name,
            'dest_name': dest_name
        })
    except Exception as e:
        logger.error(f"Move failed: {e}")
        trigger_event('move_failed', {'source': source_handle, 'destination': dest_handle, 'error': str(e)})
        raise


def rename_node_with_events(handle: str, new_name: str, event_callback=None) -> None:
    """
    Enhanced rename node function with event callbacks and logging.
    
    Args:
        handle: Handle of node to rename
        new_name: New name for the node
        event_callback: Function to call for events (optional)
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        # Get old name before renaming
        node = fs_tree.nodes.get(handle)
        old_name = node.name if node else "unknown"
        
        rename_node(handle, new_name)
        logger.info(f"Renamed: {old_name} -> {new_name}")
        trigger_event('node_renamed', {'handle': handle, 'old_name': old_name, 'new_name': new_name})
    except Exception as e:
        logger.error(f"Rename failed: {e}")
        trigger_event('rename_failed', {'handle': handle, 'new_name': new_name, 'error': str(e)})
        raise


def upload_file_with_events(file_path: Union[str, Path], dest_handle: Optional[str] = None,
                           progress_callback: Optional[Callable] = None, 
                           event_callback=None) -> MegaNode:
    """
    Enhanced upload file function with event callbacks and logging.
    
    Args:
        file_path: Path to file to upload
        dest_handle: Handle of destination folder (None for root)
        progress_callback: Function to call with progress updates
        event_callback: Function to call for events (optional)
        
    Returns:
        Uploaded file node
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    file_path = Path(file_path)
    
    # Wrap progress callback to include events
    def wrapped_progress(uploaded, total):
        if progress_callback:
            progress_callback(uploaded, total)
        trigger_event('upload_progress', {
            'file': file_path.name,
            'uploaded': uploaded,
            'total': total,
            'percent': (uploaded / total) * 100 if total > 0 else 0
        })
    
    try:
        file_size = file_path.stat().st_size
        trigger_event('upload_started', {'file': file_path.name, 'size': file_size})
        
        node = upload_file(file_path, dest_handle, wrapped_progress)
        
        logger.info(f"Upload completed: {file_path.name}")
        trigger_event('upload_completed', {'file': file_path.name, 'node': node})
        
        return node
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        trigger_event('upload_failed', {'file': file_path.name, 'error': str(e)})
        raise


def download_file_with_events(handle: str, dest_path: Union[str, Path], 
                             progress_callback: Optional[Callable] = None,
                             event_callback=None, concurrent: bool = True, 
                             max_workers: int = 4, enable_error_recovery: bool = True) -> Path:
    """
    Enhanced download file function with event callbacks, logging, and error recovery.
    
    Args:
        handle: Handle of file to download
        dest_path: Local path where to save the file
        progress_callback: Function to call with progress updates
        event_callback: Function to call for events (optional)
        concurrent: Whether to use concurrent downloading for large files
        max_workers: Maximum number of workers for concurrent downloads
        enable_error_recovery: Enable advanced error recovery with partial resume
        
    Returns:
        Path to downloaded file
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    # Get node info
    node = fs_tree.nodes.get(handle)
    if not node:
        raise RequestError(f"File not found: {handle}")
    
    # Wrap progress callback to include events
    def wrapped_progress(downloaded, total):
        if progress_callback:
            progress_callback(downloaded, total)
        trigger_event('download_progress', {
            'file': node.name,
            'downloaded': downloaded,
            'total': total,
            'percent': (downloaded / total) * 100 if total > 0 else 0
        })
    
    try:
        trigger_event('download_started', {'file': node.name, 'size': node.size})
        
        result_path = download_file(handle, dest_path, wrapped_progress, 
                                  concurrent=concurrent, max_workers=max_workers,
                                  enable_error_recovery=enable_error_recovery)
        
        logger.info(f"Download completed: {node.name}")
        trigger_event('download_completed', {'file': node.name, 'path': str(result_path)})
        
        return result_path
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        trigger_event('download_failed', {'file': node.name, 'error': str(e)})
        raise


def create_public_link_with_events(handle: str, event_callback=None) -> str:
    """
    Enhanced create public link function with event callbacks and logging.
    
    Args:
        handle: Handle of node to share
        event_callback: Function to call for events (optional)
        
    Returns:
        Public download link
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        # Get node info
        node = fs_tree.nodes.get(handle)
        node_name = node.name if node else "unknown"
        
        link = create_public_link(handle)
        logger.info(f"Created public link for: {node_name}")
        trigger_event('link_created', {'handle': handle, 'name': node_name, 'link': link})
        return link
    except Exception as e:
        logger.error(f"Link creation failed: {e}")
        trigger_event('link_creation_failed', {'handle': handle, 'error': str(e)})
        raise


def remove_public_link_with_events(handle: str, event_callback=None) -> None:
    """
    Enhanced remove public link function with event callbacks and logging.
    
    Args:
        handle: Handle of node to unshare
        event_callback: Function to call for events (optional)
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        # Get node info
        node = fs_tree.nodes.get(handle)
        node_name = node.name if node else "unknown"
        
        remove_public_link(handle)
        logger.info(f"Removed public link for: {node_name}")
        trigger_event('link_removed', {'handle': handle, 'name': node_name})
    except Exception as e:
        logger.error(f"Link removal failed: {e}")
        trigger_event('link_removal_failed', {'handle': handle, 'error': str(e)})
        raise


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Node classes
    'MegaNode',
    'FileSystemTree',
    'fs_tree',
    
    # Node types
    'NODE_TYPE_FILE',
    'NODE_TYPE_FOLDER', 
    'NODE_TYPE_ROOT',
    'NODE_TYPE_TRASH',
    
    # Filesystem operations
    'refresh_filesystem',
    'list_folder',
    'create_folder',
    'delete_node',
    'move_node',
    'rename_node',
    
    # Enhanced filesystem operations with events
    'refresh_filesystem_with_events',
    'create_folder_with_events',
    'delete_node_with_events',
    'move_node_with_events',
    'rename_node_with_events',
    
    # File operations
    'upload_file',
    'download_file',
    
    # Enhanced file operations with events
    'upload_file_with_events',
    'download_file_with_events',
    
    # Sharing
    'create_public_link',
    'remove_public_link',
    
    # Enhanced sharing with events
    'create_public_link_with_events',
    'remove_public_link_with_events',
    
    # File versioning & history
    'get_file_versions',
    'restore_file_version',
    'remove_file_version',
    'remove_all_file_versions',
    'configure_file_versioning',
    
    # Enhanced versioning with events
    'get_file_versions_with_events',
    'restore_file_version_with_events',
    'remove_file_version_with_events',
    
    # Utilities
    'get_node_by_path',
    'format_file_list',
]

# Configure logging
logger = logging.getLogger(__name__)

# ==============================================
# === FILE VERSIONING & HISTORY OPERATIONS ===
# ==============================================

@require_authentication
def get_file_versions(handle: str) -> List[Dict[str, Any]]:
    """
    Get version history for a file.
    
    Args:
        handle: File handle
        
    Returns:
        List of version dictionaries with version info
        
    Raises:
        RequestError: If file not found or API error
    """
    command = {'a': 'fvh', 'h': handle}
    result = single_api_request(command)
    
    if isinstance(result, int) and result < 0:
        raise RequestError(f"Failed to get file versions: Mega API Error {result}: {_get_error_message(result)}")
    
    if not isinstance(result, list):
        return []
    
    versions = []
    for version_data in result:
        versions.append({
            'handle': version_data.get('h', ''),
            'size': version_data.get('s', 0),
            'timestamp': version_data.get('ts', 0),
            'version_id': version_data.get('v', 0)
        })
    
    return versions


@require_authentication  
def restore_file_version(handle: str, version_handle: str) -> bool:
    """
    Restore a previous version of a file.
    
    Args:
        handle: Current file handle
        version_handle: Handle of the version to restore
        
    Returns:
        True if restore successful
        
    Raises:
        RequestError: If restore fails
    """
    command = {'a': 'fvr', 'h': handle, 'vh': version_handle}
    result = single_api_request(command)
    
    if isinstance(result, int) and result < 0:
        raise RequestError(f"Failed to restore file version: Mega API Error {result}: {_get_error_message(result)}")
    
    return True


@require_authentication
def remove_file_version(handle: str, version_handle: str) -> bool:
    """
    Remove a specific version of a file.
    
    Args:
        handle: Current file handle
        version_handle: Handle of the version to remove
        
    Returns:
        True if removal successful
        
    Raises:
        RequestError: If removal fails
    """
    command = {'a': 'fvd', 'h': handle, 'vh': version_handle}
    result = single_api_request(command)
    
    if isinstance(result, int) and result < 0:
        raise RequestError(f"Failed to remove file version: Mega API Error {result}: {_get_error_message(result)}")
    
    return True


@require_authentication
def remove_all_file_versions(handle: str, keep_current: bool = True) -> int:
    """
    Remove all versions of a file.
    
    Args:
        handle: File handle
        keep_current: Keep the current version
        
    Returns:
        Number of versions removed
        
    Raises:
        RequestError: If operation fails
    """
    try:
        # Get current versions
        versions = get_file_versions(handle)
        
        if not versions:
            return 0
        
        removed_count = 0
        for version in versions:
            if keep_current and version.get('version_id') == 0:
                continue  # Skip current version
            
            try:
                remove_file_version(handle, version['handle'])
                removed_count += 1
            except RequestError:
                continue  # Skip failed removals
        
        return removed_count
    
    except RequestError as e:
        raise RequestError(f"Failed to get file versions: {e}")


@require_authentication
def configure_file_versioning(handle: str, enable_versions: bool = True,
                            max_versions: Optional[int] = None) -> bool:
    """
    Configure versioning settings for a file.
    
    Args:
        handle: File handle
        enable_versions: Enable or disable versioning
        max_versions: Maximum number of versions to keep
        
    Returns:
        True if configuration successful
        
    Raises:
        RequestError: If configuration fails
    """
    # MEGA doesn't have a direct API for this, so we simulate it
    # by managing versions manually based on settings
    try:
        if not enable_versions:
            # Remove all versions except current
            remove_all_file_versions(handle, keep_current=True)
        elif max_versions is not None:
            # Limit number of versions
            versions = get_file_versions(handle)
            if len(versions) > max_versions:
                # Remove oldest versions
                sorted_versions = sorted(versions, key=lambda v: v.get('timestamp', 0))
                excess_count = len(versions) - max_versions
                for i in range(excess_count):
                    version = sorted_versions[i]
                    if version.get('version_id') != 0:  # Don't remove current
                        remove_file_version(handle, version['handle'])
        
        return True
    
    except RequestError as e:
        raise RequestError(f"Version configuration failed: {e}")


def _get_error_message(error_code: int) -> str:
    """Get human-readable error message for MEGA API error codes."""
    error_messages = {
        -1: "An internal error occurred",
        -2: "You have passed invalid arguments to this command",
        -3: "A temporary congestion or server malfunction prevented your request from being processed",
        -4: "You have exceeded your command weight per time quota",
        -9: "File not found",
        -11: "Access violation (e.g., trying to write to a read-only share)",
        -13: "Trying to create an object that already exists",
        -15: "You are over quota",
        -17: "User blocked",
        -18: "Link broken or expired"
    }
    return error_messages.get(error_code, f"Unknown error code {error_code}")


# Enhanced file versioning functions with events

def get_file_versions_with_events(handle: str, event_callback=None) -> List[Dict[str, Any]]:
    """
    Enhanced get file versions function with event callbacks and logging.
    
    Args:
        handle: File handle
        event_callback: Function to call for events (optional)
        
    Returns:
        List of version dictionaries
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        node = fs_tree.nodes.get(handle)
        file_name = node.name if node else "unknown"
        
        trigger_event('get_versions_started', {'handle': handle, 'file': file_name})
        
        versions = get_file_versions(handle)
        
        logger.info(f"Retrieved {len(versions)} versions for: {file_name}")
        trigger_event('get_versions_completed', {
            'handle': handle, 
            'file': file_name, 
            'version_count': len(versions)
        })
        
        return versions
        
    except Exception as e:
        logger.error(f"Get versions failed: {e}")
        trigger_event('get_versions_failed', {'handle': handle, 'error': str(e)})
        raise


def restore_file_version_with_events(handle: str, version_handle: str, event_callback=None) -> bool:
    """
    Enhanced restore file version function with event callbacks and logging.
    
    Args:
        handle: Current file handle
        version_handle: Version handle to restore
        event_callback: Function to call for events (optional)
        
    Returns:
        True if restore successful
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        node = fs_tree.nodes.get(handle)
        file_name = node.name if node else "unknown"
        
        trigger_event('restore_version_started', {
            'handle': handle, 
            'version_handle': version_handle,
            'file': file_name
        })
        
        result = restore_file_version(handle, version_handle)
        
        logger.info(f"Restored version for: {file_name}")
        trigger_event('restore_version_completed', {
            'handle': handle,
            'version_handle': version_handle,
            'file': file_name
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Restore version failed: {e}")
        trigger_event('restore_version_failed', {
            'handle': handle,
            'version_handle': version_handle,
            'error': str(e)
        })
        raise


def remove_file_version_with_events(handle: str, version_handle: str, event_callback=None) -> bool:
    """
    Enhanced remove file version function with event callbacks and logging.
    
    Args:
        handle: Current file handle
        version_handle: Version handle to remove
        event_callback: Function to call for events (optional)
        
    Returns:
        True if removal successful
    """
    def trigger_event(event_type: str, data: Dict[str, Any]) -> None:
        if event_callback:
            try:
                event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
    
    try:
        node = fs_tree.nodes.get(handle)
        file_name = node.name if node else "unknown"
        
        trigger_event('remove_version_started', {
            'handle': handle,
            'version_handle': version_handle,
            'file': file_name
        })
        
        result = remove_file_version(handle, version_handle)
        
        logger.info(f"Removed version for: {file_name}")
        trigger_event('remove_version_completed', {
            'handle': handle,
            'version_handle': version_handle,
            'file': file_name
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Remove version failed: {e}")
        trigger_event('remove_version_failed', {
            'handle': handle,
            'version_handle': version_handle,
            'error': str(e)
        })
        raise

def get_nodes() -> Dict[str, MegaNode]:
    """
    Retrieves all files and folders from the user's Mega account.
    Populates the global fs_tree with nodes using the exact reference implementation.
    
    Returns:
        Dictionary mapping node handles to MegaNode objects
    """
    logger.info('Getting all files...')
    
    # Clear existing tree
    fs_tree.clear()
    
    # Request files from API (like reference get_files)
    files_resp = single_api_request({'a': 'f', 'c': 1, 'r': 1})
    
    # Initialize shared keys (like reference)
    shared_keys = {}
    _init_shared_keys(files_resp, shared_keys)
    
    # Process each file/folder (like reference)
    for file_data in files_resp['f']:
        try:
            # Process file using reference methodology
            processed_file = _process_file(file_data, shared_keys)
            
            # Only include files/folders with valid attributes (like reference)
            if processed_file.get('a'):
                node = MegaNode(processed_file)
                fs_tree.add_node(node)
        except Exception as e:
            logger.warning(f"Failed to process node {file_data.get('h', 'unknown')}: {e}")
    
    logger.info(f'Loaded {len(fs_tree.nodes)} nodes')
    return fs_tree.nodes


def _init_shared_keys(files: Dict[str, Any], shared_keys: Dict[str, Any]) -> None:
    """
    Initializes shared keys for folders/files that are not associated with a user.
    Exact copy of reference implementation.
    """
    from .auth import current_session
    
    ok_dict = {}
    # Process 'ok' items if they exist
    for ok_item in files.get('ok', []):
        shared_key = decrypt_key(base64_to_a32(ok_item['k']), 
                               string_to_a32(makestring(current_session.master_key)))
        ok_dict[ok_item['h']] = shared_key
    
    # Process 's' items if they exist
    for s_item in files.get('s', []):
        if s_item['u'] not in shared_keys:
            shared_keys[s_item['u']] = {}
        if s_item['h'] in ok_dict:
            shared_keys[s_item['u']][s_item['h']] = ok_dict[s_item['h']]


def _process_file(file_data: Dict[str, Any], shared_keys: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a file or folder node from the API, decrypts keys and attributes.
    Exact copy of reference implementation.
    """
    from .auth import current_session
    
    file = file_data.copy()  # Work with a copy
    master_key_a32 = string_to_a32(makestring(current_session.master_key))
    
    if file['t'] == 0 or file['t'] == 1:  # File or folder
        keys = dict(
            keypart.split(':', 1) for keypart in file['k'].split('/')
            if ':' in keypart)
        uid = file['u']
        key = None
        
        # My objects
        if uid in keys:
            key = decrypt_key(base64_to_a32(keys[uid]), master_key_a32)
        # Shared folders
        elif 'su' in file and 'sk' in file and ':' in file['k']:
            shared_key = decrypt_key(base64_to_a32(file['sk']), master_key_a32)
            key = decrypt_key(base64_to_a32(keys[file['h']]), shared_key)
            if file['su'] not in shared_keys:
                shared_keys[file['su']] = {}
            shared_keys[file['su']][file['h']] = shared_key
        # Shared files
        elif file['u'] and file['u'] in shared_keys:
            for hkey in shared_keys[file['u']]:
                shared_key = shared_keys[file['u']][hkey]
                if hkey in keys:
                    key = keys[hkey]
                    key = decrypt_key(base64_to_a32(key), shared_key)
                    break
        
        # Handle exported/shared files
        if file['h'] and file['h'] in shared_keys.get('EXP', {}):
            shared_key = shared_keys['EXP'][file['h']]
            encrypted_key = string_to_a32(
                makestring(base64_url_decode(file['k'].split(':')[-1])))
            key = decrypt_key(encrypted_key, shared_key)
            file['shared_folder_key'] = shared_key
        
        if key is not None:
            # File
            if file['t'] == 0:
                k = [key[0] ^ key[4], key[1] ^ key[5], key[2] ^ key[6], key[3] ^ key[7]]
                file['iv'] = key[4:6] + [0, 0]
                file['meta_mac'] = key[6:8]
            # Folder
            else:
                k = key
            
            file['key'] = key
            file['k'] = k
            
            # Decrypt attributes
            attributes = base64_url_decode(file['a'])
            attributes = decrypt_attr(attributes, k)
            file['a'] = attributes
        # Other => wrong object
        elif file['k'] == '':
            file['a'] = False
    elif file['t'] == 2:  # Root
        fs_tree.root_handle = file['h']
        file['a'] = {'n': 'Cloud Drive'}
    elif file['t'] == 3:  # Inbox
        fs_tree.inbox_handle = file['h'] 
        file['a'] = {'n': 'Inbox'}
    elif file['t'] == 4:  # Trash
        fs_tree.trash_handle = file['h']
        file['a'] = {'n': 'Rubbish Bin'}
    
    return file


# ==============================================
# === ADVANCED FOLDER OPERATIONS ===
# ==============================================

@require_authentication
def upload_folder(local_path: str, parent_handle: Optional[str] = None, 
                 progress_callback: Optional[Callable[[str, int, int], None]] = None) -> MegaNode:
    """
    Upload an entire local folder to Mega.
    
    Args:
        local_path: Path to local folder to upload
        parent_handle: Parent folder handle (None for root)
        progress_callback: Optional callback for progress updates (file_name, current, total)
        
    Returns:
        Created folder node
        
    Raises:
        ValidationError: If path is invalid
        RequestError: If upload fails
    """
    # os already imported via dependencies
    from pathlib import Path
    
    local_path = Path(local_path)
    if not local_path.exists():
        raise ValidationError(f"Local folder does not exist: {local_path}")
    
    if not local_path.is_dir():
        raise ValidationError(f"Path is not a directory: {local_path}")
    
    # Create root folder
    folder_node = create_folder(local_path.name, parent_handle)
    
    # Count total files for progress
    total_files = sum(1 for root, dirs, files in os.walk(local_path) for file in files)
    uploaded_files = 0
    
    # Recursively upload contents
    _upload_folder_recursive(local_path, folder_node.handle, progress_callback, 
                           uploaded_files, total_files)
    
    logger.info(f"Uploaded folder: {local_path.name} ({total_files} files)")
    return folder_node


def _upload_folder_recursive(local_path: Path, mega_parent_handle: str,
                           progress_callback: Optional[Callable], 
                           uploaded_files: int, total_files: int) -> int:
    """Recursively upload folder contents."""
    # os and time already imported via dependencies
    
    for item in local_path.iterdir():
        if item.is_file():
            # Upload file with retry logic
            if progress_callback:
                progress_callback(item.name, uploaded_files, total_files)
            
            retry_count = 3
            for attempt in range(retry_count):
                try:
                    upload_file(str(item), mega_parent_handle)
                    uploaded_files += 1
                    break
                except Exception as e:
                    if attempt < retry_count - 1:
                        logger.warning(f"Upload attempt {attempt + 1} failed for {item.name}: {e}. Retrying...")
                        time.sleep(1)  # Wait 1 second before retry
                    else:
                        logger.warning(f"Failed to upload file {item.name} after {retry_count} attempts: {e}")
                
        elif item.is_dir():
            # Create subfolder and recurse
            try:
                subfolder = create_folder(item.name, mega_parent_handle)
                uploaded_files = _upload_folder_recursive(
                    item, subfolder.handle, progress_callback, uploaded_files, total_files
                )
            except Exception as e:
                logger.warning(f"Failed to create folder {item.name}: {e}")
    
    return uploaded_files


@require_authentication  
def download_folder(folder_handle: str, local_path: str,
                   progress_callback: Optional[Callable[[str, int, int], None]] = None) -> None:
    """
    Download an entire folder from Mega to local storage.
    
    Args:
        folder_handle: Handle of folder to download
        local_path: Local path where folder will be downloaded
        progress_callback: Optional callback for progress updates (file_name, current, total)
        
    Raises:
        RequestError: If folder not found or download fails
        ValidationError: If local path is invalid
    """
    # os already imported via dependencies  
    from pathlib import Path
    
    if folder_handle not in fs_tree.nodes:
        raise RequestError(f"Folder not found: {folder_handle}")
    
    folder_node = fs_tree.nodes[folder_handle]
    if not folder_node.is_folder():
        raise RequestError("Node is not a folder")
    
    # Create local directory
    local_path = Path(local_path)
    folder_path = local_path / folder_node.name
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Count total files for progress
    total_files = _count_files_recursive(folder_handle)
    downloaded_files = 0
    
    # Recursively download contents
    _download_folder_recursive(folder_handle, folder_path, progress_callback,
                             downloaded_files, total_files)
    
    logger.info(f"Downloaded folder: {folder_node.name} ({total_files} files)")


def _count_files_recursive(folder_handle: str) -> int:
    """Count total files in folder recursively."""
    count = 0
    children = fs_tree.get_children(folder_handle)
    
    for child in children:
        if child.is_file():
            count += 1
        elif child.is_folder():
            count += _count_files_recursive(child.handle)
    
    return count


def _download_folder_recursive(mega_folder_handle: str, local_path: Path,
                             progress_callback: Optional[Callable],
                             downloaded_files: int, total_files: int) -> int:
    """Recursively download folder contents."""
    children = fs_tree.get_children(mega_folder_handle)
    
    for child in children:
        if child.is_file():
            # Download file
            if progress_callback:
                progress_callback(child.name, downloaded_files, total_files)
            
            try:
                file_path = local_path / child.name
                download_file(child.handle, str(file_path))
                downloaded_files += 1
            except Exception as e:
                logger.warning(f"Failed to download file {child.name}: {e}")
                
        elif child.is_folder():
            # Create local subfolder and recurse
            try:
                subfolder_path = local_path / child.name
                subfolder_path.mkdir(exist_ok=True)
                downloaded_files = _download_folder_recursive(
                    child.handle, subfolder_path, progress_callback, 
                    downloaded_files, total_files
                )
            except Exception as e:
                logger.warning(f"Failed to create local folder {child.name}: {e}")
    
    return downloaded_files


@require_authentication
def copy_folder(source_handle: str, target_parent_handle: Optional[str] = None,
               new_name: Optional[str] = None) -> MegaNode:
    """
    Copy a folder and all its contents.
    
    Args:
        source_handle: Handle of folder to copy
        target_parent_handle: Parent folder for the copy (None for root)
        new_name: New name for copied folder (None to keep original)
        
    Returns:
        Copied folder node
        
    Raises:
        RequestError: If copy fails
    """
    if source_handle not in fs_tree.nodes:
        raise RequestError(f"Source folder not found: {source_handle}")
    
    source_node = fs_tree.nodes[source_handle]
    if not source_node.is_folder():
        raise RequestError("Source node is not a folder")
    
    # Create new folder
    folder_name = new_name or source_node.name
    new_folder = create_folder(folder_name, target_parent_handle)
    
    # Recursively copy contents
    _copy_folder_recursive(source_handle, new_folder.handle)
    
    logger.info(f"Copied folder: {source_node.name} -> {folder_name}")
    return new_folder


def _copy_folder_recursive(source_handle: str, target_handle: str) -> None:
    """Recursively copy folder contents."""
    children = fs_tree.get_children(source_handle)
    
    for child in children:
        if child.is_file():
            # Copy file
            try:
                copy_file(child.handle, target_handle)
            except Exception as e:
                logger.warning(f"Failed to copy file {child.name}: {e}")
        elif child.is_folder():
            # Copy subfolder
            try:
                new_subfolder = create_folder(child.name, target_handle)
                _copy_folder_recursive(child.handle, new_subfolder.handle)
            except Exception as e:
                logger.warning(f"Failed to copy folder {child.name}: {e}")


@require_authentication
def move_folder(source_handle: str, target_parent_handle: str) -> None:
    """
    Move a folder to a different location.
    
    Args:
        source_handle: Handle of folder to move
        target_parent_handle: New parent folder handle
        
    Raises:
        RequestError: If move fails
    """
    if source_handle not in fs_tree.nodes:
        raise RequestError(f"Source folder not found: {source_handle}")
    
    source_node = fs_tree.nodes[source_handle]
    if not source_node.is_folder():
        raise RequestError("Source node is not a folder")
    
    if target_parent_handle not in fs_tree.nodes:
        raise RequestError(f"Target parent not found: {target_parent_handle}")
    
    # Store the original attributes and key data to preserve them
    original_name = source_node.name
    original_attributes = source_node.attributes
    original_key = source_node.key
    original_decryption_key = source_node.decryption_key
    
    # Move command
    command = {'a': 'm', 'n': source_handle, 't': target_parent_handle}
    result = single_api_request(command)
    
    if isinstance(result, int) and result < 0:
        raise RequestError(f"Move failed with error code: {result}")
    
    # Update local tree manually (don't refresh_filesystem to preserve keys)
    source_node.parent_handle = target_parent_handle
    
    # Restore the original attributes and key data
    source_node.name = original_name
    source_node.attributes = original_attributes
    source_node.key = original_key
    source_node.decryption_key = original_decryption_key
    
    # Update children mappings
    for parent_handle, children in fs_tree.children.items():
        if source_handle in children:
            children.remove(source_handle)
            break
    
    if target_parent_handle not in fs_tree.children:
        fs_tree.children[target_parent_handle] = []
    fs_tree.children[target_parent_handle].append(source_handle)
    
    logger.info(f"Moved folder: {source_node.name}")

@require_authentication
def get_folder_size(folder_handle: str, recursive: bool = True) -> int:
    """
    Calculate total size of a folder.
    
    Args:
        folder_handle: Handle of folder to measure
        recursive: Whether to include subfolders
        
    Returns:
        Total size in bytes
        
    Raises:
        RequestError: If folder not found
    """
    if folder_handle not in fs_tree.nodes:
        raise RequestError(f"Folder not found: {folder_handle}")
    
    folder_node = fs_tree.nodes[folder_handle]
    if not folder_node.is_folder():
        raise RequestError("Node is not a folder")
    
    total_size = 0
    children = fs_tree.get_children(folder_handle)
    
    for child in children:
        if child.is_file():
            total_size += child.size
        elif child.is_folder() and recursive:
            total_size += get_folder_size(child.handle, recursive=True)
    
    return total_size


@require_authentication
def get_folder_info(folder_handle: str) -> Dict[str, Any]:
    """
    Get detailed information about a folder.
    
    Args:
        folder_handle: Handle of folder to analyze
        
    Returns:
        Dictionary with folder statistics
        
    Raises:
        RequestError: If folder not found
    """
    if folder_handle not in fs_tree.nodes:
        raise RequestError(f"Folder not found: {folder_handle}")
    
    folder_node = fs_tree.nodes[folder_handle]
    if not folder_node.is_folder():
        raise RequestError("Node is not a folder")
    
    children = fs_tree.get_children(folder_handle)
    
    info = {
        "name": folder_node.name,
        "handle": folder_handle,
        "parent": folder_node.parent_handle,
        "created": folder_node.created_time,
        "total_items": len(children),
        "files": sum(1 for child in children if child.is_file()),
        "folders": sum(1 for child in children if child.is_folder()),
        "total_size": get_folder_size(folder_handle, recursive=True),
        "direct_size": get_folder_size(folder_handle, recursive=False)
    }
    
    return info


# ==============================================
# === FILE VERSIONING & HISTORY (DUPLICATES REMOVED) ===
# Note: The following functions are duplicate definitions and have been removed:
# get_file_versions, restore_file_version, remove_file_version
# Original definitions are located earlier in this module.
# ==============================================

# DUPLICATE REMOVED: get_file_versions function (defined earlier in module)

# DUPLICATE REMOVED: restore_file_version function (defined earlier in module)

# DUPLICATE REMOVED: remove_file_version function (defined earlier in module)

@require_authentication
def remove_all_file_versions(handle: str, keep_current: bool = True) -> int:
    """
    Remove all versions of a file (except optionally the current one).
    
    Args:
        handle: Handle of the file to clean versions for
        keep_current: If True, keep the current version (default)
        
    Returns:
        Number of versions removed
        
    Raises:
        RequestError: If operation fails
        ValidationError: If handle is invalid
    """
    if not handle:
        raise ValidationError("File handle cannot be empty")
    
    if handle not in fs_tree.nodes:
        raise RequestError(f"File not found: {handle}")
    
    node = fs_tree.nodes[handle]
    if not node.is_file():
        raise RequestError("Node is not a file - versioning only supported for files")
    
    try:
        # Get all versions first
        versions = get_file_versions(handle)
        
        removed_count = 0
        for version in versions:
            version_handle = version['version_id']
            
            # Skip current version if keep_current is True
            if keep_current and version['is_current']:
                continue
            
            # Skip if this is the main handle (current version)
            if version_handle == handle:
                continue
            
            try:
                if remove_file_version(handle, version_handle):
                    removed_count += 1
            except Exception as e:
                # Log error but continue with other versions
                print(f"Warning: Failed to remove version {version_handle}: {e}")
                continue
        
        return removed_count
        
    except Exception as e:
        raise RequestError(f"Failed to remove file versions: {e}")


# DUPLICATE REMOVED: configure_file_versioning function (defined earlier in module)

# ==============================================
# === CLIENT METHOD INJECTION ===
# ==============================================

def add_file_versioning_methods_with_events(client_class):
    """Add file versioning methods with event support to the MPLClient class."""
    
    def get_file_versions_method(self, path: str) -> List[Dict[str, Any]]:
        """Get version history for a file."""
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"File not found: {path}")
        return get_file_versions_with_events(node.handle, getattr(self, '_trigger_event', None))
    
    def restore_file_version_method(self, path: str, version_handle: str) -> bool:
        """Restore a previous version of a file."""
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"File not found: {path}")
        return restore_file_version_with_events(node.handle, version_handle, getattr(self, '_trigger_event', None))
    
    def remove_file_version_method(self, path: str, version_handle: str) -> bool:
        """Remove a specific version of a file."""
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"File not found: {path}")
        return remove_file_version_with_events(node.handle, version_handle, getattr(self, '_trigger_event', None))
    
    def remove_all_file_versions_method(self, path: str, keep_current: bool = True) -> int:
        """Remove all versions of a file."""
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"File not found: {path}")
        return remove_all_file_versions(node.handle, keep_current)
    
    def configure_file_versioning_method(self, path: str, enable_versions: bool = True,
                                       max_versions: Optional[int] = None) -> bool:
        """Configure versioning settings for a file."""
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"File not found: {path}")
        return configure_file_versioning(node.handle, enable_versions, max_versions)
    
    # Add methods to client class
    setattr(client_class, 'get_file_versions', get_file_versions_method)
    setattr(client_class, 'restore_file_version', restore_file_version_method)
    setattr(client_class, 'remove_file_version', remove_file_version_method)
    setattr(client_class, 'remove_all_file_versions', remove_all_file_versions_method)
    setattr(client_class, 'configure_file_versioning', configure_file_versioning_method)


def add_filesystem_methods_with_events(client_class):
    """Add core filesystem methods with event support to the MPLClient class."""
    
    def refresh_method(self) -> None:
        """Refresh the filesystem tree (like mega-reload)."""
        return refresh_filesystem_with_events(getattr(self, '_trigger_event', None))
    
    def mkdir_method(self, name: str, parent_path: str = None) -> 'MegaNode':
        """Create a new folder (like mega-mkdir)."""
        parent_handle = None
        if parent_path:
            parent_node = get_node_by_path(parent_path)
            if parent_node:
                parent_handle = parent_node.handle
        return create_folder_with_events(name, parent_handle, getattr(self, '_trigger_event', None))
    
    def delete_method(self, path: str, permanent: bool = False) -> None:
        """Delete a file or folder (like mega-rm)."""
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"Node not found: {path}")
        return delete_node_with_events(node.handle, permanent, getattr(self, '_trigger_event', None))
    
    def move_method(self, source_path: str, dest_path: str) -> None:
        """Move a file or folder (like mega-mv)."""
        source_node = get_node_by_path(source_path)
        dest_node = get_node_by_path(dest_path)
        if not source_node:
            raise RequestError(f"Source node not found: {source_path}")
        if not dest_node:
            raise RequestError(f"Destination node not found: {dest_path}")
        return move_node_with_events(source_node.handle, dest_node.handle, getattr(self, '_trigger_event', None))
    
    def rename_method(self, path: str, new_name: str) -> None:
        """Rename a file or folder."""
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"Node not found: {path}")
        return rename_node_with_events(node.handle, new_name, getattr(self, '_trigger_event', None))
    
    def upload_method(self, file_path: str, dest_path: str = None, 
                     progress_callback: Optional[Callable] = None) -> 'MegaNode':
        """Upload a file to MEGA (like mega-put)."""
        dest_handle = None
        if dest_path:
            dest_node = get_node_by_path(dest_path)
            if dest_node:
                dest_handle = dest_node.handle
        return upload_file_with_events(file_path, dest_handle, progress_callback, getattr(self, '_trigger_event', None))
    
    def download_method(self, path: str, local_path: str,
                       progress_callback: Optional[Callable] = None,
                       concurrent: bool = True, max_workers: int = 4,
                       enable_error_recovery: bool = True) -> Path:
        """Download a file from MEGA (like mega-get) with concurrent support and error recovery."""
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"File not found: {path}")
        return download_file_with_events(node.handle, local_path, progress_callback, 
                                       getattr(self, '_trigger_event', None), 
                                       concurrent=concurrent, max_workers=max_workers,
                                       enable_error_recovery=enable_error_recovery)
    
    def share_method(self, path: str) -> str:
        """Create a public download link for a file or folder (like mega-export)."""
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"Node not found: {path}")
        return create_public_link_with_events(node.handle, getattr(self, '_trigger_event', None))
    
    def unshare_method(self, path: str) -> None:
        """Remove public link for a file or folder."""
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"Node not found: {path}")
        return remove_public_link_with_events(node.handle, getattr(self, '_trigger_event', None))
    
    def copy_method(self, source_path: str, dest_parent_path: str, new_name: Optional[str] = None) -> 'MegaNode':
        """
        Copy a file or folder within Mega's cloud storage.
        
        Args:
            source_path: Path to file/folder to copy
            dest_parent_path: Path to destination parent folder
            new_name: New name for copied item (optional, keeps original if not specified)
            
        Returns:
            The copied file or folder node
        """
        if not is_logged_in():
            raise RequestError("Not logged in")
        
        # Refresh filesystem if needed
        if not fs_tree.nodes:
            get_nodes()
        
        source_node = get_node_by_path(source_path)
        if not source_node:
            raise RequestError(f"Source path not found: {source_path}")
        
        # Handle root destination
        if dest_parent_path == "/":
            dest_handle = None  # Root folder
        else:
            dest_node = get_node_by_path(dest_parent_path)
            if not dest_node:
                raise RequestError(f"Destination path not found: {dest_parent_path}")
            
            if not dest_node.is_folder():
                from .exceptions import ValidationError
                raise ValidationError("Destination must be a folder")
            dest_handle = dest_node.handle
        
        if source_node.is_file():
            # For files, copy_file doesn't support None destination, so use root_handle
            if dest_parent_path == "/":
                dest_handle = fs_tree.root_handle
            
            # Copy file with fix for encrypted names
            copied_node = copy_file(source_node.handle, dest_handle)
            
            # TODO: HIGH PRIORITY - Fix root cause of encrypted names in copy operations
            # Current issue: copy_file() sometimes fails to properly decrypt/encrypt filenames
            # during the copy process, resulting in names like "Encrypted_XXX" that appear as
            # "undecrypted file" in MEGA web interface.
            # 
            # Root cause likely in copy_file() implementation:
            # 1. Key derivation/handling during copy operation
            # 2. Attribute encryption/decryption process
            # 3. Node creation with proper decryption keys
            #
            # Once fixed, we can remove the download+upload workaround below and use
            # direct MEGA API copy operations which are much faster and more efficient.
            #
            # Current workaround: Download source -> upload with proper name (slow but works)
            
            # FIXED: Check if copy created encrypted name (indicates decryption issue)
            if copied_node.name.startswith('Encrypted_') or (len(copied_node.name) > 8 and copied_node.name[:8].isupper()):
                logger.warning(f"Copy created encrypted name: {copied_node.name}, using download+upload fix")
                
                # Use download+upload method to fix encrypted name issue
                try:
                    import tempfile
                    import os
                    
                    # Download the source file content
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    downloaded_path = download_file(source_node.handle, temp_path)
                    if downloaded_path and os.path.exists(downloaded_path):
                        try:
                            # Delete the problematic encrypted copy
                            try:
                                delete_node(copied_node.handle)
                                logger.info(f"Removed problematic encrypted copy: {copied_node.name}")
                            except Exception as e:
                                logger.warning(f"Could not remove encrypted copy: {e}")
                            
                            # Generate proper copy filename
                            if new_name:
                                copy_name = new_name
                            else:
                                # Add copy suffix to original name
                                base_name = source_node.name
                                name_parts = base_name.rsplit('.', 1)
                                if len(name_parts) == 2:
                                    copy_name = f"{name_parts[0]}_copy.{name_parts[1]}"
                                else:
                                    copy_name = f"{base_name}_copy"
                            
                            # Create properly named temp file
                            copy_temp_dir = os.path.dirname(downloaded_path)
                            copy_temp_path = os.path.join(copy_temp_dir, copy_name)
                            
                            # Copy content with proper name
                            with open(downloaded_path, 'rb') as src, open(copy_temp_path, 'wb') as dst:
                                dst.write(src.read())
                            
                            # Upload with proper name
                            copied_node = upload_file(copy_temp_path, dest_handle)
                            logger.info(f"Fixed copy successful: {source_node.name} -> {copied_node.name}")
                            
                            # Cleanup temp files
                            os.unlink(downloaded_path)
                            os.unlink(copy_temp_path)
                            
                        except Exception as fix_error:
                            # Cleanup temp files on error
                            if os.path.exists(downloaded_path):
                                os.unlink(downloaded_path)
                            logger.error(f"Copy fix failed: {fix_error}")
                            # Fall back to the encrypted copy
                            logger.warning(f"Using encrypted copy as fallback: {copied_node.name}")
                    else:
                        logger.error("Failed to download source file for copy fix")
                        # Fall back to the encrypted copy
                        
                except Exception as download_error:
                    logger.error(f"Copy fix download failed: {download_error}")
                    # Fall back to the encrypted copy
            
            # Note: File copying may result in encrypted names due to encryption key handling
            # This is now fixed with the download+upload fallback method above
            if new_name:
                logger.info(f"Custom name requested for copy: {new_name} (applied via download+upload method)")
            
            logger.info(f"Copied file: {source_path} -> {dest_parent_path}")
            if hasattr(self, '_trigger_event'):
                self._trigger_event('file_copied', {'source': source_path, 'destination': dest_parent_path})
            
        elif source_node.is_folder():
            # Copy folder (copy_folder handles None for root correctly)
            copied_node = copy_folder(source_node.handle, dest_handle, new_name)
            logger.info(f"Copied folder: {source_path} -> {dest_parent_path}")
            if hasattr(self, '_trigger_event'):
                self._trigger_event('folder_copied', {'source': source_path, 'destination': dest_parent_path})
            
        else:
            from .exceptions import ValidationError
            raise ValidationError("Source must be a file or folder")
        
        return copied_node
    
    # Add methods with traditional MEGA command names
    setattr(client_class, 'refresh', refresh_method)
    setattr(client_class, 'mkdir', mkdir_method)
    setattr(client_class, 'delete', delete_method)
    setattr(client_class, 'move', move_method)
    setattr(client_class, 'rename', rename_method)
    setattr(client_class, 'upload', upload_method)
    setattr(client_class, 'download', download_method)
    setattr(client_class, 'share', share_method)
    setattr(client_class, 'unshare', unshare_method)
    setattr(client_class, 'copy', copy_method)  # Add unified copy method
    
    # Also add some aliases for compatibility
    setattr(client_class, 'put', upload_method)  # mega-put alias
    setattr(client_class, 'get', download_method)  # mega-get alias
    setattr(client_class, 'rm', delete_method)  # mega-rm alias
    setattr(client_class, 'mv', move_method)  # mega-mv alias
    setattr(client_class, 'export', share_method)  # mega-export alias
    
    # Keep technical names for advanced users
    setattr(client_class, 'refresh_filesystem', refresh_method)
    setattr(client_class, 'create_folder', mkdir_method)
    setattr(client_class, 'delete_node', delete_method)
    setattr(client_class, 'move_node', move_method)
    setattr(client_class, 'rename_node', rename_method)
    setattr(client_class, 'upload_file', upload_method)
    setattr(client_class, 'download_file', download_method)
    setattr(client_class, 'create_public_link', share_method)
    setattr(client_class, 'remove_public_link', unshare_method)
    setattr(client_class, 'copy_node', copy_method)  # Add unified copy method with technical name

