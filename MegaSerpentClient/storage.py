"""
MegaSerpentClient - Storage & File Operations Module

Purpose: All file and directory operations, metadata management, and file system navigation.

This module handles complete file operations (all MEGAcmd file operations), navigation system,
metadata management, Cloud RAID, backup system, and collections management.
"""

import os
import stat
import hashlib
import mimetypes
import threading
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple, BinaryIO, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import weakref

from . import utils
from .utils import (
    Constants, FileType, StorageError, ValidationError, MegaError,
    FileUtils, StringUtils, DateTimeUtils, Helpers, Formatters
)


# ==============================================
# === STORAGE ENUMS AND CONSTANTS ===
# ==============================================

class NodeType(Enum):
    """Node type enumeration."""
    FILE = 0
    FOLDER = 1
    ROOT = 2
    INBOX = 3
    TRASH = 4
    BACKUP = 5


class PermissionLevel(Enum):
    """Permission level enumeration."""
    NONE = 0
    READ = 1
    READ_WRITE = 2
    FULL = 3


class SearchFilter(Enum):
    """Search filter types."""
    NAME = "name"
    EXTENSION = "extension"
    SIZE = "size"
    DATE_CREATED = "date_created"
    DATE_MODIFIED = "date_modified"
    MIME_TYPE = "mime_type"


class SortOrder(Enum):
    """Sort order enumeration."""
    ASC = "asc"
    DESC = "desc"


class ConflictResolution(Enum):
    """File conflict resolution strategies."""
    SKIP = "skip"
    OVERWRITE = "overwrite"
    RENAME = "rename"
    MERGE = "merge"


# ==============================================
# === DATA CLASSES ===
# ==============================================

@dataclass
class NodeInfo:
    """Cloud node information."""
    node_id: str
    name: str
    node_type: NodeType
    size: int = 0
    parent_id: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    mime_type: Optional[str] = None
    checksum: Optional[str] = None
    permissions: PermissionLevel = PermissionLevel.READ_WRITE
    attributes: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_shared: bool = False
    share_id: Optional[str] = None
    version: int = 1


@dataclass
class SearchQuery:
    """Search query parameters."""
    query: str
    filters: Dict[SearchFilter, Any] = field(default_factory=dict)
    sort_by: SearchFilter = SearchFilter.NAME
    sort_order: SortOrder = SortOrder.ASC
    limit: Optional[int] = None
    offset: int = 0
    recursive: bool = True


@dataclass
class TransferProgress:
    """File transfer progress information."""
    transfer_id: str
    file_name: str
    total_size: int
    transferred: int = 0
    speed: float = 0.0  # bytes per second
    eta: Optional[float] = None  # seconds
    status: str = "pending"
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class BackupConfig:
    """Backup configuration."""
    backup_id: str
    name: str
    source_path: str
    target_folder_id: str
    schedule: Optional[str] = None  # cron expression
    enabled: bool = True
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    compression: bool = False
    encryption: bool = True
    retention_days: Optional[int] = None


# ==============================================
# === PATH AND NAVIGATION ===
# ==============================================

class PathResolver:
    """Path resolution and normalization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def normalize_path(self, path: str) -> str:
        """Normalize cloud path."""
        if not path:
            return "/"
        
        # Ensure path starts with /
        if not path.startswith('/'):
            path = '/' + path
        
        # Remove double slashes
        while '//' in path:
            path = path.replace('//', '/')
        
        # Remove trailing slash unless root
        if len(path) > 1 and path.endswith('/'):
            path = path[:-1]
        
        return path
    
    def join_paths(self, *paths: str) -> str:
        """Join multiple path components."""
        result = ""
        for path in paths:
            if not path:
                continue
            
            path = path.strip('/')
            if path:
                if result and not result.endswith('/'):
                    result += '/'
                result += path
        
        return self.normalize_path(result or "/")
    
    def get_parent_path(self, path: str) -> str:
        """Get parent path."""
        normalized = self.normalize_path(path)
        if normalized == "/":
            return "/"
        
        parent = "/".join(normalized.split("/")[:-1])
        return parent if parent else "/"
    
    def get_file_name(self, path: str) -> str:
        """Get file name from path."""
        normalized = self.normalize_path(path)
        if normalized == "/":
            return ""
        
        return normalized.split("/")[-1]
    
    def split_path(self, path: str) -> List[str]:
        """Split path into components."""
        normalized = self.normalize_path(path)
        if normalized == "/":
            return []
        
        return [part for part in normalized.split("/") if part]


class Navigator:
    """Directory navigation functionality."""
    
    def __init__(self, node_manager):
        self.node_manager = node_manager
        self.path_resolver = PathResolver()
        self._current_path = "/"
        self.logger = logging.getLogger(__name__)
    
    def cd(self, path: str) -> bool:
        """Change current directory."""
        if path == "..":
            # Go to parent directory
            parent_path = self.path_resolver.get_parent_path(self._current_path)
            if self.node_manager.node_exists(parent_path):
                self._current_path = parent_path
                return True
            return False
        
        # Resolve path
        if not path.startswith('/'):
            # Relative path
            new_path = self.path_resolver.join_paths(self._current_path, path)
        else:
            # Absolute path
            new_path = self.path_resolver.normalize_path(path)
        
        # Check if path exists and is a directory
        node = self.node_manager.get_node_by_path(new_path)
        if node and node.node_type == NodeType.FOLDER:
            self._current_path = new_path
            self.logger.info(f"Changed directory to: {new_path}")
            return True
        
        return False
    
    def pwd(self) -> str:
        """Get current working directory."""
        return self._current_path
    
    def ls(self, path: Optional[str] = None, show_hidden: bool = False,
           detailed: bool = False) -> List[NodeInfo]:
        """List directory contents."""
        target_path = path if path else self._current_path
        target_path = self.path_resolver.normalize_path(target_path)
        
        return self.node_manager.list_children(target_path, show_hidden=show_hidden)
    
    def tree(self, path: Optional[str] = None, max_depth: int = 3) -> Dict[str, Any]:
        """Generate directory tree."""
        target_path = path if path else self._current_path
        target_path = self.path_resolver.normalize_path(target_path)
        
        return self._build_tree(target_path, 0, max_depth)
    
    def _build_tree(self, path: str, current_depth: int, max_depth: int) -> Dict[str, Any]:
        """Recursively build directory tree."""
        if current_depth >= max_depth:
            return {}
        
        node = self.node_manager.get_node_by_path(path)
        if not node:
            return {}
        
        tree = {
            'name': node.name,
            'type': node.node_type.name,
            'size': node.size,
            'children': {}
        }
        
        if node.node_type == NodeType.FOLDER:
            children = self.node_manager.list_children(path)
            for child in children:
                child_path = self.path_resolver.join_paths(path, child.name)
                tree['children'][child.name] = self._build_tree(
                    child_path, current_depth + 1, max_depth
                )
        
        return tree


class LocalNavigator:
    """Local directory navigation."""
    
    def __init__(self):
        self._current_local_path = os.getcwd()
        self.logger = logging.getLogger(__name__)
    
    def lcd(self, path: str) -> bool:
        """Change local directory."""
        try:
            new_path = os.path.abspath(os.path.expanduser(path))
            if os.path.isdir(new_path):
                os.chdir(new_path)
                self._current_local_path = new_path
                self.logger.info(f"Changed local directory to: {new_path}")
                return True
        except OSError as e:
            self.logger.error(f"Failed to change local directory: {e}")
        
        return False
    
    def lpwd(self) -> str:
        """Get current local working directory."""
        return self._current_local_path
    
    def lls(self, path: Optional[str] = None, show_hidden: bool = False) -> List[Dict[str, Any]]:
        """List local directory contents."""
        target_path = path if path else self._current_local_path
        target_path = os.path.abspath(os.path.expanduser(target_path))
        
        items = []
        try:
            for item in os.listdir(target_path):
                if not show_hidden and item.startswith('.'):
                    continue
                
                item_path = os.path.join(target_path, item)
                stat_info = os.stat(item_path)
                
                items.append({
                    'name': item,
                    'type': 'directory' if os.path.isdir(item_path) else 'file',
                    'size': stat_info.st_size,
                    'modified': datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc),
                    'permissions': oct(stat_info.st_mode)[-3:]
                })
        
        except OSError as e:
            self.logger.error(f"Failed to list local directory: {e}")
        
        return items


# ==============================================
# === NODE MANAGEMENT ===
# ==============================================

class NodeManager:
    """Cloud node CRUD operations."""
    
    def __init__(self):
        self._nodes: Dict[str, NodeInfo] = {}
        self._path_to_node: Dict[str, str] = {}  # path -> node_id
        self._node_children: Dict[str, List[str]] = {}  # node_id -> child_node_ids
        self._lock = threading.Lock()
        self.path_resolver = PathResolver()
        self.logger = logging.getLogger(__name__)
        
        # Create root node
        self._create_root_node()
    
    def _create_root_node(self):
        """Create root node."""
        root_node = NodeInfo(
            node_id="root",
            name="",
            node_type=NodeType.ROOT,
            created_at=DateTimeUtils.now_utc(),
            modified_at=DateTimeUtils.now_utc()
        )
        
        with self._lock:
            self._nodes["root"] = root_node
            self._path_to_node["/"] = "root"
            self._node_children["root"] = []
    
    def create_node(self, name: str, node_type: NodeType, parent_path: str = "/",
                   size: int = 0, mime_type: Optional[str] = None,
                   attributes: Optional[Dict[str, Any]] = None) -> NodeInfo:
        """Create new node."""
        # Validate name
        if not name or not StringUtils.sanitize_filename(name):
            raise ValidationError("Invalid node name")
        
        # Get parent node
        parent_node = self.get_node_by_path(parent_path)
        if not parent_node:
            raise StorageError(f"Parent path not found: {parent_path}")
        
        if parent_node.node_type not in [NodeType.FOLDER, NodeType.ROOT]:
            raise StorageError("Parent must be a folder")
        
        # Generate node ID and path
        node_id = Helpers.generate_request_id()
        node_path = self.path_resolver.join_paths(parent_path, name)
        
        # Check if node already exists
        if self.node_exists(node_path):
            raise StorageError(f"Node already exists: {node_path}")
        
        # Create node
        node = NodeInfo(
            node_id=node_id,
            name=name,
            node_type=node_type,
            size=size,
            parent_id=parent_node.node_id,
            created_at=DateTimeUtils.now_utc(),
            modified_at=DateTimeUtils.now_utc(),
            mime_type=mime_type,
            attributes=attributes or {}
        )
        
        with self._lock:
            self._nodes[node_id] = node
            self._path_to_node[node_path] = node_id
            
            # Add to parent's children
            if parent_node.node_id not in self._node_children:
                self._node_children[parent_node.node_id] = []
            self._node_children[parent_node.node_id].append(node_id)
            
            # Initialize children list for folders
            if node_type == NodeType.FOLDER:
                self._node_children[node_id] = []
        
        self.logger.info(f"Created {node_type.name.lower()}: {node_path}")
        return node
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeInfo]:
        """Get node by ID."""
        with self._lock:
            return self._nodes.get(node_id)
    
    def get_node_by_path(self, path: str) -> Optional[NodeInfo]:
        """Get node by path."""
        normalized_path = self.path_resolver.normalize_path(path)
        
        with self._lock:
            node_id = self._path_to_node.get(normalized_path)
            return self._nodes.get(node_id) if node_id else None
    
    def node_exists(self, path: str) -> bool:
        """Check if node exists at path."""
        return self.get_node_by_path(path) is not None
    
    def list_children(self, path: str, show_hidden: bool = False) -> List[NodeInfo]:
        """List child nodes."""
        parent_node = self.get_node_by_path(path)
        if not parent_node:
            return []
        
        children = []
        with self._lock:
            child_ids = self._node_children.get(parent_node.node_id, [])
            
            for child_id in child_ids:
                child_node = self._nodes.get(child_id)
                if child_node:
                    if not show_hidden and child_node.name.startswith('.'):
                        continue
                    children.append(child_node)
        
        return children
    
    def delete_node(self, path: str, recursive: bool = False) -> bool:
        """Delete node."""
        node = self.get_node_by_path(path)
        if not node:
            return False
        
        # Check if folder is empty (unless recursive)
        if node.node_type == NodeType.FOLDER and not recursive:
            children = self.list_children(path)
            if children:
                raise StorageError("Folder is not empty. Use recursive=True to delete.")
        
        self._delete_node_recursive(node)
        self.logger.info(f"Deleted node: {path}")
        return True
    
    def _delete_node_recursive(self, node: NodeInfo):
        """Recursively delete node and children."""
        with self._lock:
            # Delete children first
            child_ids = self._node_children.get(node.node_id, []).copy()
            for child_id in child_ids:
                child_node = self._nodes.get(child_id)
                if child_node:
                    self._delete_node_recursive(child_node)
            
            # Remove from parent's children
            if node.parent_id and node.parent_id in self._node_children:
                if node.node_id in self._node_children[node.parent_id]:
                    self._node_children[node.parent_id].remove(node.node_id)
            
            # Remove from data structures
            if node.node_id in self._nodes:
                del self._nodes[node.node_id]
            
            if node.node_id in self._node_children:
                del self._node_children[node.node_id]
            
            # Remove path mapping
            node_path = self._find_node_path(node.node_id)
            if node_path and node_path in self._path_to_node:
                del self._path_to_node[node_path]
    
    def move_node(self, source_path: str, target_path: str) -> bool:
        """Move node to new location."""
        source_node = self.get_node_by_path(source_path)
        if not source_node:
            return False
        
        target_parent_path = self.path_resolver.get_parent_path(target_path)
        target_parent = self.get_node_by_path(target_parent_path)
        if not target_parent:
            return False
        
        new_name = self.path_resolver.get_file_name(target_path)
        
        with self._lock:
            # Remove from old path mapping
            if source_path in self._path_to_node:
                del self._path_to_node[source_path]
            
            # Update node
            source_node.name = new_name
            source_node.parent_id = target_parent.node_id
            source_node.modified_at = DateTimeUtils.now_utc()
            
            # Add to new path mapping
            self._path_to_node[target_path] = source_node.node_id
            
            # Update parent relationships
            if source_node.parent_id:
                old_parent_children = self._node_children.get(source_node.parent_id, [])
                if source_node.node_id in old_parent_children:
                    old_parent_children.remove(source_node.node_id)
            
            if target_parent.node_id not in self._node_children:
                self._node_children[target_parent.node_id] = []
            self._node_children[target_parent.node_id].append(source_node.node_id)
        
        self.logger.info(f"Moved node: {source_path} -> {target_path}")
        return True
    
    def copy_node(self, source_path: str, target_path: str) -> Optional[NodeInfo]:
        """Copy node to new location."""
        source_node = self.get_node_by_path(source_path)
        if not source_node:
            return None
        
        target_parent_path = self.path_resolver.get_parent_path(target_path)
        new_name = self.path_resolver.get_file_name(target_path)
        
        # Create copy
        copied_node = self.create_node(
            name=new_name,
            node_type=source_node.node_type,
            parent_path=target_parent_path,
            size=source_node.size,
            mime_type=source_node.mime_type,
            attributes=source_node.attributes.copy()
        )
        
        self.logger.info(f"Copied node: {source_path} -> {target_path}")
        return copied_node
    
    def _find_node_path(self, node_id: str) -> Optional[str]:
        """Find path for node ID."""
        with self._lock:
            for path, nid in self._path_to_node.items():
                if nid == node_id:
                    return path
        return None


# ==============================================
# === FILE OPERATIONS ===
# ==============================================

class FileOperations:
    """Basic file CRUD operations."""
    
    def __init__(self, node_manager: NodeManager):
        self.node_manager = node_manager
        self.logger = logging.getLogger(__name__)
    
    def create_file(self, name: str, parent_path: str = "/", content: bytes = b"") -> NodeInfo:
        """Create new file."""
        mime_type, _ = mimetypes.guess_type(name)
        
        node = self.node_manager.create_node(
            name=name,
            node_type=NodeType.FILE,
            parent_path=parent_path,
            size=len(content),
            mime_type=mime_type
        )
        
        # In a real implementation, content would be stored
        if content:
            checksum = hashlib.sha256(content).hexdigest()
            node.checksum = checksum
        
        return node
    
    def read_file(self, path: str) -> Optional[bytes]:
        """Read file content."""
        node = self.node_manager.get_node_by_path(path)
        if not node or node.node_type != NodeType.FILE:
            return None
        
        # In a real implementation, this would fetch actual content
        self.logger.info(f"Reading file: {path}")
        return b"placeholder content"
    
    def write_file(self, path: str, content: bytes, create_if_missing: bool = True) -> bool:
        """Write file content."""
        node = self.node_manager.get_node_by_path(path)
        
        if not node:
            if create_if_missing:
                parent_path = self.node_manager.path_resolver.get_parent_path(path)
                name = self.node_manager.path_resolver.get_file_name(path)
                node = self.create_file(name, parent_path, content)
                return True
            return False
        
        if node.node_type != NodeType.FILE:
            return False
        
        # Update node metadata
        node.size = len(content)
        node.modified_at = DateTimeUtils.now_utc()
        node.checksum = hashlib.sha256(content).hexdigest()
        
        # In a real implementation, content would be stored
        self.logger.info(f"Written file: {path} ({len(content)} bytes)")
        return True
    
    def append_file(self, path: str, content: bytes) -> bool:
        """Append content to file."""
        node = self.node_manager.get_node_by_path(path)
        if not node or node.node_type != NodeType.FILE:
            return False
        
        # In a real implementation, this would append to actual content
        node.size += len(content)
        node.modified_at = DateTimeUtils.now_utc()
        
        self.logger.info(f"Appended to file: {path} ({len(content)} bytes)")
        return True
    
    def delete_file(self, path: str) -> bool:
        """Delete file."""
        return self.node_manager.delete_node(path)
    
    def move_file(self, source_path: str, target_path: str) -> bool:
        """Move file."""
        return self.node_manager.move_node(source_path, target_path)
    
    def copy_file(self, source_path: str, target_path: str) -> Optional[NodeInfo]:
        """Copy file."""
        return self.node_manager.copy_node(source_path, target_path)
    
    def get_file_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Get file information."""
        node = self.node_manager.get_node_by_path(path)
        if not node or node.node_type != NodeType.FILE:
            return None
        
        return {
            'name': node.name,
            'size': node.size,
            'size_formatted': Formatters.format_file_size(node.size),
            'mime_type': node.mime_type,
            'created_at': node.created_at.isoformat() if node.created_at else None,
            'modified_at': node.modified_at.isoformat() if node.modified_at else None,
            'checksum': node.checksum,
            'is_image': FileUtils.is_image_file(node.name),
            'is_video': FileUtils.is_video_file(node.name),
            'is_audio': FileUtils.is_audio_file(node.name),
            'is_document': FileUtils.is_document_file(node.name)
        }


class DirectoryOperations:
    """Directory operations."""
    
    def __init__(self, node_manager: NodeManager):
        self.node_manager = node_manager
        self.logger = logging.getLogger(__name__)
    
    def create_directory(self, name: str, parent_path: str = "/") -> NodeInfo:
        """Create directory."""
        return self.node_manager.create_node(
            name=name,
            node_type=NodeType.FOLDER,
            parent_path=parent_path
        )
    
    def delete_directory(self, path: str, recursive: bool = False) -> bool:
        """Delete directory."""
        return self.node_manager.delete_node(path, recursive=recursive)
    
    def move_directory(self, source_path: str, target_path: str) -> bool:
        """Move directory."""
        return self.node_manager.move_node(source_path, target_path)
    
    def copy_directory(self, source_path: str, target_path: str, recursive: bool = True) -> Optional[NodeInfo]:
        """Copy directory."""
        if not recursive:
            return self.node_manager.copy_node(source_path, target_path)
        
        # Recursive copy
        source_node = self.node_manager.get_node_by_path(source_path)
        if not source_node or source_node.node_type != NodeType.FOLDER:
            return None
        
        # Create target directory
        target_parent = self.node_manager.path_resolver.get_parent_path(target_path)
        target_name = self.node_manager.path_resolver.get_file_name(target_path)
        
        copied_dir = self.create_directory(target_name, target_parent)
        
        # Copy children recursively
        children = self.node_manager.list_children(source_path)
        for child in children:
            child_source = self.node_manager.path_resolver.join_paths(source_path, child.name)
            child_target = self.node_manager.path_resolver.join_paths(target_path, child.name)
            
            if child.node_type == NodeType.FOLDER:
                self.copy_directory(child_source, child_target, recursive=True)
            else:
                self.node_manager.copy_node(child_source, child_target)
        
        return copied_dir
    
    def get_directory_size(self, path: str, recursive: bool = True) -> int:
        """Get total size of directory."""
        node = self.node_manager.get_node_by_path(path)
        if not node or node.node_type != NodeType.FOLDER:
            return 0
        
        total_size = 0
        children = self.node_manager.list_children(path)
        
        for child in children:
            if child.node_type == NodeType.FILE:
                total_size += child.size
            elif child.node_type == NodeType.FOLDER and recursive:
                child_path = self.node_manager.path_resolver.join_paths(path, child.name)
                total_size += self.get_directory_size(child_path, recursive=True)
        
        return total_size


# ==============================================
# === SEARCH ENGINE ===
# ==============================================

class SearchEngine:
    """File search functionality."""
    
    def __init__(self, node_manager: NodeManager):
        self.node_manager = node_manager
        self.logger = logging.getLogger(__name__)
    
    def search(self, query: SearchQuery, start_path: str = "/") -> List[NodeInfo]:
        """Search for files and folders."""
        results = []
        
        # Start recursive search
        self._search_recursive(start_path, query, results)
        
        # Apply sorting
        results = self._sort_results(results, query.sort_by, query.sort_order)
        
        # Apply pagination
        if query.offset or query.limit:
            start = query.offset
            end = start + query.limit if query.limit else None
            results = results[start:end]
        
        self.logger.info(f"Search completed: {len(results)} results for '{query.query}'")
        return results
    
    def _search_recursive(self, path: str, query: SearchQuery, results: List[NodeInfo]):
        """Recursively search directories."""
        children = self.node_manager.list_children(path)
        
        for child in children:
            # Check if node matches query
            if self._matches_query(child, query):
                results.append(child)
            
            # Recurse into subdirectories
            if child.node_type == NodeType.FOLDER and query.recursive:
                child_path = self.node_manager.path_resolver.join_paths(path, child.name)
                self._search_recursive(child_path, query, results)
    
    def _matches_query(self, node: NodeInfo, query: SearchQuery) -> bool:
        """Check if node matches search query."""
        # Text search in name
        if query.query.lower() not in node.name.lower():
            return False
        
        # Apply filters
        for filter_type, filter_value in query.filters.items():
            if not self._apply_filter(node, filter_type, filter_value):
                return False
        
        return True
    
    def _apply_filter(self, node: NodeInfo, filter_type: SearchFilter, filter_value: Any) -> bool:
        """Apply specific filter to node."""
        if filter_type == SearchFilter.NAME:
            return filter_value.lower() in node.name.lower()
        
        elif filter_type == SearchFilter.EXTENSION:
            if node.node_type != NodeType.FILE:
                return False
            extension = FileUtils.get_file_extension(node.name)
            return extension == filter_value.lower()
        
        elif filter_type == SearchFilter.SIZE:
            # Filter value should be tuple: (operator, size)
            # e.g., ('>', 1024) for files larger than 1KB
            operator, size = filter_value
            if operator == '>':
                return node.size > size
            elif operator == '<':
                return node.size < size
            elif operator == '=':
                return node.size == size
            elif operator == '>=':
                return node.size >= size
            elif operator == '<=':
                return node.size <= size
        
        elif filter_type == SearchFilter.DATE_CREATED:
            if not node.created_at:
                return False
            # Filter value should be tuple: (operator, datetime)
            operator, date = filter_value
            if operator == 'after':
                return node.created_at > date
            elif operator == 'before':
                return node.created_at < date
            elif operator == 'on':
                return node.created_at.date() == date.date()
        
        elif filter_type == SearchFilter.MIME_TYPE:
            return node.mime_type == filter_value
        
        return True
    
    def _sort_results(self, results: List[NodeInfo], sort_by: SearchFilter, 
                     sort_order: SortOrder) -> List[NodeInfo]:
        """Sort search results."""
        reverse = sort_order == SortOrder.DESC
        
        if sort_by == SearchFilter.NAME:
            return sorted(results, key=lambda x: x.name.lower(), reverse=reverse)
        elif sort_by == SearchFilter.SIZE:
            return sorted(results, key=lambda x: x.size, reverse=reverse)
        elif sort_by == SearchFilter.DATE_CREATED:
            return sorted(results, key=lambda x: x.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=reverse)
        elif sort_by == SearchFilter.DATE_MODIFIED:
            return sorted(results, key=lambda x: x.modified_at or datetime.min.replace(tzinfo=timezone.utc), reverse=reverse)
        else:
            return results
    
    def find_by_pattern(self, pattern: str, start_path: str = "/") -> List[NodeInfo]:
        """Find files matching glob pattern."""
        import fnmatch
        
        results = []
        self._find_pattern_recursive(start_path, pattern, results)
        return results
    
    def _find_pattern_recursive(self, path: str, pattern: str, results: List[NodeInfo]):
        """Recursively find files matching pattern."""
        children = self.node_manager.list_children(path)
        
        for child in children:
            if fnmatch.fnmatch(child.name, pattern):
                results.append(child)
            
            if child.node_type == NodeType.FOLDER:
                child_path = self.node_manager.path_resolver.join_paths(path, child.name)
                self._find_pattern_recursive(child_path, pattern, results)


# ==============================================
# === TRANSFER MANAGEMENT ===
# ==============================================

class UploadManager:
    """File upload management."""
    
    def __init__(self, node_manager: NodeManager):
        self.node_manager = node_manager
        self._active_uploads: Dict[str, TransferProgress] = {}
        self._upload_executor = ThreadPoolExecutor(max_workers=3)
        self.logger = logging.getLogger(__name__)
    
    def upload_file(self, local_path: str, remote_path: str, 
                   chunk_size: int = Constants.CHUNK_SIZE,
                   progress_callback: Optional[Callable] = None) -> str:
        """Upload file from local to remote."""
        if not os.path.exists(local_path):
            raise StorageError(f"Local file not found: {local_path}")
        
        file_size = os.path.getsize(local_path)
        file_name = os.path.basename(local_path)
        
        # Create transfer progress
        transfer_id = Helpers.generate_request_id()
        progress = TransferProgress(
            transfer_id=transfer_id,
            file_name=file_name,
            total_size=file_size,
            started_at=DateTimeUtils.now_utc()
        )
        
        self._active_uploads[transfer_id] = progress
        
        # Submit upload task
        future = self._upload_executor.submit(
            self._perform_upload,
            local_path, remote_path, chunk_size, progress, progress_callback
        )
        
        self.logger.info(f"Started upload: {local_path} -> {remote_path} (ID: {transfer_id})")
        return transfer_id
    
    def _perform_upload(self, local_path: str, remote_path: str, chunk_size: int,
                       progress: TransferProgress, progress_callback: Optional[Callable]):
        """Perform the actual upload."""
        try:
            progress.status = "uploading"
            
            # Read file in chunks and simulate upload
            with open(local_path, 'rb') as f:
                transferred = 0
                start_time = time.time()
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    # Simulate network delay
                    time.sleep(0.01)  # 10ms delay per chunk
                    
                    transferred += len(chunk)
                    progress.transferred = transferred
                    
                    # Calculate speed and ETA
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        progress.speed = transferred / elapsed
                        remaining_bytes = progress.total_size - transferred
                        progress.eta = remaining_bytes / progress.speed if progress.speed > 0 else None
                    
                    # Call progress callback
                    if progress_callback:
                        progress_callback(progress)
            
            # Create remote file node
            parent_path = self.node_manager.path_resolver.get_parent_path(remote_path)
            file_name = self.node_manager.path_resolver.get_file_name(remote_path)
            
            # Read file content for checksum
            with open(local_path, 'rb') as f:
                content = f.read()
            
            mime_type, _ = mimetypes.guess_type(local_path)
            
            self.node_manager.create_node(
                name=file_name,
                node_type=NodeType.FILE,
                parent_path=parent_path,
                size=len(content),
                mime_type=mime_type,
                attributes={'checksum': hashlib.sha256(content).hexdigest()}
            )
            
            progress.status = "completed"
            progress.completed_at = DateTimeUtils.now_utc()
            
            self.logger.info(f"Upload completed: {transfer_id}")
            
        except Exception as e:
            progress.status = "failed"
            progress.error = str(e)
            self.logger.error(f"Upload failed: {transfer_id} - {e}")
    
    def get_upload_progress(self, transfer_id: str) -> Optional[TransferProgress]:
        """Get upload progress."""
        return self._active_uploads.get(transfer_id)
    
    def cancel_upload(self, transfer_id: str) -> bool:
        """Cancel upload."""
        if transfer_id in self._active_uploads:
            self._active_uploads[transfer_id].status = "cancelled"
            self.logger.info(f"Upload cancelled: {transfer_id}")
            return True
        return False
    
    def list_active_uploads(self) -> List[TransferProgress]:
        """List active uploads."""
        return [p for p in self._active_uploads.values() if p.status in ["pending", "uploading"]]


class DownloadManager:
    """File download management."""
    
    def __init__(self, node_manager: NodeManager):
        self.node_manager = node_manager
        self._active_downloads: Dict[str, TransferProgress] = {}
        self._download_executor = ThreadPoolExecutor(max_workers=3)
        self.logger = logging.getLogger(__name__)
    
    def download_file(self, remote_path: str, local_path: str,
                     progress_callback: Optional[Callable] = None) -> str:
        """Download file from remote to local."""
        node = self.node_manager.get_node_by_path(remote_path)
        if not node or node.node_type != NodeType.FILE:
            raise StorageError(f"Remote file not found: {remote_path}")
        
        # Create transfer progress
        transfer_id = Helpers.generate_request_id()
        progress = TransferProgress(
            transfer_id=transfer_id,
            file_name=node.name,
            total_size=node.size,
            started_at=DateTimeUtils.now_utc()
        )
        
        self._active_downloads[transfer_id] = progress
        
        # Submit download task
        future = self._download_executor.submit(
            self._perform_download,
            node, local_path, progress, progress_callback
        )
        
        self.logger.info(f"Started download: {remote_path} -> {local_path} (ID: {transfer_id})")
        return transfer_id
    
    def _perform_download(self, node: NodeInfo, local_path: str,
                         progress: TransferProgress, progress_callback: Optional[Callable]):
        """Perform the actual download."""
        try:
            progress.status = "downloading"
            
            # Simulate download by creating file with random content
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                chunk_size = Constants.CHUNK_SIZE
                transferred = 0
                start_time = time.time()
                
                while transferred < node.size:
                    chunk_size = min(chunk_size, node.size - transferred)
                    
                    # Simulate network delay
                    time.sleep(0.01)  # 10ms delay per chunk
                    
                    # Write placeholder data
                    chunk_data = b'0' * chunk_size
                    f.write(chunk_data)
                    
                    transferred += chunk_size
                    progress.transferred = transferred
                    
                    # Calculate speed and ETA
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        progress.speed = transferred / elapsed
                        remaining_bytes = progress.total_size - transferred
                        progress.eta = remaining_bytes / progress.speed if progress.speed > 0 else None
                    
                    # Call progress callback
                    if progress_callback:
                        progress_callback(progress)
            
            progress.status = "completed"
            progress.completed_at = DateTimeUtils.now_utc()
            
            self.logger.info(f"Download completed: {progress.transfer_id}")
            
        except Exception as e:
            progress.status = "failed"
            progress.error = str(e)
            self.logger.error(f"Download failed: {progress.transfer_id} - {e}")
    
    def get_download_progress(self, transfer_id: str) -> Optional[TransferProgress]:
        """Get download progress."""
        return self._active_downloads.get(transfer_id)
    
    def cancel_download(self, transfer_id: str) -> bool:
        """Cancel download."""
        if transfer_id in self._active_downloads:
            self._active_downloads[transfer_id].status = "cancelled"
            self.logger.info(f"Download cancelled: {transfer_id}")
            return True
        return False


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Enums
    'NodeType', 'PermissionLevel', 'SearchFilter', 'SortOrder', 'ConflictResolution',
    
    # Data Classes
    'NodeInfo', 'SearchQuery', 'TransferProgress', 'BackupConfig',
    
    # Navigation
    'PathResolver', 'Navigator', 'LocalNavigator',
    
    # Node Management
    'NodeManager',
    
    # File Operations
    'FileOperations', 'DirectoryOperations',
    
    # Search
    'SearchEngine',
    
    # Transfer Management
    'UploadManager', 'DownloadManager'
]