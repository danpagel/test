"""
Utilities Module for Client
===============================

This module provides convenience utilities and helper functions for the MegaPythonLibrary,
including display formatting, search utilities, and common operations.

Features:
- Formatted file listing (ls command)
- File/folder search with wildcards (find command)  
- Directory tree visualization (tree command)
- Path manipulation utilities
- Display formatting helpers
- File type detection utilities
- Common utility functions

Author: Extracted for modular architecture
"""

import logging
import fnmatch
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# ==============================================
# === FILE TYPE UTILITIES (NO DEPENDENCIES) ===  
# ==============================================

def detect_file_type(file_path: str) -> Optional[str]:
    """
    Detect MIME type from file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type string or None if not detectable
    """
    import mimetypes
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type
    except Exception:
        return None


def get_file_extension(file_path: str) -> str:
    """
    Get file extension from path.
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension (including dot) or empty string
    """
    import os
    return os.path.splitext(file_path)[1].lower()


def is_image_file(file_path: str) -> bool:
    """
    Check if file is an image based on MIME type.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is an image
    """
    mime_type = detect_file_type(file_path)
    return mime_type is not None and mime_type.startswith('image/')


def is_video_file(file_path: str) -> bool:
    """
    Check if file is a video based on MIME type.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is a video
    """
    mime_type = detect_file_type(file_path)
    return mime_type is not None and mime_type.startswith('video/')


def is_audio_file(file_path: str) -> bool:
    """
    Check if file is audio based on MIME type.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is audio
    """
    mime_type = detect_file_type(file_path)
    return mime_type is not None and mime_type.startswith('audio/')


def is_media_file(file_path: str) -> bool:
    """
    Check if file is any media type (image, video, audio).
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is any media type
    """
    return is_image_file(file_path) or is_video_file(file_path) or is_audio_file(file_path)

# ==============================================
# === MEGA-SPECIFIC UTILITIES (WITH DEPENDENCIES) ===
# ==============================================

# Import MEGA-specific modules only for functions that need them
from .filesystem import fs_tree, get_node_by_path, MegaNode
from .auth import is_logged_in
from .exceptions import RequestError


# ==============================================
# === DISPLAY AND FORMATTING UTILITIES ===
# ==============================================

def format_file_list(nodes: List, show_details: bool = True) -> str:
    """
    Format list of nodes for display.
    
    Args:
        nodes: List of nodes to format
        show_details: Include size, date, and other details
        
    Returns:
        Formatted string
    """
    if not nodes:
        return "No files or folders found."
    
    lines = []
    for node in sorted(nodes, key=lambda n: (n.is_file(), n.name.lower())):
        if show_details:
            # Detailed format with size and type
            icon = "📄" if node.is_file() else "📁"
            size_str = f"{node.size:,} bytes" if node.is_file() else "Folder"
            line = f"{icon} {node.name:<40} {size_str:>15}"
        else:
            # Simple format
            icon = "📄" if node.is_file() else "📁"
            line = f"{icon} {node.name}"
        lines.append(line)
    
    return "\n".join(lines)


def format_tree_structure(start_node, max_depth: int = 3, 
                         show_files: bool = True) -> str:
    """
    Generate directory tree structure.
    
    Args:
        start_node: Starting node for tree
        max_depth: Maximum depth to show
        show_files: Include files in tree
        
    Returns:
        Tree structure as string
    """
    lines = []
    
    def build_tree(node, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return
        
        icon = "📁" if node.is_folder() else "📄"
        lines.append(f"{prefix}{icon} {node.name}")
        
        if node.is_folder() and depth < max_depth:
            from .filesystem import fs_tree
            children = fs_tree.get_children(node.handle)
            
            # Filter children based on show_files setting
            if not show_files:
                children = [child for child in children if child.is_folder()]
            
            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                child_prefix = prefix + ("└── " if is_last else "├── ")
                next_prefix = prefix + ("    " if is_last else "│   ")
                
                child_icon = "📁" if child.is_folder() else "📄"
                lines.append(f"{child_prefix}{child_icon} {child.name}")
                
                if child.is_folder() and depth + 1 < max_depth:
                    from .filesystem import fs_tree
                    grandchildren = fs_tree.get_children(child.handle)
                    if not show_files:
                        grandchildren = [gc for gc in grandchildren if gc.is_folder()]
                    
                    for j, grandchild in enumerate(grandchildren):
                        is_last_gc = j == len(grandchildren) - 1
                        gc_prefix = next_prefix + ("└── " if is_last_gc else "├── ")
                        gc_icon = "📁" if grandchild.is_folder() else "📄"
                        lines.append(f"{gc_prefix}{gc_icon} {grandchild.name}")
    
    build_tree(start_node)
    return "\n".join(lines)


def format_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


# ==============================================
# === SEARCH AND FIND UTILITIES ===
# ==============================================

def find_files_by_name(name: str, path: str = "/", case_sensitive: bool = False) -> List[MegaNode]:
    """
    Find files/folders by name pattern.
    
    Args:
        name: Name pattern to search for (supports wildcards with fnmatch)
        path: Path to search in (default: entire filesystem)
        case_sensitive: Whether search is case sensitive
        
    Returns:
        List of matching nodes
    """
    if not is_logged_in():
        raise RequestError("Not logged in")
    
    matches = []
    
    # Get starting nodes
    if path == "/":
        search_nodes = list(fs_tree.nodes.values())
    else:
        start_node = get_node_by_path(path)
        if not start_node:
            return []
        search_nodes = [start_node] + fs_tree.get_children(start_node.handle)
    
    # Search recursively
    def search_recursive(nodes):
        for node in nodes:
            node_name = node.name if case_sensitive else node.name.lower()
            search_pattern = name if case_sensitive else name.lower()
            
            if fnmatch.fnmatch(node_name, search_pattern):
                matches.append(node)
            
            if node.is_folder():
                children = fs_tree.get_children(node.handle)
                search_recursive(children)
    
    search_recursive(search_nodes)
    return matches


def find_files_by_extension(extension: str, path: str = "/") -> List[MegaNode]:
    """
    Find files by extension.
    
    Args:
        extension: File extension (with or without dot)
        path: Path to search in
        
    Returns:
        List of matching file nodes
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    return find_files_by_name(f"*{extension}", path)


def find_files_by_size(min_size: int = None, max_size: int = None, path: str = "/") -> List[MegaNode]:
    """
    Find files by size range.
    
    Args:
        min_size: Minimum size in bytes
        max_size: Maximum size in bytes  
        path: Path to search in
        
    Returns:
        List of matching file nodes
    """
    if not is_logged_in():
        raise RequestError("Not logged in")
    
    matches = []
    
    # Get starting nodes
    if path == "/":
        search_nodes = list(fs_tree.nodes.values())
    else:
        start_node = get_node_by_path(path)
        if not start_node:
            return []
        search_nodes = [start_node] + fs_tree.get_children(start_node.handle)
    
    # Search recursively
    def search_recursive(nodes):
        for node in nodes:
            if node.is_file():
                if min_size is not None and node.size < min_size:
                    continue
                if max_size is not None and node.size > max_size:
                    continue
                matches.append(node)
            
            if node.is_folder():
                children = fs_tree.get_children(node.handle)
                search_recursive(children)
    
    search_recursive(search_nodes)
    return matches


def get_folder_contents(path: str = "/", include_files: bool = True, 
                       include_folders: bool = True) -> List[MegaNode]:
    """
    Get contents of a folder with filtering options.
    
    Args:
        path: Folder path
        include_files: Include files in results
        include_folders: Include folders in results
        
    Returns:
        List of nodes in folder
    """
    if not is_logged_in():
        raise RequestError("Not logged in")
    
    # Handle path-based lookup
    if path != "/":
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"Path not found: {path}")
        if not node.is_folder():
            raise RequestError(f"Path is not a folder: {path}")
        children = fs_tree.get_children(node.handle)
    else:
        # Root folder contents
        children = fs_tree.get_children(fs_tree.root_handle) if fs_tree.root_handle else []
    
    # Filter results
    filtered = []
    for child in children:
        if child.is_file() and include_files:
            filtered.append(child)
        elif child.is_folder() and include_folders:
            filtered.append(child)
    
    return filtered


# ==============================================
# === PATH MANIPULATION UTILITIES ===
# ==============================================

def normalize_mega_path(path: str) -> str:
    """
    Normalize a MEGA path.
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized path
    """
    if not path:
        return "/"
    
    # Ensure path starts with /
    if not path.startswith("/"):
        path = "/" + path
    
    # Remove duplicate slashes
    while "//" in path:
        path = path.replace("//", "/")
    
    # Remove trailing slash unless it's root
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    
    return path


def get_parent_path(path: str) -> str:
    """
    Get parent path of a given path.
    
    Args:
        path: Child path
        
    Returns:
        Parent path
    """
    path = normalize_mega_path(path)
    if path == "/":
        return "/"
    
    parts = path.split("/")
    if len(parts) <= 2:  # ["", "folder"]
        return "/"
    
    return "/".join(parts[:-1])


def get_path_name(path: str) -> str:
    """
    Get the name (last component) of a path.
    
    Args:
        path: Full path
        
    Returns:
        Name component
    """
    path = normalize_mega_path(path)
    if path == "/":
        return ""
    
    return path.split("/")[-1]


def join_paths(*paths: str) -> str:
    """
    Join multiple path components.
    
    Args:
        paths: Path components to join
        
    Returns:
        Joined path
    """
    if not paths:
        return "/"
    
    result = ""
    for path in paths:
        if not path:
            continue
        
        path = str(path).strip("/")
        if path:
            result += "/" + path
    
    return normalize_mega_path(result or "/")


# ==============================================
# === STATISTICS AND INFO UTILITIES ===
# ==============================================

def get_folder_statistics(path: str = "/") -> Dict[str, Any]:
    """
    Get detailed statistics about a folder.
    
    Args:
        path: Folder path to analyze
        
    Returns:
        Dictionary with folder statistics
    """
    if not is_logged_in():
        raise RequestError("Not logged in")
    
    if path == "/":
        node = fs_tree.nodes.get(fs_tree.root_handle)
        children = list(fs_tree.nodes.values())
    else:
        node = get_node_by_path(path)
        if not node:
            raise RequestError(f"Path not found: {path}")
        if not node.is_folder():
            raise RequestError(f"Path is not a folder: {path}")
        children = fs_tree.get_children(node.handle)
    
    stats = {
        "path": path,
        "name": node.name if node else "Cloud Drive",
        "total_items": 0,
        "files": 0,
        "folders": 0,
        "total_size": 0,
        "largest_file": None,
        "smallest_file": None,
        "file_types": {},
    }
    
    file_sizes = []
    
    def analyze_recursive(nodes, is_root_level=True):
        for child in nodes:
            if is_root_level:
                stats["total_items"] += 1
            
            if child.is_file():
                if is_root_level:
                    stats["files"] += 1
                stats["total_size"] += child.size
                file_sizes.append(child.size)
                
                # Track largest/smallest files
                if stats["largest_file"] is None or child.size > stats["largest_file"]["size"]:
                    stats["largest_file"] = {"name": child.name, "size": child.size}
                if stats["smallest_file"] is None or child.size < stats["smallest_file"]["size"]:
                    stats["smallest_file"] = {"name": child.name, "size": child.size}
                
                # Track file types
                ext = Path(child.name).suffix.lower()
                if ext:
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                else:
                    stats["file_types"]["(no extension)"] = stats["file_types"].get("(no extension)", 0) + 1
                    
            elif child.is_folder():
                if is_root_level:
                    stats["folders"] += 1
                # Recurse into subfolders
                grandchildren = fs_tree.get_children(child.handle)
                analyze_recursive(grandchildren, is_root_level=False)
    
    analyze_recursive(children)
    
    # Add calculated stats
    if file_sizes:
        stats["average_file_size"] = sum(file_sizes) / len(file_sizes)
    else:
        stats["average_file_size"] = 0
        stats["largest_file"] = None
        stats["smallest_file"] = None
    
    return stats


# ==============================================
# === CLIENT INTEGRATION METHODS ===
# ==============================================

def add_utilities_methods(client_class):
    """Add utility methods to the MPLClient class."""
    
    def ls_method(self, path: str = "/", show_details: bool = True) -> str:
        """
        List folder contents in a formatted string.
        
        Args:
            path: Folder path to list
            show_details: Include detailed information
            
        Returns:
            Formatted file listing
        """
        nodes = get_folder_contents(path)
        return format_file_list(nodes, show_details)
    
    def find_method(self, name: str, path: str = "/") -> List[MegaNode]:
        """
        Find files/folders by name.
        
        Args:
            name: Name to search for (supports wildcards with fnmatch)
            path: Path to search in (default: entire filesystem)
            
        Returns:
            List of matching nodes
        """
        return find_files_by_name(name, path)
    
    def tree_method(self, path: str = "/", max_depth: int = 3, show_files: bool = True) -> str:
        """
        Show directory tree structure.
        
        Args:
            path: Root path for tree
            max_depth: Maximum depth to show
            show_files: Include files in tree display
            
        Returns:
            Tree structure as string
        """
        if path == "/":
            start_node = fs_tree.nodes.get(fs_tree.root_handle)
        else:
            start_node = get_node_by_path(path)
        
        if not start_node:
            return f"Path not found: {path}"
        
        return format_tree_structure(start_node, max_depth, show_files)
    
    def find_by_extension_method(self, extension: str, path: str = "/") -> List[MegaNode]:
        """Find files by extension."""
        return find_files_by_extension(extension, path)
    
    def find_by_size_method(self, min_size: int = None, max_size: int = None, path: str = "/") -> List[MegaNode]:
        """Find files by size range."""
        return find_files_by_size(min_size, max_size, path)
    
    def get_folder_stats_method(self, path: str = "/") -> Dict[str, Any]:
        """Get detailed folder statistics."""
        return get_folder_statistics(path)
    
    def get_stats_method(self) -> Dict[str, Any]:
        """Get client statistics."""
        from .filesystem import fs_tree, get_nodes
        from .auth import is_logged_in, get_current_user
        
        stats = {
            'logged_in': is_logged_in(),
            'current_user': get_current_user(),
            'filesystem_loaded': len(fs_tree.nodes) > 0,
            'node_count': len(fs_tree.nodes),
        }
        
        if is_logged_in():
            try:
                quota = self.get_quota()
                stats.update(quota)
            except Exception:
                pass
        
        return stats
    
    def refresh_filesystem_if_needed_method(self) -> None:
        """Refresh filesystem if not already loaded."""
        from .filesystem import fs_tree, get_nodes
        if not fs_tree.nodes:
            get_nodes()
    
    # Add methods to client class
    setattr(client_class, 'ls', ls_method)
    setattr(client_class, 'find', find_method)
    setattr(client_class, 'tree', tree_method)
    setattr(client_class, 'find_by_extension', find_by_extension_method)
    setattr(client_class, 'find_by_size', find_by_size_method)
    setattr(client_class, 'get_folder_stats', get_folder_stats_method)
    setattr(client_class, 'get_stats', get_stats_method)
    setattr(client_class, '_refresh_filesystem_if_needed', refresh_filesystem_if_needed_method)


# ==============================================
# === VALIDATION AND PARSING UTILITIES ===
# ==============================================


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Display utilities
    'format_file_list',
    'format_tree_structure', 
    'format_size',
    
    # Search utilities
    'find_files_by_name',
    'find_files_by_extension',
    'find_files_by_size',
    'get_folder_contents',
    
    # Path utilities
    'normalize_mega_path',
    'get_parent_path',
    'get_path_name', 
    'join_paths',
    
    # File type utilities
    'detect_file_type',
    'get_file_extension',
    'is_image_file',
    'is_video_file', 
    'is_audio_file',
    'is_media_file',
    
    # Statistics
    'get_folder_statistics',
    
    # Integration
    'add_utilities_methods',
]
