"""
Content Processing and Intelligence module for MegaPythonLibrary.

This module contains:
- File type detection and analysis
- Media processing utilities
- Content categorization
- File metadata extraction
- MIME type handling
"""

import os
import mimetypes
from typing import Optional, Dict, Any, List
from pathlib import Path

from .monitor import get_logger

# ==============================================
# === FILE TYPE DETECTION ===
# ==============================================

def detect_file_type(file_path: str) -> Optional[str]:
    """
    Detect MIME type from file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type string or None if not detectable
    """
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


def is_document_file(file_path: str) -> bool:
    """
    Check if file is a document.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is a document
    """
    mime_type = detect_file_type(file_path)
    if not mime_type:
        return False
    
    document_types = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-powerpoint',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'text/plain',
        'text/rtf',
        'application/rtf',
    ]
    
    return mime_type in document_types or mime_type.startswith('text/')


def is_archive_file(file_path: str) -> bool:
    """
    Check if file is an archive.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is an archive
    """
    mime_type = detect_file_type(file_path)
    if not mime_type:
        return False
    
    archive_types = [
        'application/zip',
        'application/x-zip-compressed',
        'application/x-rar-compressed',
        'application/x-7z-compressed',
        'application/x-tar',
        'application/gzip',
        'application/x-gzip',
        'application/x-bzip2',
        'application/x-compress',
        'application/x-compressed',
    ]
    
    return mime_type in archive_types


# ==============================================
# === FILE CATEGORIZATION ===
# ==============================================

def categorize_file(file_path: str) -> str:
    """
    Categorize file based on its type.
    
    Args:
        file_path: Path to file
        
    Returns:
        File category as string
    """
    if is_image_file(file_path):
        return 'image'
    elif is_video_file(file_path):
        return 'video'
    elif is_audio_file(file_path):
        return 'audio'
    elif is_document_file(file_path):
        return 'document'
    elif is_archive_file(file_path):
        return 'archive'
    else:
        return 'other'


def get_file_category_icon(category: str) -> str:
    """
    Get icon representation for file category.
    
    Args:
        category: File category
        
    Returns:
        Unicode icon for the category
    """
    category_icons = {
        'image': '🖼️',
        'video': '🎬',
        'audio': '🎵',
        'document': '📄',
        'archive': '📦',
        'folder': '📁',
        'other': '📄'
    }
    
    return category_icons.get(category, '📄')


# ==============================================
# === FILE METADATA EXTRACTION ===
# ==============================================

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    logger = get_logger("content")
    
    try:
        path = Path(file_path)
        if not path.exists():
            return {'error': 'File not found'}
        
        stat = path.stat()
        mime_type = detect_file_type(file_path)
        category = categorize_file(file_path)
        
        return {
            'name': path.name,
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'extension': get_file_extension(file_path),
            'mime_type': mime_type,
            'category': category,
            'icon': get_file_category_icon(category),
            'is_image': is_image_file(file_path),
            'is_video': is_video_file(file_path),
            'is_audio': is_audio_file(file_path),
            'is_document': is_document_file(file_path),
            'is_archive': is_archive_file(file_path),
        }
        
    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {e}")
        return {'error': str(e)}


# ==============================================
# === MEDIA PROCESSING UTILITIES ===
# ==============================================

def get_image_extensions() -> List[str]:
    """Get list of common image file extensions."""
    return [
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
        '.webp', '.svg', '.ico', '.raw', '.cr2', '.nef', '.arw'
    ]


def get_video_extensions() -> List[str]:
    """Get list of common video file extensions."""
    return [
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
        '.m4v', '.3gp', '.3g2', '.mts', '.m2ts', '.vob', '.ts'
    ]


def get_audio_extensions() -> List[str]:
    """Get list of common audio file extensions."""
    return [
        '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a',
        '.opus', '.aiff', '.au', '.ra', '.mid', '.midi'
    ]


def get_document_extensions() -> List[str]:
    """Get list of common document file extensions."""
    return [
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.txt', '.rtf', '.odt', '.ods', '.odp', '.pages', '.numbers',
        '.keynote', '.epub', '.mobi'
    ]


def get_archive_extensions() -> List[str]:
    """Get list of common archive file extensions."""
    return [
        '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz',
        '.tar.gz', '.tar.bz2', '.tar.xz', '.dmg', '.iso'
    ]


def is_supported_media_file(file_path: str) -> bool:
    """
    Check if file is a supported media file (image, video, audio).
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is supported media
    """
    return is_image_file(file_path) or is_video_file(file_path) or is_audio_file(file_path)


# ==============================================
# === CONTENT FILTERING ===
# ==============================================

def filter_files_by_type(file_paths: List[str], file_type: str) -> List[str]:
    """
    Filter files by type.
    
    Args:
        file_paths: List of file paths
        file_type: Type to filter by ('image', 'video', 'audio', 'document', 'archive')
        
    Returns:
        Filtered list of file paths
    """
    type_checkers = {
        'image': is_image_file,
        'video': is_video_file,
        'audio': is_audio_file,
        'document': is_document_file,
        'archive': is_archive_file,
    }
    
    checker = type_checkers.get(file_type)
    if not checker:
        return []
    
    return [path for path in file_paths if checker(path)]


def get_files_by_extension(file_paths: List[str], extensions: List[str]) -> List[str]:
    """
    Filter files by extensions.
    
    Args:
        file_paths: List of file paths
        extensions: List of extensions to match (with or without dots)
        
    Returns:
        Filtered list of file paths
    """
    # Normalize extensions to include dots
    normalized_extensions = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized_extensions.append(ext.lower())
    
    return [
        path for path in file_paths
        if get_file_extension(path) in normalized_extensions
    ]


# ==============================================
# === FILE SIZE ANALYSIS ===
# ==============================================

def analyze_file_sizes(file_paths: List[str]) -> Dict[str, Any]:
    """
    Analyze file sizes in a collection.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        Dictionary with size analysis
    """
    logger = get_logger("content")
    
    sizes = []
    total_size = 0
    largest_file = None
    smallest_file = None
    largest_size = 0
    smallest_size = float('inf')
    
    for file_path in file_paths:
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                size = path.stat().st_size
                sizes.append(size)
                total_size += size
                
                if size > largest_size:
                    largest_size = size
                    largest_file = file_path
                
                if size < smallest_size:
                    smallest_size = size
                    smallest_file = file_path
        except Exception as e:
            logger.warning(f"Could not get size for {file_path}: {e}")
    
    if not sizes:
        return {
            'total_files': 0,
            'total_size': 0,
            'average_size': 0,
            'largest_file': None,
            'smallest_file': None,
        }
    
    return {
        'total_files': len(sizes),
        'total_size': total_size,
        'average_size': total_size / len(sizes),
        'largest_file': largest_file,
        'largest_size': largest_size,
        'smallest_file': smallest_file,
        'smallest_size': smallest_size if smallest_size != float('inf') else 0,
    }


# ==============================================
# === CONTENT SEARCH ===
# ==============================================

def search_files_by_content_type(directory: str, content_type: str) -> List[str]:
    """
    Search for files of a specific content type in a directory.
    
    Args:
        directory: Directory to search in
        content_type: Type to search for ('image', 'video', 'audio', etc.)
        
    Returns:
        List of matching file paths
    """
    logger = get_logger("content")
    
    try:
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            return []
        
        all_files = [str(f) for f in path.rglob('*') if f.is_file()]
        return filter_files_by_type(all_files, content_type)
        
    except Exception as e:
        logger.error(f"Error searching directory {directory}: {e}")
        return []


# ==============================================
# === MIME TYPE UTILITIES ===
# ==============================================

def register_custom_mime_types():
    """Register additional MIME types not in standard library."""
    custom_types = {
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm',
        '.opus': 'audio/opus',
        '.flac': 'audio/flac',
        '.7z': 'application/x-7z-compressed',
        '.rar': 'application/x-rar-compressed',
    }
    
    for extension, mime_type in custom_types.items():
        mimetypes.add_type(mime_type, extension)


# Initialize custom MIME types
register_custom_mime_types()


# ==============================================
# === CONTENT VALIDATION ===
# ==============================================

def validate_file_type(file_path: str, expected_types: List[str]) -> bool:
    """
    Validate that a file matches one of the expected types.
    
    Args:
        file_path: Path to file
        expected_types: List of expected MIME types or categories
        
    Returns:
        True if file matches expected type
    """
    mime_type = detect_file_type(file_path)
    category = categorize_file(file_path)
    
    # Check against MIME types
    if mime_type and mime_type in expected_types:
        return True
    
    # Check against categories
    if category in expected_types:
        return True
    
    return False


def is_safe_file_type(file_path: str) -> bool:
    """
    Check if file type is considered safe (not executable).
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file type is safe
    """
    mime_type = detect_file_type(file_path)
    extension = get_file_extension(file_path)
    
    # Dangerous MIME types
    dangerous_mime_types = [
        'application/x-executable',
        'application/x-msdownload',
        'application/x-msdos-program',
        'application/x-dosexec',
    ]
    
    # Dangerous extensions
    dangerous_extensions = [
        '.exe', '.bat', '.cmd', '.com', '.scr', '.pif',
        '.vbs', '.js', '.jar', '.app', '.deb', '.rpm'
    ]
    
    if mime_type in dangerous_mime_types:
        return False
    
    if extension in dangerous_extensions:
        return False
    
    return True