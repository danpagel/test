"""
Sharing and Collaboration module for MegaPythonLibrary.

This module contains:
- Public link sharing functionality
- Share management
- Link generation and configuration
- Access control utilities
"""

from typing import Optional, Dict, Any, List
from .monitor import get_logger
from .network import single_api_request
from .auth import current_session, require_authentication
from .utils import RequestError, base64_url_encode, base64_url_decode

# ==============================================
# === PUBLIC LINK SHARING ===
# ==============================================

@require_authentication
def create_public_link(node_handle: str, password: Optional[str] = None, 
                      expiry_time: Optional[int] = None) -> str:
    """
    Create a public sharing link for a file or folder.
    
    Args:
        node_handle: Handle of the node to share
        password: Optional password protection
        expiry_time: Optional expiry timestamp
        
    Returns:
        Public sharing URL
        
    Raises:
        RequestError: If sharing fails
    """
    logger = get_logger("sharing")
    
    command = {
        'a': 'l',  # Create link
        'n': node_handle,
    }
    
    if password:
        # Add password protection
        command['password'] = password
    
    if expiry_time:
        command['exp'] = expiry_time
    
    try:
        result = single_api_request(command, current_session.session_id)
        
        if isinstance(result, str):
            # Link handle returned
            link_url = f"https://mega.nz/file/{result}"
            logger.info(f"Created public link for node {node_handle}")
            return link_url
        else:
            raise RequestError("Invalid response from sharing API")
            
    except Exception as e:
        logger.error(f"Failed to create public link: {e}")
        raise


@require_authentication
def remove_public_link(node_handle: str) -> bool:
    """
    Remove public sharing link for a node.
    
    Args:
        node_handle: Handle of the node
        
    Returns:
        True if link was removed successfully
    """
    logger = get_logger("sharing")
    
    command = {
        'a': 'l',  # Link operation
        'n': node_handle,
        'd': 1,    # Delete link
    }
    
    try:
        result = single_api_request(command, current_session.session_id)
        logger.info(f"Removed public link for node {node_handle}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to remove public link: {e}")
        return False


@require_authentication
def get_public_links() -> List[Dict[str, Any]]:
    """
    Get all public links for the current user.
    
    Returns:
        List of public link information
    """
    logger = get_logger("sharing")
    
    command = {'a': 'l'}  # List links
    
    try:
        result = single_api_request(command, current_session.session_id)
        
        if isinstance(result, list):
            return result
        else:
            return []
            
    except Exception as e:
        logger.error(f"Failed to get public links: {e}")
        return []


def parse_mega_url(url: str) -> Optional[Dict[str, str]]:
    """
    Parse a MEGA URL to extract file/folder information.
    
    Args:
        url: MEGA URL (e.g., https://mega.nz/file/abc123)
        
    Returns:
        Dictionary with parsed information or None if invalid
    """
    logger = get_logger("sharing")
    
    try:
        # Remove protocol and domain
        if 'mega.nz' not in url:
            return None
        
        parts = url.split('mega.nz/')
        if len(parts) < 2:
            return None
        
        path_part = parts[1]
        
        # Handle different URL formats
        if path_part.startswith('file/'):
            # File link: https://mega.nz/file/handle#key
            file_part = path_part[5:]  # Remove 'file/'
            if '#' in file_part:
                handle, key = file_part.split('#', 1)
                return {
                    'type': 'file',
                    'handle': handle,
                    'key': key,
                    'url': url
                }
        elif path_part.startswith('folder/'):
            # Folder link: https://mega.nz/folder/handle#key
            folder_part = path_part[7:]  # Remove 'folder/'
            if '#' in folder_part:
                handle, key = folder_part.split('#', 1)
                return {
                    'type': 'folder',
                    'handle': handle,
                    'key': key,
                    'url': url
                }
        elif path_part.startswith('#!'):
            # Legacy format: https://mega.nz/#!handle!key
            legacy_part = path_part[2:]  # Remove '#!'
            if '!' in legacy_part:
                handle, key = legacy_part.split('!', 1)
                return {
                    'type': 'file',
                    'handle': handle,
                    'key': key,
                    'url': url,
                    'legacy': True
                }
        elif path_part.startswith('#F!'):
            # Legacy folder format: https://mega.nz/#F!handle!key
            legacy_part = path_part[3:]  # Remove '#F!'
            if '!' in legacy_part:
                handle, key = legacy_part.split('!', 1)
                return {
                    'type': 'folder',
                    'handle': handle,
                    'key': key,
                    'url': url,
                    'legacy': True
                }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to parse MEGA URL {url}: {e}")
        return None


def get_public_file_info(url: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a publicly shared file.
    
    Args:
        url: MEGA public file URL
        
    Returns:
        File information or None if failed
    """
    logger = get_logger("sharing")
    
    parsed = parse_mega_url(url)
    if not parsed or parsed['type'] != 'file':
        return None
    
    try:
        command = {
            'a': 'g',  # Get file info
            'p': parsed['handle']
        }
        
        result = single_api_request(command)
        
        if isinstance(result, dict):
            return {
                'handle': parsed['handle'],
                'name': result.get('at', {}).get('n', 'Unknown'),
                'size': result.get('s', 0),
                'type': result.get('t', 0),
                'url': url
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get public file info: {e}")
        return None


# ==============================================
# === SHARING UTILITIES ===
# ==============================================

def is_valid_mega_url(url: str) -> bool:
    """
    Check if a URL is a valid MEGA sharing URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid MEGA URL
    """
    parsed = parse_mega_url(url)
    return parsed is not None


def get_share_type(url: str) -> Optional[str]:
    """
    Get the type of MEGA share (file or folder).
    
    Args:
        url: MEGA URL
        
    Returns:
        'file' or 'folder' or None if invalid
    """
    parsed = parse_mega_url(url)
    return parsed['type'] if parsed else None


def extract_key_from_url(url: str) -> Optional[str]:
    """
    Extract the decryption key from a MEGA URL.
    
    Args:
        url: MEGA URL
        
    Returns:
        Decryption key or None if not found
    """
    parsed = parse_mega_url(url)
    return parsed['key'] if parsed else None


def extract_handle_from_url(url: str) -> Optional[str]:
    """
    Extract the file/folder handle from a MEGA URL.
    
    Args:
        url: MEGA URL
        
    Returns:
        Handle or None if not found
    """
    parsed = parse_mega_url(url)
    return parsed['handle'] if parsed else None


# ==============================================
# === COLLABORATION FEATURES ===
# ==============================================

@require_authentication
def share_with_user(node_handle: str, target_email: str, 
                   access_level: str = 'read') -> bool:
    """
    Share a file or folder with another user.
    
    Args:
        node_handle: Handle of the node to share
        target_email: Email of the user to share with
        access_level: Access level ('read', 'readwrite', 'full')
        
    Returns:
        True if sharing was successful
    """
    logger = get_logger("sharing")
    
    # Map access levels to MEGA's numeric values
    access_levels = {
        'read': 0,
        'readwrite': 1,
        'full': 2
    }
    
    if access_level not in access_levels:
        raise ValueError("Invalid access level. Use 'read', 'readwrite', or 'full'")
    
    command = {
        'a': 's2',  # Share with user
        'n': node_handle,
        'u': target_email,
        'r': access_levels[access_level]
    }
    
    try:
        result = single_api_request(command, current_session.session_id)
        logger.info(f"Shared node {node_handle} with {target_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to share with user: {e}")
        return False


@require_authentication
def get_shared_files() -> List[Dict[str, Any]]:
    """
    Get files shared by or with the current user.
    
    Returns:
        List of shared file information
    """
    logger = get_logger("sharing")
    
    command = {'a': 's'}  # Get shares
    
    try:
        result = single_api_request(command, current_session.session_id)
        
        if isinstance(result, list):
            return result
        else:
            return []
            
    except Exception as e:
        logger.error(f"Failed to get shared files: {e}")
        return []


# ==============================================
# === LINK MANAGEMENT ===
# ==============================================

def generate_share_url(node_handle: str, node_key: str, 
                      is_folder: bool = False) -> str:
    """
    Generate a MEGA share URL from handle and key.
    
    Args:
        node_handle: Node handle
        node_key: Node key
        is_folder: Whether this is a folder
        
    Returns:
        MEGA share URL
    """
    if is_folder:
        return f"https://mega.nz/folder/{node_handle}#{node_key}"
    else:
        return f"https://mega.nz/file/{node_handle}#{node_key}"


def create_legacy_url(node_handle: str, node_key: str, 
                     is_folder: bool = False) -> str:
    """
    Create a legacy format MEGA URL.
    
    Args:
        node_handle: Node handle
        node_key: Node key
        is_folder: Whether this is a folder
        
    Returns:
        Legacy MEGA URL
    """
    if is_folder:
        return f"https://mega.nz/#F!{node_handle}!{node_key}"
    else:
        return f"https://mega.nz/#!{node_handle}!{node_key}"


# ==============================================
# === SHARING STATISTICS ===
# ==============================================

@require_authentication
def get_sharing_stats() -> Dict[str, int]:
    """
    Get sharing statistics for the current user.
    
    Returns:
        Dictionary with sharing statistics
    """
    logger = get_logger("sharing")
    
    try:
        public_links = get_public_links()
        shared_files = get_shared_files()
        
        return {
            'public_links_count': len(public_links),
            'shared_files_count': len(shared_files),
            'total_shares': len(public_links) + len(shared_files)
        }
        
    except Exception as e:
        logger.error(f"Failed to get sharing stats: {e}")
        return {
            'public_links_count': 0,
            'shared_files_count': 0,
            'total_shares': 0
        }