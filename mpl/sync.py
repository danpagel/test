# -*- coding: utf-8 -*-
"""
Advanced Synchronization Module for MegaPythonLibrary
Provides comprehensive local ↔ cloud sync functionality with real-time monitoring
"""

import os
import time
import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import logging

# File system monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# Enhanced logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SyncConfig:
    """Configuration for synchronization operations"""
    local_path: str
    remote_path: str
    sync_direction: str = "bidirectional"  # "up", "down", "bidirectional"
    conflict_resolution: str = "newer_wins"  # "newer_wins", "local_wins", "remote_wins", "ask"
    ignore_patterns: List[str] = None
    real_time: bool = True
    sync_interval: int = 300  # seconds
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_conflicts: bool = True
    
    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = [
                "*.tmp", "*.temp", "*.lock", ".DS_Store", "Thumbs.db",
                "*.swp", "*.swo", "~*", ".git/*", "__pycache__/*"
            ]

@dataclass
class FileInfo:
    """File information for synchronization tracking"""
    path: str
    size: int
    modified: datetime
    checksum: str
    is_directory: bool = False
    sync_status: str = "unknown"  # "synced", "local_newer", "remote_newer", "conflict"

class SyncDatabase:
    """Database for tracking synchronization state"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load_database()
    
    def _load_database(self) -> Dict:
        """Load sync database from disk"""
        try:
            if self.db_path.exists():
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load sync database: {e}")
        
        return {
            "files": {},
            "last_sync": None,
            "sync_history": []
        }
    
    def save_database(self):
        """Save sync database to disk"""
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save sync database: {e}")
    
    def update_file(self, file_info: FileInfo):
        """Update file information in database"""
        self._data["files"][file_info.path] = asdict(file_info)
        self.save_database()
    
    def get_file(self, path: str) -> Optional[FileInfo]:
        """Get file information from database"""
        file_data = self._data["files"].get(path)
        if file_data:
            # Convert string back to datetime
            if isinstance(file_data["modified"], str):
                file_data["modified"] = datetime.fromisoformat(file_data["modified"])
            return FileInfo(**file_data)
        return None
    
    def remove_file(self, path: str):
        """Remove file from database"""
        if path in self._data["files"]:
            del self._data["files"][path]
            self.save_database()
    
    def get_all_files(self) -> Dict[str, FileInfo]:
        """Get all files from database"""
        result = {}
        for path, file_data in self._data["files"].items():
            if isinstance(file_data["modified"], str):
                file_data["modified"] = datetime.fromisoformat(file_data["modified"])
            result[path] = FileInfo(**file_data)
        return result
    
    def update_last_sync(self):
        """Update last synchronization timestamp"""
        self._data["last_sync"] = datetime.now().isoformat()
        self.save_database()
    
    def add_sync_event(self, event: str, details: str = ""):
        """Add synchronization event to history"""
        self._data["sync_history"].append({
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details
        })
        # Keep only last 100 events
        if len(self._data["sync_history"]) > 100:
            self._data["sync_history"] = self._data["sync_history"][-100:]
        self.save_database()
    
    def is_empty(self) -> bool:
        """Check if database is empty (no tracked files or sync history)"""
        return (len(self._data["files"]) == 0 and 
                len(self._data["sync_history"]) == 0)
    
    def is_stale(self, max_age_days: int = 30) -> bool:
        """Check if database is stale (last sync older than max_age_days)"""
        if not self._data["last_sync"]:
            return True
        
        try:
            last_sync = datetime.fromisoformat(self._data["last_sync"])
            age = datetime.now() - last_sync
            return age.days > max_age_days
        except Exception:
            return True
    
    def cleanup_if_needed(self, force_empty: bool = True, max_age_days: int = 30) -> bool:
        """
        Automatically cleanup database if it meets cleanup criteria
        
        Args:
            force_empty: Remove database if it's empty
            max_age_days: Remove database if older than this many days
            
        Returns:
            bool: True if database was cleaned up, False otherwise
        """
        should_cleanup = False
        reason = ""
        
        if force_empty and self.is_empty():
            should_cleanup = True
            reason = "empty database"
        elif self.is_stale(max_age_days):
            should_cleanup = True
            reason = f"stale database (older than {max_age_days} days)"
        
        if should_cleanup:
            try:
                if self.db_path.exists():
                    self.db_path.unlink()
                    logger.info(f"🗑️ Cleaned up sync database: {self.db_path.name} ({reason})")
                    return True
            except Exception as e:
                logger.warning(f"Could not cleanup sync database {self.db_path}: {e}")
        
        return False
    
    def __del__(self):
        """Cleanup empty databases on destruction"""
        try:
            self.cleanup_if_needed(force_empty=True, max_age_days=7)
        except Exception:
            pass  # Ignore cleanup errors during destruction

class FileSystemWatcher(FileSystemEventHandler):
    """File system event handler for real-time monitoring"""
    
    def __init__(self, sync_queue: Queue, config: SyncConfig):
        super().__init__()
        self.sync_queue = sync_queue
        self.config = config
        self.ignore_patterns = config.ignore_patterns
    
    def _should_ignore(self, path: str) -> bool:
        """Check if file should be ignored based on patterns"""
        import fnmatch
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
        return False
    
    def on_modified(self, event):
        if not event.is_directory and not self._should_ignore(event.src_path):
            self.sync_queue.put(("modified", event.src_path))
    
    def on_created(self, event):
        if not self._should_ignore(event.src_path):
            self.sync_queue.put(("created", event.src_path))
    
    def on_deleted(self, event):
        if not self._should_ignore(event.src_path):
            self.sync_queue.put(("deleted", event.src_path))
    
    def on_moved(self, event):
        if not self._should_ignore(event.src_path) and not self._should_ignore(event.dest_path):
            self.sync_queue.put(("moved", event.src_path, event.dest_path))

# Utility function for global cleanup
def cleanup_all_sync_databases(directory: str = ".", max_age_days: int = 30, force_empty: bool = True) -> int:
    """
    Cleanup all sync database files in a directory
    
    Args:
        directory: Directory to search for sync databases
        max_age_days: Remove databases older than this many days
        force_empty: Remove empty databases
        
    Returns:
        int: Number of databases cleaned up
    """
    import glob
    
    cleaned_count = 0
    pattern = os.path.join(directory, ".sync_db_*.json")
    
    for db_file in glob.glob(pattern):
        try:
            # Create temporary database object for cleanup
            temp_db = SyncDatabase(db_file)
            if temp_db.cleanup_if_needed(force_empty=force_empty, max_age_days=max_age_days):
                cleaned_count += 1
        except Exception as e:
            logger.warning(f"Could not cleanup {db_file}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"🧹 Cleaned up {cleaned_count} sync database files")
    
    return cleaned_count

# Enhanced Sync Functions with Events

def create_sync_config_with_events(local_path: str, remote_path: str = "/", **kwargs) -> SyncConfig:
    """Enhanced sync config creation with event callbacks and logging.
    
    Args:
        local_path (str): Local directory path to sync
        remote_path (str): Remote directory path (default: root)
        **kwargs: Additional arguments including:
            - callback_fn (Optional[Callable]): Function to call with sync events
            - log_fn (Optional[Callable]): Function for logging sync operations
            - sync_direction (str): Direction of sync ("up", "down", "bidirectional")
            - conflict_resolution (str): How to handle conflicts ("newer_wins", "local_wins", "remote_wins", "ask")
            - real_time (bool): Enable real-time monitoring
            - sync_interval (int): Sync interval in seconds for periodic sync
            - max_file_size (int): Maximum file size to sync in bytes
            - backup_conflicts (bool): Backup conflicted files
            - ignore_patterns (List[str]): File patterns to ignore during sync
    
    Returns:
        SyncConfig: Configured synchronization settings.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('sync_config_started', {
                'local_path': local_path,
                'remote_path': remote_path
            })
        
        if log_fn:
            log_fn(f"Creating sync config: '{local_path}' ↔ '{remote_path}'")
        
        # Extract config parameters from kwargs
        config_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['callback_fn', 'log_fn']}
        
        config = SyncConfig(local_path=local_path, remote_path=remote_path, **config_kwargs)
        
        if callback_fn:
            callback_fn('sync_config_completed', {
                'local_path': local_path,
                'remote_path': remote_path,
                'sync_direction': config.sync_direction,
                'real_time': config.real_time,
                'success': True
            })
        
        if log_fn:
            log_fn(f"Sync config created successfully: {config.sync_direction} sync between '{local_path}' and '{remote_path}'")
        
        return config
        
    except Exception as e:
        if callback_fn:
            callback_fn('sync_config_error', {
                'error': str(e),
                'local_path': local_path,
                'remote_path': remote_path
            })
        if log_fn:
            log_fn(f"Sync config creation error: {e}")
        raise


def sync_directory_with_events(client_instance, config: SyncConfig, 
                              progress_callback: Optional[Callable] = None, 
                              **kwargs) -> Dict[str, Any]:
    """Enhanced directory synchronization with event callbacks and logging.
    
    Args:
        client_instance: Client instance
        config (SyncConfig): Synchronization configuration
        progress_callback (Optional[Callable]): Progress callback function
        **kwargs: Additional arguments including event callback functions
    
    Returns:
        Dict[str, Any]: Sync results with enhanced statistics and event data.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('directory_sync_started', {
                'local_path': config.local_path,
                'remote_path': config.remote_path,
                'sync_direction': config.sync_direction,
                'real_time': config.real_time
            })
        
        if log_fn:
            log_fn(f"Starting directory sync: '{config.local_path}' ↔ '{config.remote_path}' ({config.sync_direction})")
        
        # NOTE: AdvancedSynchronizer was removed - this function is broken
        return {
            "success": False,
            "error": "AdvancedSynchronizer class removed",
            "message": "sync_directory_with_events is no longer functional"
        }
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "message": f"Sync failed: {e}"
        }
        
        if callback_fn:
            callback_fn('directory_sync_error', {
                'error': str(e),
                'local_path': config.local_path,
                'remote_path': config.remote_path
            })
        if log_fn:
            log_fn(f"Directory sync error: {e}")
        
        return error_result


def start_real_time_sync_with_events(client_instance, config: SyncConfig, **kwargs) -> Dict[str, Any]:
    """Enhanced real-time sync startup with event callbacks and logging.
    
    Args:
        client_instance: Client instance
        config (SyncConfig): Synchronization configuration
        **kwargs: Event callback functions
    
    Returns:
        Dict[str, Any]: Operation result with enhanced sync information.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('realtime_sync_start_requested', {
                'local_path': config.local_path,
                'remote_path': config.remote_path,
                'sync_direction': config.sync_direction
            })
        
        if log_fn:
            log_fn(f"Starting real-time sync: '{config.local_path}' ↔ '{config.remote_path}'")
        
        if not hasattr(client_instance, '_sync_instances'):
            client_instance._sync_instances = {}
        
        sync_key = f"{config.local_path}::{config.remote_path}"
        
        if sync_key in client_instance._sync_instances:
            result = {
                "success": False,
                "sync_key": sync_key,
                "message": "Real-time sync already running for this path pair"
            }
            
            if callback_fn:
                callback_fn('realtime_sync_already_running', {
                    'sync_key': sync_key,
                    'local_path': config.local_path,
                    'remote_path': config.remote_path
                })
            
            return result
        
        # NOTE: AdvancedSynchronizer was removed - this function is broken
        return {
            "success": False,
            "error": "AdvancedSynchronizer class removed", 
            "message": "start_real_time_sync_with_events is no longer functional"
        }
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "message": f"Error starting real-time sync: {e}"
        }
        
        if callback_fn:
            callback_fn('realtime_sync_start_error', {
                'error': str(e),
                'local_path': config.local_path,
                'remote_path': config.remote_path
            })
        if log_fn:
            log_fn(f"Real-time sync start error: {e}")
        
        return result


def stop_real_time_sync_with_events(client_instance, sync_key: str, **kwargs) -> Dict[str, Any]:
    """Enhanced real-time sync stopping with event callbacks and logging.
    
    Args:
        client_instance: Client instance
        sync_key (str): Sync instance key from start_real_time_sync
        **kwargs: Event callback functions
    
    Returns:
        Dict[str, Any]: Operation result with enhanced information.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('realtime_sync_stop_requested', {'sync_key': sync_key})
        
        if log_fn:
            log_fn(f"Stopping real-time sync: {sync_key}")
        
        if not hasattr(client_instance, '_sync_instances'):
            result = {
                "success": False,
                "sync_key": sync_key,
                "message": "No sync instances found"
            }
            
            if callback_fn:
                callback_fn('realtime_sync_no_instances', {'sync_key': sync_key})
            
            return result
        
        synchronizer = client_instance._sync_instances.get(sync_key)
        if not synchronizer:
            result = {
                "success": False,
                "sync_key": sync_key,
                "message": "Sync instance not found"
            }
            
            if callback_fn:
                callback_fn('realtime_sync_not_found', {'sync_key': sync_key})
            
            return result
        
        # Get final stats before stopping
        final_stats = synchronizer.stats.copy() if hasattr(synchronizer, 'stats') else {}
        
        synchronizer.stop_real_time_sync()
        del client_instance._sync_instances[sync_key]
        
        result = {
            "success": True,
            "sync_key": sync_key,
            "final_stats": final_stats,
            "message": "Real-time sync stopped successfully"
        }
        
        if callback_fn:
            callback_fn('realtime_sync_stopped', {
                'sync_key': sync_key,
                'final_stats': final_stats,
                'success': True
            })
        
        if log_fn:
            log_fn(f"Real-time sync stopped successfully: {sync_key}")
        
        return result
        
    except Exception as e:
        result = {
            "success": False,
            "sync_key": sync_key,
            "error": str(e),
            "message": f"Error stopping real-time sync: {e}"
        }
        
        if callback_fn:
            callback_fn('realtime_sync_stop_error', {
                'error': str(e),
                'sync_key': sync_key
            })
        if log_fn:
            log_fn(f"Real-time sync stop error: {e}")
        
        return result


def get_sync_status_with_events(client_instance, sync_key: str, **kwargs) -> Dict[str, Any]:
    """Enhanced sync status retrieval with event callbacks and logging.
    
    Args:
        client_instance: Client instance
        sync_key (str): Sync instance key
        **kwargs: Event callback functions
    
    Returns:
        Dict[str, Any]: Enhanced sync status information.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('sync_status_requested', {'sync_key': sync_key})
        
        if log_fn:
            log_fn(f"Getting sync status for: {sync_key}")
        
        if not hasattr(client_instance, '_sync_instances'):
            result = {
                "success": False,
                "sync_key": sync_key,
                "message": "No sync instances found"
            }
            
            if callback_fn:
                callback_fn('sync_status_no_instances', {'sync_key': sync_key})
            
            return result
        
        synchronizer = client_instance._sync_instances.get(sync_key)
        if not synchronizer:
            result = {
                "success": False,
                "sync_key": sync_key,
                "message": "Sync instance not found"
            }
            
            if callback_fn:
                callback_fn('sync_status_not_found', {'sync_key': sync_key})
            
            return result
        
        status = synchronizer.get_sync_status()
        
        # Enhanced status with additional information
        enhanced_status = {
            "success": True,
            "sync_key": sync_key,
            "status": status,
            "config": asdict(synchronizer.config),
            "running": getattr(synchronizer, 'running', False),
            "stats": getattr(synchronizer, 'stats', {}),
            "watchdog_available": WATCHDOG_AVAILABLE
        }
        
        if callback_fn:
            callback_fn('sync_status_retrieved', {
                'sync_key': sync_key,
                'running': enhanced_status['running'],
                'stats': enhanced_status['stats'],
                'success': True
            })
        
        if log_fn:
            log_fn(f"Sync status retrieved for {sync_key}: {'running' if enhanced_status['running'] else 'stopped'}")
        
        return enhanced_status
        
    except Exception as e:
        result = {
            "success": False,
            "sync_key": sync_key,
            "error": str(e),
            "message": f"Error getting sync status: {e}"
        }
        
        if callback_fn:
            callback_fn('sync_status_error', {
                'error': str(e),
                'sync_key': sync_key
            })
        if log_fn:
            log_fn(f"Sync status error for {sync_key}: {e}")
        
        return result


def list_sync_instances_with_events(client_instance, **kwargs) -> Dict[str, Any]:
    """Enhanced sync instances listing with event callbacks and logging.
    
    Args:
        client_instance: Client instance
        **kwargs: Event callback functions
    
    Returns:
        Dict[str, Any]: Enhanced list of active sync instances.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('sync_list_requested', {})
        
        if log_fn:
            log_fn("Listing all sync instances")
        
        if not hasattr(client_instance, '_sync_instances'):
            result = {
                "success": True,
                "instances": {},
                "total_count": 0
            }
            
            if callback_fn:
                callback_fn('sync_list_completed', {'count': 0})
            
            return result
        
        instances = {}
        total_running = 0
        total_stopped = 0
        
        for key, synchronizer in client_instance._sync_instances.items():
            is_running = getattr(synchronizer, 'running', False)
            if is_running:
                total_running += 1
            else:
                total_stopped += 1
            
            instances[key] = {
                "running": is_running,
                "config": asdict(synchronizer.config),
                "stats": getattr(synchronizer, 'stats', {}),
                "watchdog_available": WATCHDOG_AVAILABLE
            }
        
        result = {
            "success": True,
            "instances": instances,
            "total_count": len(instances),
            "running_count": total_running,
            "stopped_count": total_stopped,
            "watchdog_available": WATCHDOG_AVAILABLE
        }
        
        if callback_fn:
            callback_fn('sync_list_completed', {
                'count': result['total_count'],
                'running': result['running_count'],
                'stopped': result['stopped_count'],
                'success': True
            })
        
        if log_fn:
            log_fn(f"Listed {result['total_count']} sync instances: {result['running_count']} running, {result['stopped_count']} stopped")
        
        return result
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "message": f"Error listing sync instances: {e}"
        }
        
        if callback_fn:
            callback_fn('sync_list_error', {'error': str(e)})
        if log_fn:
            log_fn(f"Sync list error: {e}")
        
        return result


def cleanup_sync_databases_with_events(max_age_days: int = 30, 
                                      force_empty: bool = True, 
                                      **kwargs) -> Dict[str, Any]:
    """Enhanced sync database cleanup with event callbacks and logging.
    
    Args:
        max_age_days (int): Remove databases older than this many days
        force_empty (bool): Remove empty databases
        **kwargs: Event callback functions
    
    Returns:
        Dict[str, Any]: Enhanced cleanup results.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('sync_cleanup_started', {
                'max_age_days': max_age_days,
                'force_empty': force_empty
            })
        
        if log_fn:
            log_fn(f"Starting sync database cleanup: max_age={max_age_days} days, force_empty={force_empty}")
        
        cleaned_count = cleanup_all_sync_databases(
            directory=".", 
            max_age_days=max_age_days, 
            force_empty=force_empty
        )
        
        result = {
            "success": True,
            "cleaned_count": cleaned_count,
            "max_age_days": max_age_days,
            "force_empty": force_empty,
            "message": f"Cleaned up {cleaned_count} sync database files"
        }
        
        if callback_fn:
            callback_fn('sync_cleanup_completed', {
                'cleaned_count': cleaned_count,
                'max_age_days': max_age_days,
                'success': True
            })
        
        if log_fn:
            log_fn(f"Sync database cleanup completed: {cleaned_count} files cleaned")
        
        return result
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "message": f"Error during cleanup: {e}"
        }
        
        if callback_fn:
            callback_fn('sync_cleanup_error', {'error': str(e)})
        if log_fn:
            log_fn(f"Sync cleanup error: {e}")
        
        return result

