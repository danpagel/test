"""
Synchronization and Transfer module for MegaPythonLibrary.

This module contains:
- File transfer operations with progress tracking
- Chunked upload and download
- Transfer queue management
- Progress monitoring and callbacks
- Bandwidth management
"""

import time
from typing import Optional, Callable, Dict, Any, List, Tuple
from pathlib import Path

from .monitor import get_logger, trigger_event, MonitoredOperation, record_performance
from .network import upload_chunk, download_chunk, get_upload_url, get_download_url, apply_bandwidth_throttling
from .auth import current_session, require_authentication
from .utils import RequestError, get_chunks

# ==============================================
# === PROGRESS TRACKING ===
# ==============================================

class TransferProgress:
    """Track progress of file transfers."""
    
    def __init__(self, total_size: int, operation: str = "transfer"):
        self.total_size = total_size
        self.transferred_size = 0
        self.operation = operation
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.speed = 0.0  # bytes per second
        self.eta = 0  # estimated time remaining
        self.logger = get_logger("transfer")
    
    def update(self, bytes_transferred: int) -> None:
        """Update transfer progress."""
        self.transferred_size += bytes_transferred
        current_time = time.time()
        
        # Calculate speed (moving average)
        time_diff = current_time - self.last_update_time
        if time_diff > 0:
            instant_speed = bytes_transferred / time_diff
            # Simple moving average
            self.speed = (self.speed * 0.8) + (instant_speed * 0.2)
        
        # Calculate ETA
        if self.speed > 0:
            remaining_bytes = self.total_size - self.transferred_size
            self.eta = remaining_bytes / self.speed
        
        self.last_update_time = current_time
        
        # Record performance metric
        record_performance(f"{self.operation}_speed", self.speed)
        
        # Trigger progress event
        trigger_event(f"{self.operation}_progress", {
            'total_size': self.total_size,
            'transferred_size': self.transferred_size,
            'percentage': self.get_percentage(),
            'speed': self.speed,
            'eta': self.eta
        })
    
    def get_percentage(self) -> float:
        """Get transfer percentage."""
        if self.total_size == 0:
            return 100.0
        return (self.transferred_size / self.total_size) * 100.0
    
    def get_elapsed_time(self) -> float:
        """Get elapsed transfer time."""
        return time.time() - self.start_time
    
    def is_complete(self) -> bool:
        """Check if transfer is complete."""
        return self.transferred_size >= self.total_size


# ==============================================
# === CHUNKED UPLOAD ===
# ==============================================

@require_authentication
def upload_file_chunked(file_path: str, upload_url: str, 
                       progress_callback: Optional[Callable] = None,
                       chunk_size: int = 1024 * 1024) -> Dict[str, Any]:
    """
    Upload file in chunks with progress tracking.
    
    Args:
        file_path: Path to file to upload
        upload_url: Upload URL from MEGA
        progress_callback: Optional progress callback function
        chunk_size: Size of each chunk in bytes
        
    Returns:
        Upload result information
        
    Raises:
        RequestError: If upload fails
    """
    logger = get_logger("upload")
    
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise RequestError(f"File not found: {file_path}")
        
        file_size = file_path_obj.stat().st_size
        chunks = get_chunks(file_size)
        
        # Initialize progress tracking
        progress = TransferProgress(file_size, "upload")
        
        logger.info(f"Starting chunked upload of {file_path} ({file_size} bytes, {len(chunks)} chunks)")
        trigger_event('upload_started', {
            'file_path': file_path,
            'file_size': file_size,
            'chunks_count': len(chunks)
        })
        
        with MonitoredOperation("file_upload"):
            with open(file_path, 'rb') as f:
                upload_results = []
                
                for i, (chunk_start, chunk_end) in enumerate(chunks):
                    # Read chunk
                    f.seek(chunk_start)
                    chunk_data = f.read(chunk_end - chunk_start)
                    chunk_size_actual = len(chunk_data)
                    
                    logger.debug(f"Uploading chunk {i+1}/{len(chunks)}: {chunk_start}-{chunk_end}")
                    
                    # Upload chunk
                    try:
                        result = upload_chunk(upload_url, chunk_data, chunk_start)
                        upload_results.append(result)
                        
                        # Update progress
                        progress.update(chunk_size_actual)
                        
                        # Apply bandwidth throttling
                        apply_bandwidth_throttling(chunk_size_actual)
                        
                        # Call progress callback
                        if progress_callback:
                            progress_callback(progress.transferred_size, progress.total_size)
                        
                    except Exception as e:
                        logger.error(f"Failed to upload chunk {i+1}: {e}")
                        trigger_event('upload_failed', {
                            'file_path': file_path,
                            'error': str(e),
                            'chunk_index': i
                        })
                        raise RequestError(f"Chunk upload failed: {e}")
                
                logger.info(f"Upload completed: {file_path} ({progress.get_elapsed_time():.2f}s)")
                trigger_event('upload_completed', {
                    'file_path': file_path,
                    'file_size': file_size,
                    'elapsed_time': progress.get_elapsed_time(),
                    'average_speed': file_size / progress.get_elapsed_time()
                })
                
                return {
                    'file_path': file_path,
                    'file_size': file_size,
                    'chunks_uploaded': len(chunks),
                    'elapsed_time': progress.get_elapsed_time(),
                    'average_speed': file_size / progress.get_elapsed_time(),
                    'upload_results': upload_results
                }
                
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        trigger_event('upload_failed', {
            'file_path': file_path,
            'error': str(e)
        })
        raise


# ==============================================
# === CHUNKED DOWNLOAD ===
# ==============================================

def download_file_chunked(download_url: str, output_path: str, file_size: int,
                         progress_callback: Optional[Callable] = None,
                         chunk_size: int = 1024 * 1024) -> Dict[str, Any]:
    """
    Download file in chunks with progress tracking.
    
    Args:
        download_url: Download URL from MEGA
        output_path: Path where file should be saved
        file_size: Size of file to download
        progress_callback: Optional progress callback function
        chunk_size: Size of each chunk in bytes
        
    Returns:
        Download result information
        
    Raises:
        RequestError: If download fails
    """
    logger = get_logger("download")
    
    try:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        chunks = get_chunks(file_size)
        
        # Initialize progress tracking
        progress = TransferProgress(file_size, "download")
        
        logger.info(f"Starting chunked download to {output_path} ({file_size} bytes, {len(chunks)} chunks)")
        trigger_event('download_started', {
            'output_path': output_path,
            'file_size': file_size,
            'chunks_count': len(chunks)
        })
        
        with MonitoredOperation("file_download"):
            with open(output_path, 'wb') as f:
                for i, (chunk_start, chunk_end) in enumerate(chunks):
                    logger.debug(f"Downloading chunk {i+1}/{len(chunks)}: {chunk_start}-{chunk_end}")
                    
                    try:
                        # Download chunk
                        chunk_data = download_chunk(download_url, chunk_start, chunk_end - 1)
                        chunk_size_actual = len(chunk_data)
                        
                        # Write chunk to file
                        f.write(chunk_data)
                        
                        # Update progress
                        progress.update(chunk_size_actual)
                        
                        # Apply bandwidth throttling
                        apply_bandwidth_throttling(chunk_size_actual)
                        
                        # Call progress callback
                        if progress_callback:
                            progress_callback(progress.transferred_size, progress.total_size)
                        
                    except Exception as e:
                        logger.error(f"Failed to download chunk {i+1}: {e}")
                        trigger_event('download_failed', {
                            'output_path': output_path,
                            'error': str(e),
                            'chunk_index': i
                        })
                        # Clean up partial file
                        if output_path_obj.exists():
                            output_path_obj.unlink()
                        raise RequestError(f"Chunk download failed: {e}")
                
                logger.info(f"Download completed: {output_path} ({progress.get_elapsed_time():.2f}s)")
                trigger_event('download_completed', {
                    'output_path': output_path,
                    'file_size': file_size,
                    'elapsed_time': progress.get_elapsed_time(),
                    'average_speed': file_size / progress.get_elapsed_time()
                })
                
                return {
                    'output_path': output_path,
                    'file_size': file_size,
                    'chunks_downloaded': len(chunks),
                    'elapsed_time': progress.get_elapsed_time(),
                    'average_speed': file_size / progress.get_elapsed_time()
                }
                
    except Exception as e:
        logger.error(f"Download failed: {e}")
        trigger_event('download_failed', {
            'output_path': output_path,
            'error': str(e)
        })
        raise


# ==============================================
# === TRANSFER QUEUE MANAGEMENT ===
# ==============================================

class TransferQueue:
    """Manage a queue of file transfers."""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.upload_queue: List[Dict[str, Any]] = []
        self.download_queue: List[Dict[str, Any]] = []
        self.active_transfers: List[Dict[str, Any]] = []
        self.completed_transfers: List[Dict[str, Any]] = []
        self.failed_transfers: List[Dict[str, Any]] = []
        self.logger = get_logger("transfer_queue")
    
    def add_upload(self, file_path: str, upload_url: str, 
                  progress_callback: Optional[Callable] = None) -> str:
        """Add upload to queue."""
        transfer_id = f"upload_{len(self.upload_queue)}_{int(time.time())}"
        
        transfer_info = {
            'id': transfer_id,
            'type': 'upload',
            'file_path': file_path,
            'upload_url': upload_url,
            'progress_callback': progress_callback,
            'status': 'queued',
            'added_time': time.time()
        }
        
        self.upload_queue.append(transfer_info)
        self.logger.info(f"Added upload to queue: {file_path}")
        
        return transfer_id
    
    def add_download(self, download_url: str, output_path: str, file_size: int,
                    progress_callback: Optional[Callable] = None) -> str:
        """Add download to queue."""
        transfer_id = f"download_{len(self.download_queue)}_{int(time.time())}"
        
        transfer_info = {
            'id': transfer_id,
            'type': 'download',
            'download_url': download_url,
            'output_path': output_path,
            'file_size': file_size,
            'progress_callback': progress_callback,
            'status': 'queued',
            'added_time': time.time()
        }
        
        self.download_queue.append(transfer_info)
        self.logger.info(f"Added download to queue: {output_path}")
        
        return transfer_id
    
    def process_queue(self) -> None:
        """Process transfers in the queue."""
        while (len(self.active_transfers) < self.max_concurrent and 
               (self.upload_queue or self.download_queue)):
            
            # Process uploads first
            if self.upload_queue:
                transfer = self.upload_queue.pop(0)
                self._start_transfer(transfer)
            elif self.download_queue:
                transfer = self.download_queue.pop(0)
                self._start_transfer(transfer)
    
    def _start_transfer(self, transfer: Dict[str, Any]) -> None:
        """Start a transfer."""
        transfer['status'] = 'active'
        transfer['start_time'] = time.time()
        self.active_transfers.append(transfer)
        
        try:
            if transfer['type'] == 'upload':
                result = upload_file_chunked(
                    transfer['file_path'],
                    transfer['upload_url'],
                    transfer['progress_callback']
                )
            else:  # download
                result = download_file_chunked(
                    transfer['download_url'],
                    transfer['output_path'],
                    transfer['file_size'],
                    transfer['progress_callback']
                )
            
            # Transfer completed
            transfer['status'] = 'completed'
            transfer['result'] = result
            transfer['end_time'] = time.time()
            
            self.active_transfers.remove(transfer)
            self.completed_transfers.append(transfer)
            
            self.logger.info(f"Transfer completed: {transfer['id']}")
            
        except Exception as e:
            # Transfer failed
            transfer['status'] = 'failed'
            transfer['error'] = str(e)
            transfer['end_time'] = time.time()
            
            self.active_transfers.remove(transfer)
            self.failed_transfers.append(transfer)
            
            self.logger.error(f"Transfer failed: {transfer['id']}: {e}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of transfer queue."""
        return {
            'queued_uploads': len(self.upload_queue),
            'queued_downloads': len(self.download_queue),
            'active_transfers': len(self.active_transfers),
            'completed_transfers': len(self.completed_transfers),
            'failed_transfers': len(self.failed_transfers),
            'total_queued': len(self.upload_queue) + len(self.download_queue),
        }
    
    def clear_completed(self) -> None:
        """Clear completed transfers from memory."""
        self.completed_transfers.clear()
        self.failed_transfers.clear()
        self.logger.info("Cleared completed transfers")


# Global transfer queue
_transfer_queue = TransferQueue()


# ==============================================
# === SYNCHRONIZATION UTILITIES ===
# ==============================================

def sync_folder(local_folder: str, remote_folder_handle: str,
               progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Synchronize a local folder with a remote MEGA folder.
    
    Args:
        local_folder: Path to local folder
        remote_folder_handle: Handle of remote folder
        progress_callback: Optional progress callback
        
    Returns:
        Synchronization results
    """
    logger = get_logger("sync")
    
    # This is a placeholder for folder synchronization
    # Full implementation would require filesystem comparison
    # and conflict resolution strategies
    
    logger.info(f"Starting folder sync: {local_folder} -> {remote_folder_handle}")
    
    return {
        'status': 'not_implemented',
        'message': 'Folder synchronization not fully implemented in this version'
    }


# ==============================================
# === TRANSFER UTILITIES ===
# ==============================================

def estimate_transfer_time(file_size: int, speed: float) -> float:
    """
    Estimate transfer time based on file size and speed.
    
    Args:
        file_size: Size of file in bytes
        speed: Transfer speed in bytes per second
        
    Returns:
        Estimated time in seconds
    """
    if speed <= 0:
        return float('inf')
    return file_size / speed


def format_transfer_speed(bytes_per_second: float) -> str:
    """
    Format transfer speed for display.
    
    Args:
        bytes_per_second: Speed in bytes per second
        
    Returns:
        Formatted speed string
    """
    if bytes_per_second < 1024:
        return f"{bytes_per_second:.1f} B/s"
    elif bytes_per_second < 1024 * 1024:
        return f"{bytes_per_second / 1024:.1f} KB/s"
    elif bytes_per_second < 1024 * 1024 * 1024:
        return f"{bytes_per_second / (1024 * 1024):.1f} MB/s"
    else:
        return f"{bytes_per_second / (1024 * 1024 * 1024):.1f} GB/s"


def format_eta(seconds: float) -> str:
    """
    Format estimated time remaining for display.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds == float('inf') or seconds < 0:
        return "Unknown"
    
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m {int(seconds % 60)}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


# ==============================================
# === QUEUE MANAGEMENT FUNCTIONS ===
# ==============================================

def add_upload_to_queue(file_path: str, upload_url: str,
                       progress_callback: Optional[Callable] = None) -> str:
    """Add upload to global transfer queue."""
    return _transfer_queue.add_upload(file_path, upload_url, progress_callback)


def add_download_to_queue(download_url: str, output_path: str, file_size: int,
                         progress_callback: Optional[Callable] = None) -> str:
    """Add download to global transfer queue."""
    return _transfer_queue.add_download(download_url, output_path, file_size, progress_callback)


def process_transfer_queue() -> None:
    """Process the global transfer queue."""
    _transfer_queue.process_queue()


def get_transfer_queue_status() -> Dict[str, Any]:
    """Get status of global transfer queue."""
    return _transfer_queue.get_queue_status()


def clear_completed_transfers() -> None:
    """Clear completed transfers from global queue."""
    _transfer_queue.clear_completed()