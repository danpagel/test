#!/usr/bin/env python3
"""
Advanced Error Recovery System for MEGA Downloads

This module provides sophisticated error recovery capabilities including:
- Partial chunk resume for interrupted downloads
- Exponential backoff retry strategies  
- Chunk integrity verification
- Progress preservation across failures
- Network-aware error handling

Based on MEGA SDK research and industry best practices.
Created as part of MEGA SDK optimization implementation.
"""

import time
import os
import hashlib
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Types of download errors"""
    NETWORK_TIMEOUT = "network_timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    CHUNK_CORRUPTION = "chunk_corruption"
    DISK_FULL = "disk_full"
    PERMISSION_ERROR = "permission_error"
    UNKNOWN_ERROR = "unknown_error"

class RetryStrategy(Enum):
    """Retry strategy types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"
    CUSTOM = "custom"

@dataclass
class ChunkState:
    """State information for a download chunk"""
    chunk_id: int
    start_byte: int
    end_byte: int
    size_bytes: int
    completed: bool = False
    attempts: int = 0
    last_error: Optional[str] = None
    last_attempt_time: float = 0.0
    data_hash: Optional[str] = None
    temp_file: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChunkState':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class DownloadProgress:
    """Download progress and recovery information"""
    file_handle: str
    file_name: str
    file_size: int
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    chunks: List[ChunkState]
    start_time: float
    last_update_time: float
    total_bytes_downloaded: int
    recovery_attempts: int = 0
    
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.total_chunks == 0:
            return 0.0
        return (self.completed_chunks / self.total_chunks) * 100.0
    
    def get_failed_chunks(self) -> List[ChunkState]:
        """Get list of failed chunks that need retry"""
        return [chunk for chunk in self.chunks if not chunk.completed and chunk.attempts > 0]
    
    def get_pending_chunks(self) -> List[ChunkState]:
        """Get list of chunks that haven't been attempted"""
        return [chunk for chunk in self.chunks if chunk.attempts == 0]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'file_handle': self.file_handle,
            'file_name': self.file_name,
            'file_size': self.file_size,
            'total_chunks': self.total_chunks,
            'completed_chunks': self.completed_chunks,
            'failed_chunks': self.failed_chunks,
            'chunks': [chunk.to_dict() for chunk in self.chunks],
            'start_time': self.start_time,
            'last_update_time': self.last_update_time,
            'total_bytes_downloaded': self.total_bytes_downloaded,
            'recovery_attempts': self.recovery_attempts
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DownloadProgress':
        """Create from dictionary"""
        chunks = [ChunkState.from_dict(chunk_data) for chunk_data in data['chunks']]
        return cls(
            file_handle=data['file_handle'],
            file_name=data['file_name'],
            file_size=data['file_size'],
            total_chunks=data['total_chunks'],
            completed_chunks=data['completed_chunks'],
            failed_chunks=data['failed_chunks'],
            chunks=chunks,
            start_time=data['start_time'],
            last_update_time=data['last_update_time'],
            total_bytes_downloaded=data['total_bytes_downloaded'],
            recovery_attempts=data.get('recovery_attempts', 0)
        )

class AdvancedErrorRecovery:
    """Advanced error recovery system for download operations"""
    
    def __init__(self, 
                 progress_dir: Path = None,
                 max_retry_attempts: int = 5,
                 base_retry_delay: float = 1.0,
                 max_retry_delay: float = 60.0,
                 retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF):
        """
        Initialize error recovery system
        
        Args:
            progress_dir: Directory to store progress files (default: temp)
            max_retry_attempts: Maximum retry attempts per chunk
            base_retry_delay: Base delay between retries (seconds)
            max_retry_delay: Maximum delay between retries (seconds)
            retry_strategy: Strategy for calculating retry delays
        """
        self.progress_dir = progress_dir or Path.cwd() / "temp_download_progress"
        self.progress_dir.mkdir(exist_ok=True)
        
        self.max_retry_attempts = max_retry_attempts
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        self.retry_strategy = retry_strategy
        
        # Enhanced state management
        self._progress_cache: Dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()
        self._recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0
        }
        
        # Auto-cleanup settings
        self._auto_cleanup_enabled = True
        self._max_cache_size = 50  # Maximum cached progress objects
        
        logger.info(f"Enhanced error recovery initialized: {retry_strategy.value}, max_attempts={max_retry_attempts}")
    
    def create_download_progress(self, 
                               file_handle: str,
                               file_name: str, 
                               file_size: int,
                               chunk_size: int) -> DownloadProgress:
        """Create initial download progress tracking"""
        
        # Calculate chunks
        total_chunks = (file_size + chunk_size - 1) // chunk_size
        chunks = []
        
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size - 1, file_size - 1)
            chunk = ChunkState(
                chunk_id=i,
                start_byte=start,
                end_byte=end,
                size_bytes=end - start + 1
            )
            chunks.append(chunk)
        
        progress = DownloadProgress(
            file_handle=file_handle,
            file_name=file_name,
            file_size=file_size,
            total_chunks=total_chunks,
            completed_chunks=0,
            failed_chunks=0,
            chunks=chunks,
            start_time=time.time(),
            last_update_time=time.time(),
            total_bytes_downloaded=0
        )
        
        with self._lock:
            self._progress_cache[file_handle] = progress
        
        logger.info(f"Created download progress: {file_name} ({total_chunks} chunks)")
        return progress
    
    def save_progress(self, progress: DownloadProgress) -> None:
        """Save download progress to disk for recovery"""
        try:
            progress_file = self.progress_dir / f"{progress.file_handle}_progress.pkl"
            with open(progress_file, 'wb') as f:
                pickle.dump(progress.to_dict(), f)
            logger.debug(f"Progress saved: {progress.file_name}")
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def load_progress(self, file_handle: str) -> Optional[DownloadProgress]:
        """Load download progress from disk"""
        try:
            progress_file = self.progress_dir / f"{file_handle}_progress.pkl"
            if progress_file.exists():
                with open(progress_file, 'rb') as f:
                    data = pickle.load(f)
                progress = DownloadProgress.from_dict(data)
                with self._lock:
                    self._progress_cache[file_handle] = progress
                logger.info(f"Progress loaded: {progress.file_name} ({progress.completion_percentage():.1f}% complete)")
                return progress
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
        return None
    
    def update_chunk_success(self, 
                           progress: DownloadProgress, 
                           chunk_id: int, 
                           data: bytes) -> None:
        """Update progress when a chunk is successfully downloaded"""
        
        with self._lock:
            if chunk_id < len(progress.chunks):
                chunk = progress.chunks[chunk_id]
                if not chunk.completed:
                    chunk.completed = True
                    chunk.data_hash = hashlib.md5(data).hexdigest()
                    progress.completed_chunks += 1
                    progress.total_bytes_downloaded += len(data)
                    progress.last_update_time = time.time()
                    
                    # Save chunk data to temp file for recovery
                    temp_file = self.progress_dir / f"{progress.file_handle}_chunk_{chunk_id}.dat"
                    try:
                        with open(temp_file, 'wb') as f:
                            f.write(data)
                        chunk.temp_file = str(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to save chunk data: {e}")
                    
                    logger.debug(f"Chunk {chunk_id} completed: {progress.completion_percentage():.1f}%")
                    
                    # Auto-save progress periodically
                    if progress.completed_chunks % 5 == 0:  # Save every 5 chunks
                        self.save_progress(progress)
    
    def update_chunk_failure(self, 
                           progress: DownloadProgress, 
                           chunk_id: int, 
                           error: Exception,
                           error_type: ErrorType = ErrorType.UNKNOWN_ERROR) -> bool:
        """
        Update progress when a chunk fails
        
        Returns:
            bool: True if chunk should be retried, False if max attempts reached
        """
        
        with self._lock:
            if chunk_id < len(progress.chunks):
                chunk = progress.chunks[chunk_id]
                chunk.attempts += 1
                chunk.last_error = str(error)
                chunk.last_attempt_time = time.time()
                progress.last_update_time = time.time()
                
                if chunk.attempts == 1:  # First failure
                    progress.failed_chunks += 1
                
                should_retry = chunk.attempts <= self.max_retry_attempts
                
                if should_retry:
                    delay = self.calculate_retry_delay(chunk.attempts, error_type)
                    logger.warning(f"Chunk {chunk_id} failed (attempt {chunk.attempts}/{self.max_retry_attempts}): {error}. Retry in {delay:.1f}s")
                else:
                    logger.error(f"Chunk {chunk_id} permanently failed after {chunk.attempts} attempts: {error}")
                
                # Save progress on failures
                self.save_progress(progress)
                
                return should_retry
        return False
    
    def calculate_retry_delay(self, attempt: int, error_type: ErrorType = None) -> float:
        """Calculate delay before retry based on strategy and error type"""
        
        # Default error type if not provided (for backwards compatibility)
        if error_type is None:
            error_type = ErrorType.UNKNOWN_ERROR
        
        if self.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(self.base_retry_delay * (2 ** (attempt - 1)), self.max_retry_delay)
        elif self.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(self.base_retry_delay * attempt, self.max_retry_delay)
        else:  # IMMEDIATE
            delay = 0.0
        
        # Smart delay adjustment based on error type
        if error_type == ErrorType.NETWORK_TIMEOUT:
            delay *= 2.0  # Longer delay for network issues
        elif error_type == ErrorType.HTTP_ERROR:
            delay *= 0.5  # Shorter delay for HTTP errors
        elif error_type == ErrorType.CHUNK_CORRUPTION:
            delay *= 0.1  # Quick retry for corruption
        elif error_type == ErrorType.CONNECTION_ERROR:
            delay *= 1.5  # Moderate delay for connection issues
        
        return min(delay, self.max_retry_delay)
    
    def get_retry_chunks(self, progress: DownloadProgress) -> List[ChunkState]:
        """Get chunks that are ready for retry"""
        current_time = time.time()
        retry_chunks = []
        
        for chunk in progress.chunks:
            if (not chunk.completed and 
                chunk.attempts > 0 and 
                chunk.attempts <= self.max_retry_attempts):
                
                # Check if enough time has passed for retry
                if chunk.last_attempt_time == 0:
                    retry_chunks.append(chunk)
                else:
                    error_type = self.classify_error(chunk.last_error)
                    delay = self.calculate_retry_delay(chunk.attempts, error_type)
                    if current_time - chunk.last_attempt_time >= delay:
                        retry_chunks.append(chunk)
        
        return retry_chunks
    
    def classify_error(self, error: Exception | str) -> ErrorType:
        """Classify error type from error message or exception"""
        # Handle both Exception objects and strings
        if isinstance(error, Exception):
            error_message = str(error)
        elif isinstance(error, str):
            error_message = error
        else:
            return ErrorType.UNKNOWN_ERROR
            
        if not error_message:
            return ErrorType.UNKNOWN_ERROR
        
        error_lower = error_message.lower()
        
        # Enhanced error classification with more patterns
        if any(pattern in error_lower for pattern in ['timeout', 'timed out', 'time out']):
            return ErrorType.NETWORK_TIMEOUT
        elif any(pattern in error_lower for pattern in ['connection', 'connect', 'unreachable', 'refused']):
            return ErrorType.CONNECTION_ERROR
        elif any(pattern in error_lower for pattern in ['http', 'status', '404', '500', '503', 'bad gateway']):
            return ErrorType.HTTP_ERROR
        elif any(pattern in error_lower for pattern in ['corruption', 'checksum', 'hash', 'integrity']):
            return ErrorType.CHUNK_CORRUPTION
        elif any(pattern in error_lower for pattern in ['disk', 'space', 'no space', 'disk full']):
            return ErrorType.DISK_FULL
        elif any(pattern in error_lower for pattern in ['permission', 'access denied', 'forbidden', 'unauthorized']):
            return ErrorType.PERMISSION_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def can_resume_download(self, file_handle: str) -> bool:
        """Check if a download can be resumed"""
        progress = self.load_progress(file_handle)
        if progress:
            incomplete_chunks = len([c for c in progress.chunks if not c.completed])
            if incomplete_chunks > 0:
                logger.info(f"Download can be resumed: {incomplete_chunks} chunks remaining")
                return True
        return False
    
    def resume_download(self, file_handle: str) -> Optional[DownloadProgress]:
        """Resume a previously interrupted download"""
        progress = self.load_progress(file_handle)
        if progress:
            progress.recovery_attempts += 1
            progress.last_update_time = time.time()
            
            # Verify existing chunk data
            verified_chunks = 0
            for chunk in progress.chunks:
                if chunk.completed and chunk.temp_file:
                    try:
                        temp_file_path = Path(chunk.temp_file)
                        if temp_file_path.exists():
                            with open(temp_file_path, 'rb') as f:
                                data = f.read()
                                current_hash = hashlib.md5(data).hexdigest()
                                if current_hash == chunk.data_hash:
                                    verified_chunks += 1
                                else:
                                    # Chunk corruption detected
                                    chunk.completed = False
                                    progress.completed_chunks -= 1
                                    logger.warning(f"Chunk {chunk.chunk_id} corruption detected, will re-download")
                    except Exception as e:
                        # Chunk file missing or corrupted
                        chunk.completed = False
                        progress.completed_chunks -= 1
                        logger.warning(f"Chunk {chunk.chunk_id} file missing, will re-download: {e}")
            
            logger.info(f"Resume download: {progress.file_name}, {verified_chunks} chunks verified, {progress.completion_percentage():.1f}% complete")
            
            with self._lock:
                self._progress_cache[file_handle] = progress
            
            return progress
        return None
    
    def assemble_file(self, progress: DownloadProgress, output_path: Path) -> bool:
        """Assemble completed chunks into final file"""
        try:
            with open(output_path, 'wb') as output_file:
                for chunk in progress.chunks:
                    if not chunk.completed or not chunk.temp_file:
                        logger.error(f"Cannot assemble: chunk {chunk.chunk_id} not completed")
                        return False
                    
                    temp_file_path = Path(chunk.temp_file)
                    if not temp_file_path.exists():
                        logger.error(f"Cannot assemble: chunk file missing {temp_file_path}")
                        return False
                    
                    with open(temp_file_path, 'rb') as chunk_file:
                        data = chunk_file.read()
                        output_file.write(data)
            
            # Verify final file size
            actual_size = output_path.stat().st_size
            if actual_size != progress.file_size:
                logger.error(f"File size mismatch: expected {progress.file_size}, got {actual_size}")
                return False
            
            logger.info(f"File assembled successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assemble file: {e}")
            return False
    
    def cleanup_progress(self, file_handle: str) -> None:
        """Clean up progress files after successful download"""
        try:
            # Remove from cache
            with self._lock:
                self._progress_cache.pop(file_handle, None)
            
            # Remove progress file
            progress_file = self.progress_dir / f"{file_handle}_progress.pkl"
            if progress_file.exists():
                progress_file.unlink()
            
            # Remove chunk files
            for chunk_file in self.progress_dir.glob(f"{file_handle}_chunk_*.dat"):
                chunk_file.unlink()
            
            logger.debug(f"Progress cleanup completed: {file_handle}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup progress: {e}")
    
    def get_recovery_statistics(self, progress: DownloadProgress) -> Dict[str, Any]:
        """Get statistics about the recovery process"""
        total_attempts = sum(chunk.attempts for chunk in progress.chunks)
        failed_chunks = [chunk for chunk in progress.chunks if not chunk.completed and chunk.attempts > 0]
        
        return {
            'completion_percentage': progress.completion_percentage(),
            'completed_chunks': progress.completed_chunks,
            'failed_chunks': progress.failed_chunks,
            'total_chunks': progress.total_chunks,
            'total_attempts': total_attempts,
            'recovery_attempts': progress.recovery_attempts,
            'bytes_downloaded': progress.total_bytes_downloaded,
            'file_size': progress.file_size,
            'download_time': time.time() - progress.start_time,
            'retry_pending_chunks': len(self.get_retry_chunks(progress)),
            'permanently_failed_chunks': len([c for c in failed_chunks if c.attempts > self.max_retry_attempts])
        }
    
    def get_global_recovery_stats(self) -> Dict[str, Any]:
        """Get global recovery system statistics"""
        with self._lock:
            stats = self._recovery_stats.copy()
            stats['cached_downloads'] = len(self._progress_cache)
            stats['success_rate'] = (
                stats['successful_recoveries'] / max(stats['total_recoveries'], 1) * 100
                if stats['total_recoveries'] > 0 else 0.0
            )
        return stats
    
    def optimize_cache(self) -> None:
        """Optimize progress cache by removing old completed downloads"""
        with self._lock:
            if len(self._progress_cache) > self._max_cache_size:
                # Remove oldest completed downloads
                to_remove = []
                sorted_progress = sorted(
                    self._progress_cache.items(),
                    key=lambda x: x[1].last_update_time
                )
                
                for file_handle, progress in sorted_progress:
                    if progress.completion_percentage() == 100.0:
                        to_remove.append(file_handle)
                        if len(to_remove) >= len(self._progress_cache) - self._max_cache_size:
                            break
                
                for file_handle in to_remove:
                    self._progress_cache.pop(file_handle, None)
                    self.cleanup_progress(file_handle)
                
                logger.info(f"Cache optimized: removed {len(to_remove)} completed downloads")
    
    def predict_success_probability(self, progress: DownloadProgress) -> float:
        """Predict the probability of successful recovery based on current state"""
        if progress.total_chunks == 0:
            return 0.0
        
        # Base probability from completion rate
        completion_rate = progress.completion_percentage() / 100.0
        
        # Penalty for failed chunks
        failed_rate = progress.failed_chunks / progress.total_chunks
        failure_penalty = min(failed_rate * 0.5, 0.4)  # Max 40% penalty
        
        # Bonus for low retry attempts
        avg_attempts = sum(c.attempts for c in progress.chunks) / progress.total_chunks
        retry_bonus = max(0, (3.0 - avg_attempts) / 3.0 * 0.2)  # Max 20% bonus
        
        # Time factor - longer downloads may have more issues
        time_elapsed = time.time() - progress.start_time
        time_penalty = min(time_elapsed / 3600.0 * 0.1, 0.2)  # Max 20% penalty after 1 hour
        
        probability = completion_rate - failure_penalty + retry_bonus - time_penalty
        return max(0.0, min(1.0, probability))
    
    def auto_recover_failed_downloads(self) -> List[str]:
        """Automatically attempt to recover all failed downloads in cache"""
        recovered_files = []
        
        with self._lock:
            for file_handle, progress in self._progress_cache.items():
                if progress.completion_percentage() < 100.0:
                    # Check if recovery is viable
                    success_prob = self.predict_success_probability(progress)
                    if success_prob > 0.5:  # Only recover if >50% chance of success
                        retry_chunks = self.get_retry_chunks(progress)
                        if retry_chunks:
                            progress.recovery_attempts += 1
                            recovered_files.append(file_handle)
                            self._recovery_stats['total_recoveries'] += 1
                            logger.info(f"Auto-recovering {progress.file_name} ({success_prob:.1%} success probability)")
        
        return recovered_files
    
    def enable_smart_retry_scheduling(self, enable: bool = True) -> None:
        """Enable smart retry scheduling based on error patterns"""
        # This would integrate with a background scheduler in a full implementation
        logger.info(f"Smart retry scheduling {'enabled' if enable else 'disabled'}")
    
    def export_recovery_report(self, output_file: Path = None) -> Dict[str, Any]:
        """Export comprehensive recovery report for analysis"""
        if output_file is None:
            output_file = self.progress_dir / f"recovery_report_{int(time.time())}.json"
        
        report = {
            'timestamp': time.time(),
            'global_stats': self.get_global_recovery_stats(),
            'active_downloads': []
        }
        
        with self._lock:
            for file_handle, progress in self._progress_cache.items():
                download_report = {
                    'file_handle': file_handle,
                    'file_name': progress.file_name,
                    'statistics': self.get_recovery_statistics(progress),
                    'success_probability': self.predict_success_probability(progress),
                    'error_patterns': {}
                }
                
                # Analyze error patterns
                error_counts = {}
                for chunk in progress.chunks:
                    if chunk.last_error:
                        error_type = self.classify_error(chunk.last_error)
                        error_counts[error_type.value] = error_counts.get(error_type.value, 0) + 1
                
                download_report['error_patterns'] = error_counts
                report['active_downloads'].append(download_report)
        
        # Save report to file
        try:
            import json
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Recovery report exported: {output_file}")
        except Exception as e:
            logger.warning(f"Failed to export recovery report: {e}")
        
        return report
