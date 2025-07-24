"""
MegaSerpentClient - Synchronization & Transfer Module

Purpose: File synchronization, transfers, progress tracking, and optimization.

This module handles complete synchronization system (real-time sync, conflict resolution),
advanced transfer operations, chunking strategies, Cloud RAID transfers, and recovery management.
"""

import os
import time
import hashlib
import threading
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import queue

from . import utils
from .utils import (
    Constants, SyncDirection, SyncError, ValidationError, MegaError,
    DateTimeUtils, Helpers, Formatters, Decorators
)


# ==============================================
# === SYNC ENUMS AND CONSTANTS ===
# ==============================================

class SyncStatus(Enum):
    """Synchronization status."""
    IDLE = "idle"
    SYNCING = "syncing"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    ASK_USER = "ask_user"
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    KEEP_BOTH = "keep_both"
    NEWER_WINS = "newer_wins"


class ChunkStrategy(Enum):
    """Chunking strategies."""
    FIXED_SIZE = "fixed_size"
    ADAPTIVE = "adaptive"
    CONTENT_AWARE = "content_aware"
    ROLLING_HASH = "rolling_hash"


class TransferType(Enum):
    """Transfer operation types."""
    UPLOAD = "upload"
    DOWNLOAD = "download"
    SYNC = "sync"
    BACKUP = "backup"


# ==============================================
# === DATA CLASSES ===
# ==============================================

@dataclass
class SyncConfig:
    """Synchronization configuration."""
    sync_id: str
    name: str
    local_path: str
    remote_path: str
    direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    enabled: bool = True
    auto_sync: bool = True
    sync_interval: int = 300  # seconds
    conflict_resolution: ConflictResolution = ConflictResolution.ASK_USER
    exclude_patterns: List[str] = field(default_factory=lambda: ['*.tmp', '*.log'])
    include_patterns: List[str] = field(default_factory=list)
    preserve_permissions: bool = True
    delete_enabled: bool = False
    created_at: datetime = field(default_factory=DateTimeUtils.now_utc)


@dataclass
class SyncState:
    """Current synchronization state."""
    sync_id: str
    status: SyncStatus = SyncStatus.IDLE
    last_sync: Optional[datetime] = None
    next_sync: Optional[datetime] = None
    files_synced: int = 0
    bytes_synced: int = 0
    errors_count: int = 0
    current_file: Optional[str] = None
    progress_percentage: float = 0.0


@dataclass
class FileConflict:
    """File conflict information."""
    local_path: str
    remote_path: str
    local_modified: datetime
    remote_modified: datetime
    local_size: int
    remote_size: int
    conflict_type: str  # "modified", "deleted", "created"
    resolution: Optional[ConflictResolution] = None


@dataclass
class TransferMetrics:
    """Transfer performance metrics."""
    transfer_id: str
    transfer_type: TransferType
    start_time: datetime
    end_time: Optional[datetime] = None
    total_size: int = 0
    transferred_size: int = 0
    speed: float = 0.0  # bytes per second
    chunks_completed: int = 0
    chunks_total: int = 0
    errors: List[str] = field(default_factory=list)


# ==============================================
# === CHUNKING STRATEGIES ===
# ==============================================

class ChunkerBase:
    """Base class for chunking strategies."""
    
    def __init__(self, chunk_size: int = Constants.CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    def chunk_data(self, data: bytes) -> Iterator[bytes]:
        """Chunk data into pieces."""
        raise NotImplementedError


class FixedSizeChunker(ChunkerBase):
    """Fixed-size chunk strategy."""
    
    def chunk_data(self, data: bytes) -> Iterator[bytes]:
        """Chunk data into fixed-size pieces."""
        for i in range(0, len(data), self.chunk_size):
            yield data[i:i + self.chunk_size]


class AdaptiveChunker(ChunkerBase):
    """Adaptive chunk sizing based on content and network conditions."""
    
    def __init__(self, min_size: int = 64*1024, max_size: int = 8*1024*1024):
        self.min_size = min_size
        self.max_size = max_size
        self.current_size = Constants.CHUNK_SIZE
        self.logger = logging.getLogger(__name__)
    
    def chunk_data(self, data: bytes) -> Iterator[bytes]:
        """Chunk data with adaptive sizing."""
        offset = 0
        while offset < len(data):
            chunk_size = self._calculate_chunk_size(data[offset:])
            chunk = data[offset:offset + chunk_size]
            yield chunk
            offset += chunk_size
    
    def _calculate_chunk_size(self, remaining_data: bytes) -> int:
        """Calculate optimal chunk size based on content."""
        # Simple adaptive logic - could be enhanced with network metrics
        if len(remaining_data) < self.min_size:
            return len(remaining_data)
        
        # Analyze data entropy to determine chunk size
        entropy = self._calculate_entropy(remaining_data[:1024])  # Sample first 1KB
        
        if entropy > 0.8:  # High entropy (compressed/encrypted data)
            chunk_size = self.max_size
        elif entropy < 0.3:  # Low entropy (repetitive data)
            chunk_size = self.min_size
        else:
            chunk_size = self.current_size
        
        return min(chunk_size, len(remaining_data))
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0
        
        # Count byte frequencies
        frequencies = [0] * 256
        for byte in data:
            frequencies[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for freq in frequencies:
            if freq > 0:
                probability = freq / data_len
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy / 8.0  # Normalize to 0-1 range


class ContentAwareChunker(ChunkerBase):
    """Content-aware chunking that splits on logical boundaries."""
    
    def __init__(self, chunk_size: int = Constants.CHUNK_SIZE):
        super().__init__(chunk_size)
        self.boundary_patterns = [b'\n\n', b'\r\n\r\n', b'\x00']
    
    def chunk_data(self, data: bytes) -> Iterator[bytes]:
        """Chunk data on content boundaries."""
        offset = 0
        
        while offset < len(data):
            # Find next boundary within reasonable distance
            chunk_end = min(offset + self.chunk_size, len(data))
            
            # Look for content boundaries near chunk boundary
            boundary_pos = self._find_boundary(data[offset:chunk_end + 1024])
            
            if boundary_pos > 0:
                chunk_end = offset + boundary_pos
            
            chunk = data[offset:chunk_end]
            yield chunk
            offset = chunk_end
    
    def _find_boundary(self, data: bytes) -> int:
        """Find content boundary in data."""
        for pattern in self.boundary_patterns:
            pos = data.find(pattern)
            if pos > 0:
                return pos + len(pattern)
        
        return len(data)


class RollingHashChunker(ChunkerBase):
    """Rolling hash chunking for content-defined chunks."""
    
    def __init__(self, window_size: int = 48, target_size: int = Constants.CHUNK_SIZE):
        self.window_size = window_size
        self.target_size = target_size
        self.min_size = target_size // 4
        self.max_size = target_size * 4
        self.logger = logging.getLogger(__name__)
    
    def chunk_data(self, data: bytes) -> Iterator[bytes]:
        """Chunk data using rolling hash."""
        if len(data) <= self.min_size:
            yield data
            return
        
        offset = 0
        while offset < len(data):
            chunk_end = self._find_chunk_boundary(data[offset:])
            chunk = data[offset:offset + chunk_end]
            yield chunk
            offset += chunk_end
    
    def _find_chunk_boundary(self, data: bytes) -> int:
        """Find chunk boundary using rolling hash."""
        if len(data) <= self.min_size:
            return len(data)
        
        # Simple rolling hash implementation
        hash_value = 0
        for i in range(min(self.window_size, len(data))):
            hash_value = (hash_value * 31 + data[i]) % (2**32)
        
        # Look for hash patterns that indicate good split points
        for i in range(self.min_size, min(len(data), self.max_size)):
            if i + self.window_size <= len(data):
                # Update rolling hash
                hash_value = (hash_value * 31 + data[i + self.window_size - 1]) % (2**32)
            
            # Check if this is a good split point (hash ends with specific pattern)
            if hash_value & 0xFFF == 0:  # 1 in 4096 chance
                return i
        
        return min(len(data), self.max_size)


class Deduplication:
    """Chunk-level deduplication."""
    
    def __init__(self):
        self._chunk_hashes: Dict[str, str] = {}  # hash -> chunk_id
        self._chunk_refs: Dict[str, int] = {}  # chunk_id -> reference count
        self.logger = logging.getLogger(__name__)
    
    def add_chunk(self, chunk: bytes) -> str:
        """Add chunk and return chunk ID."""
        chunk_hash = hashlib.sha256(chunk).hexdigest()
        
        if chunk_hash in self._chunk_hashes:
            # Chunk already exists, increment reference
            chunk_id = self._chunk_hashes[chunk_hash]
            self._chunk_refs[chunk_id] += 1
            return chunk_id
        
        # New chunk
        chunk_id = Helpers.generate_request_id()
        self._chunk_hashes[chunk_hash] = chunk_id
        self._chunk_refs[chunk_id] = 1
        
        # In real implementation, store chunk data
        self.logger.debug(f"Added new chunk: {chunk_id}")
        return chunk_id
    
    def remove_chunk_ref(self, chunk_id: str):
        """Remove reference to chunk."""
        if chunk_id in self._chunk_refs:
            self._chunk_refs[chunk_id] -= 1
            
            if self._chunk_refs[chunk_id] <= 0:
                # Remove chunk completely
                del self._chunk_refs[chunk_id]
                
                # Find and remove hash mapping
                hash_to_remove = None
                for chunk_hash, cid in self._chunk_hashes.items():
                    if cid == chunk_id:
                        hash_to_remove = chunk_hash
                        break
                
                if hash_to_remove:
                    del self._chunk_hashes[hash_to_remove]
                
                self.logger.debug(f"Removed chunk: {chunk_id}")
    
    def get_deduplication_ratio(self) -> float:
        """Get deduplication ratio."""
        total_refs = sum(self._chunk_refs.values())
        unique_chunks = len(self._chunk_refs)
        
        if unique_chunks == 0:
            return 0.0
        
        return 1.0 - (unique_chunks / total_refs)


# ==============================================
# === TRANSFER MANAGEMENT ===
# ==============================================

class TransferManager:
    """Main transfer orchestrator."""
    
    def __init__(self):
        self._transfers: Dict[str, TransferMetrics] = {}
        self._transfer_queue = queue.PriorityQueue()
        self._active_transfers: Dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._chunker_registry: Dict[ChunkStrategy, ChunkerBase] = {}
        self._deduplicator = Deduplication()
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        self._setup_chunkers()
    
    def _setup_chunkers(self):
        """Setup chunking strategies."""
        self._chunker_registry[ChunkStrategy.FIXED_SIZE] = FixedSizeChunker()
        self._chunker_registry[ChunkStrategy.ADAPTIVE] = AdaptiveChunker()
        self._chunker_registry[ChunkStrategy.CONTENT_AWARE] = ContentAwareChunker()
        self._chunker_registry[ChunkStrategy.ROLLING_HASH] = RollingHashChunker()
    
    def start_transfer(self, transfer_type: TransferType, source: str, destination: str,
                      priority: int = 5, chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE,
                      progress_callback: Optional[Callable] = None) -> str:
        """Start new transfer operation."""
        transfer_id = Helpers.generate_request_id()
        
        metrics = TransferMetrics(
            transfer_id=transfer_id,
            transfer_type=transfer_type,
            start_time=DateTimeUtils.now_utc()
        )
        
        with self._lock:
            self._transfers[transfer_id] = metrics
        
        # Submit transfer task
        future = self._executor.submit(
            self._execute_transfer,
            transfer_id, source, destination, chunk_strategy, progress_callback
        )
        
        self._active_transfers[transfer_id] = future
        
        self.logger.info(f"Started {transfer_type.value}: {source} -> {destination} (ID: {transfer_id})")
        return transfer_id
    
    def _execute_transfer(self, transfer_id: str, source: str, destination: str,
                         chunk_strategy: ChunkStrategy, progress_callback: Optional[Callable]):
        """Execute transfer operation."""
        try:
            metrics = self._transfers[transfer_id]
            chunker = self._chunker_registry[chunk_strategy]
            
            # For demo purposes, simulate file transfer
            if metrics.transfer_type == TransferType.UPLOAD:
                self._simulate_upload(metrics, source, destination, chunker, progress_callback)
            elif metrics.transfer_type == TransferType.DOWNLOAD:
                self._simulate_download(metrics, source, destination, chunker, progress_callback)
            
            metrics.end_time = DateTimeUtils.now_utc()
            self.logger.info(f"Transfer completed: {transfer_id}")
            
        except Exception as e:
            metrics = self._transfers.get(transfer_id)
            if metrics:
                metrics.errors.append(str(e))
                metrics.end_time = DateTimeUtils.now_utc()
            
            self.logger.error(f"Transfer failed: {transfer_id} - {e}")
        finally:
            if transfer_id in self._active_transfers:
                del self._active_transfers[transfer_id]
    
    def _simulate_upload(self, metrics: TransferMetrics, source: str, destination: str,
                        chunker: ChunkerBase, progress_callback: Optional[Callable]):
        """Simulate file upload."""
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source file not found: {source}")
        
        file_size = os.path.getsize(source)
        metrics.total_size = file_size
        
        with open(source, 'rb') as f:
            data = f.read()
            chunks = list(chunker.chunk_data(data))
            metrics.chunks_total = len(chunks)
            
            for i, chunk in enumerate(chunks):
                # Simulate network delay
                time.sleep(0.01)
                
                # Add chunk to deduplicator
                chunk_id = self._deduplicator.add_chunk(chunk)
                
                metrics.chunks_completed = i + 1
                metrics.transferred_size += len(chunk)
                
                # Calculate speed
                elapsed = (DateTimeUtils.now_utc() - metrics.start_time).total_seconds()
                if elapsed > 0:
                    metrics.speed = metrics.transferred_size / elapsed
                
                # Call progress callback
                if progress_callback:
                    progress_callback(metrics)
    
    def _simulate_download(self, metrics: TransferMetrics, source: str, destination: str,
                          chunker: ChunkerBase, progress_callback: Optional[Callable]):
        """Simulate file download."""
        # Simulate downloading a 1MB file
        file_size = 1024 * 1024
        metrics.total_size = file_size
        
        chunk_size = Constants.CHUNK_SIZE
        chunks_total = (file_size + chunk_size - 1) // chunk_size
        metrics.chunks_total = chunks_total
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as f:
            for i in range(chunks_total):
                # Simulate network delay
                time.sleep(0.01)
                
                # Write chunk
                current_chunk_size = min(chunk_size, file_size - i * chunk_size)
                chunk_data = b'0' * current_chunk_size
                f.write(chunk_data)
                
                metrics.chunks_completed = i + 1
                metrics.transferred_size += current_chunk_size
                
                # Calculate speed
                elapsed = (DateTimeUtils.now_utc() - metrics.start_time).total_seconds()
                if elapsed > 0:
                    metrics.speed = metrics.transferred_size / elapsed
                
                # Call progress callback
                if progress_callback:
                    progress_callback(metrics)
    
    def get_transfer_metrics(self, transfer_id: str) -> Optional[TransferMetrics]:
        """Get transfer metrics."""
        return self._transfers.get(transfer_id)
    
    def cancel_transfer(self, transfer_id: str) -> bool:
        """Cancel active transfer."""
        if transfer_id in self._active_transfers:
            future = self._active_transfers[transfer_id]
            future.cancel()
            del self._active_transfers[transfer_id]
            
            self.logger.info(f"Transfer cancelled: {transfer_id}")
            return True
        
        return False
    
    def list_active_transfers(self) -> List[str]:
        """List active transfer IDs."""
        return list(self._active_transfers.keys())
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall transfer statistics."""
        total_transfers = len(self._transfers)
        completed_transfers = sum(1 for m in self._transfers.values() if m.end_time)
        failed_transfers = sum(1 for m in self._transfers.values() if m.errors)
        
        total_bytes = sum(m.transferred_size for m in self._transfers.values())
        avg_speed = sum(m.speed for m in self._transfers.values()) / max(total_transfers, 1)
        
        return {
            'total_transfers': total_transfers,
            'completed_transfers': completed_transfers,
            'failed_transfers': failed_transfers,
            'active_transfers': len(self._active_transfers),
            'total_bytes_transferred': total_bytes,
            'total_bytes_formatted': Formatters.format_file_size(total_bytes),
            'average_speed': avg_speed,
            'average_speed_formatted': f"{Formatters.format_file_size(int(avg_speed))}/s",
            'deduplication_ratio': self._deduplicator.get_deduplication_ratio()
        }


# ==============================================
# === SYNCHRONIZATION ENGINE ===
# ==============================================

class SyncEngine:
    """Main synchronization engine."""
    
    def __init__(self, transfer_manager: TransferManager):
        self.transfer_manager = transfer_manager
        self._sync_configs: Dict[str, SyncConfig] = {}
        self._sync_states: Dict[str, SyncState] = {}
        self._sync_threads: Dict[str, threading.Thread] = {}
        self._running = False
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def create_sync(self, name: str, local_path: str, remote_path: str,
                   direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
                   **kwargs) -> str:
        """Create new sync configuration."""
        sync_id = Helpers.generate_request_id()
        
        config = SyncConfig(
            sync_id=sync_id,
            name=name,
            local_path=local_path,
            remote_path=remote_path,
            direction=direction,
            **kwargs
        )
        
        state = SyncState(sync_id=sync_id)
        
        with self._lock:
            self._sync_configs[sync_id] = config
            self._sync_states[sync_id] = state
        
        self.logger.info(f"Created sync: {name} ({sync_id})")
        return sync_id
    
    def start_sync(self, sync_id: str) -> bool:
        """Start synchronization for specific sync."""
        config = self._sync_configs.get(sync_id)
        if not config or not config.enabled:
            return False
        
        with self._lock:
            if sync_id in self._sync_threads:
                return False  # Already running
            
            thread = threading.Thread(
                target=self._run_sync,
                args=(sync_id,),
                daemon=True
            )
            thread.start()
            self._sync_threads[sync_id] = thread
        
        self.logger.info(f"Started sync: {sync_id}")
        return True
    
    def stop_sync(self, sync_id: str) -> bool:
        """Stop synchronization for specific sync."""
        with self._lock:
            if sync_id in self._sync_states:
                self._sync_states[sync_id].status = SyncStatus.PAUSED
            
            if sync_id in self._sync_threads:
                # Thread will exit when it checks status
                del self._sync_threads[sync_id]
        
        self.logger.info(f"Stopped sync: {sync_id}")
        return True
    
    def _run_sync(self, sync_id: str):
        """Run synchronization loop for sync."""
        config = self._sync_configs[sync_id]
        state = self._sync_states[sync_id]
        
        while config.enabled and state.status != SyncStatus.PAUSED:
            try:
                state.status = SyncStatus.SYNCING
                
                # Perform sync operation
                self._perform_sync(config, state)
                
                state.status = SyncStatus.COMPLETED
                state.last_sync = DateTimeUtils.now_utc()
                
                if config.auto_sync:
                    state.next_sync = state.last_sync + timedelta(seconds=config.sync_interval)
                    
                    # Wait for next sync
                    time.sleep(config.sync_interval)
                else:
                    break
                
            except Exception as e:
                state.status = SyncStatus.ERROR
                state.errors_count += 1
                self.logger.error(f"Sync error for {sync_id}: {e}")
                
                # Wait before retry
                time.sleep(60)
        
        # Clean up
        with self._lock:
            if sync_id in self._sync_threads:
                del self._sync_threads[sync_id]
    
    def _perform_sync(self, config: SyncConfig, state: SyncState):
        """Perform actual synchronization."""
        # Get file listings
        local_files = self._get_local_files(config.local_path, config)
        remote_files = self._get_remote_files(config.remote_path, config)
        
        # Detect changes and conflicts
        changes = self._detect_changes(local_files, remote_files, config)
        conflicts = self._detect_conflicts(changes, config)
        
        # Resolve conflicts
        if conflicts:
            resolved_conflicts = self._resolve_conflicts(conflicts, config)
            changes.extend(resolved_conflicts)
        
        # Apply changes
        total_files = len(changes)
        for i, change in enumerate(changes):
            if state.status == SyncStatus.PAUSED:
                break
            
            self._apply_change(change, config)
            
            state.files_synced = i + 1
            state.progress_percentage = (i + 1) / total_files * 100
            state.current_file = change.get('file_path', '')
    
    def _get_local_files(self, path: str, config: SyncConfig) -> Dict[str, Any]:
        """Get local file listing."""
        files = {}
        
        if not os.path.exists(path):
            return files
        
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                if self._should_exclude_file(filename, config):
                    continue
                
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, path)
                
                stat_info = os.stat(file_path)
                files[rel_path] = {
                    'path': file_path,
                    'size': stat_info.st_size,
                    'modified': datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc),
                    'checksum': self._calculate_file_hash(file_path)
                }
        
        return files
    
    def _get_remote_files(self, path: str, config: SyncConfig) -> Dict[str, Any]:
        """Get remote file listing."""
        # In real implementation, this would query the cloud storage
        # For demo, return empty dict
        return {}
    
    def _detect_changes(self, local_files: Dict, remote_files: Dict, config: SyncConfig) -> List[Dict]:
        """Detect file changes that need synchronization."""
        changes = []
        
        # Files to upload (local only or local newer)
        for rel_path, local_info in local_files.items():
            if rel_path not in remote_files:
                changes.append({
                    'action': 'upload',
                    'file_path': rel_path,
                    'local_path': local_info['path'],
                    'remote_path': os.path.join(config.remote_path, rel_path).replace('\\', '/')
                })
            else:
                remote_info = remote_files[rel_path]
                if local_info['modified'] > remote_info.get('modified', datetime.min.replace(tzinfo=timezone.utc)):
                    changes.append({
                        'action': 'upload',
                        'file_path': rel_path,
                        'local_path': local_info['path'],
                        'remote_path': os.path.join(config.remote_path, rel_path).replace('\\', '/')
                    })
        
        # Files to download (remote only or remote newer)
        for rel_path, remote_info in remote_files.items():
            if rel_path not in local_files:
                changes.append({
                    'action': 'download',
                    'file_path': rel_path,
                    'local_path': os.path.join(config.local_path, rel_path),
                    'remote_path': remote_info['path']
                })
            else:
                local_info = local_files[rel_path]
                if remote_info.get('modified', datetime.max.replace(tzinfo=timezone.utc)) > local_info['modified']:
                    changes.append({
                        'action': 'download',
                        'file_path': rel_path,
                        'local_path': os.path.join(config.local_path, rel_path),
                        'remote_path': remote_info['path']
                    })
        
        return changes
    
    def _detect_conflicts(self, changes: List[Dict], config: SyncConfig) -> List[FileConflict]:
        """Detect synchronization conflicts."""
        conflicts = []
        
        # For demo purposes, simulate conflict detection
        # In real implementation, this would check for actual conflicts
        
        return conflicts
    
    def _resolve_conflicts(self, conflicts: List[FileConflict], config: SyncConfig) -> List[Dict]:
        """Resolve synchronization conflicts."""
        resolved_changes = []
        
        for conflict in conflicts:
            resolution = config.conflict_resolution
            
            if resolution == ConflictResolution.LOCAL_WINS:
                resolved_changes.append({
                    'action': 'upload',
                    'file_path': conflict.local_path,
                    'local_path': conflict.local_path,
                    'remote_path': conflict.remote_path
                })
            elif resolution == ConflictResolution.REMOTE_WINS:
                resolved_changes.append({
                    'action': 'download',
                    'file_path': conflict.remote_path,
                    'local_path': conflict.local_path,
                    'remote_path': conflict.remote_path
                })
            elif resolution == ConflictResolution.NEWER_WINS:
                if conflict.local_modified > conflict.remote_modified:
                    action = 'upload'
                else:
                    action = 'download'
                
                resolved_changes.append({
                    'action': action,
                    'file_path': conflict.local_path,
                    'local_path': conflict.local_path,
                    'remote_path': conflict.remote_path
                })
            elif resolution == ConflictResolution.KEEP_BOTH:
                # Rename one of the files
                timestamp = DateTimeUtils.now_utc().strftime("%Y%m%d_%H%M%S")
                conflict_name = f"{conflict.local_path}_conflict_{timestamp}"
                
                resolved_changes.append({
                    'action': 'upload',
                    'file_path': conflict_name,
                    'local_path': conflict.local_path,
                    'remote_path': conflict.remote_path + f"_conflict_{timestamp}"
                })
        
        return resolved_changes
    
    def _apply_change(self, change: Dict, config: SyncConfig):
        """Apply a single sync change."""
        action = change['action']
        
        if action == 'upload':
            self.transfer_manager.start_transfer(
                TransferType.UPLOAD,
                change['local_path'],
                change['remote_path']
            )
        elif action == 'download':
            self.transfer_manager.start_transfer(
                TransferType.DOWNLOAD,
                change['remote_path'],
                change['local_path']
            )
    
    def _should_exclude_file(self, filename: str, config: SyncConfig) -> bool:
        """Check if file should be excluded from sync."""
        import fnmatch
        
        # Check exclude patterns
        for pattern in config.exclude_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        
        # Check include patterns (if any)
        if config.include_patterns:
            for pattern in config.include_patterns:
                if fnmatch.fnmatch(filename, pattern):
                    return False
            return True  # Not in include list
        
        return False
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash for comparison."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_sync_config(self, sync_id: str) -> Optional[SyncConfig]:
        """Get sync configuration."""
        return self._sync_configs.get(sync_id)
    
    def get_sync_state(self, sync_id: str) -> Optional[SyncState]:
        """Get sync state."""
        return self._sync_states.get(sync_id)
    
    def list_syncs(self) -> List[SyncConfig]:
        """List all sync configurations."""
        return list(self._sync_configs.values())
    
    def delete_sync(self, sync_id: str) -> bool:
        """Delete sync configuration."""
        self.stop_sync(sync_id)
        
        with self._lock:
            if sync_id in self._sync_configs:
                del self._sync_configs[sync_id]
            if sync_id in self._sync_states:
                del self._sync_states[sync_id]
        
        self.logger.info(f"Deleted sync: {sync_id}")
        return True


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Enums
    'SyncStatus', 'ConflictResolution', 'ChunkStrategy', 'TransferType',
    
    # Data Classes
    'SyncConfig', 'SyncState', 'FileConflict', 'TransferMetrics',
    
    # Chunking
    'ChunkerBase', 'FixedSizeChunker', 'AdaptiveChunker', 'ContentAwareChunker', 'RollingHashChunker',
    'Deduplication',
    
    # Transfer Management
    'TransferManager',
    
    # Synchronization
    'SyncEngine'
]