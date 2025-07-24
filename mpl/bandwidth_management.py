#!/usr/bin/env python3
"""
Bandwidth Management System for MEGA Downloads

This module provides MEGAcmd-style bandwidth management and speed limiting
capabilities for optimized download performance and network resource control.

Features:
- Dynamic bandwidth limiting with configurable speed caps
- Adaptive throttling based on network conditions
- Per-transfer and global bandwidth management
- Queue-based transfer prioritization
- Real-time bandwidth monitoring and adjustment
- Integration with existing MEGA SDK optimizations

Based on MEGAcmd speedlimit functionality and MEGA SDK research.
Created as part of MEGA SDK optimization implementation.
"""

import time
import threading
import statistics
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import deque, defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class BandwidthUnit(Enum):
    """Bandwidth measurement units"""
    BYTES_PER_SECOND = "B/s"
    KILOBYTES_PER_SECOND = "KB/s"
    MEGABYTES_PER_SECOND = "MB/s"
    GIGABYTES_PER_SECOND = "GB/s"

class TransferPriority(Enum):
    """Transfer priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class BandwidthSettings:
    """Bandwidth management configuration"""
    max_download_speed: Optional[float] = None  # MB/s, None = unlimited
    max_upload_speed: Optional[float] = None    # MB/s, None = unlimited
    adaptive_throttling: bool = True             # Enable adaptive throttling
    priority_boost_factor: float = 1.5          # Speed boost for high priority transfers
    throttle_threshold: float = 0.8             # Throttle when usage exceeds this ratio
    burst_allowance: float = 2.0                # Allow bursts up to this multiple
    monitoring_interval: float = 1.0            # Bandwidth monitoring interval (seconds)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BandwidthSettings':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class TransferMetrics:
    """Real-time transfer performance metrics"""
    transfer_id: str
    bytes_downloaded: int = 0
    bytes_uploaded: int = 0
    start_time: float = 0.0
    last_update_time: float = 0.0
    current_download_speed: float = 0.0  # MB/s
    current_upload_speed: float = 0.0    # MB/s
    average_download_speed: float = 0.0  # MB/s
    average_upload_speed: float = 0.0    # MB/s
    priority: TransferPriority = TransferPriority.NORMAL
    throttled: bool = False
    
    def update_download_progress(self, bytes_delta: int) -> None:
        """Update download progress and calculate speeds"""
        current_time = time.time()
        self.bytes_downloaded += bytes_delta
        
        if self.start_time == 0:
            self.start_time = current_time
        
        time_delta = current_time - self.last_update_time if self.last_update_time > 0 else 1.0
        total_time = current_time - self.start_time
        
        # Calculate current speed (smoothed over recent period)
        if time_delta > 0:
            self.current_download_speed = (bytes_delta / time_delta) / 1024 / 1024  # MB/s
        
        # Calculate average speed
        if total_time > 0:
            self.average_download_speed = (self.bytes_downloaded / total_time) / 1024 / 1024  # MB/s
        
        self.last_update_time = current_time

class BandwidthMonitor:
    """Real-time bandwidth monitoring and measurement"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """Initialize bandwidth monitor"""
        self.monitoring_interval = monitoring_interval
        self.active_transfers: Dict[str, TransferMetrics] = {}
        self.bandwidth_history: deque = deque(maxlen=60)  # Keep 60 samples (1 minute)
        self.global_download_speed: float = 0.0  # MB/s
        self.global_upload_speed: float = 0.0    # MB/s
        self.peak_download_speed: float = 0.0
        self.peak_upload_speed: float = 0.0
        
        self._lock = threading.Lock()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info("Bandwidth monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start bandwidth monitoring thread"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Bandwidth monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop bandwidth monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Bandwidth monitoring stopped")
    
    def register_transfer(self, transfer_id: str, priority: TransferPriority = TransferPriority.NORMAL) -> None:
        """Register a new transfer for monitoring"""
        with self._lock:
            self.active_transfers[transfer_id] = TransferMetrics(
                transfer_id=transfer_id,
                priority=priority,
                start_time=time.time()
            )
        logger.debug(f"Registered transfer: {transfer_id} (priority: {priority.value})")
    
    def unregister_transfer(self, transfer_id: str) -> None:
        """Unregister a completed transfer"""
        with self._lock:
            self.active_transfers.pop(transfer_id, None)
        logger.debug(f"Unregistered transfer: {transfer_id}")
    
    def update_transfer_progress(self, transfer_id: str, bytes_downloaded: int) -> None:
        """Update transfer progress and calculate bandwidth usage"""
        with self._lock:
            if transfer_id in self.active_transfers:
                self.active_transfers[transfer_id].update_download_progress(bytes_downloaded)
    
    def get_transfer_metrics(self, transfer_id: str) -> Optional[TransferMetrics]:
        """Get metrics for a specific transfer"""
        with self._lock:
            return self.active_transfers.get(transfer_id)
    
    def get_global_bandwidth(self) -> Dict[str, float]:
        """Get current global bandwidth usage"""
        with self._lock:
            return {
                'download_speed_mbps': self.global_download_speed,
                'upload_speed_mbps': self.global_upload_speed,
                'peak_download_mbps': self.peak_download_speed,
                'peak_upload_mbps': self.peak_upload_speed,
                'active_transfers': len(self.active_transfers)
            }
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                self._calculate_global_bandwidth()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Bandwidth monitoring error: {e}")
    
    def _calculate_global_bandwidth(self) -> None:
        """Calculate global bandwidth usage from all active transfers"""
        with self._lock:
            total_download_speed = sum(
                transfer.current_download_speed 
                for transfer in self.active_transfers.values()
            )
            total_upload_speed = sum(
                transfer.current_upload_speed 
                for transfer in self.active_transfers.values()
            )
            
            self.global_download_speed = total_download_speed
            self.global_upload_speed = total_upload_speed
            
            # Update peaks
            if total_download_speed > self.peak_download_speed:
                self.peak_download_speed = total_download_speed
            if total_upload_speed > self.peak_upload_speed:
                self.peak_upload_speed = total_upload_speed
            
            # Store in history
            self.bandwidth_history.append({
                'timestamp': time.time(),
                'download_speed': total_download_speed,
                'upload_speed': total_upload_speed,
                'active_transfers': len(self.active_transfers)
            })

class BandwidthThrottler:
    """Intelligent bandwidth throttling and speed limiting"""
    
    def __init__(self, settings: BandwidthSettings, monitor: BandwidthMonitor):
        """Initialize bandwidth throttler"""
        self.settings = settings
        self.monitor = monitor
        self.transfer_allocations: Dict[str, float] = {}  # MB/s allocation per transfer
        
        self._lock = threading.Lock()
        self._last_allocation_time = time.time()
        
        logger.info(f"Bandwidth throttler initialized with settings: {settings}")
    
    def calculate_transfer_allocation(self, transfer_id: str) -> float:
        """Calculate bandwidth allocation for a specific transfer"""
        with self._lock:
            transfer_metrics = self.monitor.get_transfer_metrics(transfer_id)
            if not transfer_metrics:
                return float('inf')  # Unlimited if not found
            
            # Get current global usage
            global_bandwidth = self.monitor.get_global_bandwidth()
            current_usage = global_bandwidth['download_speed_mbps']
            
            # Calculate base allocation
            active_transfers = len(self.monitor.active_transfers)
            if active_transfers == 0:
                return float('inf')
            
            # Apply speed limits if configured
            if self.settings.max_download_speed is not None:
                # Calculate fair share among active transfers with proper priority weighting
                total_priority_weight = 0
                priority_weights = {
                    TransferPriority.LOW: 0.5,      # Lowest priority gets least bandwidth
                    TransferPriority.NORMAL: 1.0,   # Base allocation
                    TransferPriority.HIGH: 1.5,     # 50% more than normal
                    TransferPriority.URGENT: 2.0    # Double the normal allocation
                }
                
                # Calculate total weight for all active transfers
                for metrics in self.monitor.active_transfers.values():
                    total_priority_weight += priority_weights.get(metrics.priority, 1.0)
                
                # Calculate this transfer's share based on its priority
                transfer_weight = priority_weights.get(transfer_metrics.priority, 1.0)
                base_allocation = (self.settings.max_download_speed * transfer_weight) / total_priority_weight
                
                # Apply adaptive throttling
                if self.settings.adaptive_throttling:
                    base_allocation = self._apply_adaptive_throttling(
                        base_allocation, current_usage, transfer_metrics
                    )
                
                self.transfer_allocations[transfer_id] = base_allocation
                return base_allocation
            
            return float('inf')  # Unlimited
    
    def _apply_adaptive_throttling(self, base_allocation: float, current_usage: float, 
                                 transfer_metrics: TransferMetrics) -> float:
        """Apply adaptive throttling based on current network conditions"""
        
        # Check if we're exceeding the throttle threshold
        if (self.settings.max_download_speed and 
            current_usage > self.settings.max_download_speed * self.settings.throttle_threshold):
            
            # Reduce allocation for lower priority transfers
            if transfer_metrics.priority in [TransferPriority.LOW, TransferPriority.NORMAL]:
                reduction_factor = 0.8
                base_allocation *= reduction_factor
                transfer_metrics.throttled = True
                logger.debug(f"Throttling transfer {transfer_metrics.transfer_id}: {reduction_factor:.1%}")
        
        # Allow burst for high priority transfers if under burst allowance
        elif (transfer_metrics.priority in [TransferPriority.HIGH, TransferPriority.URGENT] and
              self.settings.max_download_speed and
              current_usage < self.settings.max_download_speed * self.settings.burst_allowance):
            
            burst_factor = 1.2
            base_allocation *= burst_factor
            logger.debug(f"Burst allowance for {transfer_metrics.transfer_id}: {burst_factor:.1%}")
        
        return base_allocation
    
    def should_throttle_chunk(self, transfer_id: str, chunk_size: int) -> Tuple[bool, float]:
        """
        Determine if a chunk should be throttled and calculate delay
        
        Returns:
            Tuple of (should_throttle, delay_seconds)
        """
        allocation = self.calculate_transfer_allocation(transfer_id)
        
        if allocation == float('inf'):
            return False, 0.0
        
        transfer_metrics = self.monitor.get_transfer_metrics(transfer_id)
        if not transfer_metrics:
            return False, 0.0
        
        # Calculate expected time for this chunk at allocated speed
        chunk_size_mb = chunk_size / 1024 / 1024
        expected_time = chunk_size_mb / allocation if allocation > 0 else 0
        
        # If current speed exceeds allocation, calculate throttle delay
        if transfer_metrics.current_download_speed > allocation:
            # Calculate how much to slow down
            excess_speed = transfer_metrics.current_download_speed - allocation
            excess_ratio = excess_speed / transfer_metrics.current_download_speed
            
            # Apply proportional delay
            throttle_delay = expected_time * excess_ratio
            return True, min(throttle_delay, 5.0)  # Cap at 5 seconds
        
        return False, 0.0
    
    def update_settings(self, new_settings: BandwidthSettings) -> None:
        """Update bandwidth settings"""
        with self._lock:
            self.settings = new_settings
            # Clear transfer allocations to force recalculation with new settings
            self.transfer_allocations.clear()
            logger.info(f"Bandwidth settings updated: {new_settings}")

class BandwidthManager:
    """Main bandwidth management system coordinating monitoring and throttling"""
    
    def __init__(self, settings: Optional[BandwidthSettings] = None, 
                 config_file: Optional[Path] = None):
        """Initialize bandwidth manager"""
        
        # Load settings - prefer passed settings over config file
        if settings is not None:
            self.settings = settings
        elif config_file and config_file.exists():
            self.settings = self._load_settings(config_file)
        else:
            self.settings = BandwidthSettings()
        
        # Initialize components
        self.monitor = BandwidthMonitor(self.settings.monitoring_interval)
        self.throttler = BandwidthThrottler(self.settings, self.monitor)
        
        # Management state
        self.active = False
        self.config_file = config_file
        
        logger.info("Bandwidth manager initialized")
    
    def start(self) -> None:
        """Start bandwidth management"""
        if not self.active:
            self.monitor.start_monitoring()
            self.active = True
            logger.info("🌐 Bandwidth management started")
    
    def stop(self) -> None:
        """Stop bandwidth management"""
        if self.active:
            self.monitor.stop_monitoring()
            self.active = False
            logger.info("Bandwidth management stopped")
    
    def register_download(self, transfer_id: str, priority: TransferPriority = TransferPriority.NORMAL) -> None:
        """Register a new download for bandwidth management"""
        self.monitor.register_transfer(transfer_id, priority)
        logger.info(f"📥 Registered download: {transfer_id} (priority: {priority.value})")
    
    def unregister_download(self, transfer_id: str) -> None:
        """Unregister a completed download"""
        self.monitor.unregister_transfer(transfer_id)
        logger.info(f"✅ Unregistered download: {transfer_id}")
    
    def update_download_progress(self, transfer_id: str, bytes_downloaded: int) -> None:
        """Update download progress for bandwidth calculation"""
        self.monitor.update_transfer_progress(transfer_id, bytes_downloaded)
    
    def get_bandwidth_allocation(self, transfer_id: str) -> float:
        """Get current bandwidth allocation for a transfer (MB/s)"""
        return self.throttler.calculate_transfer_allocation(transfer_id)
    
    def check_throttling(self, transfer_id: str, chunk_size: int) -> Tuple[bool, float]:
        """Check if a chunk should be throttled and get delay"""
        return self.throttler.should_throttle_chunk(transfer_id, chunk_size)
    
    def get_bandwidth_stats(self) -> Dict[str, Any]:
        """Get comprehensive bandwidth statistics"""
        global_bandwidth = self.monitor.get_global_bandwidth()
        
        # Calculate efficiency metrics
        efficiency = 0.0
        if self.settings.max_download_speed:
            efficiency = (global_bandwidth['download_speed_mbps'] / self.settings.max_download_speed) * 100
        
        return {
            'global_bandwidth': global_bandwidth,
            'settings': self.settings.to_dict(),
            'efficiency_percent': efficiency,
            'active_transfers': len(self.monitor.active_transfers),
            'throttling_active': any(
                metrics.throttled for metrics in self.monitor.active_transfers.values()
            )
        }
    
    def set_speed_limit(self, download_mbps: Optional[float] = None, 
                       upload_mbps: Optional[float] = None) -> None:
        """Set bandwidth speed limits (MEGAcmd-style speedlimit)"""
        
        # Update settings - allow None to mean unlimited
        self.settings.max_download_speed = download_mbps
        self.settings.max_upload_speed = upload_mbps
        
        # Update throttler
        self.throttler.update_settings(self.settings)
        
        # Clear transfer allocations to force recalculation
        self.throttler.transfer_allocations.clear()
        
        # Save settings if config file is specified
        if self.config_file:
            self._save_settings(self.config_file)
        
        if download_mbps is None and upload_mbps is None:
            logger.info("🚀 Speed limits removed - unlimited bandwidth")
        else:
            logger.info(f"🚦 Speed limits updated - Download: {download_mbps} MB/s, Upload: {upload_mbps} MB/s")
    
    def remove_speed_limit(self) -> None:
        """Remove all speed limits (unlimited bandwidth)"""
        self.settings.max_download_speed = None
        self.settings.max_upload_speed = None
        
        # Update throttler
        self.throttler.update_settings(self.settings)
        
        # Clear transfer allocations to force recalculation
        self.throttler.transfer_allocations.clear()
        
        # Save settings if config file is specified
        if self.config_file:
            self._save_settings(self.config_file)
            
        logger.info("🚀 Speed limits removed - unlimited bandwidth")
    
    def _load_settings(self, config_file: Path) -> BandwidthSettings:
        """Load bandwidth settings from file"""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            return BandwidthSettings.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load bandwidth settings: {e}")
            return BandwidthSettings()
    
    def _save_settings(self, config_file: Path) -> None:
        """Save current bandwidth settings to file"""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.settings.to_dict(), f, indent=2)
            logger.debug(f"Bandwidth settings saved to {config_file}")
        except Exception as e:
            logger.warning(f"Failed to save bandwidth settings: {e}")

# Global bandwidth manager instance
_global_bandwidth_manager: Optional[BandwidthManager] = None

def get_bandwidth_manager() -> BandwidthManager:
    """Get or create global bandwidth manager instance"""
    global _global_bandwidth_manager
    if _global_bandwidth_manager is None:
        config_file = Path("bandwidth_settings.json")
        _global_bandwidth_manager = BandwidthManager(config_file=config_file)
    return _global_bandwidth_manager

def initialize_bandwidth_management(settings: Optional[BandwidthSettings] = None) -> BandwidthManager:
    """Initialize bandwidth management system"""
    global _global_bandwidth_manager
    config_file = Path("bandwidth_settings.json")
    _global_bandwidth_manager = BandwidthManager(settings, config_file)
    return _global_bandwidth_manager
