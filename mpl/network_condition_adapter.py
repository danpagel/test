"""
Network Condition Adapter Module
================================

Dynamic network condition monitoring and adaptive download optimization.
Based on MEGA SDK research and real-time network performance analysis.

This module implements intelligent network adaptation for optimal download performance:
- Real-time bandwidth detection
- Latency monitoring and adaptation
- Dynamic chunk sizing based on network conditions
- Adaptive worker count adjustment
- Network quality assessment

Author: Enhanced based on MEGA SDK research
Date: July 2025
"""

import time
import threading
import statistics
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class NetworkCondition:
    """Network condition metrics"""
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    packet_loss: float = 0.0
    stability_score: float = 0.0  # 0-1, higher is better
    quality_rating: str = "unknown"  # poor, fair, good, excellent
    
    def to_dict(self) -> Dict:
        return {
            'bandwidth_mbps': self.bandwidth_mbps,
            'latency_ms': self.latency_ms,
            'packet_loss': self.packet_loss,
            'stability_score': self.stability_score,
            'quality_rating': self.quality_rating
        }

@dataclass
class AdaptiveSettings:
    """Adaptive download settings based on network conditions"""
    chunk_size_mb: float = 8.0
    worker_count: int = 4
    timeout_seconds: int = 90
    retry_count: int = 3
    use_streaming: bool = False
    connection_pool_size: int = 20
    
    def to_dict(self) -> Dict:
        return {
            'chunk_size_mb': self.chunk_size_mb,
            'worker_count': self.worker_count,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'use_streaming': self.use_streaming,
            'connection_pool_size': self.connection_pool_size
        }

class NetworkConditionMonitor:
    """
    Monitor and analyze network conditions for adaptive optimization.
    """
    
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.bandwidth_history = deque(maxlen=history_size)
        self.latency_history = deque(maxlen=history_size)
        self.download_history = deque(maxlen=history_size)
        
        # Network condition state
        self.current_condition = NetworkCondition()
        self.is_monitoring = False
        self.monitor_lock = threading.Lock()
        
        # Calibration state
        self.is_calibrated = False
        self.baseline_bandwidth = 0.0
        self.baseline_latency = 0.0
        
    def start_monitoring(self):
        """Start network condition monitoring"""
        with self.monitor_lock:
            self.is_monitoring = True
            logger.info("Network condition monitoring started")
    
    def stop_monitoring(self):
        """Stop network condition monitoring"""
        with self.monitor_lock:
            self.is_monitoring = False
            logger.info("Network condition monitoring stopped")
    
    def record_download_sample(self, bytes_downloaded: int, download_time: float, chunk_size: int):
        """Record a download sample for network analysis"""
        if download_time <= 0:
            return
        
        bandwidth_mbps = (bytes_downloaded / download_time) / (1024 * 1024)  # MB/s to Mbps
        
        with self.monitor_lock:
            self.bandwidth_history.append(bandwidth_mbps)
            self.download_history.append({
                'bytes': bytes_downloaded,
                'time': download_time,
                'chunk_size': chunk_size,
                'bandwidth_mbps': bandwidth_mbps,
                'timestamp': time.time()
            })
            
            # Auto-calibrate after enough samples
            if len(self.download_history) >= 3 and not self.is_calibrated:
                self._auto_calibrate()
            
            # Update current condition
            self._update_network_condition()
    
    def _auto_calibrate(self):
        """Auto-calibrate baseline values from collected samples"""
        if len(self.bandwidth_history) >= 3:
            self.baseline_bandwidth = statistics.mean(list(self.bandwidth_history)[-3:])
            self.baseline_latency = 100.0  # Default latency estimate
            self.is_calibrated = True
            logger.info(f"Network auto-calibrated: {self.baseline_bandwidth:.2f} MB/s baseline bandwidth")
    
    def record_latency_sample(self, latency_ms: float):
        """Record a latency sample"""
        with self.monitor_lock:
            self.latency_history.append(latency_ms)
            self._update_network_condition()
    
    def _update_network_condition(self):
        """Update current network condition based on historical data"""
        if not self.bandwidth_history:
            return
        
        # Calculate bandwidth statistics with weighted recent samples
        bandwidths = list(self.bandwidth_history)
        if len(bandwidths) >= 3:
            # Weighted average favoring recent performance
            recent_samples = min(5, len(bandwidths))
            recent_bandwidth = bandwidths[-recent_samples:]
            weights = [1.0, 1.2, 1.5, 1.8, 2.0][:recent_samples]  # More weight on recent samples
            weighted_sum = sum(bw * w for bw, w in zip(recent_bandwidth, weights))
            avg_bandwidth = weighted_sum / sum(weights)
        else:
            avg_bandwidth = statistics.mean(bandwidths)
        
        # Calculate latency statistics
        latencies = list(self.latency_history) if self.latency_history else [100.0]  # Default 100ms
        avg_latency = statistics.mean(latencies)
        
        # Calculate stability score with improved sensitivity
        if len(bandwidths) > 1:
            recent_bw = bandwidths[-3:] if len(bandwidths) >= 3 else bandwidths
            bandwidth_variance = statistics.variance(recent_bw) if len(recent_bw) > 1 else 0.0
            # More sensitive stability calculation
            stability_score = max(0.0, 1.0 - (bandwidth_variance / max(0.1, avg_bandwidth)))
        else:
            stability_score = 0.5
        
        # Determine quality rating
        quality_rating = self._assess_quality(avg_bandwidth, avg_latency, stability_score)
        
        # Update current condition
        self.current_condition = NetworkCondition(
            bandwidth_mbps=avg_bandwidth,
            latency_ms=avg_latency,
            packet_loss=0.0,  # TODO: Implement packet loss detection
            stability_score=stability_score,
            quality_rating=quality_rating
        )
        
        logger.debug(f"Network condition updated: {self.current_condition}")
    
    def _assess_quality(self, bandwidth_mbps: float, latency_ms: float, stability: float) -> str:
        """Assess network quality rating with improved sensitivity"""
        
        # Enhanced scoring with more responsive thresholds
        bandwidth_score = min(1.0, bandwidth_mbps / 8.0)  # 8 MB/s = excellent (more achievable)
        latency_score = max(0.0, 1.0 - (latency_ms / 300.0))  # 300ms = poor (more responsive)
        
        # Weighted scoring (bandwidth most important for downloads)
        overall_score = (bandwidth_score * 0.6 + latency_score * 0.2 + stability * 0.2)
        
        # More responsive thresholds
        if overall_score >= 0.75:
            return "excellent"
        elif overall_score >= 0.55:
            return "good"
        elif overall_score >= 0.35:
            return "fair"
        else:
            return "poor"
    
    def get_current_condition(self) -> NetworkCondition:
        """Get current network condition"""
        with self.monitor_lock:
            return self.current_condition
    
    def get_performance_history(self) -> List[Dict]:
        """Get download performance history"""
        with self.monitor_lock:
            return list(self.download_history)

class AdaptiveDownloadOptimizer:
    """
    Adaptive download optimizer that adjusts settings based on network conditions.
    """
    
    def __init__(self, monitor: NetworkConditionMonitor):
        self.monitor = monitor
        self.optimization_history = []
        
        # Optimization profiles for different network conditions
        self.profiles = {
            "excellent": AdaptiveSettings(
                chunk_size_mb=16.0,
                worker_count=6,
                timeout_seconds=120,
                retry_count=2,
                use_streaming=True,
                connection_pool_size=30
            ),
            "good": AdaptiveSettings(
                chunk_size_mb=8.0,
                worker_count=4,
                timeout_seconds=90,
                retry_count=3,
                use_streaming=False,
                connection_pool_size=20
            ),
            "fair": AdaptiveSettings(
                chunk_size_mb=4.0,
                worker_count=2,
                timeout_seconds=60,
                retry_count=4,
                use_streaming=False,
                connection_pool_size=15
            ),
            "poor": AdaptiveSettings(
                chunk_size_mb=2.0,
                worker_count=1,
                timeout_seconds=45,
                retry_count=5,
                use_streaming=False,
                connection_pool_size=10
            )
        }
    
    def get_optimal_settings(self, file_size_mb: float) -> AdaptiveSettings:
        """
        Get optimal download settings based on current network conditions and file size.
        """
        condition = self.monitor.get_current_condition()
        
        # Start with profile for current network quality
        base_settings = self.profiles.get(condition.quality_rating, self.profiles["good"])
        
        # Create adaptive settings
        settings = AdaptiveSettings(
            chunk_size_mb=base_settings.chunk_size_mb,
            worker_count=base_settings.worker_count,
            timeout_seconds=base_settings.timeout_seconds,
            retry_count=base_settings.retry_count,
            use_streaming=base_settings.use_streaming,
            connection_pool_size=base_settings.connection_pool_size
        )
        
        # File size adaptations with more aggressive optimization
        if file_size_mb < 2:
            # Very small files - use sequential
            settings.worker_count = 1
            settings.chunk_size_mb = file_size_mb
        elif file_size_mb < 8:
            # Small files - conservative
            settings.worker_count = min(2, settings.worker_count)
            settings.chunk_size_mb = min(4.0, settings.chunk_size_mb)
        elif file_size_mb > 50:
            # Large files - more aggressive parallelization
            if condition.quality_rating in ["good", "excellent"]:
                settings.worker_count = min(6, settings.worker_count + 1)  # Up to 6 workers for large files
            settings.chunk_size_mb = min(24.0, max(8.0, settings.chunk_size_mb))  # Larger chunks
        elif file_size_mb > 100:
            # Very large files - enable streaming and max optimization
            settings.use_streaming = True
            settings.chunk_size_mb = min(32.0, max(16.0, settings.chunk_size_mb))
            if condition.quality_rating == "excellent":
                settings.worker_count = min(8, settings.worker_count + 2)  # Up to 8 workers for excellent connections
        
        # Network condition fine-tuning
        if condition.latency_ms > 200:
            # High latency - larger chunks, fewer workers
            settings.chunk_size_mb *= 1.5
            settings.worker_count = max(1, settings.worker_count - 1)
            settings.timeout_seconds += 30
        
        if condition.stability_score < 0.5:
            # Unstable connection - conservative approach
            settings.worker_count = max(1, settings.worker_count - 1)
            settings.retry_count += 2
            settings.timeout_seconds += 15
        
        # Bandwidth-based adaptations
        if condition.bandwidth_mbps < 1.0:
            # Low bandwidth - smaller chunks, sequential
            settings.chunk_size_mb = min(2.0, settings.chunk_size_mb)
            settings.worker_count = 1
        elif condition.bandwidth_mbps > 20.0:
            # High bandwidth - can handle larger chunks
            settings.chunk_size_mb = min(32.0, settings.chunk_size_mb * 1.5)
        
        # Ensure reasonable limits
        settings.chunk_size_mb = max(1.0, min(32.0, settings.chunk_size_mb))
        settings.worker_count = max(1, min(8, settings.worker_count))
        settings.timeout_seconds = max(30, min(300, settings.timeout_seconds))
        
        # Record optimization decision
        self.optimization_history.append({
            'timestamp': time.time(),
            'file_size_mb': file_size_mb,
            'network_condition': condition.to_dict(),
            'settings': settings.to_dict()
        })
        
        logger.info(f"Adaptive settings for {file_size_mb:.1f}MB file: "
                   f"{settings.worker_count} workers, {settings.chunk_size_mb:.1f}MB chunks, "
                   f"network={condition.quality_rating}")
        
        return settings
    
    def get_optimization_history(self) -> List[Dict]:
        """Get history of optimization decisions"""
        return self.optimization_history.copy()

# Global instances
_network_monitor = NetworkConditionMonitor()
_adaptive_optimizer = AdaptiveDownloadOptimizer(_network_monitor)

def get_network_monitor() -> NetworkConditionMonitor:
    """Get the global network monitor instance"""
    return _network_monitor

def get_adaptive_optimizer() -> AdaptiveDownloadOptimizer:
    """Get the global adaptive optimizer instance"""
    return _adaptive_optimizer

def start_network_monitoring():
    """Start global network monitoring"""
    _network_monitor.start_monitoring()

def stop_network_monitoring():
    """Stop global network monitoring"""
    _network_monitor.stop_monitoring()

def get_current_network_condition() -> NetworkCondition:
    """Get current network condition"""
    return _network_monitor.get_current_condition()

def get_adaptive_download_settings(file_size_mb: float) -> AdaptiveSettings:
    """Get adaptive download settings for a file"""
    return _adaptive_optimizer.get_optimal_settings(file_size_mb)

# Test and calibration functions
def calibrate_network(test_download_func: Callable, test_sizes: List[int] = None):
    """
    Calibrate network conditions by performing test downloads.
    
    Args:
        test_download_func: Function that downloads test data
        test_sizes: List of test sizes in bytes (default: [1MB, 5MB])
    """
    if test_sizes is None:
        test_sizes = [1024*1024, 5*1024*1024]  # 1MB, 5MB
    
    logger.info("Starting network calibration...")
    start_network_monitoring()
    
    try:
        for size in test_sizes:
            logger.info(f"Calibrating with {size/1024/1024:.1f}MB test download...")
            
            start_time = time.time()
            try:
                test_download_func(size)
                elapsed = time.time() - start_time
                
                _network_monitor.record_download_sample(size, elapsed, size)
                
                # Brief pause between tests
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Calibration test failed: {e}")
        
        condition = get_current_network_condition()
        logger.info(f"Network calibration complete: {condition.quality_rating} "
                   f"({condition.bandwidth_mbps:.1f} Mbps, {condition.latency_ms:.1f}ms)")
        
    except Exception as e:
        logger.error(f"Network calibration failed: {e}")

if __name__ == "__main__":
    # Test the network condition monitoring
    print("🌐 Network Condition Adapter Test")
    print("=" * 40)
    
    monitor = NetworkConditionMonitor()
    optimizer = AdaptiveDownloadOptimizer(monitor)
    
    # Simulate some network samples
    monitor.record_download_sample(10*1024*1024, 2.0, 8*1024*1024)  # 10MB in 2s
    monitor.record_download_sample(5*1024*1024, 1.5, 4*1024*1024)   # 5MB in 1.5s
    monitor.record_latency_sample(50.0)  # 50ms latency
    
    condition = monitor.get_current_condition()
    print(f"Network condition: {condition}")
    
    # Test adaptive settings
    for file_size in [1, 10, 50, 100]:
        settings = optimizer.get_optimal_settings(file_size)
        print(f"Settings for {file_size}MB: {settings}")
