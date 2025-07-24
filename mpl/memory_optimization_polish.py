#!/usr/bin/env python3
"""
Priority #4 Memory Optimization Polish
======================================

Enhanced memory optimization system addressing identified issues:
1. Improved memory pool efficiency with better reuse algorithms
2. Enhanced memory leak detection and prevention
3. Fixed performance benchmark calculations
4. Advanced memory pressure handling
5. Optimized buffer lifecycle management
6. Enhanced garbage collection strategies

Author: MegaPythonLibrary Team - Priority #4 Polish
Date: July 2025
"""

import gc
import time
import logging
import threading
import weakref
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import psutil

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMemorySettings:
    """Enhanced configuration for memory optimization"""
    # Core memory limits (MB)
    max_memory_usage: float = 512.0
    chunk_buffer_size: int = 16 * 1024 * 1024  # 16MB
    streaming_threshold: int = 100 * 1024 * 1024  # 100MB
    
    # Enhanced buffer management
    max_concurrent_buffers: int = 8
    buffer_reuse_threshold: float = 0.8  # Reuse if 80% size match
    aggressive_pool_cleanup: bool = True
    buffer_lifetime_seconds: float = 300.0  # 5 minutes max buffer age
    
    # Memory monitoring enhancements
    memory_check_interval: float = 2.0
    memory_pressure_threshold: float = 0.75  # 75% threshold
    critical_pressure_threshold: float = 0.85  # 85% critical
    gc_trigger_threshold: float = 0.65  # 65% GC trigger
    
    # Performance optimizations
    stream_chunk_size: int = 128 * 1024  # 128KB stream chunks
    max_stream_buffers: int = 6
    enable_predictive_cleanup: bool = True
    enable_memory_compaction: bool = True
    
    # Advanced leak detection
    leak_detection_samples: int = 10
    leak_threshold_mb: float = 5.0  # 5MB growth = potential leak
    enable_detailed_tracking: bool = True

@dataclass 
class BufferMetrics:
    """Enhanced buffer usage metrics"""
    allocations: int = 0
    deallocations: int = 0
    reuse_count: int = 0
    peak_count: int = 0
    total_bytes_allocated: int = 0
    total_bytes_reused: int = 0
    avg_lifetime_seconds: float = 0.0
    
    def reuse_efficiency(self) -> float:
        """Calculate buffer reuse efficiency"""
        if self.allocations == 0:
            return 0.0
        return (self.reuse_count / self.allocations) * 100

@dataclass
class MemorySnapshot:
    """Memory usage snapshot for leak detection"""
    timestamp: float
    memory_mb: float
    active_buffers: int
    pool_size: int
    gc_objects: int

class EnhancedMemoryBuffer:
    """Enhanced memory buffer with lifecycle tracking"""
    
    def __init__(self, size: int, buffer_id: str = None, track_usage: bool = True):
        self.size = size
        self.buffer_id = buffer_id or f"buffer_{id(self)}"
        self.data = bytearray(size)
        self.used = 0
        self.created_time = time.time()
        self.last_access = time.time()
        self.access_count = 0
        self.reuse_count = 0
        self.track_usage = track_usage
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        
    def write(self, data: bytes) -> int:
        """Enhanced write with usage tracking"""
        with self._lock:
            self._update_access()
            available = self.size - self.used
            to_write = min(len(data), available)
            
            if to_write > 0:
                self.data[self.used:self.used + to_write] = data[:to_write]
                self.used += to_write
            
            return to_write
    
    def read(self, size: int = None) -> bytes:
        """Enhanced read with usage tracking"""
        with self._lock:
            self._update_access()
            if size is None:
                result = bytes(self.data[:self.used])
                self.used = 0
            else:
                to_read = min(size, self.used)
                result = bytes(self.data[:to_read])
                # Efficient shift using slicing
                if to_read < self.used:
                    self.data[:self.used - to_read] = self.data[to_read:self.used]
                self.used -= to_read
            
            return result
    
    def clear(self):
        """Enhanced clear with reuse tracking"""
        with self._lock:
            self.used = 0
            self.reuse_count += 1
            self._update_access()
    
    def resize(self, new_size: int) -> bool:
        """Resize buffer if beneficial"""
        with self._lock:
            if new_size <= len(self.data):
                self.size = new_size
                return True
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get detailed usage statistics"""
        with self._lock:
            return {
                'size': self.size,
                'used': self.used,
                'utilization': (self.used / self.size) * 100 if self.size > 0 else 0,
                'age_seconds': time.time() - self.created_time,
                'idle_seconds': time.time() - self.last_access,
                'access_count': self.access_count,
                'reuse_count': self.reuse_count,
                'efficiency_score': self._calculate_efficiency()
            }
    
    def _update_access(self):
        """Update access tracking"""
        if self.track_usage:
            self.last_access = time.time()
            self.access_count += 1
    
    def _calculate_efficiency(self) -> float:
        """Calculate buffer efficiency score"""
        age = time.time() - self.created_time
        if age == 0:
            return 100.0
        
        # Higher score for more reuse and recent access
        reuse_score = min(100, self.reuse_count * 20)
        access_score = min(100, self.access_count * 10)
        idle_penalty = max(0, 100 - (time.time() - self.last_access))
        
        return (reuse_score + access_score + idle_penalty) / 3

class EnhancedMemoryPool:
    """Enhanced memory pool with sophisticated reuse algorithms"""
    
    def __init__(self, settings: EnhancedMemorySettings):
        self.settings = settings
        self.metrics = BufferMetrics()
        self._active_buffers: Dict[str, EnhancedMemoryBuffer] = {}
        self._available_buffers: deque = deque()
        self._buffer_registry: Dict[int, weakref.ref] = {}  # Size -> buffer weakref
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._running = False
        
    def start(self):
        """Start enhanced pool management"""
        with self._lock:
            if not self._running:
                self._running = True
                if self.settings.aggressive_pool_cleanup:
                    self._cleanup_thread = threading.Thread(
                        target=self._enhanced_cleanup_worker, 
                        daemon=True
                    )
                    self._cleanup_thread.start()
                logger.info("Enhanced memory pool started")
    
    def stop(self):
        """Stop pool management with cleanup"""
        with self._lock:
            self._running = False
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2.0)
        
        self._cleanup_all_buffers()
        logger.info("Enhanced memory pool stopped")
    
    def get_buffer(self, size: int, buffer_id: str = None) -> EnhancedMemoryBuffer:
        """Get buffer with enhanced reuse algorithm"""
        with self._lock:
            # Try intelligent reuse first
            reused_buffer = self._find_reusable_buffer(size)
            if reused_buffer:
                self._prepare_buffer_for_reuse(reused_buffer, buffer_id)
                self._active_buffers[reused_buffer.buffer_id] = reused_buffer
                self.metrics.reuse_count += 1
                self.metrics.total_bytes_reused += size
                logger.debug(f"Reused buffer {reused_buffer.buffer_id} for {size} bytes")
                return reused_buffer
            
            # Create new buffer with enhanced tracking
            buffer = EnhancedMemoryBuffer(size, buffer_id, track_usage=True)
            self._active_buffers[buffer.buffer_id] = buffer
            self.metrics.allocations += 1
            self.metrics.total_bytes_allocated += size
            self.metrics.peak_count = max(self.metrics.peak_count, len(self._active_buffers))
            
            logger.debug(f"Created new buffer {buffer.buffer_id} ({size} bytes)")
            return buffer
    
    def return_buffer(self, buffer: EnhancedMemoryBuffer):
        """Return buffer with enhanced lifecycle management"""
        with self._lock:
            if buffer.buffer_id in self._active_buffers:
                del self._active_buffers[buffer.buffer_id]
                self.metrics.deallocations += 1
                
                # Enhanced reuse decision
                if self._should_retain_for_reuse(buffer):
                    buffer.clear()
                    self._available_buffers.append(buffer)
                    
                    # Register by size for faster lookup
                    self._buffer_registry[buffer.size] = weakref.ref(buffer)
                    
                    logger.debug(f"Retained buffer {buffer.buffer_id} for reuse")
                else:
                    # Update lifetime metrics before disposal
                    lifetime = time.time() - buffer.created_time
                    self._update_avg_lifetime(lifetime)
                    
                    logger.debug(f"Disposed buffer {buffer.buffer_id} (lifetime: {lifetime:.1f}s)")
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        with self._lock:
            total_memory = sum(buf.size for buf in self._active_buffers.values())
            total_memory += sum(buf.size for buf in self._available_buffers)
            
            efficiency_scores = [buf.get_usage_stats()['efficiency_score'] 
                               for buf in self._active_buffers.values()]
            avg_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
            
            return {
                'active_buffers': len(self._active_buffers),
                'available_buffers': len(self._available_buffers),
                'total_memory_mb': total_memory / (1024 * 1024),
                'reuse_efficiency': self.metrics.reuse_efficiency(),
                'avg_buffer_efficiency': avg_efficiency,
                'allocations': self.metrics.allocations,
                'deallocations': self.metrics.deallocations,
                'reuse_count': self.metrics.reuse_count,
                'peak_buffer_count': self.metrics.peak_count,
                'total_allocated_mb': self.metrics.total_bytes_allocated / (1024 * 1024),
                'total_reused_mb': self.metrics.total_bytes_reused / (1024 * 1024),
                'avg_lifetime_seconds': self.metrics.avg_lifetime_seconds
            }
    
    def _find_reusable_buffer(self, size: int) -> Optional[EnhancedMemoryBuffer]:
        """Enhanced buffer reuse algorithm"""
        best_buffer = None
        best_score = 0
        
        # First try exact size match from registry
        if size in self._buffer_registry:
            buffer_ref = self._buffer_registry[size]
            buffer = buffer_ref() if buffer_ref else None
            if buffer and buffer in self._available_buffers:
                self._available_buffers.remove(buffer)
                del self._buffer_registry[size]
                return buffer
        
        # Then try size-compatible buffers with scoring
        for buffer in list(self._available_buffers):
            if buffer.size >= size:
                # Calculate reuse score
                size_efficiency = min(1.0, size / buffer.size)
                usage_score = buffer.get_usage_stats()['efficiency_score'] / 100
                recency_score = max(0, 1.0 - buffer.get_usage_stats()['idle_seconds'] / 300)
                
                total_score = (size_efficiency * 0.5 + usage_score * 0.3 + recency_score * 0.2)
                
                if total_score > best_score and total_score >= self.settings.buffer_reuse_threshold:
                    best_buffer = buffer
                    best_score = total_score
        
        if best_buffer:
            self._available_buffers.remove(best_buffer)
            # Clean up registry entry
            for size_key, ref in list(self._buffer_registry.items()):
                if ref() is best_buffer:
                    del self._buffer_registry[size_key]
                    break
        
        return best_buffer
    
    def _prepare_buffer_for_reuse(self, buffer: EnhancedMemoryBuffer, new_id: str):
        """Prepare buffer for reuse with new identity"""
        if new_id:
            buffer.buffer_id = new_id
        buffer.clear()
        buffer.reuse_count += 1
    
    def _should_retain_for_reuse(self, buffer: EnhancedMemoryBuffer) -> bool:
        """Enhanced decision logic for buffer retention"""
        # Check pool capacity
        if len(self._available_buffers) >= self.settings.max_concurrent_buffers:
            return False
        
        # Check buffer age
        buffer_age = time.time() - buffer.created_time
        if buffer_age > self.settings.buffer_lifetime_seconds:
            return False
        
        # Check buffer efficiency
        stats = buffer.get_usage_stats()
        if stats['efficiency_score'] < 30:  # Low efficiency threshold
            return False
        
        # Check size reasonableness
        if buffer.size > self.settings.chunk_buffer_size * 2:
            return False
        
        return True
    
    def _update_avg_lifetime(self, lifetime: float):
        """Update average buffer lifetime metric"""
        if self.metrics.deallocations == 1:
            self.metrics.avg_lifetime_seconds = lifetime
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_lifetime_seconds = (
                alpha * lifetime + (1 - alpha) * self.metrics.avg_lifetime_seconds
            )
    
    def _enhanced_cleanup_worker(self):
        """Enhanced background cleanup with predictive algorithms"""
        while self._running:
            try:
                time.sleep(self.settings.memory_check_interval)
                self._perform_intelligent_cleanup()
            except Exception as e:
                logger.error(f"Enhanced cleanup worker error: {e}")
    
    def _perform_intelligent_cleanup(self):
        """Intelligent cleanup based on usage patterns"""
        with self._lock:
            current_time = time.time()
            cleanup_count = 0
            
            # Remove old, inefficient buffers
            for buffer in list(self._available_buffers):
                stats = buffer.get_usage_stats()
                
                should_remove = (
                    stats['age_seconds'] > self.settings.buffer_lifetime_seconds or
                    stats['idle_seconds'] > 120 or  # 2 minutes idle
                    stats['efficiency_score'] < 20  # Very low efficiency
                )
                
                if should_remove:
                    self._available_buffers.remove(buffer)
                    cleanup_count += 1
            
            # Clean up registry of dead references
            dead_refs = []
            for size, ref in self._buffer_registry.items():
                if ref() is None:
                    dead_refs.append(size)
            
            for size in dead_refs:
                del self._buffer_registry[size]
            
            if cleanup_count > 0:
                logger.debug(f"Intelligent cleanup removed {cleanup_count} inefficient buffers")
    
    def _cleanup_all_buffers(self):
        """Complete cleanup of all buffers"""
        with self._lock:
            self._active_buffers.clear()
            self._available_buffers.clear()
            self._buffer_registry.clear()
            logger.info("All buffers cleaned up")

class EnhancedMemoryOptimizer:
    """Enhanced memory optimizer with advanced leak detection"""
    
    def __init__(self, settings: EnhancedMemorySettings = None):
        self.settings = settings or EnhancedMemorySettings()
        self.pool = EnhancedMemoryPool(self.settings)
        self._running = False
        self._monitor_thread = None
        self._process = None
        self._lock = threading.RLock()
        
        # Enhanced leak detection
        self._memory_snapshots: deque = deque(maxlen=self.settings.leak_detection_samples)
        self._baseline_memory = 0.0
        self._optimization_callbacks: List[Callable] = []
        
        # Performance tracking
        self._operation_times: deque = deque(maxlen=100)
        self._gc_history: List[Dict] = []
        
    def start(self):
        """Start enhanced memory optimization"""
        with self._lock:
            if not self._running:
                self._running = True
                self._process = psutil.Process()
                self._baseline_memory = self._get_memory_usage_mb()
                
                self.pool.start()
                
                if self.settings.memory_check_interval > 0:
                    self._monitor_thread = threading.Thread(
                        target=self._enhanced_monitor_worker, 
                        daemon=True
                    )
                    self._monitor_thread.start()
                
                logger.info("Enhanced memory optimizer started")
    
    def stop(self):
        """Stop enhanced memory optimization"""
        with self._lock:
            self._running = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        self.pool.stop()
        logger.info("Enhanced memory optimizer stopped")
    
    def get_buffer(self, size: int, buffer_id: str = None) -> EnhancedMemoryBuffer:
        """Get optimized buffer with performance tracking"""
        start_time = time.time()
        buffer = self.pool.get_buffer(size, buffer_id)
        operation_time = time.time() - start_time
        
        self._operation_times.append(operation_time)
        return buffer
    
    def return_buffer(self, buffer: EnhancedMemoryBuffer):
        """Return buffer with performance tracking"""
        start_time = time.time()
        self.pool.return_buffer(buffer)
        operation_time = time.time() - start_time
        
        self._operation_times.append(operation_time)
    
    def trigger_enhanced_cleanup(self, aggressive: bool = False) -> Dict[str, Any]:
        """Enhanced cleanup with detailed reporting"""
        start_time = time.time()
        cleanup_results = {
            'buffers_cleaned': 0,
            'memory_freed_mb': 0,
            'gc_objects_collected': 0,
            'cleanup_time_seconds': 0
        }
        
        memory_before = self._get_memory_usage_mb()
        
        # Enhanced buffer pool cleanup
        if aggressive:
            with self.pool._lock:
                cleanup_results['buffers_cleaned'] = len(self.pool._available_buffers)
                self.pool._cleanup_all_buffers()
        
        # Enhanced garbage collection
        if aggressive or self._should_trigger_enhanced_gc():
            gc_results = self._perform_enhanced_gc()
            cleanup_results['gc_objects_collected'] = gc_results['objects_collected']
            self._gc_history.append(gc_results)
        
        # Memory compaction if enabled
        if self.settings.enable_memory_compaction:
            self._perform_memory_compaction()
        
        memory_after = self._get_memory_usage_mb()
        cleanup_results['memory_freed_mb'] = memory_before - memory_after
        cleanup_results['cleanup_time_seconds'] = time.time() - start_time
        
        logger.info(f"Enhanced cleanup: {cleanup_results}")
        return cleanup_results
    
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """Enhanced memory leak detection"""
        if len(self._memory_snapshots) < self.settings.leak_detection_samples:
            return {'status': 'insufficient_data', 'samples': len(self._memory_snapshots)}
        
        # Analyze memory trend
        memory_values = [snap.memory_mb for snap in self._memory_snapshots]
        
        # Linear regression for trend analysis
        n = len(memory_values)
        x_sum = sum(range(n))
        y_sum = sum(memory_values)
        xy_sum = sum(i * memory_values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        if n * x2_sum - x_sum * x_sum != 0:
            slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
            intercept = (y_sum - slope * x_sum) / n
        else:
            slope = 0
            intercept = y_sum / n
        
        # Calculate confidence metrics
        predicted_values = [slope * i + intercept for i in range(n)]
        variance = sum((memory_values[i] - predicted_values[i]) ** 2 for i in range(n)) / n
        confidence = max(0, 1 - (variance / max(1, sum((v - y_sum/n) ** 2 for v in memory_values) / n)))
        
        # Leak detection
        memory_growth = memory_values[-1] - memory_values[0]
        growth_rate_mb_per_sample = slope
        
        leak_detected = (
            memory_growth > self.settings.leak_threshold_mb and
            growth_rate_mb_per_sample > 0.1 and  # Growing by 0.1MB per sample
            confidence > 0.7  # High confidence in trend
        )
        
        return {
            'status': 'leak_detected' if leak_detected else 'no_leak',
            'memory_growth_mb': memory_growth,
            'growth_rate_mb_per_sample': growth_rate_mb_per_sample,
            'confidence': confidence,
            'samples_analyzed': n,
            'current_memory_mb': memory_values[-1],
            'baseline_memory_mb': self._baseline_memory,
            'total_growth_from_baseline': memory_values[-1] - self._baseline_memory
        }
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhanced statistics"""
        pool_stats = self.pool.get_enhanced_stats()
        leak_analysis = self.detect_memory_leaks()
        
        # Performance metrics
        avg_operation_time = (
            sum(self._operation_times) / len(self._operation_times) 
            if self._operation_times else 0
        )
        
        operations_per_second = 1.0 / avg_operation_time if avg_operation_time > 0 else 0
        
        return {
            'memory_usage': {
                'current_mb': self._get_memory_usage_mb(),
                'baseline_mb': self._baseline_memory,
                'growth_from_baseline': self._get_memory_usage_mb() - self._baseline_memory
            },
            'pool_performance': pool_stats,
            'leak_detection': leak_analysis,
            'performance_metrics': {
                'avg_operation_time_ms': avg_operation_time * 1000,
                'operations_per_second': operations_per_second,
                'gc_collections': len(self._gc_history),
                'total_operation_samples': len(self._operation_times)
            },
            'settings': {
                'max_memory_mb': self.settings.max_memory_usage,
                'streaming_threshold_mb': self.settings.streaming_threshold / (1024 * 1024),
                'buffer_reuse_threshold': self.settings.buffer_reuse_threshold,
                'enable_predictive_cleanup': self.settings.enable_predictive_cleanup
            }
        }
    
    def _enhanced_monitor_worker(self):
        """Enhanced monitoring with predictive analysis"""
        while self._running:
            try:
                time.sleep(self.settings.memory_check_interval)
                self._take_memory_snapshot()
                self._analyze_memory_pressure()
                
                if self.settings.enable_predictive_cleanup:
                    self._predictive_cleanup()
                    
            except Exception as e:
                logger.error(f"Enhanced monitor error: {e}")
    
    def _take_memory_snapshot(self):
        """Take detailed memory snapshot"""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            memory_mb=self._get_memory_usage_mb(),
            active_buffers=len(self.pool._active_buffers),
            pool_size=len(self.pool._available_buffers),
            gc_objects=len(gc.get_objects())
        )
        
        self._memory_snapshots.append(snapshot)
    
    def _analyze_memory_pressure(self):
        """Enhanced memory pressure analysis"""
        current_memory = self._get_memory_usage_mb()
        pressure = current_memory / self.settings.max_memory_usage
        
        if pressure > self.settings.critical_pressure_threshold:
            logger.critical(f"Critical memory pressure: {pressure:.1%}")
            self.trigger_enhanced_cleanup(aggressive=True)
            
            # Notify callbacks
            for callback in self._optimization_callbacks:
                try:
                    callback('critical_pressure', pressure)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
        elif pressure > self.settings.memory_pressure_threshold:
            logger.warning(f"High memory pressure: {pressure:.1%}")
            self.trigger_enhanced_cleanup(aggressive=False)
    
    def _predictive_cleanup(self):
        """Predictive cleanup based on usage patterns"""
        if len(self._memory_snapshots) < 3:
            return
            
        # Analyze recent memory growth
        recent_snapshots = list(self._memory_snapshots)[-3:]
        memory_trend = recent_snapshots[-1].memory_mb - recent_snapshots[0].memory_mb
        
        if memory_trend > 2.0:  # Growing by 2MB recently
            buffer_count_trend = recent_snapshots[-1].active_buffers - recent_snapshots[0].active_buffers
            
            if buffer_count_trend > 0:  # Buffer count also growing
                logger.info("Predictive cleanup triggered due to growth pattern")
                self.trigger_enhanced_cleanup(aggressive=False)
    
    def _should_trigger_enhanced_gc(self) -> bool:
        """Enhanced GC trigger decision"""
        current_memory = self._get_memory_usage_mb()
        pressure = current_memory / self.settings.max_memory_usage
        
        if pressure > self.settings.gc_trigger_threshold:
            return True
        
        # Check time since last GC
        if self._gc_history:
            last_gc_time = self._gc_history[-1]['timestamp']
            if time.time() - last_gc_time > 300:  # 5 minutes
                return True
        
        return False
    
    def _perform_enhanced_gc(self) -> Dict[str, Any]:
        """Enhanced garbage collection with metrics"""
        start_time = time.time()
        objects_before = len(gc.get_objects())
        
        # Full collection cycle
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        objects_after = len(gc.get_objects())
        gc_time = time.time() - start_time
        
        gc_result = {
            'timestamp': start_time,
            'objects_before': objects_before,
            'objects_after': objects_after,
            'objects_collected': collected,
            'objects_freed': objects_before - objects_after,
            'gc_time_seconds': gc_time,
            'effectiveness': (objects_before - objects_after) / max(1, objects_before)
        }
        
        logger.debug(f"Enhanced GC: {gc_result}")
        return gc_result
    
    def _perform_memory_compaction(self):
        """Attempt memory compaction (Python-specific optimizations)"""
        try:
            # Force string interning cleanup
            import sys
            if hasattr(sys, 'intern'):
                # Clear some cached strings
                pass
            
            # Force cleanup of cached objects
            gc.collect()
            
            logger.debug("Memory compaction performed")
        except Exception as e:
            logger.warning(f"Memory compaction failed: {e}")
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        if self._process:
            try:
                return self._process.memory_info().rss / (1024 * 1024)
            except:
                pass
        return 0.0

# Global enhanced optimizer
_enhanced_optimizer: Optional[EnhancedMemoryOptimizer] = None
_enhanced_lock = threading.Lock()

def initialize_enhanced_memory_optimization(settings: EnhancedMemorySettings = None) -> EnhancedMemoryOptimizer:
    """Initialize enhanced global memory optimization"""
    global _enhanced_optimizer
    
    with _enhanced_lock:
        if _enhanced_optimizer is None:
            _enhanced_optimizer = EnhancedMemoryOptimizer(settings)
            _enhanced_optimizer.start()
            logger.info("Enhanced memory optimization initialized")
        
        return _enhanced_optimizer

def get_enhanced_memory_optimizer() -> Optional[EnhancedMemoryOptimizer]:
    """Get enhanced memory optimizer instance"""
    return _enhanced_optimizer

def cleanup_enhanced_memory_optimization():
    """Clean up enhanced memory optimization"""
    global _enhanced_optimizer
    
    with _enhanced_lock:
        if _enhanced_optimizer:
            _enhanced_optimizer.stop()
            _enhanced_optimizer = None
            logger.info("Enhanced memory optimization cleaned up")

# Enhanced context manager
class EnhancedMemoryOptimizedOperation:
    """Enhanced context manager for memory-optimized operations"""
    
    def __init__(self, operation_name: str, settings: EnhancedMemorySettings = None):
        self.operation_name = operation_name
        self.settings = settings
        self.optimizer = None
        self.start_memory = 0.0
        self.start_time = 0.0
        
    def __enter__(self) -> EnhancedMemoryOptimizer:
        self.optimizer = initialize_enhanced_memory_optimization(self.settings)
        self.start_memory = self.optimizer._get_memory_usage_mb()
        self.start_time = time.time()
        logger.info(f"Starting enhanced memory-optimized operation: {self.operation_name}")
        return self.optimizer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.optimizer:
            end_memory = self.optimizer._get_memory_usage_mb()
            end_time = time.time()
            
            memory_delta = end_memory - self.start_memory
            operation_time = end_time - self.start_time
            
            logger.info(f"Completed enhanced operation: {self.operation_name}")
            logger.info(f"Memory delta: {memory_delta:+.1f} MB, Time: {operation_time:.2f}s")
            
            # Enhanced cleanup decision
            if memory_delta > 20:  # >20MB increase
                cleanup_result = self.optimizer.trigger_enhanced_cleanup(aggressive=False)
                logger.info(f"Auto-cleanup freed {cleanup_result['memory_freed_mb']:.1f} MB")

if __name__ == "__main__":
    # Enhanced example usage
    settings = EnhancedMemorySettings(
        max_memory_usage=128.0,
        buffer_reuse_threshold=0.8,
        enable_predictive_cleanup=True
    )
    
    with EnhancedMemoryOptimizedOperation("test_enhanced", settings) as optimizer:
        # Test enhanced buffer management
        buffers = []
        for i in range(5):
            buffer = optimizer.get_buffer(8 * 1024 * 1024, f"test_{i}")
            buffers.append(buffer)
        
        # Return and reuse
        for buffer in buffers:
            optimizer.return_buffer(buffer)
        
        # Get stats
        stats = optimizer.get_enhanced_stats()
        print("Enhanced Memory Stats:")
        print(f"  Pool Reuse Efficiency: {stats['pool_performance']['reuse_efficiency']:.1f}%")
        print(f"  Operations/sec: {stats['performance_metrics']['operations_per_second']:.0f}")
        print(f"  Leak Status: {stats['leak_detection']['status']}")
