"""
Optimization Manager - Enterprise Download System with Smart Fallback
====================================================================

This module provides the OptimizedDownloadManager class that integrates all 5 
enterprise optimization systems with intelligent fallback to legacy methods.

Features:
- Smart fallback system for reliability
- All 5 optimization systems integrated
- Configurable optimization modes
- Performance monitoring and metrics
- Enterprise-grade reliability

Author: MegaPythonLibrary Enterprise Edition
Version: 3.0.0
Date: July 20, 2025
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass

# Import all optimization systems
try:
    from .network_condition_adapter import AdaptiveDownloadOptimizer, NetworkConditionMonitor
    NETWORK_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    NETWORK_OPTIMIZATION_AVAILABLE = False
    logging.warning(f"Network optimization not available: {e}")

try:
    from .advanced_error_recovery import AdvancedErrorRecovery
    ERROR_RECOVERY_AVAILABLE = True
except ImportError as e:
    ERROR_RECOVERY_AVAILABLE = False
    logging.warning(f"Error recovery optimization not available: {e}")

try:
    from .bandwidth_management import BandwidthManager
    BANDWIDTH_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    BANDWIDTH_MANAGEMENT_AVAILABLE = False
    logging.warning(f"Bandwidth management not available: {e}")

try:
    from .memory_optimization_polish import EnhancedMemoryOptimizer
    MEMORY_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    MEMORY_OPTIMIZATION_AVAILABLE = False
    logging.warning(f"Memory optimization not available: {e}")

try:
    from .http2_production_ready import UltimateHTTP2Client
    HTTP2_AVAILABLE = True
except ImportError as e:
    HTTP2_AVAILABLE = False
    logging.warning(f"HTTP/2 optimization not available: {e}")


class OptimizationMode(Enum):
    """Optimization modes for different use cases."""
    CONSERVATIVE = "conservative"  # Prefer legacy methods, minimal optimizations
    BALANCED = "balanced"         # Smart mix of optimizations with fallback
    AGGRESSIVE = "aggressive"     # Use all optimizations, minimal fallback
    LEGACY_ONLY = "legacy_only"   # Disable all optimizations
    OPTIMIZED_ONLY = "optimized_only"  # Force optimizations, no fallback


@dataclass
class OptimizationMetrics:
    """Metrics tracking for optimization performance."""
    optimization_attempts: int = 0
    optimization_successes: int = 0
    fallback_uses: int = 0
    total_download_time: float = 0.0
    total_bytes_downloaded: int = 0
    average_speed: float = 0.0
    optimization_failure_reasons: List[str] = None

    def __post_init__(self):
        if self.optimization_failure_reasons is None:
            self.optimization_failure_reasons = []

    @property
    def success_rate(self) -> float:
        """Calculate optimization success rate."""
        if self.optimization_attempts == 0:
            return 0.0
        return (self.optimization_successes / self.optimization_attempts) * 100

    @property
    def fallback_rate(self) -> float:
        """Calculate fallback usage rate."""
        total_downloads = self.optimization_attempts + self.fallback_uses
        if total_downloads == 0:
            return 0.0
        return (self.fallback_uses / total_downloads) * 100


class OptimizationError(Exception):
    """Custom exception for optimization failures."""
    pass


class OptimizedDownloadManager:
    """
    Enterprise download manager with 5 optimization systems and smart fallback.
    
    This class provides enterprise-grade download capabilities with:
    - Network condition adaptation
    - Advanced error recovery
    - Bandwidth management
    - Memory optimization
    - HTTP/2 support
    - Intelligent fallback to legacy methods
    """

    def __init__(self, 
                 optimization_mode: OptimizationMode = OptimizationMode.BALANCED,
                 fallback_enabled: bool = True,
                 max_optimization_failures: int = 3,
                 optimization_config: Dict[str, Any] = None):
        """
        Initialize the optimized download manager.
        
        Args:
            optimization_mode: Mode controlling optimization behavior
            fallback_enabled: Whether to fall back to legacy methods on failure
            max_optimization_failures: Max failures before switching to fallback
            optimization_config: Configuration for optimization systems
        """
        self.optimization_mode = optimization_mode
        self.fallback_enabled = fallback_enabled
        self.max_optimization_failures = max_optimization_failures
        self.config = optimization_config or {}
        
        # Initialize metrics
        self.metrics = OptimizationMetrics()
        
        # Initialize failure tracking
        self.optimization_failures = 0
        self.last_failure_time = None
        self.failure_cooldown = 300  # 5 minutes cooldown after failures
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization systems
        self._init_optimization_systems()
        
        # Legacy download method (will be set by MPLClient)
        self._legacy_download_method = None

    def _init_optimization_systems(self):
        """Initialize all available optimization systems."""
        # Network Condition Adapter
        if NETWORK_OPTIMIZATION_AVAILABLE:
            try:
                # Create monitor first, then adapter
                network_monitor = NetworkConditionMonitor()
                self.network_adapter = AdaptiveDownloadOptimizer(monitor=network_monitor)
                self.logger.info("Network condition adapter initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize network adapter: {e}")
                self.network_adapter = None
        else:
            self.network_adapter = None

        # Advanced Error Recovery
        if ERROR_RECOVERY_AVAILABLE:
            try:
                self.error_recovery = AdvancedErrorRecovery()
                self.logger.info("Advanced error recovery initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize error recovery: {e}")
                self.error_recovery = None
        else:
            self.error_recovery = None

        # Bandwidth Management
        if BANDWIDTH_MANAGEMENT_AVAILABLE:
            try:
                self.bandwidth_manager = BandwidthManager()
                self.logger.info("Bandwidth manager initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize bandwidth manager: {e}")
                self.bandwidth_manager = None
        else:
            self.bandwidth_manager = None

        # Memory Optimization
        if MEMORY_OPTIMIZATION_AVAILABLE:
            try:
                self.memory_optimizer = EnhancedMemoryOptimizer()
                self.logger.info("Memory optimizer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize memory optimizer: {e}")
                self.memory_optimizer = None
        else:
            self.memory_optimizer = None

        # HTTP/2 Support
        if HTTP2_AVAILABLE:
            try:
                self.http2_manager = UltimateHTTP2Client()
                self.logger.info("HTTP/2 manager initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize HTTP/2 manager: {e}")
                self.http2_manager = None
        else:
            self.http2_manager = None

    def set_legacy_download_method(self, method: Callable):
        """Set the legacy download method for fallback."""
        self._legacy_download_method = method

    def should_use_optimizations(self) -> bool:
        """Determine if optimizations should be used based on current state."""
        # Check optimization mode
        if self.optimization_mode == OptimizationMode.LEGACY_ONLY:
            return False
        elif self.optimization_mode == OptimizationMode.OPTIMIZED_ONLY:
            return True
        
        # Check failure count
        if self.optimization_failures >= self.max_optimization_failures:
            # Check if we're in cooldown period
            if self.last_failure_time:
                time_since_failure = time.time() - self.last_failure_time
                if time_since_failure < self.failure_cooldown:
                    self.logger.warning(f"In optimization cooldown for {self.failure_cooldown - time_since_failure:.0f}s")
                    return False
                else:
                    # Reset failure count after cooldown
                    self.optimization_failures = 0
                    self.last_failure_time = None
        
        return True

    def download(self, url: str, local_path: str, 
                progress_callback: Optional[Callable] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Enterprise download with optimization and smart fallback.
        
        Args:
            url: Download URL
            local_path: Local file path to save
            progress_callback: Optional progress callback function
            **kwargs: Additional download parameters
            
        Returns:
            Dictionary with download results and metrics
        """
        start_time = time.time()
        download_result = None
        used_optimization = False
        fallback_reason = None

        try:
            # Determine if we should use optimizations
            if self.should_use_optimizations():
                try:
                    self.metrics.optimization_attempts += 1
                    download_result = self._optimized_download(
                        url, local_path, progress_callback, **kwargs
                    )
                    used_optimization = True
                    self.metrics.optimization_successes += 1
                    self.logger.info("Optimized download completed successfully")
                    
                except OptimizationError as e:
                    fallback_reason = str(e)
                    self.optimization_failures += 1
                    self.last_failure_time = time.time()
                    self.metrics.optimization_failure_reasons.append(fallback_reason)
                    
                    if self.fallback_enabled:
                        self.logger.warning(f"Optimization failed, using fallback: {e}")
                        download_result = self._legacy_download(
                            url, local_path, progress_callback, **kwargs
                        )
                        self.metrics.fallback_uses += 1
                    else:
                        raise
            else:
                # Use legacy method directly
                download_result = self._legacy_download(
                    url, local_path, progress_callback, **kwargs
                )
                self.metrics.fallback_uses += 1
                fallback_reason = "Optimizations disabled or in cooldown"

        except Exception as e:
            # Final fallback if everything fails
            if self.fallback_enabled and not used_optimization:
                try:
                    self.logger.error(f"All methods failed, attempting final fallback: {e}")
                    download_result = self._legacy_download(
                        url, local_path, progress_callback, **kwargs
                    )
                    self.metrics.fallback_uses += 1
                    fallback_reason = f"Final fallback after error: {str(e)}"
                except Exception as final_error:
                    self.logger.error(f"Final fallback also failed: {final_error}")
                    raise
            else:
                raise

        # Update metrics
        end_time = time.time()
        download_time = end_time - start_time
        self.metrics.total_download_time += download_time

        # Prepare result
        result = {
            'success': download_result is not None,
            'used_optimization': used_optimization,
            'fallback_reason': fallback_reason,
            'download_time': download_time,
            'optimization_score': self.get_optimization_score()
        }

        if download_result:
            result.update(download_result)

        return result

    def _optimized_download(self, url: str, local_path: str, 
                          progress_callback: Optional[Callable] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Perform optimized download using all available optimization systems.
        
        This method integrates all 5 optimization systems for maximum performance.
        """
        try:
            # Step 1: Network Condition Analysis
            network_conditions = None
            if self.network_adapter:
                try:
                    network_conditions = self.network_adapter.analyze_network_conditions()
                    self.logger.debug(f"Network conditions: {network_conditions}")
                except Exception as e:
                    self.logger.warning(f"Network analysis failed: {e}")

            # Step 2: Memory Optimization Setup
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.prepare_for_download(url, local_path)
                except Exception as e:
                    self.logger.warning(f"Memory optimization setup failed: {e}")

            # Step 3: Bandwidth Management Setup
            download_priority = kwargs.get('priority', 'NORMAL')
            if self.bandwidth_manager:
                try:
                    self.bandwidth_manager.set_download_priority(download_priority)
                except Exception as e:
                    self.logger.warning(f"Bandwidth management setup failed: {e}")

            # Step 4: HTTP/2 Setup
            if self.http2_manager:
                try:
                    connection = self.http2_manager.get_optimized_connection(url)
                    kwargs['connection'] = connection
                except Exception as e:
                    self.logger.warning(f"HTTP/2 setup failed: {e}")

            # Step 5: Execute Download with Error Recovery
            if self.error_recovery:
                try:
                    download_result = self.error_recovery.download_with_recovery(
                        url, local_path, progress_callback, **kwargs
                    )
                except Exception as e:
                    raise OptimizationError(f"Error recovery failed: {e}")
            else:
                # If no error recovery, this would fail - need basic implementation
                raise OptimizationError("No error recovery system available")

            # Step 6: Post-download cleanup
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.cleanup_after_download()
                except Exception as e:
                    self.logger.warning(f"Memory cleanup failed: {e}")

            return download_result

        except Exception as e:
            raise OptimizationError(f"Optimized download failed: {e}")

    def _legacy_download(self, url: str, local_path: str,
                        progress_callback: Optional[Callable] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Fallback to legacy download method.
        """
        if not self._legacy_download_method:
            raise Exception("No legacy download method configured")
        
        try:
            return self._legacy_download_method(url, local_path, progress_callback, **kwargs)
        except Exception as e:
            self.logger.error(f"Legacy download failed: {e}")
            raise

    def get_optimization_score(self) -> float:
        """Calculate current optimization effectiveness score."""
        if self.metrics.optimization_attempts == 0:
            return 0.0
        
        base_score = self.metrics.success_rate
        
        # Adjust based on optimization mode
        if self.optimization_mode == OptimizationMode.AGGRESSIVE:
            return min(100.0, base_score * 1.2)
        elif self.optimization_mode == OptimizationMode.CONSERVATIVE:
            return base_score * 0.8
        
        return base_score

    def get_metrics(self) -> OptimizationMetrics:
        """Get current optimization metrics."""
        return self.metrics

    def reset_metrics(self):
        """Reset optimization metrics."""
        self.metrics = OptimizationMetrics()
        self.optimization_failures = 0
        self.last_failure_time = None

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization system status."""
        return {
            'optimization_mode': self.optimization_mode.value,
            'fallback_enabled': self.fallback_enabled,
            'optimization_failures': self.optimization_failures,
            'systems_available': {
                'network_adapter': self.network_adapter is not None,
                'error_recovery': self.error_recovery is not None,
                'bandwidth_manager': self.bandwidth_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'http2_manager': self.http2_manager is not None
            },
            'metrics': {
                'success_rate': self.metrics.success_rate,
                'fallback_rate': self.metrics.fallback_rate,
                'optimization_score': self.get_optimization_score()
            }
        }
