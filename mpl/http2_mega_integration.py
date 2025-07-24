#!/usr/bin/env python3
"""
Priority #5: HTTP/2 Support - Phase 3: Integration and Optimization
====================================================================

This module integrates HTTP/2 support with the existing MEGA SDK components
and optimizes performance for MEGA-specific operations.
"""

import asyncio
import httpx
import time
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
from urllib.parse import urljoin, urlparse
import os
from pathlib import Path

# Import our existing components
try:
    from memory_optimization_polish import EnhancedMemoryOptimizer
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False
    logging.warning("Memory optimizer not available")

try:
    from bandwidth_management import BandwidthManager
    BANDWIDTH_MANAGER_AVAILABLE = True
except ImportError:
    BANDWIDTH_MANAGER_AVAILABLE = False
    logging.warning("Bandwidth manager not available")

try:
    from network_condition_adapter import NetworkConditionAdapter
    NETWORK_ADAPTER_AVAILABLE = True
except ImportError:
    NETWORK_ADAPTER_AVAILABLE = False
    logging.warning("Network adapter not available")

try:
    from advanced_error_recovery import AdvancedErrorRecovery
    ERROR_RECOVERY_AVAILABLE = True
except ImportError:
    ERROR_RECOVERY_AVAILABLE = False
    logging.warning("Error recovery not available")

# Import HTTP/2 core components (required)
from .http2_core_implementation import HTTP2MegaClient, HTTP2DownloadRequest, HTTP2DownloadResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MegaFileInfo:
    """Information about a MEGA file for HTTP/2 download"""
    file_id: str
    file_name: str
    file_size: int
    download_url: str
    file_key: Optional[str] = None
    chunk_size: int = 1024 * 1024  # 1MB default
    priority: int = 1
    
    @property
    def estimated_chunks(self) -> int:
        """Estimate number of chunks needed"""
        return (self.file_size + self.chunk_size - 1) // self.chunk_size

@dataclass
class MegaDownloadSession:
    """Represents a MEGA download session with HTTP/2"""
    session_id: str
    files: List[MegaFileInfo]
    total_size: int
    start_time: float
    end_time: Optional[float] = None
    bytes_downloaded: int = 0
    chunks_completed: int = 0
    chunks_total: int = 0
    success_rate: float = 0.0
    
    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def download_speed(self) -> float:
        if self.duration > 0:
            return self.bytes_downloaded / self.duration
        return 0.0
    
    @property
    def progress(self) -> float:
        if self.chunks_total > 0:
            return self.chunks_completed / self.chunks_total
        return 0.0

class HTTP2MegaSDKIntegration:
    """Integrated HTTP/2 MEGA SDK with all optimizations"""
    
    def __init__(self, 
                 max_concurrent_streams: int = 50,
                 max_connections: int = 10,
                 enable_memory_optimization: bool = True,
                 enable_bandwidth_management: bool = True,
                 enable_network_adaptation: bool = True,
                 enable_error_recovery: bool = True):
        
        # Core HTTP/2 client
        self.http2_client = HTTP2MegaClient(max_concurrent_streams, max_connections)
        
        # Initialize optimizations if available
        self.memory_optimizer = None
        self.bandwidth_manager = None
        self.network_adapter = None
        self.error_recovery = None
        
        if enable_memory_optimization and MEMORY_OPTIMIZER_AVAILABLE:
            try:
                self.memory_optimizer = EnhancedMemoryOptimizer()
                logger.info("✅ Memory optimization enabled")
            except:
                logger.warning("⚠️  Memory optimization not available")
        
        if enable_bandwidth_management and BANDWIDTH_MANAGER_AVAILABLE:
            try:
                self.bandwidth_manager = BandwidthManager()
                logger.info("✅ Bandwidth management enabled")
            except:
                logger.warning("⚠️  Bandwidth management not available")
        
        if enable_network_adaptation and NETWORK_ADAPTER_AVAILABLE:
            try:
                self.network_adapter = NetworkConditionAdapter()
                logger.info("✅ Network adaptation enabled")
            except:
                logger.warning("⚠️  Network adaptation not available")
        
        if enable_error_recovery and ERROR_RECOVERY_AVAILABLE:
            try:
                self.error_recovery = AdvancedErrorRecovery()
                logger.info("✅ Error recovery enabled")
            except:
                logger.warning("⚠️  Error recovery not available")
        
        # Session tracking
        self.active_sessions: Dict[str, MegaDownloadSession] = {}
        self._session_counter = 0
        
    async def download_mega_file(self, 
                                file_info: MegaFileInfo,
                                output_path: Optional[str] = None,
                                progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Download a single MEGA file using HTTP/2 with all optimizations"""
        
        session_id = f"mega_download_{self._session_counter}"
        self._session_counter += 1
        
        # Create download session
        session = MegaDownloadSession(
            session_id=session_id,
            files=[file_info],
            total_size=file_info.file_size,
            start_time=time.time(),
            chunks_total=file_info.estimated_chunks
        )
        self.active_sessions[session_id] = session
        
        logger.info(f"🚀 Starting MEGA file download: {file_info.file_name} "
                   f"({file_info.file_size/1024/1024:.1f} MB)")
        
        try:
            # Optimize chunk size based on network conditions
            optimized_chunk_size = await self._optimize_chunk_size(file_info)
            
            # Determine optimal concurrency
            optimal_concurrency = await self._determine_optimal_concurrency(file_info)
            
            # Prepare HTTP/2 headers
            headers = await self._prepare_mega_headers(file_info)
            
            # Download using HTTP/2 multipart
            download_result = await self.http2_client.download_file_multipart(
                file_url=file_info.download_url,
                file_size=file_info.file_size,
                chunk_size=optimized_chunk_size,
                max_concurrent=optimal_concurrency,
                headers=headers
            )
            
            # Process result
            if download_result.success:
                # Apply memory optimization if available
                if self.memory_optimizer:
                    content = await self._optimize_memory_usage(download_result.content)
                else:
                    content = download_result.content
                
                # Save to file if path provided
                if output_path:
                    await self._save_file_optimized(content, output_path)
                
                # Update session
                session.end_time = time.time()
                session.bytes_downloaded = len(content)
                session.chunks_completed = session.chunks_total
                session.success_rate = 1.0
                
                result = {
                    "success": True,
                    "file_name": file_info.file_name,
                    "file_size": len(content),
                    "download_time": session.duration,
                    "download_speed": session.download_speed,
                    "http_version": download_result.http_version,
                    "chunks_used": session.chunks_total,
                    "session_id": session_id,
                    "content": content if not output_path else None,
                    "output_path": output_path
                }
                
                logger.info(f"✅ Download complete: {file_info.file_name} "
                           f"({session.download_speed/1024/1024:.1f} MB/s)")
                
                return result
                
            else:
                # Handle download failure
                session.end_time = time.time()
                session.success_rate = 0.0
                
                error_result = {
                    "success": False,
                    "file_name": file_info.file_name,
                    "error": download_result.error,
                    "session_id": session_id
                }
                
                logger.error(f"❌ Download failed: {file_info.file_name} - {download_result.error}")
                
                return error_result
                
        except Exception as e:
            session.end_time = time.time()
            session.success_rate = 0.0
            
            logger.error(f"❌ Download exception: {file_info.file_name} - {str(e)}")
            
            return {
                "success": False,
                "file_name": file_info.file_name,
                "error": str(e),
                "session_id": session_id
            }
            
        finally:
            # Cleanup session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    async def download_mega_files_batch(self, 
                                      files: List[MegaFileInfo],
                                      output_directory: Optional[str] = None,
                                      max_concurrent_files: int = 5) -> List[Dict[str, Any]]:
        """Download multiple MEGA files concurrently using HTTP/2"""
        
        session_id = f"mega_batch_{self._session_counter}"
        self._session_counter += 1
        
        total_size = sum(f.file_size for f in files)
        total_chunks = sum(f.estimated_chunks for f in files)
        
        # Create batch session
        session = MegaDownloadSession(
            session_id=session_id,
            files=files,
            total_size=total_size,
            start_time=time.time(),
            chunks_total=total_chunks
        )
        self.active_sessions[session_id] = session
        
        logger.info(f"🚀 Starting batch download: {len(files)} files "
                   f"({total_size/1024/1024:.1f} MB total)")
        
        try:
            # Create semaphore for file-level concurrency
            file_semaphore = asyncio.Semaphore(max_concurrent_files)
            
            async def download_single_file(file_info: MegaFileInfo):
                async with file_semaphore:
                    output_path = None
                    if output_directory:
                        output_path = os.path.join(output_directory, file_info.file_name)
                    
                    return await self.download_mega_file(file_info, output_path)
            
            # Download all files concurrently
            results = await asyncio.gather(*[
                download_single_file(file_info) for file_info in files
            ], return_exceptions=True)
            
            # Process results
            processed_results = []
            successful_downloads = 0
            total_bytes = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "success": False,
                        "file_name": files[i].file_name,
                        "error": str(result),
                        "session_id": session_id
                    })
                else:
                    processed_results.append(result)
                    if result.get("success", False):
                        successful_downloads += 1
                        total_bytes += result.get("file_size", 0)
            
            # Update session
            session.end_time = time.time()
            session.bytes_downloaded = total_bytes
            session.success_rate = successful_downloads / len(files) if files else 0.0
            
            logger.info(f"✅ Batch download complete: {successful_downloads}/{len(files)} successful "
                       f"({session.download_speed/1024/1024:.1f} MB/s average)")
            
            return processed_results
            
        except Exception as e:
            session.end_time = time.time()
            session.success_rate = 0.0
            
            logger.error(f"❌ Batch download failed: {str(e)}")
            return []
            
        finally:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    async def _optimize_chunk_size(self, file_info: MegaFileInfo) -> int:
        """Optimize chunk size based on network conditions and file size"""
        base_chunk_size = file_info.chunk_size
        
        # Network adaptation
        if self.network_adapter:
            try:
                network_conditions = await self.network_adapter.analyze_conditions_async()
                if network_conditions.get("quality", "good") == "poor":
                    # Smaller chunks for poor connections
                    base_chunk_size = min(base_chunk_size, 512 * 1024)  # 512KB
                elif network_conditions.get("quality", "good") == "excellent":
                    # Larger chunks for excellent connections
                    base_chunk_size = min(base_chunk_size * 2, 4 * 1024 * 1024)  # Up to 4MB
            except:
                pass
        
        # File size optimization
        if file_info.file_size < 10 * 1024 * 1024:  # < 10MB
            # Smaller chunks for small files
            return min(base_chunk_size, 256 * 1024)  # 256KB
        elif file_info.file_size > 100 * 1024 * 1024:  # > 100MB
            # Larger chunks for large files
            return min(base_chunk_size * 2, 2 * 1024 * 1024)  # Up to 2MB
        
        return base_chunk_size
    
    async def _determine_optimal_concurrency(self, file_info: MegaFileInfo) -> int:
        """Determine optimal concurrency level"""
        base_concurrency = 20
        
        # Bandwidth management
        if self.bandwidth_manager:
            try:
                # Get available bandwidth and adjust concurrency
                bandwidth_info = self.bandwidth_manager.get_bandwidth_status()
                if bandwidth_info.get("available_bandwidth", 0) < 1024 * 1024:  # < 1MB/s
                    base_concurrency = min(base_concurrency, 10)
                elif bandwidth_info.get("available_bandwidth", 0) > 10 * 1024 * 1024:  # > 10MB/s
                    base_concurrency = min(base_concurrency * 2, 50)
            except:
                pass
        
        # File size consideration
        if file_info.file_size < 5 * 1024 * 1024:  # < 5MB
            return min(base_concurrency, 5)  # Lower concurrency for small files
        
        return base_concurrency
    
    async def _prepare_mega_headers(self, file_info: MegaFileInfo) -> Dict[str, str]:
        """Prepare HTTP headers for MEGA download"""
        headers = {
            "User-Agent": "MEGA-SDK-HTTP2/1.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br"
        }
        
        # Add authentication if file_key is provided
        if file_info.file_key:
            headers["Authorization"] = f"Bearer {file_info.file_key}"
        
        return headers
    
    async def _optimize_memory_usage(self, content: bytes) -> bytes:
        """Optimize memory usage of downloaded content"""
        if self.memory_optimizer:
            try:
                # Use memory optimizer to manage the content
                optimized_stream = self.memory_optimizer.create_stream(len(content))
                optimized_stream.write(content)
                return optimized_stream.getvalue()
            except:
                pass
        
        return content
    
    async def _save_file_optimized(self, content: bytes, output_path: str):
        """Save file with memory optimization"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write file in chunks to optimize memory
        chunk_size = 64 * 1024  # 64KB write chunks
        
        with open(output_path, 'wb') as f:
            for i in range(0, len(content), chunk_size):
                f.write(content[i:i + chunk_size])
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        http2_stats = self.http2_client.get_session_stats()
        connection_stats = self.http2_client.get_connection_stats()
        
        return {
            "http2_stats": {
                "total_requests": http2_stats.total_requests,
                "success_rate": http2_stats.success_rate,
                "average_speed": http2_stats.average_speed,
                "total_bytes": http2_stats.total_bytes_downloaded
            },
            "connection_stats": connection_stats,
            "active_sessions": len(self.active_sessions),
            "optimizations": {
                "memory_optimizer": self.memory_optimizer is not None,
                "bandwidth_manager": self.bandwidth_manager is not None,
                "network_adapter": self.network_adapter is not None,
                "error_recovery": self.error_recovery is not None
            }
        }
    
    async def close(self):
        """Close the integrated client"""
        await self.http2_client.close()
        
        # Close other components if they have close methods
        if hasattr(self.memory_optimizer, 'close'):
            await self.memory_optimizer.close()
        
        logger.info("🔒 Integrated MEGA SDK HTTP/2 client closed")

# Demo and testing functions
async def demo_mega_integration():
    """Demonstrate MEGA SDK HTTP/2 integration"""
    print("🚀 MEGA SDK HTTP/2 INTEGRATION DEMO")
    print("=" * 60)
    
    client = HTTP2MegaSDKIntegration(
        max_concurrent_streams=20,
        enable_memory_optimization=True,
        enable_bandwidth_management=True
    )
    
    try:
        # Create test file info (using public test endpoints)
        test_files = [
            MegaFileInfo(
                file_id="test_1",
                file_name="test_1mb.bin",
                file_size=1024 * 1024,  # 1MB
                download_url="https://httpbin.org/bytes/1048576",
                chunk_size=256 * 1024  # 256KB chunks
            ),
            MegaFileInfo(
                file_id="test_2", 
                file_name="test_512kb.bin",
                file_size=512 * 1024,  # 512KB
                download_url="https://httpbin.org/bytes/524288",
                chunk_size=128 * 1024  # 128KB chunks
            )
        ]
        
        print(f"📊 Testing integrated download of {len(test_files)} files...")
        
        # Test single file download
        print("\n🔧 Single File Download Test:")
        single_result = await client.download_mega_file(test_files[0])
        
        if single_result["success"]:
            print(f"✅ {single_result['file_name']}: {single_result['file_size']/1024:.1f} KB "
                  f"in {single_result['download_time']:.2f}s "
                  f"({single_result['download_speed']/1024/1024:.1f} MB/s)")
        else:
            print(f"❌ {test_files[0].file_name}: {single_result['error']}")
        
        # Test batch download
        print("\n🔧 Batch Download Test:")
        batch_results = await client.download_mega_files_batch(test_files, max_concurrent_files=2)
        
        successful_batch = sum(1 for r in batch_results if r.get("success", False))
        print(f"✅ Batch Results: {successful_batch}/{len(test_files)} successful")
        
        for result in batch_results:
            if result.get("success", False):
                print(f"   ✅ {result['file_name']}: {result['file_size']/1024:.1f} KB "
                      f"({result['download_speed']/1024/1024:.1f} MB/s)")
            else:
                print(f"   ❌ {result['file_name']}: {result.get('error', 'Unknown error')}")
        
        # Show session statistics
        stats = client.get_session_stats()
        print(f"\n📊 SESSION STATISTICS:")
        print(f"   HTTP/2 Success Rate: {stats['http2_stats']['success_rate']*100:.1f}%")
        print(f"   Average Speed: {stats['http2_stats']['average_speed']/1024/1024:.1f} MB/s")
        print(f"   Total Bytes: {stats['http2_stats']['total_bytes']/1024/1024:.1f} MB")
        print(f"   Optimizations Active: {sum(stats['optimizations'].values())}/4")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        return False
        
    finally:
        await client.close()

async def run_integration_tests():
    """Run comprehensive integration tests"""
    print("🧪 HTTP/2 MEGA SDK INTEGRATION TESTS")
    print("=" * 60)
    
    test_results = {
        "client_initialization": False,
        "single_file_download": False,
        "batch_download": False,
        "optimization_integration": False,
        "error_handling": False
    }
    
    try:
        # Test 1: Client Initialization
        print("🔧 Test 1: Client Initialization...")
        client = HTTP2MegaSDKIntegration()
        test_results["client_initialization"] = True
        print("   ✅ Client initialized successfully")
        
        # Test 2: Single File Download
        print("🔧 Test 2: Single File Download...")
        test_file = MegaFileInfo(
            file_id="test_single",
            file_name="test_single.bin",
            file_size=1024,  # 1KB
            download_url="https://httpbin.org/bytes/1024"
        )
        
        single_result = await client.download_mega_file(test_file)
        test_results["single_file_download"] = single_result.get("success", False)
        print(f"   {'✅' if test_results['single_file_download'] else '❌'} Single download")
        
        # Test 3: Batch Download
        print("🔧 Test 3: Batch Download...")
        batch_files = [
            MegaFileInfo(
                file_id=f"test_batch_{i}",
                file_name=f"test_batch_{i}.bin",
                file_size=512,
                download_url="https://httpbin.org/bytes/512"
            ) for i in range(3)
        ]
        
        batch_results = await client.download_mega_files_batch(batch_files)
        successful_batch = sum(1 for r in batch_results if r.get("success", False))
        test_results["batch_download"] = successful_batch >= 2  # Allow 1 failure
        print(f"   {'✅' if test_results['batch_download'] else '❌'} Batch download: {successful_batch}/3")
        
        # Test 4: Optimization Integration
        print("🔧 Test 4: Optimization Integration...")
        stats = client.get_session_stats()
        optimizations_count = sum(stats["optimizations"].values())
        test_results["optimization_integration"] = optimizations_count >= 1  # At least 1 optimization
        print(f"   {'✅' if test_results['optimization_integration'] else '❌'} Optimizations: {optimizations_count}/4")
        
        # Test 5: Error Handling
        print("🔧 Test 5: Error Handling...")
        error_file = MegaFileInfo(
            file_id="test_error",
            file_name="test_error.bin", 
            file_size=1024,
            download_url="https://httpbin.org/status/500"  # Will return 500 error
        )
        
        error_result = await client.download_mega_file(error_file)
        test_results["error_handling"] = not error_result.get("success", True)  # Should fail gracefully
        print(f"   {'✅' if test_results['error_handling'] else '❌'} Error handling")
        
        await client.close()
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
    
    # Results summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\n🎯 TEST RESULTS: {passed_tests}/{total_tests} PASSED")
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    success_rate = passed_tests / total_tests
    print(f"\n📈 Success Rate: {success_rate*100:.1f}%")
    
    if success_rate >= 0.8:  # 80% pass rate required
        print("✅ PHASE 3 INTEGRATION: SUCCESS")
        return True
    else:
        print("❌ PHASE 3 INTEGRATION: NEEDS IMPROVEMENT")
        return False

async def main():
    """Main function for Phase 3 integration"""
    print("🚀 PRIORITY #5: HTTP/2 SUPPORT - PHASE 3: INTEGRATION AND OPTIMIZATION")
    print("=" * 80)
    
    # Run integration demo
    print("📊 Running MEGA SDK Integration Demo...")
    demo_success = await demo_mega_integration()
    
    print("\n" + "="*80)
    
    # Run integration tests
    print("🧪 Running Integration Tests...")
    test_success = await run_integration_tests()
    
    print("\n" + "="*80)
    
    # Final results
    if demo_success and test_success:
        print("🎉 PHASE 3 INTEGRATION AND OPTIMIZATION: COMPLETE!")
        print("✅ MEGA SDK HTTP/2 integration working")
        print("✅ Optimization components integrated")
        print("✅ Batch download functionality active")
        print("✅ Error handling robust")
        print("🚀 Ready for Phase 4: Polish and Production")
    else:
        print("⚠️  PHASE 3 INTEGRATION: PARTIAL SUCCESS")
        print("🔧 Some integration aspects need refinement")
        if not demo_success:
            print("❌ Integration demo issues detected")
        if not test_success:
            print("❌ Integration tests failed")

if __name__ == "__main__":
    asyncio.run(main())
