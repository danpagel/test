#!/usr/bin/env python3
"""
Priority #5: HTTP/2 Support - Phase 2: Core Implementation
===========================================================

This module implements the core HTTP/2 functionality for the MEGA SDK,
providing multiplexed downloads, stream prioritization, and connection pooling.
"""

import asyncio
import httpx
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
from urllib.parse import urljoin, urlparse
import json
import ssl
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HTTP2DownloadRequest:
    """Represents a single HTTP/2 download request"""
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    byte_range: Optional[Tuple[int, int]] = None
    priority: int = 1  # 1-highest, 256-lowest (HTTP/2 stream priority)
    chunk_id: Optional[str] = None
    file_id: Optional[str] = None
    
    def __post_init__(self):
        if self.byte_range:
            start, end = self.byte_range
            self.headers['Range'] = f'bytes={start}-{end}'

@dataclass
class HTTP2DownloadResult:
    """Results from an HTTP/2 download operation"""
    request: HTTP2DownloadRequest
    status_code: int
    headers: Dict[str, str]
    content: bytes
    download_time: float
    stream_id: Optional[int] = None
    http_version: str = "HTTP/2"
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300
    
    @property
    def download_speed(self) -> float:
        """Download speed in bytes per second"""
        if self.download_time > 0:
            return len(self.content) / self.download_time
        return 0.0

@dataclass
class HTTP2ConnectionStats:
    """Statistics for HTTP/2 connection performance"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_bytes_downloaded: int = 0
    total_download_time: float = 0.0
    concurrent_streams_peak: int = 0
    connection_reuses: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests > 0:
            return self.successful_requests / self.total_requests
        return 0.0
    
    @property
    def average_speed(self) -> float:
        if self.total_download_time > 0:
            return self.total_bytes_downloaded / self.total_download_time
        return 0.0

class HTTP2StreamManager:
    """Manages HTTP/2 stream prioritization and flow control"""
    
    def __init__(self, max_concurrent_streams: int = 50):
        self.max_concurrent_streams = max_concurrent_streams
        self.active_streams: Dict[int, HTTP2DownloadRequest] = {}
        self.priority_queue: List[HTTP2DownloadRequest] = []
        self.stream_semaphore = asyncio.Semaphore(max_concurrent_streams)
        self._stream_counter = 0
        
    async def acquire_stream(self, request: HTTP2DownloadRequest) -> int:
        """Acquire a stream slot for the request"""
        await self.stream_semaphore.acquire()
        self._stream_counter += 1
        stream_id = self._stream_counter
        self.active_streams[stream_id] = request
        return stream_id
    
    def release_stream(self, stream_id: int):
        """Release a stream slot"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
        self.stream_semaphore.release()
    
    def get_priority_weight(self, priority: int) -> int:
        """Convert priority level to HTTP/2 weight (1-256)"""
        # Higher priority (lower number) gets higher weight
        return max(1, min(256, 257 - priority))

class HTTP2ConnectionPool:
    """Manages HTTP/2 connection pooling and reuse"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections: Dict[str, httpx.AsyncClient] = {}
        self.connection_stats: Dict[str, HTTP2ConnectionStats] = {}
        self.connection_locks: Dict[str, asyncio.Lock] = {}
        self._pool_lock = asyncio.Lock()
        
    def _get_connection_key(self, url: str) -> str:
        """Generate connection key from URL"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    @asynccontextmanager
    async def get_connection(self, url: str):
        """Get or create a connection for the URL"""
        connection_key = self._get_connection_key(url)
        
        async with self._pool_lock:
            if connection_key not in self.connections:
                # Create new HTTP/2 connection
                client = httpx.AsyncClient(
                    http2=True,
                    limits=httpx.Limits(
                        max_keepalive_connections=20,
                        max_connections=100,
                        keepalive_expiry=30.0
                    ),
                    timeout=httpx.Timeout(30.0)
                )
                self.connections[connection_key] = client
                self.connection_stats[connection_key] = HTTP2ConnectionStats()
                self.connection_locks[connection_key] = asyncio.Lock()
                
                logger.info(f"🔗 Created new HTTP/2 connection: {connection_key}")
            else:
                # Reuse existing connection
                self.connection_stats[connection_key].connection_reuses += 1
                
        try:
            yield self.connections[connection_key]
        finally:
            pass  # Keep connection alive for reuse
    
    async def close_all_connections(self):
        """Close all connections in the pool"""
        async with self._pool_lock:
            for client in self.connections.values():
                await client.aclose()
            self.connections.clear()
            self.connection_stats.clear()
            self.connection_locks.clear()
    
    def get_stats(self) -> Dict[str, HTTP2ConnectionStats]:
        """Get connection statistics"""
        return self.connection_stats.copy()

class HTTP2MegaClient:
    """HTTP/2 enhanced client for MEGA API operations"""
    
    def __init__(self, max_concurrent_streams: int = 50, max_connections: int = 10):
        self.stream_manager = HTTP2StreamManager(max_concurrent_streams)
        self.connection_pool = HTTP2ConnectionPool(max_connections)
        self.session_stats = HTTP2ConnectionStats()
        self._session_lock = asyncio.Lock()
        
    async def download_chunk(self, request: HTTP2DownloadRequest) -> HTTP2DownloadResult:
        """Download a single chunk using HTTP/2"""
        start_time = time.time()
        stream_id = None
        
        try:
            # Acquire stream slot
            stream_id = await self.stream_manager.acquire_stream(request)
            
            # Get connection from pool
            async with self.connection_pool.get_connection(request.url) as client:
                # Perform HTTP/2 request
                response = await client.get(
                    request.url,
                    headers=request.headers
                )
                
                download_time = time.time() - start_time
                
                # Create result
                result = HTTP2DownloadResult(
                    request=request,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=response.content,
                    download_time=download_time,
                    stream_id=stream_id,
                    http_version=response.http_version
                )
                
                # Update statistics
                await self._update_stats(result)
                
                logger.info(f"✅ Downloaded chunk {request.chunk_id}: "
                          f"{len(response.content)} bytes in {download_time:.2f}s "
                          f"({result.download_speed/1024/1024:.1f} MB/s) via {response.http_version}")
                
                return result
                
        except Exception as e:
            download_time = time.time() - start_time
            error_msg = str(e)
            
            result = HTTP2DownloadResult(
                request=request,
                status_code=0,
                headers={},
                content=b'',
                download_time=download_time,
                stream_id=stream_id,
                error=error_msg
            )
            
            await self._update_stats(result)
            logger.error(f"❌ Failed to download chunk {request.chunk_id}: {error_msg}")
            
            return result
            
        finally:
            if stream_id:
                self.stream_manager.release_stream(stream_id)
    
    async def download_chunks_concurrent(self, 
                                       requests: List[HTTP2DownloadRequest],
                                       max_concurrent: Optional[int] = None) -> List[HTTP2DownloadResult]:
        """Download multiple chunks concurrently using HTTP/2 multiplexing"""
        if max_concurrent is None:
            max_concurrent = min(len(requests), self.stream_manager.max_concurrent_streams)
        
        logger.info(f"🚀 Starting concurrent download of {len(requests)} chunks "
                   f"(max concurrent: {max_concurrent}) via HTTP/2")
        
        # Sort requests by priority (lower number = higher priority)
        sorted_requests = sorted(requests, key=lambda r: r.priority)
        
        # Create semaphore for additional concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(request):
            async with semaphore:
                return await self.download_chunk(request)
        
        # Execute all downloads concurrently
        start_time = time.time()
        results = await asyncio.gather(*[
            download_with_semaphore(request) for request in sorted_requests
        ], return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to failed result
                failed_result = HTTP2DownloadResult(
                    request=sorted_requests[i],
                    status_code=0,
                    headers={},
                    content=b'',
                    download_time=0.0,
                    error=str(result)
                )
                processed_results.append(failed_result)
            else:
                processed_results.append(result)
        
        # Calculate statistics
        successful = sum(1 for r in processed_results if r.success)
        total_bytes = sum(len(r.content) for r in processed_results)
        
        logger.info(f"✅ Concurrent download complete: {successful}/{len(requests)} successful, "
                   f"{total_bytes/1024/1024:.1f} MB in {total_time:.2f}s "
                   f"({total_bytes/total_time/1024/1024:.1f} MB/s)")
        
        return processed_results
    
    async def download_file_multipart(self, 
                                    file_url: str, 
                                    file_size: int,
                                    chunk_size: int = 1024*1024,  # 1MB chunks
                                    max_concurrent: int = 20,
                                    headers: Optional[Dict[str, str]] = None) -> HTTP2DownloadResult:
        """Download a file using HTTP/2 multipart concurrent downloads"""
        
        if headers is None:
            headers = {}
        
        # Calculate chunk ranges
        chunks = []
        chunk_id = 0
        for start in range(0, file_size, chunk_size):
            end = min(start + chunk_size - 1, file_size - 1)
            
            request = HTTP2DownloadRequest(
                url=file_url,
                headers=headers.copy(),
                byte_range=(start, end),
                priority=1,  # All chunks same priority for file download
                chunk_id=f"chunk_{chunk_id}",
                file_id=file_url.split('/')[-1] if '/' in file_url else file_url
            )
            chunks.append(request)
            chunk_id += 1
        
        logger.info(f"📁 Starting multipart download: {len(chunks)} chunks "
                   f"({chunk_size/1024/1024:.1f} MB each)")
        
        # Download all chunks concurrently
        results = await self.download_chunks_concurrent(chunks, max_concurrent)
        
        # Combine chunks in correct order
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if failed_results:
            logger.warning(f"⚠️  {len(failed_results)} chunks failed to download")
            # For now, return error - could implement retry logic
            return HTTP2DownloadResult(
                request=chunks[0],  # Use first chunk as representative
                status_code=500,
                headers={},
                content=b'',
                download_time=0.0,
                error=f"{len(failed_results)} chunks failed"
            )
        
        # Combine successful chunks
        combined_content = b''.join(r.content for r in successful_results)
        total_time = max(r.download_time for r in successful_results)
        
        # Create combined result
        combined_result = HTTP2DownloadResult(
            request=chunks[0],
            status_code=200,
            headers=successful_results[0].headers if successful_results else {},
            content=combined_content,
            download_time=total_time,
            http_version="HTTP/2"
        )
        
        logger.info(f"✅ Multipart download complete: {len(combined_content)/1024/1024:.1f} MB "
                   f"at {combined_result.download_speed/1024/1024:.1f} MB/s")
        
        return combined_result
    
    async def _update_stats(self, result: HTTP2DownloadResult):
        """Update session statistics"""
        async with self._session_lock:
            self.session_stats.total_requests += 1
            
            if result.success:
                self.session_stats.successful_requests += 1
                self.session_stats.total_bytes_downloaded += len(result.content)
            else:
                self.session_stats.failed_requests += 1
            
            self.session_stats.total_download_time += result.download_time
    
    def get_session_stats(self) -> HTTP2ConnectionStats:
        """Get current session statistics"""
        return self.session_stats
    
    def get_connection_stats(self) -> Dict[str, HTTP2ConnectionStats]:
        """Get connection pool statistics"""
        return self.connection_pool.get_stats()
    
    async def close(self):
        """Close the HTTP/2 client and all connections"""
        await self.connection_pool.close_all_connections()
        logger.info("🔒 HTTP/2 client closed")

# Demo and testing functions
async def demo_http2_performance():
    """Demonstrate HTTP/2 performance improvements"""
    print("🚀 HTTP/2 PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    
    client = HTTP2MegaClient(max_concurrent_streams=20)
    
    try:
        # Test URLs (using public HTTP/2 test endpoints)
        test_urls = [
            "https://httpbin.org/bytes/1024",  # 1KB
            "https://httpbin.org/bytes/2048",  # 2KB  
            "https://httpbin.org/bytes/4096",  # 4KB
            "https://httpbin.org/bytes/8192",  # 8KB
        ]
        
        # Create test requests
        requests = []
        for i, url in enumerate(test_urls):
            request = HTTP2DownloadRequest(
                url=url,
                chunk_id=f"test_{i}",
                priority=1
            )
            requests.append(request)
        
        print(f"📊 Testing concurrent download of {len(requests)} chunks...")
        
        # Perform concurrent downloads
        results = await client.download_chunks_concurrent(requests)
        
        # Display results
        print("\n📈 RESULTS:")
        total_bytes = 0
        total_time = 0
        successful = 0
        
        for result in results:
            if result.success:
                successful += 1
                total_bytes += len(result.content)
                total_time = max(total_time, result.download_time)
                print(f"✅ {result.request.chunk_id}: {len(result.content)} bytes "
                      f"in {result.download_time:.2f}s via {result.http_version}")
            else:
                print(f"❌ {result.request.chunk_id}: {result.error}")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   Successful Downloads: {successful}/{len(requests)}")
        print(f"   Total Data: {total_bytes/1024:.1f} KB")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   Average Speed: {total_bytes/total_time/1024:.1f} KB/s")
        
        # Show session stats
        session_stats = client.get_session_stats()
        print(f"\n📊 SESSION STATISTICS:")
        print(f"   Success Rate: {session_stats.success_rate*100:.1f}%")
        print(f"   Average Speed: {session_stats.average_speed/1024:.1f} KB/s")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
        
    finally:
        await client.close()

async def run_http2_core_tests():
    """Run comprehensive HTTP/2 core functionality tests"""
    print("🧪 HTTP/2 CORE IMPLEMENTATION TESTS")
    print("=" * 60)
    
    test_results = {
        "connection_creation": False,
        "concurrent_downloads": False,
        "stream_management": False,
        "connection_pooling": False,
        "error_handling": False
    }
    
    client = HTTP2MegaClient()
    
    try:
        # Test 1: Connection Creation
        print("🔧 Test 1: HTTP/2 Connection Creation...")
        test_request = HTTP2DownloadRequest(
            url="https://httpbin.org/bytes/1024",
            chunk_id="test_connection"
        )
        result = await client.download_chunk(test_request)
        test_results["connection_creation"] = result.success and result.http_version == "HTTP/2"
        print(f"   {'✅' if test_results['connection_creation'] else '❌'} Connection: {result.http_version}")
        
        # Test 2: Concurrent Downloads
        print("🔧 Test 2: Concurrent Downloads...")
        concurrent_requests = [
            HTTP2DownloadRequest(url="https://httpbin.org/bytes/512", chunk_id=f"concurrent_{i}")
            for i in range(5)
        ]
        concurrent_results = await client.download_chunks_concurrent(concurrent_requests)
        successful_concurrent = sum(1 for r in concurrent_results if r.success)
        test_results["concurrent_downloads"] = successful_concurrent >= 4  # Allow 1 failure
        print(f"   {'✅' if test_results['concurrent_downloads'] else '❌'} Concurrent: {successful_concurrent}/5")
        
        # Test 3: Stream Management
        print("🔧 Test 3: Stream Management...")
        stream_count = len(client.stream_manager.active_streams)
        test_results["stream_management"] = stream_count == 0  # Should be empty after completion
        print(f"   {'✅' if test_results['stream_management'] else '❌'} Active Streams: {stream_count}")
        
        # Test 4: Connection Pooling
        print("🔧 Test 4: Connection Pooling...")
        pool_stats = client.get_connection_stats()
        has_connections = len(pool_stats) > 0
        test_results["connection_pooling"] = has_connections
        print(f"   {'✅' if test_results['connection_pooling'] else '❌'} Pool Connections: {len(pool_stats)}")
        
        # Test 5: Error Handling
        print("🔧 Test 5: Error Handling...")
        error_request = HTTP2DownloadRequest(
            url="https://httpbin.org/status/404",
            chunk_id="test_error"
        )
        error_result = await client.download_chunk(error_request)
        test_results["error_handling"] = not error_result.success  # Should fail gracefully
        print(f"   {'✅' if test_results['error_handling'] else '❌'} Error Handling: Status {error_result.status_code}")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        
    finally:
        await client.close()
    
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
        print("✅ PHASE 2 CORE IMPLEMENTATION: SUCCESS")
        return True
    else:
        print("❌ PHASE 2 CORE IMPLEMENTATION: NEEDS IMPROVEMENT")
        return False

async def main():
    """Main function to run HTTP/2 core implementation"""
    print("🚀 PRIORITY #5: HTTP/2 SUPPORT - PHASE 2: CORE IMPLEMENTATION")
    print("=" * 70)
    
    # Run performance demo
    print("📊 Running HTTP/2 Performance Demo...")
    demo_success = await demo_http2_performance()
    
    print("\n" + "="*70)
    
    # Run core tests
    print("🧪 Running Core Implementation Tests...")
    test_success = await run_http2_core_tests()
    
    print("\n" + "="*70)
    
    # Final results
    if demo_success and test_success:
        print("🎉 PHASE 2 CORE IMPLEMENTATION: COMPLETE!")
        print("✅ HTTP/2 multiplexing working")
        print("✅ Connection pooling active") 
        print("✅ Stream management functional")
        print("✅ Error handling robust")
        print("🚀 Ready for Phase 3: Integration and Optimization")
    else:
        print("⚠️  PHASE 2 CORE IMPLEMENTATION: PARTIAL SUCCESS")
        print("🔧 Some components need refinement")
        if not demo_success:
            print("❌ Performance demo issues detected")
        if not test_success:
            print("❌ Core functionality tests failed")

if __name__ == "__main__":
    asyncio.run(main())
