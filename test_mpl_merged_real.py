"""
Comprehensive Test Suite for mpl_merged.py - Real MEGA Operations
================================================================

This test suite performs comprehensive testing of all mpl_merged.py functionality
using real MEGA server connections and operations.

Features tested:
- Authentication (login, logout, session management)
- User information and account details
- Storage quota and space management
- File operations (upload, download, list, delete)
- Folder operations (create, navigate, list)
- Search functionality
- Sharing and public links
- Advanced features (encryption, metadata)

Requirements:
- Real MEGA account credentials in config/credentials.txt
- Internet connection for MEGA API access
- Test files for upload operations
"""

import os
import sys
import time
import tempfile
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_mpl_merged_real.log')
    ]
)
logger = logging.getLogger(__name__)

# Import the mpl_merged module
try:
    from mpl_merged import MPLClient, create_enhanced_client
    from mpl_merged import (
        validate_email, validate_password, get_error_message,
        base64_url_encode, base64_url_decode, generate_random_key,
        current_session, is_logged_in, get_current_user
    )
    print("✅ Successfully imported mpl_merged module")
except ImportError as e:
    print(f"❌ Failed to import mpl_merged: {e}")
    sys.exit(1)


class MPLRealTestSuite:
    """
    Comprehensive test suite for mpl_merged.py real operations.
    """
    
    def __init__(self):
        self.client = None
        self.temp_dir = None
        self.test_files = {}
        self.created_nodes = []  # Track created files/folders for cleanup
        self.email = None
        self.password = None
        
        # Test results tracking
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'errors': []
        }
        
        # Setup test environment
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup test environment and load credentials."""
        print("\n🏗️ Setting up test environment...")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix='mpl_test_')
        print(f"📁 Created temp directory: {self.temp_dir}")
        
        # Load credentials
        self.load_credentials()
        
        # Create test files
        self.create_test_files()
        
        print("✅ Test environment setup complete")
    
    def load_credentials(self):
        """Load MEGA credentials from file."""
        try:
            creds_path = Path(__file__).parent / "config" / "credentials.txt"
            if creds_path.exists():
                with open(creds_path, 'r') as f:
                    lines = f.read().strip().split('\n')
                    if len(lines) >= 2:
                        self.email = lines[0].strip()
                        self.password = lines[1].strip()
                        print(f"📧 Loaded credentials for: {self.email}")
                        return
            
            # Fallback: prompt for credentials
            print("⚠️ Credentials file not found, using test credentials")
            self.email = "test@example.com"
            self.password = "testpassword123"
            
        except Exception as e:
            print(f"❌ Failed to load credentials: {e}")
            raise
    
    def create_test_files(self):
        """Create test files for upload operations."""
        print("📄 Creating test files...")
        
        # Small text file
        small_file = Path(self.temp_dir) / "test_small.txt"
        with open(small_file, 'w') as f:
            f.write("This is a small test file for MPL testing.\n" * 10)
        self.test_files['small'] = small_file
        
        # Medium text file with JSON data
        medium_file = Path(self.temp_dir) / "test_medium.json"
        test_data = {
            'test': True,
            'timestamp': time.time(),
            'data': list(range(1000)),
            'description': 'Medium test file for MPL functionality testing'
        }
        with open(medium_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        self.test_files['medium'] = medium_file
        
        # Large text file
        large_file = Path(self.temp_dir) / "test_large.txt"
        with open(large_file, 'w') as f:
            for i in range(10000):
                f.write(f"Line {i}: This is a large test file for testing upload/download functionality.\n")
        self.test_files['large'] = large_file
        
        # Binary test file
        binary_file = Path(self.temp_dir) / "test_binary.dat"
        with open(binary_file, 'wb') as f:
            f.write(os.urandom(1024 * 50))  # 50KB of random data
        self.test_files['binary'] = binary_file
        
        print(f"✅ Created {len(self.test_files)} test files")
    
    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling and result tracking."""
        self.results['total_tests'] += 1
        print(f"\n🧪 Running test: {test_name}")
        
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                self.results['passed_tests'] += 1
                print(f"✅ PASSED: {test_name} ({duration:.2f}s)")
                return True
            else:
                self.results['failed_tests'] += 1
                print(f"❌ FAILED: {test_name} ({duration:.2f}s)")
                return False
                
        except Exception as e:
            self.results['failed_tests'] += 1
            error_msg = f"{test_name}: {str(e)}"
            self.results['errors'].append(error_msg)
            print(f"💥 ERROR: {test_name} - {e}")
            logger.exception(f"Test error in {test_name}")
            return False
    
    def test_client_initialization(self) -> bool:
        """Test MPL client initialization."""
        try:
            # Test basic client
            self.client = MPLClient()
            assert self.client is not None
            print("✅ Basic MPLClient initialized")
            
            # Test enhanced client
            enhanced_client = create_enhanced_client(
                max_requests_per_second=2.0,
                auto_login=False
            )
            assert enhanced_client is not None
            print("✅ Enhanced client initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Client initialization failed: {e}")
            return False
    
    def test_authentication(self) -> bool:
        """Test login/logout functionality."""
        try:
            print(f"🔐 Testing login with: {self.email}")
            
            # Test login
            result = self.client.login(self.email, self.password)
            if not result:
                print("❌ Login failed")
                return False
            
            print("✅ Login successful")
            
            # Verify logged in state
            assert is_logged_in() == True
            assert get_current_user() == self.email
            print("✅ Login state verified")
            
            # Test session info
            if hasattr(current_session, 'session_id') and current_session.session_id:
                print(f"✅ Session ID: {current_session.session_id[:20]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Authentication test failed: {e}")
            return False
    
    def test_user_info(self) -> bool:
        """Test user information retrieval."""
        try:
            # Get user info using correct method
            user_info = self.client.get_user_info()
            print(f"👤 User info: {user_info}")
            
            # Verify basic fields
            assert isinstance(user_info, dict)
            if 'email' in user_info:
                assert user_info['email'] == self.email
            
            return True
            
        except Exception as e:
            print(f"❌ User info test failed: {e}")
            return False
    
    def test_storage_quota(self) -> bool:
        """Test storage quota information."""
        try:
            # Get storage quota info using correct method
            storage_info = self.client.get_user_quota()
            print(f"💾 Storage info: {storage_info}")
            
            # Verify storage info structure
            assert isinstance(storage_info, dict)
            
            # Check for common fields
            expected_fields = ['total', 'used', 'available']
            for field in expected_fields:
                if field in storage_info:
                    value = storage_info[field]
                    assert isinstance(value, (int, float))
                    print(f"✅ {field}: {value:,} bytes")
            
            return True
            
        except Exception as e:
            print(f"❌ Storage quota test failed: {e}")
            return False
    
    def test_file_listing(self) -> bool:
        """Test file listing functionality."""
        try:
            # List root directory using correct method
            root_files = self.client.list()
            print(f"📁 Root directory has {len(root_files)} items")
            
            # Test detailed listing
            for item in root_files[:5]:  # Show first 5 items
                if hasattr(item, 'name'):
                    print(f"  📄 {item.name}")
                elif isinstance(item, dict):
                    print(f"  📄 {item.get('name', 'Unknown')}")
            
            # Test specific path listing if we have files
            if root_files:
                try:
                    files_in_root = self.client.list("/")
                    assert isinstance(files_in_root, list)
                    print("✅ Path-specific listing works")
                except:
                    print("⚠️ Path-specific listing not available")
            
            return True
            
        except Exception as e:
            print(f"❌ File listing test failed: {e}")
            return False
    
    def test_folder_operations(self) -> bool:
        """Test folder creation and navigation."""
        try:
            # Create test folder using correct method
            test_folder_name = f"MPL_Test_Folder_{int(time.time())}"
            print(f"📁 Creating folder: {test_folder_name}")
            
            folder_result = self.client.create_folder(test_folder_name)
            if folder_result:
                self.created_nodes.append(test_folder_name)
                print(f"✅ Folder created: {test_folder_name}")
                
                # Test listing to see if folder appears
                updated_list = self.client.list()
                folder_found = any(
                    (hasattr(item, 'name') and item.name == test_folder_name) or
                    (isinstance(item, dict) and item.get('name') == test_folder_name)
                    for item in updated_list
                )
                
                if folder_found:
                    print("✅ Folder appears in listing")
                else:
                    print("⚠️ Folder not found in listing")
                
                return True
            else:
                print("❌ Folder creation failed")
                return False
                
        except Exception as e:
            print(f"❌ Folder operations test failed: {e}")
            return False
    
    def test_file_upload(self) -> bool:
        """Test file upload functionality."""
        try:
            upload_results = []
            
            # Test uploading different file types
            for file_type, file_path in self.test_files.items():
                print(f"⬆️ Uploading {file_type} file: {file_path.name}")
                
                try:
                    # Upload to root directory
                    result = self.client.upload(str(file_path))
                    
                    if result:
                        upload_results.append(f"{file_type}: SUCCESS")
                        self.created_nodes.append(file_path.name)
                        print(f"✅ {file_type} file uploaded successfully")
                    else:
                        upload_results.append(f"{file_type}: FAILED")
                        print(f"❌ {file_type} file upload failed")
                
                except Exception as e:
                    upload_results.append(f"{file_type}: ERROR - {e}")
                    print(f"💥 {file_type} file upload error: {e}")
            
            # Summary
            successful_uploads = sum(1 for r in upload_results if "SUCCESS" in r)
            print(f"📊 Upload results: {successful_uploads}/{len(self.test_files)} successful")
            
            return successful_uploads > 0
            
        except Exception as e:
            print(f"❌ File upload test failed: {e}")
            return False
    
    def test_file_download(self) -> bool:
        """Test file download functionality."""
        try:
            # Get current file list using correct method
            files = self.client.list()
            
            if not files:
                print("⚠️ No files available for download test")
                return True
            
            # Try to download first available file
            download_target = None
            for item in files:
                if hasattr(item, 'name') and not item.name.endswith('/'):
                    download_target = item
                    break
                elif isinstance(item, dict) and item.get('name') and not item.get('name').endswith('/'):
                    download_target = item
                    break
            
            if not download_target:
                print("⚠️ No downloadable files found")
                return True
            
            # Get file name
            if hasattr(download_target, 'name'):
                filename = download_target.name
            else:
                filename = download_target.get('name', 'unknown_file')
            
            print(f"⬇️ Downloading file: {filename}")
            
            # Download to temp directory
            download_path = Path(self.temp_dir) / f"downloaded_{filename}"
            
            try:
                result = self.client.download(filename, str(download_path))
                
                if result and download_path.exists():
                    file_size = download_path.stat().st_size
                    print(f"✅ File downloaded successfully ({file_size:,} bytes)")
                    return True
                else:
                    print("❌ Download failed or file not created")
                    return False
                    
            except Exception as e:
                print(f"💥 Download error: {e}")
                return False
                
        except Exception as e:
            print(f"❌ File download test failed: {e}")
            return False
    
    def test_search_functionality(self) -> bool:
        """Test search functionality."""
        try:
            search_results = []
            
            # Test basic search
            try:
                results = self.client.search("test")
                search_results.append(f"Basic search: {len(results)} results")
                print(f"🔍 Basic search found {len(results)} results")
            except Exception as e:
                search_results.append(f"Basic search failed: {e}")
                print(f"❌ Basic search failed: {e}")
            
            # Test wildcard search
            try:
                results = self.client.search("*.txt")
                search_results.append(f"Wildcard search: {len(results)} results")
                print(f"🔍 Wildcard search found {len(results)} results")
            except Exception as e:
                search_results.append(f"Wildcard search failed: {e}")
                print(f"❌ Wildcard search failed: {e}")
            
            # Test empty search (should return all files)
            try:
                results = self.client.search("")
                search_results.append(f"Empty search: {len(results)} results")
                print(f"🔍 Empty search found {len(results)} results")
            except Exception as e:
                search_results.append(f"Empty search failed: {e}")
                print(f"❌ Empty search failed: {e}")
            
            print(f"📊 Search test summary:")
            for result in search_results:
                print(f"  {result}")
            
            # Consider test successful if at least one search worked
            return any("failed" not in result for result in search_results)
            
        except Exception as e:
            print(f"❌ Search functionality test failed: {e}")
            return False
    
    def test_advanced_features(self) -> bool:
        """Test advanced features and utilities."""
        try:
            # Test validation functions
            assert validate_email(self.email) == True
            assert validate_password(self.password) == True
            assert validate_email("invalid-email") == False
            assert validate_password("short") == False
            print("✅ Validation functions work")
            
            # Test cryptographic functions
            test_data = b"test data for encryption"
            random_key = generate_random_key()
            
            # Test base64 encoding/decoding
            encoded = base64_url_encode(test_data)
            decoded = base64_url_decode(encoded)
            assert decoded == test_data
            print("✅ Base64 encoding/decoding works")
            
            # Test error message function
            error_msg = get_error_message(-1)
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0
            print("✅ Error message function works")
            
            return True
            
        except Exception as e:
            print(f"❌ Advanced features test failed: {e}")
            return False
    
    def test_session_management(self) -> bool:
        """Test session management functionality."""
        try:
            # Check current session state
            if hasattr(current_session, 'is_authenticated'):
                assert current_session.is_authenticated == True
                print("✅ Session is authenticated")
            
            if hasattr(current_session, 'email'):
                assert current_session.email == self.email
                print("✅ Session email matches")
            
            # Test logout and re-login
            print("🔓 Testing logout...")
            self.client.logout()
            
            # Verify logged out state
            assert is_logged_in() == False
            print("✅ Logout successful")
            
            # Re-login
            print("🔐 Testing re-login...")
            result = self.client.login(self.email, self.password)
            assert result == True
            assert is_logged_in() == True
            print("✅ Re-login successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Session management test failed: {e}")
            return False
    
    def cleanup_created_nodes(self):
        """Clean up files and folders created during testing."""
        print("\n🧹 Cleaning up created test nodes...")
        
        cleanup_count = 0
        for node_name in self.created_nodes:
            try:
                # Try to delete the node
                if hasattr(self.client, 'delete'):
                    result = self.client.delete(node_name)
                    if result:
                        cleanup_count += 1
                        print(f"✅ Deleted: {node_name}")
                    else:
                        print(f"⚠️ Could not delete: {node_name}")
                else:
                    print(f"⚠️ Delete method not available for: {node_name}")
                    
            except Exception as e:
                print(f"❌ Error deleting {node_name}: {e}")
        
        print(f"🧹 Cleaned up {cleanup_count}/{len(self.created_nodes)} test nodes")
    
    def cleanup(self):
        """Clean up test environment."""
        print("\n🧹 Cleaning up test environment...")
        
        # Clean up created nodes
        if self.client and is_logged_in():
            self.cleanup_created_nodes()
        
        # Logout
        if self.client and is_logged_in():
            try:
                self.client.logout()
                print("✅ Logged out successfully")
            except Exception as e:
                print(f"⚠️ Logout error: {e}")
        
        # Clean up temp directory
        if self.temp_dir and Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(f"✅ Removed temp directory: {self.temp_dir}")
            except Exception as e:
                print(f"⚠️ Error removing temp directory: {e}")
    
    def print_results(self):
        """Print comprehensive test results."""
        print("\n" + "="*60)
        print("🧪 MPL MERGED REAL TEST RESULTS")
        print("="*60)
        
        print(f"📊 Total Tests: {self.results['total_tests']}")
        print(f"✅ Passed: {self.results['passed_tests']}")
        print(f"❌ Failed: {self.results['failed_tests']}")
        
        if self.results['passed_tests'] > 0:
            success_rate = (self.results['passed_tests'] / self.results['total_tests']) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.results['errors']:
            print(f"\n💥 ERRORS ({len(self.results['errors'])}):")
            for i, error in enumerate(self.results['errors'][:5], 1):
                print(f"  {i}. {error}")
            if len(self.results['errors']) > 5:
                print(f"  ... and {len(self.results['errors']) - 5} more errors")
        
        # Overall result
        if self.results['failed_tests'] == 0:
            print("\n🎉 ALL TESTS PASSED!")
        elif self.results['passed_tests'] > self.results['failed_tests']:
            print("\n✅ MOSTLY SUCCESSFUL")
        else:
            print("\n⚠️ MANY TESTS FAILED")
        
        print("="*60)
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("🚀 Starting MPL Merged Real Test Suite")
        print("="*60)
        
        test_sequence = [
            ("Client Initialization", self.test_client_initialization),
            ("Authentication", self.test_authentication),
            ("User Information", self.test_user_info),
            ("Storage Quota", self.test_storage_quota),
            ("File Listing", self.test_file_listing),
            ("Folder Operations", self.test_folder_operations),
            ("File Upload", self.test_file_upload),
            ("File Download", self.test_file_download),
            ("Search Functionality", self.test_search_functionality),
            ("Advanced Features", self.test_advanced_features),
            ("Session Management", self.test_session_management),
        ]
        
        for test_name, test_func in test_sequence:
            self.run_test(test_name, test_func)
            time.sleep(1)  # Brief pause between tests
        
        # Print final results
        self.print_results()
        
        # Cleanup
        self.cleanup()


def main():
    """Main test execution function."""
    print("🧪 MPL Merged Real Test Suite")
    print("Testing real MEGA operations with mpl_merged.py")
    print("="*60)
    
    # Create and run test suite
    test_suite = MPLRealTestSuite()
    
    try:
        test_suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n⚠️ Test suite interrupted by user")
        test_suite.cleanup()
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        logger.exception("Test suite error")
        test_suite.cleanup()
        raise


if __name__ == "__main__":
    main()
