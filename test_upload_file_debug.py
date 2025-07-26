#!/usr/bin/env python3

import sys
import os
import tempfile

# Add the current directory to the path so we can import mpl_merged
sys.path.insert(0, os.getcwd())

from mpl_merged import MPLClient

def test_upload_file_detailed():
    """Test upload_file functionality in detail"""
    client = MPLClient()
    
    try:
        # Load credentials
        with open('config/credentials.txt', 'r') as f:
            lines = f.read().strip().split('\n')
            email = lines[0]
            password = lines[1]
        
        print(f"Logging in as {email}...")
        result = client.login(email, password)
        print(f"Login result: {result}")
        
        # Create a test file
        test_content = "Test file for upload_file debugging"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            test_file_path = f.name
        
        print(f"\nCreated test file: {test_file_path}")
        
        # Test upload_file
        print("Testing upload_file...")
        try:
            uploaded_node = client.upload_file(test_file_path, "/")
            print(f"Upload_file result: {uploaded_node}")
            print(f"Upload_file result type: {type(uploaded_node)}")
            
            if uploaded_node:
                print("✅ upload_file successful!")
                if hasattr(uploaded_node, 'name'):
                    print(f"Uploaded file name: {uploaded_node.name}")
                    
                    # Test verification separately
                    print(f"\nTesting cloud verification...")
                    file_path = f"/{uploaded_node.name}"
                    
                    # Test get_node_by_path
                    print(f"Testing get_node_by_path('{file_path}')...")
                    node = client.get_node_by_path(file_path)
                    print(f"Node found: {node}")
                    print(f"Node type: {type(node)}")
                    
                    # Test download/get
                    print(f"Testing download...")
                    download_dir = tempfile.mkdtemp()
                    local_name = os.path.basename(file_path)
                    download_path = os.path.join(download_dir, local_name)
                    
                    try:
                        # Download returns bool, not path
                        download_success = client.get(file_path, download_path)
                        print(f"Download result: {download_success}")
                        if download_success and os.path.exists(download_path):
                            with open(download_path, 'r', encoding='utf-8') as f:
                                downloaded_content = f.read().strip()
                            print(f"Downloaded content: '{downloaded_content}'")
                            print(f"Original content: '{test_content}'")
                            print(f"Content matches: {downloaded_content == test_content.strip()}")
                        else:
                            print("❌ Download failed or file doesn't exist")
                            
                    except Exception as e:
                        print(f"❌ Download failed with exception: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        # Clean up
                        try:
                            if os.path.exists(download_path):
                                os.unlink(download_path)
                            os.rmdir(download_dir)
                        except Exception:
                            pass
                    
            else:
                print("❌ upload_file failed - returned None/False")
                
        except Exception as e:
            print(f"❌ upload_file failed with exception: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up
        os.unlink(test_file_path)
        client.logout()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_upload_file_detailed()
