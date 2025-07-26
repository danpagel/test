#!/usr/bin/env python3

import sys
import os
import tempfile

# Add the current directory to the path so we can import mpl_merged
sys.path.insert(0, os.getcwd())

from mpl_merged import MPLClient

def test_upload():
    """Test upload functionality specifically"""
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
        test_content = "Test file for upload"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            test_file_path = f.name
        
        print(f"\nCreated test file: {test_file_path}")
        
        # Test upload
        print("Attempting upload...")
        try:
            upload_result = client.upload(test_file_path, "/")
            print(f"Upload result: {upload_result}")
            print(f"Upload result type: {type(upload_result)}")
            
            if upload_result:
                print("✅ Upload successful!")
                if hasattr(upload_result, 'name'):
                    print(f"Uploaded file name: {upload_result.name}")
                if hasattr(upload_result, 'handle'):
                    print(f"File handle: {upload_result.handle}")
            else:
                print("❌ Upload failed - returned None/False")
                
        except Exception as e:
            print(f"❌ Upload failed with exception: {e}")
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
    test_upload()
