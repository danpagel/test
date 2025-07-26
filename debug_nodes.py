#!/usr/bin/env python3

import sys
import os

# Add the current directory to the path so we can import mpl_merged
sys.path.insert(0, os.getcwd())

from mpl_merged import MPLClient

def test_node_types():
    """Test what types of objects are being returned by get_node_by_path"""
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
        
        # Test get_node_by_path
        print("\nTesting get_node_by_path('/')...")
        root_node = client.get_node_by_path("/")
        print(f"Root node type: {type(root_node)}")
        print(f"Root node: {root_node}")
        
        if root_node:
            print(f"Has is_folder method: {hasattr(root_node, 'is_folder')}")
            if hasattr(root_node, 'is_folder'):
                print(f"is_folder(): {root_node.is_folder()}")
            else:
                print("Node is missing is_folder method!")
                print(f"Node attributes: {dir(root_node)}")
        
        client.logout()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_node_types()
