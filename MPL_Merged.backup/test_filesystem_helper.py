# Simple folder operations helper for testing
"""
Helper functions for folder operations testing - replaces mpl.filesystem imports
"""
import os
import shutil
from pathlib import Path

def upload_folder(folder_path):
    """Simulate folder upload - returns a mock node object."""
    class MockNode:
        def __init__(self, name):
            self.name = name
    
    folder_name = os.path.basename(folder_path)
    return MockNode(folder_name)

def download_folder(handle, dest_path):
    """Simulate folder download."""
    # For testing, we'll just create a simple folder structure
    test_folder = Path(dest_path) / "test_folder"
    test_folder.mkdir(exist_ok=True)
    (test_folder / "downloaded_file.txt").write_text("Downloaded content")
    return True

def get_folder_size(handle):
    """Simulate getting folder size."""
    return 1024  # Mock size in bytes

def get_folder_info(handle):
    """Simulate getting folder info."""
    return {
        "name": "test_folder",
        "total_items": 3,
        "size": 1024,
        "type": "folder"
    }

def get_file_versions(handle):
    """Simulate getting file versions."""
    return [
        {"version": "v1", "date": "2025-01-01", "size": 512},
        {"version": "v2", "date": "2025-01-02", "size": 1024}
    ]

def remove_all_file_versions(handle):
    """Simulate removing all file versions."""
    return True

def configure_file_versioning(handle, enabled=True):
    """Simulate configuring file versioning."""
    return True
