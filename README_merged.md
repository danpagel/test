# MPL Merged - Single File MEGA.nz Client

This is a merged version of the MegaPythonLibrary (MPL) that combines all 24 modules into a single working Python file while maintaining full functionality and API compatibility.

## Features

✅ **Complete MEGA.nz client functionality**
- Authentication (login, logout, registration, password management)
- File operations (upload, download, delete, move, rename)
- Folder management (create, list, navigate)
- Advanced search and file discovery
- Real-time event system with callbacks
- Cryptographic utilities and security functions
- Network optimization and rate limiting

✅ **Full API compatibility**
- Same interface as the original package
- Drop-in replacement for the entire mpl package
- All 43 exported functions and classes available

✅ **Self-contained and portable**
- Single file: `mpl_merged.py` (3,705 lines)
- Auto-installs missing dependencies (pycryptodome, requests)
- No additional setup required

## Quick Start

```python
# Import the merged client
from mpl_merged import MPLClient

# Create client instance
client = MPLClient()

# Login to MEGA
client.login("your_email@example.com", "your_password")

# Upload a file
uploaded_file = client.upload("local_file.txt", "/")

# List files in root folder
files = client.list("/")
for file in files:
    print(f"{file.name} - {file.get_size_formatted()}")

# Download a file
client.download("/remote_file.txt", "downloaded_file.txt")

# Create a folder
new_folder = client.create_folder("My Folder", "/")

# Search for files
results = client.search("*.txt")

# Logout
client.logout()
```

## Enhanced Usage

```python
# Create enhanced client with rate limiting
from mpl_merged import create_enhanced_client

client = create_enhanced_client(
    max_requests_per_second=10.0,
    auto_login=True
)

# Event handling
def on_upload_progress(data):
    print(f"Upload progress: {data['percentage']:.1f}%")

client.on('upload_progress', on_upload_progress)
```

## Available Classes and Functions

### Main Classes
- `MPLClient` - Main client class
- `MegaNode` - Represents files and folders
- `EventManager` - Event system manager

### Authentication Functions
- `login()`, `logout()`, `register()`, `verify_email()`
- `is_logged_in()`, `get_current_user()`, `get_user_info()`

### Filesystem Functions
- `upload_file()`, `download_file()`, `delete_node()`
- `create_folder()`, `move_node()`, `rename_node()`
- `list_folder()`, `get_node_by_path()`, `search_nodes_by_name()`

### Utility Functions
- `validate_email()`, `validate_password()`, `format_size()`
- `is_image_file()`, `is_video_file()`, `is_audio_file()`

### Cryptographic Functions
- `aes_cbc_encrypt()`, `aes_cbc_decrypt()`, `generate_random_key()`
- `base64_url_encode()`, `base64_url_decode()`, `hash_password()`

### Convenience Functions
- `create_client()`, `create_enhanced_client()`, `get_version_info()`

## File Statistics

- **Lines:** 3,705
- **Characters:** 110,329
- **Modules merged:** 24
- **Functions/classes:** 43 exports
- **Dependencies:** 2 (pycryptodome, requests)

## Compatibility

This merged file is a drop-in replacement for the original MPL package:

```python
# Original package usage:
# from mpl import MPLClient

# Merged file usage (same interface):
from mpl_merged import MPLClient
```

All functionality from the original 24 modules is preserved and available through the same API.

## Requirements

- Python 3.7+
- pycryptodome (auto-installed)
- requests (auto-installed)

## License

MIT License - Same as original MegaPythonLibrary package.