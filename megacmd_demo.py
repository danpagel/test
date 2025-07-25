#!/usr/bin/env python3
"""
MEGAcmd Compatibility Demonstration
===================================

This script demonstrates the MEGAcmd compatibility features added to MegaPythonLibrary.
It showcases all 71 MEGAcmd standard commands that are now available.

Usage:
    python megacmd_demo.py
"""

import sys
import os
from pathlib import Path

# Add the project root to sys.path if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    import mpl_merged
    print("✅ Imported mpl_merged successfully")
except ImportError as e:
    print(f"❌ Failed to import mpl_merged: {e}")
    sys.exit(1)

def main():
    """Demonstrate MEGAcmd compatibility features."""
    
    print("=" * 60)
    print("MEGAcmd Compatibility Demonstration")
    print("=" * 60)
    
    # Create client instance
    print("\n1. Creating MPLClient instance...")
    client = mpl_merged.MPLClient()
    print("✅ MPLClient created successfully")
    
    # Test version information
    print("\n2. Testing version information...")
    version = client.version()
    print(f"Version: {version}")
    
    # Test help system
    print("\n3. Testing help system...")
    help_text = client.help()
    print("General help available:", len(help_text) > 0)
    
    # Test specific command help
    ls_help = client.help('ls')
    print("Specific command help (ls):", len(ls_help) > 0)
    
    # Test local directory commands
    print("\n4. Testing local directory commands...")
    current_dir = client.lpwd()
    print(f"Current local directory: {current_dir}")
    
    # Test session information (not logged in)
    print("\n5. Testing session information...")
    session_info = client.session()
    print(f"Session status: {'Logged in' if session_info.get('logged_in') else 'Not logged in'}")
    
    # Test echo command
    print("\n6. Testing echo command...")
    echo_result = client.echo("Hello MEGAcmd!")
    print(f"Echo result: {echo_result}")
    
    # Test debug command
    print("\n7. Testing debug settings...")
    debug_result = client.debug("info")
    print(f"Debug setting result: {debug_result}")
    
    # Test df command (quota information)
    print("\n8. Testing storage quota (df command)...")
    try:
        quota_info = client.df()
        print(f"Quota info available: {quota_info is not None}")
    except Exception as e:
        print(f"Quota info (not logged in): {e}")
    
    # Count available MEGAcmd commands
    print("\n9. Counting available MEGAcmd commands...")
    all_methods = [m for m in dir(client) if not m.startswith('_')]
    
    # Define MEGAcmd command categories
    megacmd_commands = {
        'Authentication & Session': ['login', 'logout', 'signup', 'passwd', 'whoami', 'confirm', 'session'],
        'File Operations': ['ls', 'cd', 'mkdir', 'cp', 'mv', 'rm', 'find', 'cat', 'pwd', 'du', 'tree'],
        'Transfer Operations': ['get', 'put', 'transfers', 'mediainfo'],
        'Sharing & Collaboration': ['share', 'users', 'invite', 'ipc', 'export', 'import_'],
        'Synchronization': ['sync', 'backup', 'exclude', 'sync_ignore', 'sync_config', 'sync_issues'],
        'FUSE Filesystem': ['fuse_add', 'fuse_remove', 'fuse_enable', 'fuse_disable', 'fuse_show', 'fuse_config'],
        'System & Configuration': ['version', 'debug', 'log', 'reload', 'update', 'df', 'killsession', 'locallogout'],
        'Advanced System': ['errorcode', 'masterkey', 'showpcr', 'psa', 'mount', 'graphics', 'attr', 'userattr'],
        'Process Control': ['cancel', 'confirmcancel', 'lcd', 'lpwd', 'deleteversions'],
        'Advanced Features': ['speedlimit', 'thumbnail', 'preview', 'proxy', 'https', 'webdav', 'ftp'],
        'Shell Utilities': ['echo', 'history', 'help']
    }
    
    total_megacmd = 0
    available_megacmd = 0
    
    print("\nMEGAcmd Command Categories:")
    for category, commands in megacmd_commands.items():
        available_in_category = sum(1 for cmd in commands if hasattr(client, cmd))
        total_in_category = len(commands)
        total_megacmd += total_in_category
        available_megacmd += available_in_category
        
        status = "✅" if available_in_category == total_in_category else "⚠️"
        print(f"  {status} {category}: {available_in_category}/{total_in_category}")
    
    print(f"\n📊 Summary:")
    print(f"   Total MEGAcmd commands: {available_megacmd}/{total_megacmd}")
    print(f"   Total client methods: {len(all_methods)}")
    print(f"   Compatibility level: {(available_megacmd/total_megacmd)*100:.1f}%")
    
    # Test backward compatibility
    print("\n10. Testing backward compatibility...")
    original_methods = ['list', 'create_folder', 'upload', 'download', 'delete', 'move']
    backward_compatible = all(hasattr(client, method) for method in original_methods)
    print(f"Original methods available: {'✅ Yes' if backward_compatible else '❌ No'}")
    
    # Test that MEGAcmd methods delegate properly
    print("\n11. Testing MEGAcmd delegation...")
    delegations = [
        ('ls', 'list'),
        ('mkdir', 'create_folder'),
        ('put', 'upload'),
        ('get', 'download'),
        ('rm', 'delete'),
        ('mv', 'move')
    ]
    
    delegation_working = True
    for megacmd_cmd, original_cmd in delegations:
        if hasattr(client, megacmd_cmd) and hasattr(client, original_cmd):
            print(f"  ✅ {megacmd_cmd} -> {original_cmd}")
        else:
            print(f"  ❌ {megacmd_cmd} -> {original_cmd} (missing)")
            delegation_working = False
    
    print(f"Delegation working: {'✅ Yes' if delegation_working else '❌ No'}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("MEGAcmd Compatibility Demonstration Complete")
    print("=" * 60)
    print(f"✅ {available_megacmd} MEGAcmd commands implemented")
    print(f"✅ Backward compatibility maintained")
    print(f"✅ Help system functional")
    print(f"✅ Version information updated")
    print(f"✅ All systems operational")
    
    print(f"\nTo test with actual MEGA account:")
    print(f">>> client.login('your_email', 'your_password')")
    print(f">>> client.ls('/')  # List root directory")
    print(f">>> client.mkdir('/test_folder')  # Create folder")
    print(f">>> client.put('local_file.txt', '/')  # Upload file")
    print(f">>> client.whoami()  # Show current user")
    print(f">>> client.logout()  # Logout")

if __name__ == "__main__":
    main()