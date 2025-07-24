"""
Dependencies and Auto-Installation Module
========================================

This module handles automatic installation of missing dependencies and imports
all required packages for the Mega.nz client.

This follows the exact methodology from the reference implementation:
1. Auto-install missing dependencies if needed
2. Import all standard library modules
3. Import all third-party modules

Author: Modernized from reference implementation
Date: July 2025
"""

# ==============================================
# === AUTO-INSTALL MISSING DEPENDENCIES ===
# ==============================================

import sys
import subprocess
from typing import Optional


def _install_and_import(package: str, pip_name: Optional[str] = None) -> None:
    """
    Automatically install missing packages and import them.
    
    Args:
        package: The package name to import
        pip_name: The pip package name (if different from import name)
    """
    try:
        __import__(package)
    except ImportError:
        pip_name = pip_name or package
        print(f"Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
        __import__(package)


# Install required packages if not available
for _pkg, _pip in [("Crypto", "pycryptodome"), ("requests", "requests")]:
    _install_and_import(_pkg, _pip)

# ==============================================
# === STANDARD LIBRARY IMPORTS ===
# ==============================================

import math
import re
import json
import logging
import secrets
from pathlib import Path
import hashlib
import time
import os
import random
import binascii
import tempfile
import shutil
from typing import Union, List, Tuple, Generator, Optional, Dict, Any, Callable

# ==============================================
# === THIRD-PARTY IMPORTS ===
# ==============================================

import requests
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Util import Counter
import base64
import struct
import codecs

# ==============================================
# === VERSION COMPATIBILITY ===
# ==============================================

# Python 2/3 compatibility functions for byte handling
if sys.version_info < (3,):
    def makebyte(x: str) -> str:
        return x
    
    def makestring(x: str) -> str:
        return x
else:
    def makebyte(x: str) -> bytes:
        return codecs.latin_1_encode(x)[0]
    
    def makestring(x: bytes) -> str:
        return codecs.latin_1_decode(x)[0]

# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Standard library
    'math', 're', 'json', 'logging', 'secrets', 'Path', 'hashlib', 'time',
    'os', 'random', 'binascii', 'tempfile', 'shutil', 'sys',
    
    # Third-party
    'requests', 'AES', 'RSA', 'Counter', 'base64', 'struct', 'codecs',
    
    # Compatibility functions
    'makebyte', 'makestring',
    
    # Type hints
    'Union', 'List', 'Tuple', 'Generator', 'Optional', 'Dict', 'Any'
]

# Configure logging
logger = logging.getLogger(__name__)
