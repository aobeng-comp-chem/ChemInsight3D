"""Cross-platform path utilities for Windows, Linux and WSL.

Provides small helpers to normalise and remember last-used directories
so the GUI can open file dialogs against paths accessible from the
current runtime (native Linux/WSL or Windows).
"""
from __future__ import annotations

import os
import sys
import platform
import re

def _running_under_wsl() -> bool:
    if sys.platform != 'linux':
        return False
    try:
        with open('/proc/version', 'r', encoding='utf8') as f:
            return 'microsoft' in f.read().lower()
    except Exception:
        return False

def _is_windows() -> bool:
    return sys.platform.startswith('win') or os.name == 'nt'


def _windows_to_wsl_path(win_path: str) -> str:
    # C:\Users\Name -> /mnt/c/Users/Name
    m = re.match(r'^([A-Za-z]):\\(.*)', win_path)
    if not m:
        # UNC or other forms: replace backslashes and return as-is
        return win_path.replace('\\', '/')
    drive = m.group(1).lower()
    rest = m.group(2).replace('\\', '/')
    return f"/mnt/{drive}/{rest}"


def _wsl_to_windows_path(wsl_path: str) -> str:
    # /mnt/c/Users/Name -> C:\Users\Name
    m = re.match(r'^/mnt/([a-zA-Z])/(.*)', wsl_path)
    if m:
        drive = m.group(1).upper()
        rest = m.group(2).replace('/', '\\')
        return f"{drive}:\\{rest}"
    # Not a /mnt path — return with forward slashes converted
    return wsl_path.replace('/', '\\')


# last-used directory remembered across dialogs
_last_dir: str | None = None


def _initial_last_dir() -> str:
    # Prefer matching behaviour that existed in the repo: if running
    # inside WSL try to return the Windows user home mounted under /mnt,
    # else fall back to ~.
    try:
        if _running_under_wsl():
            up = os.environ.get('USERPROFILE', '')
            if up:
                m = re.match(r'([A-Za-z]):[/\\]+(.*)', up)
                if m:
                    path = '/mnt/' + m.group(1).lower() + '/' + m.group(2).replace('\\', '/').replace('\\', '/')
                    if os.path.isdir(path):
                        return path
            if os.path.isdir('/mnt/c/Users'):
                return '/mnt/c/Users'
        return os.path.expanduser('~')
    except Exception:
        return os.path.expanduser('~')


def default_dir() -> str:
    global _last_dir
    if _last_dir is None:
        _last_dir = _initial_last_dir()
    return _last_dir


def remember_dir(path: str) -> None:
    """Update the remembered last directory.

    Accepts Windows, WSL or POSIX paths; stores a path usable in the
    current runtime (i.e. converted to /mnt form on WSL, drive path on
    Windows).
    """
    global _last_dir
    if not path:
        return
    p = path
    p = os.path.expanduser(p)
    p = os.path.expandvars(p)
    # If running in WSL convert common Windows drive paths to /mnt
    if _running_under_wsl():
        if re.match(r'^[A-Za-z]:\\', p):
            p = _windows_to_wsl_path(p)
    elif _is_windows():
        # If running on Windows and a /mnt style path was passed, convert
        if p.startswith('/mnt/'):
            p = _wsl_to_windows_path(p)
    # ensure we store a directory path
    if os.path.isdir(p):
        _last_dir = p
    else:
        _last_dir = os.path.dirname(p) or _last_dir


def normalize_path_for_runtime(path: str) -> str:
    """Return a path string that will be accessible from the current runtime.

    - On WSL: converts Windows-style drive paths (C:\...) to `/mnt/c/...`.
    - On Windows: converts `/mnt/c/...` back to `C:\...`.
    - Expands `~` and environment variables.
    """
    if not path:
        return path
    p = os.path.expanduser(path)
    p = os.path.expandvars(p)
    if _running_under_wsl():
        # translate Windows drive paths to mnt
        if re.match(r'^[A-Za-z]:\\', p):
            return _windows_to_wsl_path(p)
        # leave other paths as-is
        return p
    if _is_windows():
        if p.startswith('/mnt/'):
            return _wsl_to_windows_path(p)
        return p
    # native POSIX (not WSL)
    return p
