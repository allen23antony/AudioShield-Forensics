"""Forensic file-integrity hashing utilities.

Computes MD5 and SHA-256 digests from raw file bytes, reading in chunks so
that large audio files do not need to be loaded entirely into memory.

SHA-256 is calculated independently from the actual uploaded file bytes and
is never obtained from embedded metadata.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


_CHUNK_SIZE = 65_536  # 64 KB chunks


def compute_md5(file_path: Path) -> str:
    """Return the hex-encoded MD5 digest of *file_path*.

    Parameters
    ----------
    file_path:
        Path to the file to hash.

    Returns
    -------
    str
        Lower-case 32-character hex string.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    OSError
        If the file cannot be read.
    """
    h = hashlib.md5()
    with Path(file_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256(file_path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of *file_path*.

    The digest is computed directly from the raw file bytes; it is never
    obtained from embedded metadata fields.

    Parameters
    ----------
    file_path:
        Path to the file to hash.

    Returns
    -------
    str
        Lower-case 64-character hex string.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    OSError
        If the file cannot be read.
    """
    h = hashlib.sha256()
    with Path(file_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_hashes(file_path: Path) -> Dict[str, str]:
    """Return MD5 and SHA-256 digests for *file_path* in a single pass pair.

    Parameters
    ----------
    file_path:
        Path to the file to hash.

    Returns
    -------
    dict
        ``{"md5": "<32-hex>", "sha256": "<64-hex>", "algorithm": "MD5 + SHA-256"}``

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    OSError
        If the file cannot be read.
    """
    md5_h = hashlib.md5()
    sha_h = hashlib.sha256()
    with Path(file_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            md5_h.update(chunk)
            sha_h.update(chunk)
    return {
        "md5": md5_h.hexdigest(),
        "sha256": sha_h.hexdigest(),
        "algorithm": "MD5 + SHA-256",
    }
