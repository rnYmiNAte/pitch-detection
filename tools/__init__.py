"""
Tools package for Pitch Detection (A440).

This package contains the pitch detection utilities used by the
GitHub Actions workflow. The main entry point is `pitch_detect.py`.

Modules:
    pitch_detect – Batch pitch detection for WAV/MP3 audio files.

Usage:
    >>> from tools.pitch_detect import process_file

This file intentionally exports only the key detection function.
"""

from .pitch_detect import process_file
