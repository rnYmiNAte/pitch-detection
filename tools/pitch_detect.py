#!/usr/bin/env python3
"""
tools/pitch_detect.py
Simple batch pitch detector for WAV and MP3 (A440 tuning).
Outputs a JSON array of results.
"""

import os
import sys
import json
import argparse
import tempfile
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import correlate

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def ensure_wav(path: Path) -> Path:
    """If path is mp3 (or not wav), convert to a WAV file in a temp dir and return Path to wav."""
    ext = path.suffix.lower()
    if ext == '.wav':
        return path
    if ext in ('.mp3', '.flac', '.m4a', '.ogg'):
        tmp = Path(tempfile.mkdtemp()) / (path.stem + '.wav')
        audio = AudioSegment.from_file(path.as_posix())
        audio.export(tmp.as_posix(), format='wav')
        return tmp
    raise RuntimeError(f"Unsupported audio extension: {ext}")

def read_mono(path: Path):
    data, sr = sf.read(path, dtype='float32')
    # If stereo, average channels -> mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr

def detect_f0_autocorr(x, sr, min_freq=50, max_freq=2000):
    """Estimate fundamental frequency by autocorrelation on a windowed signal."""
    # Use a center chunk if long
    N = len(x)
    if N < 2048:
        sig = x
    else:
        start = max(0, N // 2 - 44100 // 4)
        end = min(N, start + 44100 // 2)
        sig = x[start:end]

    # Pre-emphasis and window
    sig = sig - np.mean(sig)
    if np.all(np.abs(sig) < 1e-7):
        return None
    w = np.hanning(len(sig))
    sig = sig * w

    # Autocorrelation via FFT (fast)
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]  # keep non-negative lags
    corr /= np.max(np.abs(corr))  # normalize

    # Search lag range corresponding to min/max freq
    min_lag = int(sr / max_freq) if max_freq > 0 else 1
    max_lag = int(sr / min_freq) if min_freq > 0 else len(corr)-1
    if max_lag >= len(corr):
        max_lag = len(corr)-1
    if min_lag >= max_lag:
        return None

    # Find peak in the allowed range
    search = corr[min_lag:max_lag+1]
    peak = np.argmax(search) + min_lag
    # Parabolic interpolation for better resolution
    if 1 <= peak < len(corr)-1:
        y0, y1, y2 = corr[peak-1], corr[peak], corr[peak+1]
        p = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        peak = peak + p
    f0 = sr / peak if peak > 0 else None
    return f0

def freq_to_note_name(freq):
    """Return nearest MIDI note number, name, octave, and cents relative to A4=440"""
    if freq is None or freq <= 0:
        return None, None, None
    # MIDI note number (float)
    note_number = 69 + 12 * math.log2(freq / 440.0)
    nearest = int(round(note_number))
    cents = (note_number - nearest) * 100
    name = NOTE_NAMES[nearest % 12]
    octave = (nearest // 12) - 1
    return f"{name}{octave}", nearest, round(cents, 2)

def process_file(path: Path, min_freq, max_freq):
    wav = ensure_wav(path)
    try:
        x, sr = read_mono(wav)
    except Exception as e:
        return {"file": str(path), "error": f"read-error: {e}"}
    duration = len(x)/sr if sr > 0 else None
    f0 = detect_f0_autocorr(x, sr, min_freq, max_freq)
    note, midi, cents = freq_to_note_name(f0) if f0 else (None, None, None)
    return {
        "file": str(path),
        "detected_frequency_hz": round(float(f0), 2) if f0 else None,
        "note": note,
        "midi": midi,
        "cents": cents,
        "samplerate": int(sr),
        "duration_s": round(duration, 3) if duration else None
    }

def find_audio_files(folder: Path):
    exts = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
    for p in folder.rglob('*'):
        if p.suffix.lower() in exts and p.is_file():
            yield p

def main():
    parser = argparse.ArgumentParser(description="Batch pitch detection (A440)")
    parser.add_argument('--input-dir', type=str, default='audio', help='Directory to scan for audio')
    parser.add_argument('--output', type=str, default='results.json', help='Output JSON file')
    parser.add_argument('--min-freq', type=float, default=50.0)
    parser.add_argument('--max-freq', type=float, default=2000.0)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        print(f"Input directory {in_dir} not found.", file=sys.stderr)
        sys.exit(2)

    results = []
    for f in sorted(find_audio_files(in_dir)):
        print("Processing", f)
        try:
            r = process_file(f, args.min_freq, args.max_freq)
        except Exception as e:
            r = {"file": str(f), "error": f"exception: {repr(e)}"}
        results.append(r)

    outp = Path('results')
    outp.mkdir(parents=True, exist_ok=True)
    out_file = Path(args.output)
    with open(out_file, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print("Wrote results to", out_file)

if __name__ == '__main__':
    main()
