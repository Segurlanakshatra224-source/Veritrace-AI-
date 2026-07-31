"""
Audio Preprocessing Module

This module prepares audio inputs before they are passed to
the multimodal inference model.

Workflow:
1. Validate input
2. Load audio
3. Convert to mono
4. Resample audio
5. Normalize
6. Extract features
7. Return processed data
"""
# Import required libraries
import os
import librosa
import numpy as np


def preprocess_audio(file_path, question=None, processor=None):
    """
    Preprocess an audio file for Qwen2-Audio.

    Steps:
    1. Check if file exists
    2. Load audio
    3. Convert to mono
    4. Resample to 16 kHz
    5. Normalize audio
    6. Calculate duration
    7. Return waveform and metadata

    Parameters:
        file_path (str): Path to the audio file
        question (str): reserved for the future audio inference pipeline, unused for now
        processor: reserved for the future audio inference pipeline, unused for now

    Returns:
        dict: Processed audio waveform and metadata

    Note: audio is out of scope for this build cycle. main.py.process_audio() currently
    expects "spectrogram_image"/"time_per_column" keys that this function does not
    produce yet - that's a real gap to close when audio support is picked back up,
    not something patched here.
    """

    # Check whether the audio file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    # Load audio file
    # sr=16000 ensures audio is resampled to 16 kHz
    # mono=True converts stereo audio into a single channel
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    # Normalize audio amplitude
    # This keeps loud and soft recordings in a consistent range
    audio = librosa.util.normalize(audio)

    # Calculate audio duration in seconds
    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate
    )

    # Return processed waveform and metadata
    # Qwen2-Audio processor will handle feature extraction later
    return {
        "audio": audio,
        "sample_rate": sample_rate,
        "duration": duration
    }