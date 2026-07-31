"""
Video Keyframe Selection

Two jobs:
1. Parse an explicit timestamp out of the question ("what's happening at 0:12",
   "describe the frame at 5 seconds") - if present, we only need ONE frame,
   the one nearest that timestamp.
2. Otherwise, sample the video and pick out "main" frames: not just every Nth
   frame, but the ones that look most like distinct moments/scene changes,
   so a handful of frames actually represent the video instead of being an
   arbitrary uniform grid.

Full per-frame explainability (attention rollout + IG + cross-attention) is
expensive, so this module's whole job is deciding which small number of
frames are worth paying that cost for.
"""

import re
import cv2
import numpy as np
from PIL import Image

from config import FRAME_SAMPLING_RATE, MAX_FRAMES


def parse_requested_time(question):
    """
    Look for an explicit timestamp in the question and return it in seconds,
    or None if the question doesn't mention a time at all.

    Handles:
      "0:12", "1:05"                  -> minutes:seconds
      "12s", "12 sec", "12 seconds"   -> seconds
      "at 12"                          -> NOT matched (too ambiguous - "12" alone
                                          could be anything, e.g. "12 people")
    """
    q = question.lower()

    m = re.search(r'\b(\d+)\s*:\s*(\d{1,2})\b', q)
    if m:
        minutes, seconds = int(m.group(1)), int(m.group(2))
        return float(minutes * 60 + seconds)

    m = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b', q)
    if m:
        return float(m.group(1))

    return None


def extract_uniform_frames(video_path, sampling_rate=FRAME_SAMPLING_RATE, max_frames=MAX_FRAMES):
    """
    Sample roughly one frame every `1/sampling_rate` seconds (default: 1/sec),
    capped at max_frames so a long video can't blow up runtime.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(int(round(fps / sampling_rate)), 1)

    frames = []
    frame_index = 0
    sampled_index = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp = frame_index / fps
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append({
                "image": Image.fromarray(rgb),
                "timestamp": round(timestamp, 2),
                "frame_index": sampled_index,
            })
            sampled_index += 1

            if len(frames) >= max_frames:
                break

        frame_index += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")

    return frames


def detect_main_frames(frames, top_k=5):
    """
    Pick the frames that look most like distinct moments - ranked by how much
    each frame differs from the one before it (simple mean absolute pixel
    difference in grayscale). The very first frame is always kept to
    establish context; the rest are the biggest "jumps".
    """
    if len(frames) <= top_k:
        return frames

    diffs = [None]  # first frame has no predecessor
    prev_gray = np.array(frames[0]["image"].convert("L"), dtype=np.float32)

    for f in frames[1:]:
        gray = np.array(f["image"].convert("L"), dtype=np.float32)
        diff = float(np.abs(gray - prev_gray).mean()) if gray.shape == prev_gray.shape else 0.0
        diffs.append(diff)
        prev_gray = gray

    scored = list(zip(diffs[1:], frames[1:]))  # skip the first frame, it's auto-included
    scored.sort(key=lambda d: d[0], reverse=True)

    main_frames = [frames[0]] + [f for _, f in scored[: top_k - 1]]
    main_frames.sort(key=lambda f: f["timestamp"])  # chronological order for a sane timeline
    return main_frames


def nearest_frame_to_time(frames, target_seconds):
    """Find the sampled frame closest to an explicitly requested timestamp."""
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f["timestamp"] - target_seconds))


def select_frames_for_question(video_path, question, top_k_main=5):
    """
    Top-level entry point.

    Returns (frames, requested_time):
      - if the question mentions an explicit time: frames is a single-item
        list with the nearest sampled frame, requested_time is that time
      - otherwise: frames is the detected "main" frames, requested_time is None
    """
    all_frames = extract_uniform_frames(video_path)

    requested_time = parse_requested_time(question)
    if requested_time is not None:
        frame = nearest_frame_to_time(all_frames, requested_time)
        return ([frame] if frame else []), requested_time

    return detect_main_frames(all_frames, top_k=top_k_main), None
