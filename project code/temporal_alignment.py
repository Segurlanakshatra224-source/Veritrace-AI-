import numpy as np


def audio_columns_to_time(attention_map, time_per_column):
    """
    Convert an audio spectrogram attention/attribution map into (start_time, end_time,
    score) segments, sorted by importance, so a report can say "the model focused on
    2.1s-3.4s" instead of just showing a picture.
    """

    if attention_map is None:
        return []

    column_scores = attention_map.mean(axis=0) #average down the frequency axis, keep the time axis
    segments = []

    for col_index, score in enumerate(column_scores):
        start_time = col_index * time_per_column
        end_time = start_time + time_per_column

        segments.append({
            "start_time": round(float(start_time), 2),
            "end_time": round(float(end_time), 2),
            "score": float(score),
        })

    segments.sort(key=lambda s: s["score"], reverse=True)
    return segments


def frame_importance_scores(frame_results):
    """
    Given a list of per-keyframe results (each with a "timestamp" and an
    "attention_map"), reduce every frame's heatmap to a single importance number so
    the frames can be ranked and plotted over time.
    """

    scores = []

    for frame in frame_results:
        attention_map = frame.get("attention_map")
        if attention_map is None:
            continue

        arr = np.asarray(attention_map)
        scores.append({
            "timestamp": frame["timestamp"],
            "frame_index": frame["frame_index"],
            "importance": float(arr.mean()),
            "peak": float(arr.max()),
        })

    return scores


def build_video_timeline(frame_results):
    """
    Full timeline for a video: importance per keyframe, plus which moment the model
    seemed to care about most.
    """

    scores = frame_importance_scores(frame_results)
    if not scores:
        return {"timeline": [], "peak_moment": None}

    peak_moment = max(scores, key=lambda s: s["importance"])

    return {
        "timeline": scores,
        "peak_moment": peak_moment,
    }


def build_audio_timeline(attention_map, time_per_column, top_k=3):
    """
    Full timeline for audio: top-k most attended time segments.
    """

    segments = audio_columns_to_time(attention_map, time_per_column)
    return {
        "top_segments": segments[:top_k],
        "all_segments": segments,
    }