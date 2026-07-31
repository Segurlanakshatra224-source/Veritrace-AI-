import cv2
import numpy as np


def to_common_grid(map_array, target_shape):
    """
    Different evidence sources (attention rollout, integrated gradients, cross
    attention) can come out as slightly different grid sizes. Resize everything onto
    one common grid before combining them.
    """

    arr = np.asarray(map_array, dtype=np.float32)

    if arr.shape == target_shape:
        return arr

    resized = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return resized


def normalize_map(map_array):
    min_val = map_array.min()
    max_val = map_array.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(map_array)

    return (map_array - min_val) / (max_val - min_val)


"""
combines evidence from multiple explainability methods into one fused map. each method
sees the model from a different angle (attention rollout traces layer-to-layer flow,
integrated gradients traces pixel-level gradient contribution, cross attention looks
only at the final layer) so agreement between them is a stronger signal than any one
method alone.
"""
def fuse_evidence(maps, weights=None):
    valid_maps = [m for m in maps if m is not None]

    if not valid_maps:
        return None

    target_shape = np.asarray(valid_maps[0]).shape

    normalized_maps = [
        normalize_map(to_common_grid(m, target_shape)) for m in valid_maps
    ]

    if weights is None or len(weights) != len(normalized_maps):
        weights = [1.0 / len(normalized_maps)] * len(normalized_maps)

    fused = np.zeros(target_shape, dtype=np.float32)
    for m, w in zip(normalized_maps, weights):
        fused += m * w

    fused = normalize_map(fused)
    return fused


def fuse_attention_rollout_ig_cross(rollout_map=None, ig_map=None, cross_map=None, weights=None):
    maps = [rollout_map, ig_map, cross_map]
    return fuse_evidence(maps, weights)