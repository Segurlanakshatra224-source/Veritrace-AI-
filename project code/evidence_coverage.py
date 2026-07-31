import numpy as np
from config import IMAGE_ATTENTION_THRESHOLD


def summarize_map(attention_map, threshold=IMAGE_ATTENTION_THRESHOLD):
    """
    Pull reportable numbers out of a heatmap (attention_map, attribution_map, or
    fused evidence map): how peaky it is, where the peak sits, and how much of the
    image actually cleared the relevance threshold.
    """

    if attention_map is None:
        return None

    arr = np.asarray(attention_map, dtype=np.float32)
    grid_h, grid_w = arr.shape

    peak_flat_idx = int(arr.argmax())
    peak_row, peak_col = divmod(peak_flat_idx, grid_w)

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - arr_min) / (arr_max - arr_min)

    focused_ratio = float((norm >= threshold).sum()) / norm.size

    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "std": float(arr.std()),
        "peak_location": {"row": peak_row, "col": peak_col},
        "peak_location_normalized": {
            "row_pct": round(peak_row / max(grid_h - 1, 1) * 100, 1),
            "col_pct": round(peak_col / max(grid_w - 1, 1) * 100, 1),
        },
        "focused_region_pct": round(focused_ratio * 100, 1),
        "grid_size": f"{grid_h}x{grid_w}",
    }


def describe_region(peak_location_normalized):
    row_pct = peak_location_normalized["row_pct"]
    col_pct = peak_location_normalized["col_pct"]

    vertical = "top" if row_pct < 33 else "bottom" if row_pct > 66 else "middle"
    horizontal = "left" if col_pct < 33 else "right" if col_pct > 66 else "center"

    if vertical == "middle" and horizontal == "center":
        return "center of the image"
    return f"{vertical}-{horizontal} region of the image"


def compute_coverage(attention_map, threshold=IMAGE_ATTENTION_THRESHOLD):
    """
    How much of the image the model's evidence actually covers. Low coverage means
    the model leaned on a tiny sliver of the image to answer, which is a red flag
    even if confidence looks fine.
    """

    stats = summarize_map(attention_map, threshold)
    if stats is None:
        return None

    return {
        "coverage_pct": stats["focused_region_pct"],
        "region": describe_region(stats["peak_location_normalized"]),
        "stats": stats,
    }