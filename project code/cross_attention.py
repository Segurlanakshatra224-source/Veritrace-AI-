import os
import cv2
import torch
import numpy as np


def get_last_layer_attention(outputs):
    """
    Extract the last transformer layer attention.
    """

    if outputs is None:
        return None

    if outputs.attentions is None:
        return None

    last_layer = outputs.attentions[-1]

    nan_count = torch.isnan(last_layer).sum().item()
    inf_count = torch.isinf(last_layer).sum().item()
    if nan_count or inf_count:
        print(
            f"[cross_attention] Last layer: {nan_count} NaN, {inf_count} Inf "
            f"values out of {last_layer.numel()} - raw model output problem, "
            f"sanitizing so the pipeline can continue."
        )
        last_layer = torch.nan_to_num(last_layer, nan=0.0, posinf=0.0, neginf=0.0)

    return last_layer


def average_heads(last_layer):
    """
    Average all attention heads.
    """

    last_layer = last_layer.squeeze(0).float()

    return last_layer.mean(dim=0)


def get_token_attention(attention_matrix, token_index):
    """
    Extract attention for a generated token.
    """

    return attention_matrix[token_index]


def extract_image_attention(
    token_attention,
    image_token_start,
    image_token_end
):
    """
    Extract only image-token attention.
    """

    return token_attention[
        image_token_start:image_token_end
    ]


def reshape_attention_map(
    image_attention,
    grid_h,
    grid_w
):
    """
    Reshape image attention into patch grid.
    """

    return image_attention.reshape(
        grid_h,
        grid_w
    )


def normalize_heatmap(attention_map):
    """
    Normalize attention values to [0,1].
    """

    attention_map = attention_map.detach().cpu().float().numpy()

    if np.isnan(attention_map).any() or np.isinf(attention_map).any():
        attention_map = np.nan_to_num(attention_map, nan=0.0, posinf=0.0, neginf=0.0)

    low_val = np.percentile(attention_map, 2)
    high_val = np.percentile(attention_map, 98)

    if high_val - low_val < 1e-8:
        return np.zeros_like(attention_map)

    normalized = (attention_map - low_val) / (high_val - low_val)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def create_heatmap(
    normalized_map,
    target_size
):
    """
    Create colored heatmap.
    """

    heatmap = np.uint8(
        normalized_map * 255
    )

    heatmap = cv2.resize(
        heatmap,
        target_size,
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    return heatmap


def create_overlay(
    original_image,
    heatmap,
    alpha=0.5
):
    """
    Overlay heatmap on original image.
    """

    if original_image.shape[:2] != heatmap.shape[:2]:

        heatmap = cv2.resize(
            heatmap,
            (
                original_image.shape[1],
                original_image.shape[0]
            )
        )

    overlay = cv2.addWeighted(
        original_image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay


def save_results(
    heatmap,
    overlay,
    heatmap_path,
    overlay_path
):
    """
    Save heatmap and overlay images.
    """

    heatmap_dir = os.path.dirname(heatmap_path)
    overlay_dir = os.path.dirname(overlay_path)
    if heatmap_dir:
        os.makedirs(heatmap_dir, exist_ok=True)
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    heatmap_ok = cv2.imwrite(heatmap_path, heatmap)
    overlay_ok = cv2.imwrite(overlay_path, overlay)

    print(f"Saved heatmap to {heatmap_path}" if heatmap_ok else f"Failed to save heatmap to {heatmap_path}")
    print(f"Saved overlay to {overlay_path}" if overlay_ok else f"Failed to save overlay to {overlay_path}")


def run_cross_attention(
    outputs,
    original_image,
    image_token_start,
    image_token_end,
    grid_h,
    grid_w,
    token_index=-1,
    heatmap_path="cross_attention_heatmap.png",
    overlay_path="cross_attention_overlay.png"
):
    """
    Complete Cross-Attention visualization pipeline.
    """

    last_layer = get_last_layer_attention(
        outputs
    )

    if last_layer is None:
        return None

    attention_matrix = average_heads(
        last_layer
    )

    token_attention = get_token_attention(
        attention_matrix,
        token_index
    )

    image_attention = extract_image_attention(
        token_attention,
        image_token_start,
        image_token_end
    )

    attention_map = reshape_attention_map(
        image_attention,
        grid_h,
        grid_w
    )

    normalized_map = normalize_heatmap(
        attention_map
    )

    height, width = original_image.shape[:2]

    heatmap = create_heatmap(
        normalized_map,
        target_size=(width, height)
    )

    overlay = create_overlay(
        original_image,
        heatmap
    )

    save_results(
        heatmap,
        overlay,
        heatmap_path,
        overlay_path
    )

    return {

        "attention_map": attention_map.detach().cpu(),

        "heatmap": heatmap,

        "overlay": overlay,

        "heatmap_path": heatmap_path,

        "overlay_path": overlay_path
    }