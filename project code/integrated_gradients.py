import os
import cv2
import torch
import numpy as np
from config import DEVICE
from config import PATCH_MERGE_SIZE
from config import IG_N_STEPS

# NOTE ON PERFORMANCE:
# Qwen2-VL flattens image patches into pixel_values with shape
# (num_patches, patch_dim) - for a typical resized image that's 900+ "rows".
# captum's IntegratedGradients treats that leading dimension as a batch of
# examples, so internal_batch_size gets silently overridden up to that size
# (you'll see "Defaulting to internal batch size of 912 equal to the number
# of examples" in the log) and every one of the n_steps forwards the ENTIRE
# model with that whole batch. On CPU that's genuinely slow (minutes, not
# seconds) - it isn't frozen, there's just no progress output. The manual
# loop below does the same math without captum, but prints progress per
# step so it's clear it's working, and IG_N_STEPS in config.py controls how
# many forward/backward passes you pay for.

def forward_func(pixel_values, model, inputs, target_token_idx):
    modified_inputs = dict(inputs)
    modified_inputs["pixel_values"] = pixel_values

    outputs = model(**modified_inputs, return_dict=True)
    logits = outputs.logits
    target_logits = logits[:, target_token_idx, :]
    return target_logits

def get_target_class(model, inputs, target_token_idx):
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        target_class_idx = outputs.logits[:, target_token_idx, :].argmax(dim=-1).item()
    return target_class_idx

def compute_attributions(model, inputs, target_token_idx=-1, target_class_idx=None, n_steps=IG_N_STEPS):
    """
    Manual Integrated Gradients: straight-line path from a black-image baseline
    to the real pixel_values, averaging the gradient of the target logit at
    each interpolation step. Mathematically equivalent to captum's IG, but
    avoids captum silently expanding the batch dimension to the full patch
    count, and reports progress so a slow CPU run doesn't look hung.
    """
    pixel_values = inputs["pixel_values"].clone().detach().to(DEVICE)
    baseline = torch.zeros_like(pixel_values)

    if target_class_idx is None:
        target_class_idx = get_target_class(model, inputs, target_token_idx)

    total_grad = torch.zeros_like(pixel_values)

    for step in range(n_steps):
        alpha = step / max(n_steps - 1, 1)
        interpolated = baseline + alpha * (pixel_values - baseline)
        interpolated = interpolated.clone().detach().requires_grad_(True)

        modified_inputs = dict(inputs)
        modified_inputs["pixel_values"] = interpolated

        outputs = model(**modified_inputs, return_dict=True)
        target_logit = outputs.logits[:, target_token_idx, target_class_idx].sum()

        grad = torch.autograd.grad(target_logit, interpolated)[0]
        total_grad += grad.detach()

        print(f"Integrated Gradients: step {step + 1}/{n_steps} done")

        del outputs, grad, target_logit
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    avg_grad = total_grad / n_steps
    attributions = (pixel_values - baseline) * avg_grad

    return attributions

def process_attributions(attributions, grid_h, grid_w):
    attributions = attributions.detach().cpu().float()
    attributions = torch.abs(attributions)

    per_patch_score = attributions.sum(dim=-1)
    per_patch_score = per_patch_score.flatten()

    expected_raw = (grid_h * PATCH_MERGE_SIZE) * (grid_w * PATCH_MERGE_SIZE)
    if per_patch_score.numel() == expected_raw:
        # attributions are at raw (pre-merge) patch resolution; pool 2x2 -> merged grid
        raw_h, raw_w = grid_h * PATCH_MERGE_SIZE, grid_w * PATCH_MERGE_SIZE
        per_patch_score = per_patch_score.reshape(raw_h, raw_w)
        per_patch_score = per_patch_score.reshape(grid_h, PATCH_MERGE_SIZE, grid_w, PATCH_MERGE_SIZE)
        per_patch_score = per_patch_score.mean(dim=(1, 3))
        attribution_map = per_patch_score
    else:
        expected = grid_h * grid_w
        if per_patch_score.numel() != expected:
            print(
                f"Warning: attribution length {per_patch_score.numel()} "
                f"does not match grid {grid_h}x{grid_w}={expected}"
            )
        attribution_map = per_patch_score.reshape(grid_h, grid_w)

    return attribution_map

def normalize_heatmap(attribution_map):
    attribution_map = attribution_map.detach().cpu().float().numpy()
    low_val = np.percentile(attribution_map, 2)
    high_val = np.percentile(attribution_map, 98)
    if high_val - low_val < 1e-8:
        normalized_map = np.zeros_like(attribution_map)
    else:
        normalized_map = (attribution_map - low_val) / (high_val - low_val)
        normalized_map = np.clip(normalized_map, 0.0, 1.0)
    return normalized_map

def create_heatmap(normalized_map, target_size):
    heatmap_uint8 = np.uint8(255 * normalized_map)
    heatmap_resized = cv2.resize(heatmap_uint8, target_size, interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    return heatmap_color

def create_overlay(original_image, heatmap_color, alpha=0.5):
    if original_image.shape[:2] != heatmap_color.shape[:2]:
        heatmap_color = cv2.resize(
            heatmap_color, (original_image.shape[1], original_image.shape[0])
        )
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def save_results(heatmap_color, overlay, heatmap_path, overlay_path):
    heatmap_dir = os.path.dirname(heatmap_path)
    overlay_dir = os.path.dirname(overlay_path)
    if heatmap_dir:
        os.makedirs(heatmap_dir, exist_ok=True)
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    heatmap_ok = cv2.imwrite(heatmap_path, heatmap_color)
    overlay_ok = cv2.imwrite(overlay_path, overlay)

    print(f"Saved heatmap to {heatmap_path}" if heatmap_ok else f"Failed to save heatmap to {heatmap_path}")
    print(f"Saved overlay to {overlay_path}" if overlay_ok else f"Failed to save overlay to {overlay_path}")

def run_integrated_gradients(
    model,
    inputs,
    original_image,
    grid_h,
    grid_w,
    target_token_idx=-1,
    target_class_idx=None,
    n_steps=IG_N_STEPS,
    heatmap_path="ig_heatmap.png",
    overlay_path="ig_overlay.png",
):
    attributions = compute_attributions(
        model, inputs, target_token_idx, target_class_idx, n_steps
    )

    attribution_map = process_attributions(attributions, grid_h, grid_w)
    normalized_map = normalize_heatmap(attribution_map)

    h, w = original_image.shape[:2]
    heatmap_color = create_heatmap(normalized_map, target_size=(w, h))
    overlay = create_overlay(original_image, heatmap_color)

    save_results(heatmap_color, overlay, heatmap_path, overlay_path)

    return {
        "attributions": attributions,
        "attribution_map": attribution_map,
        "heatmap": heatmap_color,
        "overlay": overlay,
    }