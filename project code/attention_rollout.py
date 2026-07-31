import os
import cv2
import torch
import numpy as np
from config import DEVICE, ROLLOUT_LAST_N_LAYERS
def get_attention_matrices(outputs):

    if outputs is None:
        print("No outputs from model.")
        return None

    if outputs.attentions is None:
        print("Model does not return attentions.")
        return None

    # A single NaN/Inf anywhere in any layer poisons the ENTIRE rollout matrix
    # product (nan * anything = nan), which is indistinguishable downstream
    # from genuine saturation - it just always reports "degenerate" no matter
    # how many layers you restrict to. Check at the source so we know which
    # it actually is.
    attentions = outputs.attentions
    for i, layer in enumerate(attentions):
        nan_count = torch.isnan(layer).sum().item()
        inf_count = torch.isinf(layer).sum().item()
        if nan_count or inf_count:
            print(
                f"[attention_rollout] Layer {i}: {nan_count} NaN, {inf_count} Inf "
                f"values out of {layer.numel()} - this is a raw model output problem, "
                f"not a rollout math problem. Sanitizing with nan_to_num so the "
                f"pipeline can continue, but the resulting heatmap will only be as "
                f"meaningful as the fraction of real values left."
            )

    attentions = tuple(
        torch.nan_to_num(layer, nan=0.0, posinf=0.0, neginf=0.0) for layer in attentions
    )

    return attentions


def select_rollout_layers(attentions, last_n=ROLLOUT_LAST_N_LAYERS):
    """
    Rolling out ALL transformer layers (Qwen2-VL has 28+) makes the matrix
    product converge toward a near-uniform distribution - every row of the
    final rollout ends up almost identical, which is what produces a solid-
    blue heatmap (zero variance -> normalize_heatmap returns all zeros).
    Restricting rollout to the last `last_n` layers keeps a usable signal.
    Pass last_n=None to use every layer (the old, saturating behavior).
    """
    if last_n is None or last_n >= len(attentions):
        return attentions
    return attentions[-last_n:]


def avg_of_all_layers(attentions):
    averaged_layers=[]  
    for layer in attentions:       
        layer = layer.squeeze(0).float()   # <-- add .float() here
        layer = layer.mean(dim=0)
        averaged_layers.append(layer)
    return averaged_layers
    
def avg_residual_connection(averaged_layers):
    residual_layers = []
    for layer in averaged_layers:
         seq_len = layer.size(-1)
         identity = torch.eye(seq_len, device=layer.device, dtype=layer.dtype)
         residual_layers.append(layer+identity)
    return residual_layers
"""
this function connection one is very imp this function is made based on an algorithm.
here processor adds back the input value to the output that is why we are adding the I to normal matrix.
later we normalise so that sum of elements in a row becomes 1.
later we will multiply all of the matrices in the residual layers to get back the overall perfect matrix which contians all info.
"""
def normalised_attentions(residual_layers):
    normalised_values = []
    for layer in residual_layers:
        rows_sum = layer.sum(dim=-1,keepdim=True)
        layer = layer/(rows_sum +1e-8) 
        normalised_values.append(layer) #normalisation of the matix
    return normalised_values

def complete_rollout(normalised_values) :
    seq_len = normalised_values[0].size(-1)
    rollout = torch.eye(
        seq_len, device=normalised_values[0].device, dtype = normalised_values[0].dtype
    )
    for layer in normalised_values:
        rollout = torch.matmul(layer, rollout)
 
    return rollout #matmul is multiplication matrix 
#finally we have got a single matrix for the colouring the image.

def extract_image_attention(rollout, image_token_start, image_token_end, target_token_idx=-1):
    row = rollout[target_token_idx]
    image_attention = row[image_token_start:image_token_end]

    # Diagnostic: if this range is degenerate, everything downstream will be
    # degenerate too, regardless of how normalize_heatmap is written.
    with torch.no_grad():
        vals = image_attention.detach().float()
        print(
            f"[attention_rollout debug] image_token_start={image_token_start}, "
            f"image_token_end={image_token_end}, slice_len={vals.numel()}, "
            f"min={vals.min().item():.8f}, max={vals.max().item():.8f}, "
            f"mean={vals.mean().item():.8f}, std={vals.std().item():.8f}, "
            f"row_len={row.numel()}"
        )

    return image_attention
 
 
def reshape_attention_map(image_attention, grid_h, grid_w):
    expected = grid_h * grid_w
    if image_attention.numel() != expected:
        print(
            f"Warning: attention length {image_attention.numel()} "
            f"does not match grid {grid_h}x{grid_w}={expected}"
        )
    attention_map = image_attention.reshape(grid_h, grid_w)
    return attention_map
 
 
def normalize_heatmap(attention_map):
    """
    Normalize attention values to [0,1].
    """
    attention_map = attention_map.detach().cpu().float().numpy()

    if np.isnan(attention_map).any() or np.isinf(attention_map).any():
        attention_map = np.nan_to_num(attention_map, nan=0.0, posinf=0.0, neginf=0.0)

    # Percentile-based stretch is more robust than pure min/max: a couple of
    # outlier patches can otherwise compress everything else toward one end
    # of the range and make the heatmap look flat again.
    low_val = np.percentile(attention_map, 2)
    high_val = np.percentile(attention_map, 98)

    if high_val - low_val < 1e-8:
        print(
            "Warning: attention map has almost no variance even after "
            "restricting rollout depth - heatmap will look flat/solid. "
            "Try lowering ROLLOUT_LAST_N_LAYERS in config.py further."
        )
        return np.zeros_like(attention_map)

    normalized = (attention_map - low_val) / (high_val - low_val)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized
 
  #cv2 can understand only in the form of numpy not in tensor form
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
    # Create output directories if they don't exist (skip if path has no dir component)
    heatmap_dir = os.path.dirname(heatmap_path)
    overlay_dir = os.path.dirname(overlay_path)
    if heatmap_dir:
        os.makedirs(heatmap_dir, exist_ok=True)
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    # Save images and check whether cv2 actually succeeded
    heatmap_ok = cv2.imwrite(heatmap_path, heatmap_color)
    overlay_ok = cv2.imwrite(overlay_path, overlay)

    print("Current working directory:", os.getcwd())
    print("Heatmap path:", os.path.abspath(heatmap_path))
    print("Overlay path:", os.path.abspath(overlay_path))

    if heatmap_ok:
        print(f"Saved heatmap to {os.path.abspath(heatmap_path)}")
    else:
        print(f"Failed to save heatmap to {os.path.abspath(heatmap_path)}")

    if overlay_ok:
        print(f"Saved overlay to {os.path.abspath(overlay_path)}")
    else:
        print(f"Failed to save overlay to {os.path.abspath(overlay_path)}")
 
 
def run_attention_rollout(
    outputs,
    original_image,
    image_token_start,
    image_token_end,
    grid_h,
    grid_w,
    target_token_idx=-1,
    heatmap_path="heatmap.png",
    overlay_path="overlay.png",
):
    attentions = get_attention_matrices(outputs)
    if attentions is None:
        return None

    # If the requested layer count still saturates to near-zero variance,
    # automatically retry with fewer layers rather than silently returning a
    # flat map. Rollout saturation gets worse the more layers you multiply
    # through, so shrinking the window is the correct direction to try.
    candidate_layer_counts = [ROLLOUT_LAST_N_LAYERS, 4, 2, 1]
    image_attention = None
    used_layer_count = None

    for n_layers in candidate_layer_counts:
        selected = select_rollout_layers(attentions, n_layers)
        averaged_layers = avg_of_all_layers(selected)
        residual_layers = avg_residual_connection(averaged_layers)
        normalised_layers = normalised_attentions(residual_layers)
        rollout = complete_rollout(normalised_layers)

        candidate_attention = extract_image_attention(
            rollout, image_token_start, image_token_end, target_token_idx
        )

        with torch.no_grad():
            spread = candidate_attention.detach().float().max() - candidate_attention.detach().float().min()

        if spread.item() > 1e-6:
            image_attention = candidate_attention
            used_layer_count = n_layers
            break
        else:
            print(f"[attention_rollout] {n_layers} layer(s) still saturated (spread={spread.item():.2e}), trying fewer")

    if image_attention is None:
        print("[attention_rollout] Warning: even a single-layer rollout is degenerate. "
              "The image token span or model attention output itself may be wrong - "
              "check the [main debug] and [attention_rollout debug] print lines above.")
        image_attention = candidate_attention  # fall through with whatever we last computed
    else:
        print(f"[attention_rollout] Using last {used_layer_count} layer(s) for rollout")

    attention_map = reshape_attention_map(image_attention, grid_h, grid_w)
    normalized_map = normalize_heatmap(attention_map)
 
    h, w = original_image.shape[:2]
    heatmap_color = create_heatmap(normalized_map, target_size=(w, h))
    overlay = create_overlay(original_image, heatmap_color)
 
    save_results(heatmap_color, overlay, heatmap_path, overlay_path)
 
    return {
        "attention_map": attention_map.detach().cpu(),
        "heatmap": heatmap_color,
        "overlay": overlay,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path
    }
 
    
#we will get two image one is only heat map and other is overlayed heatmap on the normal image