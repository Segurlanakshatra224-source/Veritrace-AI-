import numpy as np
from PIL import Image
from config import COUNTERFACTUAL_MASK_RATIO
from confidence import compute_answer_confidence


def top_attended_mask(attention_map, mask_ratio=COUNTERFACTUAL_MASK_RATIO):
    """
    Find which grid cells the model paid the most attention to, so we can black
    them out and see if the answer actually depended on them.
    """

    arr = np.asarray(attention_map, dtype=np.float32)
    flat = arr.flatten()

    num_to_mask = max(int(len(flat) * mask_ratio), 1)
    threshold_value = np.partition(flat, -num_to_mask)[-num_to_mask]

    mask = arr >= threshold_value
    return mask


def replace_image_in_message(message, new_image):
    """
    inference.prepare_inputs(processor, message) expects the image to already be
    embedded inside the message dict, not passed as a separate argument. Swap the
    masked image into a copy of the message so the counterfactual run actually
    sees the masked pixels instead of the original ones.
    """

    updated_message = []
    for turn in message:
        new_turn = dict(turn)
        new_content = []
        for item in turn["content"]:
            if item.get("type") == "image":
                new_item = dict(item)
                new_item["image"] = new_image
                new_content.append(new_item)
            else:
                new_content.append(item)
        new_turn["content"] = new_content
        updated_message.append(new_turn)

    return updated_message


def apply_mask_to_image(original_image, mask):
    """
    Paint the top-attended grid cells black on a copy of the original image, scaled
    up from the patch grid to actual pixel coordinates.
    """

    image_array = np.array(original_image).copy()
    h, w = image_array.shape[:2]
    grid_h, grid_w = mask.shape

    cell_h = h / grid_h
    cell_w = w / grid_w

    for row in range(grid_h):
        for col in range(grid_w):
            if mask[row, col]:
                y1, y2 = int(row * cell_h), int((row + 1) * cell_h)
                x1, x2 = int(col * cell_w), int((col + 1) * cell_w)
                image_array[y1:y2, x1:x2] = 0

    return Image.fromarray(image_array)


"""
faithfulness check: if the model's own explanation (the attention map) is telling the
truth, blacking out the region it claims mattered most should meaningfully hurt its
confidence or change its answer. if confidence barely moves, the explanation was
probably not faithful to what the model actually used.
"""
def run_counterfactual(
    processor,
    model,
    image,
    message,
    attention_map,
    original_confidence,
    original_response,
    prepare_inputs_fn,
    generate_answer_fn,
    decode_response_fn,
):
    mask = top_attended_mask(attention_map)
    masked_image = apply_mask_to_image(image, mask)
    masked_message = replace_image_in_message(message, masked_image)

    masked_inputs = prepare_inputs_fn(processor, masked_message)
    masked_generated = generate_answer_fn(model, masked_inputs)
    masked_response, _ = decode_response_fn(processor, masked_generated, masked_inputs)

    masked_confidence = compute_answer_confidence(masked_generated)
    confidence_drop = original_confidence - masked_confidence

    answer_changed = masked_response != original_response
    is_faithful = answer_changed or confidence_drop > 0.15 #arbitrary but reasonable bar for "meaningful" drop

    return {
        "masked_image": masked_image,
        "masked_response": masked_response,
        "masked_confidence": masked_confidence,
        "confidence_drop": confidence_drop,
        "answer_changed": answer_changed,
        "is_faithful": is_faithful,
    }