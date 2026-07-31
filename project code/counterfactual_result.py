"""
Counterfactual faithfulness check
==================================

hallucination.py and trust_score.py both expect a `counterfactual_result` dict
with an "is_faithful" key (see counterfactual_result.get("is_faithful", True)
in hallucination.py, and score_counterfactual_component() in trust_score.py) -
this module is what produces that dict.

Idea: an explainability map (attention rollout / IG / cross-attention) is only
trustworthy if the region it says the model was "looking at" actually mattered
to the model. To test that, we mask out the model's own top-attended region in
the image, re-run inference on the masked image with the SAME question, and
compare the new answer/confidence to the original:

- If the answer changes or confidence drops noticeably  -> the region really
  was load-bearing for the answer, so the explanation is faithful.
- If masking it barely changes anything -> the model wasn't actually relying
  on the region the heatmap points to, so the explanation may be misleading
  even if it "looks" reasonable.

This is called from main.py / server.py as:

    counterfactual_result = run_counterfactual(
        processor, model, image, message, attention_map,
        confidence, response,
        prepare_inputs, generate_answer, decode_response,
    )

prepare_inputs / generate_answer / decode_response are passed in (rather than
imported directly from inference.py) purely to avoid a circular import, since
inference.py doesn't need to know anything about counterfactual testing.
"""

import os
import copy
import difflib

import numpy as np
from PIL import Image

from evidence_coverage import summarize_map, describe_region
from confidence import compute_answer_confidence

try:
    from config import (
        COUNTERFACTUAL_MASK_FRACTION,
        COUNTERFACTUAL_CONFIDENCE_DROP_THRESHOLD,
        COUNTERFACTUAL_SIMILARITY_THRESHOLD,
    )
except ImportError:
    # Not every config.py will have these yet since counterfactual testing is
    # new - fall back to sane defaults so this module works standalone.
    COUNTERFACTUAL_MASK_FRACTION = 0.25          # size of the masked box, as a fraction of image width/height
    COUNTERFACTUAL_CONFIDENCE_DROP_THRESHOLD = 0.15  # confidence drop (0-1) big enough to count as "the mask mattered"
    COUNTERFACTUAL_SIMILARITY_THRESHOLD = 0.6        # answer similarity (0-1) below which we call the answer "changed"


def _first_text(response):
    """
    response comes in as either a list (main.py passes result["response"]
    straight through) or an already-unwrapped string (server.py passes
    response[0] itself). Normalize to a plain string either way.
    """

    if isinstance(response, (list, tuple)):
        return response[0] if response else ""

    return response or ""


def compute_mask_box(attention_map, image_width, image_height, mask_frac=COUNTERFACTUAL_MASK_FRACTION):
    """
    Find the peak of the attention/attribution map and translate it into a
    pixel-space box on the original image, sized as a fraction of the image
    so the mask is big enough to actually remove information (masking a
    single patch is usually too small to move the model's answer at all).
    """

    stats = summarize_map(attention_map)

    row_pct = stats["peak_location_normalized"]["row_pct"]
    col_pct = stats["peak_location_normalized"]["col_pct"]

    center_y = row_pct / 100 * image_height
    center_x = col_pct / 100 * image_width

    box_h = image_height * mask_frac
    box_w = image_width * mask_frac

    top = int(max(center_y - box_h / 2, 0))
    bottom = int(min(center_y + box_h / 2, image_height))
    left = int(max(center_x - box_w / 2, 0))
    right = int(min(center_x + box_w / 2, image_width))

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "region_description": describe_region(stats["peak_location_normalized"]),
    }


def mask_image_region(image, box):
    """
    Replace the boxed region with the image's own mean color. Filling with
    the mean (rather than black/white) removes the local information without
    introducing a high-contrast edge that could itself become the model's new
    "most interesting" region.
    """

    arr = np.array(image.convert("RGB")).copy()
    mean_color = arr.reshape(-1, arr.shape[-1]).mean(axis=0)

    arr[box["top"]:box["bottom"], box["left"]:box["right"]] = mean_color

    return Image.fromarray(arr.astype(np.uint8))


def _build_masked_message(message, masked_image):
    """
    Swap the image inside the original chat message for the masked one,
    keeping everything else (question text, roles, etc.) identical. Built
    manually rather than via copy.deepcopy() since PIL Image objects don't
    always deep-copy cleanly.
    """

    masked_message = []

    for turn in message:
        new_turn = dict(turn)
        content = turn.get("content")

        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    new_item = dict(item)
                    new_item["image"] = masked_image
                    new_content.append(new_item)
                else:
                    new_content.append(item)
            new_turn["content"] = new_content

        masked_message.append(new_turn)

    return masked_message


def _normalize_text(text):
    return " ".join(str(text).strip().lower().split())


def responses_differ(original_answer, masked_answer, similarity_threshold=COUNTERFACTUAL_SIMILARITY_THRESHOLD):
    """
    Cheap text-similarity check (no extra model call needed) to decide whether
    masking the top-attended region changed the answer.
    """

    a = _normalize_text(original_answer)
    b = _normalize_text(masked_answer)

    if not a and not b:
        return False, 1.0

    similarity = difflib.SequenceMatcher(None, a, b).ratio()

    return similarity < similarity_threshold, similarity


def run_counterfactual(
    processor,
    model,
    image,
    message,
    attention_map,
    confidence,
    response,
    prepare_inputs,
    generate_answer,
    decode_response,
    mask_path=None,
):
    """
    Full counterfactual faithfulness pipeline for one image + one explanation
    map. Returns None if there's no attention map to test against.
    """

    if attention_map is None:
        return None

    original_answer = _first_text(response)
    image_width, image_height = image.size

    box = compute_mask_box(attention_map, image_width, image_height)
    masked_image = mask_image_region(image, box)

    if mask_path:
        try:
            mask_dir = os.path.dirname(mask_path)
            if mask_dir:
                os.makedirs(mask_dir, exist_ok=True)
            masked_image.save(mask_path)
            print(f"Saved counterfactual masked image to {mask_path}")
        except Exception as e:
            print(f"[counter_factual] Could not save masked image: {e}")

    masked_message = _build_masked_message(message, masked_image)

    try:
        masked_inputs = prepare_inputs(processor, masked_message)
        masked_generated = generate_answer(model, masked_inputs)
        masked_response, _ = decode_response(processor, masked_generated, masked_inputs)
    except Exception as e:
        print(f"[counter_factual] Masked re-inference failed, skipping counterfactual check: {e}")
        return None

    masked_answer = _first_text(masked_response)
    masked_confidence = compute_answer_confidence(masked_generated)

    confidence_drop = confidence - masked_confidence
    answer_changed, similarity = responses_differ(original_answer, masked_answer)

    is_faithful = bool(
        answer_changed or confidence_drop >= COUNTERFACTUAL_CONFIDENCE_DROP_THRESHOLD
    )

    print(
        f"[counter_factual] masked '{box['region_description']}' | "
        f"confidence {confidence:.2%} -> {masked_confidence:.2%} (drop {confidence_drop:.2%}) | "
        f"answer_changed={answer_changed} (similarity={similarity:.2f}) | "
        f"is_faithful={is_faithful}"
    )

    return {
        "is_faithful": is_faithful,
        "original_answer": original_answer,
        "masked_answer": masked_answer,
        "original_confidence": confidence,
        "masked_confidence": masked_confidence,
        "confidence_drop": round(confidence_drop, 4),
        "answer_changed": answer_changed,
        "answer_similarity": round(similarity, 4),
        "masked_region": box["region_description"],
        "mask_box": {k: v for k, v in box.items() if k != "region_description"},
        "mask_fraction": COUNTERFACTUAL_MASK_FRACTION,
    }