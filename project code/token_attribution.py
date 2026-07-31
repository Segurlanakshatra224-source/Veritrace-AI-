import torch
import torch.nn.functional as F


def get_step_scores(generated):
    """
    Pull the per-generation-step logit distributions out of model.generate() output.
    """

    if generated is None or not hasattr(generated, "scores") or generated.scores is None:
        return None

    return generated.scores


def token_probability(step_logits, token_id):
    """
    Convert one step's logits into the probability actually assigned to the chosen token.
    """

    probs = F.softmax(step_logits, dim=-1)
    return probs[0, token_id].item()


def build_token_attributions(processor, generated_ids, generated):
    """
    Build a per-token breakdown: which token was generated, and how confident the
    model was in that specific token at that specific step.
    """

    step_scores = get_step_scores(generated)
    if step_scores is None:
        return []

    ids = generated_ids[0]
    attributions = []

    for step_index, token_id in enumerate(ids):
        if step_index >= len(step_scores):
            break

        confidence = token_probability(step_scores[step_index], token_id)
        token_text = processor.tokenizer.decode([token_id])

        attributions.append({
            "step": step_index,
            "token": token_text,
            "token_id": int(token_id),
            "confidence": confidence,
        })

    return attributions


def lowest_confidence_tokens(attributions, top_k=3):
    """
    Surface the shakiest tokens in the answer, useful for flagging exactly where a
    hallucination might have crept in.
    """

    sorted_by_confidence = sorted(attributions, key=lambda a: a["confidence"])
    return sorted_by_confidence[:top_k]


def run_token_attribution(processor, generated_ids, generated, top_k=3):
    attributions = build_token_attributions(processor, generated_ids, generated)
    weakest = lowest_confidence_tokens(attributions, top_k) if attributions else []

    return {
        "attributions": attributions,
        "weakest_tokens": weakest,
    }