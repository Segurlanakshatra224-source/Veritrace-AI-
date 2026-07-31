import torch
import torch.nn.functional as F
from config import CONFIDENCE_THRESHOLD


def get_confidence_level(confidence):
    """
    Turn a raw 0-1 confidence score into a human-readable label.
    """

    if confidence >= 0.90:
        return "Very High"
    elif confidence >= 0.75:
        return "High"
    elif confidence >= CONFIDENCE_THRESHOLD:
        return "Moderate"

    return "Low"


def per_step_confidence(generated):
    """
    Softmax each generation step's logits and take the probability of the token
    that was actually chosen.
    """

    if generated is None or not hasattr(generated, "scores") or generated.scores is None:
        return []

    step_confidences = []

    for step_logits in generated.scores:
        probs = F.softmax(step_logits, dim=-1)
        top_prob = probs.max(dim=-1).values.item()
        step_confidences.append(top_prob)

    return step_confidences


def compute_answer_confidence(generated):
    """
    Aggregate per-token confidences into one overall answer confidence. Uses the
    geometric mean rather than the arithmetic mean, so one badly-guessed token pulls
    the overall score down instead of being smoothed away by confident tokens.
    """

    step_confidences = per_step_confidence(generated)

    if not step_confidences:
        return 0.0

    log_probs = torch.log(torch.tensor(step_confidences).clamp(min=1e-8))
    geometric_mean = torch.exp(log_probs.mean()).item()

    return geometric_mean


def run_confidence(generated):
    confidence = compute_answer_confidence(generated)
    level = get_confidence_level(confidence)

    return {
        "confidence": confidence,
        "level": level,
        "per_step": per_step_confidence(generated),
    }