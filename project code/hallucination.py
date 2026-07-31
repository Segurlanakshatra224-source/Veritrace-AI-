from config import (
    ENABLE_HALLUCINATION,
    CONFIDENCE_THRESHOLD,
    IMAGE_ATTENTION_THRESHOLD,
)


"""
heuristic hallucination flag, not a learned classifier: combines three signals that
each independently suggest an ungrounded answer. any one of them alone can be noise,
but two or more firing together is a decent signal something's off.
"""
def detect_hallucination(confidence, evidence_coverage_pct=None, counterfactual_result=None):
    if not ENABLE_HALLUCINATION:
        return None

    reasons = []

    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append(
            f"model confidence ({confidence:.2%}) is below the {CONFIDENCE_THRESHOLD:.2%} threshold"
        )

    if evidence_coverage_pct is not None and evidence_coverage_pct < 10:
        reasons.append(
            f"only {evidence_coverage_pct}% of the image cleared the relevance threshold, "
            f"the answer isn't well grounded in the image"
        )

    if counterfactual_result is not None and not counterfactual_result.get("is_faithful", True):
        reasons.append(
            "masking the model's own top-attended region barely changed its answer or "
            "confidence, meaning the explanation may not reflect what the model actually used"
        )

    hallucination = len(reasons) >= 2 #require at least two independent red flags before flagging

    return {
        "hallucination": hallucination,
        "reasons": reasons,
    }