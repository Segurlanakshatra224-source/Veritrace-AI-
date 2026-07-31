from config import TRUST_SCORE_WEIGHTS


def score_confidence_component(confidence):
    return confidence #already 0-1


def score_evidence_component(evidence_coverage_pct):
    if evidence_coverage_pct is None:
        return 0.5 #neutral if we don't have this signal

    #coverage that's too low (unfocused) or suspiciously total (100%, likely noise) both look worse
    if evidence_coverage_pct < 5:
        return 0.2
    if evidence_coverage_pct > 80:
        return 0.6
    return min(evidence_coverage_pct / 40, 1.0)


def score_counterfactual_component(counterfactual_result):
    if counterfactual_result is None:
        return 0.5

    return 1.0 if counterfactual_result.get("is_faithful") else 0.2


def score_hallucination_component(hallucination_result):
    if hallucination_result is None:
        return 0.5

    return 0.2 if hallucination_result.get("hallucination") else 1.0


def get_trust_level(trust_score):
    if trust_score >= 80:
        return "High Trust"
    elif trust_score >= 55:
        return "Moderate Trust"
    elif trust_score >= 30:
        return "Low Trust"
    return "Very Low Trust"


"""
rolls every explainability signal (confidence, evidence coverage, counterfactual
faithfulness, hallucination flag) into a single 0-100 number, weighted per
TRUST_SCORE_WEIGHTS in config.py. meant as a quick at-a-glance number for the
dashboard/report, not a replacement for reading the actual explanation.
"""
def compute_trust_score(confidence, evidence_coverage_pct=None, counterfactual_result=None, hallucination_result=None):
    components = {
        "confidence": score_confidence_component(confidence),
        "evidence_coverage": score_evidence_component(evidence_coverage_pct),
        "counterfactual": score_counterfactual_component(counterfactual_result),
        "hallucination": score_hallucination_component(hallucination_result),
    }

    weighted_sum = sum(
        components[key] * TRUST_SCORE_WEIGHTS[key] for key in components
    )
    trust_score = round(weighted_sum * 100, 1)

    return {
        "trust_score": trust_score,
        "trust_level": get_trust_level(trust_score),
        "components": components,
    }