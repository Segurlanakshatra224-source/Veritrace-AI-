import os
import json
from datetime import datetime
from config import MODEL_NAME, REPORT_FOLDER
from confidence import get_confidence_level
from evidence_coverage import summarize_map, describe_region, compute_coverage
from pdf_report import save_pdf_report


def generate_attention_explanation(attention_map=None):
    base = (
        "Attention Rollout highlights the image regions that received the "
        "highest attention throughout all transformer layers."
    )

    stats = summarize_map(attention_map)
    if stats is None:
        return base + " (No attention map available for this run.)"

    region = describe_region(stats["peak_location_normalized"])
    return base + (
        f"\nMost attended region: {region} (grid {stats['grid_size']}). "
        f"Peak: {stats['max']:.4f} | Mean: {stats['mean']:.4f} | "
        f"Focused area: {stats['focused_region_pct']}%."
    )


def generate_integrated_gradients_explanation(attribution_map=None):
    base = (
        "Integrated Gradients highlights the pixels that contributed the "
        "most towards the model's prediction."
    )

    stats = summarize_map(attribution_map)
    if stats is None:
        return base + " (No attribution map available for this run.)"

    region = describe_region(stats["peak_location_normalized"])
    return base + (
        f"\nStrongest contributing region: {region} (grid {stats['grid_size']}). "
        f"Peak: {stats['max']:.4f} | Mean: {stats['mean']:.4f} | "
        f"Focused area: {stats['focused_region_pct']}%."
    )


def generate_cross_attention_explanation(attention_map=None):
    base = (
        "Cross Attention highlights the image regions the final transformer "
        "layer focused on directly when producing the answer."
    )

    stats = summarize_map(attention_map)
    if stats is None:
        return base + " (No cross-attention map available for this run.)"

    region = describe_region(stats["peak_location_normalized"])
    return base + f"\nMost attended region: {region} (grid {stats['grid_size']})."


def generate_confidence_explanation(confidence):
    level = get_confidence_level(confidence)
    return f"Prediction Confidence : {confidence:.2%}\nConfidence Level : {level}"


def generate_trust_explanation(trust_result):
    if trust_result is None:
        return "Trust score not computed for this run."

    return (
        f"Trust Score : {trust_result['trust_score']}/100\n"
        f"Trust Level : {trust_result['trust_level']}"
    )


def generate_hallucination_explanation(hallucination_result):
    if hallucination_result is None:
        return "Hallucination detection was disabled for this run."

    if hallucination_result["hallucination"]:
        base = "Possible hallucination detected."
        if hallucination_result["reasons"]:
            base += " Reasons: " + "; ".join(hallucination_result["reasons"]) + "."
        return base

    return "No hallucination detected. The prediction appears to be grounded in the evidence."


def generate_counterfactual_explanation(counterfactual_result):
    """
    counterfactual_result is whatever counter_factual.run_counterfactual() returned:
    masked_response, masked_confidence, confidence_drop, answer_changed, is_faithful
    (masked_image is intentionally not surfaced here - it's a PIL image, not
    reportable text).
    """
    if counterfactual_result is None:
        return (
            "Counterfactual faithfulness check was not run for this response "
            "(no attention map was available to mask, or the check failed)."
        )

    is_faithful = counterfactual_result.get("is_faithful")
    confidence_drop = counterfactual_result.get("confidence_drop", 0.0)
    answer_changed = counterfactual_result.get("answer_changed")
    masked_response = counterfactual_result.get("masked_response", "")
    masked_confidence = counterfactual_result.get("masked_confidence")

    verdict = (
        "Faithful \u2014 masking the model's own top-attended region meaningfully "
        "changed the answer or confidence, so the explanation reflects what the "
        "model actually relied on."
        if is_faithful else
        "Potentially unfaithful \u2014 masking the model's own top-attended region "
        "barely changed its answer or confidence, meaning the highlighted "
        "region may not be what the model actually used."
    )

    lines = [verdict, ""]
    lines.append(f"Answer changed after masking : {'Yes' if answer_changed else 'No'}")
    lines.append(f"Confidence drop after masking : {confidence_drop:.2%}")
    if masked_confidence is not None:
        lines.append(f"Masked-image confidence : {masked_confidence:.2%}")
    if masked_response:
        lines.append(f"Answer on masked image : {masked_response}")

    return "\n".join(lines)


def build_explanation(
        question,
        answer,
        confidence,
        modality="image",
        image_path=None,
        attention_map=None,
        attribution_map=None,
        cross_attention_map=None,
        fused_map=None,
        coverage=None,
        hallucination_result=None,
        trust_result=None,
        counterfactual_result=None,
        timeline=None,
):
    explanation = {
        "Metadata": (
            f"Model     : {MODEL_NAME}\n"
            f"Modality  : {modality}\n"
            f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Source    : {image_path or 'N/A'}"
        ),
        "Question": question,
        "Answer": answer,
        "Confidence": generate_confidence_explanation(confidence),
        "Trust": generate_trust_explanation(trust_result),
        "Hallucination": generate_hallucination_explanation(hallucination_result),
        "Counterfactual": generate_counterfactual_explanation(counterfactual_result),
        "Attention Rollout": generate_attention_explanation(attention_map),
        "Integrated Gradients": generate_integrated_gradients_explanation(attribution_map),
        "Cross Attention": generate_cross_attention_explanation(cross_attention_map),
    }

    if coverage is not None:
        explanation["Evidence Coverage"] = (
            f"{coverage['coverage_pct']}% of the {modality} cleared the relevance "
            f"threshold, centered on the {coverage['region']}."
        )

    if timeline is not None:
        explanation["Timeline"] = timeline

    return explanation


def print_explanation(explanation):
    print("\n" + "=" * 60)
    print("VeriTrace Explainability Report".center(60))
    print("=" * 60 + "\n")
    for key, value in explanation.items():
        print(f"{key}")
        print("-" * len(str(key)))
        print(value)
        print()
    print("=" * 60 + "\n")


def save_report(explanation, filename_prefix="report", heatmap_paths=None):
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = os.path.join(REPORT_FOLDER, f"{filename_prefix}_{stamp}.txt")
    json_path = os.path.join(REPORT_FOLDER, f"{filename_prefix}_{stamp}.json")

    with open(txt_path, "w") as f:
        f.write("VeriTrace Explainability Report\n")
        f.write("=" * 60 + "\n\n")
        for key, value in explanation.items():
            f.write(f"{key}\n{'-' * len(str(key))}\n{value}\n\n")

    with open(json_path, "w") as f:
        json.dump(explanation, f, indent=2, default=str)

    print(f"Saved text report to {txt_path}")
    print(f"Saved JSON report to {json_path}")

    pdf_path = save_pdf_report(
        explanation, REPORT_FOLDER, filename_prefix=filename_prefix,
        stamp=stamp, heatmap_paths=heatmap_paths,
    )

    return txt_path, json_path, pdf_path


def run_explainability(
        question,
        answer,
        confidence,
        modality="image",
        image_path=None,
        attention_map=None,
        attribution_map=None,
        cross_attention_map=None,
        fused_map=None,
        hallucination_result=None,
        trust_result=None,
        counterfactual_result=None,
        timeline=None,
        save=True,
        heatmap_paths=None,
):
    coverage = compute_coverage(fused_map if fused_map is not None else attention_map)

    explanation = build_explanation(
        question=question,
        answer=answer,
        confidence=confidence,
        modality=modality,
        image_path=image_path,
        attention_map=attention_map,
        attribution_map=attribution_map,
        cross_attention_map=cross_attention_map,
        fused_map=fused_map,
        coverage=coverage,
        hallucination_result=hallucination_result,
        trust_result=trust_result,
        counterfactual_result=counterfactual_result,
        timeline=timeline,
    )

    print_explanation(explanation)

    if save:
        save_report(explanation, heatmap_paths=heatmap_paths)

    return explanation