import os
import json
from datetime import datetime
from config import REPORT_FOLDER


"""
explainability.py's save_report handles ONE question/answer's report. this module
sits a level above it: for a video (many keyframes) or a batch run, it rolls all
the individual explanations into one combined summary report.
"""
def summarize_run(explanations):
    if not explanations:
        return {"count": 0}

    confidences = []
    trust_scores = []
    hallucination_flags = []

    for exp in explanations:
        trust_line = exp.get("Trust", "")
        if "Trust Score" in trust_line:
            try:
                trust_scores.append(float(trust_line.split(":")[1].split("/")[0].strip()))
            except (IndexError, ValueError):
                pass

        if "Possible hallucination detected" in exp.get("Hallucination", ""):
            hallucination_flags.append(True)
        else:
            hallucination_flags.append(False)

    return {
        "count": len(explanations),
        "avg_trust_score": round(sum(trust_scores) / len(trust_scores), 1) if trust_scores else None,
        "min_trust_score": min(trust_scores) if trust_scores else None,
        "max_trust_score": max(trust_scores) if trust_scores else None,
        "hallucination_rate_pct": round(
            100 * sum(hallucination_flags) / len(hallucination_flags), 1
        ) if hallucination_flags else None,
    }


def build_full_report(source_path, modality, explanations, timeline=None):
    summary = summarize_run(explanations)

    return {
        "source": source_path,
        "modality": modality,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "timeline": timeline,
        "items": explanations,
    }


def save_full_report(full_report, filename_prefix="full_report"):
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(REPORT_FOLDER, f"{filename_prefix}_{stamp}.json")

    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"Saved full report to {json_path}")
    return json_path


def run_reports(source_path, modality, explanations, timeline=None):
    full_report = build_full_report(source_path, modality, explanations, timeline)
    path = save_full_report(full_report)

    return full_report, path