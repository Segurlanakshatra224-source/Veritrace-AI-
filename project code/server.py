"""
VeriTrace API server
=====================

Thin FastAPI wrapper around the ACTUAL VeriTrace pipeline in this project
(preprocessing, inference, attention_rollout, integrated_gradients,
cross_attention, evidence_fusion, evidence_coverage, confidence,
hallucination, trust_score, explainability) so index.html can drive it over
HTTP instead of main.py's CLI or dashboard.py's Streamlit UI.

This replaces an earlier draft of server.py that imported functions/modules
that don't exist in this codebase (hallucination_detection.run_hallucination_detection,
explainability.summarize_map, utils.pil_to_numpy/extract_image_token_information/
create_output_directories, run_inference(image=...), run_attention_rollout(answer_start=...),
run_integrated_gradients(generated_ids=...), config.ENABLE_TOKEN_CONFIDENCE). Every
call below matches the real function signatures used by main.py's process_image_item.

SETUP
-----
1. Put this file in the same folder as your VeriTrace modules (config.py,
   inference.py, attention_rollout.py, integrated_gradients.py,
   cross_attention.py, evidence_fusion.py, evidence_coverage.py,
   confidence.py, hallucination.py, trust_score.py, counter_factual.py,
   explainability.py, conversation.py, main.py).

2. Install the extra dependencies this server needs on top of what
   VeriTrace already requires:

       pip install fastapi "uvicorn[standard]" python-multipart

3. Run it:

       python server.py

   This starts the API on http://localhost:8000 and loads the model once
   at startup, so the (slow) model load only happens once, not per-request.

4. Open index.html in your browser (or serve it with any static file
   server). It talks to http://localhost:8000 by default - change
   API_BASE at the top of index.html's <script> if you deploy elsewhere.
"""

import base64
import io
import os
import uuid
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from conversation import build_convo
from preprocessing import preprocess
from video_keyframes import select_frames_for_question
from inference import (
    load_processor,
    load_model,
    run_inference,
    prepare_inputs,
    generate_answer,
    decode_response,
)
from main import locate_image_tokens
from attention_rollout import run_attention_rollout
from integrated_gradients import run_integrated_gradients
from cross_attention import run_cross_attention
from evidence_fusion import fuse_attention_rollout_ig_cross
from evidence_coverage import compute_coverage, summarize_map
from confidence import compute_answer_confidence
from hallucination import detect_hallucination
from counter_factual import run_counterfactual
from trust_score import compute_trust_score
from explainability import build_explanation, save_report
from config import (
    ENABLE_ATTENTION_ROLLOUT,
    ENABLE_INTEGRATED_GRADIENTS,
    ENABLE_CONFIDENCE,
    HEATMAP_FOLDER,
    REPORT_FOLDER,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
)


app = FastAPI(title="VeriTrace API")

# Wide-open CORS for local development. Tighten this to your actual
# frontend origin before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated once at startup so every request reuses the loaded model.
_state = {"processor": None, "model": None}


@app.on_event("startup")
def _load_model_once():
    os.makedirs(HEATMAP_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    _state["processor"] = load_processor()
    _state["model"] = load_model()
    print("VeriTrace model loaded. API ready on http://localhost:8000")


def _encode_png_base64(image_bgr):
    """cv2 BGR ndarray -> base64-encoded PNG string, for embedding straight
    into a JSON response as a data URI on the frontend."""
    if image_bgr is None:
        return None
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


def _analyze_frame(processor, model, pil_image, question, run_id, source_label="image"):
    """
    The full single-frame explainability pipeline: inference, attention
    rollout, integrated gradients, cross-attention, fusion, confidence,
    counterfactual, hallucination, trust score, and report. Used by both
    /api/analyze (one image) and /api/analyze-video (one selected frame per
    call), so the two endpoints can't drift out of sync with each other.
    """
    message = build_convo("image", question, pil_image)
    result = run_inference(processor, model, message)

    inputs = result["inputs"]
    outputs = result["outputs"]
    generated = result["generated"]
    response = result["response"][0] if result["response"] else ""

    original_image = np.array(pil_image)
    image_token_start, image_token_end, grid_h, grid_w = locate_image_tokens(processor, inputs)

    rollout_result = None
    if ENABLE_ATTENTION_ROLLOUT:
        heatmap_path = os.path.join(HEATMAP_FOLDER, f"{run_id}_{source_label}_rollout_heatmap.png")
        overlay_path = os.path.join(HEATMAP_FOLDER, f"{run_id}_{source_label}_rollout_overlay.png")
        try:
            rollout_result = run_attention_rollout(
                outputs, original_image, image_token_start, image_token_end, grid_h, grid_w,
                heatmap_path=heatmap_path, overlay_path=overlay_path,
            )
        except Exception as e:
            print(f"[server] Attention Rollout failed for {source_label}, continuing without it: {e}")
            traceback.print_exc()

    ig_result = None
    ig_overlay_path = None
    if ENABLE_INTEGRATED_GRADIENTS:
        ig_heatmap_path = os.path.join(HEATMAP_FOLDER, f"{run_id}_{source_label}_ig_heatmap.png")
        ig_overlay_path = os.path.join(HEATMAP_FOLDER, f"{run_id}_{source_label}_ig_overlay.png")
        try:
            ig_result = run_integrated_gradients(
                model, inputs, original_image, grid_h, grid_w,
                heatmap_path=ig_heatmap_path, overlay_path=ig_overlay_path,
            )
        except Exception as e:
            print(f"[server] Integrated Gradients failed for {source_label}, continuing without it: {e}")
            traceback.print_exc()

    cross_result = None
    cross_heatmap_path = os.path.join(HEATMAP_FOLDER, f"{run_id}_{source_label}_cross_heatmap.png")
    cross_overlay_path = os.path.join(HEATMAP_FOLDER, f"{run_id}_{source_label}_cross_overlay.png")
    try:
        cross_result = run_cross_attention(
            outputs, original_image, image_token_start, image_token_end, grid_h, grid_w,
            heatmap_path=cross_heatmap_path, overlay_path=cross_overlay_path,
        )
    except Exception as e:
        print(f"[server] Cross Attention failed for {source_label}, continuing without it: {e}")
        traceback.print_exc()

    attention_map = rollout_result["attention_map"] if rollout_result else None
    attribution_map = ig_result["attribution_map"] if ig_result else None
    cross_map = cross_result["attention_map"] if cross_result else None

    fused_map = fuse_attention_rollout_ig_cross(attention_map, attribution_map, cross_map)
    coverage = compute_coverage(fused_map if fused_map is not None else attention_map)

    confidence = compute_answer_confidence(generated) if ENABLE_CONFIDENCE else 0.0

    counterfactual_result = None
    if attention_map is not None:
        try:
            counterfactual_result = run_counterfactual(
                processor, model, pil_image, message, attention_map,
                confidence, response,
                prepare_inputs, generate_answer, decode_response,
            )
        except Exception as e:
            print(f"[server] Counterfactual check failed for {source_label}, continuing without it: {e}")
            traceback.print_exc()

    hallucination_result = detect_hallucination(
        confidence,
        coverage["coverage_pct"] if coverage else None,
        counterfactual_result,
    )

    trust_result = compute_trust_score(
        confidence,
        coverage["coverage_pct"] if coverage else None,
        counterfactual_result,
        hallucination_result,
    )

    explanation = build_explanation(
        question=question,
        answer=response,
        confidence=confidence,
        modality="image",
        image_path=source_label,
        attention_map=attention_map,
        attribution_map=attribution_map,
        cross_attention_map=cross_map,
        fused_map=fused_map,
        coverage=coverage,
        hallucination_result=hallucination_result,
        trust_result=trust_result,
        counterfactual_result=counterfactual_result,
    )

    heatmap_paths = [
        (rollout_result["overlay_path"] if rollout_result else None, "Attention Rollout"),
        (ig_overlay_path if ig_result else None, "Integrated Gradients"),
        (cross_result["overlay_path"] if cross_result else None, "Cross Attention"),
    ]

    txt_path, json_path, pdf_path = save_report(
        explanation, filename_prefix=f"veritrace_{run_id}_{source_label}", heatmap_paths=heatmap_paths
    )

    coverage_pct = coverage["coverage_pct"] if coverage else None

    return {
        "answer": response,
        "confidence": confidence,
        "hallucination": bool(hallucination_result["hallucination"]) if hallucination_result else None,
        "hallucination_reasons": hallucination_result["reasons"] if hallucination_result else [],
        "trust_score": trust_result["trust_score"] if trust_result else None,
        "trust_level": trust_result["trust_level"] if trust_result else None,
        "attention_mass": (coverage_pct / 100) if coverage_pct is not None else None,
        "counterfactual": {
            "is_faithful": counterfactual_result.get("is_faithful"),
            "answer_changed": counterfactual_result.get("answer_changed"),
            "confidence_drop": counterfactual_result.get("confidence_drop"),
            "masked_confidence": counterfactual_result.get("masked_confidence"),
            "masked_response": counterfactual_result.get("masked_response"),
        } if counterfactual_result else None,
        "attention_overlay": _encode_png_base64(rollout_result["overlay"]) if rollout_result else None,
        "ig_overlay": _encode_png_base64(ig_result["overlay"]) if ig_result else None,
        "cross_overlay": _encode_png_base64(cross_result["overlay"]) if cross_result else None,
        "attention_map_stats": summarize_map(attention_map) if attention_map is not None else None,
        "ig_map_stats": summarize_map(attribution_map) if attribution_map is not None else None,
        "explanation": explanation,
        "report_txt_url": f"/api/report/{os.path.basename(txt_path)}" if txt_path else None,
        "report_json_url": f"/api/report/{os.path.basename(json_path)}" if json_path else None,
        "report_pdf_url": f"/api/report/{os.path.basename(pdf_path)}" if pdf_path else None,
    }


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), question: str = Form(...)):
    processor = _state["processor"]
    model = _state["model"]

    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    extension = os.path.splitext(image.filename or "")[1].lower()
    if extension not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{extension}'. Use one of {SUPPORTED_IMAGE_FORMATS}.",
        )

    raw_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    run_id = uuid.uuid4().hex[:10]
    result = _analyze_frame(processor, model, pil_image, question, run_id, source_label="image")
    result["run_id"] = run_id
    result["question"] = question
    return result


@app.post("/api/analyze-video")
async def analyze_video(video: UploadFile = File(...), question: str = Form(...)):
    processor = _state["processor"]
    model = _state["model"]

    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    extension = os.path.splitext(video.filename or "")[1].lower()
    if extension not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{extension}'. Use one of {SUPPORTED_VIDEO_FORMATS}.",
        )

    run_id = uuid.uuid4().hex[:10]
    temp_path = os.path.join(HEATMAP_FOLDER, f"{run_id}_upload{extension}")
    raw_bytes = await video.read()
    with open(temp_path, "wb") as f:
        f.write(raw_bytes)

    try:
        # One overall answer using Qwen2-VL's native video understanding
        # (the whole video), same as main.py's process_video().
        prepared = preprocess(temp_path, question, processor)
        video_message = prepared["data"]["message"]
        overall_result = run_inference(processor, model, video_message)
        overall_answer = overall_result["response"][0] if overall_result["response"] else ""

        # Frame selection: nearest frame to an explicit timestamp in the
        # question, or a handful of auto-detected "main" (scene-change)
        # frames otherwise. Full explainability only runs on these.
        selected_frames, requested_time = select_frames_for_question(temp_path, question)

        frame_payloads = []
        for frame in selected_frames:
            label = f"frame{frame['frame_index']}_t{frame['timestamp']}s"
            frame_result = _analyze_frame(
                processor, model, frame["image"], question, run_id, source_label=label
            )
            frame_result["timestamp"] = frame["timestamp"]
            frame_result["frame_index"] = frame["frame_index"]
            frame_payloads.append(frame_result)

        return {
            "run_id": run_id,
            "question": question,
            "overall_answer": overall_answer,
            "requested_time": requested_time,
            "frames": frame_payloads,
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.get("/api/report/{filename}")
def get_report(filename: str):
    # basename() strips any path components, so this can't escape REPORT_FOLDER
    safe_name = os.path.basename(filename)
    path = os.path.join(REPORT_FOLDER, safe_name)

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Report not found.")

    return FileResponse(path, filename=safe_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)