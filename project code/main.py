import sys
from config import PATCH_MERGE_SIZE
from config import (
    ENABLE_ATTENTION_ROLLOUT,
    ENABLE_INTEGRATED_GRADIENTS,
    ENABLE_CONFIDENCE,
)
from preprocessing import preprocess
from conversation import build_convo
from inference import (
    load_processor,
    load_model,
    prepare_inputs,
    extract_attention,
    generate_answer,
    decode_response,
    run_inference,
)
from attention_rollout import run_attention_rollout
from integrated_gradients import run_integrated_gradients
from cross_attention import run_cross_attention
from evidence_fusion import fuse_attention_rollout_ig_cross
from evidence_coverage import compute_coverage
from confidence import compute_answer_confidence
from hallucination import detect_hallucination
from counter_factual import run_counterfactual
from trust_score import compute_trust_score
from token_attribution import run_token_attribution
from temporal_alignment import build_video_timeline, build_audio_timeline
from video_keyframes import select_frames_for_question
from explainability import build_explanation, print_explanation, save_report
from reports import run_reports





"""
qwen2-vl doesn't hand you image_token_start/end or grid_h/grid_w directly, so this
figures them out: image_grid_thw tells us how many patches wide/tall the image was
split into, and the vision start/end special tokens in input_ids tell us where the
image tokens sit inside the full sequence.
"""
def locate_image_tokens(processor, inputs):
    try:
        t, h, w = inputs["image_grid_thw"][0].tolist()
        grid_h = h // PATCH_MERGE_SIZE
        grid_w = w // PATCH_MERGE_SIZE

        input_ids = inputs["input_ids"][0].tolist()
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        image_token_start = input_ids.index(vision_start_id) + 1
        image_token_end = input_ids.index(vision_end_id)

        print(
            f"[main debug] image_token_start={image_token_start}, "
            f"image_token_end={image_token_end}, grid_h={grid_h}, grid_w={grid_w}, "
            f"num_image_tokens={image_token_end - image_token_start}, "
            f"total_sequence_len={len(input_ids)}"
        )

        return image_token_start, image_token_end, grid_h, grid_w

    except Exception as e:
        print(f"Warning: could not auto-locate image tokens ({e}), falling back to defaults")
        return 0, 1, 1, 1


def process_image_item(processor, model, image, question, source_label="input"):
    message = build_convo("image", question, image)
    result = run_inference(processor, model, message)

    inputs = result["inputs"]
    outputs = result["outputs"]
    generated = result["generated"]
    generated_ids = result["generated_ids"]
    response = result["response"]

    image_token_start, image_token_end, grid_h, grid_w = locate_image_tokens(processor, inputs)

    import numpy as np
    original_image = np.array(image)

    rollout_result = None
    ig_result = None
    cross_result = None

    if ENABLE_ATTENTION_ROLLOUT:
        try:
            rollout_result = run_attention_rollout(
                outputs, original_image, image_token_start, image_token_end, grid_h, grid_w,
                heatmap_path=f"outputs/heatmaps/{source_label}_rollout_heatmap.png",
                overlay_path=f"outputs/heatmaps/{source_label}_rollout_overlay.png",
            )
        except Exception as e:
            import traceback
            print(f"[main] Attention Rollout failed, continuing without it: {e}")
            traceback.print_exc()

    if ENABLE_INTEGRATED_GRADIENTS:
        try:
            ig_result = run_integrated_gradients(
                model, inputs, original_image, grid_h, grid_w,
                heatmap_path=f"outputs/heatmaps/{source_label}_ig_heatmap.png",
                overlay_path=f"outputs/heatmaps/{source_label}_ig_overlay.png",
            )
        except Exception as e:
            import traceback
            print(f"[main] Integrated Gradients failed, continuing without it: {e}")
            traceback.print_exc()

    try:
        cross_result = run_cross_attention(
            outputs, original_image, image_token_start, image_token_end, grid_h, grid_w,
            heatmap_path=f"outputs/heatmaps/{source_label}_cross_heatmap.png",
            overlay_path=f"outputs/heatmaps/{source_label}_cross_overlay.png",
        )
    except Exception as e:
        import traceback
        print(f"[main] Cross Attention failed, continuing without it: {e}")
        traceback.print_exc()

    attention_map = rollout_result["attention_map"] if rollout_result else None
    attribution_map = ig_result["attribution_map"] if ig_result else None
    cross_map = cross_result["attention_map"] if cross_result else None

    fused_map = fuse_attention_rollout_ig_cross(attention_map, attribution_map, cross_map)
    coverage = compute_coverage(fused_map if fused_map is not None else attention_map)

    confidence = compute_answer_confidence(generated) if ENABLE_CONFIDENCE else 0.0
    token_attribution = run_token_attribution(processor, generated_ids, generated)

    counterfactual_result = None
    if attention_map is not None:
        try:
            counterfactual_result = run_counterfactual(
                processor, model, image, message, attention_map,
                confidence, response,
                prepare_inputs, generate_answer, decode_response,
            )
        except Exception as e:
            import traceback
            print(f"[main] Counterfactual check failed, continuing without it: {e}")
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

    return {
        "explanation": explanation,
        "response": response,
        "confidence": confidence,
        "attention_map": attention_map,
        "attribution_map": attribution_map,
        "cross_map": cross_map,
        "fused_map": fused_map,
        "token_attribution": token_attribution,
        "trust_result": trust_result,
        "counterfactual_result": counterfactual_result,
    }


def process_video(processor, model, video_message, video_path, question):
    """
    Two things happen here:
    1. One overall answer to the question using Qwen2-VL's native video
       understanding (the whole video, via video_message from preprocessing) -
       this is the main "answer" text, same as the image/audio paths.
    2. Full per-frame explainability (attention rollout + IG + cross-attention)
       ONLY on a small number of frames selected by video_keyframes: either the
       single frame nearest an explicit timestamp mentioned in the question
       ("what's happening at 0:12"), or a handful of auto-detected "main"
       (scene-change) frames if no timestamp was given. Running full
       explainability on every frame of a video would be far too slow.
    """
    overall_result = run_inference(processor, model, video_message)
    overall_answer = overall_result["response"][0] if overall_result["response"] else ""

    selected_frames, requested_time = select_frames_for_question(video_path, question)

    explanations = []
    frame_results = []

    for frame in selected_frames:
        label = f"frame{frame['frame_index']}_t{frame['timestamp']}s"
        item = process_image_item(processor, model, frame["image"], question, source_label=label)

        explanations.append(item["explanation"])
        frame_results.append({
            "timestamp": frame["timestamp"],
            "frame_index": frame["frame_index"],
            "attention_map": item["fused_map"],
        })

    timeline = build_video_timeline(frame_results)
    timeline["requested_time"] = requested_time
    timeline["overall_answer"] = overall_answer

    return explanations, timeline, overall_answer


def process_audio(processor, model, audio_data, question):
    # NOTE: audio_preprocessing.preprocess_audio() currently returns
    # {"audio", "sample_rate", "duration"} - it does not produce "spectrogram_image"
    # or "time_per_column" yet. Audio is scoped out of this build cycle, so this
    # path will KeyError until that preprocessing step is implemented; left as-is
    # rather than fabricating a spectrogram pipeline here.
    spectrogram_image = audio_data["spectrogram_image"]
    time_per_column = audio_data["time_per_column"]

    from PIL import Image
    image = Image.fromarray(spectrogram_image)

    item = process_image_item(processor, model, image, question, source_label="audio_spectrogram")

    timeline = build_audio_timeline(item["fused_map"], time_per_column)
    return [item["explanation"]], timeline


def run_pipeline(file_path, question):
    processor = load_processor()
    model = load_model()

    prepared = preprocess(file_path, question, processor)
    modality = prepared["modality"]
    data = prepared["data"]

    if modality == "image":
        item = process_image_item(processor, model, data["image"], question, source_label="image")
        print_explanation(item["explanation"])
        heatmap_paths = [
            ("outputs/heatmaps/image_rollout_overlay.png", "Attention Rollout"),
            ("outputs/heatmaps/image_ig_overlay.png", "Integrated Gradients"),
            ("outputs/heatmaps/image_cross_overlay.png", "Cross Attention"),
        ]
        save_report(item["explanation"], heatmap_paths=heatmap_paths)
        run_reports(file_path, modality, [item["explanation"]])

    elif modality == "video":
        explanations, timeline, overall_answer = process_video(
            processor, model, data["message"], file_path, question
        )
        print(f"\nOverall video answer: {overall_answer}\n")
        for exp in explanations:
            print_explanation(exp)
        run_reports(file_path, modality, explanations, timeline)

    else:
        explanations, timeline = process_audio(processor, model, data, question)
        for exp in explanations:
            print_explanation(exp)
        run_reports(file_path, modality, explanations, timeline)


if __name__ == "__main__":
    file_path = input("Enter file path (image/video/audio): ").strip()
    question = input("Enter your question: ").strip()

    run_pipeline(file_path, question)