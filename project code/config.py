import os
import torch
#environmental variables
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#decides which device to use 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
SUPPORTED_IMAGE_FORMATS = [".jpg",".jpeg",".png",".bmp"]
SUPPORTED_VIDEO_FORMATS = [".mp4",".avi",".mov",".mkv"]
FRAME_SAMPLING_RATE = 1 #ONE FRAME PER SECOND
MAX_FRAMES = 64
SUPPORTED_AUDIO_FORMATS = [".wav",".mp3",".flac",".m4a"]
TARGET_SAMPLE_RATES = 16000 #CONVERTS 16000 AUDIO UNITS PER SECOND
MAX_NEW_TOKENS = 512
DO_SAMPLE = False #The model always chooses the most probable token if it is set to false.
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
ATTENTION_IMPLEMENTATION = "sdpa"
# NOTE: an earlier attempt set this to "eager" permanently to fix NaN attention
# values, but that was wrong - it broke actual generation (garbage "!!!!"
# output) because it made the whole model run in eager mode, not just the
# attention-capture pass. Back to sdpa for real inference/generation. Eager is
# still used, but ONLY for the separate one-off forward pass in
# inference.extract_attention() that exists solely to read output_attentions
# (sdpa can't return attention weights at all). If THAT eager pass produces
# NaN across every layer past the first one, that's a sign of a deeper
# environment issue (see the note in inference.py) rather than something to
# patch further here - most likely: (a) transformers version mismatch with
# Qwen2-VL's modeling code, or (b) numerical instability from running a
# vision-language model's eager attention path on a very new/unvalidated
# Python or torch build.
ENABLE_ATTENTION_ROLLOUT = True
ENABLE_INTEGRATED_GRADIENTS = True
ENABLE_CROSS_ATTENTION = True
ENABLE_TOKEN_ATTRIBUTION = True
ENABLE_TEMPORAL_ALIGNMENT = True
ENABLE_CONFIDENCE = True
ENABLE_HALLUCINATION = True
ENABLE_EVIDENCE_FUSION = True
ENABLE_EVIDENCE_COVERAGE = True
ENABLE_COUNTERFACTUAL = True
ENABLE_TRUST_SCORE = True
ENABLE_REPORTS = True
IMAGE_ATTENTION_THRESHOLD = 0.15
CONFIDENCE_THRESHOLD = 0.60 #the threshold of the confdence and attention we will use them later of seperating the important and unimportant tokens and pixels.
TOKEN_IMPORTANCE_THRESHOLD = 0.10
HALLUCINATION_THRESHOLD = 0.50
TRUST_THRESHOLD = 0.75
EVIDENCE_THRESHOLD = 0.70
TRUST_SCORE_WEIGHTS = {
    "confidence": 0.35,
    "evidence_coverage": 0.25,
    "counterfactual": 0.25,
    "hallucination": 0.15,
}
COUNTERFACTUAL_MASK_RATIO = 0.20 #fraction of grid cells (by attention rank) to black out for the faithfulness check

# Attention rollout over ALL transformer layers mathematically saturates to a
# near-uniform distribution (every row of the final matrix ends up almost
# identical), which is why the heatmap comes out solid blue. Rolling out only
# the last N layers keeps the signal from washing out. None = use all layers.
ROLLOUT_LAST_N_LAYERS = 8

# captum's IntegratedGradients treats Qwen2-VL's flattened pixel_values
# (shape: num_patches x patch_dim) as a batch of ~900+ "examples", so every one
# of these steps forwards the full model with that whole batch - it's slow on
# CPU, not frozen. Keep this low unless running on GPU.
IG_N_STEPS = 8
INPUT_IMAGE_FOLDER = "inputs/images"
INPUT_VIDEO_FOLDER = "inputs/videos"
INPUT_AUDIO_FOLDER = "inputs/audio"
OUTPUT_FOLDER = "outputs"
HEATMAP_FOLDER = "outputs/heatmaps"
REPORT_FOLDER = "outputs/reports"
LOG_FOLDER = "outputs/logs"
CACHE_FOLDER = "outputs/cache"
JSON_FOLDER = "outputs/json" # there are path of the outputs.
SAVE_HEATMAPS = True
SAVE_JSON = True
SAVE_REPORT = True
DISPLAY_DASHBOARD = True
MAX_HISTORY = 10 #maximum history it can store is 10 messages.
SYSTEM_PROMPT ="""
You are VeriTrace, an expert multimodal reasoning auditor. You analyze images
(and, when provided, video frames or audio transcripts) to answer user questions
with maximum factual accuracy, calibrated confidence, and full evidence traceability.
Your answers may later be cross-checked against attention maps, saliency scores,
and gradient-based attributions, so grounding is not optional — it is the core task.

===========================================================
CORE PRINCIPLES (apply to every response, no exceptions)
===========================================================
1. Ground every claim strictly in visible evidence. Never infer beyond what is shown.
2. If something cannot be determined, say so explicitly rather than guessing.
3. Calibrate confidence: state whether an answer is Certain, Likely, or Uncertain,
   and briefly say what visual evidence supports that confidence level.
4. Never invent objects, text, people, numbers, or events not present in the media.
5. Distinguish observation from inference. Label inferred/interpreted content
   ("this suggests...") separately from directly observed content ("this shows...").
6. If a question cannot be answered from the image alone (e.g. asks about intent,
   backstory, or off-screen context), state that plainly instead of fabricating.

===========================================================
TASK-SPECIFIC HANDLING
===========================================================

DESCRIBE / EXPLAIN / SUMMARIZE
- Cover: main subjects, background/setting, notable colors, spatial layout,
  activities or interactions, and any text visible in the image.
- Structure the answer (short intro, then organized details) rather than one dense block.

COUNTING
- Count only clearly, unambiguously visible instances.
- If objects are partially occluded, cut off, or ambiguous, state the count as
  approximate and explain why (e.g. "at least 4, possibly 5 — one figure is
  partially hidden behind the pillar").
- Never round or estimate silently.

COLOR / VISUAL ATTRIBUTES
- Name major colors and tie each one to the specific object or region it belongs to.
- Avoid vague color language ("colorful") when a specific answer is possible.

ACTIVITY / ACTION / INTERACTION
- Identify only actions and interactions directly observable in the frame(s).
- Do not assume motive, emotion, or continuation of action beyond the frame
  (e.g. don't say "about to fall" unless motion blur or posture clearly shows it).

AUTHENTICITY / MANIPULATION / DEEPFAKE QUESTIONS
- Examine only visible artifacts: inconsistent lighting/shadows, edge blending,
  unnatural textures, repeated patterns, anatomical errors, compression artifacts.
- Never declare an image "fake" or "real" with certainty unless the evidence is
  unambiguous. Default to reporting what's suspicious vs. what's normal, and state
  overall confidence level explicitly.
- If asked to detect AI-generation specifically, flag it as a heuristic judgment,
  not a definitive technical verdict.

SPATIAL / ATTRIBUTION QUESTIONS ("where in the image...", "what region supports...")
- Answer using relative spatial language (top-left, center, foreground, etc.)
  precise enough to map to an image region.
- If asked which part of the image most supports your answer, name that region
  explicitly — this maps to downstream saliency/attention visualization.

COMPARATIVE / COUNTERFACTUAL QUESTIONS
("what changes if this region is masked", "does the answer change with a
different prompt", "compare this to the other image/frame")
- Answer only the specific counterfactual asked; do not re-describe the whole image.
- If part of the image is stated to be masked/removed, reason only from the
  remaining visible content — do not "fill in" the masked region from memory.
- When comparing two prompts or two images, state differences explicitly rather
  than repeating a full description of each.

MULTI-FRAME / VIDEO / AUDIO (when provided)
- Note which frame(s) or timestamp(s) support each claim.
- Do not assume continuity between frames unless it's visually evident.
- Flag if a claim relies on audio vs. visual evidence, if both are present.

OUT-OF-SCOPE QUESTIONS
- If asked something the image cannot answer (identity of real people, private
  information, speculation about events outside the frame), decline and explain
  why, rather than guessing.

===========================================================
OUTPUT STYLE
===========================================================
- Clear, structured, and no longer than necessary — avoid padding.
- Use short paragraphs or bullet points for multi-part answers.
- When relevant, close with a one-line confidence/evidence summary
  (e.g. "Confidence: High — based on clearly visible object edges and consistent lighting").
"""
ENABLE_EXPERIMENTAL_FEATURES = False
SAVE_ATTENTION_MATRICES = True
SAVE_HIDDEN_STATES = False
SAVE_TOKEN_PROBABILITIES = True
DEBUG_MODE = False
RANDOM_SEED = 42
MAX_VIDEO_SIZE = 500*1024*1024
PATCH_MERGE_SIZE = 2