from pathlib import Path
from config import SUPPORTED_VIDEO_FORMATS, MAX_VIDEO_SIZE, DEVICE, FRAME_SAMPLING_RATE
from qwen_vl_utils import process_vision_info
from conversation import build_convo
from video_keyframes import extract_uniform_frames


def validate_video(video_path):
    """
    Validate video file.
    """

    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"{video_path} does not exist.")

    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        raise ValueError("Unsupported video format.")

    size = path.stat().st_size

    if size > MAX_VIDEO_SIZE:
        raise ValueError("Video file is too large.")

    return video_path


def build_message(frame_images, question, fps=FRAME_SAMPLING_RATE):
    """
    Build conversation message.

    NOTE: passes a LIST of already-decoded frame images, not a video file
    path. Handing qwen_vl_utils a raw path makes it try to decode the video
    itself via torchvision/decord/torchcodec, whichever it auto-detects -
    on this project's environment that auto-detection landed on
    torchvision.io.read_video, an API that doesn't exist in the installed
    torchvision version. Frame lists skip that video-decoding code path
    entirely since the frames are already images by the time qwen_vl_utils
    sees them.
    """

    return build_convo("video", question, frame_images, extra_content={"fps": fps})


def create_chat(message, processor):
    """
    Create chat template.
    """

    chat = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    return chat


def process_inputs(chat, message, processor):
    """
    Convert video + text into model inputs.
    """

    image_inputs, video_inputs = process_vision_info(message)

    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs


def move_to_device(inputs):
    """
    Move tensors to CPU/GPU.
    """

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    return inputs


def preprocess_video(video_path, question, processor):
    """
    Complete video preprocessing pipeline.
    """

    video_path = validate_video(video_path)

    sampled_frames = extract_uniform_frames(video_path, sampling_rate=FRAME_SAMPLING_RATE)
    frame_images = [f["image"] for f in sampled_frames]

    message = build_message(frame_images, question, fps=FRAME_SAMPLING_RATE)

    chat = create_chat(message, processor)

    inputs = process_inputs(chat, message, processor)

    inputs = move_to_device(inputs)

    return {
        "video": video_path,
        "message": message,
        "chat": chat,
        "inputs": inputs,
    }