import os
from config import (
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
)
from image_preprocessing import preprocess_image
from video_preprocessing import preprocess_video
from audio_preprocessing import preprocess_audio #we will have these three in those respective files.
def detect_input_type(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension in SUPPORTED_AUDIO_FORMATS :
        return "audio"
    elif extension in SUPPORTED_IMAGE_FORMATS :
        return "image"
    elif extension in SUPPORTED_VIDEO_FORMATS : 
        return "video"
    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )
def validate_input(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_path} does not exist."
        )

    return True
def dispatch_preprocessing(file_path, input_type, question, processor):
#from here we will send the file to respective processors
    if input_type == "image":
        return preprocess_image(file_path, question, processor)

    elif input_type == "video":
        return preprocess_video(file_path, question, processor)

    elif input_type == "audio":
        return preprocess_audio(file_path, question, processor)

    else:
        raise ValueError(
            "Unknown processor."
        )
def preprocess(file_path, question, processor):
    validate_input(file_path)

    input_type = detect_input_type(file_path)

    processed_data = dispatch_preprocessing(
        file_path,
        input_type,
        question,
        processor
    )

    return {
        "modality": input_type,
        "data": processed_data,
    }
"""
what exactly happens here is the data is send to any of the three files and there it is 
preprocessed and we get back them and here we get the data.

"""