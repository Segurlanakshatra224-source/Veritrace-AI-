from PIL import Image
from config import MAX_PIXELS, MIN_PIXELS, DEVICE
from qwen_vl_utils import process_vision_info
from conversation import build_convo


def load_image(image_path):
    return Image.open(image_path)


def convert_image(image):
    return image.convert("RGB")


def validate_image_size(image):
    width, height = image.size
    pixels = width * height

    if pixels > MAX_PIXELS:
        print("Image is large. Resizing...")
        image.thumbnail((1024, 1024))

    if image.width * image.height < MIN_PIXELS:
        raise ValueError("Image is too small")

    return image


def build_message(image, question):
    return build_convo("image", question, image)


def create_chat(message, processor):
    chat = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    return chat


def process_inputs(chat, message, processor):
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
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    return inputs


def preprocess_image(image_path, question, processor):

    image = load_image(image_path)

    image = convert_image(image)

    image = validate_image_size(image)

    message = build_message(image, question)

    chat = create_chat(message, processor)

    inputs = process_inputs(chat, message, processor)

    inputs = move_to_device(inputs)

    return {
        "image": image,
        "message": message,
        "chat": chat,
        "inputs": inputs,
    }