import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration, AutoProcessor
#these all are imported from the config file or else we ll get error of undefined variable.
try:
    # try top-level import (when running as a script)
    from config import (
        MODEL_NAME,
        DEVICE,
        DTYPE,
        MIN_PIXELS,
        MAX_PIXELS,
        ATTENTION_IMPLEMENTATION,
        MAX_NEW_TOKENS,
        DO_SAMPLE,
    )
except Exception:
    try:
        # try package-relative import (when used as a module)
        from .config import (
            MODEL_NAME,
            DEVICE,
            DTYPE,
            MIN_PIXELS,
            MAX_PIXELS,
            ATTENTION_IMPLEMENTATION,
            MAX_NEW_TOKENS,
            DO_SAMPLE,
        )
    except Exception as e:
        raise ImportError(
            "Could not import config module. Ensure a config.py is available in the project or package. "
            f"Original error: {e}"
        )      

def load_processor():
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return processor

def load_model():

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        attn_implementation=ATTENTION_IMPLEMENTATION, #this means the transformer looks different part of image and text and decides the most important part before generating new token.
        torch_dtype=DTYPE #in this data type we get back and answer , we ll send to the device either cpu or gpu
    ).to(DEVICE)

    model.eval()

    return model


def prepare_inputs(processor, message):

    text_chat = processor.apply_chat_template( #this converts the message into a chat format that the qwen understand before tokenization.
        message,
        tokenize=False, #here it should not be tokenised its just conversion to chat format.
        add_generation_prompt=True  # Adds an empty assistant prompt at the end. This tells the model: "Now it's your turn to generate the answer."
    )
    image_inputs,video_inputs= process_vision_info(message)
    inputs = processor(
        images=image_inputs, 
        videos=video_inputs,      #this is where the tokens are made and the image is converted into a tensor and sent to device.
        text=[text_chat],
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    return inputs


def _set_attn_impl(model, impl):
    """
    model.set_attn_implementation(impl) only exists on newer transformers
    (5.x+). Older versions - including 4.49.0, the version confirmed to work
    correctly with Qwen2-VL's eager attention path - require setting the
    config attribute directly instead. Support both so this project runs on
    either.
    """
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(impl)
        return

    model.config._attn_implementation = impl
    # Qwen2-VL has separate vision/text sub-configs on some transformers
    # versions; mirror the setting onto them too if present, otherwise the
    # switch can silently apply to only one half of the model.
    for sub_config_name in ("vision_config", "text_config"):
        sub_config = getattr(model.config, sub_config_name, None)
        if sub_config is not None:
            sub_config._attn_implementation = impl


def extract_attention(model, inputs):
    # NOTE: this eager forward pass exists ONLY to read output_attentions
    # (sdpa can't return attention weights). If layer 0 comes back mostly
    # intact but every later layer is 100% NaN, that's the hidden-state
    # stream itself overflowing inside eager attention on this specific
    # transformers/torch/Python build - not a bug in this project's code.
    # See config.py's ATTENTION_IMPLEMENTATION comment for what to check.
    outputs = None

    try:

        _set_attn_impl(model, "eager")

        with torch.inference_mode(): #this doesnot calculate the gradients and speeds up the inference.

            outputs = model(
                **inputs,
                return_dict=True,
                output_attentions=True,
                output_hidden_states = True
            )
        
        #this outputs contain the attentions,logits,hidden states and past key values.
    except RuntimeError as e:

        print(f"Attention maps skipped: {e}")

        outputs = None

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    finally:

        _set_attn_impl(model, ATTENTION_IMPLEMENTATION)

    return outputs


def generate_answer(model, inputs):

    generated = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS, #this place the answer is generated but u cant read in generated we have the confdence attentions and those all and we convert later.
        do_sample=DO_SAMPLE,
        output_scores=True,
        return_dict_in_generate=True
    )

    return generated


def decode_response(processor, generated, inputs):

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs["input_ids"],
            generated.sequences        #this is where the answer is generated into human readble format.
        )
    ]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return response, generated_ids


def run_inference(processor, model, message):

    inputs = prepare_inputs(
        processor,
        message
    )

    outputs = extract_attention(
        model,
        inputs
    )
                                                  #calling all the functions to get the text answer and otheer required for the further processes like attentions, logits etc..
    generated = generate_answer(
        model,
        inputs
    )

    response, generated_ids = decode_response(
        processor,
        generated,
        inputs
    )

    return {
        "inputs": inputs,
        "outputs": outputs,
        "generated": generated,
        "generated_ids": generated_ids,
        "response": response
    }