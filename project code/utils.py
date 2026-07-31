import os
import time
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def move_to_device(data, device):
    """
    Move tensors or tensor dictionaries to the specified device.
    """

    if isinstance(data, dict):
        return {k: v.to(device) if hasattr(v, "to") else v
                for k, v in data.items()}

    if hasattr(data, "to"):
        return data.to(device)

    return data


def file_exists(file_path):
    """
    Check whether a file exists.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return True


def validate_extension(file_path, supported_formats):
    """
    Validate the file extension.
    """

    extension = os.path.splitext(file_path)[1].lower()

    if extension not in supported_formats:
        raise ValueError(
            f"Unsupported file format: {extension}\n"
            f"Supported formats: {supported_formats}"
        )

    return True


def get_file_size(file_path):
    """
    Return file size in MB.
    """

    size = os.path.getsize(file_path)

    return size / (1024 * 1024)


def clear_cuda():
    """
    Clear CUDA cache.
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_memory():
    """
    Print GPU memory usage.
    """

    if torch.cuda.is_available():

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        print(f"GPU Allocated : {allocated:.2f} GB")
        print(f"GPU Reserved  : {reserved:.2f} GB")

    else:

        print("CUDA is not available.")


def start_timer():
    """
    Start a timer.
    """

    return time.time()


def stop_timer(start_time):
    """
    Stop the timer and return elapsed time.
    """

    elapsed = time.time() - start_time

    print(f"Execution Time : {elapsed:.2f} seconds")

    return elapsed


def print_header(text):
    """
    Print a formatted section header.
    """

    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)