import json
import logging
import numpy as np
import random
import tensorflow as tf
import torch


def load_config(config_path="config/config.json") -> dict:
    """
    Loads a configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML file containing the configuration. Defaults to "config/default_config.yaml".

    Returns:
        dict: The configuration as a dictionary.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, PyTorch, and TensorFlow.

    Args:
        seed: Integer seed for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


def setup_logging() -> logging.Logger:
    """
    Set up logging configuration.
    This function configures the logging settings for the application, including
    the logging level and format. It returns a logger instance that can be used
    throughout the application.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    return logger


def check_GPU_availability():
    """
    Checks if a GPU is available and configures TensorFlow to use it appropriately.

    Sets up memory growth for TensorFlow GPU usage to avoid allocating all GPU memory at once.

    Returns:
        torch.device: A torch device object ('cuda' if GPU is available, 'cpu' otherwise)
    """
    logger = setup_logging()
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        device = torch.device("cuda")
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training.")
    return device
