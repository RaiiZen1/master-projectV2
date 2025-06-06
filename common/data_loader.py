import numpy as np
import openml
import os
import pandas as pd
import pickle
from common.preprocessing import encode_target
from common.utils import setup_logging


def load_dataset(dataset_id: int, config: dict):
    """
    Loads an OpenML dataset based on its ID and processes it according to the task type.

    This function fetches the dataset using the _fetch_dataset helper, then processes
    the target variable based on whether the task is binary classification or regression.

    Parameters:
        dataset_id (int): The ID of the dataset on OpenML.
        config (dict): Configuration dictionary containing data paths and settings.

    Returns:
        tuple: A tuple containing:
            - X: The feature data (pandas DataFrame)
            - y: The processed target variable (encoded for binary classification or numpy array for regression)
            - cat_cols: List of booleans indicating which features are categorical
            - attribute_names: List of attribute names
            - task_type: String indicating the task type ('binary' or 'regression')
    """
    X, y, cat_cols, attribute_names, task_type = fetch_dataset(
        dataset_id=dataset_id,
        cache_dir=config["data"]["cache_dir_path"],
    )
    if task_type == "binary":
        # Encode the target variable if it's binary
        y = encode_target(y)
    else:
        # Transform to np.array for regression
        y = np.array(y, dtype=float)

    return X, y, cat_cols, attribute_names, task_type


def fetch_dataset(dataset_id: int, cache_dir: str = "data/cache/"):
    """
    Downloads an OpenML dataset based on its id, caches the data locally, and returns the data
    along with its metadata.

    The five returned objects are:
        - X: The feature data (typically a pandas DataFrame).
        - y: The target variable.
        - categorical_indicator: A list of booleans indicating which features are categorical.
        - attribute_names: A list of attribute names.
        - task_type: A string indicating the type of task ('binary' or 'regression').

    If the dataset has been previously downloaded and stored in the cache directory,
    it will be loaded from the local file instead of re-downloading.

    Parameters:
        dataset_id (int): The id of the dataset on OpenML.
        cache_dir (str): The directory to store the downloaded dataset. Defaults to "openml_cache".

    Returns:
        X, y, categorical_indicator, attribute_names, task_type
    """
    logger = setup_logging()

    # Ensure the cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Define a cache file name that is unique to the dataset id
    cache_file = os.path.join(cache_dir, f"openml_dataset_{dataset_id}.pkl")

    # If the cache file exists, load the data from the file.
    if os.path.exists(cache_file):
        logger.info(f"Loading dataset {dataset_id} from cache at '{cache_file}'...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        X = data["X"]
        y = data["y"]
        categorical_indicator = data["categorical_indicator"]
        attribute_names = data["attribute_names"]
        task_type = data["task_type"]
    else:
        # Download the dataset from OpenML.
        logger.info(f"Downloading dataset {dataset_id} from OpenML...")
        dataset = openml.datasets.get_dataset(dataset_id)

        # Use the default target attribute (if defined) when retrieving the data.
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )

        # Determine the task type (binary classification or regression)
        task_type = _determine_task_type(y)

        # Store the data and metadata in a dictionary.
        data = {
            "X": X,
            "y": y,
            "categorical_indicator": categorical_indicator,
            "attribute_names": attribute_names,
            "task_type": task_type,
        }

        # Save the dictionary to a local file using pickle.
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Dataset {dataset_id} stored locally at '{cache_file}'.")

    logger.info(f"Dataset {dataset_id} loaded successfully with task type: {task_type}")

    return X, y, categorical_indicator, attribute_names, task_type


def _determine_task_type(y):
    """
    Determines if the target variable is for binary classification, or regression.

    Parameters:
        y: The target variable (pandas dataframe).

    Returns:
        str: 'binary', or 'regression'
    """

    # Get unique values
    unique_values = pd.unique(y)

    # Check if it's binary (2 unique values)
    if len(unique_values) == 2:
        return "binary"
    else:
        return "regression"
