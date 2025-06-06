import numpy as np
import pandas as pd
from common.utils import setup_logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    TargetEncoder,
)


def encode_target(y):
    """
    Encode the target variable using LabelEncoder.

    Parameters
    ----------
    y : np.array
        The target variable to encode.

    Returns
    -------
    y_encoded : np.array
        The encoded target variable.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded


def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    cat_cols: list,
    config: dict,
    preprocessing_type: str,
) -> tuple:
    """
    Preprocess the training and validation datasets based on the specified model type.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training dataset features.
    y_train : pandas.Series or np.array
        The target variable for the training dataset.
    X_val : pandas.DataFrame
        The validation dataset features.
    cat_cols : list or array of bool
        Boolean mask indicating which columns in X_train and X_val are categorical (True).
    config : dict
        Configuration dictionary containing model and preprocessing details.

    Returns
    -------
    X_train : np.array
        The preprocessed training dataset features.
    X_val : np.array
        The preprocessed validation dataset features.
    """
    X_train, preprocessor_inner = _minimal_preprocess_train(
        X_train,
        y_train,
        cat_cols,
        preprocessing_type,
        config,
    )
    X_val = _minimal_preprocess_test(X_val, preprocessor_inner)

    return X_train, X_val


def _minimal_preprocess_train(X, y, categorical_features, model_type, config):
    """
    Perform minimal preprocessing on the input features for training.

    This function applies preprocessing to both numeric and categorical features
    based on the model type specified in the configuration. For Neural Networks,
    it standardizes numeric features and one-hot encodes categorical features. For
    tree-based models, it does not scale numeric features and applies different
    encoding strategies for low- and high-cardinality categorical features.

    Parameters
    ----------
    X : pandas.DataFrame
        The input features.

    categorical_features : list or array of bool
        Boolean mask indicating which columns in X are categorical (True) and
        which are numeric (False).

    Returns
    -------
    X_preprocessed : np.array
        The preprocessed features.
    preprocessor : ColumnTransformer
        The fitted preprocessor, which can be used to transform future datasets.
    """

    # Convert the boolean mask to a NumPy array (if not already) for fast indexing.
    cat_mask = np.asarray(categorical_features)

    # Check that the mask length matches the number of columns in X.
    if cat_mask.shape[0] != X.shape[1]:
        raise ValueError(
            "Length of categorical_features mask must match the number of columns in X."
        )

    # Extract column names using boolean indexing.
    categorical_feature_names = X.columns[cat_mask].tolist()
    numeric_feature_names = X.columns[~cat_mask].tolist()

    # Select pipeline based on config.
    logger = setup_logging()

    # Neural Networks: standardize numeric features and one-hot encode categoricals.
    if model_type == "nn":

        # Define the preprocessing pipeline for numeric features.
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        # Define the preprocessing pipeline for categorical features.
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers = [
            ("num", numeric_pipeline, numeric_feature_names),
            ("cat", categorical_pipeline, categorical_feature_names),
        ]

        logger.info("Preprocessing pipeline for Neural Network model.")

    elif model_type == "tree":

        # Tree-based models: leave numeric features unscaled, and differentiate encoding for categoricals.
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
            ]
        )
        # Split categorical features into low- and high-cardinality groups.
        threshold = config["preprocessing"]["threshold_high_cardinality"]
        low_card_features = []
        high_card_features = []
        for feature in categorical_feature_names:
            if X[feature].nunique() <= threshold:
                low_card_features.append(feature)
            else:
                high_card_features.append(feature)

        # For low-cardinality categoricals, use one-hot encoding.
        low_card_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                ),
            ]
        )

        # For high-cardinality categoricals, use target encoding.
        high_card_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("target_encode", TargetEncoder()),
            ]
        )

        transformers = [
            ("num", numeric_pipeline, numeric_feature_names),
            ("low_card", low_card_pipeline, low_card_features),
            ("high_card", high_card_pipeline, high_card_features),
        ]

        logger.info("Preprocessing pipeline for Decision Tree model.")

    elif model_type == "grande":
        logger.info("No preprocessing required for Grande model.")
        return X, None

    else:
        raise ValueError(
            "Unknown teacher type. Must be 'nn' for Neural Network, 'tree' for Decision Tree, or 'grande' for Gradient Based Tree."
        )

    # Create a ColumnTransformer to apply the transformations to the appropriate columns.
    preprocessor = ColumnTransformer(
        transformers=transformers,
        n_jobs=-2,
    )

    # Fit the preprocessor on the training data and transform it.
    X_preprocessed = preprocessor.fit_transform(X, y)

    return X_preprocessed, preprocessor


def _minimal_preprocess_test(X, preprocessor):
    """
    Transform the input features according to the preprocessor fitted on the training data.

    Parameters
    ----------
    X : pandas.DataFrame
        The input features.
    preprocessor : ColumnTransformer
        The preprocessor fitted on the training data.

    Returns
    -------
    X_preprocessed : np.array
        The preprocessed features.
    """
    if preprocessor is None:
        return X
    return preprocessor.transform(X)
