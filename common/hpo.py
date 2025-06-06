import optuna


def suggest_hyperparameters(
    trial: optuna.Trial, model_type: str, task_type: str, use_hpo: bool
) -> dict:
    """
    Suggest hyperparameters for the given model type using Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for suggesting hyperparameters.
    model_type : str
        Type of model ('tabpfn', 'catboost', etc.).
    task_type : str
        Type of task ('binary', 'regression').

    Returns
    -------
    dict
        Dictionary of suggested hyperparameters.
    """
    hyperparameter = {}

    if model_type == "catboost":
        if use_hpo:
            hyperparameter = {
                "iterations": trial.suggest_int("iterations", 512, 4096),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30),
                "boosting_type": "Plain",
            }
        else:
            hyperparameter = {
                "iterations": 1000,
                "max_depth": 6,
                "l2_leaf_reg": 3,
                "boosting_type": "Plain",
            }
        if task_type == "binary":
            hyperparameter["loss_function"] = "Logloss"
        else:
            hyperparameter["loss_function"] = "RMSE"

    elif model_type == "grande":
        if use_hpo:
            hyperparameter = {
                "depth": trial.suggest_int("depth", 3, 7),
                "n_estimators": trial.suggest_int("n_estimators", 512, 4096),
                "learning_rate_weights": trial.suggest_float(
                    "learning_rate_weights", 0.0001, 0.05, log=True
                ),
                "learning_rate_index": trial.suggest_float(
                    "learning_rate_index", 0.001, 0.2, log=True
                ),
                "learning_rate_values": trial.suggest_float(
                    "learning_rate_values", 0.001, 0.2, log=True
                ),
                "learning_rate_leaf": trial.suggest_float(
                    "learning_rate_leaf", 0.001, 0.2, log=True
                ),
                "optimizer": "adam",
                "cosine_decay_steps": trial.suggest_categorical(
                    "cosine_decay_steps", [0.0, 0.1, 1.0, 100.0, 1000.0]
                ),
                "dropout": trial.suggest_float("dropout", 0.0, 0.75),
                "selected_variables": trial.suggest_float(
                    "selected_variables", 0.0, 1.0
                ),
                "data_subset_fraction": trial.suggest_float(
                    "data_subset_fraction", 0.1, 1.0
                ),
                "focal_loss": False,
                "temperature": 0.0,
                "from_logits": True,
                "use_class_weights": True,
            }
        else:
            hyperparameter = {
                "depth": 5,
                "n_estimators": 2048,
                "learning_rate_weights": 0.005,
                "learning_rate_index": 0.01,
                "learning_rate_values": 0.01,
                "learning_rate_leaf": 0.01,
                "optimizer": "adam",
                "cosine_decay_steps": 0,
                "dropout": 0.0,
                "selected_variables": 0.8,
                "data_subset_fraction": 1.0,
                "focal_loss": False,
                "temperature": 0.0,
                "from_logits": True,
                "use_class_weights": True,
            }
        if task_type == "binary":
            hyperparameter["loss"] = "crossentropy"
        else:
            hyperparameter["loss"] = "mse"

    elif model_type == "tabm":
        if use_hpo:
            hyperparameter = {
                "n_blocks": trial.suggest_int("n_blocks", 1, 5),
                "d_block": trial.suggest_int("d_block", 64, 1024),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.0001, 0.005, log=True
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 0.0001, 0.1, log=True
                ),
            }
        else:
            hyperparameter = {
                "n_blocks": 3,
                "d_block": 512,
                "dropout": 0.1,
                "learning_rate": 0.002,
                "weight_decay": 0.0003,
            }

    elif model_type == "mlp":
        if use_hpo:
            hyperparameter = {
                "n_blocks": trial.suggest_int("n_blocks", 1, 5),
                "d_block": trial.suggest_int("d_block", 64, 1024),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.0001, 0.005, log=True
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 0.0001, 0.1, log=True
                ),
            }
        else:
            hyperparameter = {
                "n_blocks": 3,
                "d_block": 512,
                "dropout": 0.1,
                "learning_rate": 0.002,
                "weight_decay": 0.0003,
            }

    elif model_type == "random_forest":
        if use_hpo:
            hyperparameter = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 4096),
                "max_depth": trial.suggest_int("max_depth", 5, 100),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }
            if task_type == "binary":
                hyperparameter["criterion"] = trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                )
            else:
                hyperparameter["criterion"] = trial.suggest_categorical(
                    "criterion",
                    ["squared_error", "absolute_error", "poisson", "friedman_mse"],
                )
        else:
            hyperparameter = {
                "n_estimators": 1000,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            }
            if task_type == "binary":
                hyperparameter["criterion"] = "gini"
                hyperparameter["max_features"] = "sqrt"
            else:
                hyperparameter["criterion"] = "squared_error"
                hyperparameter["max_features"] = 1.0

    return hyperparameter
