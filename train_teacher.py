import json
import numpy as np
import optuna
import os
import pandas as pd
import time
from common.data_loader import load_dataset
from common.hpo import suggest_hyperparameters
from common.preprocessing import preprocess
from common.teacher_model_factory import get_teacher_model
from common.utils import setup_logging, load_config, check_GPU_availability, set_seed
from sklearn.model_selection import StratifiedKFold, KFold


def train_teacher(dataset_id=None, config=None):
    # =============================================================================
    # MAIN TRAINING LOOP - Process each dataset independently
    # =============================================================================
    logger = setup_logging()
    logger.info(f"Starting training for dataset {dataset_id}...")

    # -------------------------------------------------------------------------
    # SETUP AND INITIALIZATION
    # -------------------------------------------------------------------------
    logger.info("Loading configuration...")

    # Extract model configuration for this run
    model_type = config["model"]["teacher_model"]
    preprocessing_type = config["teacher_models"][model_type]["preprocessing"]
    use_hpo = config["training"]["use_hpo"]

    # -------------------------------------------------------------------------
    # STEP 0: DATA LOADING
    # -------------------------------------------------------------------------
    # Load dataset from OpenML with caching for efficiency
    X, y, cat_cols, _, task_type = load_dataset(
        dataset_id=dataset_id,
        config=config,
    )

    # -------------------------------------------------------------------------
    # STEP 1: INFRASTRUCTURE SETUP
    # -------------------------------------------------------------------------
    # Note: Checking for existing results is placeholder for future implementation
    summary_file = os.path.join(
        config["data"]["results_dir_path"], "teacher", f"{dataset_id}_results.json"
    )
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            existing_results = json.load(f)

        # Check if we already have results with the same configuration
        config_exists = False
        for key, result in existing_results.items():
            if (
                result.get("model_type") == model_type
                and result.get("use_hpo") == use_hpo
                and result.get("seed") == config["training"]["random_state"]
            ):
                config_exists = True
                logger.info(
                    f"Results already exist for dataset {dataset_id} with model {model_type}, HPO: {use_hpo}, seed: {config['training']['random_state']}"
                )
                break

        if config_exists:
            logger.info(f"Skipping dataset {dataset_id} - results already computed")
            return

    # Configure GPU/CPU usage for training
    device = check_GPU_availability()

    # Set random seed for reproducibility across all libraries
    set_seed(config["training"]["random_state"])
    logger.info(f"Random seed set to {config['training']['random_state']}")

    # -------------------------------------------------------------------------
    # STEP 2: INITIALIZE DATA STRUCTURES
    # -------------------------------------------------------------------------
    # List to store predictions from each outer fold
    output_dfs = []

    outer_fold_scores = []

    # Dictionary to store all fold indices for reproducibility and student training
    # Structure: {"outer_folds": {fold_id: {train_idx, test_idx}},
    #            "inner_folds": {outer_fold_id: {inner_fold_id: {train_idx, val_idx}}}}
    fold_indices = {"outer_folds": {}, "inner_folds": {}}

    # -------------------------------------------------------------------------
    # STEP 3: OUTER CROSS-VALIDATION SETUP
    # -------------------------------------------------------------------------
    # Choose appropriate CV strategy based on task type to maintain class balance
    if task_type == "binary":
        outer_cv = StratifiedKFold(
            n_splits=config["training"]["outer_folds"],
            shuffle=True,
            random_state=config["training"]["random_state"],
        )
    else:
        outer_cv = KFold(
            n_splits=config["training"]["outer_folds"],
            shuffle=True,
            random_state=config["training"]["random_state"],
        )

    # =========================================================================
    # STEP 4: OUTER CROSS-VALIDATION LOOP
    # =========================================================================
    # Each iteration provides one unbiased performance estimate
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):

        # Log the first 10 indices of the outer fold for debugging
        logger.info(
            f"Outer Fold {outer_fold} - Train Indices: {train_idx[:10]}, Test Indices: {test_idx[:10]}"
        )

        logger.info(
            f"-------------------- Outer Fold {outer_fold} --------------------"
        )

        # ---------------------------------------------------------------------
        # FOLD INDEX MANAGEMENT
        # ---------------------------------------------------------------------
        # Store outer fold indices for later use in student training and validation
        fold_indices["outer_folds"][f"fold_{outer_fold}"] = {
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }

        # Split data according to current outer fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ---------------------------------------------------------------------
        # INNER CROSS-VALIDATION SETUP (for model validation)
        # ---------------------------------------------------------------------
        # Choose appropriate CV strategy for inner folds
        if task_type == "binary":
            inner_cv = StratifiedKFold(
                n_splits=config["training"]["inner_folds"],
                shuffle=True,
                random_state=config["training"]["random_state"],
            )
        else:
            inner_cv = KFold(
                n_splits=config["training"]["inner_folds"],
                shuffle=True,
                random_state=config["training"]["random_state"],
            )

        # Initialize storage for inner fold indices within this outer fold
        fold_indices["inner_folds"][f"outer_fold_{outer_fold}"] = {}

        # =====================================================================
        # STEP 5: INNER CROSS-VALIDATION LOOP (hyperparameter validation)
        # =====================================================================
        def objective(trial):

            hyperparams = suggest_hyperparameters(
                trial=trial,
                model_type=model_type,
                task_type=task_type,
                use_hpo=use_hpo,
            )

            inner_fold_scores = []
            val_metrics_list = []

            # This loop would typically be used for hyperparameter optimization
            for inner_fold, (inner_train_index, inner_val_index) in enumerate(
                inner_cv.split(X_train, y_train), start=1
            ):

                # -----------------------------------------------------------------
                # INDEX MANAGEMENT (Critical for avoiding data leakage)
                # -----------------------------------------------------------------
                # Convert relative indices (within outer training set) to absolute indices
                absolute_inner_train_idx = train_idx[inner_train_index]
                absolute_inner_val_idx = train_idx[inner_val_index]

                # Store inner fold indices using absolute indices for consistency
                # Only store indices for the first trial
                if trial.number == 0:
                    fold_indices["inner_folds"][f"outer_fold_{outer_fold}"][
                        f"inner_fold_{inner_fold}"
                    ] = {
                        "train_idx": absolute_inner_train_idx.tolist(),
                        "val_idx": absolute_inner_val_idx.tolist(),
                    }

                # Split inner training data using relative indices
                X_inner_train, X_inner_val = (
                    X_train.iloc[inner_train_index],
                    X_train.iloc[inner_val_index],
                )
                y_inner_train, y_inner_val = (
                    y_train[inner_train_index],
                    y_train[inner_val_index],
                )

                # -----------------------------------------------------------------
                # PREPROCESSING: Apply model-specific data transformations
                # -----------------------------------------------------------------
                X_inner_train, X_inner_val = preprocess(
                    X_inner_train,
                    y_inner_train,
                    X_inner_val,
                    cat_cols,
                    config,
                    preprocessing_type=preprocessing_type,
                )

                # -----------------------------------------------------------------
                # MODEL TRAINING: Train teacher model on inner training data
                # -----------------------------------------------------------------
                model = get_teacher_model(
                    config=config,
                    task_type=task_type,
                    device=device,
                    hyperparams=hyperparams,
                    cat_cols=cat_cols,
                )
                logger.info(
                    f"Training Model on Outer Fold {outer_fold}, Inner Fold {inner_fold}..."
                )
                model.train(X_inner_train, y_inner_train)

                # -----------------------------------------------------------------
                # VALIDATION: Evaluate model performance on inner validation set
                # -----------------------------------------------------------------
                val_preds = model.predict(X_inner_val)
                val_metrics = model.evaluate(val_preds, y_inner_val)

                # Store metrics from this fold for later mean calculation
                val_metrics_list.append(val_metrics)

                if task_type == "binary":
                    inner_fold_scores.append(val_metrics["f1"])
                    trial.report(np.mean(inner_fold_scores), step=inner_fold)
                else:
                    inner_fold_scores.append(-val_metrics["mae"])
                    trial.report(np.mean(inner_fold_scores), step=inner_fold)

                # Check if the trial should be pruned (With 2 inner folds, pruning not recommended)
                # if trial.should_prune():
                #     logger.info("Trial pruned.")
                #     raise optuna.TrialPruned()

            # Calculate and set mean metrics as user attributes after all inner folds
            if val_metrics_list:
                # Get all metric keys from the first fold
                metric_keys = val_metrics_list[0].keys()

                for metric_key in metric_keys:
                    # Calculate mean across all inner folds for this metric
                    metric_values = [
                        fold_metrics[metric_key] for fold_metrics in val_metrics_list
                    ]
                    mean_metric = np.mean(metric_values)
                    trial.set_user_attr(f"mean_{metric_key}", mean_metric)

            return np.mean(inner_fold_scores)

        logger.info(
            f"Starting hyperparameter optimization for outer fold {outer_fold}..."
        )

        study_kwargs = dict(
            direction="maximize",
            study_name=f"{dataset_id}.{outer_fold}.{model_type}",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=config["training"]["random_state"]),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        )

        if use_hpo:
            os.makedirs(config["data"]["optuna_db_path"], exist_ok=True)
            study_kwargs["storage"] = (
                f"sqlite:///{config['data']['optuna_db_path']}/optuna.db"
            )

        study = optuna.create_study(
            **study_kwargs,
        )

        # Optimize
        completed_trials = len(study.trials)
        remaining_trials = config["training"]["trials"] - completed_trials

        if remaining_trials > 0:
            default_hyperparams = suggest_hyperparameters(
                None, model_type, task_type, False
            )
            study.enqueue_trial(default_hyperparams)
            study.optimize(
                objective,
                n_trials=remaining_trials if use_hpo else 1,
                show_progress_bar=False,
            )
        else:
            logger.info("The study has already reached the maximum number of trials.")

        if use_hpo:
            # When HPO is used, trial.suggest_ methods are called, populating trial.params
            best_hyperparams = study.best_trial.params
        else:
            # When not using HPO, a single trial runs with default parameters.
            best_hyperparams = suggest_hyperparameters(
                trial=None, model_type=model_type, task_type=task_type, use_hpo=False
            )

        logger.info(f"Best hyperparameters for fold {outer_fold}: {best_hyperparams}")
        logger.info(f"Best score: {study.best_value:.4f}")

        # =====================================================================
        # STEP 6: FINAL MODEL TRAINING (on complete outer training set)
        # =====================================================================
        # Apply same preprocessing to outer training and test sets
        X_train, X_test = preprocess(
            X_train,
            y_train,
            X_test,
            cat_cols,
            config,
            preprocessing_type=preprocessing_type,
        )

        # Train final model on complete outer training set
        logger.info("------------------------------------------------------")
        logger.info(f"Retraining Model on Outer Fold {outer_fold}")

        model = get_teacher_model(
            config=config,
            task_type=task_type,
            device=device,
            hyperparams=best_hyperparams,  # Use best hyperparameters from HPO
            cat_cols=cat_cols,
        )
        model.train(X_train, y_train)

        # =====================================================================
        # STEP 7: FINAL EVALUATION (unbiased performance on outer test set)
        # =====================================================================
        # This provides the unbiased performance estimate for this fold
        start_time = time.time()
        test_preds = model.predict(X_test)
        end_time = time.time() - start_time
        logger.info(f"\t Inference Time: {end_time:.5f} seconds")
        test_metrics = model.evaluate(test_preds, y_test)

        # Store outer fold score for later analysis
        outer_fold_results = {
            "fold": outer_fold,
            "seed": config["training"]["random_state"],
            "inference_time": end_time,
            **test_metrics,
        }
        outer_fold_scores.append(outer_fold_results)

        # =====================================================================
        # STEP 8: STORE PREDICTIONS FOR STUDENT TRAINING
        # =====================================================================
        # Save predictions with their corresponding dataset indices
        # These will be used as targets for training student models
        output_dfs.append(
            pd.DataFrame(
                {
                    "index": test_idx,
                    "output": test_preds[:, 1] if task_type == "binary" else test_preds,
                }
            )
        )

    # =========================================================================
    # STEP 9: SAVE RESULTS AND METADATA
    # =========================================================================

    # -------------------------------------------------------------------------
    # SAVE OUTER FOLD METRICS
    # -------------------------------------------------------------------------
    outer_fold_df = pd.DataFrame(outer_fold_scores)
    sub_folder = "hpo" if use_hpo else "default"
    output_dir = os.path.join(config["data"]["outer_folds_path"], "teacher", sub_folder)
    os.makedirs(output_dir, exist_ok=True)

    metrics_file = os.path.join(output_dir, f"{dataset_id}_{model_type}.csv")
    outer_fold_df.to_csv(metrics_file, index=False)
    logger.info(f"Outer fold metrics saved to: {metrics_file}")

    # Calculate and log overall performance across all folds
    mean_inference_time = outer_fold_df["inference_time"].mean()
    std_inference_time = outer_fold_df["inference_time"].std()
    mean_parameters = outer_fold_df["parameters"].mean()
    std_parameters = outer_fold_df["parameters"].std()

    if task_type == "binary":
        mean_acc = outer_fold_df["acc"].mean()
        std_acc = outer_fold_df["acc"].std()
        mean_f1 = outer_fold_df["f1"].mean()
        std_f1 = outer_fold_df["f1"].std()
        mean_roc = outer_fold_df["roc"].mean()
        std_roc = outer_fold_df["roc"].std()

        logger.info(f"=== FINAL RESULTS FOR DATASET {dataset_id} ===")
        logger.info(f"Balanced Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"ROC AUC: {mean_roc:.4f} ± {std_roc:.4f}")
    else:
        mean_mae = outer_fold_df["mae"].mean()
        std_mae = outer_fold_df["mae"].std()
        mean_mse = outer_fold_df["mse"].mean()
        std_mse = outer_fold_df["mse"].std()
        mean_rmse = outer_fold_df["rmse"].mean()
        std_rmse = outer_fold_df["rmse"].std()
        mean_r2 = outer_fold_df["r2"].mean()
        std_r2 = outer_fold_df["r2"].std()

        logger.info(f"=== FINAL RESULTS FOR DATASET {dataset_id} ===")
        logger.info(f"MAE: {mean_mae:.4f} ± {std_mae:.4f}")
        logger.info(f"MSE: {mean_mse:.4f} ± {std_mse:.4f}")
        logger.info(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        logger.info(f"R2: {mean_r2:.4f} ± {std_r2:.4f}")

    logger.info(
        f"Mean Inference Time: {mean_inference_time:.4f} ± {std_inference_time:.4f}"
    )
    logger.info(f"Mean Parameters: {mean_parameters:.4f} ± {std_parameters:.4f}")

    # Save summary statistics as well
    summary_stats = {
        "dataset_id": dataset_id,
        "model_type": model_type,
        "task_type": task_type,
        "seed": config["training"]["random_state"],
        "use_hpo": config["training"]["use_hpo"],
        "mean_inference_time": mean_inference_time,
        "std_inference_time": std_inference_time,
        "mean_parameters": mean_parameters,
        "std_parameters": std_parameters,
    }

    if task_type == "binary":
        summary_stats.update(
            {
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_roc": mean_roc,
                "std_roc": std_roc,
            }
        )
    else:
        summary_stats.update(
            {
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "mean_rmse": mean_rmse,
                "std_rmse": std_rmse,
                "mean_r2": mean_r2,
                "std_r2": std_r2,
            }
        )

    # Load existing summary file if it exists, otherwise create new
    summary_file = os.path.join(
        config["data"]["results_dir_path"], "teacher", f"{dataset_id}_results.json"
    )

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Add current model results to the dataset summary
    # Use simple incremental numbering
    next_num = len(all_results) + 1
    model_key = str(next_num)

    # Add current model results to the dataset summary
    all_results[model_key] = summary_stats

    # Save updated summary
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary statistics saved to: {summary_file}")

    # -------------------------------------------------------------------------
    # SAVE FOLD INDICES (for reproducibility and student training)
    # -------------------------------------------------------------------------
    fold_indices_file = os.path.join(
        config["data"]["fold_indices_path"], f"dataset_{dataset_id}.json"
    )
    # Create the directory if it doesn't exist
    os.makedirs(config["data"]["fold_indices_path"], exist_ok=True)
    if not os.path.exists(fold_indices_file):
        with open(fold_indices_file, "w") as f:
            json.dump(fold_indices, f, indent=2)
        logger.info(f"Fold indices saved to: {fold_indices_file}")
    else:
        logger.info(f"Fold indices file already exists: {fold_indices_file}")

    # -------------------------------------------------------------------------
    # SAVE TEACHER PREDICTIONS (targets for student training)
    # -------------------------------------------------------------------------
    # Combine predictions from all outer folds
    if output_dfs:  # Check if we have any DataFrames to concatenate
        output_df = pd.concat(output_dfs, ignore_index=True)
    else:
        output_df = pd.DataFrame(columns=["index", "output"])
    output_df = output_df.sort_values(by="index")

    # Create the full directory structure
    sub_folder = "hpo" if use_hpo else "default"
    output_dir = os.path.join(config["data"]["output_dir_path"], sub_folder, "teacher")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{dataset_id}_{model_type}.csv")
    output_df.to_csv(output_file, index=False)
    logger.info(f"{config['model']['teacher_model']} outputs saved to: {output_file}")


if __name__ == "__main__":

    config = load_config()
    datasets = config["data"]["datasets"]

    for dataset_id in datasets:
        train_teacher(dataset_id=dataset_id, config=config)
