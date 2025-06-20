import argparse
import json
import numpy as np
import optuna
import os
import pandas as pd
import time
from common.data_loader import load_dataset
from common.hpo import suggest_hyperparameters
from common.preprocessing import preprocess
from common.student_model_factory import get_student_model
from common.utils import setup_logging, load_config, check_GPU_availability, set_seed


def train_student(
    dataset_id=None, config=None, student_model_type=None, teacher_model_type=None
):
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
    preprocessing_type = config["student_models"][student_model_type]["preprocessing"]
    use_hpo = config["training"]["use_hpo"]
    train_on_logits = config["training"]["train_on_logits"]

    # -------------------------------------------------------------------------
    # STEP 0: DATA LOADING
    # -------------------------------------------------------------------------
    # Load dataset from OpenML with caching for efficiency
    X, y, cat_cols, _, original_task_type = load_dataset(
        dataset_id=dataset_id,
        config=config,
    )

    # -------------------------------------------------------------------------
    # STEP 1: INFRASTRUCTURE SETUP
    # -------------------------------------------------------------------------
    # Note: Checking for existing results is placeholder for future implementation
    model_task_type = (
        "binary"
        if (original_task_type == "binary" and not train_on_logits)
        else "regression"
    )
    summary_file = os.path.join(
        config["data"]["results_dir_path"], "student", f"{dataset_id}_results.json"
    )
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            existing_results = json.load(f)

        # Check if we already have results with the same configuration
        config_exists = False
        for key, result in existing_results.items():
            if (
                result.get("student_model_type") == student_model_type
                and result.get("teacher_model_type") == teacher_model_type
                and result.get("student_task_type") == model_task_type
                and result.get("use_hpo") == use_hpo
                and result.get("seed") == config["training"]["random_state"]
            ):
                config_exists = True
                logger.info(
                    f"Results already exist for dataset {dataset_id} with model {student_model_type}({teacher_model_type}), HPO: {use_hpo}, seed: {config['training']['random_state']}"
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
    # STEP 2: LOAD PREVIOUSLY SAVED FOLD INDICES AND TEACHER PREDICTIONS
    # -------------------------------------------------------------------------
    # Load the saved fold indices
    # Create the full directory structure
    sub_folder = "hpo" if use_hpo else "default"
    fold_indices_dir = os.path.join(
        config["data"]["fold_indices_path"], sub_folder, "teacher"
    )
    fold_indices_file = os.path.join(
        fold_indices_dir, f"{dataset_id}_{teacher_model_type}.json"
    )

    with open(fold_indices_file, "r") as f:
        fold_indices = json.load(f)
    logger.info(f"Loaded fold indices from: {fold_indices_file}")

    # -------------------------------------------------------------------------
    # STEP 3: INITIALIZE DATA STRUCTURES
    # -------------------------------------------------------------------------
    output_dfs = []

    outer_fold_scores = []

    # =========================================================================
    # STEP 4: OUTER CROSS-VALIDATION LOOP
    # =========================================================================
    # Each iteration provides one unbiased performance estimate
    for fold_key, fold_data in fold_indices["outer_folds"].items():

        # Extract outer fold number and indices from the fold key
        outer_fold = int(fold_key.split("_")[1])
        train_idx = np.array(fold_data["train_idx"])
        test_idx = np.array(fold_data["test_idx"])

        # Log the first 10 indices of the outer fold for debugging
        logger.info(
            f"Outer Fold {outer_fold} - Train Indices: {train_idx[:10]}, Test Indices: {test_idx[:10]}"
        )

        logger.info(
            f"-------------------- Outer Fold {outer_fold} --------------------"
        )

        # Split data according to current outer fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --------------------------------------------------------------------------
        # STEP 5: PREPROCESS TEACHER OUTPUTS
        # --------------------------------------------------------------------------
        if original_task_type == "regression":
            model_task_type = "regression"
            # For regression, we can use the outputs directly as they are already numeric
            teacher_targets_train = np.array(fold_data["train_preds"])
            teacher_targets_test = np.array(fold_data["test_preds"])
        else:
            if train_on_logits:
                model_task_type = "regression"
                # Convert probabilities to logits
                # Clip probabilities to avoid log(0) or log(1)
                eps = 1e-7
                teacher_probs_train = np.clip(fold_data["train_preds"], eps, 1 - eps)
                teacher_probs_test = np.clip(fold_data["test_preds"], eps, 1 - eps)

                teacher_targets_train = np.log(
                    teacher_probs_train / (1 - teacher_probs_train)
                )
                teacher_targets_test = np.log(
                    teacher_probs_test / (1 - teacher_probs_test)
                )
            else:
                model_task_type = "binary"
                # Prepare training targets (hard labels)
                teacher_targets_train = (
                    np.array(fold_data["train_preds"]) > 0.5
                ).astype(int)
                # Prepare test targets (hard labels)
                teacher_targets_test = (np.array(fold_data["test_preds"]) > 0.5).astype(
                    int
                )

        # ---------------------------------------------------------------------
        # INNER CROSS-VALIDATION SETUP (for model validation)
        # ---------------------------------------------------------------------
        inner_folds_data = fold_indices["inner_folds"][f"outer_fold_{outer_fold}"]

        # =========================================================================
        # STEP 6: INNER CROSS-VALIDATION LOOP (hyperparameter validation)
        # =========================================================================
        def objective(trial):

            hyperparams = suggest_hyperparameters(
                trial=trial,
                model_type=student_model_type,
                task_type=model_task_type,
                use_hpo=use_hpo,
            )

            inner_fold_scores = []
            val_metrics_list = []

            # This loop would typically be used for hyperparameter optimization
            for inner_fold_key, inner_fold_data in inner_folds_data.items():
                inner_fold = int(inner_fold_key.split("_")[2])

                # -----------------------------------------------------------------
                # INDEX MANAGEMENT (Critical for avoiding data leakage)
                # -----------------------------------------------------------------
                # Get absolute indices from saved data
                absolute_inner_train_idx = np.array(inner_fold_data["train_idx"])
                absolute_inner_val_idx = np.array(inner_fold_data["val_idx"])

                # Convert absolute indices to relative indices for the current outer training set
                inner_train_relative = np.where(
                    np.isin(train_idx, absolute_inner_train_idx)
                )[0]
                inner_val_relative = np.where(
                    np.isin(train_idx, absolute_inner_val_idx)
                )[0]

                # Split inner training data using relative indices
                X_inner_train, X_inner_val = (
                    X_train.iloc[inner_train_relative],
                    X_train.iloc[inner_val_relative],
                )
                y_inner_train, y_inner_val = (
                    y_train[inner_train_relative],
                    y_train[inner_val_relative],
                )  # Hard labels or Regression targets

                # Get teacher outputs for inner training and validation sets
                teacher_outputs_inner_train = teacher_targets_train[
                    inner_train_relative
                ]
                teacher_outputs_inner_val = teacher_targets_train[inner_val_relative]

                # ---------------------------------------------------------------------
                # PREPROCESSING: Apply model-specific data transformations
                # ---------------------------------------------------------------------
                X_inner_train, X_inner_val = preprocess(
                    X_inner_train,
                    y_inner_train,
                    X_inner_val,
                    cat_cols,
                    config,
                    preprocessing_type=preprocessing_type,
                )
                # ----------------------------------------------------------------------
                # MODEL TRAINING: Train student model on inner training data
                # ----------------------------------------------------------------------
                model = get_student_model(
                    config=config,
                    task_type=model_task_type,
                    device=device,
                    hyperparams=hyperparams,
                    cat_cols=cat_cols,
                    model_type=student_model_type,
                )
                logger.info(
                    f"Training Model on Outer Fold {outer_fold}, Inner Fold {inner_fold}..."
                )
                model.train(X_inner_train, teacher_outputs_inner_train)

                # ----------------------------------------------------------------------
                # VALIDATION: Evaluate model performance on inner validation set
                # ----------------------------------------------------------------------
                val_preds = model.predict(X_inner_val)  # Probs or Regression logits
                val_metrics = model.evaluate(
                    y_pred=val_preds,
                    y_true=y_inner_val,
                    y_teacher_true=teacher_outputs_inner_val,
                    original_task_type=original_task_type,
                )

                # Store metrics from this fold for later mean calculation
                val_metrics_list.append(val_metrics)

                if model_task_type == "binary":
                    inner_fold_scores.append(val_metrics["fidelity_f1"])
                    trial.report(np.mean(inner_fold_scores), step=inner_fold)
                else:
                    inner_fold_scores.append(-val_metrics["fidelity_mae"])
                    trial.report(np.mean(inner_fold_scores), step=inner_fold)

            # Calculate mean metrics as user attributes after all inner folds
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
            study_name=f"{dataset_id}.{outer_fold}.{student_model_type}({teacher_model_type}).{model_task_type[:2]}.{original_task_type[:2]}",
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
                None, student_model_type, model_task_type, False
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
                trial=None,
                model_type=student_model_type,
                task_type=model_task_type,
                use_hpo=False,
            )

        logger.info(f"Best hyperparameters for fold {outer_fold}: {best_hyperparams}")
        logger.info(f"Best score: {study.best_value:.4f}")

        # =========================================================================
        # STEP 7: FINAL MODEL TRAINING (on complete outer training set)
        # =========================================================================
        # Preprocess the outer training and test sets
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
        model = get_student_model(
            config=config,
            task_type=model_task_type,
            device=device,
            hyperparams=best_hyperparams,  # Use best hyperparameters from HPO
            cat_cols=cat_cols,
            model_type=student_model_type,
        )
        model.train(X_train, teacher_targets_train)

        # =========================================================================
        # STEP 8: FINAL EVALUATION (unbiased performance on outer test set)
        # =========================================================================
        # This provides the unbiased performance estimate for this fold
        start_time = time.time()
        test_preds = model.predict(X_test)  # Predicted logits or probabilities
        end_time = time.time() - start_time
        logger.info(f"\t Inference Time: {end_time:.5f} seconds")
        test_metrics = model.evaluate(
            test_preds, y_test, teacher_targets_test, original_task_type
        )

        # Store outer fold score for later analysis
        outer_fold_results = {
            "fold": outer_fold,
            "seed": config["training"]["random_state"],
            "inference_time": end_time,
            **test_metrics,
        }
        outer_fold_scores.append(outer_fold_results)

        # =========================================================================
        # STEP 9: STORE PREDICTIONS FOR STUDENT TRAINING
        # =========================================================================
        # Save predictions with their corresponding dataset indices
        output_dfs.append(
            pd.DataFrame(
                {
                    "index": test_idx,
                    "output": (
                        test_preds[:, 1] if model_task_type == "binary" else test_preds
                    ),
                }
            )
        )

    # =========================================================================
    # STEP 10: SAVE RESULTS AND METADATA
    # =========================================================================

    # -------------------------------------------------------------------------
    # SAVE OUTER FOLD METRICS
    # -------------------------------------------------------------------------
    outer_fold_df = pd.DataFrame(outer_fold_scores)
    sub_folder = "hpo" if use_hpo else "default"
    output_dir = os.path.join(config["data"]["outer_folds_path"], "student", sub_folder)
    os.makedirs(output_dir, exist_ok=True)

    metrics_file = os.path.join(
        output_dir,
        f"{dataset_id}_{student_model_type}({teacher_model_type})_{model_task_type[:2]}.csv",
    )
    outer_fold_df.to_csv(metrics_file, index=False)
    logger.info(f"Outer fold metrics saved to: {metrics_file}")

    # Calculate and log overall performance across all folds
    mean_inference_time = outer_fold_df["inference_time"].mean()
    std_inference_time = outer_fold_df["inference_time"].std()
    mean_parameters = outer_fold_df["parameters"].mean()
    std_parameters = outer_fold_df["parameters"].std()

    if original_task_type == "binary":
        mean_acc = outer_fold_df["acc"].mean()
        std_acc = outer_fold_df["acc"].std()
        mean_f1 = outer_fold_df["f1"].mean()
        std_f1 = outer_fold_df["f1"].std()
        mean_roc = outer_fold_df["roc"].mean()
        std_roc = outer_fold_df["roc"].std()

        mean_fidelity_acc = outer_fold_df["fidelity_acc"].mean()
        std_fidelity_acc = outer_fold_df["fidelity_acc"].std()
        mean_fidelity_f1 = outer_fold_df["fidelity_f1"].mean()
        std_fidelity_f1 = outer_fold_df["fidelity_f1"].std()
        mean_fidelity_roc = outer_fold_df["fidelity_roc"].mean()
        std_fidelity_roc = outer_fold_df["fidelity_roc"].std()
        mean_fidelity_kl_div = outer_fold_df["fidelity_kl_div"].mean()
        std_fidelity_kl_div = outer_fold_df["fidelity_kl_div"].std()

        logger.info(f"=== FINAL RESULTS FOR DATASET {dataset_id} ===")
        logger.info(f"Balanced Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"ROC AUC: {mean_roc:.4f} ± {std_roc:.4f}")
        logger.info(
            f"Fidelity Accuracy: {mean_fidelity_acc:.4f} ± {std_fidelity_acc:.4f}"
        )
        logger.info(
            f"Fidelity F1 Score: {mean_fidelity_f1:.4f} ± {std_fidelity_f1:.4f}"
        )
        logger.info(
            f"Fidelity ROC AUC: {mean_fidelity_roc:.4f} ± {std_fidelity_roc:.4f}"
        )
        logger.info(
            f"Fidelity KL Divergence: {mean_fidelity_kl_div:.4f} ± {std_fidelity_kl_div:.4f}"
        )

        if model_task_type == "regression":
            mean_fidelity_mae = outer_fold_df["fidelity_mae"].mean()
            std_fidelity_mae = outer_fold_df["fidelity_mae"].std()
            mean_fidelity_mse = outer_fold_df["fidelity_mse"].mean()
            std_fidelity_mse = outer_fold_df["fidelity_mse"].std()
            mean_fidelity_rmse = outer_fold_df["fidelity_rmse"].mean()
            std_fidelity_rmse = outer_fold_df["fidelity_rmse"].std()
            mean_fidelity_r2 = outer_fold_df["fidelity_r2"].mean()
            std_fidelity_r2 = outer_fold_df["fidelity_r2"].std()

            logger.info(
                f"Fidelity MAE: {mean_fidelity_mae:.4f} ± {std_fidelity_mae:.4f}"
            )
            logger.info(
                f"Fidelity MSE: {mean_fidelity_mse:.4f} ± {std_fidelity_mse:.4f}"
            )
            logger.info(
                f"Fidelity RMSE: {mean_fidelity_rmse:.4f} ± {std_fidelity_rmse:.4f}"
            )
            logger.info(f"Fidelity R2: {mean_fidelity_r2:.4f} ± {std_fidelity_r2:.4f}")

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

        mean_fidelity_mae = outer_fold_df["fidelity_mae"].mean()
        std_fidelity_mae = outer_fold_df["fidelity_mae"].std()
        mean_fidelity_mse = outer_fold_df["fidelity_mse"].mean()
        std_fidelity_mse = outer_fold_df["fidelity_mse"].std()
        mean_fidelity_rmse = outer_fold_df["fidelity_rmse"].mean()
        std_fidelity_rmse = outer_fold_df["fidelity_rmse"].std()
        mean_fidelity_r2 = outer_fold_df["fidelity_r2"].mean()
        std_fidelity_r2 = outer_fold_df["fidelity_r2"].std()

        logger.info(f"Fidelity MAE: {mean_fidelity_mae:.4f} ± {std_fidelity_mae:.4f}")
        logger.info(f"Fidelity MSE: {mean_fidelity_mse:.4f} ± {std_fidelity_mse:.4f}")
        logger.info(
            f"Fidelity RMSE: {mean_fidelity_rmse:.4f} ± {std_fidelity_rmse:.4f}"
        )
        logger.info(f"Fidelity R2: {mean_fidelity_r2:.4f} ± {std_fidelity_r2:.4f}")

    logger.info(
        f"Mean Inference Time: {mean_inference_time:.4f} ± {std_inference_time:.4f}"
    )
    logger.info(f"Mean Parameters: {mean_parameters:.4f} ± {std_parameters:.4f}")

    # Save summary statistics as well
    summary_stats = {
        "dataset_id": dataset_id,
        "student_model_type": student_model_type,
        "teacher_model_type": teacher_model_type,
        "original_task_type": original_task_type,
        "student_task_type": model_task_type,
        "seed": config["training"]["random_state"],
        "use_hpo": config["training"]["use_hpo"],
        "mean_inference_time": mean_inference_time,
        "std_inference_time": std_inference_time,
        "mean_parameters": mean_parameters,
        "std_parameters": std_parameters,
    }

    if original_task_type == "binary":
        summary_stats.update(
            {
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_roc": mean_roc,
                "std_roc": std_roc,
                "mean_fidelity_acc": mean_fidelity_acc,
                "std_fidelity_acc": std_fidelity_acc,
                "mean_fidelity_f1": mean_fidelity_f1,
                "std_fidelity_f1": std_fidelity_f1,
                "mean_fidelity_roc": mean_fidelity_roc,
                "std_fidelity_roc": std_fidelity_roc,
                "mean_fidelity_kl_div": mean_fidelity_kl_div,
                "std_fidelity_kl_div": std_fidelity_kl_div,
            }
        )
        if model_task_type == "regression":
            summary_stats.update(
                {
                    "mean_fidelity_mae": mean_fidelity_mae,
                    "std_fidelity_mae": std_fidelity_mae,
                    "mean_fidelity_mse": mean_fidelity_mse,
                    "std_fidelity_mse": std_fidelity_mse,
                    "mean_fidelity_rmse": mean_fidelity_rmse,
                    "std_fidelity_rmse": std_fidelity_rmse,
                    "mean_fidelity_r2": mean_fidelity_r2,
                    "std_fidelity_r2": std_fidelity_r2,
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
                "mean_fidelity_mae": mean_fidelity_mae,
                "std_fidelity_mae": std_fidelity_mae,
                "mean_fidelity_mse": mean_fidelity_mse,
                "std_fidelity_mse": std_fidelity_mse,
                "mean_fidelity_rmse": mean_fidelity_rmse,
                "std_fidelity_rmse": std_fidelity_rmse,
                "mean_fidelity_r2": mean_fidelity_r2,
                "std_fidelity_r2": std_fidelity_r2,
            }
        )

    # Load existing summary file if it exists, otherwise create new
    summary_file = os.path.join(
        config["data"]["results_dir_path"], "student", f"{dataset_id}_results.json"
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
    # SAVE STUDENT PREDICTIONS
    # -------------------------------------------------------------------------
    # Combine predictions from all outer folds
    if output_dfs:  # Check if we have any DataFrames to concatenate
        output_df = pd.concat(output_dfs, ignore_index=True)
    else:
        output_df = pd.DataFrame(columns=["index", "output"])
    output_df = output_df.sort_values(by="index")

    # Create the full directory structure
    sub_folder = "hpo" if use_hpo else "default"
    output_dir = os.path.join(config["data"]["output_dir_path"], sub_folder, "student")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"{dataset_id}_{student_model_type}({teacher_model_type})_{model_task_type[:2]}.csv",
    )
    output_df.to_csv(output_file, index=False)
    logger.info(f"{student_model_type} outputs saved to: {output_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train teacher model.")
    parser.add_argument(
        "--student_model_type", required=True, help="Type of teacher model to train."
    )
    parser.add_argument(
        "--teacher_model_type",
        default=None,
        help="Type of teacher model to use for training.",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    datasets = config["data"]["datasets"]

    # Train teacher models for each dataset
    for dataset_id in datasets:
        train_student(
            dataset_id=dataset_id,
            config=config,
            student_model_type=args.student_model_type,
            teacher_model_type=args.teacher_model_type,
        )
