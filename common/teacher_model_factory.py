import numpy as np
import os
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from catboost import CatBoostRegressor, CatBoostClassifier
from common.evaluate import evaluate_classification, evaluate_regression
from common.packages.tabm_reference import Model, make_parameter_groups
from GRANDE import GRANDE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier, TabPFNRegressor
from typing import Dict, Literal


class TeacherModelBase(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the model on the given data.

        Parameters:
            X (np.ndarray): The input features with shape (n_samples, n_features).
            y (np.ndarray): The target labels with shape (n_samples,).

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the provided data.

        Parameters:
            X (np.ndarray): The input features with shape (n_samples, n_features).

        Returns:
            np.ndarray: An array of predicted probabilities with shape (n_samples, n_classes).
                        (If the model returns logits, these should be post-processed to probabilities.)
        """
        pass

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the model's predictions against the true labels.

        Parameters:
            y_pred (np.ndarray): The predicted probabilities or values.
            y_true (np.ndarray): The true labels.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        if self.task_type == "binary":
            metrics = evaluate_classification(y_pred, y_true)
            metrics["parameters"] = self.count_parameters()
            return metrics

        elif self.task_type == "regression":
            metrics = evaluate_regression(y_pred, y_true)
            metrics["parameters"] = self.count_parameters()
            return metrics

        else:
            raise ValueError(
                f"Unknown task type: {self.task_type}. Must be 'binary' or 'regression'."
            )

    @abstractmethod
    def count_parameters(self) -> int:
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: The total number of trainable parameters.
        """
        pass


class TabPFNTeacherModel(TeacherModelBase):

    def __init__(self, **kwargs):
        """
        Initialize a TabPFN teacher model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the base
            class constructor.
        """
        self.device = kwargs.pop("device")
        self.config = kwargs.pop("config")
        self.task_type = kwargs.pop("task_type")
        super().__init__(**kwargs)

    def train(self, X, y, **training_params):
        if self.task_type == "binary":
            self.model = TabPFNClassifier()
        else:
            self.model = TabPFNRegressor()
        # If X is bigger than 10000 samples, reduce data size to 10000
        if X.shape[0] > 10000:
            X = X[:10000]
            y = y[:10000]
        self.model.fit(X, y)

    def predict(self, X):
        if self.task_type == "binary":
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)

    def count_parameters(self) -> int:
        """
        Counts the number of trainable parameters in the underlying TabPFN model.

        Returns:
            int: Total number of trainable parameters.

        Raises:
            ValueError: If the TabPFN model has not been fitted yet.
        """
        if not hasattr(self.model, "model_"):
            raise ValueError(
                "The TabPFN model is not fitted yet. Please call fit() first."
            )
        return sum(p.numel() for p in self.model.model_.parameters() if p.requires_grad)


class CatBoostTeacherModel(TeacherModelBase):
    def __init__(self, **kwargs):
        """
        Initialize a CatBoost teacher model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the base class constructor.
        """
        self.device = kwargs.pop("device")
        self.config = kwargs.pop("config")
        self.task_type = kwargs.pop("task_type")
        self.hyperparams = kwargs.pop("hyperparams")
        super().__init__(**kwargs)

    def train(self, X, y):
        """
        Train the CatBoost model using the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target data.
        **training_params : dict
            Additional training parameters to pass to the CatBoost model's fit method.
        """
        config = self.config
        random_state = config["training"]["random_state"]

        # Initialize the CatBoost model
        if self.task_type == "binary":
            self.model = CatBoostClassifier(
                random_seed=random_state,
                thread_count=-1,  # Use all available CPU cores
                logging_level="Silent",  # Suppress CatBoost output
                task_type=("GPU" if str(self.device).lower() == "cuda" else "CPU"),
                **self.hyperparams,  # Include the sampled hyperparameters
            )
        else:
            self.model = CatBoostRegressor(
                random_seed=random_state,
                thread_count=-1,  # Use all available CPU cores
                logging_level="Silent",  # Suppress CatBoost output
                task_type=("GPU" if str(self.device).lower() == "cuda" else "CPU"),
                **self.hyperparams,  # Include the sampled hyperparameters
            )

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        # Train the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=20,
        )

    def predict(self, X):
        """
        Make predictions on the given data using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The predicted probabilities for each class.
        """
        if self.task_type == "binary":
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)

    def count_parameters(self):
        """
        Count the number of parameters in the trained CatBoost model.

        Returns
        -------
        int
            The number of parameters in the model.
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("The model has not been trained yet.")

        # Get the number of trees in the model
        tree_count = self.model.tree_count_

        # For each tree, count:
        # - split values (one per non-leaf node)
        # - leaf values (one per leaf node)
        # - feature indices (one per non-leaf node)
        # For a binary tree with depth d, there are 2^d - 1 non-leaf nodes and 2^d leaf nodes
        # This is a simplified approximation
        approx_depth = self.hyperparams.get(
            "max_depth", 6
        )  # Default depth in CatBoost is 6
        non_leaf_nodes = 2**approx_depth - 1
        leaf_nodes = 2**approx_depth

        # Total parameters per tree
        params_per_tree = (
            non_leaf_nodes * 2 + leaf_nodes
        )  # split values + feature indices + leaf values
        return tree_count * params_per_tree


class GRANDETeacherModel(TeacherModelBase):
    def __init__(self, **kwargs):
        """
        Initialize a GRANDE teacher model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments. The `device` parameter is popped from kwargs and
            defaults to 'cpu' if not provided.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        self.device = kwargs.pop("device")
        self.config = kwargs.pop("config")
        self.task_type = kwargs.pop("task_type")
        self.hyperparams = kwargs.pop("hyperparams")
        self.cat_cols = kwargs.pop("cat_cols")
        super().__init__(**kwargs)

    def train(self, X, y):
        """
        Train the GRANDE teacher model using the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target data.
        **training_params : dict
            Additional training parameters to pass to the GRANDE model's fit method.
        """
        config = self.config
        random_state = config["training"]["random_state"]

        # Determine categorical indices
        cat_indices = [i for i, is_cat in enumerate(self.cat_cols) if is_cat]

        training_params = {
            "epochs": 1000,
            "early_stopping_epochs": 25,
            "batch_size": 64,
            "cat_idx": cat_indices,
            "random_seed": random_state,
        }

        if self.task_type == "binary":
            training_params["objective"] = "binary"
        else:
            training_params["objective"] = "regression"

        # Instantiate the GRANDE teacher model.
        self.model = GRANDE(params=self.hyperparams, args=training_params)

        # Split the data into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        self.model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    def predict(self, X):
        """
        Make predictions on the given input data using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        array-like
            The predicted target values.
        """
        if self.task_type == "binary":
            return self.model.predict(X)
        else:
            return self.model.predict(X).squeeze()

    def count_parameters(self):
        """
        Count the number of trainable parameters in the GRANDE teacher model based on the
        dense representation of a single tree and the number of trees in the ensemble.

        The number of parameters for a single tree is computed as:
            leaf parameters: 2^d
            split thresholds: (2^d - 1) * n
            feature selection (one-hot): (2^d - 1) * n
        so that:
            params_per_tree = 2^d + 2 * n * (2^d - 1)
        The total is then:
            total_params = n_estimators * params_per_tree
        """
        if not hasattr(self, "model"):
            raise ValueError("The model has not been trained/initialized yet.")

        # These attributes must be set when the GRANDE model is instantiated.
        # Adjust the attribute names if they are different in your GRANDE implementation.
        d = self.model.depth  # Depth of each decision tree.
        n = self.model.number_of_variables  # Number of features used in the tree.
        E = self.model.n_estimators  # Number of trees in the ensemble.

        # Calculate the number of parameters for one tree.
        params_per_tree = (2**d) + 2 * n * ((2**d) - 1)
        total_params = E * params_per_tree

        return total_params


class TabMTeacherModel(TeacherModelBase):
    def __init__(self, **kwargs):
        """
        Initialize a TabM teacher model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the base class constructor.
        """
        self.device = kwargs.pop("device")
        self.config = kwargs.pop("config")
        self.task_type = kwargs.pop("task_type")
        self.hyperparams = kwargs.pop("hyperparams")
        super().__init__(**kwargs)

        # Set up AMP if CUDA is available.
        if torch.cuda.is_available():
            self.amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            self.amp_enabled = True
        else:
            self.amp_dtype = None
            self.amp_enabled = False

    def train(self, X, y):
        """
        Train the TabM model using the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target data.
        **training_params : dict
            Additional training parameters to pass to the TabM model's fit method.
        """
        # Get parameters
        n_blocks = self.hyperparams.get("n_blocks")
        d_block = self.hyperparams.get("d_block")
        dropout = self.hyperparams.get("dropout")
        lr = self.hyperparams.get("learning_rate")
        weight_decay = self.hyperparams.get("weight_decay")

        # Set model architecture
        arch_type = "tabm"
        bins = None
        n_features = X.shape[1]

        if self.task_type == "binary":
            n_classes = len(np.unique(y))
            self.n_classes = n_classes
            cat_idx = []  # Assumes all features are continuous

            # Initialize model
            compile_model = False
            self.model = Model(
                n_num_features=n_features,
                cat_cardinalities=cat_idx,
                n_classes=n_classes,
                backbone={
                    "type": "MLP",
                    "n_blocks": n_blocks,
                    "d_block": d_block,
                    "dropout": dropout,
                },
                bins=bins,
                num_embeddings=(
                    None
                    if bins is None
                    else {
                        "type": "PiecewiseLinearEmbeddings",
                        "d_embedding": 16,
                        "activation": False,
                        "version": "B",
                    }
                ),
                arch_type=arch_type,
                k=32,
                share_training_batches=True,
            ).to(self.device)

        else:
            if len(y.shape) > 1 and y.shape[1] > 1:
                # Multiclass logits
                n_outputs = y.shape[1]
            else:
                # Binary or single-output regression
                n_outputs = 1

            # Initialize model
            compile_model = False
            self.model = Model(
                n_num_features=n_features,
                cat_cardinalities=[],  # Assumes all features are continuous
                n_classes=n_outputs,
                backbone={
                    "type": "MLP",
                    "n_blocks": n_blocks,
                    "d_block": d_block,
                    "dropout": dropout,
                },
                bins=bins,
                num_embeddings=(
                    None
                    if bins is None
                    else {
                        "type": "PiecewiseLinearEmbeddings",
                        "d_embedding": 16,
                        "activation": False,
                        "version": "B",
                    }
                ),
                arch_type=arch_type,
                k=32,
                share_training_batches=True,
            ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            make_parameter_groups(self.model), lr=lr, weight_decay=weight_decay
        )
        self.evaluation_mode = torch.no_grad if compile_model else torch.inference_mode

        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        if self.task_type == "binary":
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        n_samples = X_tensor.size(0)
        indices = torch.randperm(n_samples)
        train_end = int(0.8 * n_samples)
        train_idx, val_idx = indices[:train_end], indices[train_end:]
        self.data = {
            "train": {"x_cont": X_tensor[train_idx], "y": y_tensor[train_idx]},
            "val": {"x_cont": X_tensor[val_idx], "y": y_tensor[val_idx]},
        }
        Y_train = self.data["train"]["y"]

        # Training parameters
        n_epochs = 100000
        patience = 16
        batch_size = 256
        train_size = len(Y_train)
        best = {"val": -float("inf"), "epoch": -1}
        remaining_patience = patience

        # Training loop
        for epoch in range(n_epochs):
            batches = torch.randperm(train_size, device=self.device).split(batch_size)
            for batch_idx in batches:
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.loss_fn(
                    self.apply_model("train", batch_idx), Y_train[batch_idx]
                )
                if self.amp_enabled and self.amp_dtype == torch.float16:
                    scaler = torch.cuda.amp.GradScaler()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            val_score = self.evaluate_tabm("val")
            if val_score > best["val"]:
                best = {"val": val_score, "epoch": epoch}
                remaining_patience = patience
            else:
                remaining_patience -= 1
            if remaining_patience < 0:
                break

    def apply_model(self, part: str, idx: torch.Tensor) -> torch.Tensor:
        """
        Apply the model to a batch of data.

        Parameters
        ----------
        part : str
            The data partition to use ('train' or 'val').
        idx : torch.Tensor
            The indices of the samples to use.

        Returns
        -------
        torch.Tensor
            The model output.
        """
        with torch.autocast(
            self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            x_cont = self.data[part]["x_cont"][idx]
            x_cat = self.data[part].get("x_cat")
            if self.task_type == "binary":
                return self.model(x_cont, x_cat).float()
            else:
                return (
                    self.model(x_cont, x_cat)
                    .squeeze(-1)  # Remove the last dimension for regression tasks
                    .float()
                )

    def loss_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted outputs.
        y_true : torch.Tensor
            The ground truth.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        if self.task_type == "binary":
            # Use the second-to-last dimension for averaging over k heads.
            k = y_pred.shape[-2]
            target = y_true.repeat_interleave(k)
            return F.cross_entropy(y_pred.flatten(0, 1), target)
        else:
            # TabM produces k predictions. Each of them must be trained separately.
            # For student model (regression on logits), y_pred.shape == (batch_size, k)
            k = y_pred.shape[-1 if len(y_pred.shape) <= 2 else -2]

            # Flatten the predictions and repeat the targets
            return F.mse_loss(
                y_pred.flatten(0, 1),
                (
                    y_true.repeat_interleave(k)
                    if self.model.share_training_batches
                    else y_true
                ),
            )

    def evaluate_tabm(self, part: str) -> float:
        """
        Evaluate the model on a data partition.

        Parameters
        ----------
        part : str
            The data partition to use ('train' or 'val').

        Returns
        -------
        float
            The accuracy score.
        """
        self.model.eval()
        with self.evaluation_mode():
            eval_batch_size = 8096

            if self.task_type == "binary":
                indices = torch.arange(
                    len(self.data[part]["y"]), device=self.device
                ).split(eval_batch_size)
                y_pred = torch.cat(
                    [self.apply_model(part, idx) for idx in indices]
                ).cpu()
                y_pred = torch.softmax(y_pred, dim=-1).mean(1)
                y_true = self.data[part]["y"].cpu().numpy()
                return float(accuracy_score(y_true, y_pred.argmax(1)))
            else:
                y_pred = (
                    torch.cat(
                        [
                            self.apply_model(part, idx)
                            for idx in torch.arange(
                                len(self.data[part]["y"]), device=self.device
                            ).split(eval_batch_size)
                        ]
                    )
                    .cpu()
                    .numpy()
                )

                # Transform the predictions back to the original label space
                if hasattr(self, "regression_label_stats"):
                    y_pred = (
                        y_pred * self.regression_label_stats["std"]
                        + self.regression_label_stats["mean"]
                    )

                # Compute the mean of the k predictions
                y_pred = y_pred.mean(1)

                y_true = self.data[part]["y"].cpu().numpy()

                # For student model, we want to minimize MSE, so return negative MSE
                # The higher the value, the better (consistent with example.ipynb)
                score = -(mean_squared_error(y_true, y_pred) ** 0.5)
                return float(score)

    def predict(self, X):
        """
        Make predictions on the given data using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted class indices.
        """
        if self.task_type == "binary":
            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.model(X_tensor)
                # Since share_training_batches is True, average over the second dimension.
                probs = torch.softmax(logits, dim=-1).mean(1)
                return probs.cpu().numpy()
        else:
            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                # TabM produces output with shape (batch_size, k) or (batch_size, k, n_outputs)
                logits = self.model(X_tensor)

                # For student model (regression), we need to average over the k predictions
                # Check the dimensions of logits and average appropriately
                if logits.ndim > 2:  # Case: (batch_size, k, n_outputs)
                    logits = logits.mean(1)  # Average over k dimension
                elif (
                    logits.shape[1] > 1
                    and hasattr(self.model, "k")
                    and self.model.k > 1
                ):
                    # Case: (batch_size, k) for single output regression
                    logits = logits.mean(
                        1, keepdim=True
                    )  # Average and keep output dimension

                return logits.cpu().numpy().squeeze(-1)

    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.

        Returns
        -------
        int
            The total number of trainable parameters.
        """
        if not hasattr(self, "model"):
            raise ValueError("The model has not been trained yet")
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class MLPTeacherModel(TeacherModelBase):
    def __init__(self, **kwargs):
        """
        Initialize a TabM teacher model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the base class constructor.
        """
        self.device = kwargs.pop("device")
        self.config = kwargs.pop("config")
        self.task_type = kwargs.pop("task_type")
        self.hyperparams = kwargs.pop("hyperparams")
        super().__init__(**kwargs)

        # Set up AMP if CUDA is available.
        if torch.cuda.is_available():
            self.amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            self.amp_enabled = True
        else:
            self.amp_dtype = None
            self.amp_enabled = False

    def train(self, X, y):
        """
        Train the TabM model using the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target data.
        **training_params : dict
            Additional training parameters to pass to the TabM model's fit method.
        """
        # Get parameters
        n_blocks = self.hyperparams.get("n_blocks")
        d_block = self.hyperparams.get("d_block")
        dropout = self.hyperparams.get("dropout")
        lr = self.hyperparams.get("learning_rate")
        weight_decay = self.hyperparams.get("weight_decay")

        # Set model architecture
        arch_type = "plain"
        bins = None
        n_features = X.shape[1]

        if self.task_type == "binary":
            n_classes = len(np.unique(y))
            self.n_classes = n_classes
            cat_idx = []  # Assumes all features are continuous

            # Initialize model
            compile_model = False
            self.model = Model(
                n_num_features=n_features,
                cat_cardinalities=cat_idx,
                n_classes=n_classes,
                backbone={
                    "type": "MLP",
                    "n_blocks": n_blocks,
                    "d_block": d_block,
                    "dropout": dropout,
                },
                bins=bins,
                num_embeddings=(
                    None
                    if bins is None
                    else {
                        "type": "PiecewiseLinearEmbeddings",
                        "d_embedding": 16,
                        "activation": False,
                        "version": "B",
                    }
                ),
                arch_type=arch_type,
                k=None,
                share_training_batches=True,
            ).to(self.device)

        else:
            if len(y.shape) > 1 and y.shape[1] > 1:
                # Multiclass logits
                n_outputs = y.shape[1]
            else:
                # Binary or single-output regression
                n_outputs = 1

            # Initialize model
            compile_model = False
            self.model = Model(
                n_num_features=n_features,
                cat_cardinalities=[],  # Assumes all features are continuous
                n_classes=n_outputs,
                backbone={
                    "type": "MLP",
                    "n_blocks": n_blocks,
                    "d_block": d_block,
                    "dropout": dropout,
                },
                bins=bins,
                num_embeddings=(
                    None
                    if bins is None
                    else {
                        "type": "PiecewiseLinearEmbeddings",
                        "d_embedding": 16,
                        "activation": False,
                        "version": "B",
                    }
                ),
                arch_type=arch_type,
                k=None,
                share_training_batches=True,
            ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            make_parameter_groups(self.model), lr=lr, weight_decay=weight_decay
        )
        self.evaluation_mode = torch.no_grad if compile_model else torch.inference_mode

        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        if self.task_type == "binary":
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        n_samples = X_tensor.size(0)
        indices = torch.randperm(n_samples)
        train_end = int(0.8 * n_samples)
        train_idx, val_idx = indices[:train_end], indices[train_end:]
        self.data = {
            "train": {"x_cont": X_tensor[train_idx], "y": y_tensor[train_idx]},
            "val": {"x_cont": X_tensor[val_idx], "y": y_tensor[val_idx]},
        }
        Y_train = self.data["train"]["y"]

        # Training parameters
        n_epochs = 100000
        patience = 16
        batch_size = 256
        train_size = len(Y_train)
        best = {"val": -float("inf"), "epoch": -1}
        remaining_patience = patience

        # Training loop
        for epoch in range(n_epochs):
            batches = torch.randperm(train_size, device=self.device).split(batch_size)
            for batch_idx in batches:
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.loss_fn(
                    self.apply_model("train", batch_idx), Y_train[batch_idx]
                )
                if self.amp_enabled and self.amp_dtype == torch.float16:
                    scaler = torch.cuda.amp.GradScaler()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            val_score = self.evaluate_tabm("val")
            if val_score > best["val"]:
                best = {"val": val_score, "epoch": epoch}
                remaining_patience = patience
            else:
                remaining_patience -= 1
            if remaining_patience < 0:
                break

    def apply_model(self, part: str, idx: torch.Tensor) -> torch.Tensor:
        """
        Apply the model to a batch of data.

        Parameters
        ----------
        part : str
            The data partition to use ('train' or 'val').
        idx : torch.Tensor
            The indices of the samples to use.

        Returns
        -------
        torch.Tensor
            The model output.
        """
        with torch.autocast(
            self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            x_cont = self.data[part]["x_cont"][idx]
            x_cat = self.data[part].get("x_cat")
            if self.task_type == "binary":
                return self.model(x_cont, x_cat).float()
            else:
                return (
                    self.model(x_cont, x_cat)
                    .squeeze(-1)  # Remove the last dimension for regression tasks
                    .float()
                )

    def loss_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted outputs.
        y_true : torch.Tensor
            The ground truth.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        if self.task_type == "binary":
            # Use the second-to-last dimension for averaging over k heads.
            k = y_pred.shape[-2]
            target = y_true.repeat_interleave(k)
            return F.cross_entropy(y_pred.flatten(0, 1), target)
        else:
            # TabM produces k predictions. Each of them must be trained separately.
            # For student model (regression on logits), y_pred.shape == (batch_size, k)
            k = y_pred.shape[-1 if len(y_pred.shape) <= 2 else -2]

            # Flatten the predictions and repeat the targets
            return F.mse_loss(
                y_pred.flatten(0, 1),
                (
                    y_true.repeat_interleave(k)
                    if self.model.share_training_batches
                    else y_true
                ),
            )

    def evaluate_tabm(self, part: str) -> float:
        """
        Evaluate the model on a data partition.

        Parameters
        ----------
        part : str
            The data partition to use ('train' or 'val').

        Returns
        -------
        float
            The accuracy score.
        """
        self.model.eval()
        with self.evaluation_mode():
            eval_batch_size = 8096

            if self.task_type == "binary":
                indices = torch.arange(
                    len(self.data[part]["y"]), device=self.device
                ).split(eval_batch_size)
                y_pred = torch.cat(
                    [self.apply_model(part, idx) for idx in indices]
                ).cpu()
                y_pred = torch.softmax(y_pred, dim=-1).mean(1)
                y_true = self.data[part]["y"].cpu().numpy()
                return float(accuracy_score(y_true, y_pred.argmax(1)))
            else:
                y_pred = (
                    torch.cat(
                        [
                            self.apply_model(part, idx)
                            for idx in torch.arange(
                                len(self.data[part]["y"]), device=self.device
                            ).split(eval_batch_size)
                        ]
                    )
                    .cpu()
                    .numpy()
                )

                # Transform the predictions back to the original label space
                if hasattr(self, "regression_label_stats"):
                    y_pred = (
                        y_pred * self.regression_label_stats["std"]
                        + self.regression_label_stats["mean"]
                    )

                # Compute the mean of the k predictions
                y_pred = y_pred.mean(1)

                y_true = self.data[part]["y"].cpu().numpy()

                # For student model, we want to minimize MSE, so return negative MSE
                # The higher the value, the better (consistent with example.ipynb)
                score = -(mean_squared_error(y_true, y_pred) ** 0.5)
                return float(score)

    def predict(self, X):
        """
        Make predictions on the given data using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted class indices.
        """
        if self.task_type == "binary":
            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.model(X_tensor)
                # Since share_training_batches is True, average over the second dimension.
                probs = torch.softmax(logits, dim=-1).mean(1)
                return probs.cpu().numpy()
        else:
            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                # TabM produces output with shape (batch_size, k) or (batch_size, k, n_outputs)
                logits = self.model(X_tensor)

                # For student model (regression), we need to average over the k predictions
                # Check the dimensions of logits and average appropriately
                if logits.ndim > 2:  # Case: (batch_size, k, n_outputs)
                    logits = logits.mean(1)  # Average over k dimension
                elif (
                    logits.shape[1] > 1
                    and hasattr(self.model, "k")
                    and self.model.k > 1
                ):
                    # Case: (batch_size, k) for single output regression
                    logits = logits.mean(
                        1, keepdim=True
                    )  # Average and keep output dimension

                return logits.cpu().numpy().squeeze(-1)

    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.

        Returns
        -------
        int
            The total number of trainable parameters.
        """
        if not hasattr(self, "model"):
            raise ValueError("The model has not been trained yet")
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class RandomForestTeacherModel(TeacherModelBase):
    def __init__(self, **kwargs):
        """
        Initialize a RandomForest teacher model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the base class constructor.
        """
        self.device = kwargs.pop("device")
        self.config = kwargs.pop("config")
        self.task_type = kwargs.pop("task_type")
        self.hyperparams = kwargs.pop("hyperparams")
        super().__init__(**kwargs)

    def train(self, X, y):
        """
        Train the RandomForest teacher model using the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target data.
        **training_params : dict
            Additional training parameters (not used for RandomForest).
        """
        config = self.config
        random_state = config["training"]["random_state"]

        if self.task_type == "binary":
            # Initialize the RandomForest model with the provided hyperparameters
            self.model = RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,  # Use all available CPU cores
                **self.hyperparams,
            )
        else:
            self.model = RandomForestRegressor(
                random_state=random_state,
                n_jobs=-1,  # Use all available CPU cores
                **self.hyperparams,
            )

        # Train the model
        self.model.fit(X, y)

    def predict(self, X):
        """
        Make probability predictions on the given data using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The predicted probabilities for each class.
        """
        if self.task_type == "binary":
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)

    def count_parameters(self):
        """
        Count the number of parameters in the trained RandomForest model.

        For each tree, we count:
        - The feature indices at each node (1 parameter per node except leaf nodes)
        - The thresholds at each node (1 parameter per node except leaf nodes)
        - The values at each leaf node (1 parameter per leaf node)

        Returns
        -------
        int
            The number of parameters in the model.
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("The model has not been trained yet.")

        n_params = 0
        for tree in self.model.estimators_:
            tree = tree.tree_
            # Number of nodes in the tree
            n_nodes = tree.node_count
            # Number of leaf nodes
            n_leaves = tree.n_leaves
            # Number of internal nodes
            n_internal = n_nodes - n_leaves
            # Each internal node has a feature index and a threshold
            n_params += 2 * n_internal

            # Each leaf node has value(s) - depends on task type
            if self.task_type == "binary":
                # For classification, each leaf stores class distribution
                n_params += n_leaves * self.model.n_classes_
            else:
                # For regression, each leaf stores a single value
                n_params += n_leaves

        return n_params


def get_teacher_model(
    config: dict,
    task_type: Literal["binary", "regression"],
    device,
    hyperparams,
    cat_cols=None,
) -> TeacherModelBase:
    """
    Returns an instance of the teacher model based on the configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and dataset details.
    device : torch.device
        The device to use for training (CPU or GPU).

    Returns
    -------
    TeacherModelBase
        An instance of the teacher model.
    """
    model_type = config["model"]["teacher_model"]
    if model_type == "tabpfn":
        return TabPFNTeacherModel(
            config=config, task_type=task_type, device=device, hyperparams=hyperparams
        )
    elif model_type == "catboost":
        return CatBoostTeacherModel(
            config=config, task_type=task_type, device=device, hyperparams=hyperparams
        )
    elif model_type == "grande":
        return GRANDETeacherModel(
            config=config,
            task_type=task_type,
            device=device,
            hyperparams=hyperparams,
            cat_cols=cat_cols,
        )
    elif model_type == "tabm":
        return TabMTeacherModel(
            config=config, task_type=task_type, device=device, hyperparams=hyperparams
        )
    elif model_type == "mlp":
        return MLPTeacherModel(
            config=config, task_type=task_type, device=device, hyperparams=hyperparams
        )
    elif model_type == "random_forest":
        return RandomForestTeacherModel(
            config=config, task_type=task_type, device=device, hyperparams=hyperparams
        )
    else:
        raise ValueError(f"Unknown teacher type: {config['model']['teacher_model']}")
