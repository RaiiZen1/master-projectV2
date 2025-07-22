# Knowledge Distillation with Gradient-Based Decision Trees

**Author:** Markus Johannes Herre (University of Mannheim)  
**Date:** August 4, 2025

## 1. Project Overview

This project investigates the effectiveness of knowledge distillation for tabular data modeling. Knowledge distillation is a machine learning technique where a compact "student" model is trained to reproduce the behavior of a larger, more complex "teacher" model. The goal is to create smaller, faster models that retain the high performance of the teacher, making them suitable for deployment in resource-constrained environments.

This repository contains the code and resources to reproduce the experiments and results of our research, which evaluates various teacher-student model combinations across a range of tabular datasets.

## 2. Repository Structure

The repository is organized as follows:

```
.
├── README.md                    # This file, providing an overview of the project.
├── requirements.txt             # A list of Python dependencies required to run the code.
├── train_student.py             # Main Python script for training student models.
├── train_teacher.py             # Main Python script for training teacher models.
├── common/                      # Directory for shared Python modules.
│   ├── data_loader.py           # Module for loading and preprocessing datasets.
│   ├── evaluate.py              # Module with functions for model evaluation.
│   ├── hpo.py                   # Module for hyperparameter optimization using Optuna.
│   ├── preprocessing.py         # Module for data preprocessing utilities.
│   ├── student_model_factory.py # Factory for creating student model architectures.
│   ├── teacher_model_factory.py # Factory for creating teacher model architectures.
│   └── utils.py                 # General utility functions.
├── config/
│   └── config.json              # Configuration file for experiments.
├── notebooks/
│   ├── results_analysis.ipynb   # Jupyter notebook for analyzing experiment results.
│   └── training_pipeline.ipynb  # Jupyter notebook demonstrating the training pipeline.
└── results/                     # Directory for storing final, aggregated results.
```

## 3. How to Use This Repository

To get started with this project and reproduce the experiments, follow these steps:

### 3.1. Prerequisites

Ensure you have Python 3.9 or higher installed. You will also need `pip` to manage Python packages.

### 3.2. Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/RaiiZen1/master-projectV2.git
    cd master-projectV2
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3.3. Configuration

The experiments are configured using the `config/config.json` file. You can modify this file to select different datasets, models, and hyperparameters. The file specifies which datasets from OpenML to use and which teacher and student models to train and evaluate.

### 3.4. Running the Experiments

The training process is divided into two main stages: training the teacher model and then training the student model.

1.  **Train the Teacher Model:**
    Use the provided python scripts to train the teacher models. The scripts handle the entire pipeline, including data loading, optional hyperparameter optimization (HPO), training, and saving the model and its predictions. For GPU-accelerated training, ensure that a CUDA-compatible device is available (recommended for larger models):

    *   
        ```bash
        python train_teacher.py
        ```

2.  **Train the Student Model:**
    After the teacher model is trained and its predictions (logits) are saved, you can train the student model. The student model learns from the teacher's logits.

    *   
        ```bash
        python train_student.py
        ```
 
### 3.5. Analyzing the Results

The results of the experiments, including performance metrics and model comparisons, are saved in the `results/` directory. You can use the `notebooks/results_analysis.ipynb` Jupyter notebook to visualize and analyze the results in detail.

## 4. Key Components and Files

*   **`train_teacher.py`**: This script orchestrates the training of the teacher models. It loads the configuration, prepares the data, performs nested cross-validation, and, if enabled, runs hyperparameter optimization using Optuna. The best-performing teacher model is then retrained on the full training data and its predictions on the test set are saved.
*   **`train_student.py`**: This script handles the training of the student models. It loads the teacher's saved predictions (logits) and uses them as the target for training various student model architectures. It evaluates the student's performance against both the teacher's predictions and the ground truth labels.
*   **`common/` directory**: This directory contains a collection of shared modules used by both training scripts.
    *   `data_loader.py`: Manages the download and preprocessing of datasets from OpenML.
    *   `evaluate.py`: Provides functions to calculate various performance metrics like accuracy, F1-score, and Mean Absolute Error (MAE).
    *   `hpo.py`: Implements the hyperparameter optimization logic using the Optuna framework.
    *   `*_model_factory.py`: These factory modules are responsible for creating instances of different teacher and student models (e.g., TabPFN, CatBoost, MLP).
*   **`data/` directory**: This is the central location for all data related to the experiments. It includes cached datasets, cross-validation fold indices, Optuna databases for HPO, and the outputs (models and predictions) from the training runs. It will be created during training.
*   **`results/` directory**: This directory stores the final, aggregated results from the experiments, typically in CSV format, along with summary reports. It will be created during training.

## 5. Citation

If you use this repository or the accompanying thesis in your work, please cite:

```bibtex
@misc{herre_gradientkd_2025,
  author       = {Markus Johannes Herre},
  title        = {Evaluation of Gradient-Based Decision Tree Methods for Model Distillation},
  school       = {University of Mannheim},
  year         = {2025},
  month        = {August},
  type         = {Master's Thesis},
  address      = {Mannheim, Germany},
  url          = {https://github.com/RaiiZen1/master-projectV2},
  note         = {Code and supplementary materials available at GitHub repository},
}
```

## 6. Contact

For questions or collaborations, please contact me through LinkedIn (https://www.linkedin.com/in/markus-herre/).
