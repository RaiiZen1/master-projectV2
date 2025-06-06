# Evaluation of Gradient-Based Decision Tree Methods for Model Distillation
Author: Markus Johannes Herre (University of Mannheim)
Date: June 6, 2025

## 1. Introduction
This project explores the efficacy of knowledge distillation techniques in tabular data modeling. Specifically, we investigate how complex teacher models can guide simpler student models to achieve competitive performance while reducing computational cost.

## 2. Objectives
- Understand and implement **teacher-student** frameworks for tabular prediction tasks
- Compare different teacher architectures (e.g., TabPFN, CatBoost, Random Forest, TabM) on a diverse set of OpenML datasets
- Evaluate student models (MLP, CatBoost, Random Forest, TabM variants) trained on teacher outputs versus ground truth labels
- Assess the impact of training on logits (teacher predictions) and hyperparameter optimization (HPO)

## 3. Research Questions
1. How well do student models trained on teacher logits replicate teacher performance?
2. Which combinations of teacher-student architectures yield the best trade-off between accuracy and resource efficiency?
3. Does hyperparameter optimization significantly improve distilled student models over default configurations?
4. How do different ensemble and adapter methods (e.g., BatchEnsemble / TabM) affect knowledge transfer?

## 4. Data and Experimental Setup
- **Datasets**: A curated selection of 20 OpenML tabular datasets, spanning binary classification and regression tasks
- **Cross-validation**: Nested cross-validation with outer and inner folds to ensure unbiased performance estimates
- **Caching**: Data, fold indices, Optuna databases, and prediction outputs are stored under `data/` for reproducibility

## 5. Model Architectures
- **Teacher Models**:
  - TabPFN (Bayesian-inspired neural network)
  - Tree-based models (CatBoost, Random Forest)
  - BatchEnsemble-based architectures (TabM, Grande)
- **Student Models**:
  - Multilayer Perceptron (MLP)
  - Tree-based models
  - TabM variants (mini, packed, normal)

## 6. Methodology
1. **Teacher Training**:
   - Perform nested HPO (optional) on outer folds
   - Retrain the best teacher on the full outer training set
   - Save predictions on the outer test set
2. **Student Training**:
   - Load teacher predictions (logits) as targets
   - Optionally include ground-truth labels for combined objectives
   - Train student models under the same cross-validation scheme
3. **Evaluation**:
   - Compare metrics (accuracy, F1, MAE, etc.) of teacher vs. student
   - Analyze inference time and model size (parameters)
   - Report aggregated results across datasets and folds

## 7. Expected Contributions
- A systematic comparison of knowledge distillation strategies for tabular data
- Insights into the performance-resource trade-offs of distilled student models
- Open-source implementation and reproducible pipelines for further research

## 8. Project Structure
```
├── config/config.json         # Experiment configuration and dataset selection
├── train_teacher.py           # Teacher training and HPO pipeline
├── train_student.py           # Student training on teacher outputs
├── common/                    # Shared utilities and factories
└── packages/tabm_reference.py # Reference implementation of TabM and BatchEnsemble modules
```

---
*June 2025*
