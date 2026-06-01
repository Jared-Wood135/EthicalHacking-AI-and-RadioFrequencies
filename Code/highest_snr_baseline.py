'''
FILE OVERVIEW:
- Code and functions specifically for highest_snr_baseline.ipynb
- This is primarily for reproducibility and maintaining presentability of highest_snr_baseline.ipynb
- This file pulls the MOD_TYPE_MAPPING variable from "acquire.py"

=================================================

MISC COMMENTS:
- THIS ASSUMES YOU HAVE THE REDUCED DATASET '../Datasets/highest_snr_reduced_df.hdf5'
- Because of the nature of the dataset, there will NOT be any data preparation since
  the data appears to already be uniformly distributed and represents signals accurately

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - get_rfc
    - train_and_test_model
    - save_baseline
- Main Function
'''
# ----- Imports -----------------------------------------------------------------------------------
import os
import json
import joblib

import numpy as np              # Iteration over arrays (I/Q signal array)
import pandas as pd             # Easier dataset manipulation
import h5py                     # Reading .hdf5 datasets
import matplotlib.pyplot as plt # Visualizations

# Random Forest Classifier (ML)
from sklearn.ensemble import RandomForestClassifier

# Hyper-parameter tuning
from sklearn.model_selection import GridSearchCV, train_test_split

# Standard model performance metrics
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# ----- Global Variables --------------------------------------------------------------------------
from acquire import MOD_TYPE_MAPPING    # Modulation ID and name mapping

HIGHEST_SNR_REDUCED_DF = '../Datasets/highest_snr_reduced_df.hdf5'

BASELINE_MODEL_OUTPUT_PATH = "../Models/highest_snr_random_forest_baseline.joblib"
BASELINE_METRICS_OUTPUT_PATH = "../Models/highest_snr_random_forest_baseline_metrics.json"

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def get_rfc(
        random_state: int = 35
) -> RandomForestClassifier:
    """
    About
    -----
    - Creates a base RandomForestClassifier for GridSearchCV
    - This model is used as the baseline model for the highest-SNR reduced dataset

    Parameters
    ----------
    - random_state (int):
        - DEFAULT: 35
        - Controls reproducibility

    Raises
    ------
    - None

    Returns
    -------
    - RandomForestClassifier
        - Untrained Random Forest model
    """
    # Just a simple Random Forest Classifier
    rfc = RandomForestClassifier(
        random_state=random_state,
        n_jobs=1,
        class_weight=None
    )

    return rfc


def train_and_test_model(
        hdf5_data_filepath: str = HIGHEST_SNR_REDUCED_DF,
        mod_type_mapping: dict = MOD_TYPE_MAPPING,
        test_size: float = 0.20,
        random_state: int = 35,
        cv: int = 3
) -> tuple:
    """
    About
    -----
    - Loads the reduced highest-SNR .hdf5 dataset
    - Flattens raw I/Q signals from shape (1024, 2) into shape (2048,)
    - Trains a RandomForestClassifier baseline using GridSearchCV
    - Uses weighted one-vs-rest multiclass ROC-AUC as the GridSearchCV scoring metric
    - Prints each parameter combination's cross-validation performance
    - Evaluates the best model on the held-out test set

    Dependencies
    ------------
    - get_rfc

    Parameters
    ----------
    - hdf5_data_filepath (str):
        - DEFAULT: HIGHEST_SNR_REDUCED_DF
        - Filepath to reduced .hdf5 dataset

    - mod_type_mapping (dict):
        - DEFAULT: MOD_TYPE_MAPPING
        - Dictionary mapping modulation IDs to modulation names

    - test_size (float):
        - DEFAULT: 0.20
        - Percentage of dataset reserved for testing

    - random_state (int):
        - DEFAULT: 35
        - Controls reproducibility

    - cv (int):
        - DEFAULT: 3
        - Number of cross-validation folds for GridSearchCV

    Raises
    ------
    - None

    Returns
    -------
    - tuple:
        - best_model
        - metrics dictionary
        - grid_search object
        - cv_results_df
    """

    # ========== Load Dataset =====================================================================
    with h5py.File(hdf5_data_filepath, "r") as f:
        X = f["X"][:]
        Y = f["Y"][:]
        Z = f["Z"][:].flatten()

    # ========== Prepare Features and Labels ======================================================
    num_signals = X.shape[0]

    # Flatten each signal:
    # Original shape:  (num_signals, 1024, 2)
    # Flattened shape: (num_signals, 2048)
    X_flat = X.reshape(num_signals, -1)

    # Convert one-hot labels to integer class IDs
    y_labels = np.argmax(Y, axis=1)

    print("=" * 100)
    print("DATASET INFORMATION")
    print("=" * 100)
    print("Dataset filepath:", hdf5_data_filepath)
    print("Original X shape:", X.shape)
    print("Flattened X shape:", X_flat.shape)
    print("Y shape:", Y.shape)
    print("Z shape:", Z.shape)
    print("Unique SNR values:", np.unique(Z))
    print("Number of classes:", len(np.unique(y_labels)))
    print()

    # ========== Train/Test Split =================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat,
        y_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labels
    )

    print("=" * 100)
    print("TRAIN/TEST SPLIT")
    print("=" * 100)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print()

    # ========== Define Model and Hyperparameter Grid =============================================
    rfc = get_rfc(random_state=random_state)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        scoring="roc_auc_ovr_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    # ========== Run Grid Search ==================================================================
    print("=" * 100)
    print("GRID SEARCH STARTED")
    print("=" * 100)
    print("Scoring metric: roc_auc_ovr_weighted")
    print("Total parameter combinations:", np.prod([len(v) for v in param_grid.values()]))
    print()

    grid_search.fit(X_train, y_train)

    # ========== Print Grid Search Results ========================================================
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    display_cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "param_max_features"
    ]

    cv_results_display = (
        cv_results_df[display_cols]
        .sort_values(by="rank_test_score")
        .reset_index(drop=True)
    )

    print("=" * 100)
    print("GRID SEARCH RESULTS")
    print("=" * 100)
    print(cv_results_display.to_string(index=False))
    print()

    print("=" * 100)
    print("BEST GRID SEARCH MODEL")
    print("=" * 100)
    print("Best ROC-AUC CV Score:", grid_search.best_score_)
    print("Best Parameters:")
    for key, value in grid_search.best_params_.items():
        print(f"  {key}: {value}")
    print()

    # ========== Evaluate Best Model on Test Set ==================================================
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)

    test_precision_weighted = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    test_recall_weighted = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    test_roc_auc_ovr_weighted = roc_auc_score(
        y_test,
        y_pred_proba,
        multi_class="ovr",
        average="weighted"
    )

    target_names = [
        mod_type_mapping[class_id]
        for class_id in sorted(np.unique(y_labels))
    ]

    class_report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0
    )

    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_type": "RandomForestClassifier",
        "dataset_filepath": hdf5_data_filepath,
        "input_type": "Flattened raw I/Q",
        "original_X_shape": list(X.shape),
        "flattened_X_shape": list(X_flat.shape),
        "test_size": test_size,
        "random_state": random_state,
        "cv": cv,
        "grid_search_scoring": "roc_auc_ovr_weighted",
        "best_cv_roc_auc_ovr_weighted": float(grid_search.best_score_),
        "best_params": grid_search.best_params_,
        "test_accuracy": float(test_accuracy),
        "test_precision_weighted": float(test_precision_weighted),
        "test_recall_weighted": float(test_recall_weighted),
        "test_roc_auc_ovr_weighted": float(test_roc_auc_ovr_weighted),
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist()
    }

    print("=" * 100)
    print("FINAL TEST SET PERFORMANCE")
    print("=" * 100)
    print(f"Accuracy:              {test_accuracy:.4f}")
    print(f"Weighted Precision:    {test_precision_weighted:.4f}")
    print(f"Weighted Recall:       {test_recall_weighted:.4f}")
    print(f"Weighted ROC-AUC OVR:  {test_roc_auc_ovr_weighted:.4f}")
    print()
    print("Classification Report:")
    print(class_report)

    return best_model, metrics, grid_search, cv_results_df


def save_baseline(
        best_model,
        metrics: dict,
        model_output_path: str = BASELINE_MODEL_OUTPUT_PATH,
        metrics_output_path: str = BASELINE_METRICS_OUTPUT_PATH
) -> None:
    """
    About
    -----
    - Saves the best RandomForestClassifier baseline model
    - Saves the model metrics as a JSON file

    Parameters
    ----------
    - best_model:
        - Trained best model from GridSearchCV

    - metrics (dict):
        - Dictionary of model performance metrics

    - model_output_path (str):
        - DEFAULT: BASELINE_MODEL_OUTPUT_PATH
        - Filepath where the trained model will be saved

    - metrics_output_path (str):
        - DEFAULT: BASELINE_METRICS_OUTPUT_PATH
        - Filepath where model metrics will be saved

    Raises
    ------
    - None

    Returns
    -------
    - None
    """
    # ========== Ensure Output Directories Exist ==================================================
    model_output_dir = os.path.dirname(model_output_path)
    metrics_output_dir = os.path.dirname(metrics_output_path)

    if model_output_dir != "":
        os.makedirs(model_output_dir, exist_ok=True)

    if metrics_output_dir != "":
        os.makedirs(metrics_output_dir, exist_ok=True)

    # ========== Save Model =======================================================================
    joblib.dump(best_model, model_output_path)

    # ========== Save Metrics =====================================================================
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("=" * 100)
    print("BASELINE SAVED")
    print("=" * 100)
    print("Model saved to:", model_output_path)
    print("Metrics saved to:", metrics_output_path)
    print()

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def main():
    """
    About
    -----
    - Runs the full highest-SNR baseline workflow
    - Loads the reduced .hdf5 dataset
    - Trains a Random Forest baseline using GridSearchCV
    - Evaluates the best model
    - Saves the best model and metrics
    - WARNING: THIS MAY TAKE A REALLY LONG TIME

    Dependencies
    ------------
    - '../Datasets/highest_snr_reduced_df.hdf5'
    - acquire.py containing MOD_TYPE_MAPPING
    - train_and_test_model
    - save_baseline

    Parameters
    ----------
    - None

    Raises
    ------
    - FileNotFoundError:
        - If the reduced .hdf5 dataset cannot be found

    Returns
    -------
    - None
    """
    best_model, metrics, grid_search, cv_results_df = train_and_test_model(
        hdf5_data_filepath=HIGHEST_SNR_REDUCED_DF,
        mod_type_mapping=MOD_TYPE_MAPPING,
        test_size=0.20,
        random_state=35,
        cv=3
    )

    save_baseline(
        best_model=best_model,
        metrics=metrics,
        model_output_path=BASELINE_MODEL_OUTPUT_PATH,
        metrics_output_path=BASELINE_METRICS_OUTPUT_PATH
    )

# =================================================================================================
# END Main Function
# =================================================================================================