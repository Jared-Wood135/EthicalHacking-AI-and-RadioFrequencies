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
    - make_json_safe
    - get_rfc
    - train_and_test_model
    - save_baseline
    - reproduce_model_from_json
- Main Function
'''
# ----- Imports -----------------------------------------------------------------------------------
import os                       # For model saving
import json                     # For model saving
import joblib                   # For model saving
import platform                 # For model saving
import sklearn                  # For model saving

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

# File pathings
HIGHEST_SNR_REDUCED_DF = '../Datasets/highest_snr_reduced_df.hdf5'
BASELINE_MODEL_OUTPUT_PATH = '../Models/highest_snr_random_forest_baseline.joblib'
BASELINE_METRICS_OUTPUT_PATH = '../Models/highest_snr_random_forest_baseline_metrics.json'

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def make_json_safe(obj):
    """
    Converts NumPy / pandas / sklearn objects into JSON-safe Python objects.
    """

    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(value) for value in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    return str(obj)


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
        cv: int = 3,
        metrics_output_path: str = BASELINE_METRICS_OUTPUT_PATH,
        save_metrics_json: bool = True
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

    - metrics_output_path (str):
        - DEFAULT: BASELINE_METRICS_OUTPUT_PATH
        - The path of the baseline metrics

    - save_metrics_json (bool):
        - DEFAULT: True
        - The json file of the best baseline model for reproducibility

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

    X_flat = X.reshape(num_signals, -1)
    y_labels = np.argmax(Y, axis=1)

    all_indices = np.arange(num_signals)

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
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labels
    )

    X_train = X_flat[train_indices]
    X_test = X_flat[test_indices]

    y_train = y_labels[train_indices]
    y_test = y_labels[test_indices]

    print("=" * 100)
    print("TRAIN/TEST SPLIT")
    print("=" * 100)
    print("Train indices:", len(train_indices))
    print("Test indices:", len(test_indices))
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

    scoring_metric = "roc_auc_ovr_weighted"

    grid_search = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    # ========== Run Grid Search ==================================================================
    print("=" * 100)
    print("GRID SEARCH STARTED")
    print("=" * 100)
    print("Scoring metric:", scoring_metric)
    print("Total parameter combinations:", np.prod([len(v) for v in param_grid.values()]))
    print()

    grid_search.fit(X_train, y_train)

    # ========== Grid Search Results ==============================================================
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

    # ========== Test Set Evaluation ==============================================================
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

    # ========== Reproducibility + Metrics JSON ===================================================
    metrics = {
        "project_stage": "highest_snr_baseline",
        "model_type": "RandomForestClassifier",
        "input_type": "flattened_raw_iq",

        "dataset": {
            "dataset_filepath": hdf5_data_filepath,
            "original_X_shape": list(X.shape),
            "flattened_X_shape": list(X_flat.shape),
            "Y_shape": list(Y.shape),
            "Z_shape": list(Z.shape),
            "unique_snr_values": np.unique(Z).tolist(),
            "num_signals": int(num_signals),
            "num_classes": int(len(np.unique(y_labels))),
            "mod_type_mapping": mod_type_mapping
        },

        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "stratify": "modulation_id",
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "num_train": int(len(train_indices)),
            "num_test": int(len(test_indices))
        },

        "grid_search": {
            "scoring": scoring_metric,
            "cv": cv,
            "param_grid": param_grid,
            "best_cv_roc_auc_ovr_weighted": grid_search.best_score_,
            "best_params": grid_search.best_params_
        },

        "test_metrics": {
            "accuracy": test_accuracy,
            "precision_weighted": test_precision_weighted,
            "recall_weighted": test_recall_weighted,
            "roc_auc_ovr_weighted": test_roc_auc_ovr_weighted
        },

        "classification_report": class_report,
        "confusion_matrix": conf_matrix,

        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "h5py_version": h5py.__version__
        }
    }

    metrics = make_json_safe(metrics)

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

    # ========== Save Metrics JSON ================================================================
    if save_metrics_json:
        metrics_output_dir = os.path.dirname(metrics_output_path)

        if metrics_output_dir != "":
            os.makedirs(metrics_output_dir, exist_ok=True)

        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print("=" * 100)
        print("METRICS JSON SAVED")
        print("=" * 100)
        print("Metrics saved to:", metrics_output_path)
        print()

    return best_model, metrics, grid_search, cv_results_df


def save_baseline(
        best_model,
        metrics: dict,
        model_output_path: str = BASELINE_MODEL_OUTPUT_PATH,
        metrics_output_path: str = BASELINE_METRICS_OUTPUT_PATH,
        save_model: bool = True,
        save_metrics_json: bool = True
) -> None:
    """
    About
    -----
    - Saves the best RandomForestClassifier baseline model
    - Saves model metrics and reproducibility metadata as a JSON file

    Parameters
    ----------
    - best_model:
        - Trained best model from GridSearchCV

    - metrics (dict):
        - Dictionary of model performance metrics and reproducibility metadata

    - model_output_path (str):
        - DEFAULT: BASELINE_MODEL_OUTPUT_PATH
        - Filepath where the trained model will be saved

    - metrics_output_path (str):
        - DEFAULT: BASELINE_METRICS_OUTPUT_PATH
        - Filepath where model metrics will be saved

    - save_model (bool):
        - DEFAULT: True
        - Whether to save the trained .joblib model

    - save_metrics_json (bool):
        - DEFAULT: True
        - Whether to save the metrics JSON

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
    if save_model:
        joblib.dump(best_model, model_output_path)

    # ========== Save Metrics JSON ================================================================
    if save_metrics_json:
        metrics = make_json_safe(metrics)

        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=4)

    print("=" * 100)
    print("BASELINE SAVE COMPLETE")
    print("=" * 100)

    if save_model:
        print("Model saved to:", model_output_path)
    else:
        print("Model saving skipped.")

    if save_metrics_json:
        print("Metrics JSON saved to:", metrics_output_path)
    else:
        print("Metrics JSON saving skipped.")


def reproduce_model_from_json(
        metrics_json_filepath: str = BASELINE_METRICS_OUTPUT_PATH,
        model_output_path: str = "../Models/highest_snr_random_forest_reproduced.joblib",
        save_model: bool = True
) -> tuple:
    """
    About
    -----
    - Reproduces the RandomForestClassifier baseline using a saved metrics JSON file
    - Loads:
        - dataset filepath
        - train/test split indices
        - best model hyperparameters
        - random_state
    - Retrains the model using the same training rows
    - Evaluates the reproduced model on the same test rows
    - Optionally saves the reproduced model as a .joblib file

    Parameters
    ----------
    - metrics_json_filepath (str):
        - DEFAULT: BASELINE_METRICS_OUTPUT_PATH
        - Path to the saved baseline metrics JSON file

    - model_output_path (str):
        - DEFAULT: '../Models/highest_snr_random_forest_reproduced.joblib'
        - Path where the reproduced model should be saved

    - save_model (bool):
        - DEFAULT: True
        - Whether to save the reproduced model

    Raises
    ------
    - None

    Returns
    -------
    - tuple:
        - reproduced_model
        - reproduction_metrics
    """

    # ========== Load Metrics JSON ================================================================
    with open(metrics_json_filepath, "r") as f:
        saved_metrics = json.load(f)

    # ========== Pull Reproducibility Info ========================================================
    dataset_filepath = saved_metrics["dataset"]["dataset_filepath"]

    train_indices = np.array(saved_metrics["split"]["train_indices"])
    test_indices = np.array(saved_metrics["split"]["test_indices"])

    random_state = saved_metrics["split"]["random_state"]
    best_params = saved_metrics["grid_search"]["best_params"]

    # ========== Load Dataset =====================================================================
    with h5py.File(dataset_filepath, "r") as f:
        X = f["X"][:]
        Y = f["Y"][:]
        Z = f["Z"][:].flatten()

    # ========== Prepare Features and Labels ======================================================
    num_signals = X.shape[0]

    X_flat = X.reshape(num_signals, -1)
    y_labels = np.argmax(Y, axis=1)

    X_train = X_flat[train_indices]
    X_test = X_flat[test_indices]

    y_train = y_labels[train_indices]
    y_test = y_labels[test_indices]

    # ========== Rebuild Model ====================================================================
    reproduced_model = RandomForestClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=1,
        class_weight=None
    )

    print("=" * 100)
    print("REPRODUCING RANDOM FOREST BASELINE")
    print("=" * 100)
    print("Metrics JSON filepath:", metrics_json_filepath)
    print("Dataset filepath:", dataset_filepath)
    print("X shape:", X.shape)
    print("Flattened X shape:", X_flat.shape)
    print("Train rows:", len(train_indices))
    print("Test rows:", len(test_indices))
    print("Random state:", random_state)
    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()

    # ========== Train Reproduced Model ===========================================================
    reproduced_model.fit(X_train, y_train)

    # ========== Evaluate Reproduced Model ========================================================
    y_pred = reproduced_model.predict(X_test)
    y_pred_proba = reproduced_model.predict_proba(X_test)

    reproduced_accuracy = accuracy_score(y_test, y_pred)

    reproduced_precision_weighted = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    reproduced_recall_weighted = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    reproduced_roc_auc_ovr_weighted = roc_auc_score(
        y_test,
        y_pred_proba,
        multi_class="ovr",
        average="weighted"
    )

    class_report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    conf_matrix = confusion_matrix(y_test, y_pred)

    reproduction_metrics = {
        "model_type": "RandomForestClassifier",
        "reproduction_source_json": metrics_json_filepath,
        "dataset_filepath": dataset_filepath,
        "input_type": "Flattened raw I/Q",
        "original_X_shape": list(X.shape),
        "flattened_X_shape": list(X_flat.shape),
        "train_indices_used": train_indices.tolist(),
        "test_indices_used": test_indices.tolist(),
        "random_state": random_state,
        "best_params": best_params,

        "reproduced_test_accuracy": float(reproduced_accuracy),
        "reproduced_test_precision_weighted": float(reproduced_precision_weighted),
        "reproduced_test_recall_weighted": float(reproduced_recall_weighted),
        "reproduced_test_roc_auc_ovr_weighted": float(reproduced_roc_auc_ovr_weighted),

        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist()
    }

    # ========== Print Comparison =================================================================
    print("=" * 100)
    print("REPRODUCED MODEL PERFORMANCE")
    print("=" * 100)
    print(f"Accuracy:              {reproduced_accuracy:.4f}")
    print(f"Weighted Precision:    {reproduced_precision_weighted:.4f}")
    print(f"Weighted Recall:       {reproduced_recall_weighted:.4f}")
    print(f"Weighted ROC-AUC OVR:  {reproduced_roc_auc_ovr_weighted:.4f}")
    print()
    print("Classification Report:")
    print(class_report)

    # ========== Compare to Original Saved Metrics ================================================
    if "test_metrics" in saved_metrics:
        original_metrics = saved_metrics["test_metrics"]

        print("=" * 100)
        print("ORIGINAL JSON METRICS VS REPRODUCED METRICS")
        print("=" * 100)

        metric_comparison = pd.DataFrame([
            {
                "metric": "accuracy",
                "original": original_metrics.get("accuracy"),
                "reproduced": reproduced_accuracy,
                "difference": reproduced_accuracy - original_metrics.get("accuracy")
            },
            {
                "metric": "precision_weighted",
                "original": original_metrics.get("precision_weighted"),
                "reproduced": reproduced_precision_weighted,
                "difference": reproduced_precision_weighted - original_metrics.get("precision_weighted")
            },
            {
                "metric": "recall_weighted",
                "original": original_metrics.get("recall_weighted"),
                "reproduced": reproduced_recall_weighted,
                "difference": reproduced_recall_weighted - original_metrics.get("recall_weighted")
            },
            {
                "metric": "roc_auc_ovr_weighted",
                "original": original_metrics.get("roc_auc_ovr_weighted"),
                "reproduced": reproduced_roc_auc_ovr_weighted,
                "difference": reproduced_roc_auc_ovr_weighted - original_metrics.get("roc_auc_ovr_weighted")
            }
        ])

        print(metric_comparison.to_string(index=False))
        print()

        reproduction_metrics["original_vs_reproduced_comparison"] = metric_comparison.to_dict(
            orient="records"
        )

    # ========== Save Reproduced Model ============================================================
    if save_model:
        model_output_dir = os.path.dirname(model_output_path)

        if model_output_dir != "":
            os.makedirs(model_output_dir, exist_ok=True)

        joblib.dump(reproduced_model, model_output_path)

        print("=" * 100)
        print("REPRODUCED MODEL SAVED")
        print("=" * 100)
        print("Model saved to:", model_output_path)
        print()

    return reproduced_model, reproduction_metrics

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
    best_model, metrics = train_and_test_model(
        hdf5_data_filepath=HIGHEST_SNR_REDUCED_DF,
        mod_type_mapping=MOD_TYPE_MAPPING,
        test_size=0.20,
        random_state=42,
        cv=3,
        metrics_output_path=BASELINE_METRICS_OUTPUT_PATH,
        save_metrics_json=True
    )

    save_baseline(
        best_model=best_model,
        metrics=metrics,
        model_output_path=BASELINE_MODEL_OUTPUT_PATH,
        metrics_output_path=BASELINE_METRICS_OUTPUT_PATH,
        save_model=True,
        save_metrics_json=True
    )

    reproduce_model_from_json()

# =================================================================================================
# END Main Function
# =================================================================================================