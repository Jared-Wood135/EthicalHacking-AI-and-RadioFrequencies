'''
FILE OVERVIEW:
- Code and functions specifically for highest_snr_feature_engineering_modeling.ipynb
- This is primarily for reproducibility and maintaining presentability of the modeling notebook
- This file trains a RandomForestClassifier on selected feature-engineered columns
- The selected features come from highest_snr_feature_engineering_explore.py information-gain results
- The model and a reproducibility JSON are saved similarly to highest_snr_baseline.py

=================================================

MISC COMMENTS:
- THIS ASSUMES YOU HAVE THE FEATURE ENGINEERED DATASET '../Datasets/highest_snr_feature_engineered.parquet'
- THIS ASSUMES YOU HAVE THE SELECTED FEATURES JSON '../Reports/highest_snr_selected_features.json'
- The selected features JSON should be created by highest_snr_feature_engineering_explore.py
- The primary model here is a RandomForestClassifier using engineered signal features
- This is different from the baseline model, which used flattened raw I/Q samples

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - make_json_safe
    - load_feature_engineered_dataset
    - load_selected_feature_names
    - get_feature_columns
    - prepare_modeling_data
    - get_rfc
    - train_and_test_model
    - save_feature_engineered_model
    - reproduce_model_from_json
- Main Function
'''
# ----- Imports -----------------------------------------------------------------------------------
import os
import json
import joblib
import platform

import sklearn
import numpy as np              # Numerical operations
import pandas as pd             # Feature-engineered tabular dataset

# Random Forest Classifier (ML)
from sklearn.ensemble import RandomForestClassifier

# Hyper-parameter tuning and splitting
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
from acquire import MOD_TYPE_MAPPING    # Modulation ID and Name mapping

# File pathings
FEATURE_ENGINEERED_DATASET_PARQUET = '../Datasets/highest_snr_feature_engineered.parquet'
SELECTED_FEATURES_INPUT_PATH = '../Reports/highest_snr_selected_features.json'
FEATURE_ENGINEERED_MODEL_OUTPUT_PATH = '../Models/highest_snr_feature_engineered_random_forest_gridsearch.joblib'
FEATURE_ENGINEERED_METRICS_OUTPUT_PATH = '../Models/highest_snr_feature_engineered_random_forest_gridsearch_metrics.json'
FEATURE_ENGINEERED_CV_RESULTS_OUTPUT_PATH = '../Reports/highest_snr_feature_engineered_random_forest_gridsearch_cv_results.csv'

METADATA_COLUMNS = [
    'signal_index',
    'modulation_id',
    'modulation_type',
    'snr'
]

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def make_json_safe(obj):
    '''
    About
    -----
    - Converts NumPy / pandas / sklearn objects into JSON-safe Python objects
    - This is useful when saving reproducibility metadata and model metrics

    Parameters
    ----------
    - obj:
        - Any object that may need conversion before JSON serialization

    Raises
    ------
    - None

    Returns
    -------
    - object
        - JSON-safe version of obj
    '''
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
        return obj.to_dict(orient='records')

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    return str(obj)


def load_feature_engineered_dataset(
        feature_dataset_filepath: str = FEATURE_ENGINEERED_DATASET_PARQUET
) -> pd.DataFrame:
    '''
    About
    -----
    - Loads the feature-engineered highest-SNR dataset
    - Supports .parquet, .hdf5/.h5, and .csv files

    Parameters
    ----------
    - feature_dataset_filepath (str):
        - DEFAULT: FEATURE_ENGINEERED_DATASET_PARQUET
        - Path to the feature-engineered dataset

    Raises
    ------
    - FileNotFoundError:
        - If feature_dataset_filepath is not found

    - ValueError:
        - If the file extension is unsupported

    Returns
    -------
    - pd.DataFrame
        - Loaded feature-engineered dataset
    '''
    if not os.path.exists(feature_dataset_filepath):
        raise FileNotFoundError(f'Could not find dataset at: {feature_dataset_filepath}')

    file_extension = os.path.splitext(feature_dataset_filepath)[1].lower()

    if file_extension == '.parquet':
        df_features = pd.read_parquet(feature_dataset_filepath)

    elif file_extension in ['.hdf5', '.h5']:
        df_features = pd.read_hdf(feature_dataset_filepath, key='features')

    elif file_extension == '.csv':
        df_features = pd.read_csv(feature_dataset_filepath)

    else:
        raise ValueError(
            "Unsupported feature dataset file extension. "
            "Use '.parquet', '.hdf5', '.h5', or '.csv'."
        )

    print('\033[32mFeature-engineered dataset loaded successfully!\033[0m')
    print('Dataset filepath:', feature_dataset_filepath)
    print('Dataset shape:', df_features.shape)
    print()

    return df_features


def load_selected_feature_names(
        selected_features_filepath: str = SELECTED_FEATURES_INPUT_PATH,
        selected_feature_key: str = 'selected_feature_names'
) -> tuple:
    '''
    About
    -----
    - Loads selected feature names from the feature-engineering exploration JSON
    - These features are usually the top features ranked by information gain

    Parameters
    ----------
    - selected_features_filepath (str):
        - DEFAULT: SELECTED_FEATURES_INPUT_PATH
        - Path to selected features JSON file

    - selected_feature_key (str):
        - DEFAULT: 'selected_feature_names'
        - JSON key containing the selected feature name list

    Raises
    ------
    - FileNotFoundError:
        - If selected_features_filepath is not found

    - KeyError:
        - If selected_feature_key is not found in the JSON file

    Returns
    -------
    - tuple
        - selected_feature_names
        - selected_features_payload
    '''
    if not os.path.exists(selected_features_filepath):
        raise FileNotFoundError(f'Could not find selected features JSON at: {selected_features_filepath}')

    with open(selected_features_filepath, 'r') as f:
        selected_features_payload = json.load(f)

    if selected_feature_key not in selected_features_payload:
        raise KeyError(f'{selected_feature_key} was not found in {selected_features_filepath}')

    selected_feature_names = selected_features_payload[selected_feature_key]

    print('\033[32mSelected feature names loaded successfully!\033[0m')
    print('Selected features filepath:', selected_features_filepath)
    print('Selection method:', selected_features_payload.get('selection_method'))
    print('Top N:', selected_features_payload.get('top_n'))
    print('Number of selected features:', len(selected_feature_names))
    print()

    return selected_feature_names, selected_features_payload


def get_feature_columns(
        df_features: pd.DataFrame,
        metadata_columns: list = METADATA_COLUMNS
) -> list:
    '''
    About
    -----
    - Identifies every engineered feature column available in the feature-engineered dataset
    - Removes metadata and target columns

    Parameters
    ----------
    - df_features (pd.DataFrame):
        - Feature-engineered dataset

    - metadata_columns (list):
        - DEFAULT: METADATA_COLUMNS
        - Columns that should not be used as engineered model features

    Raises
    ------
    - None

    Returns
    -------
    - list
        - List of engineered feature columns
    '''
    feature_columns = [
        col for col in df_features.columns
        if col not in metadata_columns
    ]

    print('Number of available engineered feature columns:', len(feature_columns))

    return feature_columns


def prepare_modeling_data(
        df_features: pd.DataFrame,
        selected_feature_names: list = None,
        target_column: str = 'modulation_id',
        metadata_columns: list = METADATA_COLUMNS
) -> tuple:
    '''
    About
    -----
    - Creates X, y, and metadata objects for model training
    - Uses selected features when provided
    - If selected_feature_names is None, uses every engineered feature column
    - Replaces inf/-inf with NaN and fills NaN values with 0

    Parameters
    ----------
    - df_features (pd.DataFrame):
        - Feature-engineered dataset

    - selected_feature_names (list):
        - DEFAULT: None
        - Selected engineered feature columns to use for model training

    - target_column (str):
        - DEFAULT: 'modulation_id'
        - Target label column

    - metadata_columns (list):
        - DEFAULT: METADATA_COLUMNS
        - Metadata columns excluded from X

    Raises
    ------
    - KeyError:
        - If target_column or a selected feature is missing from df_features

    Returns
    -------
    - tuple
        - X_features
        - y_labels
        - metadata_df
        - feature_columns_used
    '''
    if target_column not in df_features.columns:
        raise KeyError(f'{target_column} was not found in df_features')

    if selected_feature_names is None:
        feature_columns_used = get_feature_columns(
            df_features=df_features,
            metadata_columns=metadata_columns
        )
    else:
        feature_columns_used = selected_feature_names

    missing_features = [feature for feature in feature_columns_used if feature not in df_features.columns]

    if len(missing_features) > 0:
        raise KeyError(f'Selected features not found in df_features: {missing_features}')

    X_features = (
        df_features[feature_columns_used]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y_labels = df_features[target_column]

    metadata_df = df_features[[col for col in metadata_columns if col in df_features.columns]].copy()

    print('=' * 100)
    print('MODELING DATASET PREPARED')
    print('=' * 100)
    print('X_features shape:', X_features.shape)
    print('y_labels shape:', y_labels.shape)
    print('metadata_df shape:', metadata_df.shape)
    print('Target column:', target_column)
    print('Number of features used:', len(feature_columns_used))
    print()

    return X_features, y_labels, metadata_df, feature_columns_used


def get_rfc(
        random_state: int = 35,
        n_jobs: int = 1
) -> RandomForestClassifier:
    '''
    About
    -----
    - Creates a base RandomForestClassifier for GridSearchCV
    - This model is used for the engineered-feature model

    Parameters
    ----------
    - random_state (int):
        - DEFAULT: 35
        - Controls reproducibility

    - n_jobs (int):
        - DEFAULT: 1
        - Number of parallel jobs for the base RandomForestClassifier
        - 1 is preferred for maximum reproducibility

    Raises
    ------
    - None

    Returns
    -------
    - RandomForestClassifier
        - Untrained Random Forest model
    '''
    rfc = RandomForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=None
    )

    return rfc


def train_and_test_model(
        feature_dataset_filepath: str = FEATURE_ENGINEERED_DATASET_PARQUET,
        selected_features_filepath: str = SELECTED_FEATURES_INPUT_PATH,
        mod_type_mapping: dict = MOD_TYPE_MAPPING,
        target_column: str = 'modulation_id',
        test_size: float = 0.20,
        random_state: int = 35,
        cv: int = 3,
        rfc_n_jobs: int = 1,
        grid_search_n_jobs: int = -1,
        use_selected_features: bool = True,
        save_metrics_json: bool = True,
        metrics_output_path: str = FEATURE_ENGINEERED_METRICS_OUTPUT_PATH,
        cv_results_output_path: str = FEATURE_ENGINEERED_CV_RESULTS_OUTPUT_PATH
) -> tuple:
    '''
    About
    -----
    - Loads the feature-engineered dataset
    - Loads selected top information-gain features from JSON when requested
    - Trains a RandomForestClassifier using GridSearchCV
    - Uses weighted one-vs-rest multiclass ROC-AUC as the GridSearchCV scoring metric
    - Evaluates the best model on the held-out test set
    - Builds and optionally saves a reproducibility JSON similar to highest_snr_baseline.py

    Dependencies
    ------------
    - load_feature_engineered_dataset
    - load_selected_feature_names
    - prepare_modeling_data
    - get_rfc
    - make_json_safe

    Parameters
    ----------
    - feature_dataset_filepath (str):
        - DEFAULT: FEATURE_ENGINEERED_DATASET_PARQUET
        - Path to the feature-engineered dataset

    - selected_features_filepath (str):
        - DEFAULT: SELECTED_FEATURES_INPUT_PATH
        - Path to selected features JSON file

    - mod_type_mapping (dict):
        - DEFAULT: MOD_TYPE_MAPPING
        - Dictionary mapping modulation IDs to modulation names

    - target_column (str):
        - DEFAULT: 'modulation_id'
        - Target label column

    - test_size (float):
        - DEFAULT: 0.20
        - Percentage of dataset reserved for testing

    - random_state (int):
        - DEFAULT: 35
        - Controls reproducibility

    - cv (int):
        - DEFAULT: 3
        - Number of cross-validation folds for GridSearchCV

    - rfc_n_jobs (int):
        - DEFAULT: 1
        - Number of jobs used inside RandomForestClassifier

    - grid_search_n_jobs (int):
        - DEFAULT: -1
        - Number of jobs used by GridSearchCV

    - use_selected_features (bool):
        - DEFAULT: True
        - Whether to train on selected information-gain features
        - If False, trains on all engineered features

    - save_metrics_json (bool):
        - DEFAULT: True
        - Whether to save the metrics JSON inside this function

    - metrics_output_path (str):
        - DEFAULT: FEATURE_ENGINEERED_METRICS_OUTPUT_PATH
        - Where metrics JSON should be saved

    - cv_results_output_path (str):
        - DEFAULT: FEATURE_ENGINEERED_CV_RESULTS_OUTPUT_PATH
        - Where GridSearchCV results should be saved as CSV

    Raises
    ------
    - FileNotFoundError:
        - If the feature dataset or selected features JSON is missing

    Returns
    -------
    - tuple
        - best_model
        - metrics
        - cv_results_df
    '''
    # ========== Load Dataset =====================================================================
    df_features = load_feature_engineered_dataset(
        feature_dataset_filepath=feature_dataset_filepath
    )

    selected_features_payload = None
    selected_feature_names = None

    if use_selected_features:
        selected_feature_names, selected_features_payload = load_selected_feature_names(
            selected_features_filepath=selected_features_filepath
        )

    X_features, y_labels, metadata_df, feature_columns_used = prepare_modeling_data(
        df_features=df_features,
        selected_feature_names=selected_feature_names,
        target_column=target_column,
        metadata_columns=METADATA_COLUMNS
    )

    all_indices = np.arange(df_features.shape[0])

    # ========== Train/Test Split =================================================================
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labels
    )

    X_train = X_features.iloc[train_indices]
    X_test = X_features.iloc[test_indices]
    y_train = y_labels.iloc[train_indices]
    y_test = y_labels.iloc[test_indices]

    print('=' * 100)
    print('TRAIN/TEST SPLIT')
    print('=' * 100)
    print('Train rows:', len(train_indices))
    print('Test rows:', len(test_indices))
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    print()

    # ========== Define Model and Hyperparameter Grid =============================================
    rfc = get_rfc(
        random_state=random_state,
        n_jobs=rfc_n_jobs
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    scoring_metric = 'roc_auc_ovr_weighted'

    grid_search = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=cv,
        n_jobs=grid_search_n_jobs,
        verbose=2,
        return_train_score=True
    )

    # ========== Run Grid Search ==================================================================
    print('=' * 100)
    print('ENGINEERED-FEATURE GRID SEARCH STARTED')
    print('=' * 100)
    print('Scoring metric:', scoring_metric)
    print('Total parameter combinations:', np.prod([len(v) for v in param_grid.values()]))
    print('Features used:', len(feature_columns_used))
    print()

    grid_search.fit(X_train, y_train)

    # ========== Grid Search Results ==============================================================
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    display_cols = [
        'rank_test_score',
        'mean_test_score',
        'std_test_score',
        'mean_train_score',
        'std_train_score',
        'param_n_estimators',
        'param_max_depth',
        'param_min_samples_split',
        'param_min_samples_leaf',
        'param_max_features'
    ]

    cv_results_display = (
        cv_results_df[display_cols]
        .sort_values(by='rank_test_score')
        .reset_index(drop=True)
    )

    print('=' * 100)
    print('GRID SEARCH RESULTS')
    print('=' * 100)
    print(cv_results_display.to_string(index=False))
    print()

    print('=' * 100)
    print('BEST ENGINEERED-FEATURE MODEL')
    print('=' * 100)
    print('Best ROC-AUC CV Score:', grid_search.best_score_)
    print('Best Parameters:')
    for key, value in grid_search.best_params_.items():
        print(f'  {key}: {value}')
    print()

    # ========== Evaluate Best Model on Test Set ==================================================
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)

    test_precision_weighted = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    test_recall_weighted = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    test_roc_auc_ovr_weighted = roc_auc_score(
        y_test,
        y_pred_proba,
        multi_class='ovr',
        average='weighted'
    )

    label_order = sorted(y_labels.unique())
    target_names = [mod_type_mapping[label] for label in label_order]

    class_report = classification_report(
        y_test,
        y_pred,
        labels=label_order,
        target_names=target_names,
        zero_division=0
    )

    conf_matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=label_order
    )

    feature_importance_df = pd.DataFrame({
        'feature': feature_columns_used,
        'random_forest_importance': best_model.feature_importances_
    }).sort_values(
        by='random_forest_importance',
        ascending=False
    ).reset_index(drop=True)

    feature_importance_df['random_forest_importance_rank'] = np.arange(
        1,
        len(feature_importance_df) + 1
    )

    metrics = {
        'project_stage': 'highest_snr_feature_engineering_modeling',
        'model_type': 'RandomForestClassifier',
        'input_type': 'selected_engineered_features' if use_selected_features else 'all_engineered_features',

        'dataset': {
            'feature_dataset_filepath': feature_dataset_filepath,
            'dataset_shape': list(df_features.shape),
            'num_rows': int(df_features.shape[0]),
            'target_column': target_column,
            'mod_type_mapping': mod_type_mapping
        },

        'selected_features_source': {
            'use_selected_features': use_selected_features,
            'selected_features_filepath': selected_features_filepath if use_selected_features else None,
            'selected_features_payload': selected_features_payload
        },

        'features_used': {
            'num_features_used': int(len(feature_columns_used)),
            'feature_columns_used': feature_columns_used
        },

        'split': {
            'test_size': test_size,
            'random_state': random_state,
            'stratify': target_column,
            'train_indices': train_indices.tolist(),
            'test_indices': test_indices.tolist(),
            'num_train': int(len(train_indices)),
            'num_test': int(len(test_indices))
        },

        'grid_search': {
            'scoring': scoring_metric,
            'cv': cv,
            'param_grid': param_grid,
            'best_cv_roc_auc_ovr_weighted': grid_search.best_score_,
            'best_params': grid_search.best_params_
        },

        'test_metrics': {
            'accuracy': test_accuracy,
            'precision_weighted': test_precision_weighted,
            'recall_weighted': test_recall_weighted,
            'roc_auc_ovr_weighted': test_roc_auc_ovr_weighted
        },

        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'feature_importances': feature_importance_df.to_dict(orient='records'),

        'environment': {
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'sklearn_version': sklearn.__version__
        }
    }

    metrics = make_json_safe(metrics)

    print('=' * 100)
    print('FINAL TEST SET PERFORMANCE')
    print('=' * 100)
    print(f'Accuracy:              {test_accuracy:.4f}')
    print(f'Weighted Precision:    {test_precision_weighted:.4f}')
    print(f'Weighted Recall:       {test_recall_weighted:.4f}')
    print(f'Weighted ROC-AUC OVR:  {test_roc_auc_ovr_weighted:.4f}')
    print()
    print('Classification Report:')
    print(class_report)
    print('Top feature importances from final model:')
    print(feature_importance_df.head(25).to_string(index=False))
    print()

    # ========== Save Metrics JSON and CV Results =================================================
    if cv_results_output_path is not None:
        cv_results_output_dir = os.path.dirname(cv_results_output_path)

        if cv_results_output_dir != '':
            os.makedirs(cv_results_output_dir, exist_ok=True)

        cv_results_df.to_csv(cv_results_output_path, index=False)
        print('GridSearchCV results saved to:', cv_results_output_path)
        print()

    if save_metrics_json:
        metrics_output_dir = os.path.dirname(metrics_output_path)

        if metrics_output_dir != '':
            os.makedirs(metrics_output_dir, exist_ok=True)

        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print('=' * 100)
        print('METRICS JSON SAVED')
        print('=' * 100)
        print('Metrics saved to:', metrics_output_path)
        print()

    return best_model, metrics, cv_results_df


def save_feature_engineered_model(
        best_model,
        metrics: dict,
        model_output_path: str = FEATURE_ENGINEERED_MODEL_OUTPUT_PATH,
        metrics_output_path: str = FEATURE_ENGINEERED_METRICS_OUTPUT_PATH,
        save_model: bool = True,
        save_metrics_json: bool = True
) -> None:
    '''
    About
    -----
    - Saves the best engineered-feature RandomForestClassifier model
    - Saves model metrics and reproducibility metadata as a JSON file
    - Mirrors the save_baseline pattern from highest_snr_baseline.py

    Parameters
    ----------
    - best_model:
        - Trained best model from GridSearchCV

    - metrics (dict):
        - Dictionary of model performance metrics and reproducibility metadata

    - model_output_path (str):
        - DEFAULT: FEATURE_ENGINEERED_MODEL_OUTPUT_PATH
        - Filepath where the trained model will be saved

    - metrics_output_path (str):
        - DEFAULT: FEATURE_ENGINEERED_METRICS_OUTPUT_PATH
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
    '''
    model_output_dir = os.path.dirname(model_output_path)
    metrics_output_dir = os.path.dirname(metrics_output_path)

    if model_output_dir != '':
        os.makedirs(model_output_dir, exist_ok=True)

    if metrics_output_dir != '':
        os.makedirs(metrics_output_dir, exist_ok=True)

    if save_model:
        joblib.dump(best_model, model_output_path)

    if save_metrics_json:
        metrics = make_json_safe(metrics)

        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    print('=' * 100)
    print('FEATURE-ENGINEERED MODEL SAVE COMPLETE')
    print('=' * 100)

    if save_model:
        print('Model saved to:', model_output_path)
    else:
        print('Model saving skipped.')

    if save_metrics_json:
        print('Metrics JSON saved to:', metrics_output_path)
    else:
        print('Metrics JSON saving skipped.')

    print()


def reproduce_model_from_json(
        metrics_json_filepath: str = FEATURE_ENGINEERED_METRICS_OUTPUT_PATH,
        model_output_path: str = '../Models/highest_snr_feature_engineered_random_forest_reproduced.joblib',
        save_model: bool = True
) -> tuple:
    '''
    About
    -----
    - Reproduces the engineered-feature RandomForestClassifier using a saved metrics JSON file
    - Loads the feature-engineered dataset, feature columns, train/test indices, and best parameters
    - Retrains the model on the exact same training rows
    - Evaluates on the exact same test rows

    Parameters
    ----------
    - metrics_json_filepath (str):
        - DEFAULT: FEATURE_ENGINEERED_METRICS_OUTPUT_PATH
        - Path to the saved model metrics JSON file

    - model_output_path (str):
        - DEFAULT: '../Models/highest_snr_feature_engineered_random_forest_reproduced.joblib'
        - Path where the reproduced model should be saved

    - save_model (bool):
        - DEFAULT: True
        - Whether to save the reproduced model

    Raises
    ------
    - FileNotFoundError:
        - If metrics_json_filepath or dataset filepath cannot be found

    Returns
    -------
    - tuple
        - reproduced_model
        - reproduction_metrics
    '''
    with open(metrics_json_filepath, 'r') as f:
        saved_metrics = json.load(f)

    feature_dataset_filepath = saved_metrics['dataset']['feature_dataset_filepath']
    feature_columns_used = saved_metrics['features_used']['feature_columns_used']
    train_indices = np.array(saved_metrics['split']['train_indices'])
    test_indices = np.array(saved_metrics['split']['test_indices'])
    random_state = saved_metrics['split']['random_state']
    best_params = saved_metrics['grid_search']['best_params']
    target_column = saved_metrics['dataset']['target_column']

    df_features = load_feature_engineered_dataset(
        feature_dataset_filepath=feature_dataset_filepath
    )

    X_features, y_labels, metadata_df, feature_columns_used = prepare_modeling_data(
        df_features=df_features,
        selected_feature_names=feature_columns_used,
        target_column=target_column,
        metadata_columns=METADATA_COLUMNS
    )

    X_train = X_features.iloc[train_indices]
    X_test = X_features.iloc[test_indices]
    y_train = y_labels.iloc[train_indices]
    y_test = y_labels.iloc[test_indices]

    reproduced_model = RandomForestClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=1,
        class_weight=None
    )

    print('=' * 100)
    print('REPRODUCING FEATURE-ENGINEERED RANDOM FOREST MODEL')
    print('=' * 100)
    print('Metrics JSON filepath:', metrics_json_filepath)
    print('Feature dataset filepath:', feature_dataset_filepath)
    print('Train rows:', len(train_indices))
    print('Test rows:', len(test_indices))
    print('Features used:', len(feature_columns_used))
    print('Random state:', random_state)
    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')
    print()

    reproduced_model.fit(X_train, y_train)

    y_pred = reproduced_model.predict(X_test)
    y_pred_proba = reproduced_model.predict_proba(X_test)

    reproduced_accuracy = accuracy_score(y_test, y_pred)
    reproduced_precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    reproduced_recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    reproduced_roc_auc_ovr_weighted = roc_auc_score(
        y_test,
        y_pred_proba,
        multi_class='ovr',
        average='weighted'
    )

    reproduction_metrics = {
        'model_type': 'RandomForestClassifier',
        'reproduction_source_json': metrics_json_filepath,
        'feature_dataset_filepath': feature_dataset_filepath,
        'features_used': feature_columns_used,
        'train_indices_used': train_indices.tolist(),
        'test_indices_used': test_indices.tolist(),
        'random_state': random_state,
        'best_params': best_params,
        'reproduced_test_accuracy': reproduced_accuracy,
        'reproduced_test_precision_weighted': reproduced_precision_weighted,
        'reproduced_test_recall_weighted': reproduced_recall_weighted,
        'reproduced_test_roc_auc_ovr_weighted': reproduced_roc_auc_ovr_weighted
    }

    reproduction_metrics = make_json_safe(reproduction_metrics)

    print('=' * 100)
    print('REPRODUCED MODEL PERFORMANCE')
    print('=' * 100)
    print(f'Accuracy:              {reproduced_accuracy:.4f}')
    print(f'Weighted Precision:    {reproduced_precision_weighted:.4f}')
    print(f'Weighted Recall:       {reproduced_recall_weighted:.4f}')
    print(f'Weighted ROC-AUC OVR:  {reproduced_roc_auc_ovr_weighted:.4f}')
    print()

    if save_model:
        model_output_dir = os.path.dirname(model_output_path)

        if model_output_dir != '':
            os.makedirs(model_output_dir, exist_ok=True)

        joblib.dump(reproduced_model, model_output_path)

        print('=' * 100)
        print('REPRODUCED MODEL SAVED')
        print('=' * 100)
        print('Model saved to:', model_output_path)
        print()

    return reproduced_model, reproduction_metrics

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def main():
    '''
    About
    -----
    - Runs the full feature-engineered modeling workflow
    - Loads the feature-engineered dataset
    - Loads top information-gain features from the exploration JSON
    - Trains a RandomForestClassifier with GridSearchCV
    - Evaluates the best model
    - Saves the best model and metrics JSON
    - WARNING: THIS MAY TAKE A WHILE

    Dependencies
    ------------
    - train_and_test_model
    - save_feature_engineered_model

    Parameters
    ----------
    - None

    Raises
    ------
    - FileNotFoundError:
        - If the feature-engineered dataset or selected feature JSON cannot be found

    Returns
    -------
    - None
    '''
    best_model, metrics, cv_results_df = train_and_test_model(
        feature_dataset_filepath=FEATURE_ENGINEERED_DATASET_PARQUET,
        selected_features_filepath=SELECTED_FEATURES_INPUT_PATH,
        mod_type_mapping=MOD_TYPE_MAPPING,
        target_column='modulation_id',
        test_size=0.20,
        random_state=35,
        cv=3,
        rfc_n_jobs=1,
        grid_search_n_jobs=-1,
        use_selected_features=True,
        save_metrics_json=True,
        metrics_output_path=FEATURE_ENGINEERED_METRICS_OUTPUT_PATH,
        cv_results_output_path=FEATURE_ENGINEERED_CV_RESULTS_OUTPUT_PATH
    )

    save_feature_engineered_model(
        best_model=best_model,
        metrics=metrics,
        model_output_path=FEATURE_ENGINEERED_MODEL_OUTPUT_PATH,
        metrics_output_path=FEATURE_ENGINEERED_METRICS_OUTPUT_PATH,
        save_model=True,
        save_metrics_json=True
    )

# =================================================================================================
# END Main Function
# =================================================================================================
