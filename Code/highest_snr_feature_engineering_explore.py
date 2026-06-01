'''
FILE OVERVIEW:
- Code and functions specifically for highest_snr_feature_engineering_explore.ipynb
- This is primarily for reproducibility and maintaining presentability of the explore notebook
- This file heavily pulls global variables and functions created in acquire.py
- This file explores the feature-engineered highest-SNR dataset created from the reduced SNR-30 .hdf5 file

=================================================

MISC COMMENTS:
- THIS ASSUMES YOU HAVE THE REDUCED DATASET '../Datasets/highest_snr_reduced_df.hdf5'
- THIS ASSUMES YOU HAVE THE FEATURE ENGINEERED DATASET '../Datasets/highest_snr_feature_engineered.parquet'
- The primary exploration interest is information gain / mutual information of each engineered feature
- The information-gain results can help identify which engineered features are most useful for later model training

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - vis_all_mod_types
    - load_feature_engineered_dataset
    - get_feature_columns
    - summarize_feature_engineered_dataset
    - check_feature_quality
    - calculate_information_gain
    - plot_top_information_gain_features
    - calculate_random_forest_feature_importance
    - combine_feature_importance_results
    - save_feature_exploration_results
    - get_top_feature_names
    - create_modeling_dataset_from_top_features
    - save_selected_feature_names
    - make_json_safe
    - save_engineered_feature_model
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
import matplotlib.pyplot as plt # Visualizations

# Feature exploration and model-related utilities
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Easy visualization of signals
from acquire import vis_signal

# ----- Global Variables --------------------------------------------------------------------------
from acquire import MOD_TYPE_MAPPING    # Modulation ID and Name mapping

# File pathings
HIGHEST_SNR_REDUCED_DF = '../Datasets/highest_snr_reduced_df.hdf5'
FEATURE_ENGINEERED_DATASET_PARQUET = '../Datasets/highest_snr_feature_engineered.parquet'
FEATURE_EXPLORATION_OUTPUT_PATH = '../Reports/highest_snr_feature_exploration.csv'
FEATURE_QUALITY_OUTPUT_PATH = '../Reports/highest_snr_feature_quality.csv'
SELECTED_FEATURES_OUTPUT_PATH = '../Reports/highest_snr_selected_features.json'
ENGINEERED_FEATURE_MODEL_OUTPUT_PATH = '../Models/highest_snr_feature_engineered_random_forest.joblib'
ENGINEERED_FEATURE_METRICS_OUTPUT_PATH = '../Models/highest_snr_feature_engineered_random_forest_metrics.json'

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
    """
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
        return obj.to_dict(orient='records')

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    return str(obj)


def vis_all_mod_types():
    """
    About
    -----
    - Visualizes a sample of each 24 modulation signal types

    Dependencies
    ------------
    - acquire.vis_signal
    - acquire.MOD_TYPE_MAPPING
    - ../Datasets/highest_snr_reduced_df.hdf5

    Parameters
    ----------
    - None

    Raises
    ------
    - None

    Returns
    -------
    - None
    """
    # Get the NAME of every modulation type
    mod_type_list = [MOD_TYPE_MAPPING[idx] for idx in MOD_TYPE_MAPPING]

    # Iterate through the modulation types to visualize
    for mod_type in mod_type_list:
        vis_signal(
            hdf5_data_filepath=HIGHEST_SNR_REDUCED_DF,
            mod_type=mod_type
        )

        # Nice visual separator between visualizations
        print('\033[35m\n=============================================\033[0m')


def load_feature_engineered_dataset(
        feature_dataset_filepath: str = FEATURE_ENGINEERED_DATASET_PARQUET
) -> pd.DataFrame:
    """
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
    """
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


def get_feature_columns(
        df_features: pd.DataFrame,
        metadata_columns: list = METADATA_COLUMNS
) -> list:
    """
    About
    -----
    - Identifies feature columns to use for exploration and later modeling
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
    """
    feature_columns = [
        col for col in df_features.columns
        if col not in metadata_columns
    ]

    print('Number of feature columns:', len(feature_columns))

    return feature_columns


def summarize_feature_engineered_dataset(
        df_features: pd.DataFrame,
        feature_columns: list = None
) -> None:
    """
    About
    -----
    - Prints a compact summary of the feature-engineered dataset
    - Includes shape, modulation distribution, SNR distribution, and feature count

    Parameters
    ----------
    - df_features (pd.DataFrame):
        - Feature-engineered dataset

    - feature_columns (list):
        - DEFAULT: None
        - List of engineered feature columns

    Raises
    ------
    - None

    Returns
    -------
    - None
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(df_features)

    print('=' * 100)
    print('FEATURE-ENGINEERED DATASET SUMMARY')
    print('=' * 100)
    print('Dataset shape:', df_features.shape)
    print('Number of rows/signals:', df_features.shape[0])
    print('Number of engineered features:', len(feature_columns))
    print()

    if 'modulation_type' in df_features.columns:
        print('Modulation distribution:')
        print(df_features['modulation_type'].value_counts().sort_index())
        print()

    if 'snr' in df_features.columns:
        print('SNR distribution:')
        print(df_features['snr'].value_counts().sort_index())
        print()

    print('First 5 rows:')
    print(df_features.head())
    print()


def check_feature_quality(
        df_features: pd.DataFrame,
        feature_columns: list,
        output_filepath: str = FEATURE_QUALITY_OUTPUT_PATH,
        save_results: bool = True
) -> pd.DataFrame:
    """
    About
    -----
    - Checks the quality of each engineered feature
    - Looks for missing values, infinite values, constant features, and basic summary statistics
    - Saves the feature quality report if requested

    Parameters
    ----------
    - df_features (pd.DataFrame):
        - Feature-engineered dataset

    - feature_columns (list):
        - List of engineered feature columns

    - output_filepath (str):
        - DEFAULT: FEATURE_QUALITY_OUTPUT_PATH
        - Filepath where the feature quality report should be saved

    - save_results (bool):
        - DEFAULT: True
        - Whether to save the feature quality report

    Raises
    ------
    - None

    Returns
    -------
    - pd.DataFrame
        - Feature quality report
    """
    results = []

    for feature in feature_columns:
        values = pd.to_numeric(df_features[feature], errors='coerce')

        num_missing = values.isna().sum()
        num_infinite = np.isinf(values.replace([np.inf, -np.inf], np.nan)).sum()
        num_unique = values.nunique(dropna=True)

        results.append({
            'feature': feature,
            'num_missing': int(num_missing),
            'num_infinite': int(num_infinite),
            'num_unique': int(num_unique),
            'is_constant': bool(num_unique <= 1),
            'mean': values.replace([np.inf, -np.inf], np.nan).mean(),
            'std': values.replace([np.inf, -np.inf], np.nan).std(),
            'min': values.replace([np.inf, -np.inf], np.nan).min(),
            'max': values.replace([np.inf, -np.inf], np.nan).max()
        })

    feature_quality_df = pd.DataFrame(results)

    print('=' * 100)
    print('FEATURE QUALITY CHECK')
    print('=' * 100)
    print('Total features checked:', len(feature_columns))
    print('Constant features:', int(feature_quality_df['is_constant'].sum()))
    print('Features with missing values:', int((feature_quality_df['num_missing'] > 0).sum()))
    print('Features with infinite values:', int((feature_quality_df['num_infinite'] > 0).sum()))
    print()
    print('Top feature quality issues:')
    print(
        feature_quality_df
        .sort_values(by=['num_missing', 'num_infinite', 'is_constant'], ascending=False)
        .head(15)
        .to_string(index=False)
    )
    print()

    if save_results:
        output_dir = os.path.dirname(output_filepath)

        if output_dir != '':
            os.makedirs(output_dir, exist_ok=True)

        feature_quality_df.to_csv(output_filepath, index=False)
        print('Feature quality report saved to:', output_filepath)
        print()

    return feature_quality_df


def calculate_information_gain(
        df_features: pd.DataFrame,
        feature_columns: list,
        target_column: str = 'modulation_id',
        random_state: int = 35
) -> pd.DataFrame:
    """
    About
    -----
    - Calculates information gain / mutual information between each engineered feature and modulation type
    - Higher values suggest the feature contains more useful information about the modulation label
    - This is the primary exploration metric before training on engineered features

    Parameters
    ----------
    - df_features (pd.DataFrame):
        - Feature-engineered dataset

    - feature_columns (list):
        - List of engineered feature columns

    - target_column (str):
        - DEFAULT: 'modulation_id'
        - Target label column

    - random_state (int):
        - DEFAULT: 35
        - Controls reproducibility of mutual information estimation

    Raises
    ------
    - KeyError:
        - If target_column is not found in df_features

    Returns
    -------
    - pd.DataFrame
        - Ranked information-gain results
    """
    if target_column not in df_features.columns:
        raise KeyError(f'{target_column} was not found in df_features')

    X = (
        df_features[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y = df_features[target_column]

    information_gain_scores = mutual_info_classif(
        X,
        y,
        random_state=random_state
    )

    information_gain_df = pd.DataFrame({
        'feature': feature_columns,
        'information_gain': information_gain_scores
    })

    information_gain_df = (
        information_gain_df
        .sort_values(by='information_gain', ascending=False)
        .reset_index(drop=True)
    )

    information_gain_df['information_gain_rank'] = np.arange(1, len(information_gain_df) + 1)

    print('=' * 100)
    print('TOP INFORMATION-GAIN FEATURES')
    print('=' * 100)
    print(information_gain_df.head(25).to_string(index=False))
    print()

    return information_gain_df


def plot_top_information_gain_features(
        information_gain_df: pd.DataFrame,
        top_n: int = 25,
        figsize: tuple = (12, 8)
) -> None:
    """
    About
    -----
    - Creates a horizontal bar chart of the top information-gain features
    - Useful for visually communicating which engineered features matter most

    Parameters
    ----------
    - information_gain_df (pd.DataFrame):
        - Ranked information-gain results

    - top_n (int):
        - DEFAULT: 25
        - Number of top features to plot

    - figsize (tuple):
        - DEFAULT: (12, 8)
        - Matplotlib figure size

    Raises
    ------
    - None

    Returns
    -------
    - None
    """
    top_features = information_gain_df.head(top_n).sort_values(
        by='information_gain',
        ascending=True
    )

    plt.figure(figsize=figsize)
    plt.barh(top_features['feature'], top_features['information_gain'])
    plt.xlabel('Information Gain / Mutual Information')
    plt.ylabel('Engineered Feature')
    plt.title(f'Top {top_n} Engineered Features by Information Gain')
    plt.tight_layout()
    plt.show()


def calculate_random_forest_feature_importance(
        df_features: pd.DataFrame,
        feature_columns: list,
        target_column: str = 'modulation_id',
        feature_dataset_filepath: str = FEATURE_ENGINEERED_DATASET_PARQUET,
        test_size: float = 0.20,
        random_state: int = 35,
        n_estimators: int = 200,
        n_jobs: int = 1
) -> tuple:
    """
    About
    -----
    - Trains a RandomForestClassifier on the engineered features
    - Calculates model-based feature importance as a secondary comparison to information gain
    - Prints core performance metrics to show whether engineered features are useful for modeling
    - Creates a JSON-ready metrics dictionary with train/test indices and reproducibility details

    Parameters
    ----------
    - df_features (pd.DataFrame):
        - Feature-engineered dataset

    - feature_columns (list):
        - List of engineered feature columns

    - target_column (str):
        - DEFAULT: 'modulation_id'
        - Target label column

    - feature_dataset_filepath (str):
        - DEFAULT: FEATURE_ENGINEERED_DATASET_PARQUET
        - Filepath to the feature-engineered dataset used for this model

    - test_size (float):
        - DEFAULT: 0.20
        - Test-set percentage

    - random_state (int):
        - DEFAULT: 35
        - Controls reproducibility

    - n_estimators (int):
        - DEFAULT: 200
        - Number of Random Forest trees

    - n_jobs (int):
        - DEFAULT: 1
        - Number of parallel jobs for RandomForestClassifier
        - 1 is preferred for maximum reproducibility

    Raises
    ------
    - KeyError:
        - If target_column is not found in df_features

    Returns
    -------
    - tuple
        - rf_model
        - feature_importance_df
        - model_metrics
    """
    if target_column not in df_features.columns:
        raise KeyError(f'{target_column} was not found in df_features')

    X = (
        df_features[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y = df_features[target_column]
    all_indices = np.arange(df_features.shape[0])

    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=None
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    test_recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    test_roc_auc_ovr_weighted = roc_auc_score(
        y_test,
        y_pred_proba,
        multi_class='ovr',
        average='weighted'
    )

    label_order = sorted(y.unique())
    target_names = [MOD_TYPE_MAPPING[label] for label in label_order]

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
        'feature': feature_columns,
        'random_forest_importance': rf_model.feature_importances_
    })

    feature_importance_df = (
        feature_importance_df
        .sort_values(by='random_forest_importance', ascending=False)
        .reset_index(drop=True)
    )

    feature_importance_df['random_forest_importance_rank'] = np.arange(
        1,
        len(feature_importance_df) + 1
    )

    model_metrics = {
        'project_stage': 'highest_snr_feature_engineering_explore',
        'model_type': 'RandomForestClassifier',
        'input_type': 'engineered_features',

        'dataset': {
            'feature_dataset_filepath': feature_dataset_filepath,
            'dataset_shape': list(df_features.shape),
            'num_rows': int(df_features.shape[0]),
            'num_features_used': int(len(feature_columns)),
            'target_column': target_column,
            'mod_type_mapping': MOD_TYPE_MAPPING
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

        'model_params': {
            'n_estimators': n_estimators,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'class_weight': None
        },

        'selected_features': {
            'feature_columns': feature_columns
        },

        'test_metrics': {
            'accuracy': float(test_accuracy),
            'precision_weighted': float(test_precision_weighted),
            'recall_weighted': float(test_recall_weighted),
            'roc_auc_ovr_weighted': float(test_roc_auc_ovr_weighted)
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

    model_metrics = make_json_safe(model_metrics)

    print('=' * 100)
    print('RANDOM FOREST ON ENGINEERED FEATURES')
    print('=' * 100)
    print('Feature dataset filepath:', feature_dataset_filepath)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print(f'Accuracy:              {test_accuracy:.4f}')
    print(f'Weighted Precision:    {test_precision_weighted:.4f}')
    print(f'Weighted Recall:       {test_recall_weighted:.4f}')
    print(f'Weighted ROC-AUC OVR:  {test_roc_auc_ovr_weighted:.4f}')
    print()
    print('Top Random Forest feature importances:')
    print(feature_importance_df.head(25).to_string(index=False))
    print()

    return rf_model, feature_importance_df, model_metrics


def combine_feature_importance_results(
        information_gain_df: pd.DataFrame,
        random_forest_importance_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    About
    -----
    - Combines information-gain results with optional Random Forest feature importance results
    - Creates a single ranked feature exploration table

    Parameters
    ----------
    - information_gain_df (pd.DataFrame):
        - Information-gain ranking results

    - random_forest_importance_df (pd.DataFrame):
        - DEFAULT: None
        - Random Forest feature importance results

    Raises
    ------
    - None

    Returns
    -------
    - pd.DataFrame
        - Combined feature exploration DataFrame
    """
    combined_df = information_gain_df.copy()

    if random_forest_importance_df is not None:
        combined_df = combined_df.merge(
            random_forest_importance_df,
            on='feature',
            how='left'
        )

    sort_columns = ['information_gain']

    if 'random_forest_importance' in combined_df.columns:
        sort_columns.append('random_forest_importance')

    combined_df = (
        combined_df
        .sort_values(by=sort_columns, ascending=False)
        .reset_index(drop=True)
    )

    return combined_df


def save_feature_exploration_results(
        combined_feature_results_df: pd.DataFrame,
        output_filepath: str = FEATURE_EXPLORATION_OUTPUT_PATH
) -> None:
    """
    About
    -----
    - Saves the feature exploration results
    - Default output is a CSV file for easy review and GitHub tracking

    Parameters
    ----------
    - combined_feature_results_df (pd.DataFrame):
        - Combined feature exploration results

    - output_filepath (str):
        - DEFAULT: FEATURE_EXPLORATION_OUTPUT_PATH
        - Output filepath for saved feature exploration results

    Raises
    ------
    - None

    Returns
    -------
    - None
    """
    output_dir = os.path.dirname(output_filepath)

    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    combined_feature_results_df.to_csv(output_filepath, index=False)

    print('=' * 100)
    print('FEATURE EXPLORATION RESULTS SAVED')
    print('=' * 100)
    print('Output filepath:', output_filepath)
    print('Saved shape:', combined_feature_results_df.shape)
    print()


def get_top_feature_names(
        information_gain_df: pd.DataFrame,
        top_n: int = 25
) -> list:
    """
    About
    -----
    - Gets the top feature names by information gain
    - Useful for selecting a smaller feature set for later model training

    Parameters
    ----------
    - information_gain_df (pd.DataFrame):
        - Ranked information-gain DataFrame

    - top_n (int):
        - DEFAULT: 25
        - Number of top feature names to return

    Raises
    ------
    - None

    Returns
    -------
    - list
        - Top feature names
    """
    top_feature_names = information_gain_df.head(top_n)['feature'].tolist()

    print(f'Top {top_n} feature names:')
    for idx, feature in enumerate(top_feature_names, start=1):
        print(f'{idx}. {feature}')

    return top_feature_names


def create_modeling_dataset_from_top_features(
        df_features: pd.DataFrame,
        top_feature_names: list,
        metadata_columns: list = METADATA_COLUMNS
) -> tuple:
    """
    About
    -----
    - Creates X and y objects for model training using selected engineered features
    - Keeps metadata separate from modeling features

    Parameters
    ----------
    - df_features (pd.DataFrame):
        - Feature-engineered dataset

    - top_feature_names (list):
        - Selected engineered feature names

    - metadata_columns (list):
        - DEFAULT: METADATA_COLUMNS
        - Metadata columns excluded from X

    Raises
    ------
    - KeyError:
        - If any selected feature is missing from df_features

    Returns
    -------
    - tuple
        - X_features
        - y_labels
        - metadata_df
    """
    missing_features = [feature for feature in top_feature_names if feature not in df_features.columns]

    if len(missing_features) > 0:
        raise KeyError(f'Selected features not found in df_features: {missing_features}')

    X_features = (
        df_features[top_feature_names]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y_labels = df_features['modulation_id']

    metadata_df = df_features[[col for col in metadata_columns if col in df_features.columns]].copy()

    print('=' * 100)
    print('MODELING DATASET CREATED FROM TOP FEATURES')
    print('=' * 100)
    print('X_features shape:', X_features.shape)
    print('y_labels shape:', y_labels.shape)
    print('metadata_df shape:', metadata_df.shape)
    print()

    return X_features, y_labels, metadata_df


def save_selected_feature_names(
        selected_feature_names: list,
        output_filepath: str = SELECTED_FEATURES_OUTPUT_PATH,
        selection_method: str = 'information_gain',
        top_n: int = 25
) -> None:
    """
    About
    -----
    - Saves selected feature names to a JSON file
    - This allows later model training files to reuse the same feature subset

    Parameters
    ----------
    - selected_feature_names (list):
        - Selected engineered feature names

    - output_filepath (str):
        - DEFAULT: SELECTED_FEATURES_OUTPUT_PATH
        - Path where selected feature names should be saved

    - selection_method (str):
        - DEFAULT: 'information_gain'
        - Description of how features were selected

    - top_n (int):
        - DEFAULT: 25
        - Number of selected features

    Raises
    ------
    - None

    Returns
    -------
    - None
    """
    output_dir = os.path.dirname(output_filepath)

    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    selected_features_payload = {
        'selection_method': selection_method,
        'top_n': top_n,
        'selected_feature_names': selected_feature_names
    }

    with open(output_filepath, 'w') as f:
        json.dump(selected_features_payload, f, indent=4)

    print('=' * 100)
    print('SELECTED FEATURE NAMES SAVED')
    print('=' * 100)
    print('Output filepath:', output_filepath)
    print('Number of selected features:', len(selected_feature_names))
    print()


def save_engineered_feature_model(
        model,
        model_metrics: dict,
        model_output_path: str = ENGINEERED_FEATURE_MODEL_OUTPUT_PATH,
        metrics_output_path: str = ENGINEERED_FEATURE_METRICS_OUTPUT_PATH,
        information_gain_df: pd.DataFrame = None,
        combined_feature_results_df: pd.DataFrame = None,
        selected_feature_names: list = None,
        save_model: bool = True,
        save_metrics_json: bool = True
) -> None:
    """
    About
    -----
    - Saves the engineered-feature RandomForestClassifier model
    - Saves the model metrics and reproducibility metadata as a JSON file
    - Optionally appends information-gain rankings, combined feature rankings, and selected features

    Parameters
    ----------
    - model:
        - Trained engineered-feature RandomForestClassifier model

    - model_metrics (dict):
        - Dictionary of model performance metrics and reproducibility metadata

    - model_output_path (str):
        - DEFAULT: ENGINEERED_FEATURE_MODEL_OUTPUT_PATH
        - Filepath where the trained model will be saved

    - metrics_output_path (str):
        - DEFAULT: ENGINEERED_FEATURE_METRICS_OUTPUT_PATH
        - Filepath where model metrics will be saved

    - information_gain_df (pd.DataFrame):
        - DEFAULT: None
        - Information-gain ranking results to include in the JSON

    - combined_feature_results_df (pd.DataFrame):
        - DEFAULT: None
        - Combined information-gain and Random Forest feature-ranking results to include in the JSON

    - selected_feature_names (list):
        - DEFAULT: None
        - Selected top feature names to include in the JSON

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
    model_output_dir = os.path.dirname(model_output_path)
    metrics_output_dir = os.path.dirname(metrics_output_path)

    if model_output_dir != '':
        os.makedirs(model_output_dir, exist_ok=True)

    if metrics_output_dir != '':
        os.makedirs(metrics_output_dir, exist_ok=True)

    metrics_payload = dict(model_metrics)

    if information_gain_df is not None:
        metrics_payload['information_gain_results'] = information_gain_df.to_dict(orient='records')

    if combined_feature_results_df is not None:
        metrics_payload['combined_feature_results'] = combined_feature_results_df.to_dict(orient='records')

    if selected_feature_names is not None:
        metrics_payload['selected_features']['selected_feature_names'] = selected_feature_names
        metrics_payload['selected_features']['selected_feature_count'] = len(selected_feature_names)
        metrics_payload['selected_features']['selection_method'] = 'information_gain'

    metrics_payload = make_json_safe(metrics_payload)

    if save_model:
        joblib.dump(model, model_output_path)

    if save_metrics_json:
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics_payload, f, indent=4)

    print('=' * 100)
    print('ENGINEERED-FEATURE MODEL SAVE COMPLETE')
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

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def main():
    """
    About
    -----
    - Runs the feature-engineering exploration workflow
    - Loads the feature-engineered highest-SNR dataset
    - Checks feature quality
    - Calculates information gain / mutual information for each engineered feature
    - Optionally compares information gain with Random Forest feature importance
    - Saves feature exploration results and selected top feature names

    Dependencies
    ------------
    - load_feature_engineered_dataset
    - get_feature_columns
    - summarize_feature_engineered_dataset
    - check_feature_quality
    - calculate_information_gain
    - calculate_random_forest_feature_importance
    - combine_feature_importance_results
    - save_feature_exploration_results
    - get_top_feature_names
    - save_selected_feature_names
    - save_engineered_feature_model

    Parameters
    ----------
    - None

    Raises
    ------
    - FileNotFoundError:
        - If the feature-engineered dataset cannot be found

    Returns
    -------
    - None
    """
    df_features = load_feature_engineered_dataset(
        feature_dataset_filepath=FEATURE_ENGINEERED_DATASET_PARQUET
    )

    feature_columns = get_feature_columns(df_features)

    summarize_feature_engineered_dataset(
        df_features=df_features,
        feature_columns=feature_columns
    )

    check_feature_quality(
        df_features=df_features,
        feature_columns=feature_columns,
        output_filepath=FEATURE_QUALITY_OUTPUT_PATH,
        save_results=True
    )

    information_gain_df = calculate_information_gain(
        df_features=df_features,
        feature_columns=feature_columns,
        target_column='modulation_id',
        random_state=35
    )

    rf_model, random_forest_importance_df, model_metrics = calculate_random_forest_feature_importance(
        df_features=df_features,
        feature_columns=feature_columns,
        target_column='modulation_id',
        feature_dataset_filepath=FEATURE_ENGINEERED_DATASET_PARQUET,
        test_size=0.20,
        random_state=35,
        n_estimators=200,
        n_jobs=1
    )

    combined_feature_results_df = combine_feature_importance_results(
        information_gain_df=information_gain_df,
        random_forest_importance_df=random_forest_importance_df
    )

    save_feature_exploration_results(
        combined_feature_results_df=combined_feature_results_df,
        output_filepath=FEATURE_EXPLORATION_OUTPUT_PATH
    )

    selected_feature_names = get_top_feature_names(
        information_gain_df=information_gain_df,
        top_n=25
    )

    save_selected_feature_names(
        selected_feature_names=selected_feature_names,
        output_filepath=SELECTED_FEATURES_OUTPUT_PATH,
        selection_method='information_gain',
        top_n=25
    )

    save_engineered_feature_model(
        model=rf_model,
        model_metrics=model_metrics,
        model_output_path=ENGINEERED_FEATURE_MODEL_OUTPUT_PATH,
        metrics_output_path=ENGINEERED_FEATURE_METRICS_OUTPUT_PATH,
        information_gain_df=information_gain_df,
        combined_feature_results_df=combined_feature_results_df,
        selected_feature_names=selected_feature_names,
        save_model=True,
        save_metrics_json=True
    )

# =================================================================================================
# END Main Function
# =================================================================================================
