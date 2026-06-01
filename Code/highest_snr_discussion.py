'''
FILE OVERVIEW:
- Code and functions specifically for highest_snr_discussion.ipynb
- This is primarily for reproducibility and maintaining presentability of the discussion notebook
- This file pulls together the saved artifacts from the highest_snr series:
    - baseline Random Forest on flattened raw I/Q
    - Random Forest on all engineered features
    - GridSearchCV Random Forest on selected top information-gain features
    - feature exploration / information-gain rankings
    - feature-quality outputs

=================================================

MISC COMMENTS:
- THIS ASSUMES YOU HAVE ALREADY RUN THE HIGHEST_SNR BASELINE, FEATURE ENGINEERING,
  FEATURE EXPLORATION, AND FEATURE MODELING STEPS
- The discussion notebook is intended to answer the major project questions:
    - How did raw I/Q baseline performance compare against engineered features?
    - How many engineered features were created?
    - How much performance was retained when training on only the top selected features?
    - Which engineered features were most useful by information gain?
    - Was the simpler selected-feature model a reasonable tradeoff?
- Feature counts are detected dynamically from the saved artifacts so the notebook remains accurate
  if the local feature-engineering file produces a different number of features

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - resolve_existing_path
    - load_json_file
    - load_csv_file
    - load_discussion_artifacts
    - extract_test_metrics
    - create_model_comparison_dataframe
    - create_feature_count_summary
    - create_performance_retention_dataframe
    - create_gridsearch_summary_dataframe
    - get_top_information_gain_features
    - get_top_random_forest_features
    - plot_model_performance_comparison
    - plot_feature_count_summary
    - plot_performance_retention
    - plot_top_information_gain_features
    - plot_top_random_forest_features
    - plot_gridsearch_top_results
    - create_discussion_summary_markdown
    - print_discussion_keypoints
    - save_discussion_outputs
- Main Function
'''
# ----- Imports -----------------------------------------------------------------------------------
import os
import json

import numpy as np              # Numerical operations
import pandas as pd             # Tabular summaries of metrics and feature rankings
import matplotlib.pyplot as plt # Visualizations

# ----- Global Variables --------------------------------------------------------------------------
BASELINE_METRICS_PATH = '../Models/highest_snr_random_forest_baseline_metrics.json'
ALL_FEATURE_METRICS_PATH = '../Models/highest_snr_feature_engineered_random_forest_metrics.json'
TOP_FEATURE_METRICS_PATH = '../Models/highest_snr_feature_engineered_random_forest_gridsearch_metrics.json'
GRIDSEARCH_CV_RESULTS_PATH = '../Reports/highest_snr_feature_engineered_random_forest_gridsearch_cv_results.csv'
FEATURE_EXPLORATION_PATH = '../Reports/highest_snr_feature_exploration.csv'
FEATURE_QUALITY_PATH = '../Reports/highest_snr_feature_quality.csv'
SELECTED_FEATURES_PATH = '../Reports/highest_snr_selected_features.json'
DISCUSSION_OUTPUT_DIR = '../Reports/discussion'

MODEL_LABELS = {
    'baseline': 'Raw I/Q RFC Baseline',
    'all_features': 'All Engineered Features RFC',
    'top_features': 'Top Selected Features RFC'
}

METRIC_COLUMNS = [
    'accuracy',
    'precision_weighted',
    'recall_weighted',
    'roc_auc_ovr_weighted'
]

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def resolve_existing_path(
        filepath: str
) -> str:
    '''
    About
    -----
    - Resolves a filepath for local project usage or uploaded-notebook usage
    - First checks the filepath as provided
    - Then checks the basename in the current working directory
    - Then checks the basename in /mnt/data for ChatGPT sandbox usage

    Parameters
    ----------
    - filepath (str):
        - Filepath to resolve

    Raises
    ------
    - FileNotFoundError:
        - If the file cannot be found in any expected location

    Returns
    -------
    - str
        - Resolved existing filepath
    '''
    candidate_paths = [
        filepath,
        os.path.basename(filepath),
        os.path.join('/mnt/data', os.path.basename(filepath))
    ]

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    raise FileNotFoundError(f'Could not find file: {filepath}')


def load_json_file(
        filepath: str
) -> dict:
    '''
    About
    -----
    - Loads a JSON file and returns the parsed dictionary

    Parameters
    ----------
    - filepath (str):
        - Path to JSON file

    Raises
    ------
    - FileNotFoundError:
        - If the filepath cannot be resolved

    Returns
    -------
    - dict
        - Parsed JSON dictionary
    '''
    resolved_filepath = resolve_existing_path(filepath)

    with open(resolved_filepath, 'r') as f:
        json_data = json.load(f)

    return json_data


def load_csv_file(
        filepath: str
) -> pd.DataFrame:
    '''
    About
    -----
    - Loads a CSV file and returns a pandas DataFrame

    Parameters
    ----------
    - filepath (str):
        - Path to CSV file

    Raises
    ------
    - FileNotFoundError:
        - If the filepath cannot be resolved

    Returns
    -------
    - pd.DataFrame
        - Loaded CSV data
    '''
    resolved_filepath = resolve_existing_path(filepath)

    return pd.read_csv(resolved_filepath)


def load_discussion_artifacts(
        baseline_metrics_path: str = BASELINE_METRICS_PATH,
        all_feature_metrics_path: str = ALL_FEATURE_METRICS_PATH,
        top_feature_metrics_path: str = TOP_FEATURE_METRICS_PATH,
        gridsearch_cv_results_path: str = GRIDSEARCH_CV_RESULTS_PATH,
        feature_exploration_path: str = FEATURE_EXPLORATION_PATH,
        feature_quality_path: str = FEATURE_QUALITY_PATH,
        selected_features_path: str = SELECTED_FEATURES_PATH
) -> dict:
    '''
    About
    -----
    - Loads all saved highest_snr artifacts needed for discussion
    - Returns a dictionary containing parsed JSON objects and CSV DataFrames

    Parameters
    ----------
    - baseline_metrics_path (str):
        - Path to raw I/Q baseline metrics JSON

    - all_feature_metrics_path (str):
        - Path to all-feature engineered Random Forest metrics JSON

    - top_feature_metrics_path (str):
        - Path to top-feature GridSearchCV model metrics JSON

    - gridsearch_cv_results_path (str):
        - Path to top-feature GridSearchCV CV results CSV

    - feature_exploration_path (str):
        - Path to feature exploration CSV

    - feature_quality_path (str):
        - Path to feature quality CSV

    - selected_features_path (str):
        - Path to selected features JSON

    Raises
    ------
    - FileNotFoundError:
        - If any required artifact cannot be found

    Returns
    -------
    - dict
        - Dictionary of loaded discussion artifacts
    '''
    artifacts = {
        'baseline_metrics': load_json_file(baseline_metrics_path),
        'all_feature_metrics': load_json_file(all_feature_metrics_path),
        'top_feature_metrics': load_json_file(top_feature_metrics_path),
        'gridsearch_cv_results': load_csv_file(gridsearch_cv_results_path),
        'feature_exploration': load_csv_file(feature_exploration_path),
        'feature_quality': load_csv_file(feature_quality_path),
        'selected_features': load_json_file(selected_features_path)
    }

    print('\033[32mDiscussion artifacts loaded successfully!\033[0m')
    print('Loaded artifacts:')
    for artifact_name in artifacts:
        print(f'- {artifact_name}')
    print()

    return artifacts


def extract_test_metrics(
        metrics_json: dict
) -> dict:
    '''
    About
    -----
    - Extracts the shared test metrics from a saved model metrics JSON

    Parameters
    ----------
    - metrics_json (dict):
        - Model metrics JSON dictionary

    Raises
    ------
    - KeyError:
        - If test_metrics is missing

    Returns
    -------
    - dict
        - Extracted test metrics
    '''
    if 'test_metrics' not in metrics_json:
        raise KeyError('Expected key "test_metrics" not found in metrics_json')

    return metrics_json['test_metrics']


def create_model_comparison_dataframe(
        artifacts: dict
) -> pd.DataFrame:
    '''
    About
    -----
    - Creates a comparison DataFrame for the three major highest_snr model stages
    - Compares:
        - Raw I/Q Random Forest baseline
        - All engineered feature Random Forest
        - Top selected engineered feature Random Forest

    Parameters
    ----------
    - artifacts (dict):
        - Dictionary returned by load_discussion_artifacts

    Raises
    ------
    - None

    Returns
    -------
    - pd.DataFrame
        - Model comparison table
    '''
    baseline_metrics = artifacts['baseline_metrics']
    all_feature_metrics = artifacts['all_feature_metrics']
    top_feature_metrics = artifacts['top_feature_metrics']

    baseline_test_metrics = extract_test_metrics(baseline_metrics)
    all_feature_test_metrics = extract_test_metrics(all_feature_metrics)
    top_feature_test_metrics = extract_test_metrics(top_feature_metrics)

    baseline_feature_count = baseline_metrics.get('dataset', {}).get('flattened_X_shape', [None, None])[-1]
    all_feature_count = all_feature_metrics.get('dataset', {}).get('num_features_used')
    top_feature_count = top_feature_metrics.get('features_used', {}).get('num_features_used')

    comparison_rows = [
        {
            'model_key': 'baseline',
            'model_name': MODEL_LABELS['baseline'],
            'input_type': baseline_metrics.get('input_type', 'flattened_raw_iq'),
            'num_features_used': baseline_feature_count,
            'accuracy': baseline_test_metrics.get('accuracy'),
            'precision_weighted': baseline_test_metrics.get('precision_weighted'),
            'recall_weighted': baseline_test_metrics.get('recall_weighted'),
            'roc_auc_ovr_weighted': baseline_test_metrics.get('roc_auc_ovr_weighted')
        },
        {
            'model_key': 'all_features',
            'model_name': MODEL_LABELS['all_features'],
            'input_type': all_feature_metrics.get('input_type', 'engineered_features'),
            'num_features_used': all_feature_count,
            'accuracy': all_feature_test_metrics.get('accuracy'),
            'precision_weighted': all_feature_test_metrics.get('precision_weighted'),
            'recall_weighted': all_feature_test_metrics.get('recall_weighted'),
            'roc_auc_ovr_weighted': all_feature_test_metrics.get('roc_auc_ovr_weighted')
        },
        {
            'model_key': 'top_features',
            'model_name': MODEL_LABELS['top_features'],
            'input_type': top_feature_metrics.get('input_type', 'selected_engineered_features'),
            'num_features_used': top_feature_count,
            'accuracy': top_feature_test_metrics.get('accuracy'),
            'precision_weighted': top_feature_test_metrics.get('precision_weighted'),
            'recall_weighted': top_feature_test_metrics.get('recall_weighted'),
            'roc_auc_ovr_weighted': top_feature_test_metrics.get('roc_auc_ovr_weighted')
        }
    ]

    model_comparison_df = pd.DataFrame(comparison_rows)

    return model_comparison_df


def create_feature_count_summary(
        artifacts: dict
) -> pd.DataFrame:
    '''
    About
    -----
    - Creates a compact feature-count summary
    - This is useful for discussing dimensionality reduction from all engineered features to top selected features

    Parameters
    ----------
    - artifacts (dict):
        - Dictionary returned by load_discussion_artifacts

    Raises
    ------
    - None

    Returns
    -------
    - pd.DataFrame
        - Feature count summary
    '''
    feature_quality_df = artifacts['feature_quality']
    all_feature_metrics = artifacts['all_feature_metrics']
    top_feature_metrics = artifacts['top_feature_metrics']
    selected_features = artifacts['selected_features']

    feature_quality_count = int(feature_quality_df.shape[0])
    all_feature_metric_count = all_feature_metrics.get('dataset', {}).get('num_features_used')
    top_feature_count = top_feature_metrics.get('features_used', {}).get('num_features_used')
    selected_feature_count = len(selected_features.get('selected_feature_names', []))

    # Use the most reliable available count for all engineered features.
    # The feature quality file has one row per engineered feature.
    all_engineered_count = feature_quality_count

    reduction_from_all_to_top = 1 - (selected_feature_count / all_engineered_count)

    feature_count_summary_df = pd.DataFrame([
        {
            'feature_set': 'Raw flattened I/Q baseline',
            'num_features': artifacts['baseline_metrics'].get('dataset', {}).get('flattened_X_shape', [None, None])[-1],
            'notes': '1024 I/Q samples flattened into 2048 raw values'
        },
        {
            'feature_set': 'All engineered features',
            'num_features': all_engineered_count,
            'notes': 'Detected from feature-quality artifact; one row per engineered feature'
        },
        {
            'feature_set': 'Top selected features',
            'num_features': selected_feature_count,
            'notes': f'Selected by {selected_features.get("selection_method", "information_gain")}'
        }
    ])

    feature_count_summary_df.attrs['all_feature_metric_count'] = all_feature_metric_count
    feature_count_summary_df.attrs['top_feature_metric_count'] = top_feature_count
    feature_count_summary_df.attrs['reduction_from_all_to_top'] = reduction_from_all_to_top

    return feature_count_summary_df


def create_performance_retention_dataframe(
        model_comparison_df: pd.DataFrame
) -> pd.DataFrame:
    '''
    About
    -----
    - Compares top selected-feature model performance against the all-feature engineered model
    - Calculates absolute metric drop and retained performance percentage

    Parameters
    ----------
    - model_comparison_df (pd.DataFrame):
        - DataFrame returned by create_model_comparison_dataframe

    Raises
    ------
    - ValueError:
        - If expected model rows are missing

    Returns
    -------
    - pd.DataFrame
        - Performance retention summary
    '''
    all_feature_row = model_comparison_df[model_comparison_df['model_key'] == 'all_features']
    top_feature_row = model_comparison_df[model_comparison_df['model_key'] == 'top_features']

    if all_feature_row.empty or top_feature_row.empty:
        raise ValueError('Expected all_features and top_features rows in model_comparison_df')

    all_feature_row = all_feature_row.iloc[0]
    top_feature_row = top_feature_row.iloc[0]

    retention_rows = []

    for metric in METRIC_COLUMNS:
        all_feature_value = all_feature_row[metric]
        top_feature_value = top_feature_row[metric]
        absolute_drop = all_feature_value - top_feature_value
        retained_ratio = top_feature_value / all_feature_value if all_feature_value != 0 else np.nan

        retention_rows.append({
            'metric': metric,
            'all_feature_model': all_feature_value,
            'top_feature_model': top_feature_value,
            'absolute_drop': absolute_drop,
            'absolute_drop_percentage_points': absolute_drop * 100,
            'retained_ratio': retained_ratio,
            'retained_percent': retained_ratio * 100
        })

    performance_retention_df = pd.DataFrame(retention_rows)

    return performance_retention_df


def create_gridsearch_summary_dataframe(
        artifacts: dict,
        top_n: int = 10
) -> pd.DataFrame:
    '''
    About
    -----
    - Creates a compact view of the top GridSearchCV results for the selected-feature model

    Parameters
    ----------
    - artifacts (dict):
        - Dictionary returned by load_discussion_artifacts

    - top_n (int):
        - DEFAULT: 10
        - Number of top GridSearchCV rows to return

    Raises
    ------
    - None

    Returns
    -------
    - pd.DataFrame
        - Top GridSearchCV results
    '''
    cv_results_df = artifacts['gridsearch_cv_results'].copy()

    display_columns = [
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

    available_display_columns = [col for col in display_columns if col in cv_results_df.columns]

    gridsearch_summary_df = (
        cv_results_df[available_display_columns]
        .sort_values(by='rank_test_score')
        .head(top_n)
        .reset_index(drop=True)
    )

    return gridsearch_summary_df


def get_top_information_gain_features(
        artifacts: dict,
        top_n: int = 25
) -> pd.DataFrame:
    '''
    About
    -----
    - Gets the top features by information gain / mutual information

    Parameters
    ----------
    - artifacts (dict):
        - Dictionary returned by load_discussion_artifacts

    - top_n (int):
        - DEFAULT: 25
        - Number of top features to return

    Raises
    ------
    - None

    Returns
    -------
    - pd.DataFrame
        - Top information-gain features
    '''
    feature_exploration_df = artifacts['feature_exploration'].copy()

    return (
        feature_exploration_df
        .sort_values(by='information_gain', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def get_top_random_forest_features(
        artifacts: dict,
        top_n: int = 25
) -> pd.DataFrame:
    '''
    About
    -----
    - Gets the top engineered features by Random Forest feature importance

    Parameters
    ----------
    - artifacts (dict):
        - Dictionary returned by load_discussion_artifacts

    - top_n (int):
        - DEFAULT: 25
        - Number of top features to return

    Raises
    ------
    - None

    Returns
    -------
    - pd.DataFrame
        - Top Random Forest feature-importance features
    '''
    feature_exploration_df = artifacts['feature_exploration'].copy()

    return (
        feature_exploration_df
        .sort_values(by='random_forest_importance', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_model_performance_comparison(
        model_comparison_df: pd.DataFrame,
        metrics: list = None,
        figsize: tuple = (12, 6)
) -> None:
    '''
    About
    -----
    - Plots model performance metrics across the highest_snr model series

    Parameters
    ----------
    - model_comparison_df (pd.DataFrame):
        - DataFrame returned by create_model_comparison_dataframe

    - metrics (list):
        - DEFAULT: None
        - Metrics to plot

    - figsize (tuple):
        - DEFAULT: (12, 6)
        - Matplotlib figure size

    Raises
    ------
    - None

    Returns
    -------
    - None
    '''
    if metrics is None:
        metrics = METRIC_COLUMNS

    plot_df = model_comparison_df.set_index('model_name')[metrics]

    ax = plot_df.plot(kind='bar', figsize=figsize)
    ax.set_title('Highest-SNR Model Performance Comparison')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.show()


def plot_feature_count_summary(
        feature_count_summary_df: pd.DataFrame,
        figsize: tuple = (10, 5)
) -> None:
    '''
    About
    -----
    - Plots number of input features used by each major model/feature set

    Parameters
    ----------
    - feature_count_summary_df (pd.DataFrame):
        - DataFrame returned by create_feature_count_summary

    - figsize (tuple):
        - DEFAULT: (10, 5)
        - Matplotlib figure size

    Raises
    ------
    - None

    Returns
    -------
    - None
    '''
    ax = feature_count_summary_df.plot(
        x='feature_set',
        y='num_features',
        kind='bar',
        figsize=figsize,
        legend=False
    )

    ax.set_title('Feature Count Reduction Across Highest-SNR Series')
    ax.set_xlabel('Feature Set')
    ax.set_ylabel('Number of Input Features')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.show()


def plot_performance_retention(
        performance_retention_df: pd.DataFrame,
        figsize: tuple = (10, 5)
) -> None:
    '''
    About
    -----
    - Plots how much performance the top selected-feature model retained relative to all features

    Parameters
    ----------
    - performance_retention_df (pd.DataFrame):
        - DataFrame returned by create_performance_retention_dataframe

    - figsize (tuple):
        - DEFAULT: (10, 5)
        - Matplotlib figure size

    Raises
    ------
    - None

    Returns
    -------
    - None
    '''
    ax = performance_retention_df.plot(
        x='metric',
        y='retained_percent',
        kind='bar',
        figsize=figsize,
        legend=False
    )

    ax.axhline(95, linestyle='--')
    ax.set_title('Performance Retained by Top Selected Features vs All Engineered Features')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Retained Performance (%)')
    ax.set_ylim(0, 105)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.show()


def plot_top_information_gain_features(
        top_information_gain_df: pd.DataFrame,
        top_n: int = 25,
        figsize: tuple = (12, 8)
) -> None:
    '''
    About
    -----
    - Plots the top information-gain features

    Parameters
    ----------
    - top_information_gain_df (pd.DataFrame):
        - Top information-gain features

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
    '''
    plot_df = (
        top_information_gain_df
        .head(top_n)
        .sort_values(by='information_gain', ascending=True)
    )

    plt.figure(figsize=figsize)
    plt.barh(plot_df['feature'], plot_df['information_gain'])
    plt.xlabel('Information Gain / Mutual Information')
    plt.ylabel('Engineered Feature')
    plt.title(f'Top {top_n} Features by Information Gain')
    plt.tight_layout()
    plt.show()


def plot_top_random_forest_features(
        top_random_forest_df: pd.DataFrame,
        top_n: int = 25,
        figsize: tuple = (12, 8)
) -> None:
    '''
    About
    -----
    - Plots the top Random Forest feature importances

    Parameters
    ----------
    - top_random_forest_df (pd.DataFrame):
        - Top Random Forest feature-importance features

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
    '''
    plot_df = (
        top_random_forest_df
        .head(top_n)
        .sort_values(by='random_forest_importance', ascending=True)
    )

    plt.figure(figsize=figsize)
    plt.barh(plot_df['feature'], plot_df['random_forest_importance'])
    plt.xlabel('Random Forest Feature Importance')
    plt.ylabel('Engineered Feature')
    plt.title(f'Top {top_n} Features by Random Forest Importance')
    plt.tight_layout()
    plt.show()


def plot_gridsearch_top_results(
        gridsearch_summary_df: pd.DataFrame,
        figsize: tuple = (10, 5)
) -> None:
    '''
    About
    -----
    - Plots top GridSearchCV mean test scores for the selected-feature model

    Parameters
    ----------
    - gridsearch_summary_df (pd.DataFrame):
        - DataFrame returned by create_gridsearch_summary_dataframe

    - figsize (tuple):
        - DEFAULT: (10, 5)
        - Matplotlib figure size

    Raises
    ------
    - None

    Returns
    -------
    - None
    '''
    plot_df = gridsearch_summary_df.copy()
    plot_df['model_rank'] = 'Rank ' + plot_df['rank_test_score'].astype(str)

    ax = plot_df.plot(
        x='model_rank',
        y='mean_test_score',
        kind='bar',
        figsize=figsize,
        legend=False
    )

    ax.set_title('Top GridSearchCV Selected-Feature RFC Results')
    ax.set_xlabel('GridSearchCV Rank')
    ax.set_ylabel('Mean CV ROC-AUC OVR Weighted')
    ax.set_ylim(max(0, plot_df['mean_test_score'].min() - 0.01), 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def create_discussion_summary_markdown(
        model_comparison_df: pd.DataFrame,
        feature_count_summary_df: pd.DataFrame,
        performance_retention_df: pd.DataFrame,
        artifacts: dict,
        expected_engineered_feature_count: int = None
) -> str:
    '''
    About
    -----
    - Creates a markdown-formatted discussion summary based on the current artifacts
    - Intended for display in highest_snr_discussion.ipynb

    Parameters
    ----------
    - model_comparison_df (pd.DataFrame):
        - DataFrame returned by create_model_comparison_dataframe

    - feature_count_summary_df (pd.DataFrame):
        - DataFrame returned by create_feature_count_summary

    - performance_retention_df (pd.DataFrame):
        - DataFrame returned by create_performance_retention_dataframe

    - artifacts (dict):
        - Dictionary returned by load_discussion_artifacts

    - expected_engineered_feature_count (int):
        - DEFAULT: None
        - Optional expected feature count for discussion context

    Raises
    ------
    - None

    Returns
    -------
    - str
        - Markdown discussion summary
    '''
    all_features_row = model_comparison_df[model_comparison_df['model_key'] == 'all_features'].iloc[0]
    top_features_row = model_comparison_df[model_comparison_df['model_key'] == 'top_features'].iloc[0]
    baseline_row = model_comparison_df[model_comparison_df['model_key'] == 'baseline'].iloc[0]

    all_engineered_count = int(
        feature_count_summary_df[feature_count_summary_df['feature_set'] == 'All engineered features']['num_features'].iloc[0]
    )
    selected_feature_count = int(
        feature_count_summary_df[feature_count_summary_df['feature_set'] == 'Top selected features']['num_features'].iloc[0]
    )

    feature_reduction_percent = (1 - (selected_feature_count / all_engineered_count)) * 100

    accuracy_retention = performance_retention_df[
        performance_retention_df['metric'] == 'accuracy'
    ]['retained_percent'].iloc[0]

    roc_retention = performance_retention_df[
        performance_retention_df['metric'] == 'roc_auc_ovr_weighted'
    ]['retained_percent'].iloc[0]

    accuracy_drop_pp = performance_retention_df[
        performance_retention_df['metric'] == 'accuracy'
    ]['absolute_drop_percentage_points'].iloc[0]

    roc_drop_pp = performance_retention_df[
        performance_retention_df['metric'] == 'roc_auc_ovr_weighted'
    ]['absolute_drop_percentage_points'].iloc[0]

    selected_features_payload = artifacts['selected_features']
    selection_method = selected_features_payload.get('selection_method', 'information_gain')

    expected_feature_note = ''
    if expected_engineered_feature_count is not None and expected_engineered_feature_count != all_engineered_count:
        expected_feature_note = (
            f'\n\n> Note: The discussion prompt referenced {expected_engineered_feature_count} engineered features. '
            f'The currently loaded artifacts contain {all_engineered_count} engineered features. '
            'This notebook reports the artifact-derived value so rerunning with updated local artifacts will update the count automatically.'
        )

    markdown_summary = f'''
## Highest-SNR Series Discussion Summary

The highest-SNR modeling series compared three progressively more interpretable Random Forest approaches:

1. **Raw I/Q baseline:** flattened each signal from `(1024, 2)` into `{int(baseline_row['num_features_used'])}` raw input features.
2. **All engineered features:** trained on `{all_engineered_count}` engineered features derived from I/Q statistics, magnitude, power, phase, instantaneous frequency, FFT/spectral behavior, and constellation geometry.
3. **Top selected features:** trained on the top `{selected_feature_count}` engineered features selected by `{selection_method}`.

The all-feature engineered model performed best overall, with accuracy of `{all_features_row['accuracy']:.4f}` and ROC-AUC of `{all_features_row['roc_auc_ovr_weighted']:.4f}`. The top-{selected_feature_count} model reduced the feature set by approximately `{feature_reduction_percent:.2f}%` while retaining `{accuracy_retention:.2f}%` of the all-feature accuracy and `{roc_retention:.2f}%` of the all-feature ROC-AUC.

In practical terms, the selected-feature model lost about `{accuracy_drop_pp:.2f}` percentage points of accuracy and `{roc_drop_pp:.2f}` percentage points of ROC-AUC. This suggests that a relatively compact feature subset preserves most of the discriminative information, even though the full engineered-feature model remains the strongest performer.

The major interpretation is that the feature engineering step provided a large improvement over the raw flattened I/Q Random Forest baseline. The top selected features then showed that much of that performance can be maintained with fewer, more interpretable features.{expected_feature_note}
'''

    return markdown_summary


def print_discussion_keypoints(
        model_comparison_df: pd.DataFrame,
        feature_count_summary_df: pd.DataFrame,
        performance_retention_df: pd.DataFrame
) -> None:
    '''
    About
    -----
    - Prints key project discussion points in a compact format

    Parameters
    ----------
    - model_comparison_df (pd.DataFrame):
        - DataFrame returned by create_model_comparison_dataframe

    - feature_count_summary_df (pd.DataFrame):
        - DataFrame returned by create_feature_count_summary

    - performance_retention_df (pd.DataFrame):
        - DataFrame returned by create_performance_retention_dataframe

    Raises
    ------
    - None

    Returns
    -------
    - None
    '''
    all_features = model_comparison_df[model_comparison_df['model_key'] == 'all_features'].iloc[0]
    top_features = model_comparison_df[model_comparison_df['model_key'] == 'top_features'].iloc[0]
    baseline = model_comparison_df[model_comparison_df['model_key'] == 'baseline'].iloc[0]

    all_count = int(feature_count_summary_df[feature_count_summary_df['feature_set'] == 'All engineered features']['num_features'].iloc[0])
    top_count = int(feature_count_summary_df[feature_count_summary_df['feature_set'] == 'Top selected features']['num_features'].iloc[0])
    reduction_percent = (1 - top_count / all_count) * 100

    print('=' * 100)
    print('DISCUSSION KEYPOINTS')
    print('=' * 100)
    print(f'Raw I/Q RFC baseline accuracy:      {baseline["accuracy"]:.4f}')
    print(f'All engineered features accuracy:  {all_features["accuracy"]:.4f}')
    print(f'Top selected features accuracy:    {top_features["accuracy"]:.4f}')
    print()
    print(f'All engineered feature count:      {all_count}')
    print(f'Top selected feature count:        {top_count}')
    print(f'Feature reduction:                 {reduction_percent:.2f}%')
    print()
    print('Top selected-feature retention relative to all engineered features:')
    print(performance_retention_df[['metric', 'retained_percent', 'absolute_drop_percentage_points']].to_string(index=False))
    print()


def save_discussion_outputs(
        model_comparison_df: pd.DataFrame,
        feature_count_summary_df: pd.DataFrame,
        performance_retention_df: pd.DataFrame,
        gridsearch_summary_df: pd.DataFrame,
        output_dir: str = DISCUSSION_OUTPUT_DIR
) -> None:
    '''
    About
    -----
    - Saves discussion tables to CSV so the final notebook conclusions can be reproduced

    Parameters
    ----------
    - model_comparison_df (pd.DataFrame):
        - Model comparison table

    - feature_count_summary_df (pd.DataFrame):
        - Feature count summary table

    - performance_retention_df (pd.DataFrame):
        - Performance retention table

    - gridsearch_summary_df (pd.DataFrame):
        - GridSearchCV summary table

    - output_dir (str):
        - DEFAULT: DISCUSSION_OUTPUT_DIR
        - Directory where discussion outputs should be saved

    Raises
    ------
    - None

    Returns
    -------
    - None
    '''
    os.makedirs(output_dir, exist_ok=True)

    model_comparison_df.to_csv(
        os.path.join(output_dir, 'highest_snr_model_comparison.csv'),
        index=False
    )

    feature_count_summary_df.to_csv(
        os.path.join(output_dir, 'highest_snr_feature_count_summary.csv'),
        index=False
    )

    performance_retention_df.to_csv(
        os.path.join(output_dir, 'highest_snr_performance_retention.csv'),
        index=False
    )

    gridsearch_summary_df.to_csv(
        os.path.join(output_dir, 'highest_snr_gridsearch_summary.csv'),
        index=False
    )

    print('=' * 100)
    print('DISCUSSION OUTPUTS SAVED')
    print('=' * 100)
    print('Output directory:', output_dir)
    print()

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def main():
    '''
    About
    -----
    - Runs the full highest_snr discussion workflow
    - Loads saved model/feature artifacts
    - Creates model comparison, feature count, performance retention, and GridSearchCV summaries
    - Saves discussion tables
    - Prints key discussion points

    Dependencies
    ------------
    - load_discussion_artifacts
    - create_model_comparison_dataframe
    - create_feature_count_summary
    - create_performance_retention_dataframe
    - create_gridsearch_summary_dataframe
    - save_discussion_outputs
    - print_discussion_keypoints

    Parameters
    ----------
    - None

    Raises
    ------
    - FileNotFoundError:
        - If required discussion artifacts cannot be found

    Returns
    -------
    - tuple
        - artifacts
        - model_comparison_df
        - feature_count_summary_df
        - performance_retention_df
        - gridsearch_summary_df
    '''
    artifacts = load_discussion_artifacts()

    model_comparison_df = create_model_comparison_dataframe(artifacts)
    feature_count_summary_df = create_feature_count_summary(artifacts)
    performance_retention_df = create_performance_retention_dataframe(model_comparison_df)
    gridsearch_summary_df = create_gridsearch_summary_dataframe(artifacts)

    print_discussion_keypoints(
        model_comparison_df=model_comparison_df,
        feature_count_summary_df=feature_count_summary_df,
        performance_retention_df=performance_retention_df
    )

    save_discussion_outputs(
        model_comparison_df=model_comparison_df,
        feature_count_summary_df=feature_count_summary_df,
        performance_retention_df=performance_retention_df,
        gridsearch_summary_df=gridsearch_summary_df
    )

    return artifacts, model_comparison_df, feature_count_summary_df, performance_retention_df, gridsearch_summary_df


if __name__ == '__main__':
    main()

# =================================================================================================
# END Main Function
# =================================================================================================
