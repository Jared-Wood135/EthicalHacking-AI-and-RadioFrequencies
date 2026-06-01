'''
FILE OVERVIEW:
- Code and functions specifically for feature engineering the reduced highest-SNR dataset
- This file assumes the reduced dataset exists at '../Datasets/highest_snr_reduced_df.hdf5'
- The goal is to convert raw I/Q signal frames into tabular engineered features

=================================================

MISC COMMENTS:
- THIS ASSUMES YOU HAVE THE REDUCED DATASET '../Datasets/highest_snr_reduced_df.hdf5'
- This does NOT save raw I/Q arrays into pandas
- Each row of the output dataset represents one signal
- Each column represents metadata or an engineered feature
- The reduced dataset should only contain the highest-SNR signals, likely SNR = 30

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - safe_divide
    - safe_skew
    - safe_kurtosis
    - extract_basic_iq_features
    - extract_magnitude_power_features
    - extract_phase_frequency_features
    - extract_fft_spectral_features
    - extract_constellation_features
    - extract_all_signal_features
    - create_feature_engineered_dataset
    - save_feature_engineered_dataset
- Main Function
'''
# ----- Imports -----------------------------------------------------------------------------------
import os           # File saving

import numpy as np  # Numerical operations and signal math
import pandas as pd # Feature-engineered tabular dataset
import h5py         # Reading .hdf5 datasets

# ----- Global Variables --------------------------------------------------------------------------
from acquire import MOD_TYPE_MAPPING    # Modulation ID and name mapping

# File pathings
HIGHEST_SNR_REDUCED_DF = '../Datasets/highest_snr_reduced_df.hdf5'
FEATURE_ENGINEERED_DATASET_PARQUET = '../Datasets/highest_snr_feature_engineered.parquet'
FEATURE_ENGINEERED_DATASET_HDF5 = '../Datasets/highest_snr_feature_engineered_df.hdf5'
FEATURE_ENGINEERED_DATASET_CSV = '../Datasets/highest_snr_feature_engineered.csv'

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def safe_divide(
        numerator: float,
        denominator: float,
        default_value: float = 0.0
) -> float:
    '''
    About
    -----
    - Safely divides two values
    - Prevents divide-by-zero errors during feature engineering

    Parameters
    ----------
    - numerator (float):
        - Value in the numerator

    - denominator (float):
        - Value in the denominator

    - default_value (float):
        - DEFAULT: 0.0
        - Value returned if denominator is zero

    Raises
    ------
    - None

    Returns
    -------
    - float
        - Division result or default value
    '''
    if denominator == 0:
        return default_value

    return numerator / denominator


def safe_skew(
        values: np.ndarray
) -> float:
    '''
    About
    -----
    - Calculates skewness without requiring scipy
    - Skewness helps describe asymmetry in a signal distribution

    Parameters
    ----------
    - values (np.ndarray):
        - Array of numeric values

    Raises
    ------
    - None

    Returns
    -------
    - float
        - Skewness value
    '''
    mean_value = np.mean(values)
    std_value = np.std(values)

    if std_value == 0:
        return 0.0

    skew_value = np.mean(((values - mean_value) / std_value) ** 3)

    return float(skew_value)


def safe_kurtosis(
        values: np.ndarray
) -> float:
    '''
    About
    -----
    - Calculates excess kurtosis without requiring scipy
    - Kurtosis helps describe how heavy-tailed or peaked a distribution is

    Parameters
    ----------
    - values (np.ndarray):
        - Array of numeric values

    Raises
    ------
    - None

    Returns
    -------
    - float
        - Excess kurtosis value
    '''
    mean_value = np.mean(values)
    std_value = np.std(values)

    if std_value == 0:
        return 0.0

    kurtosis_value = np.mean(((values - mean_value) / std_value) ** 4) - 3

    return float(kurtosis_value)


def extract_basic_iq_features(
        signal: np.ndarray
) -> dict:
    '''
    About
    -----
    - Extracts basic statistical features directly from I and Q
    - These features summarize the raw in-phase and quadrature signal values

    Parameters
    ----------
    - signal (np.ndarray):
        - One signal frame with shape (1024, 2)
        - signal[:, 0] represents I values
        - signal[:, 1] represents Q values

    Raises
    ------
    - None

    Returns
    -------
    - dict
        - Basic I/Q feature dictionary
    '''
    I = signal[:, 0]
    Q = signal[:, 1]

    features = {
        # I-channel statistics
        'mean_I': np.mean(I),
        'std_I': np.std(I),
        'min_I': np.min(I),
        'max_I': np.max(I),
        'median_I': np.median(I),
        'range_I': np.max(I) - np.min(I),
        'q25_I': np.percentile(I, 25),
        'q75_I': np.percentile(I, 75),
        'iqr_I': np.percentile(I, 75) - np.percentile(I, 25),
        'skew_I': safe_skew(I),
        'kurtosis_I': safe_kurtosis(I),

        # Q-channel statistics
        'mean_Q': np.mean(Q),
        'std_Q': np.std(Q),
        'min_Q': np.min(Q),
        'max_Q': np.max(Q),
        'median_Q': np.median(Q),
        'range_Q': np.max(Q) - np.min(Q),
        'q25_Q': np.percentile(Q, 25),
        'q75_Q': np.percentile(Q, 75),
        'iqr_Q': np.percentile(Q, 75) - np.percentile(Q, 25),
        'skew_Q': safe_skew(Q),
        'kurtosis_Q': safe_kurtosis(Q),

        # I/Q relationship
        'mean_abs_I': np.mean(np.abs(I)),
        'mean_abs_Q': np.mean(np.abs(Q)),
        'std_ratio_I_Q': safe_divide(np.std(I), np.std(Q)),
        'mean_abs_ratio_I_Q': safe_divide(np.mean(np.abs(I)), np.mean(np.abs(Q))),
        'corr_I_Q': np.corrcoef(I, Q)[0, 1] if np.std(I) != 0 and np.std(Q) != 0 else 0.0
    }

    return features


def extract_magnitude_power_features(
        signal: np.ndarray
) -> dict:
    '''
    About
    -----
    - Extracts magnitude and power features from complex I/Q
    - These are useful because some modulation types differ strongly by amplitude behavior

    Parameters
    ----------
    - signal (np.ndarray):
        - One signal frame with shape (1024, 2)

    Raises
    ------
    - None

    Returns
    -------
    - dict
        - Magnitude and power feature dictionary
    '''
    I = signal[:, 0]
    Q = signal[:, 1]

    iq = I + 1j * Q

    magnitude = np.abs(iq)
    power = magnitude ** 2

    mean_power = np.mean(power)
    peak_power = np.max(power)

    features = {
        # Magnitude features
        'mean_magnitude': np.mean(magnitude),
        'std_magnitude': np.std(magnitude),
        'min_magnitude': np.min(magnitude),
        'max_magnitude': np.max(magnitude),
        'median_magnitude': np.median(magnitude),
        'range_magnitude': np.max(magnitude) - np.min(magnitude),
        'q25_magnitude': np.percentile(magnitude, 25),
        'q75_magnitude': np.percentile(magnitude, 75),
        'iqr_magnitude': np.percentile(magnitude, 75) - np.percentile(magnitude, 25),
        'skew_magnitude': safe_skew(magnitude),
        'kurtosis_magnitude': safe_kurtosis(magnitude),

        # Power features
        'mean_power': mean_power,
        'std_power': np.std(power),
        'min_power': np.min(power),
        'max_power': peak_power,
        'median_power': np.median(power),
        'total_energy': np.sum(power),
        'rms_power': np.sqrt(mean_power),
        'peak_to_average_power_ratio': safe_divide(peak_power, mean_power),

        # Normalized amplitude behavior
        'magnitude_cv': safe_divide(np.std(magnitude), np.mean(magnitude)),
        'power_cv': safe_divide(np.std(power), mean_power)
    }

    return features


def extract_phase_frequency_features(
        signal: np.ndarray
) -> dict:
    '''
    About
    -----
    - Extracts phase and instantaneous-frequency-like features from complex I/Q
    - These are useful because PSK, FM, and related modulations often differ in phase behavior

    Parameters
    ----------
    - signal (np.ndarray):
        - One signal frame with shape (1024, 2)

    Raises
    ------
    - None

    Returns
    -------
    - dict
        - Phase and instantaneous frequency feature dictionary
    '''
    I = signal[:, 0]
    Q = signal[:, 1]

    iq = I + 1j * Q

    phase = np.angle(iq)
    unwrapped_phase = np.unwrap(phase)

    # Approximate instantaneous frequency using phase difference
    inst_freq = np.diff(unwrapped_phase)

    # Circular phase statistics
    circular_mean_complex = np.mean(np.exp(1j * phase))
    circular_mean_phase = np.angle(circular_mean_complex)
    circular_resultant_length = np.abs(circular_mean_complex)

    features = {
        # Raw phase statistics
        'mean_phase': np.mean(phase),
        'std_phase': np.std(phase),
        'min_phase': np.min(phase),
        'max_phase': np.max(phase),
        'range_phase': np.max(phase) - np.min(phase),

        # Circular phase statistics
        'circular_mean_phase': circular_mean_phase,
        'circular_resultant_length': circular_resultant_length,

        # Unwrapped phase behavior
        'mean_unwrapped_phase': np.mean(unwrapped_phase),
        'std_unwrapped_phase': np.std(unwrapped_phase),
        'unwrapped_phase_range': np.max(unwrapped_phase) - np.min(unwrapped_phase),

        # Instantaneous frequency approximation
        'mean_inst_freq': np.mean(inst_freq),
        'std_inst_freq': np.std(inst_freq),
        'min_inst_freq': np.min(inst_freq),
        'max_inst_freq': np.max(inst_freq),
        'median_inst_freq': np.median(inst_freq),
        'range_inst_freq': np.max(inst_freq) - np.min(inst_freq),
        'q25_inst_freq': np.percentile(inst_freq, 25),
        'q75_inst_freq': np.percentile(inst_freq, 75),
        'iqr_inst_freq': np.percentile(inst_freq, 75) - np.percentile(inst_freq, 25),
        'skew_inst_freq': safe_skew(inst_freq),
        'kurtosis_inst_freq': safe_kurtosis(inst_freq)
    }

    return features


def extract_fft_spectral_features(
        signal: np.ndarray
) -> dict:
    '''
    About
    -----
    - Extracts frequency-domain features using FFT
    - These features summarize where signal energy appears in normalized frequency
    - These can be helpful for OFDM, FM, and other spectrally distinctive modulations

    Parameters
    ----------
    - signal (np.ndarray):
        - One signal frame with shape (1024, 2)

    Raises
    ------
    - None

    Returns
    -------
    - dict
        - FFT and spectral feature dictionary
    '''
    I = signal[:, 0]
    Q = signal[:, 1]

    iq = I + 1j * Q

    fft_values = np.fft.fftshift(np.fft.fft(iq))
    fft_magnitude = np.abs(fft_values)
    fft_power = fft_magnitude ** 2

    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq)))

    total_fft_power = np.sum(fft_power) + 1e-12
    normalized_power = fft_power / total_fft_power

    spectral_centroid = np.sum(freqs * normalized_power)
    spectral_spread = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * normalized_power)
    )

    spectral_entropy = -np.sum(
        normalized_power * np.log2(normalized_power + 1e-12)
    )

    geometric_mean_power = np.exp(np.mean(np.log(fft_power + 1e-12)))
    arithmetic_mean_power = np.mean(fft_power + 1e-12)
    spectral_flatness = safe_divide(geometric_mean_power, arithmetic_mean_power)

    peak_index = np.argmax(fft_power)
    dominant_freq = freqs[peak_index]
    peak_fft_power = fft_power[peak_index]

    cumulative_power = np.cumsum(normalized_power)
    rolloff_85_index = np.searchsorted(cumulative_power, 0.85)
    rolloff_95_index = np.searchsorted(cumulative_power, 0.95)

    rolloff_85_freq = freqs[min(rolloff_85_index, len(freqs) - 1)]
    rolloff_95_freq = freqs[min(rolloff_95_index, len(freqs) - 1)]

    # Approximate occupied bandwidth containing 90% of power
    lower_power_index = np.searchsorted(cumulative_power, 0.05)
    upper_power_index = np.searchsorted(cumulative_power, 0.95)

    lower_freq = freqs[min(lower_power_index, len(freqs) - 1)]
    upper_freq = freqs[min(upper_power_index, len(freqs) - 1)]

    occupied_bandwidth_90 = upper_freq - lower_freq

    features = {
        # FFT magnitude statistics
        'mean_fft_magnitude': np.mean(fft_magnitude),
        'std_fft_magnitude': np.std(fft_magnitude),
        'min_fft_magnitude': np.min(fft_magnitude),
        'max_fft_magnitude': np.max(fft_magnitude),
        'median_fft_magnitude': np.median(fft_magnitude),
        'skew_fft_magnitude': safe_skew(fft_magnitude),
        'kurtosis_fft_magnitude': safe_kurtosis(fft_magnitude),

        # FFT power statistics
        'mean_fft_power': np.mean(fft_power),
        'std_fft_power': np.std(fft_power),
        'max_fft_power': peak_fft_power,
        'total_fft_power': total_fft_power,

        # Spectral summary features
        'spectral_centroid': spectral_centroid,
        'spectral_spread': spectral_spread,
        'spectral_entropy': spectral_entropy,
        'spectral_flatness': spectral_flatness,
        'dominant_freq': dominant_freq,
        'peak_fft_power_ratio': safe_divide(peak_fft_power, total_fft_power),
        'spectral_rolloff_85_freq': rolloff_85_freq,
        'spectral_rolloff_95_freq': rolloff_95_freq,
        'occupied_bandwidth_90': occupied_bandwidth_90
    }

    return features


def extract_constellation_features(
        signal: np.ndarray
) -> dict:
    '''
    About
    -----
    - Extracts constellation-style features from I/Q samples
    - These features summarize the geometry of the I/Q scatter plot
    - Helpful for distinguishing ASK, PSK, QAM, and similar modulation families

    Parameters
    ----------
    - signal (np.ndarray):
        - One signal frame with shape (1024, 2)

    Raises
    ------
    - None

    Returns
    -------
    - dict
        - Constellation geometry feature dictionary
    '''
    I = signal[:, 0]
    Q = signal[:, 1]

    iq = I + 1j * Q

    magnitude = np.abs(iq)
    phase = np.angle(iq)

    covariance_matrix = np.cov(I, Q)

    # Quadrant percentages
    quadrant_1 = np.mean((I >= 0) & (Q >= 0))
    quadrant_2 = np.mean((I < 0) & (Q >= 0))
    quadrant_3 = np.mean((I < 0) & (Q < 0))
    quadrant_4 = np.mean((I >= 0) & (Q < 0))

    # Distance from origin thresholds
    mean_magnitude = np.mean(magnitude)
    std_magnitude = np.std(magnitude)

    near_origin_ratio = np.mean(magnitude < mean_magnitude)
    far_origin_ratio = np.mean(magnitude > (mean_magnitude + std_magnitude))

    features = {
        # Geometry / covariance
        'cov_I_I': covariance_matrix[0, 0],
        'cov_I_Q': covariance_matrix[0, 1],
        'cov_Q_I': covariance_matrix[1, 0],
        'cov_Q_Q': covariance_matrix[1, 1],
        'constellation_area_proxy': np.std(I) * np.std(Q),

        # Quadrant distribution
        'quadrant_1_ratio': quadrant_1,
        'quadrant_2_ratio': quadrant_2,
        'quadrant_3_ratio': quadrant_3,
        'quadrant_4_ratio': quadrant_4,
        'quadrant_balance_std': np.std([
            quadrant_1,
            quadrant_2,
            quadrant_3,
            quadrant_4
        ]),

        # Origin / radius behavior
        'near_origin_ratio': near_origin_ratio,
        'far_origin_ratio': far_origin_ratio,
        'radius_mean': np.mean(magnitude),
        'radius_std': np.std(magnitude),
        'radius_unique_rounded_count': len(np.unique(np.round(magnitude, 2))),

        # Phase-bin behavior
        'phase_unique_rounded_count': len(np.unique(np.round(phase, 2)))
    }

    return features


def extract_all_signal_features(
        signal: np.ndarray
) -> dict:
    '''
    About
    -----
    - Runs all feature engineering functions on one I/Q signal frame
    - Combines the returned dictionaries into one feature dictionary

    Dependencies
    ------------
    - extract_basic_iq_features
    - extract_magnitude_power_features
    - extract_phase_frequency_features
    - extract_fft_spectral_features
    - extract_constellation_features

    Parameters
    ----------
    - signal (np.ndarray):
        - One signal frame with shape (1024, 2)

    Raises
    ------
    - None

    Returns
    -------
    - dict
        - Combined feature dictionary for one signal
    '''
    features = {}

    features.update(extract_basic_iq_features(signal))
    features.update(extract_magnitude_power_features(signal))
    features.update(extract_phase_frequency_features(signal))
    features.update(extract_fft_spectral_features(signal))
    features.update(extract_constellation_features(signal))

    return features


def create_feature_engineered_dataset(
        hdf5_data_filepath: str = HIGHEST_SNR_REDUCED_DF,
        mod_type_mapping: dict = MOD_TYPE_MAPPING,
        max_signals: int | None = None,
        print_progress: bool = True
) -> pd.DataFrame:
    '''
    About
    -----
    - Creates a feature-engineered pandas DataFrame from the reduced highest-SNR .hdf5 dataset
    - Each row represents one signal
    - Each column represents metadata or an engineered signal feature

    Dependencies
    ------------
    - extract_all_signal_features

    Parameters
    ----------
    - hdf5_data_filepath (str):
        - DEFAULT: HIGHEST_SNR_REDUCED_DF
        - Path to the reduced highest-SNR .hdf5 dataset

    - mod_type_mapping (dict):
        - DEFAULT: MOD_TYPE_MAPPING
        - Dictionary mapping modulation ID to modulation name

    - max_signals (int | None):
        - DEFAULT: None
        - If provided, only this many signals will be processed
        - Useful for quick testing

    - print_progress (bool):
        - DEFAULT: True
        - Whether to print progress while feature engineering

    Raises
    ------
    - FileNotFoundError
        - If the hdf5_data_filepath is not found

    Returns
    -------
    - pd.DataFrame
        - Feature-engineered dataset
    '''
    if not os.path.exists(hdf5_data_filepath):
        raise FileNotFoundError(f"Could not find dataset at: {hdf5_data_filepath}")

    results = []

    with h5py.File(hdf5_data_filepath, 'r') as f:
        X = f['X']
        Y = f['Y']
        Z = f['Z']

        num_signals = X.shape[0]

        if max_signals is not None:
            num_signals = min(num_signals, max_signals)

        if print_progress:
            print("=" * 100)
            print("FEATURE ENGINEERING STARTED")
            print("=" * 100)
            print("Dataset filepath:", hdf5_data_filepath)
            print("Signals to process:", num_signals)
            print("X shape:", X.shape)
            print("Y shape:", Y.shape)
            print("Z shape:", Z.shape)
            print()

        for idx in range(num_signals):
            signal = X[idx]
            label = Y[idx]
            snr = Z[idx][0]

            mod_id = int(np.argmax(label))
            mod_name = mod_type_mapping[mod_id]

            signal_features = extract_all_signal_features(signal)

            signal_features['signal_index'] = idx
            signal_features['modulation_id'] = mod_id
            signal_features['modulation_type'] = mod_name
            signal_features['snr'] = snr

            results.append(signal_features)

            if print_progress and idx > 0 and idx % 1000 == 0:
                print(f"Processed {idx} / {num_signals} signals...")

    df_features = pd.DataFrame(results)

    # Move metadata columns to the front
    metadata_cols = [
        'signal_index',
        'modulation_id',
        'modulation_type',
        'snr'
    ]

    feature_cols = [
        col for col in df_features.columns
        if col not in metadata_cols
    ]

    df_features = df_features[metadata_cols + feature_cols]

    if print_progress:
        print()
        print("=" * 100)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 100)
        print("Feature-engineered dataset shape:", df_features.shape)
        print("Number of feature columns:", len(feature_cols))
        print()
        print(df_features.head())

    return df_features


def save_feature_engineered_dataset(
        df_features: pd.DataFrame,
        output_filepath: str = FEATURE_ENGINEERED_DATASET_PARQUET,
        hdf5_key: str = 'features'
) -> None:
    '''
    About
    -----
    - Saves the feature-engineered dataset
    - Supports saving as:
        - .parquet
        - .hdf5 / .h5
        - .csv
    - Parquet is recommended for pandas feature datasets
    - HDF5 is also acceptable if you want to stay consistent with the rest of the project

    Parameters
    ----------
    - df_features (pd.DataFrame):
        - Feature-engineered dataset to save

    - output_filepath (str):
        - DEFAULT: FEATURE_ENGINEERED_DATASET_PARQUET
        - Output filepath for saved dataset

    - hdf5_key (str):
        - DEFAULT: 'features'
        - Key used when saving a pandas DataFrame to HDF5

    Raises
    ------
    - ValueError
        - If the output file extension is unsupported

    Returns
    -------
    - None
    '''
    output_dir = os.path.dirname(output_filepath)

    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    file_extension = os.path.splitext(output_filepath)[1].lower()

    if file_extension == '.parquet':
        df_features.to_parquet(output_filepath, index=False)

    elif file_extension in ['.hdf5', '.h5']:
        df_features.to_hdf(
            output_filepath,
            key=hdf5_key,
            mode='w',
            format='table'
        )

    elif file_extension == '.csv':
        df_features.to_csv(output_filepath, index=False)

    else:
        raise ValueError(
            "Unsupported output file extension. "
            "Use '.parquet', '.hdf5', '.h5', or '.csv'."
        )

    print("=" * 100)
    print("FEATURE-ENGINEERED DATASET SAVED")
    print("=" * 100)
    print("Output filepath:", output_filepath)
    print("Dataset shape:", df_features.shape)
    print()

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def main():
    '''
    About
    -----
    - Runs the full feature-engineering workflow
    - Loads the reduced highest-SNR dataset
    - Extracts useful signal features from I/Q data
    - Saves the feature-engineered dataset

    Dependencies
    ------------
    - create_feature_engineered_dataset
    - save_feature_engineered_dataset

    Parameters
    ----------
    - None

    Raises
    ------
    - FileNotFoundError
        - If the reduced highest-SNR dataset is not found

    Returns
    -------
    - None
    '''
    df_features = create_feature_engineered_dataset(
        hdf5_data_filepath=HIGHEST_SNR_REDUCED_DF,
        mod_type_mapping=MOD_TYPE_MAPPING,
        max_signals=None,
        print_progress=True
    )

    save_feature_engineered_dataset(
        df_features=df_features,
        output_filepath=FEATURE_ENGINEERED_DATASET_PARQUET
    )

# =================================================================================================
# END Main Function
# =================================================================================================