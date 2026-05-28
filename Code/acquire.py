'''
FILE OVERVIEW:
- Code and functions specifically for acquire.ipynb
- This is primarily for reproducibility and maintaing presentability of acquire.ipynb

=================================================

MISC COMMENTS:
- If you download the full 42GB file, please ensure that it lives in the root of this repository in a directory "Datasets"
- The created highest_snr_reduced_df.hdf5 should only be ~1.5GB

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - print_hdf5_dataset_info
    - print_mod_type_mapping
    - print_mod_type_counts
    - print_highest_snr_distribution
    - vis_signal
    - create_highest_snr_reduced_hdf5
- Main Function
'''
# ----- Imports -----------------------------------------------------------------------------------
import os                       # Reduced dataset creation      
import numpy as np              # Reading of signal array data
import pandas as pd             # Easier dataset manipulation
import h5py                     # Reading of hdf5 dataset
import matplotlib.pyplot as plt # Visualization of signal

# ----- Global Variables --------------------------------------------------------------------------
# Define the file pathing of the raw dataset and signal modulation type ID mapping
FILEPATH_RAWDATA = "../Datasets/GOLD_XYZ_OSC.0001_1024.hdf5"
FILEPATH_MODCLASS = "../Datasets/classes-fixed.txt"

# Define the signal modulation type mapping for quicker access
MOD_TYPE_MAPPING = {
    0: 'OOK', 
    1: '4ASK', 
    2: '8ASK', 
    3: 'BPSK', 
    4: 'QPSK', 
    5: '8PSK', 
    6: '16PSK', 
    7: '32PSK', 
    8: '16APSK', 
    9: '32APSK', 
    10: '64APSK', 
    11: '128APSK', 
    12: '16QAM', 
    13: '32QAM', 
    14: '64QAM', 
    15: '128QAM', 
    16: '256QAM', 
    17: 'AM-SSB-WC', 
    18: 'AM-SSB-SC', 
    19: 'AM-DSB-WC', 
    20: 'AM-DSB-SC', 
    21: 'FM', 
    22: 'GMSK', 
    23: 'OQPSK'
}

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def print_hdf5_dataset_info(hdf5_data_filepath:str=FILEPATH_RAWDATA) -> None:
    """
    About
    -----
    - Prints off general information about the passed in .hdf5 dataset

    Parameters
    ----------
    - hdf5_data_filepath (str) :
        - DEFAULT: '../Datasets/GOLD_XYZ_OSC.0001_1024.hdf5'
        - The filepathing to the .hdf5 dataset to print information on

    Raises
    ------
    - FileNotFoundError:
        - If 'hdf5_data_filepath' is not found

    Returns
    -------
    - None
    """
    # Try reading and printing off the .hdf5 dataset
    try:
        with h5py.File(hdf5_data_filepath, "r") as f:
            for key in f.keys():
                print(key, f[key].shape, f[key].dtype)

    except FileNotFoundError:
        print(f'\033[31m{hdf5_data_filepath} not found!\033[0m')
    

def print_mod_type_mapping(txt_mod_filepath:str=FILEPATH_MODCLASS) -> dict:
    """
    About
    -----
    - Prints off the signal modulation type ID mapping and returns a dictionary

    Parameters
    ----------
    - txt_mod_filepath (str) :
        - DEFAULT: '../Datasets/classes-fixed.txt'
        - The filepathing to the signal modulation type ID mapping .txt file

    Raises
    ------
    - FileNotFoundError:
        - If 'txt_mod_filepath' is not found

    Returns
    -------
    - mod_type_mapping (dict):
        - The signal modulation type ID mapping
    """
    # Initialize the dictionary
    mod_type_mapping = {}

    # Iterate through the file and extract desired information mapping
    with open(txt_mod_filepath, 'r') as f:
        for line in f:
            if line.startswith('|'):
                line_val_list = line.split('|')
                if line_val_list[1].strip().isnumeric():
                    try:
                        index = int(line_val_list[1].strip()) 
                        fixed_mod_class = str(line_val_list[3].strip().strip("'"))
                        mod_type_mapping[index] = fixed_mod_class
                    except TypeError:
                        continue

    # Print off an easy to read list of modulation mapping
    print('\033[35mID\tMod Type\033[0m')
    print('\033[35m' + '====='*3 + '\033[0m')
    for id in mod_type_mapping:
        print(f'\033[35m{id}:\033[0m\t{mod_type_mapping[id]}')

    # Return the modulation mapping
    return mod_type_mapping


def print_mod_type_counts(
        hdf5_data_filepath:str=FILEPATH_RAWDATA,
        mod_type_mapping:dict=MOD_TYPE_MAPPING
) -> None:
    """
    About
    -----
    - Prints off the counts of signals per modulation type in the .hdf5 dataset

    Parameters
    ----------
    - hdf5_data_filepath (str) :
        - DEFAULT: '../Datasets/GOLD_XYZ_OSC.0001_1024.hdf5'
        - The filepathing to the .hdf5 dataset to print information on

    - mod_type_mapping (dict):
        - DEFAULT: MOD_TYPE_MAPPING
        - The default dictionary comes from the 'print_mod_type_mapping' function
        - Dictionary mapping class index to modulation name.
        - Example: {0: "OOK", 1: "4ASK", ...}

    Throws
    ------
    - FileNotFoundError:
        - If 'hdf5_data_filepath' is not found

    Returns
    -------
    - None
    """
    # Instantiate variable to house results in
    results = []

    # Open the file
    with h5py.File(hdf5_data_filepath, "r") as f:
        Y = f["Y"][:]

    # Convert one-hot labels to class IDs
    mod_labels = np.argmax(Y, axis=1)

    # Count signals per unique mod type
    unique_mod_type, counts = np.unique(mod_labels, return_counts=True)

    total_signals = len(mod_labels)

    # Pass information into results
    for mod_id, count in zip(unique_mod_type, counts):
        mod_name = mod_type_mapping[mod_id]
        percentage = (count / total_signals) * 100

        results.append({
            "modulation_id": mod_id,
            "modulation_type": mod_name,
            "num_signals": count,
            "percentage": percentage
        })

    # Turn into a Pandas dataframe for a cleaner print
    df_counts = pd.DataFrame(results)

    # Print information
    print("Total signals:", total_signals)
    print(df_counts)


def print_highest_snr_distribution(
        hdf5_data_filepath: str = FILEPATH_RAWDATA,
        mod_type_mapping: dict = MOD_TYPE_MAPPING
) -> None:
    """
    About
    -----
    - Filters the .hdf5 dataset to only the signals with the highest SNR
    - Prints the count of signals per modulation type within that highest-SNR subset
    - Also prints what percentage each modulation type takes up within the highest-SNR subset
    - Also prints what percentage each modulation type takes up of the whole dataset

    Parameters
    ----------
    - hdf5_data_filepath (str) :
        - DEFAULT: '../Datasets/GOLD_XYZ_OSC.0001_1024.hdf5'
        - The filepath to the .hdf5 dataset to print information on

    - mod_type_mapping (dict):
        - DEFAULT: MOD_TYPE_MAPPING
        - Dictionary mapping class index to modulation name.
        - Example: {0: "OOK", 1: "4ASK", ...}

    Throws
    ------
    - FileNotFoundError:
        - If 'hdf5_data_filepath' is not found

    Returns
    -------
    - None
    """

    # Instantiate variable to house results in
    results = []

    # Open the file
    with h5py.File(hdf5_data_filepath, "r") as f:
        Y = f["Y"][:]                  # shape: (num_signals, 24)
        Z = f["Z"][:].flatten()        # shape: (num_signals,)

    # Convert one-hot labels to class IDs
    mod_labels = np.argmax(Y, axis=1)

    # Get total number of signals in full dataset
    total_signals = len(mod_labels)

    # Find highest SNR value
    highest_snr = np.max(Z)

    # Filter to only highest-SNR signals
    highest_snr_mask = Z == highest_snr

    highest_snr_labels = mod_labels[highest_snr_mask]

    # Count number of highest-SNR signals
    total_highest_snr_signals = len(highest_snr_labels)

    # Count signals per modulation type within highest-SNR subset
    unique_mod_type, counts = np.unique(highest_snr_labels, return_counts=True)

    # Pass information into results
    for mod_id, count in zip(unique_mod_type, counts):
        mod_name = mod_type_mapping[mod_id]

        percentage_of_highest_snr_subset = (
            count / total_highest_snr_signals
        ) * 100

        percentage_of_whole_dataset = (
            count / total_signals
        ) * 100

        results.append({
            "modulation_id": mod_id,
            "modulation_type": mod_name,
            "num_highest_snr_signals": count,
            "percentage_of_highest_snr_subset": percentage_of_highest_snr_subset,
            "percentage_of_whole_dataset": percentage_of_whole_dataset
        })

    # Turn into a Pandas dataframe for cleaner printing
    df_highest_snr_counts = pd.DataFrame(results)

    # Round percentages
    df_highest_snr_counts["percentage_of_highest_snr_subset"] = (
        df_highest_snr_counts["percentage_of_highest_snr_subset"].round(2)
    )

    df_highest_snr_counts["percentage_of_whole_dataset"] = (
        df_highest_snr_counts["percentage_of_whole_dataset"].round(4)
    )

    # Print information
    print("Total signals in full dataset:", total_signals)
    print("Highest SNR:", highest_snr)
    print("Total signals with highest SNR:", total_highest_snr_signals)
    print(df_highest_snr_counts)


def vis_signal(
    hdf5_data_filepath:str=FILEPATH_RAWDATA,
    mod_type_mapping:dict=MOD_TYPE_MAPPING,
    idx:int=None,
    mod_type:str=None,
    save_path:int=None,
    random_state:int=35
) -> int:
    """
    About
    -----
    - Visualizes one DeepSig I/Q signal from an HDF5 dataset
    - Returns the index of the signal visualized

    Parameters
    ----------
    - hdf5_data_filepath (str):
        - DEFAULT: '../Datasets/GOLD_XYZ_OSC.0001_1024.hdf5'
        - Path to the DeepSig .hdf5 data file

    - mod_type_mapping (dict):
        - DEFAULT: MOD_TYPE_MAPPING
        - The default dictionary comes from the 'print_mod_type_mapping' function
        - Dictionary mapping class index to modulation name.
        - Example: {0: "OOK", 1: "4ASK", ...}

    - idx (int, optional):
        - DEFAULT: None
        - Exact signal index to plot.
        - If filtering by signal modulation, this will be overridden with a random index from the filter

    - mod_type (str, optional):
        - DEFAULT: None
        - Modulation type to randomly select from.
        - Example: "QPSK", "BPSK", "16QAM"

    - save_path (str, optional):
        - DEFAULT: None
        - Filepath to save the plot, for example "plots/qpsk_signal.png".

    - random_state (int):
        - DEFAULT: 35
        - Random seed used when selecting by modulation type

    Raises
    ------
    - ValueError:
        - If passed mod_type is not found
        - If no matching signal type is found

    Returns
    -------
    idx (int):
        The signal index that was plotted.
    """
    # ========== Initial Setup ====================================================================
    # Set a randomizer to randomly choose a signal if filtered by mod_type
    rng = np.random.default_rng(random_state)

    # Open the data file
    with h5py.File(hdf5_data_filepath, "r") as f:
        X = f["X"]
        Y = f["Y"]
        Z = f["Z"]

        # ========== Filter By Modulation Type ====================================================
        if idx is None and mod_type is not None:
            reverse_mod_dict = {v: k for k, v in mod_type_mapping.items()}

            if mod_type not in reverse_mod_dict:
                raise ValueError(
                    "\033[31m"
                    f"Unknown mod_type '{mod_type}' "
                    f"Available types: {list(reverse_mod_dict.keys())}"
                    "\033[0m"
                )

            target_mod = reverse_mod_dict[mod_type]

            # Identify matching signal modulation type indicies
            labels = np.argmax(Y[:], axis=1)
            matching_indices = np.where(labels == target_mod)[0]

            if len(matching_indices) == 0:
                raise ValueError(
                    "\033[31m"
                    f"No examples found for modulation type {mod_type}"
                    "\033[0m"
                )

            # Assign a random index from mod filtering
            idx = rng.choice(matching_indices)

        # ========== Final Setup ==================================================================
        # If no index and filter was given
        elif idx is None:
            idx = 0
        
        x = X[idx]       # shape: (1024, 2)
        y = Y[idx]       # shape: (24,)
        z = Z[idx]       # shape: (1,)

    # Define I/Q signal data split
    I = x[:, 0]
    Q = x[:, 1]

    # Variables for info prints
    mod_class = np.argmax(y)
    mod_name = mod_type_mapping[mod_class]
    snr = z[0]

    # ========== Standard Info Print ==============================================================
    print("Signal index:", idx)
    print("Signal shape:", x.shape)
    print("Modulation Class:", mod_name)
    print("SNR:", snr)

    # ========== Visualization ====================================================================
    plt.figure(figsize=(12, 4))
    plt.plot(I, label="I")
    plt.plot(Q, label="Q")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.title(f"I/Q Time-Domain Signal | Index={idx}, Class={mod_name}, SNR={snr}")
    plt.legend()
    plt.grid(True)
    plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return idx


def create_highest_snr_reduced_hdf5(
        hdf5_data_filepath: str = FILEPATH_RAWDATA,
        output_hdf5_filepath: str = "../Datasets/highest_snr_reduced_df.hdf5",
        mod_type_mapping: dict = MOD_TYPE_MAPPING,
        chunk_size: int = 5000
) -> None:
    """
    About
    -----
    - Creates a reduced .hdf5 file containing only the highest-SNR (30) signals
      from the original DeepSig .hdf5 dataset
    - Keeps all modulation types, but only includes examples where SNR is the
      maximum SNR value found in the dataset
    - Saves the reduced dataset with X, Y, Z, and original_indices datasets

    Parameters
    ----------
    - hdf5_data_filepath (str):
        - DEFAULT: '../Datasets/GOLD_XYZ_OSC.0001_1024.hdf5'
        - The filepath to the original .hdf5 dataset

    - output_hdf5_filepath (str):
        - DEFAULT: '../Datasets/highest_snr_reduced_df.hdf5'
        - The filepath where the reduced .hdf5 dataset will be saved

    - mod_type_mapping (dict):
        - DEFAULT: MOD_TYPE_MAPPING
        - Dictionary mapping class index to modulation name.
        - Example: {0: "OOK", 1: "4ASK", ...}

    - chunk_size (int):
        - DEFAULT: 5000
        - Number of signals copied at a time to avoid loading too much into memory

    Throws
    ------
    - FileNotFoundError:
        - If 'hdf5_data_filepath' is not found

    Returns
    -------
    - None
    """
    # ========== Ensure Output Path Exists ========================================================
    output_dir = os.path.dirname(output_hdf5_filepath)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    #========== Filter Highest SNR Signals ========================================================
    with h5py.File(hdf5_data_filepath, "r") as f_in:

        # Load mod types and SNRs
        Y = f_in["Y"][:]
        Z = f_in["Z"][:].flatten()

        # Convert one-hot labels to class IDs
        mod_labels = np.argmax(Y, axis=1)

        # Find highest SNR value
        highest_snr = np.max(Z)

        # Create mask for highest SNR signals
        highest_snr_mask = np.isclose(Z, highest_snr)

        # Get indices of highest SNR signals
        highest_snr_indices = np.where(highest_snr_mask)[0]

        # Count total reduced signals
        total_reduced_signals = len(highest_snr_indices)

        # Get original dataset shapes
        X_shape = f_in["X"].shape
        Y_shape = f_in["Y"].shape
        Z_shape = f_in["Z"].shape

        # Create reduced shapes
        reduced_X_shape = (total_reduced_signals, X_shape[1], X_shape[2])
        reduced_Y_shape = (total_reduced_signals, Y_shape[1])
        reduced_Z_shape = (total_reduced_signals, Z_shape[1])

        # ========== Create Reduced Dataset =======================================================
        with h5py.File(output_hdf5_filepath, "w") as f_out:

            # Create reduced datasets
            X_out = f_out.create_dataset(
                "X",
                shape=reduced_X_shape,
                dtype=f_in["X"].dtype,
                compression="gzip",
                compression_opts=4
            )

            Y_out = f_out.create_dataset(
                "Y",
                shape=reduced_Y_shape,
                dtype=f_in["Y"].dtype,
                compression="gzip",
                compression_opts=4
            )

            Z_out = f_out.create_dataset(
                "Z",
                shape=reduced_Z_shape,
                dtype=f_in["Z"].dtype,
                compression="gzip",
                compression_opts=4
            )

            indices_out = f_out.create_dataset(
                "original_indices",
                data=highest_snr_indices,
                compression="gzip",
                compression_opts=4
            )

            # Store useful metadata as attributes
            f_out.attrs["source_file"] = hdf5_data_filepath
            f_out.attrs["highest_snr"] = highest_snr
            f_out.attrs["num_signals"] = total_reduced_signals
            f_out.attrs["description"] = "Reduced DeepSig dataset containing only highest-SNR signals"

            # Copy data in chunks
            for start in range(0, total_reduced_signals, chunk_size):
                end = min(start + chunk_size, total_reduced_signals)

                batch_indices = highest_snr_indices[start:end]

                X_out[start:end] = f_in["X"][batch_indices]
                Y_out[start:end] = f_in["Y"][batch_indices]
                Z_out[start:end] = f_in["Z"][batch_indices]

    # ========== Print Summary ====================================================================
    reduced_mod_labels = mod_labels[highest_snr_indices]

    unique_mod_type, counts = np.unique(reduced_mod_labels, return_counts=True)

    results = []

    for mod_id, count in zip(unique_mod_type, counts):
        mod_name = mod_type_mapping[mod_id]
        percentage = (count / total_reduced_signals) * 100

        results.append({
            "modulation_id": mod_id,
            "modulation_type": mod_name,
            "num_signals": count,
            "percentage": round(percentage, 2)
        })

    df_counts = pd.DataFrame(results)

    print("\033[32mhighest_snr_reduced_df.hdf5 file created successfully!\033[0m")
    print("Output filepath:", output_hdf5_filepath)
    print("Highest SNR:", highest_snr)
    print("Total highest-SNR signals:", total_reduced_signals)
    print()
    print(df_counts)

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def main():
    """
    About
    -----
    - Simply runs all the functions in this file

    Dependencies
    ------------
    - print_hdf5_dataset_info
    - print_mod_type_mapping
    - print_mod_type_counts
    - print_highest_snr_distribution
    - vis_signal
    - create_highest_snr_distribution

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
    print_hdf5_dataset_info()
    print_mod_type_mapping()
    print_mod_type_counts()
    print_highest_snr_distribution()
    vis_signal()
    create_highest_snr_reduced_hdf5()

# =================================================================================================
# END Main Function
# =================================================================================================