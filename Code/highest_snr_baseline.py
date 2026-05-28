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
import numpy as np              # Iteration over arrays (I/Q signal array)
import pandas as pd             # Easier dataset manipulation
import h5py                     # Reading .hdf5 datasets
import matplotlib.pyplot as plt # Visualizations

# Random Forest Classifier (ML)
from sklearn.ensemble import RandomForestClassifier

# Hyper-parameter tuning
from sklearn.model_selection import GridSearchCV

# Standard model performance metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# ----- Global Variables --------------------------------------------------------------------------
from acquire import MOD_TYPE_MAPPING    # Modulation ID and name mapping
HIGHEST_SNR_REDUCED_DF = '../Datasets/highest_snr_reduced_df.hdf5'

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def get_rfc():
    pass


def train_and_test_model():
    pass


def save_baseline():
    pass

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def main():
    """
    About
    -----
    - Simply runs everything from above

    Dependencies
    ------------
    - 

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
    pass

# =================================================================================================
# END Main Function
# =================================================================================================