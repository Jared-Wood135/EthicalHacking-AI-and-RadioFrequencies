'''
FILE OVERVIEW:
- Code and functions specifically for acquire.ipynb
- This is primarily for reproducibility and maintaing presentability of acquire.ipynb

=================================================

MISC COMMENTS:
- If you download the full 42GB file, please ensure that it lives in the root of this repository in a directory "Datasets"

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - get_hdf5_dataset_as_pd
    - print_mod_type_mapping
    - vis_signal
    - create_reduced_dataset
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

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def get_hdf5_dataset_as_pd():
    pass


def print_mod_type_mapping():
    pass


def vis_signal():
    pass


def create_reduced_dataset():
    pass

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================



# =================================================================================================
# END Main Function
# =================================================================================================