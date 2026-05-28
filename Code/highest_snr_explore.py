'''
FILE OVERVIEW:
- Code and functions specifically for highest_snr_explore.ipynb
- This is primarily for reproducibility and maintaining presentability of highest_snr_explore.ipynb
- This file heavily pulls global variables and functions created in acquire.py

=================================================

MISC COMMENTS:
- THIS ASSUMES YOU HAVE THE REDUCED DATASET '../Datasets/highest_snr_reduced_df.hdf5'
- Because of the nature of the dataset, there will NOT be any data preparation since
  the data appears to already be uniformly distributed and represents signals accurately

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - vis_all_mod_types
'''
# ----- Imports -----------------------------------------------------------------------------------
from acquire import vis_signal          # Easy visualization of signals

# ----- Global Variables --------------------------------------------------------------------------
from acquire import MOD_TYPE_MAPPING    # Modulation ID and Name mapping
HIGHEST_SNR_REDUCED_DF = '../Datasets/highest_snr_reduced_df.hdf5'

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

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
        
        # Nice visual seperator between visualizations
        print('\033[35m\n=============================================\033[0m')

# =================================================================================================
# END Helper Functions
# =================================================================================================