'''
FILE OVERVIEW:
- This is primarily for visTree.ipynb
- This is just for visualizing the tree

=================================================

MISC COMMENTS:
- You must have a created .joblib model for this to run properly

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - plot_simplified_random_forest_tree
'''
# ----- Imports -----------------------------------------------------------------------------------
import joblib
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree

# ----- Global Variables --------------------------------------------------------------------------

MODEL_PATH = "../Models/highest_snr_random_forest_baseline.joblib"

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def plot_simplified_random_forest_tree(
        model_filepath: str = MODEL_PATH,
        output_filepath: str = "../Reports/simplified_random_forest_tree.png",
        tree_index: int = 0,
        max_depth: int = 2,
        figsize: tuple = (24, 12)
) -> None:
    """
    About
    -----
    - Loads a saved Random Forest model
    - Plots one simplified decision tree from the forest
    - Saves the tree as a presentation-friendly PNG image

    Parameters
    ----------
    - model_filepath (str):
        - DEFAULT: MODEL_PATH
        - Filepath to the saved Random Forest model

    - output_filepath (str):
        - DEFAULT: '../Reports/simplified_random_forest_tree.png'
        - Filepath where the simplified tree image will be saved

    - tree_index (int):
        - DEFAULT: 0
        - Which tree in the forest to visualize

    - max_depth (int):
        - DEFAULT: 2
        - Maximum depth to display for readability

    - figsize (tuple):
        - DEFAULT: (24, 12)
        - Figure size for the plot

    Raises
    ------
    - FileNotFoundError:
        - If the model filepath does not exist

    Returns
    -------
    - None
    """
    # ========== Load Saved Model =================================================================
    rf_model = joblib.load(model_filepath)

    # ========== Pull One Tree ====================================================================
    single_tree = rf_model.estimators_[tree_index]

    # ========== Feature Names ====================================================================
    if hasattr(rf_model, "feature_names_in_"):
        feature_names = list(rf_model.feature_names_in_)
    else:
        feature_names = [f"feature_{i}" for i in range(single_tree.n_features_in_)]

    class_names = [str(class_name) for class_name in rf_model.classes_]

    # ========== Plot Simplified Tree =============================================================
    plt.figure(figsize=figsize)

    plot_tree(
        decision_tree=single_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=10,
        proportion=True
    )

    plt.title(
        f"Simplified Random Forest Decision Tree (Tree {tree_index}, Max Depth = {max_depth})",
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Simplified tree saved to: {output_filepath}")

# =================================================================================================
# END Helper Functions
# =================================================================================================