"""
FILE OVERVIEW:
- Command-line interactive demo for the Simulation folder
- Allows the user to enter text and toggle modulation/encryption correctness

=================================================

MISC COMMENTS:
- Run from project root:
    python Simulation/run_text_simulation.py

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - ask_yes_no
    - ask_choice
- Main Function
"""

# ----- Imports -----------------------------------------------------------------------------------
from simulation_core import (
    AVAILABLE_MODULATION_TYPES,
    run_text_simulation,
    print_simulation_report
)

# ----- Global Variables --------------------------------------------------------------------------
DEFAULT_TRUE_MODULATION = "QPSK"

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """
    About
    -----
    - Asks a yes/no question and returns a boolean

    Parameters
    ----------
    - prompt (str):
        - Prompt to display

    - default (bool):
        - DEFAULT: True
        - Default answer if user presses Enter

    Returns
    -------
    - bool
        - User choice
    """
    default_text = "Y/n" if default else "y/N"

    answer = input(f"{prompt} [{default_text}]: ").strip().lower()

    if answer == "":
        return default

    return answer in ["y", "yes", "true", "1"]


def ask_choice(prompt: str, choices: list, default: str) -> str:
    """
    About
    -----
    - Asks user to choose from a list of options

    Parameters
    ----------
    - prompt (str):
        - Prompt to display

    - choices (list):
        - Allowed choices

    - default (str):
        - Default choice

    Returns
    -------
    - str
        - Selected choice
    """
    choices_display = "/".join(choices)

    answer = input(f"{prompt} ({choices_display}) [default={default}]: ").strip().upper()

    if answer == "":
        return default

    if answer not in choices:
        print(f"Invalid choice '{answer}'. Using default: {default}")
        return default

    return answer

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def main():
    """
    About
    -----
    - Runs the command-line text simulation interactively

    Parameters
    ----------
    - None

    Returns
    -------
    - None
    """
    print("=" * 100)
    print("RF TEXT SIMULATION")
    print("=" * 100)
    print("This demo shows text -> bits -> encryption -> modulation -> I/Q -> recovery.")
    print()

    input_text = input("Enter text to transmit: ").strip()

    if input_text == "":
        input_text = "HELLO RF WORLD"

    true_modulation = ask_choice(
        prompt="Choose TRUE transmitter modulation",
        choices=AVAILABLE_MODULATION_TYPES,
        default=DEFAULT_TRUE_MODULATION
    )

    correct_modulation_enabled = ask_yes_no(
        prompt="Should the receiver use the CORRECT modulation?",
        default=True
    )

    receiver_modulation = None

    if not correct_modulation_enabled:
        receiver_modulation = ask_choice(
            prompt="Choose WRONG receiver modulation",
            choices=AVAILABLE_MODULATION_TYPES,
            default="BPSK" if true_modulation != "BPSK" else "QPSK"
        )

    encryption_enabled = ask_yes_no(
        prompt="Enable XOR encryption?",
        default=True
    )

    encryption_key = "secret"
    correct_encryption_enabled = True
    receiver_encryption_key = None

    if encryption_enabled:
        encryption_key = input("Enter true encryption key [secret]: ").strip()

        if encryption_key == "":
            encryption_key = "secret"

        correct_encryption_enabled = ask_yes_no(
            prompt="Should the receiver use the CORRECT encryption key?",
            default=True
        )

        if not correct_encryption_enabled:
            receiver_encryption_key = input("Enter WRONG receiver key [wrong_key]: ").strip()

            if receiver_encryption_key == "":
                receiver_encryption_key = "wrong_key"

    results = run_text_simulation(
        input_text=input_text,
        true_modulation_type=true_modulation,
        receiver_modulation_type=receiver_modulation,
        correct_modulation_enabled=correct_modulation_enabled,
        encryption_enabled=encryption_enabled,
        encryption_key=encryption_key,
        receiver_encryption_key=receiver_encryption_key,
        correct_encryption_enabled=correct_encryption_enabled
    )

    print_simulation_report(
        simulation_results=results,
        show_full_keys=False
    )


if __name__ == "__main__":
    main()

# =================================================================================================
# END Main Function
# =================================================================================================
