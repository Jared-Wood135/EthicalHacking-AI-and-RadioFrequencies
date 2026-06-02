"""
FILE OVERVIEW:
- Core logic for the text-based modulation/encryption simulation
- Converts text to bits, optionally encrypts bits, modulates into I/Q samples,
  demodulates with correct or incorrect modulation assumptions, decrypts with
  correct or incorrect keys, and attempts to recover text
- Can convert generated I/Q samples into DeepSig-style frames shaped as:
    (num_frames, 1024, 2)

=================================================

MISC COMMENTS:
- This is an educational simulation, not a production radio receiver
- The Random Forest models in the main project classify modulation type;
  they do not decrypt messages
- Encryption and modulation are separate layers:
    text -> bits -> encryption -> modulation -> I/Q
    I/Q -> demodulation -> decryption -> recovered text

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - text_to_bits
    - bits_to_text
    - format_bits
    - repeat_key_bits
    - xor_bits
    - get_constellation
    - modulate_bits
    - demodulate_iq
    - iq_to_deepsig_frames
    - calculate_bit_error_rate
    - run_text_simulation
    - print_simulation_report
- Main Function
"""

# ----- Imports -----------------------------------------------------------------------------------
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any

import numpy as np


# ----- Global Variables --------------------------------------------------------------------------
AVAILABLE_MODULATION_TYPES = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM"
]

BITS_PER_SYMBOL = {
    "BPSK": 1,
    "QPSK": 2,
    "8PSK": 3,
    "16QAM": 4
}


# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def text_to_bits(
        text: str,
        encoding: str = "utf-8"
) -> List[int]:
    """
    About
    -----
    - Converts a string into a list of bits using the provided encoding

    Parameters
    ----------
    - text (str):
        - Input message

    - encoding (str):
        - DEFAULT: 'utf-8'
        - Encoding used to convert text into bytes

    Raises
    ------
    - None

    Returns
    -------
    - list[int]
        - Bit list representation of the input text
    """
    byte_data = text.encode(encoding)

    bits = []

    for byte in byte_data:
        for bit_position in range(7, -1, -1):
            bits.append((byte >> bit_position) & 1)

    return bits


def bits_to_text(
        bits: List[int],
        encoding: str = "utf-8"
) -> str:
    """
    About
    -----
    - Converts a list of bits back into text
    - Invalid byte sequences are replaced so wrong modulation/encryption still displays something

    Parameters
    ----------
    - bits (list[int]):
        - Bit list to decode

    - encoding (str):
        - DEFAULT: 'utf-8'
        - Encoding used to decode bytes back into text

    Raises
    ------
    - None

    Returns
    -------
    - str
        - Decoded text
    """
    usable_length = (len(bits) // 8) * 8
    usable_bits = bits[:usable_length]

    byte_values = []

    for start in range(0, usable_length, 8):
        byte = 0

        for bit in usable_bits[start:start + 8]:
            byte = (byte << 1) | int(bit)

        byte_values.append(byte)

    return bytes(byte_values).decode(encoding, errors="replace")


def format_bits(
        bits: List[int],
        max_bits: int = 128
) -> str:
    """
    About
    -----
    - Formats a bit list for readable display
    - Truncates long bitstreams while still showing total length

    Parameters
    ----------
    - bits (list[int]):
        - Bit list to format

    - max_bits (int):
        - DEFAULT: 128
        - Maximum number of bits to display before truncating

    Raises
    ------
    - None

    Returns
    -------
    - str
        - Human-readable bit string
    """
    shown_bits = bits[:max_bits]
    bit_string = "".join(str(int(bit)) for bit in shown_bits)

    grouped = " ".join(
        bit_string[start:start + 8]
        for start in range(0, len(bit_string), 8)
    )

    if len(bits) > max_bits:
        grouped += f" ... ({len(bits)} total bits)"

    else:
        grouped += f" ({len(bits)} total bits)"

    return grouped


def repeat_key_bits(
        key_text: str,
        target_length: int
) -> List[int]:
    """
    About
    -----
    - Converts a key string into bits and repeats/truncates it to match target length

    Parameters
    ----------
    - key_text (str):
        - Encryption/decryption key text

    - target_length (int):
        - Desired key bit length

    Raises
    ------
    - ValueError:
        - If key_text is empty

    Returns
    -------
    - list[int]
        - Repeated key bits
    """
    if key_text == "":
        raise ValueError("key_text cannot be empty")

    key_bits = text_to_bits(key_text)

    repeat_count = math.ceil(target_length / len(key_bits))
    repeated_key_bits = (key_bits * repeat_count)[:target_length]

    return repeated_key_bits


def xor_bits(
        bits: List[int],
        key_text: str
) -> List[int]:
    """
    About
    -----
    - Applies simple XOR encryption/decryption to a bitstream
    - The same function is used for encryption and decryption

    Parameters
    ----------
    - bits (list[int]):
        - Bitstream to encrypt/decrypt

    - key_text (str):
        - Key text used to generate repeated XOR key bits

    Raises
    ------
    - ValueError:
        - If key_text is empty

    Returns
    -------
    - list[int]
        - XORed bitstream
    """
    key_bits = repeat_key_bits(key_text, len(bits))

    return [
        int(bit) ^ int(key_bit)
        for bit, key_bit in zip(bits, key_bits)
    ]


def _int_to_bits(
        value: int,
        width: int
) -> Tuple[int, ...]:
    """
    About
    -----
    - Converts an integer into a fixed-width tuple of bits

    Parameters
    ----------
    - value (int):
        - Integer value

    - width (int):
        - Number of output bits

    Raises
    ------
    - None

    Returns
    -------
    - tuple[int, ...]
        - Fixed-width bit tuple
    """
    return tuple((value >> bit_position) & 1 for bit_position in range(width - 1, -1, -1))


def _bits_to_int(
        bits: Tuple[int, ...]
) -> int:
    """
    About
    -----
    - Converts a tuple of bits to an integer

    Parameters
    ----------
    - bits (tuple[int, ...]):
        - Bits to convert

    Raises
    ------
    - None

    Returns
    -------
    - int
        - Integer representation
    """
    value = 0

    for bit in bits:
        value = (value << 1) | int(bit)

    return value


def get_constellation(
        modulation_type: str
) -> Dict[Tuple[int, ...], complex]:
    """
    About
    -----
    - Creates a simple constellation mapping for supported modulation types

    Parameters
    ----------
    - modulation_type (str):
        - One of AVAILABLE_MODULATION_TYPES

    Raises
    ------
    - ValueError:
        - If modulation_type is unsupported

    Returns
    -------
    - dict
        - Mapping from bit tuples to complex constellation points
    """
    modulation_type = modulation_type.upper()

    if modulation_type == "BPSK":
        return {
            (0,): -1 + 0j,
            (1,): 1 + 0j
        }

    if modulation_type == "QPSK":
        scale = 1 / np.sqrt(2)

        return {
            (0, 0): (1 + 1j) * scale,
            (0, 1): (-1 + 1j) * scale,
            (1, 1): (-1 - 1j) * scale,
            (1, 0): (1 - 1j) * scale
        }

    if modulation_type == "8PSK":
        constellation = {}

        for value in range(8):
            bits = _int_to_bits(value, 3)
            angle = 2 * np.pi * value / 8
            constellation[bits] = np.exp(1j * angle)

        return constellation

    if modulation_type == "16QAM":
        # Simple 16QAM map normalized to approximately unit average power
        levels = [-3, -1, 1, 3]
        scale = 1 / np.sqrt(10)
        constellation = {}

        for value in range(16):
            bits = _int_to_bits(value, 4)
            i_idx = _bits_to_int(bits[:2])
            q_idx = _bits_to_int(bits[2:])
            constellation[bits] = complex(levels[i_idx], levels[q_idx]) * scale

        return constellation

    raise ValueError(
        f"Unsupported modulation_type '{modulation_type}'. "
        f"Choose from: {AVAILABLE_MODULATION_TYPES}"
    )


def modulate_bits(
        bits: List[int],
        modulation_type: str = "QPSK",
        samples_per_symbol: int = 8
) -> Tuple[np.ndarray, int]:
    """
    About
    -----
    - Modulates a bitstream into complex I/Q samples
    - Pads bits as needed so the final bit count is divisible by bits per symbol

    Parameters
    ----------
    - bits (list[int]):
        - Bitstream to modulate

    - modulation_type (str):
        - DEFAULT: 'QPSK'
        - Modulation type to use

    - samples_per_symbol (int):
        - DEFAULT: 8
        - Number of repeated samples per symbol

    Raises
    ------
    - ValueError:
        - If samples_per_symbol is less than 1

    Returns
    -------
    - tuple
        - iq_samples (np.ndarray)
        - num_padding_bits (int)
    """
    if samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be >= 1")

    modulation_type = modulation_type.upper()

    constellation = get_constellation(modulation_type)
    bits_per_symbol = BITS_PER_SYMBOL[modulation_type]

    num_padding_bits = (-len(bits)) % bits_per_symbol
    padded_bits = list(bits) + ([0] * num_padding_bits)

    symbols = []

    for start in range(0, len(padded_bits), bits_per_symbol):
        bit_group = tuple(padded_bits[start:start + bits_per_symbol])
        symbols.append(constellation[bit_group])

    symbol_array = np.array(symbols, dtype=np.complex64)

    iq_samples = np.repeat(symbol_array, samples_per_symbol)

    return iq_samples, num_padding_bits


def demodulate_iq(
        iq_samples: np.ndarray,
        modulation_type: str = "QPSK",
        samples_per_symbol: int = 8,
        num_padding_bits: int = 0,
        expected_num_bits: int | None = None
) -> List[int]:
    """
    About
    -----
    - Demodulates complex I/Q samples into bits by nearest constellation point
    - If the wrong modulation type is used, the output bitstream will likely be corrupted

    Parameters
    ----------
    - iq_samples (np.ndarray):
        - Complex I/Q samples

    - modulation_type (str):
        - DEFAULT: 'QPSK'
        - Receiver-assumed modulation type

    - samples_per_symbol (int):
        - DEFAULT: 8
        - Number of samples per symbol

    - num_padding_bits (int):
        - DEFAULT: 0
        - Number of padding bits to remove

    - expected_num_bits (int | None):
        - DEFAULT: None
        - Optional output bit length to force

    Raises
    ------
    - ValueError:
        - If samples_per_symbol is less than 1

    Returns
    -------
    - list[int]
        - Recovered bitstream
    """
    if samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be >= 1")

    modulation_type = modulation_type.upper()

    constellation = get_constellation(modulation_type)

    bit_groups = list(constellation.keys())
    points = np.array(list(constellation.values()), dtype=np.complex64)

    usable_sample_count = (len(iq_samples) // samples_per_symbol) * samples_per_symbol
    usable_samples = iq_samples[:usable_sample_count]

    symbol_estimates = usable_samples.reshape(-1, samples_per_symbol).mean(axis=1)

    recovered_bits = []

    for symbol in symbol_estimates:
        nearest_idx = np.argmin(np.abs(symbol - points))
        recovered_bits.extend(bit_groups[nearest_idx])

    if num_padding_bits > 0:
        recovered_bits = recovered_bits[:-num_padding_bits]

    if expected_num_bits is not None:
        if len(recovered_bits) < expected_num_bits:
            recovered_bits = recovered_bits + ([0] * (expected_num_bits - len(recovered_bits)))

        recovered_bits = recovered_bits[:expected_num_bits]

    return [int(bit) for bit in recovered_bits]


def iq_to_deepsig_frames(
        iq_samples: np.ndarray,
        frame_size: int = 1024
) -> np.ndarray:
    """
    About
    -----
    - Converts complex I/Q samples into DeepSig-style frame format
    - Output X shape is:
        (num_frames, 1024, 2)

    Parameters
    ----------
    - iq_samples (np.ndarray):
        - Complex I/Q samples

    - frame_size (int):
        - DEFAULT: 1024
        - Number of I/Q samples per frame

    Raises
    ------
    - ValueError:
        - If frame_size is less than 1

    Returns
    -------
    - np.ndarray
        - DeepSig-style X array
    """
    if frame_size < 1:
        raise ValueError("frame_size must be >= 1")

    if len(iq_samples) == 0:
        return np.zeros((0, frame_size, 2), dtype=np.float32)

    num_frames = math.ceil(len(iq_samples) / frame_size)
    padded_length = num_frames * frame_size
    padded_iq = np.zeros(padded_length, dtype=np.complex64)
    padded_iq[:len(iq_samples)] = iq_samples

    framed_iq = padded_iq.reshape(num_frames, frame_size)

    X = np.zeros((num_frames, frame_size, 2), dtype=np.float32)
    X[:, :, 0] = framed_iq.real
    X[:, :, 1] = framed_iq.imag

    return X


def calculate_bit_error_rate(
        true_bits: List[int],
        recovered_bits: List[int]
) -> float:
    """
    About
    -----
    - Calculates bit error rate between the original and recovered bitstreams

    Parameters
    ----------
    - true_bits (list[int]):
        - Original bitstream

    - recovered_bits (list[int]):
        - Recovered bitstream

    Raises
    ------
    - None

    Returns
    -------
    - float
        - Bit error rate from 0.0 to 1.0
    """
    compare_length = min(len(true_bits), len(recovered_bits))

    if compare_length == 0:
        return 0.0

    true_array = np.array(true_bits[:compare_length])
    recovered_array = np.array(recovered_bits[:compare_length])

    bit_errors = np.sum(true_array != recovered_array)

    return float(bit_errors / compare_length)


def run_text_simulation(
        input_text: str,
        true_modulation_type: str = "QPSK",
        receiver_modulation_type: str | None = None,
        correct_modulation_enabled: bool = True,
        encryption_enabled: bool = True,
        encryption_key: str = "secret",
        receiver_encryption_key: str | None = None,
        correct_encryption_enabled: bool = True,
        samples_per_symbol: int = 8,
        frame_size: int = 1024
) -> Dict[str, Any]:
    """
    About
    -----
    - Runs the full text-to-bits, optional encryption, modulation, demodulation,
      optional decryption, and recovered-text simulation
    - Supports toggling correct/wrong modulation and correct/wrong encryption

    Parameters
    ----------
    - input_text (str):
        - Text message to transmit

    - true_modulation_type (str):
        - DEFAULT: 'QPSK'
        - True modulation used by transmitter

    - receiver_modulation_type (str | None):
        - DEFAULT: None
        - Modulation assumed by receiver when correct_modulation_enabled is False

    - correct_modulation_enabled (bool):
        - DEFAULT: True
        - If True, receiver uses true modulation
        - If False, receiver uses receiver_modulation_type

    - encryption_enabled (bool):
        - DEFAULT: True
        - If True, bitstream is XOR encrypted before modulation

    - encryption_key (str):
        - DEFAULT: 'secret'
        - True encryption key

    - receiver_encryption_key (str | None):
        - DEFAULT: None
        - Receiver key when correct_encryption_enabled is False

    - correct_encryption_enabled (bool):
        - DEFAULT: True
        - If True, receiver uses encryption_key
        - If False, receiver uses receiver_encryption_key

    - samples_per_symbol (int):
        - DEFAULT: 8
        - Number of I/Q samples per symbol

    - frame_size (int):
        - DEFAULT: 1024
        - DeepSig-style frame size

    Raises
    ------
    - ValueError:
        - If unsupported modulation or empty encryption key is used

    Returns
    -------
    - dict
        - Simulation results
    """
    true_modulation_type = true_modulation_type.upper()

    if receiver_modulation_type is None:
        receiver_modulation_type = "BPSK" if true_modulation_type != "BPSK" else "QPSK"

    receiver_modulation_type = receiver_modulation_type.upper()

    if correct_modulation_enabled:
        assumed_modulation_type = true_modulation_type

    else:
        assumed_modulation_type = receiver_modulation_type

    if receiver_encryption_key is None:
        receiver_encryption_key = "wrong_key"

    if correct_encryption_enabled:
        assumed_encryption_key = encryption_key

    else:
        assumed_encryption_key = receiver_encryption_key

    original_bits = text_to_bits(input_text)

    if encryption_enabled:
        transmitted_bits = xor_bits(original_bits, encryption_key)

    else:
        transmitted_bits = list(original_bits)

    iq_samples, num_padding_bits = modulate_bits(
        bits=transmitted_bits,
        modulation_type=true_modulation_type,
        samples_per_symbol=samples_per_symbol
    )

    deepsig_X = iq_to_deepsig_frames(
        iq_samples=iq_samples,
        frame_size=frame_size
    )

    recovered_transmitted_bits = demodulate_iq(
        iq_samples=iq_samples,
        modulation_type=assumed_modulation_type,
        samples_per_symbol=samples_per_symbol,
        num_padding_bits=num_padding_bits,
        expected_num_bits=len(transmitted_bits)
    )

    if encryption_enabled:
        recovered_original_bits = xor_bits(recovered_transmitted_bits, assumed_encryption_key)

    else:
        recovered_original_bits = list(recovered_transmitted_bits)

    recovered_text = bits_to_text(recovered_original_bits)

    plain_text_roundtrip = bits_to_text(original_bits)

    transmitted_bit_error_rate = calculate_bit_error_rate(
        true_bits=transmitted_bits,
        recovered_bits=recovered_transmitted_bits
    )

    original_bit_error_rate = calculate_bit_error_rate(
        true_bits=original_bits,
        recovered_bits=recovered_original_bits
    )

    return {
        "input_text": input_text,
        "plain_text_roundtrip": plain_text_roundtrip,

        "true_modulation_type": true_modulation_type,
        "assumed_modulation_type": assumed_modulation_type,
        "correct_modulation_enabled": correct_modulation_enabled,

        "encryption_enabled": encryption_enabled,
        "true_encryption_key": encryption_key,
        "assumed_encryption_key": assumed_encryption_key,
        "correct_encryption_enabled": correct_encryption_enabled,

        "original_bits": original_bits,
        "transmitted_bits": transmitted_bits,
        "recovered_transmitted_bits": recovered_transmitted_bits,
        "recovered_original_bits": recovered_original_bits,

        "original_bits_display": format_bits(original_bits),
        "transmitted_bits_display": format_bits(transmitted_bits),
        "recovered_transmitted_bits_display": format_bits(recovered_transmitted_bits),
        "recovered_original_bits_display": format_bits(recovered_original_bits),

        "iq_samples": iq_samples,
        "iq_shape": tuple(iq_samples.shape),
        "deepsig_X": deepsig_X,
        "deepsig_X_shape": tuple(deepsig_X.shape),
        "num_padding_bits": num_padding_bits,

        "recovered_text": recovered_text,
        "transmitted_bit_error_rate": transmitted_bit_error_rate,
        "original_bit_error_rate": original_bit_error_rate,
        "successful_recovery": recovered_text == input_text
    }


def print_simulation_report(
        simulation_results: Dict[str, Any],
        show_full_keys: bool = False
) -> None:
    """
    About
    -----
    - Prints a readable report for one simulation run

    Parameters
    ----------
    - simulation_results (dict):
        - Output from run_text_simulation

    - show_full_keys (bool):
        - DEFAULT: False
        - Whether to display true/assumed encryption keys

    Raises
    ------
    - None

    Returns
    -------
    - None
    """
    print("=" * 100)
    print("TRUE PLAINTEXT")
    print("=" * 100)
    print("Input text:", simulation_results["input_text"])
    print("Plain text roundtrip:", simulation_results["plain_text_roundtrip"])
    print("True bits:", simulation_results["original_bits_display"])
    print()

    print("=" * 100)
    print("TRANSMISSION SETUP")
    print("=" * 100)
    print("True modulation:", simulation_results["true_modulation_type"])
    print("Receiver assumed modulation:", simulation_results["assumed_modulation_type"])
    print("Correct modulation enabled:", simulation_results["correct_modulation_enabled"])
    print("Encryption enabled:", simulation_results["encryption_enabled"])

    if show_full_keys:
        print("True encryption key:", simulation_results["true_encryption_key"])
        print("Receiver encryption key:", simulation_results["assumed_encryption_key"])

    else:
        print("True encryption key:", "[masked]")
        print("Receiver encryption key:", "[masked]")

    print("Correct encryption enabled:", simulation_results["correct_encryption_enabled"])
    print("Transmitted bits:", simulation_results["transmitted_bits_display"])
    print("I/Q shape:", simulation_results["iq_shape"])
    print("DeepSig-style X shape:", simulation_results["deepsig_X_shape"])
    print()

    print("=" * 100)
    print("RECEIVER OUTPUT")
    print("=" * 100)
    print("Recovered transmitted bits:", simulation_results["recovered_transmitted_bits_display"])
    print("Recovered original bits:", simulation_results["recovered_original_bits_display"])
    print("Recovered text:", simulation_results["recovered_text"])
    print("Transmitted bit error rate:", f"{simulation_results['transmitted_bit_error_rate']:.4f}")
    print("Original bit error rate:", f"{simulation_results['original_bit_error_rate']:.4f}")
    print("Successful recovery:", simulation_results["successful_recovery"])
    print()

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def main():
    """
    About
    -----
    - Runs a default simulation example

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
    results = run_text_simulation(
        input_text="HELLO RF WORLD",
        true_modulation_type="QPSK",
        correct_modulation_enabled=True,
        encryption_enabled=True,
        encryption_key="secret",
        correct_encryption_enabled=True
    )

    print_simulation_report(results)


if __name__ == "__main__":
    main()

# =================================================================================================
# END Main Function
# =================================================================================================
