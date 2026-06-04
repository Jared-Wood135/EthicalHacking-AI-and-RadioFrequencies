# EthicalHacking-AI-and-RadioFrequencies

The intent is primarily a proof-of-concept, information gain, and broad applications of Ethcial Hacking concepts (i.e. Security Layers) and Artificial Intelligence (AI).

This project takes DEEPSIG's RADIOML 2018.01A dataset and simplifies it for the initial objective to train a model in signal modulation classification to then apply ethical hacking concepts.





## Table of Contents

- [Acknowledgements](#acknowledgements)
- [Environment Setup](#environment-setup)
- [General Project Process](#general-project-process)
- [Objectives](#objectives)
- [Known Issues](#known-issues)





## Acknowledgements

[Back to Table of Contents](#table-of-contents)

- **DATASET:** [DEEPSIG RADIOML 2018.01A](https://www.deepsig.ai/datasets/)
    - **LICENSE:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
    - **CHANGES MADE:**
        - Dataset reduced down to only SNR 30 for baseline establishment
        - This reduced dataset was then used to feature engineer a multitude of seemingly typical information of a signals intelligence anaylsts' pipeline





## Environment Setup

[Back to Table of Contents](#table-of-contents)

Python version in both environments: 3.13.5

You have two options for setting up your Python environment:

### Option 1: Conda (Recommended)

**Conda** is an open-source environment and package manager that makes it easy to manage Python versions and dependencies. If you do not already use an environment manager, you may want to familiarize yourself with one since it helps avoid conflicts and makes reproducibility easier.  I use Conda and I think it's the easiest (Though I haven't used other packages)

**Steps:**
1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Clone this repository (Or just download ```environment.yml```).
3. Create the environment using the provided `environment.yml`:
	```bash
	conda env create -f environment.yml
	conda activate RF_HACKING_AI
	```

### Option 2: pip (Use with Caution)

You can also use `pip` with the `environment.txt` file. Using pip does not manage Python versions, so you must ensure your Python version matches the requirements.

**Steps:**
1. Ensure you are using a compatible Python version (see above).
2. Clone this repository (Or just download environment.txt).
3. Install dependencies:
	```bash
	pip install -r environment.txt
    ```





## General Project Process

[Back to Table of Contents](#table-of-contents)

**NOTE:**
- The `Presentations` folder has a presentation with/without notes and may give a good enough summation of the entire project in ~10 minutes
- The `KeyImages` folder has some images of what I deem to be in-line with the key takeaways from the current state of the project

### High Level Overview

1. Acquire dataset from DeepSig
2. Initial validation and exploration of data
3. Reduce dataset to only SNR 30
4. Decision to use Random Forest Classification
5. Make baseline model (~36% accuracy; 2048 features; 24 targets)
6. Feature Engineer SIGINT analyst-like features
7. Make Feature Engineered baseline (~72% accuracy; 105 features; 24 targets)
8. Extract top 25 information gain features
9. Make top 25 feature engineered information gain model (~69% accuracy; 25 features; 24 targets)
10. Make simulation with bit-masking for demonstration purposes
11. Create presentation
12. Create paper
13. Review findings and project future implementations

### Code Level Replication
**NOTE:** The notebooks have their codebases located in the same filename, but with the file extension `.py`
1. `Code/acquire.ipynb`
2. `Code/highest_snr_explore.ipynb`
3. `Code/highest_snr_baseline.ipynb`
4. `Code/highest_snr_feature_engineering.ipynb`
5. `Code/highest_snr_feature_engineering_explore.ipynb`
6. `Code/highest_snr_feature_engineering_modeling.ipynb`
7. `Code/vis_tree.ipynb`
8. `Code/highest_snr_discussion.ipynb`
9. `Simulation/demo_text_simulation.ipynb`





## Objectives

[Back to Table of Contents](#table-of-contents)

- [x] Acquire Dataset
- [x] Determine Signal Modulation Classifiers
- [x] Clean Dataset
    - [x] Remove modulation classifiers not intended to keep
    - [x] Handle nulls
    - [x] Make data consistent (i.e. Strings lowercased)
- [x] Determine ML/NN Type and/or Structure
- [x] Initial Exploration of Clean Dataset
- [x] Prepare Dataset
    - [x] Apply encoding if necessary
    - [x] Apply normalization if necessary
    - [x] Apply feature engineering if necessary
- [x] Acquire Baseline Model
- [x] Feature Engineering
    - [x] Extract I/Q Stats/Metrics
    - [x] Extract Magnitude Stats/Metrics
    - [x] Extract Power Stats/Metrics
    - [x] Extract Phase Stats/Metrics
    - [x] Extract Frequency Stats/Metrics
    - [x] Extract FFT Stats/Metrics
    - [x] Extract Spectral Stats/Metrics
    - [x] Extract Constellation
    - [x] Create and save feature engineered dataset
- [x] Train and Test Models
- [x] Identify Best Model
- [x] Signal Simulation?
    - [x] Textual Input To Bits
    - [x] Bits to Textual Output
    - [x] Modulation Masking of Bits
    - [x] Mock Encryption Masking of Bits
    - [x] Simulate Wrong Modulation, Wrong Encryption
    - [x] Simulate Wrong Modulation, Correct Encryption
    - [x] Simulate Correct Modulation, Wrong Encryption
    - [x] Simulate Correct Modulation, Correct Encryption
    - [x] Add audio variant?
- [] Future Implementations
    - [] Explore and model with noisy data
    - [] Use a clustering model particularly with constellations
    - [] Explore and understand what is required to clean a noisy signal to look more like SNR 30
    - [] Explore and understand how to extract different signals effectively in a signal capture with more than one distinct signal
    - [] Explore complex network theories with clustering and any signal math applications where it may be applicable
    - [] If possible, for more legitimacy in work, capture OWN signals and attempt to run through the algortihm
    - [] Flesh out the simulation more to implement more features that would be seen in actual signals
    - [] Attempt to transmit OWN modulated signal, run through refined trained pipeline and see if my code can demodulate and receive the information transmitted on the signal





## Known Issues

[Back to Table of Contents](#table-of-contents)

- Generally, the notebooks are meant to be clean and have reduced code by pulling from their respective codebases.  Near the end of the project due to time constraints, much of the code remained in the notebooks and will be cleaned up in the future
- The simulation portion output is a bit of a mess at the moment, but gets the point across.  This will be cleaned up in the future
- The simulation does a generalized bit-mask methodology to mimic modulation and encryption, I don't believe it actually simulates exactly what is happening however (I need more information to validate my methodology)
