# EthicalHacking-AI-and-RadioFrequencies

The intent is primarily a proof-of-concept, information gain, and broad applications of Ethcial Hacking concepts (i.e. Security Layers) and Artificial Intelligence (AI).

This project takes DEEPSIG's RADIOML 2018.01A dataset and simplifies it for the initial objective to train a model in signal modulation classification to then apply ethical hacking concepts.





## Table of Contents

- [Acknowledgements](#acknowledgements)
- [Environment Setup](#environment-setup)
- [Objectives](#objectives)
- [Known Issues](#known-issues)





## Acknowledgements

[Back to Table of Contents](#table-of-contents)

- **DATASET:** [DEEPSIG RADIOML 2018.01A](https://www.deepsig.ai/datasets/)
    - **LICENSE:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
    - **CHANGES MADE:**
        - Dataset reduced down to only SNR 30 for baseline establishment
        - This reduced dataset was then used to feature engineer a multitude of seemingly typical information of a signals anaylsis' pipeline





## Environment Setup

[Back to Table of Contents](#table-of-contents)

You have two options for setting up your Python environment:

### Option 1: Conda (Recommended)

**Conda** is an open-source environment and package manager that makes it easy to manage Python versions and dependencies. If you do not already use an environment manager, you may want to familiarize yourself with one since it helps avoid conflicts and makes reproducibility easier.  I use Conda and I think it's the easiest (Though I haven't used other packages)

**Steps:**
1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Clone this repository (Or just download ```environment.yml```).
3. Navigate to the `final` directory.
4. Create the environment using the provided `environment.yml`:
	```bash
	conda env create -f environment.yml
	conda activate COMP3703
	```

### Option 2: pip (Use with Caution)

You can also use `pip` with the `environment.txt` file. Using pip does not manage Python versions, so you must ensure your Python version matches the requirements.

**Steps:**
1. Ensure you are using a compatible Python version (see above).
2. Clone this repository (Or just download environment.txt).
3. Navigate to the `final` directory.
4. Install dependencies:
	```bash
	pip install -r environment.txt
    ```





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
- [] Feature Engineering
    - [] Extract I/Q Stats/Metrics
    - [] Extract Magnitude Stats/Metrics
    - [] Extract Power Stats/Metrics
    - [] Extract Phase Stats/Metrics
    - [] Extract Frequency Stats/Metrics
    - [] Extract FFT Stats/Metrics
    - [] Extract Spectral Stats/Metrics
    - [] Extract Constellation
    - [] Create and save feature engineered dataset
- [] Train and Test Models
- [] Identify Best Model
- [] Signal Simulation?
    - [] Determine "Hacking" Equivalence
    - [] Create a simulated encrypted signal
    - [] Simulate decrypting the signal via hacking?
- [] TBD: Ethical Hacking Portion





## Known Issues

[Back to Table of Contents](#table-of-contents)
