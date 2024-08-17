# Bearing Fault Diagnosis Framework

This framework is designed for fault diagnosis in bearings using vibration data and machine learning algorithms.

## Installation Guide

Before you start, it's recommended to create a virtual environment to manage your dependencies. Below are two methods for installing the necessary dependencies: **automatic** and **manual**.

### 1. Automatic Installation (Recommended)

To automatically install the dependencies, use the provided installation script. This method is the easiest and ensures that all required libraries are installed correctly.

**Steps:**

1. Clone the repository:

    ```bash
    git clone https://github.com/fboldt/bearing-fault-diagnosis.git
    cd bearing-fault-diagnosis
  

2. Create a virtual environment (optional but recommended):

    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
  

3. Run the installation script:
   
    ```bash
    bash install_dependencies.sh
    

The script will prompt you to select the installation type:

* **Option 1**: Install minimal dependencies
* **Option 2**: Install dependencies for running a CNN
* **Option 3**: Install all dependencies

### 2. Manual Installation

If you prefer to install the dependencies manually, follow the steps below.

**Steps:**

1. Install the minimal dependencies:
   
    ```bash
    pip install numpy scipy requests pyunpack rarfile scikit-learn imblearn PyWavelets
  

2. If you want to run a CNN, additionally install TensorFlow:

    ```bash
    pip install tensorflow
  
 
## Running Experiments

To run an experiment, use the provided **experimenter_kfold.py** script.

    ```bash
    python experimenter_kfold.py
