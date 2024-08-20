from datasets.cwru import CWRU
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.uored_vafcls import UORED_VAFCLS
from utils.acquisition_handler import get_acquisitions
from collections.abc import Iterable
import numpy as np
import time
import os

from estimators.estimator_factory import EstimatorFactory
from utils.model_validation import run_kfold

import logging
from datetime import datetime

# Configure the logger
os.makedirs('experiments', exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('experiments', exist_ok=True)
log_filename = f"experiments/{current_time}_experiment_log.txt"

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler(log_filename),
    logging.StreamHandler()
])

# Define the k-fold procedure
def kfold(datasets, clfmaker, repetitions=3):
    total = np.array([])
    if isinstance(datasets, Iterable):
        X, y, groups = get_acquisitions(datasets)
        n_folds = 10
        for dataset in datasets:
            n_folds = min(n_folds, dataset.n_folds)
    else:
        signal, groups = datasets.get_acquisitions()
        X, y = signal.data, signal.labels        
        n_folds = datasets.n_folds
        logging.info(datasets)
    logging.info('-----------------------------------------')
    init = time.time()        
    for i in range(repetitions):
        logging.info(f"X.shape {X.shape}")
        logging.info(f"{i+1}/{repetitions} repetitions: ")
        accuracies = []
        accuracies = run_kfold(X, y, groups, n_folds, clfmaker)
        mean_accuracy = sum(accuracies)/len(accuracies)
        logging.info(f" Mean accuracy: {mean_accuracy}")
        total = np.append(total, mean_accuracy)
    logging.info('-----------------------------------------')
    logging.info(f"Total Mean Accuracy: {np.mean(total)}")
    logging.info(f"Standard Deviation: {np.std(total)}")
    final = time.time()
    logging.info(f'Processing time: {final-init}')
    logging.info('-----------------------------------------')


# Set the debug mode and define the datasets
debug = True
datasets = [  # debug mode 
    CWRU(config='dbg'),
] if debug else [
    CWRU(config='all'),
    Hust(config='all'),
    MFPT(config='all'),
    Ottawa(config='niob'),
    Paderborn(config='all'),
    UORED_VAFCLS(config='all'),
]


# Initialize the estimator factory
factory = EstimatorFactory()
factory.set_estimator('random_forest')


# Define and run the experimenter
def experimenter(datasets=datasets, clfmaker=factory, repetitions=1):
    kfold(datasets, clfmaker=clfmaker, repetitions=repetitions)


# Run the experimenter
if __name__ == "__main__":
    experimenter(clfmaker=factory, repetitions=10)