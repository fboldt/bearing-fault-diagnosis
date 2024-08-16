from datasets.cwru import CWRU
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.phm import PHM
from utils.train_estimator import train_estimator
from utils.get_acquisitions import get_acquisitions
from collections.abc import Iterable
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import time
from estimators.estimator_factory import EstimatorFactory

import logging
from datetime import datetime

# Configure the logger
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        logging.info(f"{i+1}/{repetitions}: ")
        accuracies = []
        kf = StratifiedGroupKFold(n_splits=n_folds)
        logging.info(f"X.shape {X.shape}")
        for train_index, test_index in kf.split(X, y, groups):
            Xtr, ytr = X[train_index], y[train_index]
            Xte, yte = X[test_index], y[test_index]  
            # training
            clf = clfmaker.get_estimator()
            train_estimator(clf.fit, Xtr, ytr, groups[train_index])
            # predicting
            ypr = clf.predict(Xte)
            accuracies.append(accuracy_score(yte, ypr))
            # experiment log
            logging.info(f"fold {len(accuracies)}/{n_folds} accuracy: {accuracies[-1]}")
            labels = list(set(yte))
            logging.info(f" {labels}")
            logging.info(confusion_matrix(yte, ypr, labels=labels))
        mean_accuracy = sum(accuracies)/len(accuracies)
        logging.info(f"mean accuracy: {mean_accuracy}")
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
    Ottawa(config='all'),
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
    experimenter(clfmaker=factory, repetitions=1)
