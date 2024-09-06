import numpy as np
import time
import os

from collections.abc import Iterable

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix

from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.hust import Hust
# from datasets.mfpt import MFPT

from utils.acquisition_handler import get_acquisitions
from utils.model_training import train_estimator
from utils.logger import log_message, configure_logger

from estimators.estimator_factory import EstimatorFactory


# Define the k-fold procedure
def kfold(datasets, clfmaker, repetitions=3):
    if isinstance(datasets, Iterable):
        X, y, groups = get_acquisitions(datasets)
        n_folds = 10
        for dataset in datasets:
            n_folds = min(n_folds, dataset.n_folds)
    else:
        signal, groups = datasets.get_acquisitions()
        X, y = signal.data, signal.labels        
        n_folds = datasets.n_folds
    
    # set the minimum number of fold
    group_size = np.size(np.unique(groups))
    n_folds = group_size if n_folds > group_size else n_folds
    log_message(f' Number of folds: {n_folds}')    
    
    start_time = time.time()        
    total_accuracies = np.array([])
    for i in range(repetitions):
        log_message('--------------------------')
        log_message({"X.shape": X.shape, f"{i+1} ":repetitions})
        kf = StratifiedGroupKFold(n_splits=n_folds)    
        accuracies = []    
        for x, (train_index, test_index) in enumerate(kf.split(X, y, groups)):
            Xtr, ytr = X[train_index], y[train_index]
            Xte, yte = X[test_index], y[test_index]        
            clf = clfmaker.get_estimator()        
            train_estimator(clf.fit, Xtr, ytr, groups[train_index])        
            ypr = clf.predict(Xte)        
            accuracies.append(accuracy_score(yte, ypr))
        
            # Logging info
            labels = list(set(yte))
            mean_accuracy = sum(accuracies)/len(accuracies)        
            total_accuracies = np.append(total_accuracies, mean_accuracy)

            # Recording results
            log_message({f' Fold {x+1} accuracy': accuracies[-1]})
            log_message(f' {labels}')
            log_message(confusion_matrix(yte, ypr, labels=labels))
            log_message({"Mean accuracy": mean_accuracy})
    
    # processing time
    end_time = time.time()
    processing_time = (end_time - start_time)
    minutes = int((processing_time / 60) % 60)
    hours = int((processing_time / 60) // 60) 
    formated_time = f'{minutes}min{int(processing_time % 60)}s' if minutes < 60 else f'{hours}h{minutes}min{int(processing_time%60)}s'   
    
    # show and record accuracy and processing time
    log_message({
        "Total Mean Accuracy": np.mean(total_accuracies),
        "Standard Deviation": np.std(total_accuracies),
        "Processing time": formated_time
    })


# Set the debug mode and define the datasets
debug = True
datasets = [  # debug mode 
    Ottawa(config='all'),
] if debug else [
    CWRU(config='48k'),
    UORED_VAFCLS(config='all'),
    Ottawa(config='niob'),
    Paderborn(config='all'),
    Hust(config='all'),
    # MFPT(config='all'),
    # Paderborn(config='all'),
]


# Initialize the estimator factory
factory = EstimatorFactory()
factory.set_estimator('random_forest')


# Define and run the experimenter
def experimenter(datasets=datasets, clfmaker=factory, repetitions=1):
    kfold(datasets, clfmaker=clfmaker, repetitions=repetitions)


# Run the experimenter
if __name__ == "__main__":
    configure_logger()
    experimenter(clfmaker=factory, repetitions=1)