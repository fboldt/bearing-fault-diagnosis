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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
import numpy as np
import time
from utils.resampling import resample_data
from estimators.estimator_factory import EstimatorFactory

import librosa

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
def kfold(datasets, clfmaker, repetitions=3, target_sr=12000):
    total = np.array([])
    if isinstance(datasets, Iterable):
        X, y, groups, orig_sr = get_acquisitions(datasets)
        n_folds = 10
        for dataset in datasets:
            n_folds = min(n_folds, dataset.n_folds)
    else:
        X, y, groups, orig_sr = datasets.get_acquisitions()        
        n_folds = datasets.n_folds
        logging.info(datasets)
    logging.info('-----------------------------------------')
    total_accuracies = []
    for i in range(repetitions):
        logging.info(f"{i+1}/{repetitions}: ")
        accuracies = []
        f1_scores = []
        kf = StratifiedGroupKFold(n_splits=n_folds) #, shuffle=True)
        init = time.time()        
        logging.info(f"X.shape {X.shape}")
        for train_index, test_index in kf.split(X, y, groups):
            Xtr, ytr = X[train_index], y[train_index]
            Xte, yte = X[test_index], y[test_index]  
            print(f"set(yte): {np.unique(yte)}")          
            # training
            clf = clfmaker.get_estimator(orig_sr=orig_sr, target_sr=target_sr)
            train_estimator(clf.fit, Xtr, ytr, groups[train_index])
            # predicting
            Xte_rs = resample_data(Xte, orig_sr=orig_sr, target_sr=target_sr)
            ypr = clf.predict(Xte_rs)
            accuracies.append(accuracy_score(yte, ypr))
            f1_scores.append(f1_score(yte, ypr, average='macro', labels=['N', 'I', 'O', 'B']))
            # experiment log
            logging.info(f"fold {len(accuracies)}/{n_folds} accuracy: {accuracies[-1]}")
            labels = list(set(yte))
            logging.info(f" {labels}")
            logging.info(confusion_matrix(yte, ypr, labels=labels))
        final = time.time()
        logging.info(f'Processing time: {final-init}')
        mean_accuracy = sum(accuracies)/len(accuracies)
        mean_f1_score = sum(f1_scores)/len(f1_scores)
        logging.info(f"mean accuracy: {mean_accuracy}")
        logging.info(f"mean f1-score: {mean_f1_score}")
        total = np.append(total, mean_accuracy)
        total_accuracies.extend(accuracies)
    logging.info('-----------------------------------------')
    logging.info(f"Total Mean Accuracy: {np.mean(total)}")
    logging.info(f"Standard Deviation: {np.std(total)}")
    logging.info(f"Total Accuracies:\n {total_accuracies}")
    logging.info('-----------------------------------------')

# Set the debug mode and define the datasets
debug = True
datasets = [  # debug mode 
    # CWRU(cache_file = "cache/cwru_FE.npy"),
    # CWRU(cache_file = "cache/cwru_DE.npy"),
    # CWRU(cache_file = "cache/cwru_FEDE.npy"),    
    CWRU(cache_file = "cache/cwru_48k.npy"),    
        
    # CWRU(cache_file = "cache/cwru_48k.npy"),    
    # Hust(cache_file="cache/hust_B5.npy")
    # Hust(cache_file="cache/hust_all.npy")    
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

# SAMPLING RATES
# hust -> 51200
# cwru -> 12000/48000
target_sr = 48000

# Define and run the experimenter
def experimenter(datasets=datasets, clfmaker=factory, repetitions=1, target_sr=target_sr):
    kfold(datasets, clfmaker=clfmaker, repetitions=repetitions, target_sr=target_sr)

# Run the experimenter
if __name__ == "__main__":
    experimenter(clfmaker=factory, repetitions=1)
