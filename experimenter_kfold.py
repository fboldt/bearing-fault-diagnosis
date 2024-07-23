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
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
import numpy as np
import time

from estimators.estimator_factory import EstimatorFactory

# Define the k-fold procedure
def kfold(datasets, clfmaker, repetitions=3):
    total = np.array([])
    if isinstance(datasets, Iterable):
        X, y, groups = get_acquisitions(datasets)
        n_folds = 10
        for dataset in datasets:
            n_folds = min(n_folds, dataset.n_folds)
    else:
        X, y, groups = datasets.get_acquisitions()        
        n_folds = datasets.n_folds
        print(datasets)
    for i in range(repetitions):
        print(f"{i+1}/{repetitions}: ")
        accuracies = []
        kf = StratifiedGroupKFold(n_splits=n_folds) #, shuffle=True)
        init = time.time()        
        print("X.shape", X.shape)
        for train_index, test_index in kf.split(X, y, groups):
            Xtr, ytr = X[train_index], y[train_index]
            Xte, yte = X[test_index], y[test_index]
            clf = clfmaker.get_estimator()
            train_estimator(clf.fit, Xtr, ytr, groups[train_index])
            ypr = clf.predict(Xte)
            accuracies.append(accuracy_score(yte, ypr))
            print(f"fold {len(accuracies)}/{n_folds} accuracy: {accuracies[-1]}")
            labels = list(set(yte))
            print(f" {labels}")
            print(confusion_matrix(yte, ypr, labels=labels))
        final = time.time()
        print('Processing time:', final-init)
        mean_accuracy = sum(accuracies)/len(accuracies)
        print(f"mean accuracy: {mean_accuracy}")
        total = np.append(total, mean_accuracy)
    print('-----------------------------------------')
    print(f"Total Mean Accuracy: {np.mean(total)}")
    print(f"Standard Deviation: {np.std(total)}")
    print('-----------------------------------------')

# Initialize the estimator factory
factory = EstimatorFactory()
factory.set_estimator('random_forest')

# Set the debug mode and define the datasets
debug = True
datasets = [  # debug mode 
    # CWRU(cache_file = "cache/cwru_FE.npy"),
    # CWRU(cache_file = "cache/cwru_DE.npy"),
    # CWRU(cache_file = "cache/cwru_FE_DE.npy"),
    Hust(cache_file="cache/hust_all.npy")
] if debug else [
    CWRU(config='all'),
    Hust(config='all'),
    MFPT(config='all'),
    Ottawa(config='all'),
    Paderborn(config='all'),
    UORED_VAFCLS(config='all'),
]

# Define and run the experimenter
def experimenter(datasets=datasets, clfmaker=factory, repetitions=1):
    kfold(datasets, clfmaker=clfmaker, repetitions=repetitions)

# Run the experimenter
if __name__ == "__main__":
    experimenter(clfmaker=factory, repetitions=5)
