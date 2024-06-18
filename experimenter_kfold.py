from datasets.cwru48 import CWRU48k
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
        kf = StratifiedGroupKFold(n_splits=n_folds)
        clf = clfmaker.estimator()
        init = time.time()        
        print("X.shape", X.shape)
        for train_index, test_index in kf.split(X, y, groups):
            Xtr, ytr = X[train_index], y[train_index]
            Xte, yte = X[test_index], y[test_index]
            clf = clfmaker.estimator()
            train_estimator(clf.fit, Xtr, ytr, groups[train_index])
            ypr = clf.predict(Xte)
            accuracies.append(accuracy_score(yte, ypr))
            # print(clf)
            print(f"fold {len(accuracies)}/{n_folds} accuracy: {accuracies[-1]}")
            labels = list(set(yte))
            print(f" {labels}")
            print(confusion_matrix(yte, ypr, labels=labels))
        final = time.time()
        print('Processing time:', final-init)
        mean_accuracy = sum(accuracies)/len(accuracies)
        print(f"mean accuracy: {mean_accuracy}")
        total = np.append(total, mean_accuracy)
    print(f"total mean accuracy: {np.mean(total)}")
    print(f"standard deviation: {np.std(total)}")

'''
from estimators.estimator_factory import RandomForestEstimator 
clfmaker = RandomForestEstimator(n_estimators=1000, max_features=25)
'''
from estimators.estimator_factory import SGDEstimator
clfmaker = SGDEstimator()
# '''
# from estimators.estimator_factory import CNN1DEstimator 
# clfmaker = CNN1DEstimator(epochs=100, verbose=0)


debug = True

datasets = [
    # Paderborn(cache_file = "paderborn_dbg.npy"),
    # Ottawa(cache_file = "ottawa_all.npy"),
    # Hust(cache_file = "hust_dbg.npy"),
    UORED_VAFCLS(cache_file = "uored_all.npy"),
    # MFPT(cache_file = "mfpt_all.npy"),
    # CWRU48k(cache_file = "cwru48k_dbg.npy"),
    # CWRU48k(cache_file = "cwru48k_balanced.npy"),
    # PHM(cache_file = "phm_motor_tr.npy"),
    # PHM(cache_file = "phm_gearbox_tr.npy"),
    # PHM(cache_file = "pzzhm_leftaxlebox_tr.npy"),
    # PHM(cache_file = "phm_18ch_tr100.npy"),
    # PHM(cache_file = "phm_18ch_tr.npy"),
] if debug else [
    CWRU48k(config='all'),
    Hust(config='all'),
    MFPT(config='all'),
    Ottawa(config='all'),
    Paderborn(config='all'),
    UORED_VAFCLS(config='all'),
]

def experimenter(datasets=datasets, clfmaker=clfmaker, repetitions=1):
    kfold(datasets, clfmaker=clfmaker, repetitions=repetitions)

if __name__ == "__main__":
    experimenter(clfmaker=clfmaker, repetitions=1)
