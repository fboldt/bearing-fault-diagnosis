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
import copy

def kfold(datasets, repetitions=3, clf=None):
    clf = copy.copy(clf)
    total = []
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
        for train_index, test_index in kf.split(X, y, groups):
            Xtr, ytr = X[train_index], y[train_index]
            Xte, yte = X[test_index], y[test_index]
            train_estimator(clf.fit, Xtr, ytr, groups[train_index])
            ypr = clf.predict(Xte)
            accuracies.append(accuracy_score(yte, ypr))
            print(clf)
            print(f"fold {len(accuracies)}/{n_folds} accuracy: {accuracies[-1]}")
            labels = list(set(yte))
            print(f" {labels}")
            print(confusion_matrix(yte, ypr, labels=labels))
        mean_accuracy = sum(accuracies)/len(accuracies)
        print(f"mean accuracy: {mean_accuracy}")
        total.append(mean_accuracy)
    print(f"total mean accuracy: {sum(total)/len(total)}")

debug = True
'''
from estimators.randomforest import RandomForest
clf = RandomForest(1000, 25)
'''
from estimators.cnn1d import CNN1D
epochs = 100
verbose = 0
clf = CNN1D(epochs=epochs,verbose=verbose)
# '''

datasets = [
    # CWRU(cache_file = "cwru_all_de.npy"),
    PHM(cache_file = "phm_motor_tr100.npy"),
] if debug else [
    CWRU(config='all'),
    Hust(config='all'),
    MFPT(config='all'),
    Ottawa(config='all'),
    Paderborn(config='all'),
    UORED_VAFCLS(config='all'),
]

def experimenter(datasets=datasets, repetitions=3, clf=None):
    kfold(datasets, repetitions=repetitions, clf=clf)

if __name__ == "__main__":
    experimenter(repetitions=3, clf=clf)
