from datasets.cwru import CWRU
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.phm import PHM
from estimators.cnn1d import CNN1D
from utils.train_estimator import train_estimator
from utils.get_acquisitions import get_acquisitions
from collections.abc import Iterable
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import copy

def kfold(datasets, repetitions=3, clf=CNN1D()):
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
            print(f"fold {len(accuracies)}/{n_folds} accuracy: {accuracies[-1]}")
            labels = list(set(yte))
            print(f" {labels}")
            print(confusion_matrix(yte, ypr, labels=labels))
        mean_accuracy = sum(accuracies)/len(accuracies)
        print(f"mean accuracy: {mean_accuracy}")
        total.append(mean_accuracy)
    print(f"total mean accuracy: {sum(total)/len(total)}")

debug = True
epochs = 50
verbose = 2
acquisition_maxsize=None

datasets = [
    PHM(config="all_tr", acquisition_maxsize=acquisition_maxsize),
    # PHM(config="motor_tr", acquisition_maxsize=acquisition_maxsize),
    # PHM(config="gearbox_tr", acquisition_maxsize=acquisition_maxsize),
    # PHM(config="leftaxlebox_tr", acquisition_maxsize=acquisition_maxsize),
] if debug else [
    CWRU(config='all'),
    Hust(config='all'),
    MFPT(config='all'),
    Ottawa(config='all'),
    Paderborn(config='all'),
    UORED_VAFCLS(config='all'),
]

def experimenter(datasets=datasets, repetitions=3, clf=CNN1D()):
    kfold(datasets, repetitions=repetitions, clf=clf)

if __name__ == "__main__":
    experimenter(repetitions=1, clf=CNN1D(epochs=epochs,verbose=verbose))
