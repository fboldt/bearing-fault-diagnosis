from datasets.cwru import CWRU
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.uored_vafcls import UORED_VAFCLS
from estimators.cnn1d import CNN1D
from utils.get_acquisitions import get_acquisitions
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

def train_estimator(estimator_training_function, Xtr, ytr, groups=None):
    if groups is not None:
        group_kfold = StratifiedGroupKFold(n_splits=len(set(groups)))
        for (train_partial_index, val_index) in group_kfold.split(Xtr, ytr, groups):
            Xtr_partial, ytr_partial = Xtr[train_partial_index], ytr[train_partial_index]
            Xva, yva = Xtr[val_index], ytr[val_index]
            break
        estimator_training_function(Xtr_partial, ytr_partial, Xva, yva)
    else:
        estimator_training_function(Xtr, ytr)

def kfold(datasets, repetitions=3, clf=CNN1D()):
    total = []
    X, y, groups = get_acquisitions(datasets)
    n_folds = 10
    for dataset in datasets:
        n_folds = min(n_folds, dataset.n_folds)
    for i in range(repetitions):
        print(f"{i+1}/{repetitions}: ")
        for dataset in datasets:
            print(dataset)
        accuracies = []
        kf = StratifiedGroupKFold(n_splits=n_folds)
        for train_index, test_index in kf.split(X, y, groups):
            Xtr, ytr = X[train_index], y[train_index]
            Xte, yte = X[test_index], y[test_index]
            print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)
            train_estimator(clf.fit, Xtr, ytr, groups[train_index])
            print(clf)
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

datasets = [
    CWRU(config='nio', acquisition_maxsize=84_000),
    Hust(config='dbg', acquisition_maxsize=84_000),
    MFPT(config='dbg', acquisition_maxsize=84_000),
    Ottawa(config='dbg', acquisition_maxsize=84_000),
    Paderborn(config='dbg', acquisition_maxsize=84_000),
    UORED_VAFCLS(config='dbg', acquisition_maxsize=84_000),
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

epochs = 50
verbose = 2

if __name__ == "__main__":
    experimenter(repetitions=1, clf=CNN1D(epochs=epochs,verbose=verbose))
