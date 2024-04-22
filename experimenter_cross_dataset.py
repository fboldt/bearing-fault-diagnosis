from datasets.cwru import CWRU
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.uored_vafcls import UORED_VAFCLS
from estimators.cnn1d import CNN1D
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from experimenter_kfold import train_estimator
from utils.get_acquisitions import get_acquisitions

def cross_dataset(sources, targets, clf=CNN1D()):
    print("loading sources acquisitions...")
    Xtr, ytr, groups = get_acquisitions(sources)
    print("training estimator...")
    train_estimator(clf.fit, Xtr, ytr, groups)
    print("loading target acquisitions...")
    Xte, yte, _ = get_acquisitions(targets)
    print("inferencing predictions...")
    ypr = clf.predict(Xte)
    print(f"Accuracy {accuracy_score(yte, ypr)}")
    labels = list(set(yte))
    print(f" {labels}")
    print(confusion_matrix(yte, ypr, labels=labels))

datasets = [
    CWRU(config='nio', acquisition_maxsize=84_000),
    Hust(config='dbg', acquisition_maxsize=84_000),
    MFPT(config='dbg', acquisition_maxsize=84_000),
    Ottawa(config='dbg', acquisition_maxsize=84_000),
    Paderborn(config='dbg', acquisition_maxsize=84_000),
    UORED_VAFCLS(config='dbg', acquisition_maxsize=84_000),
]

sources = datasets[:-1]
target = list(set(datasets) - set(sources))

def experimenter(sources=sources, target=target, clf=CNN1D()):
    print("cross dataset")
    cross_dataset(sources, target, clf=clf)

if __name__ == "__main__":
    experimenter(clf=CNN1D(epochs=20))
    