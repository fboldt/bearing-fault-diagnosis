from datasets.cwru import CWRU
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.uored_vafcls import UORED_VAFCLS
from estimators.cnn1d import CNN1D
from sklearn.metrics import accuracy_score, confusion_matrix
from experimenter_kfold import train_estimator
from utils.get_acquisitions import get_acquisitions
import copy

def cross_dataset(sources, targets, clf=CNN1D()):
    clf = copy.copy(clf)
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

debug = True

datasets = [
    # PHM(config="motor_tr", acquisition_maxsize=64_000)
    CWRU(config='nio', acquisition_maxsize=21_000),
    MFPT(config='dbg', acquisition_maxsize=21_000),
    Hust(config='dbg', acquisition_maxsize=21_000),
    # Ottawa(config='dbg', acquisition_maxsize=21_000),
    # Paderborn(config='dbg', acquisition_maxsize=21_000),
    # UORED_VAFCLS(config='dbg', acquisition_maxsize=21_000),
] if debug else [
    CWRU(config='all'),
    Hust(config='all'),
    MFPT(config='all'),
    Ottawa(config='all'),
    Paderborn(config='all'),
    UORED_VAFCLS(config='all'),
]

sources = datasets[:-1]
target = list(set(datasets) - set(sources))

def experimenter(sources=sources, target=target, clf=CNN1D()):
    print("cross dataset")
    cross_dataset(sources, target, clf=clf)

if __name__ == "__main__":
    experimenter(clf=CNN1D(epochs=20))
    