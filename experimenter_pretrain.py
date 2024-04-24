from datasets.cwru import CWRU
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.uored_vafcls import UORED_VAFCLS
from estimators.cnn1d import CNN1D
from experimenter_kfold import kfold
from utils.train_estimator import train_estimator
from experimenter_cross_dataset import get_acquisitions
from sklearn.metrics import accuracy_score, confusion_matrix
import copy

def transfer_learning(sources, target, repetitions=3, clf=CNN1D()):
    clf = copy.copy(clf)
    print("loading sources acquisitions...")
    Xtr, ytr, groups = get_acquisitions(sources)
    print("pretraining estimator...")
    train_estimator(clf.prefit, Xtr, ytr, groups)
    print(clf)
    ypred = clf.predict(Xtr)
    print(accuracy_score( ytr, ypred))
    print(confusion_matrix(ytr, ypred))
    kfold(target, clf=clf, repetitions=repetitions)

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
target = list(set(datasets) - set(sources))[0]

def experimenter(sources=sources, target=target, repetitions=1, clf=CNN1D()):
    print("Transfer learning")
    transfer_learning(sources, target, repetitions=repetitions, clf=clf)

if __name__ == "__main__":
    experimenter(clf=CNN1D(epochs=5))
    