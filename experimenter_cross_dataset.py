from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import time

def get_acquisitions(datasets):
    first_dataset = False
    for dataset in datasets:
        print(dataset)
        Xtmp, ytmp = dataset.get_acquisitions()
        if not first_dataset:
            X, y =  Xtmp, ytmp
            first_dataset = True
        else:
            X = np.concatenate((X, Xtmp))
            y = np.concatenate((y, ytmp))
    return X, y

def cross_dataset(sources, targets, clf=CNN1D()):
    print("loading sources acquisitions...")
    Xtr, ytr = get_acquisitions(sources)
    print("training estimator...")
    clf.fit(Xtr, ytr)
    print("loading target acquisitions...")
    Xte, yte = get_acquisitions(targets)
    print("inferencing predictions...")
    ypr = clf.predict(Xte)
    print(f"Accuracy {accuracy_score(yte, ypr)}")
    labels = list(set(yte))
    print(f" {labels}")
    print(confusion_matrix(yte, ypr, labels=labels))

datasets = [
    CWRU(config='all'),
    MFPT(config='all'),
    Paderborn(config='all'),
    Hust(config='niob'),
    UORED_VAFCLS(config='mert'),
]

sources = datasets[:-1]
target = list(set(datasets) - set(sources))

def experimenter(sources=sources, target=target):
    tm_init = time.time()
    print("cross dataset")
    cross_dataset(sources, target)
    tm_final = time.time()
    print(f"Processing time: {round(tm_final-tm_init, 2)} seconds")

if __name__ == "__main__":
    experimenter()
    