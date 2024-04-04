from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from experimenter_kfold import train_estimator

def get_acquisitions(datasets):
    first_dataset = True
    for dataset in datasets:
        print(dataset)
        Xtmp, ytmp, gtmp = dataset.get_acquisitions()
        if first_dataset:
            X, y, g =  Xtmp, ytmp, gtmp
            first_dataset = False
        else:
            gtmp += np.max(gtmp)
            X = np.concatenate((X, Xtmp))
            y = np.concatenate((y, ytmp))
            g = np.concatenate((g, gtmp))
    return X, y, g

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
    MFPT(config='dbg'),
    Paderborn(config='dbg'),
    Hust(config='dbg'),
    UORED_VAFCLS(config='dbg'),
    CWRU(config='nio'),
]

sources = datasets[:-1]
target = list(set(datasets) - set(sources))

def experimenter(sources=sources, target=target, clf=CNN1D()):
    print("cross dataset")
    cross_dataset(sources, target, clf=clf)

if __name__ == "__main__":
    experimenter(clf=CNN1D(epochs=20))
    