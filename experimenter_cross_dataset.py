from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def get_acquisitions(datasets):
    first_dataset = False
    for dataset in datasets:
        print(f"Dataset: {dataset[0]}")
        Xtmp, ytmp = dataset[1].get_acquisitions()
        if not first_dataset:
            X, y =  Xtmp, ytmp
            first_dataset = True
        else:
            X = np.concatenate((X, Xtmp))
            y = np.concatenate((y, ytmp))
    return X, y

def experimenter(sources, targets, clf=CNN1D()):
    print("loading sources acquisitions...")
    Xtr, ytr = get_acquisitions(sources)
    print("training estimator...")
    clf.fit(Xtr, ytr)
    print("loading target acquisitions...")
    Xte, yte = get_acquisitions(targets)
    print("inrferencing predictions...")
    ypr = clf.predict(Xte)
    print(f"Accuracy {accuracy_score(yte, ypr)}")
    labels = list(set(yte))
    print(f" {labels}")
    print(confusion_matrix(yte, ypr, labels=labels))

datasets = [
    # ("Paderborn (dbg)", Paderborn(config='dbg')),
    # ("MFPT (dbg)", MFPT(config='dbg')),
    ("CWRU (mert)", CWRU(config='mert')),
    ("Hust (mert)", Hust(config='mert')),
    ("UORED (mert)", UORED_VAFCLS(config='mert')),
]

if __name__ == "__main__":
    source = datasets[:2]
    target = list(set(datasets) - set(source))
    experimenter(source, target)
