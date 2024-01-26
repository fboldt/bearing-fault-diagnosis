from datasets.cwru import CWRU
from datasets.mfpt import MFPT
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

if __name__ == "__main__":
    datasets = [("CWRU", CWRU()),
                ("MFPT", MFPT())]
    experimenter(datasets, datasets)
