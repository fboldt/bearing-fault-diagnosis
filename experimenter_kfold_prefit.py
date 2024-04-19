from datasets.cwru import CWRU
from datasets.mfpt import MFPT
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.paderborn import Paderborn
from estimators.cnn1d_p import CNN1DP
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

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

def kfold(dataset, split='groupkfold_acquisition', clf=CNN1DP()):
    accuracies = []
    print(f"Slipt type: {split}")
    for Xtr, ytr, Xte, yte in getattr(dataset, split)():
        print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)
        clf.fit(Xtr, ytr)
        ypr = clf.predict(Xte)
        accuracies.append(accuracy_score(yte, ypr))
        print(f"fold {len(accuracies)} accuracy: {accuracies[-1]}")
        labels = list(set(yte))
        print(f" {labels}")
        print(confusion_matrix(yte, ypr, labels=labels))
    print(f"mean accuracy: {sum(accuracies)/len(accuracies)}")

def experimenter(source, target, clf=CNN1DP()):
    X, y = get_acquisitions(source)
    clf.prefit(X, y)
    kfold(target, clf=clf)

datasets = [
    # ("Paderborn (dbg)", Paderborn(config='dbg')),
    ("MFPT (dbg)", MFPT(config='dbg')),
    ("CWRU (nio)", CWRU(config='nio')),
]

if __name__ == "__main__":
    source = datasets[:2]
    target = list(set(datasets) - set(source))
    experimenter(source, target)