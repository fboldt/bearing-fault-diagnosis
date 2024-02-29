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

def experimenter_kfold(dataset, split='groupkfold_acquisition', clf=None):
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

def experimenter(sources, target, clf=CNN1D()):
    print("loading sources acquisitions...")
    Xtr, ytr = get_acquisitions(sources)
    print("training estimator...")
    clf.fit(Xtr, ytr)
    print("loading target acquisitions...")
    # doing kfold for acquisition    
    experimenter_kfold(dataset=target[0][1], split='groupkfold_acquisition', clf=clf)

datasets = [
    # ("Paderborn (dbg)", Paderborn(config='dbg')),
    ("CWRU (mert)", CWRU(config='mert')),
    ("UORED (mert)", UORED_VAFCLS(config='mert')),
    # ("MFPT (dbg)", MFPT(config='dbg')),
    ("Hust (mert)", Hust(config='mert')),
]

if __name__ == "__main__":
    source = datasets[:2]
    print(f'source -> {source}')
    target = list(set(datasets) - set(source))
    print(f'target -> {target}')
    experimenter(source, target)
