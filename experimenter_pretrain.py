from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def get_acquisitions(datasets):
    accelerometers = 1
    sample_size = 8400
    X = np.empty((0, sample_size, accelerometers))
    y = np.empty((0,))
    for dataset in datasets:
        print(f"Dataset: {dataset[0]}")
        Xtmp, ytmp = dataset[1].get_acquisitions()
        print(f"Dataset: {datasets[0][0]} -> X={Xtmp.shape}, Y={ytmp.shape}")
        X = np.concatenate((X, Xtmp))
        y = np.concatenate((y, ytmp))
        print(f"Final - X={X.shape}, Y={y.shape}")
    return X, y


def experimenter(sources, target, split='groupkfold_acquisition', clf=CNN1D()):
    accuracies = []
    
    print(f"Slipt type: {split}")
    for Xtr, ytr, Xte, yte in getattr(target[1], split)():
        # loading sources acquisitions
        Xtrs, ytrs = get_acquisitions(sources)
        # loading target acquisitions
        print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)
        # gathering training data
        Xtr = np.concatenate((Xtrs, Xtr))
        ytr = np.concatenate((ytrs, ytr))
        # training the model 
        clf.fit(Xtr, ytr)
        # getting the prediction
        ypr = clf.predict(Xte)
        accuracies.append(accuracy_score(yte, ypr))
        print(f"fold {len(accuracies)} accuracy: {accuracies[-1]}")
        labels = list(set(yte))
        print(f" {labels}")
        print(confusion_matrix(yte, ypr, labels=labels))
    print(f"mean accuracy: {sum(accuracies)/len(accuracies)}")


datasets = [
    # ("Paderborn (dbg)", Paderborn(config='dbg')),
    ("CWRU (mert)", CWRU(config='mert')),
    ("UORED (mert)", UORED_VAFCLS(config='mert')),
    # ("MFPT (dbg)", MFPT(config='dbg')),
    ("Hust (mert)", Hust(config='mert')),
]

if __name__ == "__main__":
    target = datasets[-1]
    sources = datasets[:-1]
    print(f'sources -> {sources}')
    print(f'target -> {target}')
    experimenter(sources, target)
