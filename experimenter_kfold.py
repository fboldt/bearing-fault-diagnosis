from datasets.cwru import CWRU
from datasets.mfpt import MFPT
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from sklearn.metrics import accuracy_score, confusion_matrix

def kfold(dataset, split='groupkfold_acquisition', repetitions=3, clf=CNN1D()):
    total = []
    for i in range(repetitions):
        print(f"{i+1}/{repetitions}: {dataset}")
        accuracies = []
        print(f"Slipt type: {split}")
        for Xtr, ytr, Xte, yte in getattr(dataset, split)():
            print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)
            clf.fit(Xtr, ytr)
            ypr = clf.predict(Xte)
            accuracies.append(accuracy_score(yte, ypr))
            print(f"fold {len(accuracies)}/{dataset.n_folds} accuracy: {accuracies[-1]}")
            labels = list(set(yte))
            print(f" {labels}")
            print(confusion_matrix(yte, ypr, labels=labels))
        mean_accuracy = sum(accuracies)/len(accuracies)
        print(f"mean accuracy: {mean_accuracy}")
        total.append(mean_accuracy)
    print(f"total mean accuracy: {sum(total)/len(total)}")


def experimenter():
    print("KFold acquisition")
    # kfold(CWRU(config='mert'))
    # kfold(MFPT(config='all'))
    # kfold(Paderborn(config='reduced'))
    # kfold(Hust(config='mert'))
    kfold(UORED_VAFCLS(config='mert'))

if __name__ == "__main__":
    experimenter()