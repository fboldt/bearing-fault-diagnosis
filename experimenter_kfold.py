from datasets.cwru import CWRU
from datasets.mfpt import MFPT
from estimators.cnn1d import CNN1D
from sklearn.metrics import accuracy_score


def experimenter(dataset, split='kfold', clf=CNN1D()):
    accuracies = []
    print(f"Slipt type: {split}")
    for Xtr, ytr, Xte, yte in getattr(dataset, split)():
        print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        accuracies.append(accuracy_score(yte, ypred))
        print(f"fold {len(accuracies)} accuracy: {accuracies[-1]}")
    print(f"mean accuracy: {sum(accuracies)/len(accuracies)}")


if __name__ == "__main__":
    experimenter(CWRU(bearing_names_file="cwru_bearings.csv"))
    experimenter(MFPT())
    