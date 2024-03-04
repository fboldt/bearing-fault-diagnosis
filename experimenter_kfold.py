from datasets.cwru import CWRU
from datasets.mfpt import MFPT
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from estimators.cnn1dpre import CNN1DPre
from sklearn.metrics import accuracy_score, confusion_matrix

def experimenter(dataset, split='groupkfold_acquisition', clf=CNN1DPre()):
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


if __name__ == "__main__":
    # experimenter(CWRU(config='mert'), clf=CNN1DPre())
    # experimenter(MFPT(config='dbg'))
    # experimenter(Paderborn(config='dbg'))
    experimenter(UORED_VAFCLS(config='mert'))
    # experimenter(Hust(config='faulty_healthy'))
