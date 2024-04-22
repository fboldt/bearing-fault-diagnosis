from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from experimenter_kfold import kfold, train_estimator
from experimenter_cross_dataset import get_acquisitions
from sklearn.metrics import accuracy_score, confusion_matrix

def transfer_learning(sources, target, repetitions=3, clf=CNN1D()):
    print("loading sources acquisitions...")
    Xtr, ytr, groups = get_acquisitions(sources)
    print("pretraining estimator...")
    train_estimator(clf.prefit, Xtr, ytr, groups)
    print(clf)
    ypred = clf.predict(Xtr)
    print(accuracy_score( ytr, ypred))
    print(confusion_matrix(ytr, ypred))
    kfold(target, clf=clf, repetitions=repetitions)

datasets = [
    MFPT(config='dbg'),
    # Paderborn(config='dbg'),
    Hust(config='dbg'),
    UORED_VAFCLS(config='dbg'),
    CWRU(config='nio'),
]
sources = datasets[:-1]
target = list(set(datasets) - set(sources))[0]

def experimenter(sources=sources, target=target, repetitions=1, clf=CNN1D()):
    print("Transfer learning")
    transfer_learning(sources, target, repetitions=repetitions, clf=clf)

if __name__ == "__main__":
    experimenter(clf=CNN1D(epochs=5))
    