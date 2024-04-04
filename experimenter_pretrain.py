from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from experimenter_kfold import kfold
from experimenter_cross_dataset import get_acquisitions

def transfer_learning(sources, target, repetitions=3, clf=CNN1D()):
    print("loading sources acquisitions...")
    Xtr, ytr = get_acquisitions(sources)
    print("pretraining estimator...")
    clf.prefit(Xtr, ytr)
    print(clf)
    kfold(target, clf=clf, repetitions=repetitions)

datasets = [
    CWRU(config='nio'),
    MFPT(config='dbg'),
    # Paderborn(config='dbg'),
    # Hust(config='dbg'),
    # UORED_VAFCLS(config='dbg'),
]

sources = datasets[:-1]
target = list(set(datasets) - set(sources))[0]

def experimenter(sources=sources, target=target, repetitions=1, clf=CNN1D()):
    print("Transfer learning")
    transfer_learning(sources, target, repetitions=repetitions, clf=clf)

if __name__ == "__main__":
    experimenter(clf=CNN1D(epochs=20))
    