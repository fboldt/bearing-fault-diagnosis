from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from experimenter_kfold import kfold
from experimenter_cross_dataset import get_acquisitions

def transfer_learning(sources, target, split='groupkfold_acquisition', repetitions=3, clf=CNN1D()):
    print("loading sources acquisitions...")
    Xtr, ytr = get_acquisitions(sources)
    print("pretraining estimator...")
    clf.prefit(Xtr, ytr)
    kfold(target, clf=clf, split=split, repetitions=repetitions)

datasets = [
    MFPT(config='dbg'),
    CWRU(config='nio'),
    # Paderborn(config='dbg'),
    # Hust(config='dbg'),
    # UORED_VAFCLS(config='dbg'),
]

sources = datasets[:-1]
target = list(set(datasets) - set(sources))[0]

def experimenter(sources=sources, target=target, split='groupkfold_acquisition', repetitions=1, clf=CNN1D()):
    print("Transfer learning")
    transfer_learning(sources, target, split=split, repetitions=repetitions, clf=clf)

if __name__ == "__main__":
    experimenter(clf=CNN1D(epochs=20))
    