from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from experimenter_kfold import kfold
from experimenter_cross_dataset import get_acquisitions

def transfer_learning(sources, target, clf=CNN1D()):
    print("loading sources acquisitions...")
    Xtr, ytr = get_acquisitions(sources)
    print("pretraining estimator...")
    clf.prefit(Xtr, ytr)
    kfold(target, clf=clf)

sources = [
    CWRU(config='all'),
    MFPT(config='all'),
    Paderborn(config='all'),
    Hust(config='niob'),
]
target = UORED_VAFCLS(config='mert')

def experimenter():
    print("Transfer learning")
    transfer_learning(sources, target)

if __name__ == "__main__":
    experimenter()
    