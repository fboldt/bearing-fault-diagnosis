import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from experimenter_kfold import experimenter as kfold
from experimenter_cross_dataset import experimenter as cross_dataset
from experimenter_pretrain import experimenter as transfer_learning

datasets = [
    CWRU(config='mert'),
    # MFPT(config='all'),
    # Paderborn(config='reduced'),
    UORED_VAFCLS(config='mert'),
    Hust(config='niob'),
]

sources = datasets[:-1]
target = list(set(datasets) - set(sources))
kfold_repetitions = 3

def experimenter():
    kfold(target, kfold_repetitions)
    cross_dataset(sources, target)
    transfer_learning(sources, target[0], kfold_repetitions)

if __name__ == "__main__":
    experimenter()
