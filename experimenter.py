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
    # MFPT(config='all'),
    # Paderborn(config='all'),
    UORED_VAFCLS(config='all'),
    CWRU(config='all'),
    Hust(config='all'),
]

sources = datasets[:-1]
target = list(set(datasets) - set(sources))
kfold_repetitions = 10
split= 'groupkfold_severity' if "CWRU" in target.__str__() else 'groupkfold_acquisition'

def experimenter():
    kfold(target, split=split, repetitions=kfold_repetitions)
    # cross_dataset(sources, target)
    transfer_learning(sources, target[0], split=split, repetitions=kfold_repetitions)

if __name__ == "__main__":
    experimenter()
