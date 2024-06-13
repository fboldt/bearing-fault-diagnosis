import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets.cwru48 import CWRU48k
from datasets.hust import Hust
# from datasets.mafaulda import Mafaulda
from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.uored_vafcls import UORED_VAFCLS
from estimators.cnn1d import CNN1D
from experimenter_kfold import experimenter as kfold
from experimenter_cross_dataset import experimenter as cross_dataset
from experimenter_pretrain import experimenter as transfer_learning
from datetime import datetime
import time

from utils.show_info import show_title
from utils.save_output import ConsoleOutputToFile

debug = True

datasets = [
    CWRU48k(config='nio', acquisition_maxsize=21_000),
    Hust(config='dbg', acquisition_maxsize=21_000),
    MFPT(config='dbg', acquisition_maxsize=21_000),
    # Ottawa(config='dbg', acquisition_maxsize=21_000),
    # Paderborn(config='dbg', acquisition_maxsize=21_000),
    # UORED_VAFCLS(config='dbg', acquisition_maxsize=21_000),
] if debug else [
    CWRU48k(config='all'),
    Hust(config='all'),
    MFPT(config='all'),
    Ottawa(config='all'),
    Paderborn(config='all'),
    UORED_VAFCLS(config='all'),
]

kfold_repetitions = 1 if debug else 10
epochs = 5 if debug else 1000
verbose = 2 if debug else 0
clf=CNN1D(epochs=epochs, verbose=verbose)

def experimenter():
    for target in datasets:
        sources = list(set(datasets) - set([target]))
        show_title("Kfold")
        kfold(target, repetitions=kfold_repetitions, clf=clf)
        show_title("Transfer Learning")
        transfer_learning(sources, target, repetitions=kfold_repetitions, 
                          clf=clf)


if __name__ == "__main__":
    now = datetime.now()
    date_time = now.strftime("%Y.%m.%d_%H.%M.%S")
    direxp = "experiments"
    if not os.path.exists(direxp):
        os.makedirs(direxp)
    filename = direxp + "/" + date_time + ".txt"
    with ConsoleOutputToFile(filename):        
        it = time.time()    
        experimenter()
        ft = time.time()
        print('Total processing time:', round(ft-it, 2))
