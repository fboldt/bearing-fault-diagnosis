import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from experimenter_kfold import experimenter as kfold
from experimenter_cross_dataset import experimenter as cross_dataset
from experimenter_pretrain import experimenter as transfer_learning
from datetime import datetime
import time

from utils.show_info import show_title
from utils.save_output import ConsoleOutputToFile


datasets = [
    MFPT(config='dbg'),
    CWRU(config='nio'),
    # # Paderborn(config='all'),
    # UORED_VAFCLS(config='all'),
    # Hust(config='all'),
    # CWRU(config='balanced'),
]

sources = datasets[:-1]
target = list(set(datasets) - set(sources))
kfold_repetitions = 1
split= 'groupkfold_severity' if "CWRU" in target.__str__() else 'groupkfold_acquisition'
clf = CNN1D(epochs=10)

def experimenter():

    show_title("Kfold")
    kfold(target, split=split, repetitions=kfold_repetitions, clf=clf)
    
    show_title("Cross Dataset")
    cross_dataset(sources, target, clf=clf)
    
    show_title("Transfer Learning")
    transfer_learning(sources, target[0], split=split, repetitions=kfold_repetitions, clf=clf)


if __name__ == "__main__":
    
    now = datetime.now()
    date_time = now.strftime("%Y.%m.%d_%H.%M.%S")
    filename = date_time + ".txt"

    with ConsoleOutputToFile(filename):        
        it = time.time()    
        experimenter()
        ft = time.time()
        print('Total processing time:', round(ft-it, 2))