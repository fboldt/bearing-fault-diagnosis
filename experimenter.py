import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.mafaulda import Mafaulda
from datasets.paderborn import Paderborn
from estimators.cnn1d import CNN1D
from experimenter_kfold import experimenter as kfold
from experimenter_cross_dataset import experimenter as cross_dataset
from experimenter_pretrain import experimenter as transfer_learning
from datetime import datetime
import time

from utils.show_info import show_title
from utils.save_output import ConsoleOutputToFile

debug = False

datasets = [
    MFPT(config='all'),
    Hust(config='mert'),
    CWRU(config='nio'),
] if debug else [
    CWRU(config='all'),
    Hust(config='all'),
    UORED_VAFCLS(config='all'),
    Paderborn(config='all'),
]

kfold_repetitions = 1 if debug else 10
epochs = 10 if debug else 1000
clf = CNN1D(epochs=epochs)

def experimenter():
    for i in range(len(datasets)):
        target = datasets[i:i+1]
        sources = list(set(datasets) - set(target))
        show_title("Kfold")
        kfold(target, repetitions=kfold_repetitions, clf=clf)    
        # show_title("Cross Dataset")
        # cross_dataset(sources, target, clf=clf)
        show_title("Transfer Learning")
        transfer_learning(sources, target[0], repetitions=kfold_repetitions, clf=clf)


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
