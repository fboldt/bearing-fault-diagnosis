import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets.cwru import CWRU
from datasets.hust import Hust
# from datasets.mafaulda import Mafaulda
from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.phm import PHM
from datasets.uored_vafcls import UORED_VAFCLS
from experimenter_kfold import experimenter as kfold
from experimenter_cross_dataset import experimenter as cross_dataset
from experimenter_pretrain import experimenter as transfer_learning
from datetime import datetime
import time

from utils.show_info import show_title
from utils.save_output import ConsoleOutputToFile

debug = False

datasets = [
    PHM(cache_file = "phm_motor_tr.npy"),
    PHM(cache_file = "phm_gearbox_tr.npy"),
    PHM(cache_file = "phm_leftaxlebox_tr.npy"),
# '''
]
'''
] if debug else [
    CWRU(config='all'),
    Hust(config='all'),
    MFPT(config='all'),
    Ottawa(config='all'),
    Paderborn(config='all'),
    UORED_VAFCLS(config='all'),
]
#'''

kfold_repetitions = 1 if debug else 3
epochs = 5 if debug else 100
verbose = 2 if debug else 0
from estimators.cnn1d import Contructor
clfmaker = Contructor(epochs=epochs, verbose=verbose)

def experimenter():
    show_title("Kfold")
    kfold(datasets, clfmaker=clfmaker, repetitions=kfold_repetitions)
    # for target in datasets:
    #     sources = list(set(datasets) - set([target]))
    #     show_title("Kfold")
    #     kfold(target, clfmaker=clfmaker, repetitions=kfold_repetitions)
        # show_title("Transfer Learning")
        # transfer_learning(sources, target, repetitions=kfold_repetitions, 
        #                   clf=clf)


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
