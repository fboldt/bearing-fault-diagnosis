import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from experimenter_kfold import experimenter as kfold
from experimenter_cross_dataset import experimenter as cross_dataset
from experimenter_pretrain import experimenter as transfer_learning

def experimenter():
    kfold()
    cross_dataset()
    transfer_learning()

if __name__ == "__main__":
    experimenter()