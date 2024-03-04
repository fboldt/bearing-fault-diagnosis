from datasets.cwru import CWRU
from datasets.uored_vafcls import UORED_VAFCLS
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn
from estimators.cnn1dpre import CNN1DPre
from experimenter_kfold import experimenter as experimenter_kfold
from experimenter_cross_dataset import get_acquisitions

def experimenter(sources, target, clf=CNN1DPre()):
    print("loading sources acquisitions...")
    Xtr, ytr = get_acquisitions(sources)
    print("pretraining estimator...")
    clf.prefit(Xtr, ytr)
    experimenter_kfold(target, clf=clf)


sources = [
    ("Paderborn (all)", Paderborn(config='all')),
    ("CWRU (12k)", CWRU(config='12k')),
    ("Hust (all)", Hust(config='all')),  
]

target = UORED_VAFCLS(config='mert')

if __name__ == "__main__":
    print(f'sources -> {sources}')
    print(f'target -> {target}')
    experimenter(sources, target)
