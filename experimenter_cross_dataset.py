import os
import copy

from datasets.cwru import CWRU
from datasets.hust import Hust
# from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.uored_vafcls import UORED_VAFCLS
from estimators.estimator_factory import EstimatorFactory
from sklearn.metrics import accuracy_score, confusion_matrix
from experimenter_kfold import train_estimator
from utils.acquisition_handler import get_acquisitions

from utils.logger import configure_logger, log_message

def cross_dataset(sources, target, clf):
    log_message('Sources:')
    for source in sources:
        log_message(f' {source}')    
    clf = copy.copy(clf)
    print(" loading sources acquisitions...")
    Xtr, ytr, groups = get_acquisitions(sources)
    print(" training estimator...")
    train_estimator(clf.fit, Xtr, ytr, groups)
    
    log_message(f'Target:')
    log_message(f' {target[0]}')
    print(" loading target acquisitions...")
    for t in target:
        print(t)
        Xte, yte, _ = get_acquisitions([t])
        print(" inferencing predictions...")
        ypr = clf.predict(Xte)
        print(f"Accuracy {accuracy_score(yte, ypr)}")
        labels = list(set(yte))
        print(f" {labels}")
        print(confusion_matrix(yte, ypr, labels=labels))

paderborn_config = os.getenv('PADERBORN_CONFIG', 'artificial')
hust_config = os.getenv('HUST_CONFIG', '6204,6205,6206,6207,6208')
list_hust_config = hust_config.split(',')

datasets = [
    Paderborn(config=paderborn_config),
    Hust(config=list_hust_config[0]),
    Hust(config=list_hust_config[1]),
    Hust(config=list_hust_config[2]),
    Hust(config=list_hust_config[3]),
    Hust(config=list_hust_config[4]),
    CWRU(config='FE'),
    CWRU(config='DE'),
]

# sources = datasets[:-1]
# target = list(set(datasets) - set(sources))
sources = [datasets[0]]
target = datasets[1:]

# Initialize the estimator factory
factory = EstimatorFactory()
factory.set_estimator('cnn1d')


def experimenter(sources=sources, target=target, clf=factory):
    print("cross dataset")
    cross_dataset(sources, target, clf=clf)

if __name__ == "__main__":
    configure_logger()
    experimenter(clf=factory.get_estimator())
    