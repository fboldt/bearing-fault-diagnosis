from datasets.cwru import CWRU
from datasets.hust import Hust
from datasets.mfpt import MFPT
from datasets.ottawa import Ottawa
from datasets.paderborn import Paderborn
from datasets.uored_vafcls import UORED_VAFCLS
from estimators.cnn1d import CNN1D
from experimenter_kfold import kfold
from utils.acquisition_handler import get_acquisitions
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.model_training import train_estimator
from estimators.estimator_factory import EstimatorFactory
import copy

def transfer_learning(sources, target, repetitions=3, clf=CNN1D()):
    clf = copy.copy(clf)
    print('type sources:', type(sources))
    Xtr, ytr, groups = get_acquisitions(sources)
    print("Pretraining estimator...")
    train_estimator(clf.prefit, Xtr, ytr, groups)
    print(clf)
    ypred = clf.predict(Xtr)
    print(accuracy_score( ytr, ypred))
    print(confusion_matrix(ytr, ypred))
    kfold(target, clf=clf, repetitions=repetitions)

debug = True
datasets = [
    CWRU(config='dbg', acquisition_maxsize=21_000),
    Ottawa(config='dbg', acquisition_maxsize=21_000),
] if debug else [
    CWRU(config='all'),
    Hust(config='all'),
    MFPT(config='all'),
    Ottawa(config='all'),
    Paderborn(config='all'),
    UORED_VAFCLS(config='all'),
]

# Initialize the estimator factory
factory = EstimatorFactory()
factory.set_estimator('cnn1d')

sources = datasets[:-1]
target = list(set(datasets) - set(sources))[0]

def experimenter(sources=sources, target=target, repetitions=1, clf=CNN1D()):
    print("Transfer learning")
    transfer_learning(sources, target, repetitions, clf)

if __name__ == "__main__":
    experimenter(clf=CNN1D())
    