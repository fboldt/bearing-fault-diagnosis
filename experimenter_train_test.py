from datasets.phm import PHM
from utils.train_estimator import train_estimator
from sklearn.model_selection import cross_validate, StratifiedGroupKFold
from sklearn.metrics import accuracy_score
import csv

from estimators.cnn1d_phm import Contructor
epochs = 100
verbose = 0
basename = "phm_18ch" # "phm_leftaxlebox" # "phm_gearbox" # "phm_motor" # 
checkpoint = f"{basename}.keras"
clfmaker = Contructor(epochs=epochs, checkpoint=checkpoint, verbose=verbose)
dataset_tr = PHM(cache_file = f"{basename}_tr.npy")
dataset_te = PHM(cache_file = f"{basename}_te.npy")

def kfold(clfmaker, dataset):
    clf = clfmaker.estimator()
    X, y, groups = dataset.get_acquisitions()
    scores = cross_validate(clf, X, y, groups=groups, 
                            cv=StratifiedGroupKFold(n_splits=3))
    print(scores['test_score'], sum(scores['test_score'])/len(scores['test_score']))

def train(clf, dataset):
    X, y, groups = dataset.get_acquisitions()
    train_estimator(clf.fit, X, y, groups)

def test(clf, dataset, csvfile=None):
    X, y, groups = dataset.get_acquisitions()
    keys = dataset.keys
    ypred = clf.predict(X)
    if len(y) > 0:
        print("Accuracy: ", accuracy_score(y, ypred))
        answers = [[keys.split("_")[0],ypred[i]] for i, keys in enumerate(keys)]
    else:
        answers = [[keys.split("_")[1][6:],f"TYPE{ypred[i]}"] for i, keys in enumerate(keys)]
    if csvfile is not None:
        with open(csvfile, 'w') as f:
            write = csv.writer(f)        
            write.writerow(["Sample Number", "Fault Code"])
            write.writerows(answers)
    # for answer in answers:
    #     print(answer)

if __name__ == "__main__":
    for i in range(5):
        # '''
        kfold(clfmaker, dataset_tr)
        '''
        clf = clfmaker.estimator()
        train(clf, dataset_tr)
        print(clf)
        test(clf, dataset_tr)
        test(clf, dataset_te, f"{basename}{i}.csv")
        # '''
