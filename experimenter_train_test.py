from datasets.phm import PHM
from utils.train_estimator import train_estimator
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import csv

from estimators.cnn1d import Contructor
epochs = 500
verbose = 2
basename = "phm_18ch"
checkpoint = f"{basename}.keras"
clfmaker = Contructor(epochs=epochs, checkpoint=checkpoint, verbose=verbose)
dataset_tr = PHM(cache_file = f"{basename}_tr.npy")
dataset_te = PHM(cache_file = f"{basename}_te.npy")

def kfold(clf, dataset):
    X, y, groups = dataset.get_acquisitions()
    scores = cross_validate(clf, X, y, groups=groups)
    print(scores['test_score'], sum(scores['test_score'])/5)

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
    for answer in answers:
        print(answer)

if __name__ == "__main__":
    clf = clfmaker.estimator()
    # kfold(clfmaker, dataset)
    # '''
    train(clf, dataset_tr)
    test(clf, dataset_te, "answers5.csv")
    '''
    test(clf, dataset_tr)
    # '''
