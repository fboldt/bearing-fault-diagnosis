from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.model_training import train_estimator
import logging

def run_kfold(X, y, groups, n_folds, clfmaker):
    kf = StratifiedGroupKFold(n_splits=n_folds)    
    accuracies = []    
    for x, (train_index, test_index) in enumerate(kf.split(X, y, groups)):
        Xtr, ytr = X[train_index], y[train_index]
        Xte, yte = X[test_index], y[test_index]        
        clf = clfmaker.get_estimator()        
        train_estimator(clf.fit, Xtr, ytr, groups[train_index])        
        ypr = clf.predict(Xte)        
        accuracies.append(accuracy_score(yte, ypr))
        
        # Logging info
        logging.info(f" Fold {x+1} accuracy: {accuracies[-1]}")
        labels = list(set(yte))
        logging.info(f" {labels}")
        logging.info(confusion_matrix(yte, ypr, labels=labels))
    
    return accuracies
