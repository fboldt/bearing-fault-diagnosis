from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import inspect

def train_estimator(estimator_training_function, Xtr, ytr, groups=None):
    if "Xva" in str(inspect.signature(estimator_training_function)):
        n_splits=10
        splitter = StratifiedGroupKFold(n_splits) if groups is not None else StratifiedKFold(n_splits)
        for (train_partial_index, val_index) in splitter.split(Xtr, ytr, groups):
            Xtr_partial, ytr_partial = Xtr[train_partial_index], ytr[train_partial_index]
            Xva, yva = Xtr[val_index], ytr[val_index]
            break
        estimator_training_function(Xtr_partial, ytr_partial, Xva, yva)
    else:
        estimator_training_function(Xtr, ytr)
