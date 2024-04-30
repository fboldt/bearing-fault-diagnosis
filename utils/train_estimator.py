from sklearn.model_selection import StratifiedGroupKFold

def train_estimator(estimator_training_function, Xtr, ytr, groups=None):
    if groups is not None:
        group_kfold = StratifiedGroupKFold(n_splits=5)
        for (train_partial_index, val_index) in group_kfold.split(Xtr, ytr, groups):
            Xtr_partial, ytr_partial = Xtr[train_partial_index], ytr[train_partial_index]
            Xva, yva = Xtr[val_index], ytr[val_index]
            break
        try:
            estimator_training_function(Xtr_partial, ytr_partial, Xva, yva)
        except:
            estimator_training_function(Xtr, ytr)
    else:
        estimator_training_function(Xtr, ytr)