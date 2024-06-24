from abc import ABC, abstractmethod

class EstimatorFactory(ABC):
    @abstractmethod
    def estimator(self):
        pass

    def get_estimator(self):
        return self.estimator()


class CNN1DEstimator(EstimatorFactory):
    def __init__(self, epochs=100, checkpoint="model.checkpoint.keras", verbose=2):
        self.epochs=epochs
        self.checkpoint=checkpoint
        self.verbose=verbose
    def estimator(self):
        from estimators import cnn1d
        return cnn1d.CNN1D(epochs=self.epochs, checkpoint=self.checkpoint, verbose=self.verbose)


class RandomForestEstimator(EstimatorFactory):
    def __init__(self, n_estimators=100, max_features=None):
        self.n_estimators = n_estimators
        self.max_features = max_features        
    def estimator(self):
        from estimators import randomforest
        return randomforest.RandomForest(n_estimators=self.n_estimators, max_features=self.max_features)


class SGDEstimator(EstimatorFactory):
    def __init__(self, max_iter=1000, tol=1e-3):
        self.max_iter = max_iter
        self.tol = tol        
    def estimator(self):
        from estimators import sgd
        return sgd.SGDClf(max_iter=self.max_iter, tol=self.tol)