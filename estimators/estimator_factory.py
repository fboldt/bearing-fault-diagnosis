from abc import ABC, abstractmethod
from estimators import randomforest, cnn1d

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
        return cnn1d.CNN1D(epochs=self.epochs, checkpoint=self.checkpoint, verbose=self.verbose)


class RandomForestEstimator(EstimatorFactory):
    def __init__(self, n_estimators=100, max_features=None):
        self.n_estimators = n_estimators
        self.max_features = max_features        
    def estimator(self):
        return randomforest.RandomForest(n_estimators=self.n_estimators, max_features=self.max_features)