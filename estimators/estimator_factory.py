from abc import ABC, abstractmethod

class EstimatorFactory(ABC):
    @abstractmethod
    def estimator(self):
        pass

    def get_estimator(self):
        return self.estimator()

class RandomForestEstimator(EstimatorFactory):
    def __init__(self, n_estimators=100, max_features=None):
        self.n_estimators = n_estimators
        self.max_features = max_features        
    def estimator(self):
        from estimators import randomforest
        return randomforest.RandomForest(n_estimators=self.n_estimators, max_features=self.max_features)
