from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from estimators.features.heterogeneous import Heterogeneous

class Reshape(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    return X.reshape(X.shape[0], X.shape[1]*X.shape[2])

class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_features=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        super().__init__()
        model = RandomForestClassifier(n_estimators=self.n_estimators, 
                                       max_features=self.max_features) 
        steps = [('reshape', Reshape()), 
                 ('over', SMOTE()), 
                 ('under', RandomUnderSampler()), 
                 ('feature', Heterogeneous()),
                 ('model', model)]
        self.clf = Pipeline(steps=steps) 
    def fit(self, X, y):
        self.clf.fit(X, y)
    def predict(self,X):
        return self.clf.predict(X)
