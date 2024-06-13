from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import SGDClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# from estimators.features.heterogeneous import Heterogeneous
from estimators.features.statisticaltime import StatisticalTime
from sklearn.preprocessing import StandardScaler


class Reshape(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    return X.reshape(X.shape[0], X.shape[1]*X.shape[2])

class SGDClf(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iter=1000, tol=1e-3, loss="hinge"):
        self.max_iter = max_iter
        self.tol = tol
        self.loss = loss
        super().__init__()
        model = SGDClassifier(max_iter=1000, tol=1e-2, loss=self.loss) 
        steps = [('reshape', Reshape()), 
                 ('scale', StandardScaler()),
                 ('over', SMOTE()), 
                 ('under', RandomUnderSampler()), 
                 ('feature', StatisticalTime()),
                 ('model', model)]
        self.clf = Pipeline(steps=steps)
    def model(self):
        return self.clf
    def fit(self, X, y):
        self.clf.fit(X, y)
    def predict(self,X):
        return self.clf.predict(X)
