from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class Reshape(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    return X.reshape(X.shape[0], X.shape[1])
  
model = RandomForestClassifier(1000, max_features=25) 
over = SMOTE()
under = RandomUnderSampler()
steps = [('reshape', Reshape()), ('over', over), ('under', under), ('model', model)]
clf = Pipeline(steps=steps) 

class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_features=None):
        super().__init__()
        model = RandomForestClassifier(n_estimators=n_estimators, 
                                       max_features=max_features) 
        steps = [('reshape', Reshape()), 
                 ('over', SMOTE()), 
                 ('under', RandomUnderSampler()), 
                 ('model', model)]
        self.clf = Pipeline(steps=steps) 
    def fit(self, X, y):
        self.clf.fit(X, y)
    def predict(self,X):
        return self.clf.predict(X)
