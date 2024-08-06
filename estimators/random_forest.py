import librosa
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from estimators.features.heterogeneous import Heterogeneous
from sklearn.preprocessing import StandardScaler
from utils.resampling import resample_data


class Reshape(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    return X.reshape(X.shape[0], X.shape[1]*X.shape[2])
  
class ResampleData(BaseEstimator, TransformerMixin):
    def __init__(self, target_sr=12000, orig_sr=48000):
        self.target_sr = target_sr
        self.orig_sr = orig_sr
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return resample_data(X, orig_sr=self.orig_sr, target_sr=self.target_sr)        

class RandomForestEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_features=None, orig_sr=48000, target_sr=12000):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.target_sr = target_sr
        self.orig_sr = orig_sr
        super().__init__()
        model = RandomForestClassifier(n_estimators=self.n_estimators, 
                                       max_features=self.max_features, max_depth=10) 
        
        steps = [
            ('resample', ResampleData(target_sr=self.target_sr, orig_sr=self.orig_sr)),
            ('reshape', Reshape()), 
            # ('over', SMOTE()), 
            # ('under', RandomUnderSampler()), 
            ('feature', Heterogeneous()),
            ('scale', StandardScaler()),
            ('model', model)]
        self.clf = Pipeline(steps=steps) 
    def fit(self, X, y):
        self.clf.fit(X, y)
    def predict(self,X):
        return self.clf.predict(X)
