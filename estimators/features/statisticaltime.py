
import numpy as np
import scipy.stats as stats
from sklearn.base import TransformerMixin

def rms(x):
  '''
  root mean square
  '''
  x = np.array(x)
  return np.sqrt(np.mean(np.square(x)))

def sra(x):
  '''
  square root amplitude
  '''
  x = np.array(x)
  return np.mean(np.sqrt(np.absolute(x)))**2

def ppv(x):
  '''
  peak to peak value
  '''
  x = np.array(x)
  return np.max(x)-np.min(x)

def cf(x):
  '''
  crest factor
  '''
  x = np.array(x)
  return np.max(np.absolute(x))/rms(x)

def ifa(x):
  '''
  impact factor
  '''
  x = np.array(x)
  return np.max(np.absolute(x))/np.mean(np.absolute(x))

def mf(x):
  '''
  margin factor
  '''
  x = np.array(x)
  return np.max(np.absolute(x))/sra(x)

def sf(x):
  '''
  shape factor
  '''
  x = np.array(x)
  return rms(x)/np.mean(np.absolute(x))

def kf(x):
  '''
  kurtosis factor
  '''
  x = np.array(x)
  return stats.kurtosis(x)/(np.mean(x**2)**2)


class StatisticalTime(TransformerMixin):
  '''
  Extracts statistical features from the time domain.
  '''
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    return np.array([
                     [
                      rms(x), # root mean square
                      sra(x), # square root amplitude
                      stats.kurtosis(x), # kurtosis
                      stats.skew(x), # skewness
                      ppv(x), # peak to peak value
                      cf(x), # crest factor
                      ifa(x), # impact factor
                      mf(x), # margin factor
                      sf(x), # shape factor
                      kf(x), # kurtosis factor
                      ] for x in X[:]
                     ])

