
import numpy as np
from sklearn.base import TransformerMixin
from estimators.features.statisticaltime import rms

class StatisticalFrequency(TransformerMixin):
  '''
  Extracts statistical features from the frequency domain.
  '''
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    sig = []
    for x in X[:]:
      fx = np.absolute(np.fft.fft(x)) # transform x from time to frequency domain
      fc = np.mean(fx) # frequency center
      sig.append([
                  fc, # frequency center
                  rms(fx), # RMS from the frequency domain
                  rms(fx-fc), # Root Variance Frequency
                  ])
    return np.array(sig)
