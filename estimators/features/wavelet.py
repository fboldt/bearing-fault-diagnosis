import numpy as np
from sklearn.base import TransformerMixin
import pywt

class WaveletPackage(TransformerMixin):
  '''
  Extracts Wavelet Package features.
  The features are calculated by the energy of the recomposed signal
  of the leaf nodes coefficients.
  '''
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    def Energy(coeffs, k):
      return np.sqrt(np.sum(np.array(coeffs[-k]) ** 2)) / len(coeffs[-k])
    def getEnergy(wp):
      coefs = np.asarray([n.data for n in wp.get_leaf_nodes(True)])
      return np.asarray([Energy(coefs,i) for i in range(2**wp.maxlevel)])
    return np.array([getEnergy(pywt.WaveletPacket(data=x, wavelet='db4',
                                                  mode='symmetric', maxlevel=4)
                                                  ) for x in X[:]])
  