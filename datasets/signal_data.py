import numpy as np
import os


class Signal:

    def __init__(self, dataset_name, cache_filepath):
        self._dataset_name = dataset_name
        self._labels = np.array([], dtype=str)
        self._keys = np.array([], dtype=str)
        self._data = None
        self._is_cached = os.path.exists(cache_filepath)


    def add_acquisitions(self, bearing_label, acquisitions):
        if self.data is None:
            self._data = np.empty((0, acquisitions.shape[1], 1))
        self._data = np.append(self._data, acquisitions, axis=0)
        for _ in range(acquisitions.shape[0]):
            self._labels = np.append(self._labels, bearing_label[0])
            self._keys = np.append(self._keys, bearing_label)

    def check_is_cached(self):
        return self._is_cached

    def save_cache(self, cache_filepath):
        print(' Saving cache')
        directory = cache_filepath.split('/')[0]
        os.makedirs(directory, exist_ok=True)
        with open(cache_filepath, 'wb') as f:
            np.save(f, self._data)
            np.save(f, self._labels)
            np.save(f, self._keys)
        self._is_cached = True
    
    def load_cache(self, cache_filepath):
        print(' Loading cache')
        with open(cache_filepath, 'rb') as f:
            self._data = np.load(f)
            self._labels = np.load(f)
            self._keys = np.load(f)
    

    @property
    def data(self):
        return self._data
    
    @property
    def labels(self):
        return self._labels

    @property
    def keys(self):
        return self._keys
    
    @property
    def dataset_name(self):
        return self._dataset_name
    