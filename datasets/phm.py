"""
Class definition of PHM Bearing dataset download and acquisitions extraction.
"""

import scipy.io
import numpy as np
import os
import re
from sklearn.model_selection import KFold, StratifiedGroupKFold
import sys


# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)


list_of_bearings_all = [file[:-4] for file in os.listdir('raw_phm')]

list_of_bearings_reduced = [file[:-4] for file in os.listdir('raw_phm') if file.find('_S1_') != -1]

def download_file(url, dirname, bearing):
    pass


class PHM():   

    def get_bearings(self):
        list_of_bearings = eval("list_of_bearings_"+self.config)
        bearing_file_names = [name+'.mat' for name in list_of_bearings]
        bearing_label = [label for label in bearing_file_names]    
        return np.array(bearing_label), np.array(bearing_file_names)

    def __str__(self):
        return f"PHM ({self.config})"

    def __init__(self, sample_size=8400, n_channels=1, acquisition_maxsize=420_000, 
                 config="all", resampled_rate=42_000):
        self.sample_size = sample_size
        self.n_channels = n_channels
        self.acquisition_maxsize = acquisition_maxsize
        self.config = config
        self.resampled_rate = resampled_rate
        self.rawfilesdir = "raw_phm"
        # self.url = ""
        self.n_folds = 3
        self.bearing_labels, self.bearing_names = self.get_bearings()
        # self.accelerometers = [][:self.n_channels]
        self.signal_data = np.empty((0, self.sample_size, 1))
        self.labels = []
        self.keys = []

        # Files Paths ordered by bearings
        files_path = {}
        for key, bearing in zip(self.bearing_labels, self.bearing_names):
            files_path[key] = os.path.join(self.rawfilesdir, bearing)
        self.files = files_path

    def download(self):
        pass

    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        cwd = os.getcwd()
        for x, key in enumerate(self.files):
            matlab_file = scipy.io.loadmat(os.path.join(cwd, self.files[key]))
            acquisition = []
            print('\r', f" loading acquisitions {100*(x+1)/len(self.files):.2f} %", end='')
            label = re.findall(r'TYPE\d+', self.files[key])[0]
            acquisition.append(matlab_file['data'].reshape(1, -1)[0][:self.acquisition_maxsize])
            acquisition = np.array(acquisition)
            if len(acquisition.shape)<2 or acquisition.shape[0]<self.n_channels:
                continue
            for i in range(acquisition.shape[1]//self.sample_size):
                sample = acquisition[:,(i * self.sample_size):((i + 1) * self.sample_size)]
                self.signal_data = np.append(self.signal_data, np.array([sample.T]), axis=0)
                self.labels = np.append(self.labels, label)
                self.keys = np.append(self.keys, key)
        print(f"  ({len(self.labels)} examples) | labels: {set(self.labels)}")

    def get_acquisitions(self):
        if len(self.labels) == 0:
            self.load_acquisitions()
        groups = self.groups()
        return self.signal_data, self.labels, groups
             
    def group_acquisition(self):
        groups = []
        hash = dict()
        for i in self.keys:
            if i not in hash:
                hash[i] = len(hash)
            groups = np.append(groups, hash[i])
        return groups

    def groups(self):
        return self.group_acquisition()

if __name__ == "__main__":
    dataset = PHM(config='reduced')
    dataset.download()
    dataset.load_acquisitions()
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    keys = list(set(dataset.keys))
    print("keys", np.array(keys), f"({len(keys)})")
