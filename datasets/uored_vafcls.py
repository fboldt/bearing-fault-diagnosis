"""
Class definition of UORED-VAFCLS Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedShuffleSplit, GroupShuffleSplit, StratifiedGroupKFold
import urllib
import sys
import csv

# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)

list_of_bearings_dbg = [
    ("H_1_0", "H_1_0.mat"), ("H_2_0", "H_2_0.mat"), ("H_3_0", "H_3_0.mat"), ("H_4_0", "H_4_0.mat"),
    ("I_1_1", "I_1_1.mat"), ("I_2_2", "I_2_2.mat"), ("I_3_1", "I_3_1.mat"), ("I_4_2", "I_4_2.mat"),
    ("O_6_1", "O_6_1.mat"), ("O_7_2", "O_7_2.mat"), ("O_8_1", "O_8_1.mat"), ("O_9_2", "O_9_2.mat"),
    ("B_11_1", "B_11_1.mat"), ("B_12_2", "B_12_2.mat"), ("B_13_1", "B_13_1.mat"), ("B_14_2", "B_14_2.mat"),
    ("C_16_1", "C_16_1.mat"), ("C_17_2", "C_17_2.mat"), ("C_18_1", "C_18_1.mat"), ("C_19_2", "C_19_2.mat")
]

list_of_bearings_normal = [
    ("H_1_0", "H_1_0.mat"),   ("H_2_0", "H_2_0.mat"),   ("H_3_0", "H_3_0.mat"),   ("H_4_0", "H_4_0.mat"),
    ("H_5_0", "H_5_0.mat"),   ("H_6_0", "H_6_0.mat"),   ("H_7_0", "H_7_0.mat"),   ("H_8_0", "H_8_0.mat"),
    ("H_9_0", "H_9_0.mat"),   ("H_10_0", "H_10_0.mat"), ("H_11_0", "H_11_0.mat"), ("H_12_0", "H_12_0.mat"),
    ("H_13_0", "H_13_0.mat"), ("H_14_0", "H_14_0.mat"), ("H_15_0", "H_15_0.mat"), ("H_16_0", "H_16_0.mat"),
    ("H_17_0", "H_17_0.mat"), ("H_18_0", "H_18_0.mat"), ("H_19_0", "H_19_0.mat"), ("H_20_0", "H_20_0.mat"),
    ("I_1_1", "I_1_1.mat"),   ("I_1_2", "I_1_2.mat"),   ("I_2_1", "I_2_1.mat"),   ("I_2_2", "I_2_2.mat"), 
    ("I_3_1", "I_3_1.mat"),   ("I_3_2", "I_3_2.mat"),   ("I_4_1", "I_4_1.mat"),   ("I_4_2", "I_4_2.mat"), 
    ("I_5_1", "I_5_1.mat"),   ("I_5_2", "I_5_2.mat"),   ("O_6_1", "O_6_1.mat"),   ("O_6_2", "O_6_2.mat"),  
    ("O_7_1", "O_7_1.mat"),   ("O_7_2", "O_7_2.mat"),   ("O_8_1", "O_8_1.mat"),   ("O_8_2", "O_8_2.mat"), 
    ("O_9_1", "O_9_1.mat"),   ("O_9_2", "O_9_2.mat"),   ("O_10_1", "O_10_1.mat"), ("B_11_1", "B_11_1.mat"),  
    ("B_11_2", "B_11_2.mat"), ("B_12_1", "B_12_1.mat"), ("B_12_2", "B_12_2.mat"), ("B_13_1", "B_13_1.mat"),
    ("B_13_2", "B_13_2.mat"), ("B_14_1", "B_14_1.mat"), ("B_14_2", "B_14_2.mat"), ("B_15_1", "B_15_1.mat"), 
    ("B_15_2", "B_15_2.mat"), ("C_16_1", "C_16_1.mat"), ("C_16_2", "C_16_2.mat"), ("C_17_1", "C_17_1.mat"), 
    ("C_17_2", "C_17_2.mat"), ("C_18_1", "C_18_1.mat"), ("C_18_2", "C_18_2.mat"), ("C_19_1", "C_19_1.mat"), 
    ("C_19_2", "C_19_2.mat"), ("C_20_1", "C_20_1.mat"), ("C_20_2", "C_20_2.mat")
]


def read_csv(file_path):
    
    try:
        
        with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            urls = {}          
            for row in csv_reader:
                label = row[0]
                url = row[1]
                urls[label] = url
            return urls
    
    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
    
    except Exception as e:
        print(f"An error occurred: {e}")


def download_file(url, dirname, bearing):
    print("Downloading Bearing Data:", bearing)   
    file_name = bearing

    try:
        req = urllib.request.Request(url, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])
        dir_path = os.path.join(dirname, file_name)        
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        if not os.path.exists(dir_path):
            urllib.request.urlretrieve(url, dir_path)
            downloaded_file_size = os.stat(dir_path).st_size
            if file_size != downloaded_file_size:
                os.remove(dir_path)
                download_file(url, dirname, bearing)
        else:
            return
        
    except Exception as e:
        print("Error occurs when downloading file: " + str(e))
        print("Trying do download again")
        download_file(url, dirname, bearing)


   


class UORED_VAFCLS():
    """
    UORED_VAFCLS class wrapper for database download and acquisition.

    ...
    Attributes
    ----------
    rawfilesdir : str
      directory name where the files will be downloaded
    url : str
      website from the raw files are downloaded
    files : dict
      the keys represent the conditions_acquisition and the values are the files names

    Methods
    -------
    download()
      Download raw files from UORED_VAFCLS website
    load_acquisitions()
      Extract vibration data from files
    """


    def get_cwru_bearings(self):
        list_of_bearings = eval("list_of_bearings_"+self.config)
        bearing_label, bearing_file_names = zip(*list_of_bearings)
        return np.array(bearing_label), np.array(bearing_file_names)


    def __init__(self, sample_size=8400, n_channels=1, acquisition_maxsize=420_000, config="dbg"):
        self.sample_size = sample_size
        self.n_channels = n_channels
        self.acquisition_maxsize = acquisition_maxsize
        self.config = config
        self.rawfilesdir = "raw_uored_vafcls"
        # self.url = "https://engineering.case.edu/sites/default/files/"   list of urls
        self.n_folds = 4
        self.bearing_labels, self.bearing_names = self.get_cwru_bearings()
        self.accelerometers = ['DE'][:self.n_channels]
        self.signal_data = np.empty((0, self.sample_size, len(self.accelerometers)))
        self.labels = []
        self.keys = []

        # Files Paths ordered by bearings
        files_path = {}
        for key, bearing in zip(self.bearing_labels, self.bearing_names):
            files_path[key] = os.path.join(self.rawfilesdir, bearing)
        self.files = files_path

    def download(dirname):
        dir_path_urls = os.path.join("datasets", "uored_vafcls_urls.csv")
        urls = read_csv(dir_path_urls)

        for label, url in urls.items():
            download_file(url, dirname, bearing=label+".mat")

    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        cwd = os.getcwd()
        for x, key in enumerate(self.files):
            matlab_file = scipy.io.loadmat(os.path.join(cwd, self.files[key]))
            acquisition = []
            print('\r', f" loading acquisitions {100*(x+1)/len(self.files):.2f} %", end='')
            for position in self.accelerometers:
                label = self.files[key][len(self.rawfilesdir)+1:-4]
                acquisition.append(matlab_file[label].reshape(1, -1)[0][:self.acquisition_maxsize])
            acquisition = np.array(acquisition)
            if len(acquisition.shape)<2 or acquisition.shape[0]<self.n_channels:
                continue
            for i in range(acquisition.shape[1]//self.sample_size):
                sample = acquisition[:,(i * self.sample_size):((i + 1) * self.sample_size)]
                self.signal_data = np.append(self.signal_data, np.array([sample.T]), axis=0)
                self.labels = np.append(self.labels, key[0])
                self.keys = np.append(self.keys, key)
        print()

    def get_acquisitions(self):
        if len(self.labels) == 0:
            self.load_acquisitions()
        return self.signal_data, self.labels

    def kfold(self):
        if len(self.signal_data) == 0:
            self.load_acquisitions()
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for train, test in kf.split(self.signal_data):
            yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

    def stratifiedkfold(self):
        if len(self.signal_data) == 0:
            self.load_acquisitions()
        kf = StratifiedShuffleSplit(n_splits=self.n_folds, random_state=42)
        for train, test in kf.split(self.signal_data, self.labels):
            yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

    def groupkfold_acquisition(self):
        if len(self.signal_data) == 0:
            self.load_acquisitions()
        groups = []
        for i in self.keys:
            groups = np.append(groups, i)
        kf = StratifiedGroupKFold(n_splits=self.n_folds)
        for train, test in kf.split(self.signal_data, self.labels, groups):
            yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

    def groupkfold_load(self):
        if len(self.signal_data) == 0:
            self.load_acquisitions()
        groups = []
        for i in self.keys:
            groups = np.append(groups, int(i[-1]) % self.n_folds)
        kf = GroupShuffleSplit(n_splits=self.n_folds)
        for train, test in kf.split(self.signal_data, self.labels, groups):
            yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

    def groupkfold_settings(self):
        if len(self.signal_data) == 0:
            self.load_acquisitions()
        groups = []
        for i in self.keys:
            load = i[-1]
            groups = np.append(groups, load)
        kf = GroupShuffleSplit(n_splits=self.n_folds)
        for train, test in kf.split(self.signal_data, self.labels, groups):
            yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

    def groupkfold_severity(self):
        if len(self.signal_data) == 0:
            self.load_acquisitions()
        groups = []
        for i in self.keys:
            if i[0] == "N":
                load_severity = str(i[-1])
            else:
                load_severity = i[2:5]
            groups = np.append(groups, load_severity)
        kf = GroupShuffleSplit(n_splits=self.n_folds)
        for train, test in kf.split(self.signal_data, self.labels, groups):
            yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

if __name__ == "__main__":
    dataset = UORED_VAFCLS(config='normal')
    dataset.download()
    dataset.load_acquisitions()
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    keys = list(set(dataset.keys))
    print("keys", np.array(keys), f"({len(keys)})")
