"""
Class definition of Mafalda Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedGroupKFold
import urllib
import sys
import os
import requests
import zipfile
from pyunpack import Archive

# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)

list_of_bearings_dbg = [
    'N_0_0_0', 'N_0_0_1', 'N_0_0_2', 'N_0_0_3',  
    'B_O_0_0', 'B_O_0_1', 'B_O_0_2', 'B_O_0_3', 
    'C_O_0_0', 'C_O_0_1', 'C_O_0_2', 'C_O_0_3', 
    'O_O_0_0', 'O_O_0_1', 'O_O_0_2', 'O_O_0_3', 
]

list_of_bearings_healthy = []

list_of_bearings_nio = []

list_of_bearings_all = []

list_of_bearings_cmert = []

list_of_bearings_mert = []


list_of_bearings_reduced = [
    'N_0_0_0', 'N_0_0_24', 'N_0_0_48', 'N_0_6_0', 'N_0_6_24', 'N_0_6_48',
    'N_0_10_0', 'N_0_10_24', 'N_0_10_47', 'N_0_15_0', 'N_0_15_24', 'N_0_15_47',
    'N_0_20_0', 'N_0_20_24', 'N_0_20_48', 'N_0_25_0', 'N_0_25_24', 'N_0_25_46',
    'N_0_30_0', 'N_0_30_24', 'N_0_30_46', 'N_0_35_0', 'N_0_35_24', 'N_0_35_44',
    'B_O_0_0',  'B_O_0_24',  'B_O_0_48',  'B_O_6_0',  'B_O_6_24',  'B_O_6_42', 
    'B_O_20_0', 'B_O_20_12', 'B_O_20_24', 'B_O_35_0', 'B_O_35_10', 'B_O_35_19', 
    'C_O_0_0',  'C_O_0_24',  'C_O_0_48',  'C_O_6_0',  'C_O_6_24',  'C_O_6_48', 
    'C_O_20_0', 'C_O_20_24', 'C_O_20_48', 'C_O_35_0', 'C_O_35_20', 'C_O_35_40', 
    'O_O_0_0',  'O_O_0_24',  'O_O_0_48',  'O_O_6_0',  'O_O_6_24',  'O_O_6_48', 
    'O_O_20_0', 'O_O_20_24', 'O_O_20_48', 'O_O_35_0', 'O_O_35_20', 'O_O_35_40', 
]


def download_file(url, dirname, filename):
    response = requests.get(url)
    
    if response.status_code == 200:
        filepath = os.path.join(dirname, filename)
        
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print(f"File saved as '{filepath}'")
    else:
        print(f"Error downloading the file: Status code {response.status_code}")


def extract_zip(zip_file, extract_dir):
    # Create the extract directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Open the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract all contents to the extract directory
        zip_ref.extractall(extract_dir)
    
    print("Files extracted successfully.")


def read_csv_data(file_name):
    df = pd.read_csv(file_name)
    data = {}
    for column in df.columns:
        data[column] = df[column].tolist()
    return data



def downsample(data, original_freq, post_freq):
    if original_freq < post_freq:
        print('Downsample is not required')
        return    
    step = 1 / abs(original_freq - post_freq)
    indices_to_delete = [i for i in range(0, np.size(data), round(step*original_freq))]
    return np.delete(data, indices_to_delete)


class Mafaulda():
    """
    Mafalda class wrapper for database download and acquisition.
    """

    def get_bearings(self):
        list_of_bearings = eval("list_of_bearings_"+self.config)
        bearing_file_names = [name+'.mat' for name in list_of_bearings]
        bearing_label = [label for label in bearing_file_names]    
        return np.array(bearing_label), np.array(bearing_file_names)


    def __str__(self):
        return f"Mafaulda ({self.config})"


    def __init__(self, sample_size=8400, n_channels=1, acquisition_maxsize=420_000, 
                 config="dbg", resampled_rate=42_000):
        self.sample_size = sample_size
        self.n_channels = n_channels
        self.acquisition_maxsize = acquisition_maxsize
        self.config = config
        self.resampled_rate = resampled_rate
        self.rawfilesdir = "raw_mafaulda_reduced"
        self.url = [
            "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/normal.tgz",
            "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/imbalance.tgz",
            "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/underhang.tgz"
        ]
        self.n_folds = 3
        self.bearing_labels, self.bearing_names = self.get_bearings()
        self.accelerometers = ['Underhang_Axial', 'Underhang_Radial', 'Underhang_Tangential',
                               'Overhang_Axial', 'Overhang_Radial', 'Overhang_Tangential'][:self.n_channels]
        self.signal_data = np.empty((0, self.sample_size, len(self.accelerometers)))
        self.labels = []
        self.keys = []
        
        # Files Paths ordered by bearings
        files_path = {}
        for key, bearing in zip(self.bearing_labels, self.bearing_names):
            files_path[key] = os.path.join(self.rawfilesdir, bearing)
        self.files = files_path


    def download(self):

        urls = self.url
        dirname = self.rawfilesdir
        
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        zip_name = [
            "normal.zip",
            "imbalance.zip",
            "underhang.zip",
            "onverhang.zip"
        ]


        for idx, url in enumerate(urls):
            # download_file(url, dirname, zip_name[idx])
            zip_file = os.path.join('raw_mafaulda_test', zip_name[idx])
            extract_zip = self.rawfilesdir
            extract_zip(zip_file, extract_zip)



    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        cwd = os.getcwd()
        # print(cwd)
        for x, key in enumerate(self.files):
            matlab_file = scipy.io.loadmat(os.path.join(cwd, self.files[key]))           
            acquisition = []
            print('\r', f" loading acquisitions {100*(x+1)/len(self.files):.2f} %", end='')
            acquisition.append(matlab_file["Overhang_Radial"].reshape(1, -1)[0][:self.acquisition_maxsize])
            acquisition = np.array(acquisition)
            acquisition = downsample(acquisition, 50000, self.resampled_rate).reshape(1, -1)
            if len(acquisition.shape)<2 or acquisition.shape[0]<self.n_channels:
                continue
            for i in range(acquisition.shape[1]//self.sample_size):
                sample = acquisition[:,(i * self.sample_size):((i + 1) * self.sample_size)]
                self.signal_data = np.append(self.signal_data, np.array([sample.T]), axis=0)
                self.labels = np.append(self.labels, key[0])
                self.keys = np.append(self.keys, key)
        print(f"  ({len(self.labels)} examples) | labels: {set(self.labels)}")
    
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


if __name__ == "__main__":
    dataset = Mafaulda(config='dbg')
    # dataset.download()
    dataset.load_acquisitions()
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    keys = list(set(dataset.keys))
    print("keys", np.array(keys), f"({len(keys)})")