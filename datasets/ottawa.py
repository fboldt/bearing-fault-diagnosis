"""
Class definition of Ottawa Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import scipy.signal
import numpy as np
import os
from sklearn.model_selection import KFold, GroupKFold, StratifiedShuffleSplit
import shutil
import zipfile
import sys

# Unpack Tools
from pyunpack import Archive

# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)


def download_file(url, dirname, zip_name):
    print("Downloading Bearings Data.")

    try:
        req = urllib.request.Request(url, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])

        dir_path = os.path.join(dirname, zip_name)
        if not os.path.exists(dir_path):
            urllib.request.urlretrieve(url, dir_path)
            downloaded_file_size = os.stat(dir_path).st_size
        else:
            downloaded_file_size = os.stat(dir_path).st_size

        if file_size != downloaded_file_size:
            os.remove(dir_path)
            print("File Size Incorrect. Downloading Again.")
            download_file(url, dirname, zip_name)
    except Exception as e:
        print("Error occurs when downloading file: " + str(e))
        print("Trying do download again")
        download_file(url, dirname, zip_name)


def extract_zip(dirname, zip_name):
    print("Extracting Bearings Data.")
    dir_bearing_zip = os.path.join(dirname, zip_name)

    file_name = dir_bearing_zip
    Archive(file_name).extractall(dirname)
    extracted_files_qnt = sum([len(files) for r, d, files in os.walk(dirname)])

    zf = zipfile.ZipFile(dir_bearing_zip)
    zip_files_qnt = len(zf.namelist())

    if zip_files_qnt != (extracted_files_qnt-1):
        shutil.rmtree(dirname)
        print("Extracted Files Incorrect. Extracting Again.")
        extract_zip(dirname, zip_name)


class Ottawa():
    """
    Ottawa class wrapper for database download and acquisition.

    ...
    Attributes
    ----------
    rawfilesdir : str
      directory name where the files will be downloaded
    url : str
      website from the raw files are downloaded
    conditions : dict
      the keys represent the condition code and the values the number of acquisitions and its lengh
    files : dict
      the keys represent the conditions_acquisition and the values are the files names

    Methods
    -------
    download()
      Download raw compressed files from Ottawa website
    load_acquisitions()
      Extract data from files
    """
    def __str__(self):
        return f"Ottawa ({self.config})"
    
    def __init__(self, sample_size=8400, n_channels=1, acquisition_maxsize=None, config='all'):
        self.n_channels = n_channels
        self.sample_size = sample_size
        self.acquisition_maxsize = acquisition_maxsize
        self.config = config
        self.rawfilesdir = "raw_ottawa"
        self.url="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/v43hmbwxpm-1.zip"
        self.n_folds = 4
        self.n_samples_acquisition = 100  # used for FaultNet
        self.signal_data = np.empty((0, self.sample_size, self.n_channels))
        self.labels = []
        self.keys = []

        """
        Ottawa data set description.
        """

        files_path = {}
        # Normal
        files_path["H-A-1"] = os.path.join(self.rawfilesdir, "H-A-1.mat")
        if self.config != 'dbg':
            files_path["H-A-2"] = os.path.join(self.rawfilesdir, "H-A-2.mat")
            files_path["H-A-3"] = os.path.join(self.rawfilesdir, "H-A-3.mat")
            files_path["H-B-1"] = os.path.join(self.rawfilesdir, "H-B-1.mat")
        files_path["H-B-2"] = os.path.join(self.rawfilesdir, "H-B-2.mat")
        if self.config != 'dbg':
            files_path["H-B-3"] = os.path.join(self.rawfilesdir, "H-B-3.mat")
            files_path["H-C-1"] = os.path.join(self.rawfilesdir, "H-C-1.mat")
            files_path["H-C-2"] = os.path.join(self.rawfilesdir, "H-C-2.mat")
        files_path["H-C-3"] = os.path.join(self.rawfilesdir, "H-C-3.mat")
        files_path["H-D-1"] = os.path.join(self.rawfilesdir, "H-D-1.mat")
        if self.config != 'dbg':
            files_path["H-D-2"] = os.path.join(self.rawfilesdir, "H-D-2.mat")
            files_path["H-D-3"] = os.path.join(self.rawfilesdir, "H-D-3.mat")
        # OR
        if self.config != 'dbg':
            files_path["O-A-1"] = os.path.join(self.rawfilesdir, "O-A-1.mat")
            files_path["O-A-2"] = os.path.join(self.rawfilesdir, "O-A-2.mat")
        files_path["O-A-3"] = os.path.join(self.rawfilesdir, "O-A-3.mat")
        files_path["O-B-1"] = os.path.join(self.rawfilesdir, "O-B-1.mat")
        if self.config != 'dbg':
            files_path["O-B-2"] = os.path.join(self.rawfilesdir, "O-B-2.mat")
            files_path["O-B-3"] = os.path.join(self.rawfilesdir, "O-B-3.mat")
            files_path["O-C-1"] = os.path.join(self.rawfilesdir, "O-C-1.mat")
        files_path["O-C-2"] = os.path.join(self.rawfilesdir, "O-C-2.mat")
        if self.config != 'dbg':
            files_path["O-C-3"] = os.path.join(self.rawfilesdir, "O-C-3.mat")
            files_path["O-D-1"] = os.path.join(self.rawfilesdir, "O-D-1.mat")
            files_path["O-D-2"] = os.path.join(self.rawfilesdir, "O-D-2.mat")
        files_path["O-D-3"] = os.path.join(self.rawfilesdir, "O-D-3.mat")
        # IR
        if self.config != 'dbg':
            files_path["I-A-1"] = os.path.join(self.rawfilesdir, "I-A-1.mat")
        files_path["I-A-2"] = os.path.join(self.rawfilesdir, "I-A-2.mat")
        if self.config != 'dbg':
            files_path["I-A-3"] = os.path.join(self.rawfilesdir, "I-A-3.mat")
            files_path["I-B-1"] = os.path.join(self.rawfilesdir, "I-B-1.mat")
            files_path["I-B-2"] = os.path.join(self.rawfilesdir, "I-B-2.mat")
        files_path["I-B-3"] = os.path.join(self.rawfilesdir, "I-B-3.mat")
        files_path["I-C-1"] = os.path.join(self.rawfilesdir, "I-C-1.mat")
        if self.config != 'dbg':
            files_path["I-C-2"] = os.path.join(self.rawfilesdir, "I-C-2.mat")
            files_path["I-C-3"] = os.path.join(self.rawfilesdir, "I-C-3.mat")
            files_path["I-D-1"] = os.path.join(self.rawfilesdir, "I-D-1.mat")
        files_path["I-D-2"] = os.path.join(self.rawfilesdir, "I-D-2.mat")
        if self.config != 'dbg':
            files_path["I-D-3"] = os.path.join(self.rawfilesdir, "I-D-3.mat")
        self.files = files_path

    def download(self):
        """
        Download and extract compressed files from Ottawa website.
        """
        url = self.url
        dirname = self.rawfilesdir
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        zip_name = "v43hmbwxpm-1.zip"
        print("Downloading and Extracting ZIP file:")
        download_file(url, dirname, zip_name)
        extract_zip(dirname, zip_name)
        print("Dataset Loaded.")

    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        for x, key in enumerate(self.files):
            print('\r', f" loading acquisitions {100*(x+1)/len(self.files):.2f} %", end='')
            matlab_file = scipy.io.loadmat(self.files[key])
            if self.acquisition_maxsize:
                vibration_data = np.array([elem for singleList in matlab_file['Channel_1'] for elem in singleList][:self.acquisition_maxsize])
            else:
                vibration_data = np.array([elem for singleList in matlab_file['Channel_1'] for elem in singleList])
            for i in range(len(vibration_data)//self.sample_size):
                sample = vibration_data[(i * self.sample_size):((i + 1) * self.sample_size)]
                sample = np.array([sample]).reshape(1, -1, self.n_channels)
                self.signal_data = np.append(self.signal_data, sample, axis=0)
                self.labels = np.append(self.labels, key[0])
                self.keys = np.append(self.keys, key)
        self.labels[self.labels=='H'] = 'N'
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
    dataset = Ottawa(config='dbg', acquisition_maxsize=84_000)
    # dataset.download()
    dataset.load_acquisitions()
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    keys = list(set(dataset.keys))
    print("keys", np.array(keys), f"({len(keys)})")
