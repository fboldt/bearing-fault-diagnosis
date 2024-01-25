"""
Class definition of MFPT Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedShuffleSplit, GroupShuffleSplit
import shutil
import zipfile
import sys
import ssl
import requests

# Unpack Tools
from pyunpack import Archive

# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)

def download_file(url, dirname, zip_name):
    print("Downloading Bearings Data.")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'XYZ/3.0'})
        gcontext = ssl.SSLContext()
        f = urllib.request.urlopen(req, timeout=10, context=gcontext)
        file_size = int(f.headers['Content-Length'])
        dir_path = os.path.join(dirname, zip_name)
        if not os.path.exists(dir_path):
            downloaded_obj = requests.get(url)
            with open(dir_path, "wb") as file:
                file.write(downloaded_obj.content)
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
    dir_mfpt_data = "MFPT Fault Data Sets"
    dir_bearing_data = os.path.join(dirname, dir_mfpt_data)
    if not os.path.exists(dir_bearing_data):
        file_name = dir_bearing_zip
        Archive(file_name).extractall(dirname)
        extracted_files_qnt = sum([len(files) for r, d, files in os.walk(dir_bearing_data)])
    else:
        extracted_files_qnt = sum([len(files) for r, d, files in os.walk(dir_bearing_data)])
    zf = zipfile.ZipFile(dir_bearing_zip)
    zip_files_qnt = len(zf.namelist())

    if zip_files_qnt != extracted_files_qnt:
        shutil.rmtree(dir_bearing_data)
        print("Extracted Files Incorrect. Extracting Again.")
        extract_zip(dirname, zip_name)


class MFPT():
    """
    MFPT class wrapper for database download and acquisition.

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
      Download raw compressed files from MFPT website
    load_acquisitions()
      Extract data from files
    """
    def __init__(self):
        self.rawfilesdir = "mfpt_raw"
        self.url="https://mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip"
        self.n_folds = 5
        #self.sample_size = 8192
        self.n_channels = 2
        self.sample_size = 4096
        self.n_samples_acquisition = 100  # used for FaultNet
        self.signal_data = np.empty((0, self.sample_size // self.n_channels, self.n_channels))
        self.labels = []
        self.keys = []
        """
        The MFPT dataset is divided into 3 kinds of states: normal state, inner race
        fault state, and outer race fault state (N, IR, and OR), where three baseline
        data and three outer race fault were gathered at a sampling frequency of 97656 Hz and under 270 lbs of
        load for 6 seconds; seven outer race fault data were gathered at a sampling frequency of
        48828 Hz and, respectively, under 25, 50, 100, 150, 200, 250, and 300 lbs 
        of load, and seven inner race fault data were gathered at a sampling 
        frequency of 48828 Hz and, respectively, under 0, 50, 100, 150, 200, 250, 
        and 300 lbs of load, all for 3 seconds.
        """
        files_path = {}
        # Normal
        files_path["Normal_0"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_1")
        files_path["Normal_1"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_2")
        files_path["Normal_2"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_3")
        # OR
        files_path["OR_0"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions/OuterRaceFault_1")
        files_path["OR_1"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions/OuterRaceFault_2")
        files_path["OR_2"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions/OuterRaceFault_3")
        files_path["OR_3"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_1")
        files_path["OR_4"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_2")
        files_path["OR_5"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_3")
        files_path["OR_6"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_4")
        files_path["OR_7"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_5")
        files_path["OR_8"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_6")
        files_path["OR_9"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_7")
        # IR
        files_path["IR_0"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_1")
        files_path["IR_1"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_2")
        files_path["IR_2"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_3")
        files_path["IR_3"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_4")
        files_path["IR_4"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_5")
        files_path["IR_5"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_6")
        files_path["IR_6"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_7")
        self.files = files_path

    def download(self):
        """
        Download and extract compressed files from MFPT website.
        """
        url = self.url
        dirname = self.rawfilesdir
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        zip_name = "MFPT-Fault-Data-Sets-20200227T131140Z-001.zip"
        print("Downloading and Extracting ZIP file:")
        download_file(url, dirname, zip_name)
        extract_zip(dirname, zip_name)
        print("Dataset Loaded.")

    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        for key in self.files:
            matlab_file = scipy.io.loadmat(self.files[key])
            if len(key) == 8:
                vibration_data_raw = matlab_file['bearing'][0][0][1]
            else:
                vibration_data_raw = matlab_file['bearing'][0][0][2]
            vibration_data = np.array([ elem for singleList in vibration_data_raw for elem in singleList])
            for i in range(len(vibration_data)//self.sample_size):
                sample = vibration_data[(i * self.sample_size):((i + 1) * self.sample_size)]
                sample = np.array([sample]).reshape(1, -1, 2)
                self.signal_data = np.append(self.signal_data, sample, axis=0)
                self.labels = np.append(self.labels, key[0])
                self.keys = np.append(self.keys, key)

    def kfold(self):
        if len(self.signal_data) == 0:
            self.load_acquisitions()
        kf = KFold(n_splits=self.n_folds, shuffle=True)
        for train, test in kf.split(self.signal_data):
            # print("Train Index: ", train, "Test Index: ", test)
            yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

    def stratifiedkfold(self):
        if len(self.signal_data) == 0:
            self.load_acquisitions()
        kf = StratifiedShuffleSplit(n_splits=self.n_folds)
        for train, test in kf.split(self.signal_data, self.labels):
            # print("Train Index: ", train, "Test Index: ", test)
            yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

    def groupkfold_acquisition(self):
        if len(self.signal_data) == 0:
            self.load_acquisitions()
        groups = []
        for i in self.keys:
            groups = np.append(groups, i)
        kf = GroupShuffleSplit(n_splits=self.n_folds)
        for train, test in kf.split(self.signal_data, self.labels, groups):
            # print("Train Index: ", train, "Test Index: ", test)
            yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

if __name__ == "__main__":
    dataset = MFPT()
    # dataset.download()
    dataset.load_acquisitions()
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    keys = list(set(dataset.keys))
    print("keys", np.array(keys), f"({len(keys)})")