"""
Class definition of Paderborn Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
import urllib
import rarfile
import shutil
import sys

# Unpack Tools
from pyunpack import Archive

# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)

bearing_names_all = [
    "K001", "K002", "K003", "K004", "K005", "K006", 
    "KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA09", "KA15", "KA16", "KA22", "KA30", 
    "KI01", "KI03", "KI04", "KI05", "KI07", "KI08", "KI14", "KI16", "KI17", "KI18", "KI21", 
]

bearing_names_reduced = [
    "K002", "KA03", "KI03",
]

bearing_names_dbg = [
    "K001", "KA01", "KI01",
]

def download_file(url, dirname, dir_rar, bearing):
    print("Downloading Bearing Data:", bearing)
    file_name = bearing + ".rar"
    try:
        req = urllib.request.Request(url + file_name, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])

        dir_path = os.path.join(dirname, dir_rar, file_name)
        if not os.path.exists(dir_path):
            urllib.request.urlretrieve(url + file_name, dir_path)
            downloaded_file_size = os.stat(dir_path).st_size
        else:
            downloaded_file_size = os.stat(dir_path).st_size
        if file_size != downloaded_file_size:
            os.remove(dir_path)
            print("File Size Incorrect. Downloading Again.")
            download_file(url, dirname, dir_rar, bearing)
    except Exception as e:
        print("Error occurs when downloading file: " + str(e))
        print("Trying do download again")
        download_file(url, dirname, dir_rar, bearing)


def extract_rar(dirname, dir_rar, bearing):
    print("Extracting Bearing Data:", bearing)
    dir_bearing_rar = os.path.join(dirname, dir_rar, bearing + ".rar")
    dir_bearing_data = os.path.join(dirname, bearing)
    if not os.path.exists(dir_bearing_data):
        file_name = dir_bearing_rar
        Archive(file_name).extractall(dirname)
        extracted_files_qnt = len([name for name in os.listdir(dir_bearing_data)
                                   if os.path.isfile(os.path.join(dir_bearing_data, name))])
    else:
        extracted_files_qnt = len([name for name in os.listdir(dir_bearing_data)
                   if os.path.isfile(os.path.join(dir_bearing_data, name))])
    rf = rarfile.RarFile(dir_bearing_rar)
    rar_files_qnt = len(rf.namelist())
    if rar_files_qnt != extracted_files_qnt + 1:
        shutil.rmtree(dir_bearing_data)
        print("Extracted Files Incorrect. Extracting Again.")
        extract_rar(dirname, dir_rar, bearing)


class Paderborn():
    """
    Paderborn class wrapper for database download and acquisition.

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
      Download and extract raw compressed files from PADERBORN website
    load_acquisitions()
      Extract vibration data from files
    """

    def get_paderborn_bearings(self):
        # Get bearings to be considered to be
        bearing_names = eval("bearing_names_"+self.config)
        return bearing_names

    def __str__(self):
        return f"Paderborn ({self.config})"

    def __init__(self, sample_size=8400, n_channels=1, acquisition_maxsize=420_000, config="all"):
        self.n_channels = n_channels
        self.sample_size = sample_size
        self.acquisition_maxsize = acquisition_maxsize
        self.rawfilesdir = "raw_paderborn"
        self.config = config
        self.url = "http://groups.uni-paderborn.de/kat/BearingDataCenter/"
        self.n_folds = 4
        self.bearing_names = self.get_paderborn_bearings()
        self.n_acquisitions = 2 if self.config == 'dbg' else 20
        self.signal_data = np.empty((0, self.sample_size, self.n_channels))
        self.labels = []
        self.keys = []

        """
        Associate each file name to a bearing condition in a Python dictionary. 
        The dictionary keys identify the conditions.
    
        In total, experiments with 32 different bearings were performed:
        12 bearings with artificial damages and 14 bearings with damages
        from accelerated lifetime tests. Moreover, experiments with 6 healthy
        bearings and a different time of operation were performed as
        reference states.
    
        The rotational speed of the drive system, the radial force onto the test
        bearing and the load torque in the drive train are the main operation
        parameters. To ensure comparability of the experiments, fixed levels were
        defined for each parameter. All three parameters were kept constant for
        the time of each measurement. At the basic setup (Set no. 0) of the 
        operation parameters, the test rig runs at n = 1,500 rpm with a load 
        torque of M = 0.7 Nm and a radial force on the bearing of F = 1,000 N. Three
        additional settings are used by reducing the parameters one
        by one to n = 900 rpm, M = 0.1 Nm and F = 400 N (set No. 1-3), respectively.
    
        For each of the settings, 20 measurements of 4 seconds each were recorded
        for each bearing. There are a total of 2.560 files.
    
        All files start with the bearing code, followed by the conditions, by an
        algarism representing the setting and end with an algarism representing 
        the sample sequential. All features are separated by an underscore character.
        """
        settings_files = ["N15_M07_F10_", "N09_M07_F10_", "N15_M01_F10_", "N15_M07_F04_"]
        # Files Paths ordered by bearings
        files_path = {}
        for bearing in self.bearing_names:
            if bearing[1] == '0':
                tp = "Normal_"
            elif bearing[1] == 'A':
                tp = "OR_"
            else:
                tp = "IR_"
            for idx, setting in enumerate(settings_files):
                for i in range(1, self.n_acquisitions + 1):
                    key = tp + bearing + "_" + str(idx) + "_" + str(i)
                    files_path[key] = os.path.join(self.rawfilesdir, bearing, setting + bearing +
                                                   "_" + str(i) + ".mat")
        self.files = files_path

    def download(self):
        """
        Download and extract compressed files from Paderborn website.
        """
        # Download RAR Files
        url = self.url
        dirname = self.rawfilesdir
        dir_rar = "rar_files"
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        if not os.path.isdir(os.path.join(dirname, dir_rar)):
            os.mkdir(os.path.join(dirname, dir_rar))
        print("Downloading and Extracting RAR files:")
        for bearing in self.bearing_names:
            download_file(url, dirname, dir_rar, bearing)
            extract_rar(dirname, dir_rar, bearing)
        print("Dataset Loaded.")

    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        cwd = os.getcwd()
        for x, key in enumerate(self.files):
            # print("Loading vibration data:", key)
            print('\r', f" loading acquisitions {100*(x+1)/len(self.files):.2f} %", end='')
            matlab_file = scipy.io.loadmat(os.path.join(cwd, self.files[key]))
            if len(self.files[key]) > 41:
                vibration_data_raw = matlab_file[self.files[key][19:38]]['Y'][0][0][0][6][2]
            else:
                vibration_data_raw = matlab_file[self.files[key][19:37]]['Y'][0][0][0][6][2]

            vibration_data = vibration_data_raw[0][:self.acquisition_maxsize]
            self.n_samples_acquisition = len(vibration_data)//self.sample_size
            for i in range(self.n_samples_acquisition):
                sample = np.empty((self.sample_size, self.n_channels))
                for j in range(self.n_channels):
                    sample[:,j] = vibration_data[(i * self.sample_size):((i + 1) * self.sample_size)]
                sample = np.array([sample]).reshape(1, -1, self.n_channels)
                self.signal_data = np.append(self.signal_data, sample, axis=0)
                self.labels = np.append(self.labels, key[0])
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

    def group_settings(self):
        groups = []
        for i in range(len(self.bearing_names)):
            for k in range(4): # Number of Settings - 4
                for j in range(self.n_samples_acquisition*self.n_acquisitions):
                    groups = np.append(groups, k)
        return groups

    def group_bearings(self):
        if len(self.signal_data) == 0:
            self.load_acquisitions()
        groups = []
        n_keys_bearings = 1
        n_normal = 1
        n_outer = 1
        n_inner = 1
        for key in self.keys:
            if key[0] == 'N':
                groups = np.append(groups, n_normal % self.n_folds)
            if key[0] == 'O':
                groups = np.append(groups, n_outer % self.n_folds)
            if key[0] == 'I':
                groups = np.append(groups, n_inner % self.n_folds)
            if n_keys_bearings < 4 * self.n_acquisitions*self.n_samples_acquisition:
                n_keys_bearings = n_keys_bearings + 1
            elif key[0] == 'N':
                n_normal = n_normal + 1
                n_keys_bearings = 1
            elif key[0] == 'O':
                n_outer = n_outer + 1
                n_keys_bearings = 1
            elif key[0] == 'I':
                n_inner = n_inner + 1
                n_keys_bearings = 1
        return groups


if __name__ == "__main__":
    dataset = Paderborn(config='all')
    dataset.download()
    dataset.load_acquisitions()
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    keys = list(set(dataset.keys))
    print("keys", np.array(keys), f"({len(keys)})")
