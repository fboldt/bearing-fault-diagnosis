"""
Class definition of Paderborn Bearing dataset download and acquisitions extraction.
"""

import scipy.io
import numpy as np
import os
import rarfile
import shutil
import logging
import requests
import time

from datasets.signal_data import Signal

# Unpack Tools
# from pyunpack import Archive

def bearing_names_all():
    return [
    "K001", "K002", "K003", "K004", "K005", "K006", 
    "KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA09", "KA15", "KA16", "KA22", "KA30", 
    "KI01", "KI03", "KI04", "KI05", "KI07", "KI08", "KI14", "KI16", "KI17", "KI18", "KI21", 
]

def bearing_names_reference():
    return [
        "K001", "K002", "K003", "K004", "K005", "K006",
    ]

def bearing_names_artificial():
    return bearing_names_reference() + [
        "KA01", "KA03", "KA05", "KA06", "KA07", "KA09", "KI01", "KI03", "KI05", "KI07", "KI08"
    ]

def bearing_names_real():
    return bearing_names_reference() + [
        "KA04", "KA15", "KA16", "KA22", "KA30", "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"
    ]

def bearing_names_dbg():
    return [
    "K001", "KA01", "KI16",
]

def manufacturers(): 
    return {
        "K001": "IBU", "K002": "IBU", "K003": "IBU", "K004": "IBU", "K005": "IBU", "K006": "IBU", 
        "KA01": "MTK", "KA03": "MTK", "KA04": "FAG", "KA05": "IBU", "KA06": "IBU", "KA07": "IBU", "KA09": "IBU", "KA15": "FAG", "KA16": "MTK", "KA22": "IBU", "KA30": "MTK", 
        "KI01": "MTK", "KA03": "MTK", "KI03": "MTK", "KI04": "MTK", "KI05": "IBU", "KI07": "IBU", "KI08": "IBU", "KI14": "MTK", "KI16": "FAG", "KI17": "MTK", "KI18": "MTK", "KI21": "FAG", 
    }


def download_file(url, dirname, dir_rar, bearing):    
    print("\nDownloading Bearing Data:", bearing)
    file_name = bearing + ".rar"
    dir_path = os.path.join(dirname, dir_rar, file_name)
    full_url = url + file_name
    try:
        response = requests.get(full_url, stream=True)
        response.raise_for_status() # check the response
        
        # download progress
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192
        progress = 0  
        
        with open(dir_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                progress += len(chunk)
                done = int(50 * progress / total_size)
                print(f"\r[{'=' * done}{' ' * (50-done)}] {progress / (1024*1024):.2f}/{total_size / (1024*1024):.2f} MB", end='')
    
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
        print(f"Error downloading file: {e}. Trying again...)")
        time.sleep(5)
        download_file(url, dirname, dir_rar, bearing)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

def extract_rar(dirname, dir_rar, bearing):
    print("\nExtracting Bearing Data:", bearing)
    dir_bearing_rar = os.path.join(dirname, dir_rar, bearing + ".rar")
    dir_bearing_data = os.path.join(dirname, bearing)
    
    if not os.path.exists(dir_bearing_data):
        os.makedirs(dir_bearing_data)
        file_name = dir_bearing_rar
        
        rf = rarfile.RarFile(file_name)
        total_files = len(rf.namelist())
        
        with rarfile.RarFile(file_name) as rf:
            for i, member in enumerate(rf.infolist(), 1):
                rf.extract(member, path=dirname)
                # Update and display progress
                done = int(50 * i / total_files)
                print(f"\r[{'=' * done}{' ' * (50-done)}] {i}/{total_files} files extracted", end='')
        
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

def get_list_of_bearings(n_acquisitions, config):
        bearing_names = eval("bearing_names_"+config+"()")
        settings_files = ["N15_M07_F10_", "N09_M07_F10_", "N15_M01_F10_", "N15_M07_F04_"]
        list_of_bearings = []
        for bearing in bearing_names:
            if bearing[1] == '0':
                tp = "Normal_"
            elif bearing[1] == 'A':
                tp = "OR_"
            else:
                tp = "IR_"
            for idx, setting in enumerate(settings_files):
                for i in range(1, n_acquisitions + 1):
                    key = tp + bearing + "_" + str(idx) + "_" + str(i)
                    list_of_bearings.append((key, os.path.join(bearing, setting + bearing +
                                                   "_" + str(i) + ".mat")))
        return list_of_bearings


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
    

    def __str__(self):
        return f"Paderborn ({self.config})"


    def __init__(self, sample_size=4096, n_channels=1, acquisition_maxsize=None, config="all"):
        self.sample_rate = 64000
        self.n_channels = n_channels
        self.sample_size = sample_size
        self.acquisition_maxsize = acquisition_maxsize
        self.rawfilesdir = "data_raw/raw_paderborn"
        self.config = config
        self.cache_filepath = f'cache/paderborn_{self.config}.npy'
        self.url = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
        self.n_folds = 2
        self.n_acquisitions = 2 if self.config == 'dbg' else 20
        self.list_of_bearings = get_list_of_bearings(self.n_acquisitions, self.config)
        self.signal = Signal('paderborn', self.cache_filepath)
        self.n_samples_acquisition = None

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
        

    def download(self, config='all'):
        """
        Download and extract compressed files from Paderborn website.
        """
        url = self.url
        dirname = self.rawfilesdir
        dir_rar = "rar_files"
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        if not os.path.isdir(os.path.join(dirname, dir_rar)):
            os.mkdir(os.path.join(dirname, dir_rar))
        print("Downloading and Extracting RAR files:")
        for bearing in eval(f'bearing_names_{config}()'):
            download_file(url, dirname, dir_rar, bearing)
            extract_rar(dirname, dir_rar, bearing)
        shutil.rmtree(os.path.join(dirname,dir_rar)) # remove the rar files 
        print("\nDataset Loaded.")


    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        for x, (key, filename_path) in enumerate(self.list_of_bearings):
            print('\r', f" Loading acquisitions {100*(x+1)/len(self.list_of_bearings):.2f} %", end='')
            matlab_file = scipy.io.loadmat(os.path.join(self.rawfilesdir, filename_path))
            bearing_label = os.path.splitext(os.path.split(filename_path)[-1])[0]
            vibration_data_raw = matlab_file[bearing_label]['Y'][0][0][0][6][2]
            if self.acquisition_maxsize:
                vibration_data = vibration_data_raw[0][:self.acquisition_maxsize]
            else:
                vibration_data = vibration_data_raw[0]
            self.n_samples_acquisition = len(vibration_data)//self.sample_size
            for i in range(self.n_samples_acquisition):
                sample = np.empty((self.sample_size, self.n_channels))
                for j in range(self.n_channels):
                    sample[:,j] = vibration_data[(i * self.sample_size):((i + 1) * self.sample_size)]
                sample = np.array([sample]).reshape(1, -1, self.n_channels)
                self.signal.add_acquisitions(key, sample)
        print(f"  ({np.size(self.signal.labels)} examples) | labels: {np.unique(self.signal.labels)}")


    def get_acquisitions(self):
        logging.info(self)
        if self.signal.check_is_cached():
            self.signal.load_cache(self.cache_filepath)
        else:
            os.path.exists(self.rawfilesdir) or self.download()
            self.load_acquisitions()
            self.signal.save_cache(self.cache_filepath)
        groups = self.groups()
        return self.signal, groups
    
                 
    def group_acquisition(self):
        logging.info(' Grouping the data by acquisition.')
        groups = []
        hash = dict()
        for i in self.signal.keys:
            if i not in hash:
                hash[i] = len(hash)
            groups = np.append(groups, hash[i])
        return groups
    
    def group_manufacturer(self):
        logging.info(' Grouping the data by manufacturer.')
        groups = []
        hash = dict()
        for key in self.signal.keys:
            manufacturer = manufacturers()[key.split('_')[1]]
            if manufacturer not in hash:    
                hash[manufacturer] = len(hash)
            groups = np.append(groups, hash[manufacturer])
        return groups

    def group_settings(self):
        logging.info(' Grouping the data by settings.')
        groups = []
        hash = dict()
        for key in self.signal.keys:
            setting = key[:11]
            if setting not in hash:
                hash[setting] = len(hash)
            groups = np.append(groups, hash[setting])
        return groups


    def group_bearings(self):
        logging.info(' Grouping the data by bearing.')
        groups = []
        hash = dict()
        for key in self.signal.keys:
            id = key.split('_')[1]
            if id not in hash:
                hash[id] = len(hash)
            groups = np.append(groups, hash[id])
        return groups


    def groups(self):
        return self.group_manufacturer()