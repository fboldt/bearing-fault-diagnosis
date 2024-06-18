"""
Class definition of CWRU Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
import urllib
import sys

# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)

def list_of_bearings_dbg():
    return [
        ("N.000.NN_0","97.mat"),        ("N.000.NN_1","98.mat"),        ("N.000.NN_2","99.mat"),        ("N.000.NN_3","100.mat"),
        ("I.007.DE_0","109.mat"),       ("I.007.DE_1","110.mat"),       ("I.007.DE_2","111.mat"),       ("I.007.DE_3","112.mat"),   
        ("B.007.DE_0","122.mat"),       ("B.007.DE_1","123.mat"),       ("B.007.DE_2","124.mat"),       ("B.007.DE_3","125.mat"),    
        ("O.007.DE.@6_0","135.mat"),    ("O.007.DE.@6_1","136.mat"),    ("O.007.DE.@6_2","137.mat"),    ("O.007.DE.@6_3","138.mat"),    
    ]

def list_of_bearings_all():
    return [
        ("N.000.NN_0","97.mat"),        ("N.000.NN_1","98.mat"),        ("N.000.NN_2","99.mat"),        ("N.000.NN_3","100.mat"),
        ("I.007.DE_0","109.mat"),       ("I.007.DE_1","110.mat"),       ("I.007.DE_2","111.mat"),       ("I.007.DE_3","112.mat"),
        ("B.007.DE_0","122.mat"),       ("B.007.DE_1","123.mat"),       ("B.007.DE_2","124.mat"),       ("B.007.DE_3","125.mat"),    
        ("O.007.DE.@6_0","135.mat"),    ("O.007.DE.@6_1","136.mat"),    ("O.007.DE.@6_2","137.mat"),    ("O.007.DE.@6_3","138.mat"),    
        ("O.007.DE.@3_0","148.mat"),    ("O.007.DE.@3_1","149.mat"),    ("O.007.DE.@3_2","150.mat"),    ("O.007.DE.@3_3","151.mat"),    
        ("O.007.DE.@12_0","161.mat"),   ("O.007.DE.@12_1","162.mat"),   ("O.007.DE.@12_2","163.mat"),   ("O.007.DE.@12_3","164.mat"),    
        ("I.014.DE_0","174.mat"),       ("I.014.DE_1","175.mat"),       ("I.014.DE_2","176.mat"),       ("I.014.DE_3","177.mat"),    
        ("B.014.DE_0","189.mat"),       ("B.014.DE_1","190.mat"),       ("B.014.DE_2","191.mat"),       ("B.014.DE_3","192.mat"),    
        ("O.014.DE.@6_0","201.mat"),    ("O.014.DE.@6_1","202.mat"),    ("O.014.DE.@6_2","203.mat"),    ("O.014.DE.@6_3","204.mat"),    
        ("I.021.DE_0","213.mat"),       ("I.021.DE_1","214.mat"),       ("I.021.DE_2","215.mat"),       ("I.021.DE_3","217.mat"),    
        ("B.021.DE_0","226.mat"),       ("B.021.DE_1","227.mat"),       ("B.021.DE_2","228.mat"),       ("B.021.DE_3","229.mat"),    
        ("O.021.DE.@6_0","238.mat"),    ("O.021.DE.@6_1","239.mat"),    ("O.021.DE.@6_2","240.mat"),    ("O.021.DE.@6_3","241.mat"),    
        ("O.021.DE.@3_0","250.mat"),    ("O.021.DE.@3_1","251.mat"),    ("O.021.DE.@3_2","252.mat"),    ("O.021.DE.@3_3","253.mat"),    
        ("O.021.DE.@12_0","262.mat"),   ("O.021.DE.@12_1","263.mat"),   ("O.021.DE.@12_2","264.mat"),   ("O.021.DE.@12_3","265.mat"),    
    ]

def list_of_bearings_balanced():
    return [
        ("N.000.NN_0","97.mat"), ("N.000.NN_1","98.mat"), ("N.000.NN_2","99.mat"), ("N.000.NN_3","100.mat"),
        ('I.007.DE_0', '109.mat'), ('I.007.DE_1', '110.mat'), ('I.007.DE_2', '111.mat'), ('I.007.DE_3', '112.mat'),
        ('B.007.DE_0', '122.mat'), ('B.007.DE_1', '123.mat'), ('B.007.DE_2', '124.mat'), ('B.007.DE_3', '125.mat'),
        ('O.007.DE.@6_0', '135.mat'), ('O.007.DE.@6_1', '136.mat'), ('O.007.DE.@6_2', '137.mat'), ('O.007.DE.@6_3', '138.mat'),
        ('I.014.DE_0', '174.mat'), ('I.014.DE_1', '175.mat'), ('I.014.DE_2', '176.mat'), ('I.014.DE_3', '177.mat'),
        ('B.014.DE_0', '189.mat'), ('B.014.DE_1', '190.mat'), ('B.014.DE_2', '191.mat'), ('B.014.DE_3', '192.mat'),
        ('O.014.DE.@6_0', '201.mat'), ('O.014.DE.@6_1', '202.mat'), ('O.014.DE.@6_2', '203.mat'), ('O.014.DE.@6_3', '204.mat'),
        ('I.021.DE_0', '213.mat'), ('I.021.DE_1', '214.mat'), ('I.021.DE_2', '215.mat'), ('I.021.DE_3', '217.mat'),
        ('B.021.DE_0', '226.mat'), ('B.021.DE_1', '227.mat'), ('B.021.DE_2', '228.mat'), ('B.021.DE_3', '229.mat'),
        ('O.021.DE.@6_0', '238.mat'), ('O.021.DE.@6_1', '239.mat'), ('O.021.DE.@6_2', '240.mat'), ('O.021.DE.@6_3', '241.mat')
    ]

def list_of_bearings_mert():
    return [
        ("N.000.NN_0","97.mat"),        ("N.000.NN_1","98.mat"),        ("N.000.NN_2","99.mat"),        ("N.000.NN_3","100.mat"),
        ("I.021.DE_0","213.mat"),       ("I.021.DE_1","214.mat"),       ("I.021.DE_2","215.mat"),       ("I.021.DE_3","217.mat"),    
        ("B.021.DE_0","226.mat"),       ("B.021.DE_1","227.mat"),       ("B.021.DE_2","228.mat"),       ("B.021.DE_3","229.mat"),    
        ("O.021.DE.@6_0","238.mat"),    ("O.021.DE.@6_1","239.mat"),    ("O.021.DE.@6_2","240.mat"),    ("O.021.DE.@6_3","241.mat"),    
    ]

def list_of_bearings_cmert():
    return list(set(list_of_bearings_all()) - set(list_of_bearings_mert()))

def download_file(url, dirname, bearing):
    print("Downloading Bearing Data:", bearing)
    file_name = bearing
    try:
        req = urllib.request.Request(url + file_name, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])

        dir_path = os.path.join(dirname, file_name)
        if not os.path.exists(dir_path):
            urllib.request.urlretrieve(url + file_name, dir_path)
            downloaded_file_size = os.stat(dir_path).st_size
        else:
            downloaded_file_size = os.stat(dir_path).st_size
        if file_size != downloaded_file_size:
            os.remove(dir_path)
            print("File Size Incorrect. Downloading Again.")
            download_file(url, dirname, bearing)
    except Exception as e:
        print("Error occurs when downloading file: " + str(e))
        print("Trying do download again")
        download_file(url, dirname, bearing)

class CWRU48k():
    """
    CWRU class wrapper for database download and acquisition.

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
      Download and extract raw files from CWRU website
    load_acquisitions()
      Extract vibration data from files
    """

    def get_cwru_bearings(self):
        list_of_bearings = eval("list_of_bearings_"+self.config+"()")
        bearing_label, bearing_file_names = zip(*list_of_bearings)
        return np.array(bearing_label), np.array(bearing_file_names)

    def __str__(self):
        return f"CWRU ({self.config})"

    def __init__(self, sample_size=4096, acquisition_maxsize=None, 
                 config="all", cache_file=None):
        self.sample_rate = 48_000
        self.sample_size = sample_size
        self.acquisition_maxsize = acquisition_maxsize
        self.config = config
        self.rawfilesdir = "raw_cwru"
        self.url = "https://engineering.case.edu/sites/default/files/"
        self.n_folds = 3
        self.bearing_labels, self.bearing_names = self.get_cwru_bearings()
        self.signal_data = np.empty((0, self.sample_size, 1))
        self.labels = []
        self.keys = []

        """
        Associate each file name to a bearing condition in a Python dictionary. 
        The dictionary keys identify the conditions.

        There are only four normal conditions, with loads of 0, 1, 2 and 3 hp. 
        All conditions end with an underscore character followed by an algarism 
        representing the load applied during the acquisitions. 
        The remaining conditions follow the pattern:
        
        First two characters represent the bearing location, 
        i.e. drive end (DE) and fan end (FE). 
        The following two characters represent the failure location in the bearing, 
        i.e. ball (BA), Inner Race (IR) and Outer Race (OR). 
        The next three algarisms indicate the severity of the failure, 
        where 007 stands for 0.007 inches and 0021 for 0.021 inches. 
        For Outer Race failures, the character @ is followed by a number 
        that indicates different load zones.
        """
        # Files Paths ordered by bearings
        files_path = {}
        for key, bearing in zip(self.bearing_labels, self.bearing_names):
            files_path[key] = os.path.join(self.rawfilesdir, bearing)
        self.files = files_path
        if cache_file is not None:
            self.load_cache(cache_file)

    def download(self):
        """
        Download and extract compressed files from CWRU website.
        """
        # Download MAT Files
        url = self.url
        dirname = self.rawfilesdir
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print("Downloading MAT files:")
        for bearing in self.bearing_names:
            download_file(url, dirname, bearing)
        print("Dataset Loaded.")

    def extract_acquisition(self, key, position):
        cwd = os.getcwd()
        matlab_file = scipy.io.loadmat(os.path.join(cwd, self.files[key]))
        acquisition = []
        file_number = self.files[key][len(self.rawfilesdir)+1:-4]
        signal_key = [key for key in matlab_file if key.endswith(file_number+ "_" + position + "_time")]            
        if len(signal_key) == 0:
            signal_key = [key for key in matlab_file if key.endswith("_" + position + "_time")]            
        if len(signal_key) > 0:
            if self.acquisition_maxsize:
                acquisition.append(matlab_file[signal_key[0]].reshape(1, -1)[0][:self.acquisition_maxsize])
            else:
                acquisition.append(matlab_file[signal_key[0]].reshape(1, -1)[0])
        acquisition = np.array(acquisition)       
        for i in range(acquisition.shape[1]//self.sample_size):
            sample = acquisition[:,(i * self.sample_size):((i + 1) * self.sample_size)]
            self.signal_data = np.append(self.signal_data, np.array([sample.T]), axis=0)
            self.labels = np.append(self.labels, key[0])
            self.keys = np.append(self.keys, key)
            
    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        for x, key in enumerate(self.files):
            print('\r', f" loading acquisitions {100*(x+1)/len(self.files):.2f} %", end='')
            position = key[6:8]
            if position == 'NN':
                for position in ['DE', 'FE']:
                    self.extract_acquisition(key, position)
            else:
                self.extract_acquisition(key, position)
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

    def group_load(self):    
        groups = []
        for i in self.keys:
            groups = np.append(groups, int(i[-1]) % self.n_folds)
        return groups

    def group_settings(self):
        groups = []
        hash = dict()
        for i in self.keys:
            load = i[-1]
            if load not in hash:
                hash[load] = len(hash)
            groups = np.append(groups, hash[load])
        return groups

    def group_severity(self):
        groups = []
        hash = dict()
        for i in self.keys:
            if i[0] == "N":
                load_severity = str(i[-1])
            else:
                load_severity = i[2:5]
            if load_severity not in hash:
                hash[load_severity] = len(hash)
            groups = np.append(groups, hash[load_severity])
        return groups
        
    def groups(self):
        return self.group_severity()

    def save_cache(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.signal_data)
            np.save(f, self.labels)
            np.save(f, self.keys)
            np.save(f, self.config)
    
    def load_cache(self, filename):
        with open(filename, 'rb') as f:
            self.signal_data = np.load(f)
            self.labels = np.load(f)
            self.keys = np.load(f)
            self.config = np.load(f)

if __name__ == "__main__":
    config = "balanced" # "dbg" # "all" # "balanced" # "cmert" # "mert"
    cache_name = f"cwru48k_{config}.npy"
    dataset = CWRU48k(config=config, acquisition_maxsize=None)
    os.path.exists("raw_cwru") or dataset.download()    
    if not os.path.exists(cache_name):
        dataset.load_acquisitions()  
        dataset.save_cache(cache_name)
    else: 
        dataset.load_cache(cache_name)
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    keys = list(set(dataset.keys))
    print("keys", np.array(keys), f"({len(keys)})")
    