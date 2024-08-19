"""
Class definition of CWRU Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
import urllib
import sys
import logging
import re

from datasets.signal_data import Signal
from utils.acquisition_handler import split_acquisition

# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)

def list_of_bearings_dbg():
    return [
        ("N.000.NN_0&12000","97.mat"),        ("N.000.NN_1&12000","98.mat"),        ("N.000.NN_2&12000","99.mat"),        ("N.000.NN_3&12000","100.mat"),
        ("I.007.DE_0&48000","109.mat"),       ("I.007.DE_1&48000","110.mat"),       ("I.007.DE_2&48000","111.mat"),       ("I.007.DE_3&48000","112.mat"),   
        ("B.007.DE_0&48000","122.mat"),       ("B.007.DE_1&48000","123.mat"),       ("B.007.DE_2&48000","124.mat"),       ("B.007.DE_3&48000","125.mat"),    
        ("O.007.DE.@6_0&48000","135.mat"),    ("O.007.DE.@6_1&48000","136.mat"),    ("O.007.DE.@6_2&48000","137.mat"),    ("O.007.DE.@6_3&48000","138.mat"),    
    ]

def list_of_bearings_48k():
    return [
        ("I.007.DE_0&48000","109.mat"),       ("I.007.DE_1&48000","110.mat"),       ("I.007.DE_2&48000","111.mat"),       ("I.007.DE_3&48000","112.mat"),
        ("B.007.DE_0&48000","122.mat"),       ("B.007.DE_1&48000","123.mat"),       ("B.007.DE_2&48000","124.mat"),       ("B.007.DE_3&48000","125.mat"),    
        ("O.007.DE.@6_0&48000","135.mat"),    ("O.007.DE.@6_1&48000","136.mat"),    ("O.007.DE.@6_2&48000","137.mat"),    ("O.007.DE.@6_3&48000","138.mat"),    
        ("O.007.DE.@3_0&48000","148.mat"),    ("O.007.DE.@3_1&48000","149.mat"),    ("O.007.DE.@3_2&48000","150.mat"),    ("O.007.DE.@3_3&48000","151.mat"),    
        ("O.007.DE.@12_0&48000","161.mat"),   ("O.007.DE.@12_1&48000","162.mat"),   ("O.007.DE.@12_2&48000","163.mat"),   ("O.007.DE.@12_3&48000","164.mat"),    
        ("I.014.DE_0&48000","174.mat"),       ("I.014.DE_1&48000","175.mat"),       ("I.014.DE_2&48000","176.mat"),       ("I.014.DE_3&48000","177.mat"),    
        ("B.014.DE_0&48000","189.mat"),       ("B.014.DE_1&48000","190.mat"),       ("B.014.DE_2&48000","191.mat"),       ("B.014.DE_3&48000","192.mat"),    
        ("O.014.DE.@6_0&48000","201.mat"),    ("O.014.DE.@6_1&48000","202.mat"),    ("O.014.DE.@6_2&48000","203.mat"),    ("O.014.DE.@6_3&48000","204.mat"),    
        ("I.021.DE_0&48000","213.mat"),       ("I.021.DE_1&48000","214.mat"),       ("I.021.DE_2&48000","215.mat"),       ("I.021.DE_3&48000","217.mat"),    
        ("B.021.DE_0&48000","226.mat"),       ("B.021.DE_1&48000","227.mat"),       ("B.021.DE_2&48000","228.mat"),       ("B.021.DE_3&48000","229.mat"),    
        ("O.021.DE.@6_0&48000","238.mat"),    ("O.021.DE.@6_1&48000","239.mat"),    ("O.021.DE.@6_2&48000","240.mat"),    ("O.021.DE.@6_3&48000","241.mat"),    
        ("O.021.DE.@3_0&48000","250.mat"),    ("O.021.DE.@3_1&48000","251.mat"),    ("O.021.DE.@3_2&48000","252.mat"),    ("O.021.DE.@3_3&48000","253.mat"),    
        ("O.021.DE.@12_0&48000","262.mat"),   ("O.021.DE.@12_1&48000","263.mat"),   ("O.021.DE.@12_2&48000","264.mat"),   ("O.021.DE.@12_3&48000","265.mat"),    
    ]

def list_of_bearings_12k():
    return [
    ("N.000.NN_0&12000","97.mat"),        ("N.000.NN_1&12000","98.mat"),        ("N.000.NN_2&12000","99.mat"),        ("N.000.NN_3&12000","100.mat"),
    ("I.007.DE_0&12000","105.mat"),       ("I.007.DE_1&12000","106.mat"),       ("I.007.DE_2&12000","107.mat"),       ("I.007.DE_3&12000","108.mat"),
    ("B.007.DE_0&12000","118.mat"),       ("B.007.DE_1&12000","119.mat"),       ("B.007.DE_2&12000","120.mat"),       ("B.007.DE_3&12000","121.mat"),    
    ("O.007.DE.@6_0&12000","130.mat"),    ("O.007.DE.@6_1&12000","131.mat"),    ("O.007.DE.@6_2&12000","132.mat"),    ("O.007.DE.@6_3&12000","133.mat"),    
    ("O.007.DE.@3_0&12000","144.mat"),    ("O.007.DE.@3_1&12000","145.mat"),    ("O.007.DE.@3_2&12000","146.mat"),    ("O.007.DE.@3_3&12000","147.mat"),    
    ("O.007.DE.@12_0&12000","156.mat"),   ("O.007.DE.@12_1&12000","158.mat"),   ("O.007.DE.@12_2&12000","159.mat"),   ("O.007.DE.@12_3&12000","160.mat"),    
    ("I.014.DE_0&12000","169.mat"),       ("I.014.DE_1&12000","170.mat"),       ("I.014.DE_2&12000","171.mat"),       ("I.014.DE_3&12000","172.mat"),    
    ("B.014.DE_0&12000","185.mat"),       ("B.014.DE_1&12000","186.mat"),       ("B.014.DE_2&12000","187.mat"),       ("B.014.DE_3&12000","188.mat"),    
    ("O.014.DE.@6_0&12000","197.mat"),    ("O.014.DE.@6_1&12000","198.mat"),    ("O.014.DE.@6_2&12000","199.mat"),    ("O.014.DE.@6_3&12000","200.mat"),    
    ("I.021.DE_0&12000","209.mat"),       ("I.021.DE_1&12000","210.mat"),       ("I.021.DE_2&12000","211.mat"),       ("I.021.DE_3&12000","212.mat"),    
    ("B.021.DE_0&12000","222.mat"),       ("B.021.DE_1&12000","223.mat"),       ("B.021.DE_2&12000","224.mat"),       ("B.021.DE_3&12000","225.mat"),    
    ("O.021.DE.@6_0&12000","234.mat"),    ("O.021.DE.@6_1&12000","235.mat"),    ("O.021.DE.@6_2&12000","236.mat"),    ("O.021.DE.@6_3&12000","237.mat"),    
    ("O.021.DE.@3_0&12000","246.mat"),    ("O.021.DE.@3_1&12000","247.mat"),    ("O.021.DE.@3_2&12000","248.mat"),    ("O.021.DE.@3_3&12000","249.mat"),    
    ("O.021.DE.@12_0&12000","258.mat"),   ("O.021.DE.@12_1&12000","259.mat"),   ("O.021.DE.@12_2&12000","260.mat"),   ("O.021.DE.@12_3&12000","261.mat"),    
    ("I.028.DE_0&12000","3001.mat"),      ("I.028.DE_1&12000","3002.mat"),      ("I.028.DE_2&12000","3003.mat"),      ("I.028.DE_3&12000","3004.mat"),    
    ("B.028.DE_0&12000","3005.mat"),      ("B.028.DE_1&12000","3006.mat"),      ("B.028.DE_2&12000","3007.mat"),      ("B.028.DE_3&12000","3008.mat"),
    ("I.007.FE_0&12000","278.mat"),       ("I.007.FE_1&12000","279.mat"),       ("I.007.FE_2&12000","280.mat"),       ("I.007.FE_3&12000","281.mat"),    
    ("B.007.FE_0&12000","282.mat"),       ("B.007.FE_1&12000","283.mat"),       ("B.007.FE_2&12000","284.mat"),       ("B.007.FE_3&12000","285.mat"),    
    ("O.007.FE.@6_0&12000","294.mat"),    ("O.007.FE.@6_1&12000","295.mat"),    ("O.007.FE.@6_2&12000","296.mat"),    ("O.007.FE.@6_3&12000","297.mat"),    
    ("O.007.FE.@3_0&12000","298.mat"),    ("O.007.FE.@3_1&12000","299.mat"),    ("O.007.FE.@3_2&12000","300.mat"),    ("O.007.FE.@3_3&12000","301.mat"),    
    ("O.007.FE.@12_0&12000","302.mat"),   ("O.007.FE.@12_1&12000","305.mat"),   ("O.007.FE.@12_2&12000","306.mat"),   ("O.007.FE.@12_3&12000","307.mat"),    
    ("I.014.FE_0&12000","274.mat"),       ("I.014.FE_1&12000","275.mat"),       ("I.014.FE_2&12000","276.mat"),       ("I.014.FE_3&12000","277.mat"),    
    ("B.014.FE_0&12000","286.mat"),       ("B.014.FE_1&12000","287.mat"),       ("B.014.FE_2&12000","288.mat"),       ("B.014.FE_3&12000","289.mat"),    
    ("O.014.FE.@3_0&12000","310.mat"),    ("O.014.FE.@3_1&12000","309.mat"),    ("O.014.FE.@3_2&12000","311.mat"),    ("O.014.FE.@3_3&12000","312.mat"),    
    ("O.014.FE.@6_0&12000","313.mat"),    
    ("I.021.FE_0&12000","270.mat"),       ("I.021.FE_1&12000","271.mat"),       ("I.021.FE_2&12000","272.mat"),       ("I.021.FE_3&12000","273.mat"),    
    ("B.021.FE_0&12000","290.mat"),       ("B.021.FE_1&12000","291.mat"),       ("B.021.FE_2&12000","292.mat"),       ("B.021.FE_3&12000","293.mat"),    
    ("O.021.FE.@6_0&12000","315.mat"),    ("O.021.FE.@3_1&12000","316.mat"),    ("O.021.FE.@3_2&12000","317.mat"),    ("O.021.FE.@3_3&12000","318.mat"),    
]

def list_of_bearings_DE():
    return [
        ("N.000.NN_0&12000","97.mat"),        ("N.000.NN_1&12000","98.mat"),        ("N.000.NN_2&12000","99.mat"),        ("N.000.NN_3&12000","100.mat"),
        ("I.007.DE_0&12000","105.mat"),       ("I.007.DE_1&12000","106.mat"),       ("I.007.DE_2&12000","107.mat"),       ("I.007.DE_3&12000","108.mat"),
        ("B.007.DE_0&12000","118.mat"),       ("B.007.DE_1&12000","119.mat"),       ("B.007.DE_2&12000","120.mat"),       ("B.007.DE_3&12000","121.mat"),    
        ("O.007.DE.@6_0&12000","130.mat"),    ("O.007.DE.@6_1&12000","131.mat"),    ("O.007.DE.@6_2&12000","132.mat"),    ("O.007.DE.@6_3&12000","133.mat"),    
        ("O.007.DE.@3_0&12000","144.mat"),    ("O.007.DE.@3_1&12000","145.mat"),    ("O.007.DE.@3_2&12000","146.mat"),    ("O.007.DE.@3_3&12000","147.mat"),    
        ("O.007.DE.@12_0&12000","156.mat"),   ("O.007.DE.@12_1&12000","158.mat"),   ("O.007.DE.@12_2&12000","159.mat"),   ("O.007.DE.@12_3&12000","160.mat"),
        ("I.014.DE_0&12000","169.mat"),       ("I.014.DE_1&12000","170.mat"),       ("I.014.DE_2&12000","171.mat"),       ("I.014.DE_3&12000","172.mat"),    
        ("B.014.DE_0&12000","185.mat"),       ("B.014.DE_1&12000","186.mat"),       ("B.014.DE_2&12000","187.mat"),       ("B.014.DE_3&12000","188.mat"),
        ("O.014.DE.@6_0&12000","197.mat"),    ("O.021.DE.@6_0&12000","234.mat"),    ("O.021.DE.@3_1&12000","247.mat"),    ("O.021.DE.@3_2&12000","248.mat"),    
        ("O.021.DE.@3_3&12000","249.mat"),
    ]

def list_of_bearings_FE():
    return [
        ("N.000.NN_0&12000","97.mat"),        ("N.000.NN_1&12000","98.mat"),        ("N.000.NN_2&12000","99.mat"),        ("N.000.NN_3&12000","100.mat"),
        ("I.007.FE_0&12000","278.mat"),       ("I.007.FE_1&12000","279.mat"),       ("I.007.FE_2&12000","280.mat"),       ("I.007.FE_3&12000","281.mat"),    
        ("B.007.FE_0&12000","282.mat"),       ("B.007.FE_1&12000","283.mat"),       ("B.007.FE_2&12000","284.mat"),       ("B.007.FE_3&12000","285.mat"),    
        ("O.007.FE.@6_0&12000","294.mat"),    ("O.007.FE.@6_1&12000","295.mat"),    ("O.007.FE.@6_2&12000","296.mat"),    ("O.007.FE.@6_3&12000","297.mat"),    
        ("O.007.FE.@3_0&12000","298.mat"),    ("O.007.FE.@3_1&12000","299.mat"),    ("O.007.FE.@3_2&12000","300.mat"),    ("O.007.FE.@3_3&12000","301.mat"),    
        ("O.007.FE.@12_0&12000","302.mat"),   ("O.007.FE.@12_1&12000","305.mat"),   ("O.007.FE.@12_2&12000","306.mat"),   ("O.007.FE.@12_3&12000","307.mat"), 
        ("I.014.FE_0&12000","274.mat"),       ("I.014.FE_1&12000","275.mat"),       ("I.014.FE_2&12000","276.mat"),       ("I.014.FE_3&12000","277.mat"),    
        ("B.014.FE_0&12000","286.mat"),       ("B.014.FE_1&12000","287.mat"),       ("B.014.FE_2&12000","288.mat"),       ("B.014.FE_3&12000","289.mat"),
        ("O.014.FE.@3_0&12000","310.mat"),    ("O.021.FE.@6_0&12000","315.mat"),    ("O.021.FE.@3_1&12000","316.mat"),    ("O.021.FE.@3_2&12000","317.mat"),    
        ("O.021.FE.@3_3&12000","318.mat"),    
    ]

def list_of_bearings_FEDE():
    return [
        ("N.000.NN_0&12000","97.mat"),        ("N.000.NN_1&12000","98.mat"),        ("N.000.NN_2&12000","99.mat"),        ("N.000.NN_3&12000","100.mat"),
        ("I.007.DE_0&12000","105.mat"),       ("I.007.DE_1&12000","106.mat"),       ("I.007.DE_2&12000","107.mat"),       ("I.007.DE_3&12000","108.mat"),
        ("I.007.FE_0&12000","278.mat"),       ("I.007.FE_1&12000","279.mat"),       ("I.007.FE_2&12000","280.mat"),       ("I.007.FE_3&12000","281.mat"),    
        ("B.007.DE_0&12000","118.mat"),       ("B.007.DE_1&12000","119.mat"),       ("B.007.DE_2&12000","120.mat"),       ("B.007.DE_3&12000","121.mat"),
        ("B.007.FE_0&12000","282.mat"),       ("B.007.FE_1&12000","283.mat"),       ("B.007.FE_2&12000","284.mat"),       ("B.007.FE_3&12000","285.mat"),    
        ("O.007.DE.@6_0&12000","130.mat"),    ("O.007.DE.@6_1&12000","131.mat"),    ("O.007.DE.@6_2&12000","132.mat"),    ("O.007.DE.@6_3&12000","133.mat"),
        ("O.007.FE.@6_0&12000","294.mat"),    ("O.007.FE.@6_1&12000","295.mat"),    ("O.007.FE.@6_2&12000","296.mat"),    ("O.007.FE.@6_3&12000","297.mat"),    
        ("O.007.DE.@3_0&12000","144.mat"),    ("O.007.DE.@3_1&12000","145.mat"),    ("O.007.DE.@3_2&12000","146.mat"),    ("O.007.DE.@3_3&12000","147.mat"),
        ("O.007.FE.@3_0&12000","298.mat"),    ("O.007.FE.@3_1&12000","299.mat"),    ("O.007.FE.@3_2&12000","300.mat"),    ("O.007.FE.@3_3&12000","301.mat"),    
        ("O.007.DE.@12_0&12000","156.mat"),   ("O.007.DE.@12_1&12000","158.mat"),   ("O.007.DE.@12_2&12000","159.mat"),   ("O.007.DE.@12_3&12000","160.mat"),
        ("O.007.FE.@12_0&12000","302.mat"),   ("O.007.FE.@12_1&12000","305.mat"),   ("O.007.FE.@12_2&12000","306.mat"),   ("O.007.FE.@12_3&12000","307.mat"), 
        ("I.014.DE_0&12000","169.mat"),       ("I.014.DE_1&12000","170.mat"),       ("I.014.DE_2&12000","171.mat"),       ("I.014.DE_3&12000","172.mat"),    
        ("I.014.FE_0&12000","274.mat"),       ("I.014.FE_1&12000","275.mat"),       ("I.014.FE_2&12000","276.mat"),       ("I.014.FE_3&12000","277.mat"),
        ("B.014.DE_0&12000","185.mat"),       ("B.014.DE_1&12000","186.mat"),       ("B.014.DE_2&12000","187.mat"),       ("B.014.DE_3&12000","188.mat"),
        ("B.014.FE_0&12000","286.mat"),       ("B.014.FE_1&12000","287.mat"),       ("B.014.FE_2&12000","288.mat"),       ("B.014.FE_3&12000","289.mat"),
        ("O.014.DE.@6_0&12000","197.mat"),    ("O.021.DE.@6_0&12000","234.mat"),    ("O.021.DE.@3_1&12000","247.mat"),    ("O.021.DE.@3_2&12000","248.mat"),    
        ("O.014.FE.@3_0&12000","310.mat"),    ("O.021.FE.@6_0&12000","315.mat"),    ("O.021.FE.@3_1&12000","316.mat"),    ("O.021.FE.@3_2&12000","317.mat"),    
        ("O.021.DE.@3_3&12000","249.mat"),   ("O.021.FE.@3_3&12000","318.mat"),
    ]


def list_of_bearings_all():
    return list_of_bearings_48k() + list_of_bearings_12k()


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


class CWRU():
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
    def __str__(self):        
        return f"CWRU ({self.config})"

    def __init__(self, sample_size=4096, acquisition_maxsize=None,
                 config="12k"):
        self.config = config
        self.cache_filepath = f'cache/cwru_{self.config}.npy'
        self.sample_size = sample_size
        self.acquisition_maxsize = acquisition_maxsize
        self.rawfilesdir = "raw_cwru"
        self.url = "https://engineering.case.edu/sites/default/files/"
        self.n_folds = 3
        self.signal = Signal('CWRU', self.cache_filepath)

        

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

    def download(self):
        """
        Download and extract compressed files from CWRU website.
        """
        url = self.url
        dirname = self.rawfilesdir
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print("Downloading MAT files:")
        bearings = eval(f"list_of_bearings_{self.config}()")
        filenames = [b[1] for b in bearings]
        for file in filenames:
            download_file(url, dirname, file)
        print("Dataset Loaded.")

    def extract_acquisition(self, bearing_label, bearing_file):     
        full_path = os.path.join(f'{self.rawfilesdir}/{bearing_file}')
        matlab_file = scipy.io.loadmat(full_path)
        keys = re.findall(r'X\d{3}_[A-Z]{2}_time', str(matlab_file.keys()))
        for key in keys:
            if self.acquisition_maxsize:
                data = matlab_file[key].reshape(1, -1)[:, :self.acquisition_maxsize]
            else:
                data = matlab_file[key].reshape(1, -1)
            acquisitions = split_acquisition(data, self.sample_size)
            self.signal.add_acquisitions(bearing_label, acquisitions)
                    
    def load_acquisitions(self):
        os.path.exists(self.rawfilesdir) or self.download()
        list_of_bearings = eval(f"list_of_bearings_{self.config}()")
        for x, (bearing_label, bearing_file) in enumerate(list_of_bearings):
            print('\r', f" Loading acquisitions {100*(x+1)/len(list_of_bearings):.2f} %", end='')
            self.extract_acquisition(bearing_label, bearing_file)
        print(f"  ({np.size(self.signal.labels)} examples) | labels: {np.unique(self.signal.labels)}")
    
    def get_acquisitions(self):
        logging.info(self) # show name of dataset
        if self.signal.check_is_cached():
            self.signal.load_cache(self.cache_filepath)
        else:
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

    def group_load(self):    
        logging.info(' Grouping the data by load.')
        groups = []
        for i in self.signal.keys:
            groups = np.append(groups, int(i[-7]) % self.n_folds)
        return groups

    def group_settings(self):
        logging.info(' Grouping the data by settings.')
        groups = []
        hash = dict()
        for i in self.signal.keys:
            load = i[-7]
            if load not in hash:
                hash[load] = len(hash)
            groups = np.append(groups, hash[load])
        return groups

    def group_severity(self):
        logging.info(' Grouping the data by severity.')
        groups = []
        hash = dict()
        for i in self.signal.keys:
            if i[0] == "N":
                load_severity = str(i[-7])
            else:
                load_severity = i[2:5]
            if load_severity not in hash:
                hash[load_severity] = len(hash)
            groups = np.append(groups, hash[load_severity])
        return groups
    
    def group_sensor_position(self):
        logging.info(' Grouping the data by accelerometer position: FE, DE, and BA.')
        self.n_folds = 2
        groups = []
        hash = dict()
        for key in self.signal.acquisition_keys:
            acc_position = key[-7:-5]
            if acc_position not in hash:
                hash[acc_position] = len(hash)
            groups = np.append(groups, hash[acc_position])
        return groups
        
    def groups(self):
        return self.group_severity()    
