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

# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)

list_of_bearings_dbg = [
    ("H_1_0", "H_1_0.mat"), ("H_2_0", "H_2_0.mat"), ("H_3_0", "H_3_0.mat"), ("H_4_0", "H_4_0.mat"),
    ("I_1_1", "I_1_1.mat"), ("I_2_2", "I_2_2.mat"), ("I_3_1", "I_3_1.mat"), ("I_4_2", "I_4_2.mat"),
    ("O_6_1", "O_6_1.mat"), ("O_7_2", "O_7_2.mat"), ("O_8_1", "O_8_1.mat"), ("O_9_2", "O_9_2.mat"),
    ("B_11_1", "B_11_1.mat"), ("B_12_2", "B_12_2.mat"), ("B_13_1", "B_13_1.mat"), ("B_14_2", "B_14_2.mat"),
    ("C_16_1", "C_16_1.mat"), ("C_17_2", "C_17_2.mat"), ("C_18_1", "C_18_1.mat"), ("C_19_2", "C_19_2.mat")
]



import csv

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


def download(dirname):
    dir_path_urls = os.path.join("datasets", "uored_vafcls_urls.csv")
    urls = read_csv(dir_path_urls)

    for label, url in urls.items():
        download_file(url, dirname, bearing=label+".mat")   


download("raw_uored_vafcls")
