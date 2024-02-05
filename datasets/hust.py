"""
Class definition of Hust Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedShuffleSplit, GroupShuffleSplit, StratifiedGroupKFold
import urllib
import sys
import requests
from tqdm import tqdm
import re
import zipfile36 as zipfile
# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)

list_of_bearings_dbg = []

list_of_bearings_normal = []


def download_file(url, dirname, bearing, progress_bar=None):

    path = os.path.join(dirname, bearing)

    if progress_bar is None:
        print("Downloading MAT files:")
        with requests.get(url, stream=True) as response, open(path, 'wb') as file:
            file_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=file_size)

            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

                try:
                    progress_bar.refresh()
                except KeyboardInterrupt:
                    print("Download stopped manually.")
                    progress_bar.close()
                    file.close()
                    os.remove(path)
                    raise

            progress_bar.close()
    else:
        print("A download is already in progress.")

    print(f'The file has been downloaded to the : {path}')


def extract_zip(zip_file_path, target_dir, pattern=r'([^/]+\.mat)$'):
    print("Extracting Bearings Data...")
    
    if not os.path.exists(zip_file_path):
        print(f"Zip file {zip_file_path} not found.")
        return
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    regex = re.compile(pattern)

    counter = 0
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        
        for file_info in zip_ref.infolist():
            filename = file_info.filename
            match = regex.search(filename)

            if match:
                matched_part = match.group()
                output_path = os.path.join(target_dir, matched_part)
                with open(output_path, 'wb') as output_file:
                    output_file.write(zip_ref.read(filename))
                    counter += 1        

    print(f'{counter} files were extracted into {target_dir} directory.')



class Hust():
    """
    Hust class wrapper for database download and acquisition.

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

    def get_hust_bearings(self):
        list_of_bearings = eval("list_of_bearings_"+self.config)
        bearing_label, bearing_file_names = zip(*list_of_bearings)
        return np.array(bearing_label), np.array(bearing_file_names)

    def __init__(self, sample_size=8400, n_channels=1, acquisition_maxsize=420_000, config="dbg"):
        self.sample_size = sample_size
        self.n_channels = n_channels
        self.acquisition_maxsize = acquisition_maxsize
        self.config = config
        self.rawfilesdir = "raw_hust"
        self.url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/cbv7jyx4p9-2.zip"
        self.n_folds = 4
        # self.bearing_labels, self.bearing_names = self.get_hust_bearings()
        self.accelerometers = ['DE'][:self.n_channels]
        self.signal_data = np.empty((0, self.sample_size, len(self.accelerometers)))
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
        # # Files Paths ordered by bearings
        # files_path = {}
        # for key, bearing in zip(self.bearing_labels, self.bearing_names):
        #     files_path[key] = os.path.join(self.rawfilesdir, bearing)
        # self.files = files_path

    def download(self):
        """
        Download and extract compressed files from CWRU website.
        """
        # Download MAT Files
        url = self.url
        dirname = self.rawfilesdir

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
                
        download_file(url, dirname, "hust_bearing.zip")
        extract_zip(os.path.join(dirname, "hust_bearing.zip"), dirname)


Hust().download()

#     def load_acquisitions(self):
#         """
#         Extracts the acquisitions of each file in the dictionary files_names.
#         """
#         cwd = os.getcwd()
#         for x, key in enumerate(self.files):
#             matlab_file = scipy.io.loadmat(os.path.join(cwd, self.files[key]))
#             acquisition = []
#             print('\r', f" loading acquisitions {100*(x+1)/len(self.files):.2f} %", end='')
#             for position in self.accelerometers:
#                 file_number = self.files[key][len(self.rawfilesdir)+1:-4]
#                 signal_key = [key for key in matlab_file if key.endswith(file_number+ "_" + position + "_time")]
#                 if len(signal_key) == 0:
#                     signal_key = [key for key in matlab_file if key.endswith("_" + position + "_time")]
#                 if len(signal_key) > 0:
#                     # print(f"  {key}: {matlab_file[signal_key[0]].reshape(1, -1)[0].shape}")
#                     acquisition.append(matlab_file[signal_key[0]].reshape(1, -1)[0][:self.acquisition_maxsize])
#             acquisition = np.array(acquisition)
#             if len(acquisition.shape)<2 or acquisition.shape[0]<self.n_channels:
#                 continue
#             for i in range(acquisition.shape[1]//self.sample_size):
#                 sample = acquisition[:,(i * self.sample_size):((i + 1) * self.sample_size)]
#                 self.signal_data = np.append(self.signal_data, np.array([sample.T]), axis=0)
#                 self.labels = np.append(self.labels, key[0])
#                 self.keys = np.append(self.keys, key)
#         print(f"  ({len(self.labels)} examples)")
    
#     def get_acquisitions(self):
#         if len(self.labels) == 0:
#             self.load_acquisitions()
#         return self.signal_data, self.labels

#     def kfold(self):
#         if len(self.signal_data) == 0:
#             self.load_acquisitions()
#         kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
#         for train, test in kf.split(self.signal_data):
#             yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

#     def stratifiedkfold(self):
#         if len(self.signal_data) == 0:
#             self.load_acquisitions()
#         kf = StratifiedShuffleSplit(n_splits=self.n_folds, random_state=42)
#         for train, test in kf.split(self.signal_data, self.labels):
#             yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

#     def groupkfold_acquisition(self):
#         if len(self.signal_data) == 0:
#             self.load_acquisitions()
#         groups = []
#         for i in self.keys:
#             groups = np.append(groups, i)
#         kf = StratifiedGroupKFold(n_splits=self.n_folds)
#         for train, test in kf.split(self.signal_data, self.labels, groups):
#             yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

#     def groupkfold_load(self):
#         if len(self.signal_data) == 0:
#             self.load_acquisitions()
#         groups = []
#         for i in self.keys:
#             groups = np.append(groups, int(i[-1]) % self.n_folds)
#         kf = GroupShuffleSplit(n_splits=self.n_folds)
#         for train, test in kf.split(self.signal_data, self.labels, groups):
#             yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

#     def groupkfold_settings(self):
#         if len(self.signal_data) == 0:
#             self.load_acquisitions()
#         groups = []
#         for i in self.keys:
#             load = i[-1]
#             groups = np.append(groups, load)
#         kf = GroupShuffleSplit(n_splits=self.n_folds)
#         for train, test in kf.split(self.signal_data, self.labels, groups):
#             yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

#     def groupkfold_severity(self):
#         if len(self.signal_data) == 0:
#             self.load_acquisitions()
#         groups = []
#         for i in self.keys:
#             if i[0] == "N":
#                 load_severity = str(i[-1])
#             else:
#                 load_severity = i[2:5]
#             groups = np.append(groups, load_severity)
#         kf = GroupShuffleSplit(n_splits=self.n_folds)
#         for train, test in kf.split(self.signal_data, self.labels, groups):
#             yield self.signal_data[train], self.labels[train], self.signal_data[test], self.labels[test]

# if __name__ == "__main__":
#     dataset = CWRU(config='nio')
#     # dataset.download()
#     dataset.load_acquisitions()
#     print("Signal datase shape", dataset.signal_data.shape)
#     labels = list(set(dataset.labels))
#     print("labels", labels, f"({len(labels)})")
#     keys = list(set(dataset.keys))
#     print("keys", np.array(keys), f"({len(keys)})")
    