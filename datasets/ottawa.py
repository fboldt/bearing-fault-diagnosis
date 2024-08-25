"""
Class definition of Ottawa Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import scipy.signal
import numpy as np
import os
import logging

from datasets.signal_data import Signal
from utils.acquisition_handler import split_acquisition


def files_hash():
    return {
    "H-A-1": "3d666929-39e2-4e91-9028-5df96cb4f0cc",
    "H-A-2": "2d502b39-e863-46ff-b466-927c06d5deb6",
    "H-A-3": "66420867-38e1-47b6-9db0-6126f89f584b",
    "H-B-1": "d4cd7559-2ede-470f-a3da-58fd8f72f176",
    "H-B-2": "4559f9a0-ba9e-4e5b-b0d5-e56fed968405",
    "H-B-3": "1cec2af9-a4ed-4f1c-8afe-7f6b39800b3e",
    "H-C-1": "1aee60b5-98fb-435b-a594-6645c993c396",
    "H-C-2": "ab80616b-c77d-400b-8e6b-8f5023e00b04",
    "H-C-3": "71cac4ca-4fa9-4143-bbec-5e8293deddfb",
    "H-D-1": "0d69c81c-3458-4ac5-8351-2dd5598bebc6",
    "H-D-2": "78389e5e-ff1c-45d7-9d99-49c405d8b432",
    "H-D-3": "8bedfdfe-04e6-414e-a529-cf7a5a2335b1",
    "I-A-1": "88403a69-f7fe-4811-9b48-50862cd98254",
    "I-A-2": "6fc987d1-78bb-4453-872f-418c11883b38",
    "I-A-3": "c968ff9a-8aa8-4fb9-9a83-4f5bc614ee65",
    "I-B-1": "635b5080-56ee-4baf-aa66-c4afab60e86d",
    "I-B-2": "bd215a9b-65cc-40f7-870a-2e224a1ddf9c",
    "I-B-3": "b8d3d029-5cd3-43de-8d7b-5c7cdaf82536",
    "I-C-1": "c6416b77-d978-40d3-b027-ed51c35a45ae",
    "I-C-2": "52003e3e-64ee-4df4-a54a-0313ef8e0b8f",
    "I-C-3": "547cf379-4797-48b0-bb5b-92cc33e0a011",
    "I-D-1": "c4e2f020-0cc9-48df-8eca-54a3589ff26e",
    "I-D-2": "92c6e6fc-2ffa-4c0d-ae8c-dbc1ae8204d5",
    "I-D-3": "d7d9dc77-6bfc-4159-8bd8-0a76a357f5d1",
    "O-A-1": "0d680ac1-81a9-4f12-a4e0-c8a55877fbf2",
    "O-A-2": "d0aa841a-0fd0-414f-9a5e-fb9f043ff631",
    "O-A-3": "917d4e7b-c2b6-48cf-a201-79e1f7fdab8f",
    "O-B-1": "45a6369f-d759-45be-a8dd-4a334a9914c6",
    "O-B-2": "9f009c49-dc7d-4988-96f8-cbc59d948a43",
    "O-B-3": "0952c95b-ed49-46f1-9075-097e01ade99b",
    "O-C-1": "cf4f2263-93d4-4aaf-af5a-0cd1093f508d",
    "O-C-2": "93cb5c8a-a522-4732-8589-1e7028cba72a",
    "O-C-3": "139b8073-12ef-43db-ac81-ed484946ef7f",
    "O-D-1": "14493825-0e5b-444d-a435-c9ff9b99ce13",
    "O-D-2": "27a2cef4-63ed-40c2-9a1e-81c26ee7396c",
    "O-D-3": "a112c222-3318-4735-869b-ecdf9eafa7f8",
    "B-A-1": "c161f6c5-4cfb-4a89-8d3f-6b5c4adf74b9",
    "B-A-2": "3d74e6b1-75aa-4f42-808d-a0ced61be48f",
    "B-A-3": "4b1ce977-6d04-444b-977c-68362505b9a7",
    "B-B-1": "053467ce-5bf6-43a3-bc74-b62b988c06f7",
    "B-B-2": "d8451dbc-3e3a-4b91-905e-7aaef2c6482f",
    "B-B-3": "0e7bb330-b7e8-40e6-a5f5-633dbc836301",
    "B-C-1": "1ca826f9-66c7-40f7-bf89-b5519e2bdb9b",
    "B-C-2": "637793c3-1005-40f8-8a3f-b3f7e58268c8",
    "B-C-3": "632343c9-9aa6-489c-9b61-aad864b53c48",
    "B-D-1": "b24c6f21-34d1-49e9-b81a-8f15e4862c37",
    "B-D-2": "da64bb45-681a-400a-bd2d-57851e8981d8",
    "B-D-3": "a22bcbc1-6449-4272-a163-aafe04bc0a80",
    "C-A-1": "e5542644-029b-498d-a474-6eeac0b414b2",
    "C-A-2": "9ce4ce7f-6178-4ee0-a30d-88dad4cc41cc",
    "C-A-3": "e35676b4-710e-4fe8-995a-360640a17cf7",
    "C-B-1": "e2202cbc-4203-4e54-a925-0fcf2f1ed29d",
    "C-B-2": "2bb383ed-edf5-4b03-96ef-63b56c1ea3da",
    "C-B-3": "6bd23e2a-0ab8-49e4-ba35-ce90a6a2832c",
    "C-C-1": "6730dad0-c10c-4dd3-ab51-f0826092106b",
    "C-C-2": "6cc31603-0502-4a3f-ba5c-8652794cfac3",
    "C-C-3": "b0e296f3-42f8-45e0-99e7-51e4ab5b2e58",
    "C-D-1": "ecc5c18d-2b53-4589-8226-ab7882ffa9b8",
    "C-D-2": "9ee26954-aba1-45df-ba4b-6220fa49835f",
    "C-D-3": "675e085a-f6c9-44a0-81b2-20a939f0f6e8",
}


def list_of_bearings_dbg():
    return [
        ("H-A-1&200000","H-A-1.mat"), ("H-B-1&200000", "H-B-1.mat"), ("H-C-1&200000", "H-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("I-A-1&200000","I-A-1.mat"), ("I-B-1&200000", "I-B-1.mat"), ("I-C-1&200000", "I-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("O-A-1&200000","O-A-1.mat"), ("O-B-1&200000", "O-B-1.mat"), ("O-C-1&200000", "O-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("B-A-1&200000","B-A-1.mat"), ("B-B-1&200000", "B-B-1.mat"), ("B-C-1&200000", "B-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("C-A-1&200000","C-A-1.mat"), ("C-B-1&200000", "C-B-1.mat"), ("C-C-1&200000", "C-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
    ]

def list_of_bearings_all():
    return [
        ("H-A-1&200000","H-A-1.mat"), ("H-B-1&200000", "H-B-1.mat"), ("H-C-1&200000", "H-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("H-A-2&200000","H-A-2.mat"), ("H-B-2&200000", "H-B-2.mat"), ("H-C-2&200000", "H-C-2.mat"), ("H-D-2&200000", "H-D-2.mat"),    
        ("H-A-3&200000","H-A-3.mat"), ("H-B-3&200000", "H-B-3.mat"), ("H-C-3&200000", "H-C-3.mat"), ("H-D-3&200000", "H-D-3.mat"),    
        ("I-A-1&200000","I-A-1.mat"), ("I-B-1&200000", "I-B-1.mat"), ("I-C-1&200000", "I-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("I-A-2&200000","I-A-2.mat"), ("I-B-2&200000", "I-B-2.mat"), ("I-C-2&200000", "I-C-2.mat"), ("H-D-2&200000", "H-D-2.mat"),    
        ("I-A-3&200000","I-A-3.mat"), ("I-B-3&200000", "I-B-3.mat"), ("I-C-3&200000", "I-C-3.mat"), ("H-D-3&200000", "H-D-3.mat"),    
        ("O-A-1&200000","O-A-1.mat"), ("O-B-1&200000", "O-B-1.mat"), ("O-C-1&200000", "O-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("O-A-2&200000","O-A-2.mat"), ("O-B-2&200000", "O-B-2.mat"), ("O-C-2&200000", "O-C-2.mat"), ("H-D-2&200000", "H-D-2.mat"),    
        ("O-A-3&200000","O-A-3.mat"), ("O-B-3&200000", "O-B-3.mat"), ("O-C-3&200000", "O-C-3.mat"), ("H-D-3&200000", "H-D-3.mat"),    
        ("B-A-1&200000","B-A-1.mat"), ("B-B-1&200000", "B-B-1.mat"), ("B-C-1&200000", "B-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("B-A-2&200000","B-A-2.mat"), ("B-B-2&200000", "B-B-2.mat"), ("B-C-2&200000", "B-C-2.mat"), ("H-D-2&200000", "H-D-2.mat"),    
        ("B-A-3&200000","B-A-3.mat"), ("B-B-3&200000", "B-B-3.mat"), ("B-C-3&200000", "B-C-3.mat"), ("H-D-3&200000", "H-D-3.mat"),    
        ("C-A-1&200000","C-A-1.mat"), ("C-B-1&200000", "C-B-1.mat"), ("C-C-1&200000", "C-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("C-A-2&200000","C-A-2.mat"), ("C-B-2&200000", "C-B-2.mat"), ("C-C-2&200000", "C-C-2.mat"), ("H-D-2&200000", "H-D-2.mat"),    
        ("C-A-3&200000","C-A-3.mat"), ("C-B-3&200000", "C-B-3.mat"), ("C-C-3&200000", "C-C-3.mat"), ("H-D-3&200000", "H-D-3.mat"),    
    ]

def list_of_bearings_niob():
    return [
        ("H-A-1&200000","H-A-1.mat"), ("H-B-1&200000", "H-B-1.mat"), ("H-C-1&200000", "H-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("H-A-2&200000","H-A-2.mat"), ("H-B-2&200000", "H-B-2.mat"), ("H-C-2&200000", "H-C-2.mat"), ("H-D-2&200000", "H-D-2.mat"),    
        ("H-A-3&200000","H-A-3.mat"), ("H-B-3&200000", "H-B-3.mat"), ("H-C-3&200000", "H-C-3.mat"), ("H-D-3&200000", "H-D-3.mat"),    
        ("I-A-1&200000","I-A-1.mat"), ("I-B-1&200000", "I-B-1.mat"), ("I-C-1&200000", "I-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("I-A-2&200000","I-A-2.mat"), ("I-B-2&200000", "I-B-2.mat"), ("I-C-2&200000", "I-C-2.mat"), ("H-D-2&200000", "H-D-2.mat"),    
        ("I-A-3&200000","I-A-3.mat"), ("I-B-3&200000", "I-B-3.mat"), ("I-C-3&200000", "I-C-3.mat"), ("H-D-3&200000", "H-D-3.mat"),    
        ("O-A-1&200000","O-A-1.mat"), ("O-B-1&200000", "O-B-1.mat"), ("O-C-1&200000", "O-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("O-A-2&200000","O-A-2.mat"), ("O-B-2&200000", "O-B-2.mat"), ("O-C-2&200000", "O-C-2.mat"), ("H-D-2&200000", "H-D-2.mat"),    
        ("O-A-3&200000","O-A-3.mat"), ("O-B-3&200000", "O-B-3.mat"), ("O-C-3&200000", "O-C-3.mat"), ("H-D-3&200000", "H-D-3.mat"),    
        ("B-A-1&200000","B-A-1.mat"), ("B-B-1&200000", "B-B-1.mat"), ("B-C-1&200000", "B-C-1.mat"), ("H-D-1&200000", "H-D-1.mat"),    
        ("B-A-2&200000","B-A-2.mat"), ("B-B-2&200000", "B-B-2.mat"), ("B-C-2&200000", "B-C-2.mat"), ("H-D-2&200000", "H-D-2.mat"),    
        ("B-A-3&200000","B-A-3.mat"), ("B-B-3&200000", "B-B-3.mat"), ("B-C-3&200000", "B-C-3.mat"), ("H-D-3&200000", "H-D-3.mat"),    
    ]


def download_file(url, dirname, bearing):
    print("Downloading Bearing Data:", bearing)   
    file_name = bearing
    try:
        req = urllib.request.Request(url, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])
        dir_path = os.path.join(dirname, file_name)              
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
    
    def __init__(self, sample_size=4096, acquisition_maxsize=None, config='all'):
        self.config = config
        self.n_channels = 2 # channel 1: vibration data, channel 2: rotational speed data
        self.cache_filepath = f'cache/ottawa_{self.config}.npy'
        self.sample_size = sample_size
        self.acquisition_maxsize = acquisition_maxsize
        self.rawfilesdir = "data_raw/raw_ottawa"
        self.url="https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/"
        self.n_folds = 4
        self.signal = Signal('Ottawa', self.cache_filepath)

        


    def download(self):
        list_of_bearings = list_of_bearings_all()
        dirname = self.rawfilesdir
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for bearing in list_of_bearings:
            url = self.url + files_hash()[bearing[1][:-4]]
            files = bearing[1]            
            download_file(url, dirname, files)
                    

    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        os.path.exists(self.rawfilesdir) or self.download()
        list_of_bearings = eval(f"list_of_bearings_{self.config}()")
        for x, (bearing_label, bearing_file) in enumerate(list_of_bearings):
            print('\r', f" Loading acquisitions {100*(x+1)/len(list_of_bearings):.2f} %", end='')
            matlab_file = scipy.io.loadmat(f"{self.rawfilesdir}/{bearing_file}")
            if self.acquisition_maxsize:
                vibration_data = np.array([elem for singleList in matlab_file['Channel_1'] for elem in singleList][:self.acquisition_maxsize])
            else:
                vibration_data = np.array([elem for singleList in matlab_file['Channel_1'] for elem in singleList])
            vibration_data = vibration_data[np.newaxis, :]
            acquisitions = split_acquisition(vibration_data, self.sample_size)
            
            if bearing_label.startswith('H'): 
                bl = bearing_label.replace('H', 'N', 1)
            elif bearing_label.startswith('C'):
                bl = bearing_label.replace('C', 'M', 1)
            else:
                bl = bearing_label
            self.signal.add_acquisitions(bl, acquisitions)
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

    def groups(self):
        return self.group_acquisition()