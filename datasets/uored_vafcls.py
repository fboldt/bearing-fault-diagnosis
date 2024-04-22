"""
Class definition of UORED-VAFCLS Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
import urllib
import sys
import csv

# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)

files_hash = {
    "H_1_0": "31863372-55f7-4c9c-91a5-4f3c907a85af",
    "H_2_0": "7615c4b8-7c8f-41fd-8034-5b6e8438eb16",
    "H_3_0": "e5abb9af-727e-4fd5-8238-0d007f4be6d6",
    "H_4_0": "371e6895-8ae8-4e30-9925-0aacb815dfe9",
    "H_5_0": "ac2a3654-a2ae-4d99-a6a8-9b8e3f7eab27",
    "H_6_0": "960e11a3-e88b-40a2-8d05-e5051951599d",
    "H_7_0": "75bf8b30-beaf-4267-9190-db1ee68c3f14",
    "H_8_0": "0714450f-ebbb-4039-b415-60fc4bef3601",
    "H_9_0": "9cd35272-6b46-4fb5-b482-05ab53cf1c95",
    "H_10_0": "128dde70-6511-4404-9779-28d4a9da7438",
    "H_11_0": "8f7cc1ad-04eb-4b68-99ef-d2adb9ed8825",
    "H_12_0": "2ca4ba4a-146c-4d56-a309-889888713491",
    "H_13_0": "c1cd9d5d-61e2-468b-bf25-9ddeeff4b487",
    "H_14_0": "30fa3b7b-db98-4328-be88-1317ebedbbb6",
    "H_15_0": "75552a72-ed20-4723-8217-afa86e341099",
    "H_16_0": "4e4b52bb-f657-4347-8965-c77b7342f905",
    "H_17_0": "edecdcaf-c6eb-4125-bb63-2b704f33956f",
    "H_18_0": "3768c855-d514-42b3-ae11-700282d57089",
    "H_19_0": "2c74bcb6-d879-4a9a-863c-b257c91a5dc0",
    "H_20_0": "45ab549d-075f-4409-b76e-b8443fa0ca0b",
    "I_1_1": "d383ee8d-75ae-49bd-9fb5-6148695ab69b",
    "I_1_2": "fb1cca43-124d-4b74-b944-f0abc3f28132",
    "I_2_1": "25e8e452-7bee-40ae-9d8a-63f32c3325a4",
    "I_2_2": "0f7addcb-ff4d-4f15-9b91-5493b5bc0dcf",
    "I_3_1": "9c815a92-e72d-48a3-8bc9-14a51f6483fa",
    "I_3_2": "05922ac8-52cd-4189-b23c-a8bc5c9b88eb",
    "I_4_1": "883ba133-1610-4a7c-8084-bb0e941125e2",
    "I_4_2": "113815c9-ddfe-4f63-8f0e-76c75b6c0581",
    "I_5_1": "770d3546-6895-4c99-bfb4-7e57b1d7228d",
    "I_5_2": "c5415d06-2607-4878-b6f5-06cc67b30540",
    "O_6_1": "b3b96601-3e54-4634-95b3-6dd8aeddf613",
    "O_6_2": "c280f3a7-90e6-4438-9ce2-db7b84eb097c",
    "O_7_1": "805c124e-df28-40a5-933e-2e1976fbb73a",
    "O_7_2": "eeb7d7a3-87f3-468a-9038-6224cb256a0a",
    "O_8_1": "66834062-c2d5-45bd-9a72-8960828a8902",
    "O_8_2": "0500d6e7-02fa-4fcb-8df6-96ade544e3e4",
    "O_9_1": "ea795582-032f-4cbd-9601-5bce5873d1f8",
    "O_9_2": "4a8bfeb4-c031-450d-86cc-78e9e7cec17d",
    "O_10_1": "e3b6e7c5-f4c9-492c-ba14-7da2f1955709",
    "O_10_2": "798664fc-ce4f-4d0c-99e0-a30bc08cd6dc",
    "B_11_1": "8d34facf-e922-46cb-9ee7-f04edf7a89d2",
    "B_11_2": "0c1ad1e3-196e-466d-968a-41803f2d020e",
    "B_12_1": "7b9f2066-b6b5-4b0e-8fce-30f0bd15574f",
    "B_12_2": "8837b687-1f08-4cb7-9c98-1bb1823723b4",
    "B_13_1": "a0a2c68c-478b-495d-868c-ffae62b5b445",
    "B_13_2": "a3bdcff7-b9d4-417b-b902-b5cfd08e0985",
    "B_14_1": "7320c429-b6dc-4d4c-9dc4-57df635e603e",
    "B_14_2": "1e5409ec-d96c-4d02-9ef6-69bfd3d6ff2a",
    "B_15_1": "03de7b86-b6e5-4e1b-8e40-d7f9fbed8a12",
    "B_15_2": "17cbbffb-987a-4706-b1b4-778012b3ac16",
    "C_16_1": "ce267f69-4027-4db6-9f23-81c96029b523",
    "C_16_2": "12e2565a-99da-41df-ae49-70f1f5b2fddf",
    "C_17_1": "59b8814c-5a04-40fe-bfd2-7e712414d68f",
    "C_17_2": "1849dd9a-55b0-4313-90d8-fd91e3f3fde6",
    "C_18_1": "c2167e9e-e981-4ca2-ac7c-7dc31e058795",
    "C_18_2": "2654cd65-e54c-4d88-9554-9dbf56a6c60e",
    "C_19_1": "95083600-04f6-4ac4-854e-d5462f8aefbb",
    "C_19_2": "b4a6118d-ddec-4e66-a2f5-ff58a4e08388",
    "C_20_1": "4ceb3588-37b6-4f62-baa4-c64cbf0179a5",
    "C_20_2": "8e8a485f-6fe9-4439-8f93-743a7ac431ec",
}

list_of_bearings_all = [
    "H_1_0",   "H_2_0",   "H_3_0",   "H_4_0",   "H_5_0",   
    "H_6_0",   "H_7_0",   "H_8_0",   "H_9_0",   "H_10_0",  
    "H_11_0",  "H_12_0",  "H_13_0",  "H_14_0",  "H_15_0", 
    "H_16_0",  "H_17_0",  "H_18_0",  "H_19_0",  "H_20_0",  
    "I_1_1",   "I_1_2",   "I_2_1",   "I_2_2",   "I_3_1",   
    "I_3_2",   "I_4_1",   "I_4_2",   "I_5_1",   "I_5_2",   
    "O_6_1",   "O_6_2",   "O_7_1",   "O_7_2",   "O_8_1",   
    "O_8_2",   "O_9_1",   "O_9_2",   "O_10_1",  "O_10_2", 
    "B_11_1",  "B_11_2",  "B_12_1",  "B_12_2",  "B_13_1",
    "B_13_2",  "B_14_1",  "B_14_2",  "B_15_1",  "B_15_2",
    "C_16_1",  "C_16_2",  "C_17_1",  "C_17_2",  "C_18_1", 
    "C_18_2",  "C_19_1",  "C_19_2",  "C_20_1",  "C_20_2"
]

list_of_bearings_nio = [
    "H_1_0",   "H_2_0",   "H_3_0",   "H_4_0",   "H_5_0",   
    "H_6_0",   "H_7_0",   "H_8_0",   "H_9_0",   "H_10_0",  
    "H_11_0",  "H_12_0",  "H_13_0",  "H_14_0",  "H_15_0", 
    "H_16_0",  "H_17_0",  "H_18_0",  "H_19_0",  "H_20_0",  
    "I_1_1",   "I_1_2",   "I_2_1",   "I_2_2",   "I_3_1",   
    "I_3_2",   "I_4_1",   "I_4_2",   "I_5_1",   "I_5_2",   
    "O_6_1",   "O_6_2",   "O_7_1",   "O_7_2",   "O_8_1",   
    "O_8_2",   "O_9_1",   "O_9_2",   "O_10_1",  "O_10_2"   
]

list_of_bearings_niob = [
    "H_1_0",   "H_2_0",   "H_3_0",   "H_4_0",   "H_5_0",   
    "H_6_0",   "H_7_0",   "H_8_0",   "H_9_0",   "H_10_0",  
    "H_11_0",  "H_12_0",  "H_13_0",  "H_14_0",  "H_15_0", 
    "H_16_0",  "H_17_0",  "H_18_0",  "H_19_0",  "H_20_0",  
    "I_1_1",   "I_1_2",   "I_2_1",   "I_2_2",   "I_3_1",   
    "I_3_2",   "I_4_1",   "I_4_2",   "I_5_1",   "I_5_2",   
    "O_6_1",   "O_6_2",   "O_7_1",   "O_7_2",   "O_8_1",   
    "O_8_2",   "O_9_1",   "O_9_2",   "O_10_1",  "O_10_2", 
    "B_11_1",  "B_11_2",  "B_12_1",  "B_12_2",  "B_13_1",
    "B_13_2",  "B_14_1",  "B_14_2",  "B_15_1",  "B_15_2",
]

list_of_bearings_faulty_healthy = [
    "H_1_0",   "H_2_0",   "H_3_0",   "H_4_0",   "H_5_0",   
    "H_6_0",   "H_7_0",   "H_8_0",   "H_9_0",   "H_10_0",  
    "H_11_0",  "H_12_0",  "H_13_0",  "H_14_0",  "H_15_0", 
    "H_16_0",  "H_17_0",  "H_18_0",  "H_19_0",  "H_20_0",  
    "I_1_2",   "I_2_2",   "I_3_2",   "I_4_2",   "I_5_2",   
    "O_6_2",   "O_7_2",   "O_8_2",   "O_9_2",   "O_10_2",  
    "B_11_2",  "B_12_2",  "B_13_2",  "B_14_2",  "B_15_2",  
    "C_16_2",  "C_17_2",  "C_18_2",  "C_19_2",  "C_20_2"
]

list_of_bearings_mert = [
    "H_1_0",   "H_2_0",   "H_3_0", "H_4_0", "H_5_0", 
    "I_1_2",   "I_2_2",   "I_3_2", "I_4_2", "I_5_2", 
    "O_6_2",   "O_7_2",   "O_8_2", "O_9_2", "O_10_2",
    "B_11_2",  "B_12_1",  "B_13_2", "B_14_2", "B_15_2",
]

list_of_bearings_dbg = list_of_bearings_mert

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
   


class UORED_VAFCLS():
    """
    UORED_VAFCLS class wrapper for database download and acquisition.

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
      Download raw files from UORED_VAFCLS website
    load_acquisitions()
      Extract vibration data from files
    """


    def get_uored_vafcls_bearings(self):
        list_of_bearings = eval("list_of_bearings_"+self.config)
        bearing_file_names = [name+'.mat' for name in list_of_bearings]
        bearing_label = [label for label in bearing_file_names]    
        return np.array(bearing_label), np.array(bearing_file_names)

    def __str__(self):
        return f"UORED_VAFCLS ({self.config})"


    def __init__(self, sample_size=8400, n_channels=1, acquisition_maxsize=None, config="dbg"):
        self.url = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/"
        self.sample_size = sample_size
        self.n_channels = n_channels
        self.acquisition_maxsize = acquisition_maxsize
        self.config = config
        self.rawfilesdir = "raw_uored_vafcls"
        self.n_folds = 5
        self.bearing_labels, self.bearing_names = self.get_uored_vafcls_bearings()
        self.accelerometers = ['DE'][:self.n_channels]
        self.signal_data = np.empty((0, self.sample_size, len(self.accelerometers)))
        self.labels = []
        self.keys = []

        # Files Paths ordered by bearings
        files_path = {}
        for key, bearing in zip(self.bearing_labels, self.bearing_names):
            files_path[key] = os.path.join(self.rawfilesdir, bearing)
        self.files = files_path


    def download(self):
        list_of_bearings = eval("list_of_bearings_"+self.config)
        dirname = self.rawfilesdir
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for acquisition in list_of_bearings:
            url = self.url + files_hash[acquisition] 
            files_name = acquisition + '.mat'       
            download_file(url, dirname, files_name)


    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        cwd = os.getcwd()
        for x, key in enumerate(self.files):
            matlab_file = scipy.io.loadmat(os.path.join(cwd, self.files[key]))
            acquisition = []
            print('\r', f" loading acquisitions {100*(x+1)/len(self.files):.2f} %", end='')
            label = self.files[key][len(self.rawfilesdir)+1:-4]
            if self.acquisition_maxsize:
                acquisition.append(matlab_file[label].reshape(1, -1)[0][:self.acquisition_maxsize] )
            else: 
                acquisition.append(matlab_file[label].reshape(1, -1)[0])
            acquisition = np.array(acquisition)
            if len(acquisition.shape)<2 or acquisition.shape[0]<self.n_channels:
                continue
            for i in range(acquisition.shape[1]//self.sample_size):
                sample = acquisition[:,(i * self.sample_size):((i + 1) * self.sample_size)]
                self.signal_data = np.append(self.signal_data, np.array([sample.T]), axis=0)
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
    dataset = UORED_VAFCLS(config='dbg', acquisition_maxsize=84_000)
    # dataset.download()
    dataset.load_acquisitions()
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    keys = list(set(dataset.keys))
    print("keys", np.array(keys), f"({len(keys)})")
