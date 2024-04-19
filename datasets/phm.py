import numpy as np
import pathlib
import os

def training_pre_stage_cfg():
    training_pre_stage_dir = "raw_phm/Data_Pre Stage/Training data"
    training_pre_stage_files = []
    for fault in pathlib.Path(training_pre_stage_dir).iterdir():
        if fault.is_dir():
            for sample in pathlib.Path(fault).iterdir(): 
                if sample.is_file():
                    training_pre_stage_files.append((f"{os.path.basename(fault)}_{os.path.basename(sample)[:-4]}", f"{sample}"))
    return training_pre_stage_files

list_of_bearings_trpre = training_pre_stage_cfg()

class PHM():
    """
    PHM class wrapper for database download and acquisition.
    """
    def __str__(self):
        return f"PHM ({self.config})"
    
    def get_phm_bearings(self):
        list_of_bearings = eval("list_of_bearings_"+self.config)
        bearing_label, bearing_file_names = zip(*list_of_bearings)
        return np.array(bearing_label), np.array(bearing_file_names)

    def __init__(self, sample_size=6400, n_channels=18, acquisition_maxsize=420_000, config='trpre'):
        self.n_channels = n_channels
        self.sample_size = sample_size
        self.acquisition_maxsize = acquisition_maxsize
        self.config = config
        self.n_folds = 3
        self.signal_data = np.empty((0, self.sample_size, self.n_channels))
        self.labels = []
        self.keys = []
        self.bearing_labels, self.bearing_names = self.get_phm_bearings()
        files_path = {}
        for key, bearing in zip(self.bearing_labels, self.bearing_names):
            files_path[key] = bearing
        self.files = files_path


    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        for x, key in enumerate(self.files):
            print('\r', f" loading acquisitions {100*(x+1)/len(self.files):.2f} %", end='')
            vibration_data_raw = np.loadtxt(self.files[key], delimiter=',', skiprows=1)
            vibration_data = np.array([elem for singleList in vibration_data_raw for elem in singleList])[:self.acquisition_maxsize]
            for i in range(len(vibration_data)//self.sample_size):
                sample = np.empty((self.sample_size, self.n_channels))
                for j in range(self.n_channels):
                    sample[:,j] = vibration_data[(i * self.sample_size):((i + 1) * self.sample_size)]
                sample = np.array([sample]).reshape(1, -1, self.n_channels)
                self.signal_data = np.append(self.signal_data, sample, axis=0)
                self.labels = np.append(self.labels, key.split("_")[0][4:])
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

if __name__ == "__main__":
    dataset = PHM()
    dataset.load_acquisitions()
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    keys = list(set(dataset.keys))
    print("keys", np.array(keys), f"({len(keys)})")
