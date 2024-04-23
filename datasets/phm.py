import numpy as np
import pathlib
import os

def get_sources():
    sources = {
        "data_motor": [f"TYPE{i}" for i in range(1, 5)], 
        "data_gearbox": [f"TYPE{i}" for i in range(5, 13)], 
        "data_leftaxlebox": [f"TYPE{i}" for i in range(13, 17)], 
        "data_rightaxlebox": []
    }
    for k in sources.keys():
        sources[k].insert(0,"TYPE0")
    return {
        "data_motor": [f"TYPE{i}" for i in range(0, 17)], 
        "data_gearbox": [f"TYPE{i}" for i in range(0, 17)], 
        "data_leftaxlebox": [f"TYPE{i}" for i in range(0, 17)], 
        "data_rightaxlebox": [f"TYPE{i}" for i in range(0, 17)], 
    }
    return sources

def pre_stage_tr(data_sources = ["data_motor", "data_gearbox", "data_leftaxlebox", "data_rightaxlebox"]):    
    data_dir = "raw_phm/Data_Pre Stage/Training data"
    data_files = []
    sources = get_sources()
    for fault in pathlib.Path(data_dir).iterdir():
        if fault.is_dir():
            for sample in pathlib.Path(fault).iterdir(): 
                if sample.is_dir():
                    files = []
                    for source in data_sources:
                        print(os.path.basename(sample), sources[source], source, sources)
                        if os.path.basename(sample) in sources[source]:
                            files.append(f"{os.path.join(sample,source)}.csv")
                    if len(files) > 0:
                        data_files.append((f"{os.path.basename(fault)}_{os.path.basename(sample)}", files))
    return data_files

list_of_data_all_tr = pre_stage_tr(["data_motor", "data_gearbox", "data_leftaxlebox", "data_rightaxlebox"])
list_of_data_motor_tr = pre_stage_tr(["data_motor"])
list_of_data_gearbox_tr = pre_stage_tr(["data_gearbox"])
list_of_data_leftaxlebox_tr = pre_stage_tr(["data_leftaxlebox"])
list_of_data_rightaxlebox_tr = pre_stage_tr(["data_rightaxlebox"])


class PHM():
    """
    PHM class wrapper for database adaptation.
    """
    def __str__(self):
        return f"PHM ({self.config})"
    
    def get_phm_bearings(self):
        list_of_bearings = eval("list_of_data_"+self.config)
        bearing_label, bearing_file_names = zip(*list_of_bearings)
        return np.array(bearing_label), np.array(bearing_file_names)

    def __init__(self, sample_size=6400, n_channels=None, acquisition_maxsize=None, config='motor_tr'):
        self.n_channels = n_channels
        self.sample_size = sample_size
        self.acquisition_maxsize = acquisition_maxsize
        self.config = config
        self.n_folds = 3
        self.signal_data = None
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
            vibration_data = None
            for file in self.files[key]:
                vibration_data_raw = np.loadtxt(file, delimiter=',', skiprows=1)
                if vibration_data is None:
                    vibration_data = vibration_data_raw
                else:
                    vibration_data = np.concatenate((vibration_data, vibration_data_raw), axis=1)
            if self.acquisition_maxsize is not None:
                vibration_data = vibration_data[:self.acquisition_maxsize]
            for i in range(len(vibration_data)//self.sample_size):
                sample = vibration_data[(i * self.sample_size):((i + 1) * self.sample_size),:]
                n_channels = sample.shape[1]
                sample = np.array([sample]).reshape(1, -1, n_channels)
                if self.signal_data is None:
                    self.signal_data = np.empty((0, self.sample_size, n_channels))
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
    dataset = PHM(config="motor_tr", acquisition_maxsize=64_000)
    dataset.load_acquisitions()
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    keys = list(set(dataset.keys))
    print("keys", np.array(keys), f"({len(keys)})")
