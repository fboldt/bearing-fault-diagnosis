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
    return sources

def pre_stage_tr(data_sources = ["data_motor", "data_gearbox", "data_leftaxlebox", "data_rightaxlebox"]):    
    data_dir = "raw_phm/Data_Pre Stage/Training data"
    data_files = []
    conditions = set()
    sources = get_sources()
    for source in data_sources:
        for condition in sources[source]:
            conditions.add(condition)
    for fault in pathlib.Path(data_dir).iterdir():
        if fault.is_dir() and os.path.basename(fault) in conditions:
            for sample in pathlib.Path(fault).iterdir(): 
                if sample.is_dir():
                    files = []
                    for file in pathlib.Path(sample).iterdir():
                        if os.path.basename(file)[:-4] in data_sources:
                            files.append(f"{file}")
                    if len(files) > 0:
                        data_files.append((f"{os.path.basename(fault)}_{os.path.basename(sample)}", files))
    return data_files

def pre_stage_te(data_sources = ["data_motor", "data_gearbox", "data_leftaxlebox", "data_rightaxlebox"]):    
    data_dir = "raw_phm/Data_Pre Stage/Test data"
    data_files = []
    for sample in pathlib.Path(data_dir).iterdir(): 
        if sample.is_dir():
            files = []
            for file in pathlib.Path(sample).iterdir():
                if os.path.basename(file)[:-4] in data_sources:
                    files.append(f"{file}")
            if len(files) > 0:
                data_files.append((f"test_{os.path.basename(sample)}", files))
    return data_files

def list_of_data_all_te():
    files = pre_stage_te(["data_motor", 
                          "data_gearbox", 
                          "data_leftaxlebox", 
                          "data_rightaxlebox"
                          ])
    return files

def list_of_data_18ch_te():
    files = pre_stage_te(["data_motor", 
                          "data_gearbox", 
                          "data_leftaxlebox", 
                          ])
    return files

def list_of_data_motor_te():
    files = pre_stage_te(["data_motor"])
    return files

def list_of_data_gearbox_te():
    files = pre_stage_te(["data_gearbox"])
    return files

def list_of_data_leftaxlebox_te():
    files = pre_stage_te(["data_leftaxlebox"])
    return files

def list_of_data_rightaxlebox_te():
    files = pre_stage_te(["data_rightaxlebox"])
    return files

def list_of_data_all_tr():
    files = pre_stage_tr(["data_motor", 
                          "data_gearbox", 
                          "data_leftaxlebox", 
                          "data_rightaxlebox"
                          ])
    return files

def list_of_data_18ch_tr():
    files = pre_stage_tr(["data_motor", 
                          "data_gearbox", 
                          "data_leftaxlebox",
                          ])
    return files

def list_of_data_motor_tr():
    return pre_stage_tr(["data_motor"])

def list_of_data_gearbox_tr():
    return pre_stage_tr(["data_gearbox"])

def list_of_data_leftaxlebox_tr():
    return pre_stage_tr(["data_leftaxlebox"])

def list_of_data_rightaxlebox_tr():
    return pre_stage_tr(["data_rightaxlebox"])

class PHM():
    """
    PHM class wrapper for database adaptation.
    """
    def __str__(self):
        return f"PHM ({self.config})"
    
    def get_phm_bearings(self):
        list_of_bearings = eval("list_of_data_"+self.config+"()")
        bearing_label, bearing_file_names = zip(*list_of_bearings)
        return np.array(bearing_label), bearing_file_names

    def __init__(self, sample_size=64000, n_channels=None, acquisition_maxsize=None, config='all_tr', cache_file=None):
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
            if key not in files_path.keys():
                files_path[key] = []
            files_path[key].append(bearing)
        self.files = files_path
        if cache_file is not None:
            self.load_cache(cache_file)
        self.cache_file = cache_file

    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """
        for x, key in enumerate(self.files):
            print('\r', f" loading acquisitions {100*(x+1)/len(self.files):.2f} %", end='')
            vibration_data = None
            for files in self.files[key]:
                vibration_data_raw = None
                for file in files:
                    loaded_data = np.loadtxt(file, delimiter=',', skiprows=1)
                    if vibration_data_raw is None:
                        vibration_data_raw = loaded_data
                    else: 
                        vibration_data_raw = np.concatenate((vibration_data_raw, loaded_data), axis=1)
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
                if not key.startswith('test'):
                    self.labels.append(int(key.split("_")[0][4:]))
                self.keys = np.append(self.keys, key)
        self.labels = np.array(self.labels)
        print(f"  ({len(self.labels)} examples) | labels: {set(self.labels)}")
        
    def get_acquisitions(self):
        if len(self.labels) == 0 and self.cache_file is None:
            self.load_acquisitions()
        groups = self.groups()
        # if str(self.config).endswith("tr"):
        #     groups = groups % 3
        return self.signal_data, self.labels, groups
             
    def group_acquisition(self):
        groups = []
        hash = dict()
        for i in self.keys:
            if i not in hash:
                hash[i] = len(hash)
            groups = np.append(groups, hash[i])
        return np.array(groups, dtype=int)

    def groups(self):
        return self.group_acquisition()
    
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
    config = "motor_tr" # "leftaxlebox_tr" # "gearbox_tr" # "18ch_tr" # "all_tr" # 
    cache_name = f"phm_{config}.npy"
    dataset = PHM(config=config, sample_size=64000, acquisition_maxsize=None)
    # '''
    dataset.load_acquisitions()
    dataset.save_cache(cache_name)
    '''
    dataset.load_cache(cache_name)
    # '''
    print("Signal datase shape", dataset.signal_data.shape)
    labels = list(set(dataset.labels))
    print("labels", labels, f"({len(labels)})")
    # keys = (dataset.keys)
    # print("keys", keys, f"({len(keys)})")
    # groups = (dataset.groups())
    # print("groups", groups, f"({len(groups)})")
    # print([[keys,groups[i]] for i, keys in enumerate(keys)])
