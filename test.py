import os
import shutil
import numpy as np
import random

def list_of_bearings_test():
    return [
        ("N.000.NN_0&12000","97.mat"), ("I.007.DE_0&48000","109.mat"), ("O.021.DE.@6_1&12000","235.mat")   
    ]

class TestCWRU():
    
    def setUp(self, remove_download=False):

        from datasets.cwru import CWRU

        self.cwru = CWRU(config="test")

        self.cwru.rawfilesdir = "test_raw_cwru"
        self.cwru.acquisition_maxsize = None
        self.cwru.sample_size = 4096
        self.cwru.target_sr = 12000
        self.cwru.signal_data = np.empty((0, 0, 1))
        self.cwru.labels = np.array([])
        self.cwru.keys = np.array([])

        if remove_download:
            if os.path.exists(self.cwru.rawfilesdir):
                shutil.rmtree(self.cwru.rawfilesdir)
            os.makedirs(self.cwru.rawfilesdir, exist_ok=True)
    
          
    def test_download(self):
        if os.path.exists(self.cwru.rawfilesdir):
            shutil.rmtree(self.cwru.rawfilesdir)
        self.cwru.download()

    def test_extract_acquisition_when_acquisition_maxsize_is_none(self):        
        self.cwru.acquisition_maxsize = None
        self.cwru.extract_acquisition(bearing_label="N.000.NN_0&12000", filename="97.mat")
        assert self.cwru.signal_data.shape[0] == 118, f"The shape of the signal should be 118 and it was {self.cwru.signal_data.shape[0]}"
        assert self.cwru.signal_data.shape[1] == 4096, f"The signal should have samples with size 4096, but instead it is presenting {self.cwru.signal_data.shape[1]}"
        assert self.cwru.signal_data.shape[2] == 1, f"The signal should have only 1 channel, but instead it is presenting {self.cwru.signal_data.shape[2]}"
        assert self.cwru.labels.shape[0] == 118, f"The self.labels should have 118 labels, but instead it is presenting {self.cwru.labels.shape[0]}"
        assert self.cwru.keys[random.randint(0,117)] == "N.000.NN_0&12000", f"The keys, in any position, should have shown the value 'N.000.NN_0&12000', but instead it is showing {self.cwru.keys[0]}"


    def test_extract_acquisition_when_acquisition_maxsize_is_10000(self):
        self.cwru.acquisition_maxsize = 10000
        self.cwru.extract_acquisition(bearing_label="N.000.NN_0&12000", filename="97.mat")
        assert self.cwru.signal_data.shape[0] == 4, f"The shape of the signal should be 4 and it was {self.cwru.signal_data.shape[0]}"
        assert self.cwru.signal_data.shape[1] == 4096, f"The signal should have samples with size 4096, but instead it is presenting {self.cwru.signal_data.shape[1]}"
        assert self.cwru.signal_data.shape[2] == 1, f"The signal should have only 1 channel, but instead it is presenting {self.cwru.signal_data.shape[2]}"
        assert self.cwru.labels.shape[0] == 4, f"The self.labels should have 4 labels, but instead it is presenting {self.cwru.labels.shape[0]}"
        assert self.cwru.keys[random.randint(0,3)] == "N.000.NN_0&12000", f"The keys, in any position, should have shown the value 'N.000.NN_0&12000', but instead it is showing {self.cwru.keys[0]}"


    def test_extract_acquisition_when_acquisition_maxsize_is_100000000000000(self):
        # should display a message stating that 'max_acquisition is greater than the signal contained in sample N.000.NN_0&12000'
        # The length of the total sample contained in the raw signal will be considered.
        self.cwru.acquisition_maxsize = 100000000000000
        self.cwru.extract_acquisition(bearing_label="N.000.NN_0&12000", filename="97.mat")
        assert self.cwru.signal_data.shape[0] == 118, f"The shape of the signal should be 118 and it was {self.cwru.signal_data.shape[0]}"
        assert self.cwru.signal_data.shape[1] == 4096, f"The signal should have samples with size 4096, but instead it is presenting {self.cwru.signal_data.shape[1]}"
        assert self.cwru.signal_data.shape[2] == 1, f"The signal should have only 1 channel, but instead it is presenting {self.cwru.signal_data.shape[2]}"
        assert self.cwru.labels.shape[0] == 118, f"The self.labels should have 118 labels, but instead it is presenting {self.cwru.labels.shape[0]}"
        assert self.cwru.keys[random.randint(0,117)] == "N.000.NN_0&12000", f"The keys, in any position, should have shown the value 'N.000.NN_0&12000', but instead it is showing {self.cwru.keys[0]}"

    def test_extract_acquisition_when_bearing_label_is_I_007_DE_0_48000_109_mat(self):
        self.cwru.acquisition_maxsize = None
        self.cwru.extract_acquisition(bearing_label="I.007.DE_0&48000", filename="109.mat")
        assert self.cwru.signal_data.shape[0] == 28, f"The shape of the signal should be 59 and it was {self.cwru.signal_data.shape[0]}"
        assert self.cwru.signal_data.shape[1] == 16384, f"The signal should have samples with size 4096, but instead it is presenting {self.cwru.signal_data.shape[1]}"
        assert self.cwru.signal_data.shape[2] == 1, f"The signal should have only 1 channel, but instead it is presenting {self.cwru.signal_data.shape[2]}"
        assert self.cwru.labels.shape[0] == 28, f"The self.labels should have 59 labels, but instead it is presenting {self.cwru.labels.shape[0]}"
        assert self.cwru.keys[random.randint(0,27)] == "I.007.DE_0&48000", f"The keys, in any position, should have shown the value 'I.007.DE_0&48000', but instead it is showing {self.cwru.keys[0]}"


    def to_test(self):
        # self.setUp(remove_download=True)
        # self.test_download()
        self.setUp()
        self.test_extract_acquisition_when_acquisition_maxsize_is_none()
        self.setUp()
        self.test_extract_acquisition_when_acquisition_maxsize_is_10000()
        self.setUp()
        self.test_extract_acquisition_when_acquisition_maxsize_is_100000000000000()
        self.setUp()
        self.test_extract_acquisition_when_bearing_label_is_I_007_DE_0_48000_109_mat()
   
    # # test_load()
class Utils():
    def adjust_sample_size_when_sample_size_1000_orig_sr_100_and_target_sr_50(self):
        from utils.resampling import compute_resample_size
        assert compute_resample_size(1000, 100, 50) == 500, f"computer_resample_size should return 500, but instead it returned {compute_resample_size(1000, 100, 50)}"
    def resample_sample_when_sample_size_1000_orig_sr_100_and_target_sr_50(self):
        pass
    def to_test(self):
        self.adjust_sample_size_when_sample_size_1000_orig_sr_100_and_target_sr_50()

if __name__ == "__main__":
    test_cwru = TestCWRU()
    test_cwru.to_test()

