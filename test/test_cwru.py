import os
import sys
import shutil
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.cwru import CWRU

def list_of_bearings_test():
    return [
        ("N.000.NN_0&12000","97.mat"), ("I.007.DE_0&48000","109.mat"), ("O.021.DE.@6_1&12000","235.mat")   
    ]

class TestCWRU():
    
    def setUp(self, remove_download=False):
        self.cwru = CWRU(config="test")
        self.cwru.rawfilesdir = "test_raw_cwru"
        self.cwru.cache_dirname = 'test_cache'
        self.cwru.acquisition_maxsize = None
        self.cwru.sample_size = 4096

        if remove_download:
            if os.path.exists(self.cwru.rawfilesdir):
                shutil.rmtree(self.cwru.rawfilesdir)
            os.makedirs(self.cwru.rawfilesdir, exist_ok=True)

        if os.path.isdir(self.cwru.cache_dirname):
            shutil.rmtree(self.cwru.cache_dirname)

          
    def test_download(self):
        if os.path.exists(self.cwru.rawfilesdir):
            shutil.rmtree(self.cwru.rawfilesdir)
        self.cwru.download()


    def test_extract_acquisition_when_acquisition_maxsize_is_none(self):        
        self.cwru.acquisition_maxsize = None
        self.cwru.extract_acquisition(bearing_label="N.000.NN_0&12000", bearing_file="97.mat")
        assert self.cwru.signal.data.shape[0] == 118, f"The shape of the signal should be 118 and it was {self.cwru.data.shape[0]}"
        assert self.cwru.signal.data.shape[1] == 4096, f"The signal should have samples with size 4096, but instead it is presenting {self.cwru.data.shape[1]}"
        assert self.cwru.signal.data.shape[2] == 1, f"The signal should have only 1 channel, but instead it is presenting {self.cwru.data.shape[2]}"
        assert self.cwru.signal.labels.shape[0] == 118, f"The self.labels should have 118 labels, but instead it is presenting {self.cwru.labels.shape[0]}"
        assert self.cwru.signal.keys[random.randint(0,117)] == "N.000.NN_0&12000", f"The keys, in any position, should have shown the value 'N.000.NN_0&12000', but instead it is showing {self.cwru.keys[0]}"


    def test_extract_acquisition_when_acquisition_maxsize_is_10000(self):
        self.cwru.acquisition_maxsize = 10000
        self.cwru.extract_acquisition(bearing_label="N.000.NN_0&12000", bearing_file="97.mat")
        assert self.cwru.signal.data.shape[0] == 4, f"The shape of the signal should be 4 and it was {self.cwru.signal.data.shape[0]}"
        assert self.cwru.signal.data.shape[1] == 4096, f"The signal should have samples with size 4096, but instead it is presenting {self.cwru.signal.data.shape[1]}"
        assert self.cwru.signal.data.shape[2] == 1, f"The signal should have only 1 channel, but instead it is presenting {self.cwru.signal.data.shape[2]}"
        assert self.cwru.signal.labels.shape[0] == 4, f"The self.labels should have 4 labels, but instead it is presenting {self.cwru.signal.labels.shape[0]}"
        assert self.cwru.signal.keys[random.randint(0,3)] == "N.000.NN_0&12000", f"The keys, in any position, should have shown the value 'N.000.NN_0&12000', but instead it is showing {self.cwru.signal.keys[0]}"


    def test_extract_acquisition_when_acquisition_maxsize_is_100000000000000(self):
        # should display a message stating that 'max_acquisition is greater than the signal contained in sample N.000.NN_0&12000'
        # The length of the total sample contained in the raw signal will be considered.
        self.cwru.acquisition_maxsize = 100000000000000
        self.cwru.extract_acquisition(bearing_label="N.000.NN_0&12000", bearing_file="97.mat")
        assert self.cwru.signal.data.shape[0] == 118, f"The shape of the signal should be 118 and it was {self.cwru.signal.data.shape[0]}"
        assert self.cwru.signal.data.shape[1] == 4096, f"The signal should have samples with size 4096, but instead it is presenting {self.cwru.signal.data.shape[1]}"
        assert self.cwru.signal.data.shape[2] == 1, f"The signal should have only 1 channel, but instead it is presenting {self.cwru.signal.data.shape[2]}"
        assert self.cwru.signal.labels.shape[0] == 118, f"The self.labels should have 118 labels, but instead it is presenting {self.cwru.signal.labels.shape[0]}"
        assert self.cwru.signal.keys[random.randint(0,117)] == "N.000.NN_0&12000", f"The keys, in any position, should have shown the value 'N.000.NN_0&12000', but instead it is showing {self.cwru.signal.keys[0]}"


    def test_extract_acquisition_when_bearing_label_is_I_007_DE_0_48000_109_mat(self):        
        self.cwru.extract_acquisition(bearing_label="I.007.DE_0&48000", bearing_file="109.mat")
        self.cwru.acquisition_maxsize = None
        assert self.cwru.signal.data.shape[0] == 118, f"The shape of the signal should be 28 and it was {self.cwru.signal.data.shape[0]}"
        assert self.cwru.signal.data.shape[1] == 4096, f"The signal should have samples with size 4096, but instead it is presenting {self.cwru.signal.data.shape[1]}"
        assert self.cwru.signal.data.shape[2] == 1, f"The signal should have only 1 channel, but instead it is presenting {self.cwru.signal.data.shape[2]}"
        assert self.cwru.signal.labels.shape[0] == 118, f"The self.labels should have 59 labels, but instead it is presenting {self.cwru.signal.labels.shape[0]}"
        assert self.cwru.signal.keys[random.randint(0,117)] == "I.007.DE_0&48000", f"The keys, in any position, should have shown the value 'I.007.DE_0&48000', but instead it is showing {self.cwru.signal.keys[0]}"

    """
    Testing load_acquisition
    """
    def test_load_acquisition_when_list_of_bearing_is_config_test(self):
        self.cwru.load_acquisitions()
        assert self.cwru.signal.data.shape[0] == 323, f"The shape of the signal should be 222 and it was {signal.data.shape[0]}"
        assert self.cwru.signal.data.shape[1] == 4096, f"The signal should have samples with size 4096, but instead it is presenting {signal.data.shape[1]}"
        assert self.cwru.signal.data.shape[2] == 1, f"The signal should have only 1 channel, but instead it is presenting {signal.data.shape[2]}"
        assert self.cwru.signal.labels.shape[0] == 323, f"The self.labels should have 59 labels, but instead it is presenting {signal.labels.shape[0]}"

    """
    Testing get_acquisitions when data cached
    """
    def test_get_acquisitions(self):
        self.test_load_acquisition_when_list_of_bearing_is_config_test()
        self.cwru.get_acquisitions()
        
    """
    Testing save_cache function
    """
    def test_save_cache(self):
        self.test_get_acquisitions()
        self.cwru.signal.save_cache('cache/test.npy')

    """
    Testing load_cache function
    """
    def test_load_cache(self):
        self.cwru.signal.load_cache('cache/test.npy')
        assert self.cwru.signal.data.shape[0] != 0, 'The signal was not loaded.'

    """
    Execute CWRU test
    """
    def to_test(self):
        self.setUp(remove_download=True)
        self.test_download()
        self.setUp()
        self.test_extract_acquisition_when_acquisition_maxsize_is_none()
        self.setUp()
        self.test_extract_acquisition_when_acquisition_maxsize_is_10000()
        self.setUp()
        self.test_extract_acquisition_when_acquisition_maxsize_is_100000000000000()
        self.setUp()
        self.test_extract_acquisition_when_bearing_label_is_I_007_DE_0_48000_109_mat()
        self.setUp()
        self.test_load_acquisition_when_list_of_bearing_is_config_test()
        self.setUp()
        self.test_save_cache()
        self.test_load_cache()
   
