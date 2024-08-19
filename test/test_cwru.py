import os
import sys
import shutil
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.cwru import CWRU

class TestCWRU():
    
    def setUp(self, remove_download=False):
        self.cwru = CWRU(config="dbg")
        self.cwru.rawfilesdir = "raw_cwru"
        self.cwru.acquisition_maxsize = None
        self.cwru.sample_size = 4096

        if remove_download:
            if os.path.exists(self.cwru.rawfilesdir):
                shutil.rmtree(self.cwru.rawfilesdir)
            os.makedirs(self.cwru.rawfilesdir, exist_ok=True)

    """
    Test functions
    """    
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

    def test_load_acquisition(self):
        print("Testing load_acquisition function")
        self.cwru.load_acquisitions()
        print(self.cwru.signal.data.shape)

    def test_get_acquisitions(self):
        print("Testing get_acquisitions function")
        self.cwru.get_acquisitions()
        print(self.cwru.signal.data.shape)

    """
    Execute CWRU test
    """
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
        self.setUp()
        self.test_load_acquisition()
        self.setUp()
        self.test_get_acquisitions() # saving cache
        self.test_get_acquisitions() # loading cache
