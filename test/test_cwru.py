import os
import sys
import shutil
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.cwru import CWRU

def list_of_bearings_48k():
    return [
        ("I.007.DE_0&48000","109.mat"),       ("I.007.DE_1&48000","110.mat"),       ("I.007.DE_2&48000","111.mat"),       ("I.007.DE_3&48000","112.mat"),
        ("B.007.DE_0&48000","122.mat"),       ("B.007.DE_1&48000","123.mat"),       ("B.007.DE_2&48000","124.mat"),       ("B.007.DE_3&48000","125.mat"),    
        ("O.007.DE.@6_0&48000","135.mat"),    ("O.007.DE.@6_1&48000","136.mat"),    ("O.007.DE.@6_2&48000","137.mat"),    ("O.007.DE.@6_3&48000","138.mat"),    
        ("O.007.DE.@3_0&48000","148.mat"),    ("O.007.DE.@3_1&48000","149.mat"),    ("O.007.DE.@3_2&48000","150.mat"),    ("O.007.DE.@3_3&48000","151.mat"),    
        ("O.007.DE.@12_0&48000","161.mat"),   ("O.007.DE.@12_1&48000","162.mat"),   ("O.007.DE.@12_2&48000","163.mat"),   ("O.007.DE.@12_3&48000","164.mat"),    
        ("I.014.DE_0&48000","174.mat"),       ("I.014.DE_1&48000","175.mat"),       ("I.014.DE_2&48000","176.mat"),       ("I.014.DE_3&48000","177.mat"),    
        ("B.014.DE_0&48000","189.mat"),       ("B.014.DE_1&48000","190.mat"),       ("B.014.DE_2&48000","191.mat"),       ("B.014.DE_3&48000","192.mat"),    
        ("O.014.DE.@6_0&48000","201.mat"),    ("O.014.DE.@6_1&48000","202.mat"),    ("O.014.DE.@6_2&48000","203.mat"),    ("O.014.DE.@6_3&48000","204.mat"),    
        ("I.021.DE_0&48000","213.mat"),       ("I.021.DE_1&48000","214.mat"),       ("I.021.DE_2&48000","215.mat"),       ("I.021.DE_3&48000","217.mat"),    
        ("B.021.DE_0&48000","226.mat"),       ("B.021.DE_1&48000","227.mat"),       ("B.021.DE_2&48000","228.mat"),       ("B.021.DE_3&48000","229.mat"),    
        ("O.021.DE.@6_0&48000","238.mat"),    ("O.021.DE.@6_1&48000","239.mat"),    ("O.021.DE.@6_2&48000","240.mat"),    ("O.021.DE.@6_3&48000","241.mat"),    
        ("O.021.DE.@3_0&48000","250.mat"),    ("O.021.DE.@3_1&48000","251.mat"),    ("O.021.DE.@3_2&48000","252.mat"),    ("O.021.DE.@3_3&48000","253.mat"),    
        ("O.021.DE.@12_0&48000","262.mat"),   ("O.021.DE.@12_1&48000","263.mat"),   ("O.021.DE.@12_2&48000","264.mat"),   ("O.021.DE.@12_3&48000","265.mat"),    
    ]

def list_of_bearings_12k():
    return [
    ("N.000.NN_0&12000","97.mat"),        ("N.000.NN_1&12000","98.mat"),        ("N.000.NN_2&12000","99.mat"),        ("N.000.NN_3&12000","100.mat"),
    ("I.007.DE_0&12000","105.mat"),       ("I.007.DE_1&12000","106.mat"),       ("I.007.DE_2&12000","107.mat"),       ("I.007.DE_3&12000","108.mat"),
    ("B.007.DE_0&12000","118.mat"),       ("B.007.DE_1&12000","119.mat"),       ("B.007.DE_2&12000","120.mat"),       ("B.007.DE_3&12000","121.mat"),    
    ("O.007.DE.@6_0&12000","130.mat"),    ("O.007.DE.@6_1&12000","131.mat"),    ("O.007.DE.@6_2&12000","132.mat"),    ("O.007.DE.@6_3&12000","133.mat"),    
    ("O.007.DE.@3_0&12000","144.mat"),    ("O.007.DE.@3_1&12000","145.mat"),    ("O.007.DE.@3_2&12000","146.mat"),    ("O.007.DE.@3_3&12000","147.mat"),    
    ("O.007.DE.@12_0&12000","156.mat"),   ("O.007.DE.@12_1&12000","158.mat"),   ("O.007.DE.@12_2&12000","159.mat"),   ("O.007.DE.@12_3&12000","160.mat"),    
    ("I.014.DE_0&12000","169.mat"),       ("I.014.DE_1&12000","170.mat"),       ("I.014.DE_2&12000","171.mat"),       ("I.014.DE_3&12000","172.mat"),    
    ("B.014.DE_0&12000","185.mat"),       ("B.014.DE_1&12000","186.mat"),       ("B.014.DE_2&12000","187.mat"),       ("B.014.DE_3&12000","188.mat"),    
    ("O.014.DE.@6_0&12000","197.mat"),    ("O.014.DE.@6_1&12000","198.mat"),    ("O.014.DE.@6_2&12000","199.mat"),    ("O.014.DE.@6_3&12000","200.mat"),    
    ("I.021.DE_0&12000","209.mat"),       ("I.021.DE_1&12000","210.mat"),       ("I.021.DE_2&12000","211.mat"),       ("I.021.DE_3&12000","212.mat"),    
    ("B.021.DE_0&12000","222.mat"),       ("B.021.DE_1&12000","223.mat"),       ("B.021.DE_2&12000","224.mat"),       ("B.021.DE_3&12000","225.mat"),    
    ("O.021.DE.@6_0&12000","234.mat"),    ("O.021.DE.@6_1&12000","235.mat"),    ("O.021.DE.@6_2&12000","236.mat"),    ("O.021.DE.@6_3&12000","237.mat"),    
    ("O.021.DE.@3_0&12000","246.mat"),    ("O.021.DE.@3_1&12000","247.mat"),    ("O.021.DE.@3_2&12000","248.mat"),    ("O.021.DE.@3_3&12000","249.mat"),    
    ("O.021.DE.@12_0&12000","258.mat"),   ("O.021.DE.@12_1&12000","259.mat"),   ("O.021.DE.@12_2&12000","260.mat"),   ("O.021.DE.@12_3&12000","261.mat"),    
    ("I.028.DE_0&12000","3001.mat"),      ("I.028.DE_1&12000","3002.mat"),      ("I.028.DE_2&12000","3003.mat"),      ("I.028.DE_3&12000","3004.mat"),    
    ("B.028.DE_0&12000","3005.mat"),      ("B.028.DE_1&12000","3006.mat"),      ("B.028.DE_2&12000","3007.mat"),      ("B.028.DE_3&12000","3008.mat"),
    ("I.007.FE_0&12000","278.mat"),       ("I.007.FE_1&12000","279.mat"),       ("I.007.FE_2&12000","280.mat"),       ("I.007.FE_3&12000","281.mat"),    
    ("B.007.FE_0&12000","282.mat"),       ("B.007.FE_1&12000","283.mat"),       ("B.007.FE_2&12000","284.mat"),       ("B.007.FE_3&12000","285.mat"),    
    ("O.007.FE.@6_0&12000","294.mat"),    ("O.007.FE.@6_1&12000","295.mat"),    ("O.007.FE.@6_2&12000","296.mat"),    ("O.007.FE.@6_3&12000","297.mat"),    
    ("O.007.FE.@3_0&12000","298.mat"),    ("O.007.FE.@3_1&12000","299.mat"),    ("O.007.FE.@3_2&12000","300.mat"),    ("O.007.FE.@3_3&12000","301.mat"),    
    ("O.007.FE.@12_0&12000","302.mat"),   ("O.007.FE.@12_1&12000","305.mat"),   ("O.007.FE.@12_2&12000","306.mat"),   ("O.007.FE.@12_3&12000","307.mat"),    
    ("I.014.FE_0&12000","274.mat"),       ("I.014.FE_1&12000","275.mat"),       ("I.014.FE_2&12000","276.mat"),       ("I.014.FE_3&12000","277.mat"),    
    ("B.014.FE_0&12000","286.mat"),       ("B.014.FE_1&12000","287.mat"),       ("B.014.FE_2&12000","288.mat"),       ("B.014.FE_3&12000","289.mat"),    
    ("O.014.FE.@3_0&12000","310.mat"),    ("O.014.FE.@3_1&12000","309.mat"),    ("O.014.FE.@3_2&12000","311.mat"),    ("O.014.FE.@3_3&12000","312.mat"),    
    ("O.014.FE.@6_0&12000","313.mat"),    
    ("I.021.FE_0&12000","270.mat"),       ("I.021.FE_1&12000","271.mat"),       ("I.021.FE_2&12000","272.mat"),       ("I.021.FE_3&12000","273.mat"),    
    ("B.021.FE_0&12000","290.mat"),       ("B.021.FE_1&12000","291.mat"),       ("B.021.FE_2&12000","292.mat"),       ("B.021.FE_3&12000","293.mat"),    
    ("O.021.FE.@6_0&12000","315.mat"),    ("O.021.FE.@3_1&12000","316.mat"),    ("O.021.FE.@3_2&12000","317.mat"),    ("O.021.FE.@3_3&12000","318.mat"),    
]

def list_of_bearings_all():
    return list_of_bearings_48k() + list_of_bearings_12k()

class TestCWRU():
    
    def setUp(self, remove_download=False):
        self.cwru = CWRU(config="dbg")
        self.cwru.rawfilesdir = "data_raw_test/raw_cwru"
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


    def test_extract_acquisition(self):        
        self.cwru.acquisition_maxsize = None
        for bearing_label, bearing_file in list_of_bearings_all():
            self.cwru.extract_acquisition(bearing_label, bearing_file)

    def test_extract_acquisition_when_acquisition_maxsize_is_10000(self):
        self.cwru.acquisition_maxsize = 10000
        for bearing_label, bearing_file in [("N.000.NN_0&12000","97.mat"),
                                            ("N.000.NN_1&12000","98.mat"),
                                            ("N.000.NN_2&12000","99.mat"),
                                            ("N.000.NN_3&12000","100.mat"),
                                            ("I.007.DE_0&48000","109.mat"),       
                                            ("I.007.DE_1&48000","110.mat"),       
                                            ("I.007.DE_2&48000","111.mat"),       
                                            ("I.007.DE_3&48000","112.mat"),]:
            self.cwru.extract_acquisition(bearing_label, bearing_file)
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
        self.test_extract_acquisition()
        # self.setUp()
        # self.test_extract_acquisition_when_acquisition_maxsize_is_10000()
        # self.setUp()
        # self.test_extract_acquisition_when_acquisition_maxsize_is_100000000000000()
        # self.setUp()
        # self.test_extract_acquisition_when_bearing_label_is_I_007_DE_0_48000_109_mat()
        # self.setUp()
        # self.test_load_acquisition()
        # self.setUp()
        # self.test_get_acquisitions() # saving cache
        # self.test_get_acquisitions() # loading cache

if __name__ == '__main__':
    TestCWRU().to_test()