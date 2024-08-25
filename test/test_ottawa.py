import os
import sys
import shutil
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ottawa import Ottawa

class TestOttawa():
    
    def setUp(self, remove_download=False):
        self.ottawa = Ottawa(config="dbg")
        self.ottawa.rawfilesdir = "test_raw_ottawa"
        self.ottawa.acquisition_maxsize = None
        self.ottawa.sample_size = 4096

        if remove_download:
            if os.path.exists(self.ottawa.rawfilesdir):
                shutil.rmtree(self.ottawa.rawfilesdir)
            os.makedirs(self.ottawa.rawfilesdir, exist_ok=True)

    """
    Test functions
    """
    def test_download(self):
        if os.path.exists(self.ottawa.rawfilesdir):
            shutil.rmtree(self.ottawa.rawfilesdir)
        self.ottawa.download()
  
    def test_load_acquisition(self):
        print("Testing load_acquisition function")
        self.ottawa.load_acquisitions()
        print(self.ottawa.signal.data.shape)

    def test_get_acquisitions(self):
        print("Testing get_acquisitions function")
        self.ottawa.get_acquisitions()
        print('Test get_acquisition ok!')


    """
    Execute Ottawa test
    """
    def to_test(self):
        # self.setUp(remove_download=True)
        # self.test_download()
        self.setUp()
        self.test_load_acquisition()
        self.setUp()
        self.test_get_acquisitions() # saving cache
        self.setUp() 
        self.test_get_acquisitions() # loading cache
    
if __name__ == '__main__':
    ottawa = TestOttawa()
    ottawa.to_test()
   
