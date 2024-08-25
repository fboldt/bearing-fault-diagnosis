import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.uored_vafcls import UORED_VAFCLS

class TestUORED():
    
    def setUp(self, remove_download=False):
        self.uored = UORED_VAFCLS(config="dbg")
        self.uored.rawfilesdir = "test_raw_uored"
        self.uored.acquisition_maxsize = None
        self.uored.sample_size = 4096

        if remove_download:
            if os.path.exists(self.uored.rawfilesdir):
                shutil.rmtree(self.uored.rawfilesdir)
            os.makedirs(self.uored.rawfilesdir, exist_ok=True)

    """
    Test functions
    """
    def test_download(self):
        if os.path.exists(self.uored.rawfilesdir):
            shutil.rmtree(self.uored.rawfilesdir)
        self.uored.download()
  
    def test_load_acquisition(self):
        print("Testing load_acquisition function")
        self.uored.load_acquisitions()
        print(self.uored.signal.data.shape)

    def test_get_acquisitions(self):
        print("Testing get_acquisitions function")
        self.uored.get_acquisitions()
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
    ottawa = TestUORED()
    ottawa.to_test()
   
