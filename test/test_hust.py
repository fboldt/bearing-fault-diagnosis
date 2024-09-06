import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.hust import Hust

class TestHust():
    
    def setUp(self, remove_download=False, remove_cache=False):
        self.hust = Hust(config="dbg")
        self.hust.acquisition_maxsize = None
        self.hust.sample_size = 4096

        # if remove_download:
        #     if os.path.exists(self.hust.rawfilesdir):
        #         shutil.rmtree(self.hust.rawfilesdir)
        #     os.makedirs(self.hust.rawfilesdir, exist_ok=True)

        # if remove_cache:
        #     if os.path.exists(self.hust.cache_filepath):
        #         shutil.rmtree(self.hust.cache_filepath)
        #     os.makedirs(self.hust.rawfilesdir, exist_ok=True)

    """
    Test functions
    """
    def test_download(self):
        print('Testing download function')        
        self.hust.download(config='dbg')
    
    def test_load_acquisition(self):
        print("Testing load_acquisition function")
        self.hust.load_acquisitions()
    
    def test_get_acquisitions(self):
        print("Testing get_acquisitions function")
        self.hust.get_acquisitions()
    

    """
    Run test
    """
    def to_test(self):
        # self.setUp(remove_download=True)
        # self.test_download()
        self.setUp()
        self.test_load_acquisition()
        self.setUp()
        self.test_get_acquisitions() # saving cache       
    
if __name__ == '__main__':
    TestHust().to_test()
   
