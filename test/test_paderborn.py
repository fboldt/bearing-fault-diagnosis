import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.paderborn import Paderborn, download_file

class TestPaderborn():
    
    def setUp(self, remove_download=False):
        self.paderborn = Paderborn(config="dbg")
        self.paderborn.rawfilesdir = "test_raw_paderborn"
        self.paderborn.acquisition_maxsize = None
        self.paderborn.sample_size = 4096

        if remove_download:
            if os.path.exists(self.paderborn.rawfilesdir):
                shutil.rmtree(self.paderborn.rawfilesdir)
            os.makedirs(self.paderborn.rawfilesdir, exist_ok=True)

    """
    Test functions
    """
    def test_download(self):
        if os.path.exists(self.paderborn.rawfilesdir):
            shutil.rmtree(self.paderborn.rawfilesdir)
        self.paderborn.download()
  
    def test_load_acquisition(self):
        print("Testing load_acquisition function")
        self.paderborn.load_acquisitions()
        print(self.paderborn.signal.data.shape)

    def test_get_acquisitions(self):
        print("Testing get_acquisitions function")
        self.paderborn.get_acquisitions()
        print('Test get_acquisition ok!')


    """
    Execute Ottawa test
    """
    def to_test(self):
        self.setUp(remove_download=True)
        self.test_download()
        # self.setUp()
        # self.test_load_acquisition()
        # self.setUp()
        # self.test_get_acquisitions() # saving cache
        # self.setUp() 
        # self.test_get_acquisitions() # loading cache
    
if __name__ == '__main__':
    ottawa = TestPaderborn()
    ottawa.to_test()
   
