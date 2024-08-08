import os
import shutil


def test_cwru_dataset():
    from datasets.cwru import CWRU
    cwru = CWRU(config="test")
    def test_download():
        cwru.rawfilesdir = "test_raw_cwru"
        if os.path.exists(cwru.rawfilesdir):
            shutil.rmtree(cwru.rawfilesdir)
        cwru.download()
    def test_load():
        cwru.rawfilesdir = "test_raw_cwru"
        cwru.load()
    test_download()
    test_load()

if __name__ == "__main__":
    test_cwru_dataset()