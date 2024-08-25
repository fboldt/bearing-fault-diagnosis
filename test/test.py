from test_cwru import TestCWRU
from test_utils import TestUtils
from test_ottawa import TestOttawa
from test_uored import TestUORED
from test_experimenter_kfold import TestExperimenterKFold

if __name__ == "__main__":
    tests = [
        TestOttawa(),
        TestCWRU(),
        TestUORED(),
        TestUtils(),
        TestExperimenterKFold(),
    ]
    
    for test in tests:
        test.to_test()

