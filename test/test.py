from test_cwru import TestCWRU
from test_utils import TestUtils
from test_ottawa import TestOttawa
from test_uored import TestUORED

if __name__ == "__main__":
    tests = [
        TestOttawa(),
        TestCWRU(),
        TestUORED(),
        TestUtils(),
    ]
    
    for test in tests:
        test.to_test()

