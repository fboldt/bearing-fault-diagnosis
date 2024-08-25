from datasets.cwru import CWRU
from experimenter_kfold import kfold
from estimators.estimator_factory import EstimatorFactory

class TestExperimenter():

    """
    Testing kfold function
    """
    def test_kfold(self):
        fac = EstimatorFactory()
        fac.set_estimator('random_forest')
        kfold(CWRU(config='12k'), clfmaker=fac, repetitions=1)

    
    """
    Testing transfer_learning function
    """
    def test_kfold(self):
        fac = EstimatorFactory()
        fac.set_estimator('random_forest')
        kfold(CWRU(config='12k'), clfmaker=fac, repetitions=1)

    """
    Execute the code
    """
    def to_test(self):
        self.test_kfold()
