class TestUtils():
    
    """
    Testing split_acquisition function
    """
    def test_split_acquisition_when_acquisition_has_shape_1_243938_and_sample_size_4096(self):
        import scipy.io
        from utils.get_acquisitions import split_acquisition
        data = scipy.io.loadmat(f'raw_cwru/97.mat')
        acquisition = data['X097_DE_time'].reshape(1, -1)
        acquisitions = split_acquisition(acquisition, 4096)
        assert acquisitions.shape[0] == 59, f"The amount of acquisitions should be 59 and it was {acquisitions.shape[0]}"
        assert acquisitions.shape[1] == 4096, f"The amount of samples should be 4096 and it was {acquisitions.shape[1]}"
        assert acquisitions.shape[2] == 1, f"The amount of channles should be 1 and it was {acquisitions.shape[2]}"

    """
    Execute the code
    """
    def to_test(self):
        self.test_split_acquisition_when_acquisition_has_shape_1_243938_and_sample_size_4096()
