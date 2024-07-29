import librosa
from scipy.interpolate import CubicSpline
import numpy as np

def resample_data(data, orig_sr, target_sr):
    samples = []
    for i in range(data.shape[0]):
        # resampling
        data_rs = librosa.resample(np.squeeze(data[i]), orig_sr=orig_sr[i], target_sr=target_sr)
        
        # cubic interpolation
        original_time = np.linspace(0, len(data[i]) / orig_sr[i], len(data[i]))
        resampled_time = np.linspace(0, len(data[i]) / orig_sr[i], len(data_rs))
        cs = CubicSpline(resampled_time, data_rs)
        data_interpolated = cs(original_time)
        
        samples.append(data_interpolated)

    return np.reshape(np.array(samples), data.shape)
