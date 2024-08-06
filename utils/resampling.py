import librosa
import numpy as np

def resample_data(data, orig_sr, target_sr):
    print('original sr:', orig_sr[0])
    print('target sr:', target_sr)
    print('data.shape', data.shape)
    samples = []
    for i in range(data.shape[0]):
        data_rs = librosa.resample(np.squeeze(data[i]), orig_sr=orig_sr[i], target_sr=target_sr)           
        samples.append(data_rs)
    samples_rs = np.array(samples)
    print("samples resampled:", samples_rs.shape)
    return np.reshape(samples_rs, (data.shape[0], samples_rs.shape[1] , data.shape[2]))
