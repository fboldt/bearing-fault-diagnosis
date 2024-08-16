import numpy as np

def get_acquisitions(datasets):
    first_dataset = True
    for dataset in datasets:
        signal, gtmp = dataset.get_acquisitions()
        Xtmp, ytmp = signal.data, signal.labels
        if first_dataset:
            X, y, g =  Xtmp, ytmp, gtmp
            first_dataset = False
        else:
            gtmp += np.max(gtmp)
            X = np.concatenate((X, Xtmp))
            y = np.concatenate((y, ytmp))
            g = np.concatenate((g, gtmp))
    return X, y, g


def split_acquisition(acquisition, sample_size):
    acquisitions = np.empty((0, sample_size, 1))
    for i in range(acquisition.shape[1]//sample_size):
        sample = acquisition[:,(i * sample_size):((i + 1) * sample_size)]
        sample = sample[:, :, np.newaxis] # add a dimension to channel
        acquisitions = np.append(acquisitions, sample, axis=0)
    return acquisitions
    