import numpy as np

def get_acquisitions(datasets):
    first_dataset = True
    for dataset in datasets:
        # print(dataset)
        Xtmp, ytmp, gtmp, srtmp = dataset.get_acquisitions()
        if first_dataset:
            X, y, g, sr =  Xtmp, ytmp, gtmp, srtmp
            first_dataset = False
        else:
            gtmp += np.max(gtmp)
            X = np.concatenate((X, Xtmp))
            y = np.concatenate((y, ytmp))
            g = np.concatenate((g, gtmp))
            sr = np.concatenate((sr, srtmp))
    return X, y, g, sr