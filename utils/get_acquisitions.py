import numpy as np

def get_acquisitions(datasets):
    first_dataset = True
    for dataset in datasets:
        print(dataset)
        Xtmp, ytmp, gtmp = dataset.get_acquisitions()
        if first_dataset:
            X, y, g =  Xtmp, ytmp, gtmp
            first_dataset = False
        else:
            gtmp += np.max(gtmp)
            X = np.concatenate((X, Xtmp))
            y = np.concatenate((y, ytmp))
            g = np.concatenate((g, gtmp))
    return X, y, g