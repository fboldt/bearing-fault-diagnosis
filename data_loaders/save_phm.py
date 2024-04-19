from scipy.io import savemat
import pandas as pd
import numpy as np
import re
import os
import sys


np.set_printoptions(threshold=sys.maxsize)

# utils
def list_files_in_directory(directory):
    try:        
        files = os.listdir(directory)
        return files
    except Exception as e:
        print("Error listing files:", e)
        return None

def get_subdirectories(parent_directory):
    subdirectories = []
    for dirpath, dirnames, filenames in os.walk(parent_directory):
        if not dirnames:
            subdirectories.append(dirpath)
    return subdirectories


maxsize_acquisition = 420_000

def reorganize_and_save_phm():
    subdirectories = get_subdirectories('Data_Pre Stage/Training data')
    count=0
    for subdirectory in subdirectories:
        files = list_files_in_directory(subdirectory)
        print('\r', f" saving data {100*(count+1)/len(subdirectories):.2f} %", end='')
        count += 1
        for file in files:
            filepath = os.path.join(subdirectory, file)
            match = re.search(r'/(TYPE\d+)/Sample(\d+)/', filepath)
            if match:
                type = match.groups()[0]
                n_sample = match.groups()[1]
                data = pd.read_csv(filepath)
                # data_dict = dict(zip(list(data.columns), list(data.values)))
                for col in list(data.columns):
                    filename = f"{type}_S{n_sample}_{col}.mat"
                    path = os.path.join('raw_phm', filename)
                    try:
                        savemat(path, {'data': data[col].values[:maxsize_acquisition]})
                        # print("Data saved")
                    except Exception as e:
                        print("Error saving data:", e)
                        

if __name__ == '__main__':

    if not os.path.exists('raw_phm'):
        os.mkdir('raw_phm')

    reorganize_and_save_phm()