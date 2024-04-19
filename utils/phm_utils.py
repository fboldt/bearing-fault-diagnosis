import pandas as pd
import os
import pathlib

data_sources = ["data_motor", "data_gearbox", "data_leftaxlebox"] #, "data_rightaxlebox"]
original_files = "raw_phm/Data_Pre Stage"

def joinFiles(dir, data_sources):
    files = [os.path.join(dir,file+".csv") for file in data_sources]
    dfs = [pd.read_csv(file) for file in files]
    merged_df = pd.concat(dfs, axis=1)
    merged_file = f"{dir}.csv"
    merged_df.to_csv(merged_file, index=False)
    print(merged_file)

def joinSamples(path):
    for sample in pathlib.Path(path).iterdir():
        if sample.is_dir():
            joinFiles(sample, data_sources)

def joinTrainingData():
    path = os.path.join(original_files,"Training data")
    for faultType in pathlib.Path(path).iterdir():
        if faultType.is_dir():
            joinSamples(faultType)

def joinTestData():
    path = os.path.join(original_files,"Test data")
    joinSamples(os.path.join(path))

if __name__ == "__main__":
    joinTrainingData()
