import numpy as np
import pandas as pd
import re
import os

from util import create_directory_structure, read_matlab_file, get_subdirectories, list_files_in_directory



# cwru
list_of_bearings_cwru_48k = [
    ("N.000.NN_0","97.mat"),        ("N.000.NN_1","98.mat"),        ("N.000.NN_2","99.mat"),        ("N.000.NN_3","100.mat"),
    ("I.007.DE_0","109.mat"),       ("I.007.DE_1","110.mat"),       ("I.007.DE_2","111.mat"),       ("I.007.DE_3","112.mat"),
    ("B.007.DE_0","122.mat"),       ("B.007.DE_1","123.mat"),       ("B.007.DE_2","124.mat"),       ("B.007.DE_3","125.mat"),    
    ("O.007.DE.@6_0","135.mat"),    ("O.007.DE.@6_1","136.mat"),    ("O.007.DE.@6_2","137.mat"),    ("O.007.DE.@6_3","138.mat"),    
    ("O.007.DE.@3_0","148.mat"),    ("O.007.DE.@3_1","149.mat"),    ("O.007.DE.@3_2","150.mat"),    ("O.007.DE.@3_3","151.mat"),    
    ("O.007.DE.@12_0","161.mat"),   ("O.007.DE.@12_1","162.mat"),   ("O.007.DE.@12_2","163.mat"),   ("O.007.DE.@12_3","164.mat"),    
    ("I.014.DE_0","174.mat"),       ("I.014.DE_1","175.mat"),       ("I.014.DE_2","176.mat"),       ("I.014.DE_3","177.mat"),    
    ("B.014.DE_0","189.mat"),       ("B.014.DE_1","190.mat"),       ("B.014.DE_2","191.mat"),       ("B.014.DE_3","192.mat"),    
    ("O.014.DE.@6_0","201.mat"),    ("O.014.DE.@6_1","202.mat"),    ("O.014.DE.@6_2","203.mat"),    ("O.014.DE.@6_3","204.mat"),    
    ("I.021.DE_0","213.mat"),       ("I.021.DE_1","214.mat"),       ("I.021.DE_2","215.mat"),       ("I.021.DE_3","217.mat"),    
    ("B.021.DE_0","226.mat"),       ("B.021.DE_1","227.mat"),       ("B.021.DE_2","228.mat"),       ("B.021.DE_3","229.mat"),    
    ("O.021.DE.@6_0","238.mat"),    ("O.021.DE.@6_1","239.mat"),    ("O.021.DE.@6_2","240.mat"),    ("O.021.DE.@6_3","241.mat"),    
    ("O.021.DE.@3_0","250.mat"),    ("O.021.DE.@3_1","251.mat"),    ("O.021.DE.@3_2","252.mat"),    ("O.021.DE.@3_3","253.mat"),    
    ("O.021.DE.@12_0","262.mat"),   ("O.021.DE.@12_1","263.mat"),   ("O.021.DE.@12_2","264.mat"),   ("O.021.DE.@12_3","265.mat"),    
]

list_of_bearings_cmert_cwru = [
    # ("N.000.NN_0","97.mat"),        ("N.000.NN_1","98.mat"),        ("N.000.NN_2","99.mat"),        ("N.000.NN_3","100.mat"),
    ("I.007.DE_0","109.mat"),       ("I.007.DE_1","110.mat"),       ("I.007.DE_2","111.mat"),       ("I.007.DE_3","112.mat"),
    ("B.007.DE_0","122.mat"),       ("B.007.DE_1","123.mat"),       ("B.007.DE_2","124.mat"),       ("B.007.DE_3","125.mat"),    
    ("O.007.DE.@6_0","135.mat"),    ("O.007.DE.@6_1","136.mat"),    ("O.007.DE.@6_2","137.mat"),    ("O.007.DE.@6_3","138.mat"),    
    ("O.007.DE.@3_0","148.mat"),    ("O.007.DE.@3_1","149.mat"),    ("O.007.DE.@3_2","150.mat"),    ("O.007.DE.@3_3","151.mat"),    
    ("O.007.DE.@12_0","161.mat"),   ("O.007.DE.@12_1","162.mat"),   ("O.007.DE.@12_2","163.mat"),   ("O.007.DE.@12_3","164.mat"),    
    ("I.014.DE_0","174.mat"),       ("I.014.DE_1","175.mat"),       ("I.014.DE_2","176.mat"),       ("I.014.DE_3","177.mat"),    
    ("B.014.DE_0","189.mat"),       ("B.014.DE_1","190.mat"),       ("B.014.DE_2","191.mat"),       ("B.014.DE_3","192.mat"),    
    ("O.014.DE.@6_0","201.mat"),    ("O.014.DE.@6_1","202.mat"),    ("O.014.DE.@6_2","203.mat"),    ("O.014.DE.@6_3","204.mat"),    
    # ("I.021.DE_0","213.mat"),       ("I.021.DE_1","214.mat"),       ("I.021.DE_2","215.mat"),       ("I.021.DE_3","217.mat"),    
    # ("B.021.DE_0","226.mat"),       ("B.021.DE_1","227.mat"),       ("B.021.DE_2","228.mat"),       ("B.021.DE_3","229.mat"),    
    # ("O.021.DE.@6_0","238.mat"),    ("O.021.DE.@6_1","239.mat"),    ("O.021.DE.@6_2","240.mat"),    ("O.021.DE.@6_3","241.mat"),    
    ("O.021.DE.@3_0","250.mat"),    ("O.021.DE.@3_1","251.mat"),    ("O.021.DE.@3_2","252.mat"),    ("O.021.DE.@3_3","253.mat"),    
    ("O.021.DE.@12_0","262.mat"),   ("O.021.DE.@12_1","263.mat"),   ("O.021.DE.@12_2","264.mat"),   ("O.021.DE.@12_3","265.mat"),    
]

# hust
list_of_bearings_niob_hust = [
        "B500", "B502", "B504", "B600", "B602", "B604", 
        "B700", "B702", "B704", "B800", "B802", "B804", 
        "I400", "I402", "I404", "I500", "I502", "I504", 
        "I600", "I602", "I604", "I700", "I702", "I704", 
        "I800", "I802", "I804", "N400", "N402", "N404", 
        "N500", "N502", "N504", "N600", "N602", "N604", 
        "N700", "N702", "N704", "N800", "N802", "N804", 
        "O400", "O402", "O404", "O500", "O502", "O504", 
        "O600", "O602", "O604", "O700", "O702", "O704", 
        "O800", "O802", "O804"
]

# uoread
list_of_bearings_faulty_healthy_uoread = [
    "H_1_0",   "H_2_0",   "H_3_0",   "H_4_0",   "H_5_0",   
    "H_6_0",   "H_7_0",   "H_8_0",   "H_9_0",   "H_10_0",  
    "H_11_0",  "H_12_0",  "H_13_0",  "H_14_0",  "H_15_0", 
    "H_16_0",  "H_17_0",  "H_18_0",  "H_19_0",  "H_20_0",  
    "I_1_2",   "I_2_2",   "I_3_2",   "I_4_2",   "I_5_2",   
    "O_6_2",   "O_7_2",   "O_8_2",   "O_9_2",   "O_10_2",  
    "B_11_2",  "B_12_2",  "B_13_2",  "B_14_2",  "B_15_2",  
    "C_16_2",  "C_17_2",  "C_18_2",  "C_19_2",  "C_20_2"
]

list_of_bearings_all_uoread = [
    "H_1_0",   "H_2_0",   "H_3_0",   "H_4_0",   "H_5_0",   
    "H_6_0",   "H_7_0",   "H_8_0",   "H_9_0",   "H_10_0",  
    "H_11_0",  "H_12_0",  "H_13_0",  "H_14_0",  "H_15_0", 
    "H_16_0",  "H_17_0",  "H_18_0",  "H_19_0",  "H_20_0",  
    "I_1_1",   "I_1_2",   "I_2_1",   "I_2_2",   "I_3_1",   
    "I_3_2",   "I_4_1",   "I_4_2",   "I_5_1",   "I_5_2",   
    "O_6_1",   "O_6_2",   "O_7_1",   "O_7_2",   "O_8_1",   
    "O_8_2",   "O_9_1",   "O_9_2",   "O_10_1",  "O_10_2", 
    "B_11_1",  "B_11_2",  "B_12_1",  "B_12_2",  "B_13_1",
    "B_13_2",  "B_14_1",  "B_14_2",  "B_15_1",  "B_15_2",
    "C_16_1",  "C_16_2",  "C_17_1",  "C_17_2",  "C_18_1", 
    "C_18_2",  "C_19_1",  "C_19_2",  "C_20_1",  "C_20_2"
]

# paderborn
list_of_bearings_artificial_demange_and_healthy = [
    'K001', 'K002', 'K003', 'K004', 'K005', 'K006',
    'KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08',
    'KA09', 'KI01', 'KI03', 'KI05', 'KI07', 'KI08'
]


def save_cwru_data(directory, list_bearings):
    counter = { 'N': 0, 'I': 0, 'O': 0, 'C': 0, 'B': 0 }
    for bearing in list_bearings:
        label = bearing[0][0]
        filename = bearing[1]
        data = read_matlab_file(f"raw_cwru/{filename}")
        keys_file = data.keys()
        
        
        for key in keys_file:
            match = re.search(r'X[0-9]{3}_[A-Z]+_time', key)
            if match:
                for key, value in {'normal': 'N', 'inner_fault': 'I', 
                                   'outer_fault': 'O', 'cage_fault': 'C', 'ball_fault': 'B'}.items():
                    if label == value:
                        path = f"{directory}/{key}/cwru_{counter[value]}"
                        np.save(path, data[match.group(0)][:420_000])
                        counter[value] += 1


    print(counter)
    

def save_hust_data(list_bearings):
    counter = { 'N': 0, 'I': 0, 'O': 0, 'C': 0 }

    for bearing in list_bearings:
        file_path = f"raw_hust/{bearing}.mat"
        data = read_matlab_file(file_path)['data']
        for key, value in {'normal': 'N', 'inner_fault': 'I', 'outer_fault': 'O', 'cage_fault': 'C'}.items():
            if bearing[0] == value:
                path = f"data/{key}/hust_{counter[value]}"
                np.save(path, data[:420_000])
                counter[value] += 1

    print(counter)


def save_uoread_data(list_bearings):
    counter = { 'N': 0, 'I': 0, 'O': 0, 'C': 0 }

    for bearing in list_bearings:
        file_path = f"raw_uored_vafcls/{bearing}.mat"
        data = read_matlab_file(file_path)[bearing]
        for key, value in {'normal': 'H', 'inner_fault': 'I', 'outer_fault': 'O', 'cage_fault': 'C'}.items():
            if bearing[0] == value:
                path = f"data/{key}/uoread_{counter['N' if value == 'H' else value]}"
                np.save(path, data[:420_000])
                counter['N' if value == 'H' else value] += 1

    print(counter)


def save_mfpt_data(list_bearings=None):
    # get all bearings
    list_directories = get_subdirectories('raw_mfpt')
    counter = { 'N': 0, 'I': 0, 'O': 0, 'C': 0 }
    for directory in list_directories:
        files = list_files_in_directory(directory)
        for filename in files:
            match = re.search(r'(?:baseline_|OuterRace|InnerRace)\w*.mat$', filename)
            if match:
                file_path = f"{directory}/{match.group(0)}"
                data = None
                if match.group(0).find('baseline')==0:
                    data = read_matlab_file(file_path)['bearing'][0][0][1]
                else:
                    data = read_matlab_file(file_path)['bearing'][0][0][2]
                label = match.group(0)[0]            
                for key, value in {'normal': 'b', 'inner_fault': 'I', 'outer_fault': 'O', 'cage_fault': 'C'}.items():
                    if label == value:
                        path = f"data/{key}/mfpt_{counter['N' if value=='b' else value]}"
                        np.save(path, data[:420_000])
                        counter['N' if value=='b' else value] += 1

    print(counter)


def save_paderborn_data(list_bearings):
    counter = { 'N': 0, 'I': 0, 'O': 0, 'C': 0 }
    map = {'normal': 'N', 'inner_fault': 'I', 'outer_fault': 'O', 'cage_fault': 'C'}

    directories = get_subdirectories('raw_paderborn')

    for directory in directories:
        bearings_file = list_files_in_directory(directory)
        for bearing in bearings_file:
            file_path = os.path.join(directory, bearing)
            match = re.search(r'\bK[0AI]0\d\b', file_path)
            if match:
                if match.group(0) in list_bearings:
                    filename_match = re.search(r'/(\w+).mat$', file_path)
                    if filename_match:
                        filename = filename_match.group(1)

                        try:
                            data = read_matlab_file(file_path)[filename]
                        except:
                            print(f"File {file_path} not found.")

                        if match.group(0)[:2] == 'K0':
                            key = 'normal'
                        elif match.group(0)[:2] == 'KI':
                            key = 'inner_fault'
                        elif match.group(0)[:2] == 'KA':
                            key = 'outer_fault'
                        path = f"data/{key}/paderborn_{counter[map[key]]}"
                        np.save(path, data[:420_000])
                        counter[map[key]] += 1
            
    print(counter)


def save_mafaulda_data(list_bearings=None):
    list_of_bearings = get_subdirectories("raw_mafaulda")

    counter = { 'N': 0, 'I': 0, 'O': 0, 'C': 0 }
    map = {'normal': 'N', 'imbalance':'N',  'ball_fault': 'I', 'outer_race': 'O', 'cage_fault': 'C'}
    map_dir = {'N': 'normal', 'B': 'ball_fault', 'O': 'outer_fault', 'I': 'inner_fault', 'C': 'cage_fault'}
    for path in list_of_bearings:
        match = re.search(r'normal|imbalance|ball_fault|cage_fault|outer_race', path)
        files = list_files_in_directory(path)    
        if match:
            for filename in files:
                filepath = os.path.join(path, filename)
                for i in [2, 5]:
                    npy_filename = f"mafaulda_{counter[map[match.group(0)]]}"
                    data = pd.read_csv(filepath).iloc[:420_000, [i]]
                    out_dir = f'data/{map_dir[map[match.group(0)]]}/{npy_filename}'
                    np.save(out_dir, data.to_numpy())
                    counter[map[match.group(0)]] += 1


    print(counter)

if __name__ == "__main__":
    directory = 'data_test_cwru'
    # creates a structure that will receive the data 
    create_directory_structure(directory)
    save_cwru_data(directory, list_bearings=list_of_bearings_cmert_cwru)