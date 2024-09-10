import sys
import os
import numpy as np
from utils.logger import log_message

def groups_by(group_name, keys):
    log_message(f' Grouping the data by {group_name}.')
    groups = []
    hash = dict()
    for i in keys:
        if i not in hash:
            hash[i] = len(hash)
        groups = np.append(groups, hash[i])
    return groups

def calc_groups(keys):
    register = dict()
    for i in keys:
        if i not in register:
            register[i] = 1
        register[i] += 1   
    # show info
    print("{:<10} {:<25} {:<25}".format("Seq.", "Group", "Amount"))
    for x, (key, value) in enumerate(register.items()):
        print("{:<10} {:<25} {:<25}".format(x, key, value))
        
