"""
This script contains all the util functions for file systems.
"""

import os
import glob

import yaml

def create_folder(folder_path):
    try:
        os.mkdir(folder_path)
        print("Directory " , folder_path,  " Created ")
    except FileExistsError:
        print("Directory " , folder_path,  " already exists")


def load_config(filename):
    params = {}
    
    with open(filename) as f:
        params = yaml.load(f)

    return params