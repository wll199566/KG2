"""
This script contains all the util functions for file systems.
"""

import os
import glob

def create_folder(folder_path):
    try:
        os.mkdir(folder_path)
        print("Directory " , folder_path,  " Created ")
    except FileExistsError:
        print("Directory " , folder_path,  " already exists")