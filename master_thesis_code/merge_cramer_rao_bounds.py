# script to merge the cramer rao bounds for the different simulations

import os
import sys
import numpy as np
import pandas as pd
from typing import List
from master_thesis_code.constants import CRAMER_RAO_BOUNDS_PATH, CRAMER_RAO_BOUNDS_OUTPUT_PATH

def delete_files(used_files: List[str]) -> None:
    for file in used_files:
        os.remove(file)
        print(f"Deleted {file}.")

if __name__ == "__main__":
    # get directory of the script
    directory = os.path.dirname(os.path.realpath(CRAMER_RAO_BOUNDS_OUTPUT_PATH))
    files = os.listdir(directory)
    used_files = []
    common_file_name = CRAMER_RAO_BOUNDS_PATH.split(".")[0].replace("$index", "").split("/")[1]
    cramer_rao_bounds_files = [
        file for file in files if common_file_name in file
    ]
    cramer_rao_bounds_files = [directory + "/" + file for file in cramer_rao_bounds_files]

    # check if there are any files to merge
    if len(cramer_rao_bounds_files) == 0:
        sys.exit("No cramer rao bounds files found.")
    
    # check for existing output file
    try:
        with open(CRAMER_RAO_BOUNDS_OUTPUT_PATH, "r") as f:
            cramer_rao_bounds = pd.read_csv(f)
    except FileNotFoundError:
        # use first file as base
        first_file = cramer_rao_bounds_files.pop(0)
        cramer_rao_bounds = pd.read_csv(first_file)
        used_files.append(first_file)
    
    if len(cramer_rao_bounds_files) == 0:
        # save the file
        cramer_rao_bounds.to_csv(CRAMER_RAO_BOUNDS_OUTPUT_PATH, index=False)
        print(f"Only one cramer rao bounds file found. Saved as {CRAMER_RAO_BOUNDS_OUTPUT_PATH}.")
        to_be_deleted = input("Do you want to delete the used files? [y/n] ")
        if to_be_deleted == "y":
            delete_files(used_files)
        sys.exit()
    else:
        print(f"Merging {len(cramer_rao_bounds_files)} cramer rao bounds files.")

    # merge all files
    for file in cramer_rao_bounds_files:
        cramer_rao_bounds = pd.concat([cramer_rao_bounds, pd.read_csv(file)])
        used_files.append(file)
    
    # save the file
    cramer_rao_bounds.to_csv(CRAMER_RAO_BOUNDS_OUTPUT_PATH, index=False)
    print(f"Saved merged cramer rao bounds as {CRAMER_RAO_BOUNDS_OUTPUT_PATH}.")
    to_be_deleted = input("Do you want to delete the used files? [y/n] ")
    if to_be_deleted == "y":
        delete_files(used_files)
    sys.exit()
