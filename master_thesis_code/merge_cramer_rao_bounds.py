# script to merge the cramer rao bounds for the different simulations

import os
import sys
import numpy as np
import pandas as pd
from master_thesis_code.constants import CRAMER_RAO_BOUNDS_PATH, CRAMER_RAO_BOUNDS_OUTPUT_PATH



if __name__ == "__main__":
    # get directory of the script
    script_directory = os.path.dirname(os.path.realpath(CRAMER_RAO_BOUNDS_OUTPUT_PATH))
    files = os.listdir(script_directory)
    cramer_rao_bounds_files = [
        file for file in files if CRAMER_RAO_BOUNDS_PATH.split(".")[0].replace("$index", "") in file
    ]

    # check if there are any files to merge
    if len(cramer_rao_bounds_files) == 0:
        sys.exit("No cramer rao bounds files found.")
    
    # check for existing output file
    try:
        with open(CRAMER_RAO_BOUNDS_OUTPUT_PATH, "r") as f:
            pd.read_csv(f)
    except FileNotFoundError:
        # use first file as base
        cramer_rao_bounds = pd.read_csv(cramer_rao_bounds_files.pop(0))
    
    if len(cramer_rao_bounds_files) == 0:
        # save the file
        cramer_rao_bounds.to_csv(CRAMER_RAO_BOUNDS_OUTPUT_PATH, index=False)
        sys.exit("Only one cramer rao bounds file found.")
    else:
        print(f"Merging {len(cramer_rao_bounds_files)} cramer rao bounds files.")

    # merge all files
    for file in cramer_rao_bounds_files:
        cramer_rao_bounds = pd.concat([cramer_rao_bounds, pd.read_csv(file)])
    
    # save the file
    cramer_rao_bounds.to_csv(CRAMER_RAO_BOUNDS_OUTPUT_PATH, index=False)