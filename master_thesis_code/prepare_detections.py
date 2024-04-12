import pandas as pd
import numpy as np
import sys

from master_thesis_code.cosmological_model import Detection
from master_thesis_code.constants import CRAMER_RAO_BOUNDS_OUTPUT_PATH, PREPARED_CRAMER_RAO_BOUNDS_PATH


if __name__ == '__main__':
    with open(CRAMER_RAO_BOUNDS_OUTPUT_PATH, 'r') as f:
        print(f"Reading cramer rao bounds and detection parameters from {CRAMER_RAO_BOUNDS_OUTPUT_PATH}")
        cramer_rao_bounds = pd.read_csv(f)

    print("Converting detection parameters to best guess parameters")
    for detection_index, detection in cramer_rao_bounds.iterrows():
        print(f"progess: {detection_index}/{len(cramer_rao_bounds)}")
        detection = Detection(detection)
        detection.convert_to_best_guess_parameters()
        cramer_rao_bounds.at[detection_index, 'M'] = detection.M
        cramer_rao_bounds.at[detection_index, 'dist'] = detection.d_L
        cramer_rao_bounds.at[detection_index, 'phiS'] = detection.phi
        cramer_rao_bounds.at[detection_index, 'qS'] = detection.theta

    cramer_rao_bounds.to_csv(PREPARED_CRAMER_RAO_BOUNDS_PATH, index=False)
    print(f"Saved prepared cramer rao bounds to {PREPARED_CRAMER_RAO_BOUNDS_PATH}")
    sys.exit("Finished.")