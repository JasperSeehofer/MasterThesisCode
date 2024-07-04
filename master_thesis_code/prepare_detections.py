import pandas as pd
import numpy as np
import sys

from master_thesis_code.cosmological_model import Detection
from master_thesis_code.constants import (
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    PREPARED_CRAMER_RAO_BOUNDS_PATH,
)


if __name__ == "__main__":
    with open(CRAMER_RAO_BOUNDS_OUTPUT_PATH, "r") as f:
        print(
            f"Reading cramer rao bounds and detection parameters from {CRAMER_RAO_BOUNDS_OUTPUT_PATH}"
        )
        cramer_rao_bounds = pd.read_csv(f)
    prepared_cramer_rao_bounds = cramer_rao_bounds.copy()
    unusable_detections = []
    print("Converting detection parameters to best guess parameters")
    for detection_index, detection in cramer_rao_bounds.iterrows():
        print(f"progess: {detection_index}/{len(cramer_rao_bounds)}")
        if detection["delta_phiS_delta_phiS"] < 0:
            unusable_detections.append(detection_index)
            continue
        detection = Detection(detection)
        detection.convert_to_best_guess_parameters()
        prepared_cramer_rao_bounds.at[detection_index, "M"] = detection.M
        prepared_cramer_rao_bounds.at[detection_index, "dist"] = detection.d_L
        prepared_cramer_rao_bounds.at[detection_index, "phiS"] = detection.phi
        prepared_cramer_rao_bounds.at[detection_index, "qS"] = detection.theta

    if len(unusable_detections) > 0:
        print(f"{len(unusable_detections)} detections had to be deleted")
        cramer_rao_bounds.drop(unusable_detections, inplace=True)
        prepared_cramer_rao_bounds.drop(unusable_detections, inplace=True)
        cramer_rao_bounds.to_csv(CRAMER_RAO_BOUNDS_OUTPUT_PATH, index=False)

    prepared_cramer_rao_bounds.to_csv(PREPARED_CRAMER_RAO_BOUNDS_PATH, index=False)
    print(f"Saved prepared cramer rao bounds to {PREPARED_CRAMER_RAO_BOUNDS_PATH}")
    sys.exit("Finished.")
