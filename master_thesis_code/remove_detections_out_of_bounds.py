import pandas as pd
import numpy as np
import sys

from master_thesis_code.cosmological_model import Detection
from master_thesis_code.constants import (
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
)


if __name__ == "__main__":
    with open(CRAMER_RAO_BOUNDS_OUTPUT_PATH, "r") as f:
        print(
            f"Reading cramer rao bounds and detection parameters from {CRAMER_RAO_BOUNDS_OUTPUT_PATH}"
        )
        cramer_rao_bounds = pd.read_csv(f)
    unusable_detections = []
    for detection_index, detection in cramer_rao_bounds.iterrows():
        print(f"progess: {detection_index}/{len(cramer_rao_bounds)}")
        detection = Detection(detection)
        if (detection.M <= 10**4) or (detection.M >= 10**6):
            print(f"unusable M: {detection.M}")
            unusable_detections.append(detection_index)
            continue
        if (detection.d_L <= 0.0) or (detection.d_L >= 5 * 10**3):
            print(f"unusable d_L: {detection.d_L}")
            unusable_detections.append(detection_index)
            continue

    if len(unusable_detections) > 0:
        print(f"{len(unusable_detections)} detections had to be deleted")
        cramer_rao_bounds.drop(unusable_detections, inplace=True)
        cramer_rao_bounds.to_csv(CRAMER_RAO_BOUNDS_OUTPUT_PATH, index=False)

    print(f"Saved reduced cramer rao bounds to {CRAMER_RAO_BOUNDS_OUTPUT_PATH}")
    sys.exit("Finished.")
