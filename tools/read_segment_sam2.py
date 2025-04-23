import numpy as np

# Replace with the actual filename you want to inspect
file_path = "/scratch/shared/dataset/GaussTR/nuscenes_grounded_sam2/n008-2018-07-26-12-13-50-0400__CAM_FRONT_RIGHT__1532621922620482.npy"

# Load the .npy file
segment = np.load(file_path)

# Print shape and basic stats
print("Shape:", segment.shape)
print("Dtype:", segment.dtype)
print("Min value:", np.min(segment))
print("Max value:", np.max(segment))
print("Mean value:", np.mean(segment))
print("Number of NaNs:", np.isnan(segment).sum())
# Shape: (900, 1600)
# Dtype: float32
# Min value: 0.0
# Max value: 16.0
# Mean value: 6.051571
# Number of NaNs: 0
