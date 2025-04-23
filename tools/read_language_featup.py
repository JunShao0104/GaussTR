import numpy as np

# Replace with the actual filename you want to inspect
file_path = "/scratch/shared/dataset/GaussTR/nuscenes_featup/n008-2018-07-26-12-13-50-0400__CAM_FRONT_LEFT__1532621761654799.npy"

# Load the .npy file
featup = np.load(file_path)

# Print shape and basic stats
print("Shape:", featup.shape)
print("Dtype:", featup.dtype)
print("Min value:", np.min(featup))
print("Max value:", np.max(featup))
print("Mean value:", np.mean(featup))
print("Number of NaNs:", np.isnan(featup).sum())
# Shape: (512, 27, 48) 27*16=432, 48*16=768
# Dtype: float32
# Min value: -2.3621945
# Max value: 7.4038925
# Mean value: 0.010775001
# Number of NaNs: 0