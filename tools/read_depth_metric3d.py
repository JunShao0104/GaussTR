import numpy as np

# Replace with the actual filename you want to inspect
file_path = "/scratch/shared/dataset/GaussTR/nuscenes_metric3d/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151432187558.npy"

# Load the .npy file
depth = np.load(file_path)

# Print shape and basic stats
print("Shape:", depth.shape) # Shape: (900, 1600)
print("Dtype:", depth.dtype) # Dtype: float32
print("Min value:", np.min(depth)) # Min value: 2.8132453
print("Max value:", np.max(depth)) # Max value: 105.98651
print("Mean value:", np.mean(depth)) # Mean value: 31.18067
print("Number of NaNs:", np.isnan(depth).sum()) # Number of NaNs: 0