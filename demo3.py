import h5py
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import mayavi.mlab as mlab

# 📂 Folder path containing HDF5 files
folder_path = "C:/Users/jeshn/OneDrive/Documents/BrainTwin Copy/Brats/"

# 🔍 Get list of all .h5 files
h5_files = sorted(glob.glob(os.path.join(folder_path, "*.h5")))

if not h5_files:
    print(f"Error: No HDF5 files found in '{folder_path}'.")
    exit()

num_files_to_process = min(16, len(h5_files))
print(f"Found {len(h5_files)} HDF5 files. Processing the first {num_files_to_process}.")

# Function to visualize MRI data using Mayavi
def visualize_mri(mri_volume):
    try:
        mlab.figure(bgcolor=(0, 0, 0))
        if mri_volume.shape[-1] == 4:
            mri_volume = mri_volume[..., 0]  # Select first modality
        
        mri_volume = np.array(mri_volume, dtype=np.float32)  # Ensure NumPy array
        mlab.contour3d(mri_volume, contours=10, opacity=0.5)
        mlab.show()
    except Exception as e:
        print(f"Error during MRI visualization: {e}")

# Process the MRI data
for file_path in h5_files[:num_files_to_process]:
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\nProcessing file: {file_path}")
            print("Keys in HDF5 file:", list(f.keys()))

            if "image" in f:
                mri_image = f["image"][:]  # Shape: (240, 240, 4)
                print(f"Shape of 'image': {mri_image.shape}")

                # Select single channel if 4-channel image
                if mri_image.shape[-1] == 4:
                    mri_image = mri_image[..., 0]
                
                # Normalize MRI images (0-1 range)
                if np.max(mri_image) > 1:
                    mri_image = (mri_image - np.min(mri_image)) / (np.max(mri_image) - np.min(mri_image))
                
                visualize_mri(mri_image)
            else:
                print("Error: 'image' dataset not found in the HDF5 file.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
