import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from diffusers import StableDiffusionPipeline
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import nibabel as nib
import cv2


# 📂 Folder path containing HDF5 files
folder_path = "C:/Users/jeshn/OneDrive/Documents/BrainTwin Copy/brats2/"

# 🔍 Get list of all .h5 files
h5_files = sorted(glob.glob(os.path.join(folder_path, "*.h5")))

if not h5_files:
    print(f"Error: No HDF5 files found in '{folder_path}'.")
    exit()

print(f"Found {len(h5_files)} HDF5 files. Processing all available files.")

# 📈 Set up a grid for MRI modalities
fig, axes = plt.subplots(len(h5_files), 4, figsize=(20, 10 * len(h5_files)))

os.makedirs("processed_mri", exist_ok=True)

for i, file_path in enumerate(h5_files):  
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\nProcessing file: {file_path}")
            print("Keys in HDF5 file:", list(f.keys()))

            if "image" in f:
                mri_image = f["image"][:]  # Shape: (240, 240, 4)

                print(f"Shape of 'image': {mri_image.shape}")

                # Normalize MRI images (0-1 range)
                if np.max(mri_image) > 1:
                    mri_image = (mri_image - np.min(mri_image)) / (np.max(mri_image) - np.min(mri_image))

                # Convert to 8-bit
                mri_image_8bit = (mri_image * 255).astype(np.uint8)

                # 🎯 Extract all four MRI modalities
                modality_names = ["T1", "T1ce", "T2", "FLAIR"]

                for j, name in enumerate(modality_names):
                    img_path = f"processed_mri/MRI_{i+1}_{name}.png"
                    cv2.imwrite(img_path, mri_image_8bit[:, :, j])

            else:
                print("Error: 'image' dataset not found in the HDF5 file.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

plt.tight_layout()
plt.show()

# 🔹 Step 1: MRI Enhancement using Stable Diffusion + ControlNet
print("\n🔹 Enhancing MRI Scans using Stable Diffusion...")
os.makedirs("enhanced_mri", exist_ok=True)

model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
model.to("cuda")

# Get all MRI files in processed_mri directory
mri_files = sorted([f for f in os.listdir("processed_mri") if f.endswith(".png")])

for i, mri_file in enumerate(mri_files, start=1):
    input_mri_path = os.path.join("processed_mri", mri_file)
    enhanced_mri = model(input_mri_path).images[0]
    
    enhanced_mri_path = os.path.join("enhanced_mri", f"Enhanced_{mri_file}")
    enhanced_mri.save(enhanced_mri_path)
    
    print(f"✅ Enhanced MRI saved: {enhanced_mri_path}")
