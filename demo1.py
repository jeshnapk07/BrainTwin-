import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import subprocess

# 📂 Folder path containing HDF5 files
folder_path = "C:/Users/jeshn/OneDrive/Documents/BrainTwin Copy/Brats/"

# 🔍 Get list of all .h5 files
h5_files = sorted(glob.glob(os.path.join(folder_path, "*.h5")))

if not h5_files:
    print(f"Error: No HDF5 files found in '{folder_path}'.")
    exit()

print(f"Found {len(h5_files)} HDF5 files. Processing the first {min(16, len(h5_files))}.")

# 📈 Set up a 4x4 grid for MRI modalities
fig, axes = plt.subplots(4, 4, figsize=(20, 10))  

for i, file_path in enumerate(h5_files[:16]):  
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

                row, col = divmod(i, 4)
                for j in range(4):
                    axes[row, col * 4 + j].imshow(mri_image_8bit[:, :, j], cmap="gray", vmin=0, vmax=255)
                    axes[row, col * 4 + j].set_title(f"{modality_names[j]} MRI {i+1}")
                    axes[row, col * 4 + j].axis("off")

                # Save MRI images for further processing
                os.makedirs("processed_mri", exist_ok=True)
                for j, name in enumerate(modality_names):
                    img_path = f"processed_mri/MRI_{i+1}_{name}.png"
                    cv2.imwrite(img_path, mri_image_8bit[:, :, j])
            else:
                print("Error: 'image' dataset not found in the HDF5 file.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

plt.tight_layout()
plt.show()

# 🔹 Tumor Segmentation using MedSAM
print("\n🔹 Performing Tumor Segmentation...")
sam_checkpoint_path = "C:/Users/jeshn/OneDrive/Documents/BrainTwin Copy/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
mask_generator = SamAutomaticMaskGenerator(sam)

os.makedirs("tumor_masks", exist_ok=True)

# Process all images in the processed_mri folder
mri_images = sorted(glob.glob("processed_mri/*.png"))

for i, mri_path in enumerate(mri_images):
    if not os.path.exists(mri_path):
        print(f"Error: File {mri_path} not found.")
        continue
    
    mri_data = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
    if mri_data is None:
        print(f"Error: Unable to read image {mri_path}. Skipping...")
        continue
    
    mri_data = cv2.cvtColor(mri_data, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    
    try:
        tumor_mask = mask_generator.generate(mri_data)
        
        # Save segmented tumor mask
        mask_filename = os.path.basename(mri_path).replace(".png", "_mask.png")
        mask_path = os.path.join("tumor_masks", mask_filename)
        cv2.imwrite(mask_path, tumor_mask[0]['segmentation'] * 255)
        print(f"✅ Tumor Mask saved: {mask_path}")
        
        # Display the segmented mask
        plt.figure(figsize=(5, 5))
        plt.imshow(tumor_mask[0]['segmentation'], cmap="gray")
        plt.title(f"Tumor Mask {os.path.basename(mri_path)}")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Error generating tumor mask for {mri_path}: {e}")