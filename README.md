# BrainTwin-

This is a Generative AI-enhanced virtual brain model that enables neurosurgeons to visualize tumor locations and their relationships with critical structures using MRI scans. This solution enhances MRI scan quality using AI, automates tumor segmentation, and generates a 3D virtual brain model to simulate surgeries before they take place. By leveraging cutting-edge AI innovations such as Stable Diffusion + ControlNet for MRI enhancement and MedSAM (Segment Anything Model) for tumor segmentation, it aims to improve medical imaging, support pre-surgical planning, and enhance accuracy in tumor localization. Additionally, this model creates a 3D MRI representation from multiple 2D JPG slices using OpenCV for image processing, NumPy for stacking images into a 3D volume, and NiBabel for saving the model in NIfTI (.nii.gz) format, which can be viewed in medical imaging tools like ITK-SNAP. The .gz file is then rendered using VTK, allowing doctors and researchers to analyze MRI scans in a 3D format for better visualization and diagnosis.

MRI enhancement using Stable Diffusion works like a smart AI-powered photo editor for medical scans. Stable Diffusion, a type of AI that usually creates images, can be trained to clean up MRI scans by reducing noise, sharpening details, and making them clearer. To make sure the AI doesn’t change important medical details, control mechanisms like ControlNet act as guides, ensuring the image stays accurate while improving its quality. This helps doctors get a clearer view of scans for better diagnosis and treatment.  

MedSAM is a type of Generative AI because it can automatically "generate" segmentations (outlines) of organs, tumors, or other important areas in medical scans. Instead of just identifying or classifying images, it creates precise boundaries around these areas based on patterns it has learned from medical data.

It works like a smart assistant for doctors—just by pointing to a spot on an MRI or CT scan, MedSAM predicts and draws the correct shape of an organ or tumor, saving time and effort. This makes it a powerful AI tool for medical imaging.

This code creates a 3D MRI model from multiple 2D JPG slices using:
1️⃣ OpenCV – Reads and processes 2D MRI images.

2️⃣ NumPy – Stacks the 2D images into a 3D volume.

3️⃣ NiBabel – Saves the 3D model as a NIfTI (.nii.gz) file, which can be viewed in medical imaging tools like ITK-SNAP.

This helps doctors and researchers see and analyze MRI scans in 3D.

app4.py - streamlit page
