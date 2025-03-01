import vtk
import numpy as np
import nibabel as nib

# 📂 Load MRI from NIfTI
nii_file = "brain_mri.nii.gz"
nii_img = nib.load(nii_file)
nii_data = nii_img.get_fdata().astype(np.float32)  # Ensure float32 type

# Normalize and scale data to [0, 255] (8-bit grayscale)
nii_data = 255 * (nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data))
nii_data = nii_data.astype(np.uint8)  # Convert to unsigned 8-bit

# 📏 Get image dimensions (Z, Y, X) and convert to (X, Y, Z) for VTK
depth, height, width = nii_data.shape  # NumPy order (Z, Y, X)
nii_data = np.transpose(nii_data, (2, 0, 1))  # Convert to (X, Z, Y) to fix orientation

# 🏗 Convert NumPy array to VTK image
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(width, depth, height)  # Set corrected dimensions
vtk_image.SetSpacing(1, 1, 1)  # Adjust based on actual voxel size

# 🔹 Create VTK data array (Unsigned Char for grayscale)
flat_data = nii_data.flatten(order="F")  # Fortran order ensures correct voxel arrangement
vtk_array = vtk.vtkUnsignedCharArray()
vtk_array.SetNumberOfValues(len(flat_data))

for i in range(len(flat_data)):
    vtk_array.SetValue(i, int(flat_data[i]))  # Assign values

vtk_image.GetPointData().SetScalars(vtk_array)

# 🔄 **Fix Orientation (Reslice)**
reslice = vtk.vtkImageReslice()
reslice.SetInputData(vtk_image)
reslice.SetResliceAxesDirectionCosines(1, 0, 0,  0, -1, 0,  0, 0, -1)  # Adjusted for proper 3D positioning
reslice.SetInterpolationModeToLinear()
reslice.Update()

# 🎨 Opacity Transfer Function (Grayscale)
opacity_tf = vtk.vtkPiecewiseFunction()
opacity_tf.AddPoint(0, 0.0)
opacity_tf.AddPoint(50, 0.2)
opacity_tf.AddPoint(100, 0.6)
opacity_tf.AddPoint(255, 1.0)

# 🎭 Volume Property (No Color Transfer Function)
volume_property = vtk.vtkVolumeProperty()
volume_property.SetScalarOpacity(opacity_tf)
volume_property.ShadeOn()

# 🎭 Volume rendering setup
mapper = vtk.vtkSmartVolumeMapper()
mapper.SetInputConnection(reslice.GetOutputPort())  # Use resliced data

# 🏗 Create volume actor
volume = vtk.vtkVolume()
volume.SetMapper(mapper)
volume.SetProperty(volume_property)

# 🎥 Renderer & window setup
renderer = vtk.vtkRenderer()
renderer.AddVolume(volume)
renderer.SetBackground(0, 0, 0)  # Black background
renderer.GetActiveCamera().Elevation(90)  # Adjust camera to correct 3D view
renderer.ResetCamera()  # Ensure correct positioning

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 🚀 Render the 3D MRI in Correct Orientation (Grayscale)
print("✅ Rendering 3D MRI with Correct Orientation (Grayscale)...")
render_window.Render()
interactor.Start()
