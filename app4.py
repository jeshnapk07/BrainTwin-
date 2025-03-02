import streamlit as st
import os
from PIL import Image
import plotly.graph_objects as go
import tempfile
import shutil

# Function to create a simple 3D plot (replace with actual 3D model visualization logic later)
# def display_3d_model(model_file_path):
#     fig = go.Figure(data=[go.Mesh3d(
#         x=[1, 2, 3, 4, 5],  
#         y=[1, 2, 3, 4, 5],
#         z=[1, 2, 3, 4, 5],
#         opacity=0.5,
#         color='blue'
#     )])

#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(title='X'),
#             yaxis=dict(title='Y'),
#             zaxis=dict(title='Z')
#         ),
#         title="3D Model Example"
#     )
    
#     st.plotly_chart(fig)

# Function to simulate the processing of images to a 3D model
def generate_3d_model_from_images(image_files):
    # Here you would process the images with a photogrammetry tool or library
    # Simulate that a 3D model is generated
    st.write(f"Processing {len(image_files)} images to generate a 3D model...")

    # In practice, you might run an external script to process the images into a 3D model file.
    # For demonstration, we'll just create a temporary .obj file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmpfile:
        tmpfile.write(b"# 3D Model generated from images\n")
        tmpfile.write(b"v 0 0 0\n")  # Example vertex data
        tmpfile.write(b"v 1 0 0\n")
        tmpfile.write(b"v 0 1 0\n")
        tmpfile.write(b"f 1 2 3\n")  # Example face data
        tmpfile_path = tmpfile.name

    return tmpfile_path

# Streamlit UI
st.title("3D Model Creation from Images")
st.markdown("""
    Upload images taken from different angles of the same object. 
    These images will be used to generate a 3D model.
""")

# Upload multiple images
uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # Display thumbnails of the uploaded images
    st.subheader("Uploaded Images")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)

    # After images are uploaded, simulate the creation of a 3D model
    if st.button("Generate 3D Model"):
        model_file_path = generate_3d_model_from_images(uploaded_files)

        # st.write(f"3D model generated: {model_file_path}")
        
        # # Display the 3D model (here we use a simulated .obj file)
        # display_3d_model(model_file_path)

        # Optionally, allow users to download the generated model
        with open(model_file_path, "rb") as file:
            st.download_button(
                label="Download 3D Model (.obj)",
                data=file,
                file_name="generated_3d_model.obj",
                mime="application/octet-stream"
            )

        # Clean up the temporary file after use
        os.remove(model_file_path)
else:
    st.write("Please upload images to start the 3D model generation process.")
