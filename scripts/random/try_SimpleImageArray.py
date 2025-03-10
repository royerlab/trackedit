#******************
wd = "/hpc/projects/jacobo_group/iSim_processed_files/steady_state_timelapses/20241003_2dpf_myo6b_bactin_GFP_she_h2b_gfp_cldnb_lyn_mScarlet/46hpf_fish1_1/2_deconvolution_and_registration/" 
zarr_file = "deconvolved_and_registered_corrected_cropped850.zarr"
channel = '0/4/0/0'
#******************


# Import the class (adjust import path as needed)
from trackedit.arrays.ImagingArray import SimpleImageArray
import napari

# Initialize with your existing data
image_data = SimpleImageArray(imaging_zarr_file=wd+zarr_file, channel=channel)

# # Print initial shape (should be full dataset)
print("Full dataset shape:", image_data.shape)

# # Set a time window of 100 frames
image_data.time_window = (100, 200)
print("Windowed shape:", image_data.shape)

# # Create viewer with windowed data
viewer = napari.Viewer()
# viewer.add_image(image_data.membrane,
#                 name='membrane',
#                 colormap='red')
# viewer.add_image(image_data.nuclear,
#                 name='nucleus',
#                 opacity=0.5,
#                 colormap='green')