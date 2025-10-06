import os
import random
import time
import zarr
import numpy as np
import matplotlib.pyplot as plt

def get_random_crop(
    zarr_path: str,
    crop_size: tuple[int, int, int],
    resolution: str = "s0",
) -> np.ndarray | None:
    """
    Selects a random volume from the data directory, opens its Zarr array,
    and extracts a random 3D crop.

    The expected directory structure for each volume is:
    <data_dir>/<volume_name>/<volume_name>.zarr/recon-1/em/fibsem-uint8/s0

    Args:
        data_dir: The path to the top-level directory containing volume folders.
        subdir_name: Subdirectory name containing the EM data.
        crop_size: A tuple of (depth, height, width) for the desired crop.
        resolution: Resolution at which to retrieve the data (default: 's0').

    Returns:
        A NumPy array containing the cropped data, or None if an error occurs.
    """    
    try:
        # Open the Zarr array without loading it into memory
        # We open the group first and then access the dataset at the requested resolution. 
        # If 'fibsem-uint8' is the array itself, zarr.open
        # would return an array object directly. This approach is more robust.
        print(f"Opening Zarr store at: {zarr_path}")
        zarr_group = zarr.open(zarr_path, mode='r')
        print(f"Available resolutions: {list(zarr_group.keys())}")

        # Assuming the highest resolution data is at scale 's0'
        if resolution not in zarr_group:
            print(f"Error: Could not find {resolution} dataset in '{zarr_path}'.")
            print(f"Available resolutions: {list(zarr_group.keys())}")
            return None
            
        zarr_array = zarr_group[resolution]
        full_shape = zarr_array.shape
        print(f"Full array shape: {full_shape}")
        
        # Determine the coordinates for a random crop
        cz, cy, cx = crop_size
        
        # Ensure the crop size is not larger than the full array
        if any(c > f for c, f in zip(crop_size, full_shape)):
             print("Error: Crop size is larger than the array dimensions.")
             return None

        # Calculate the maximum possible starting index for the crop in each dimension
        max_z = full_shape[0] - cz
        max_y = full_shape[1] - cy
        max_x = full_shape[2] - cx
        
        # Generate a random starting point
        start_z = random.randint(0, max_z)
        start_y = random.randint(0, max_y)
        start_x = random.randint(0, max_x)
        
        print(f"Extracting crop of size {crop_size} from starting coordinate: {(start_z, start_y, start_x)}")

        # Read the specific crop from the Zarr array into a NumPy array
        # This is the step where the data is actually read from disk. Zarr is
        # optimized to only read the chunks necessary to fulfill this slice.
        crop_slice = (
            slice(start_z, start_z + cz),
            slice(start_y, start_y + cy),
            slice(start_x, start_x + cx)
        )
        numpy_crop = zarr_array[crop_slice]
        
        return numpy_crop

    except FileNotFoundError:
        print(f"Error: The specified path or a part of it was not found. Check the path: {zarr_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def visualize_crop(crop: np.ndarray, filename: str, num_slices: int = 36):
    """
    Visualizes a 3D crop by displaying a grid of 2D slices.

    Args:
        crop: The 3D NumPy array to visualize.
        num_slices: The total number of slices to display in the grid (should be a perfect square).
    """
    depth = crop.shape[0]
    # determine the grid size (e.g., 4x4 for 16 slices)
    grid_size = int(np.sqrt(num_slices))
    if grid_size**2 != num_slices:
        print("Warning: num_slices is not a perfect square. Visualization may be incomplete.")
        
    # calculate the indices of the slices to show
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle('Regularly Spaced Slices from the Selected Crop', fontsize=16)

    for i, ax in enumerate(axes.flat):
        slice_idx = slice_indices[i]
        img_slice = crop[slice_idx, :, :]
        
        ax.imshow(img_slice, cmap='gray', interpolation='nearest')
        ax.set_title(f'Slice: {slice_idx}')
        ax.axis('off') # hide axes ticks and labels for a cleaner look

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # save figure to a file
    try:
        plt.savefig(filename, bbox_inches='tight')
        print(f"\nVisualization saved to: {filename}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    finally:
        plt.close(fig)


if __name__ == '__main__':
    # --- configuration ---
    ZARR_PATH = "data/jrc_ut21-1413-003/jrc_ut21-1413-003.zarr/recon-1/labels/groundtruth/crop191/all"
    CROP_SIZE = (256, 256, 256)
    RESOLUTION = "s0"
    # --- configuration ---

    # --- crop ---
    s_time = time.time()
    random_crop = get_random_crop(ZARR_PATH, CROP_SIZE, RESOLUTION)
    e_time = time.time()
    print(f"Crop time: {e_time - s_time} seconds")

    if random_crop is not None:
        print("\n--- Success! ---")
        print(f"Returned crop shape: {random_crop.shape}")
        print(f"Returned crop data type: {random_crop.dtype}")
        # you can now work with 'random_crop' as a regular numpy array
        print(f"Min value in crop: {np.min(random_crop)}")
        print(f"Max value in crop: {np.max(random_crop)}")
        print(f"Unique values: {np.unique(random_crop)}")

        # --- visualize crop ---
        output_filename = f"example_crop_label_visualization.jpeg"
        visualize_crop(random_crop, filename=output_filename)