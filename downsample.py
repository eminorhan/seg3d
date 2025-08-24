import zarr
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import sys

# --- Configuration ---
# Path to your large numpy array
zarr_path = "data/jrc_mus-granule-neurons-1/jrc_mus-granule-neurons-1.zarr/recon-2/em/fibsem-int16/s0"

# Path for the output numpy file
numpy_file_path = "resized_array.npy"

# Define the downsampling factor. For a factor of k, the new array will be 1/k the size in each dimension.
downsample_factor = 8

def worker_function(args):
    """
    This function processes a single 'chunk' or slice of the data.
    It is designed to run in a separate process.

    Args:
        args (tuple): A tuple containing the zarr_path, the index of the slice to process, the new_shape, and the downsample_factor.
    """
    try:
        zarr_path, i, new_shape, downsample_factor = args
        
        # Each worker process opens its own read-only reference to the Zarr array.
        zarr_array = zarr.open(zarr_path, mode='r')

        # Create a 2D array to hold the processed slice.
        output_slice = np.zeros(new_shape[1:], dtype=zarr_array.dtype)

        # Iterate through the y and z dimensions for the current x slice (i).
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                # Determine the start and end coordinates of the block in the original array.
                start_x = i * downsample_factor
                end_x = (i + 1) * downsample_factor
                start_y = j * downsample_factor
                end_y = (j + 1) * downsample_factor
                start_z = k * downsample_factor
                end_z = (k + 1) * downsample_factor

                # Read a small, manageable block of data from the Zarr array.
                block = zarr_array[start_x:end_x, start_y:end_y, start_z:end_z]

                # Perform the downsampling (e.g., calculate the mean of the block).
                downsampled_value = np.mean(block)

                # Store the downsampled value in the output slice.
                output_slice[j, k] = downsampled_value

        # The worker returns the completed slice.
        return i, output_slice

    except Exception as e:
        print(f"Error in worker process for slice {i}: {e}", file=sys.stderr)
        return i, None


if __name__ == '__main__':
    print(f"Opening Zarr array from: {zarr_path}")

    try:
        # Open the Zarr array to get its shape dynamically.
        zarr_array = zarr.open(zarr_path, mode='r')
        original_shape = zarr_array.shape
        print(f"Zarr array basic info: {zarr_array.info}")
        z = zarr_array[:]
        print(f"Zarr array min: {np.min(z[:])}")
        print(f"Zarr array max: {np.max(z[:])}")

        # Calculate the new shape based on the original shape and downsample factor.
        # This uses integer division, which will truncate any remainder.
        new_shape = (
            original_shape[0] // downsample_factor,
            original_shape[1] // downsample_factor,
            original_shape[2] // downsample_factor
        )

        # Add a check for perfect divisibility and warn the user if it's not.
        if not (original_shape[0] % downsample_factor == 0 and
                original_shape[1] % downsample_factor == 0 and
                original_shape[2] % downsample_factor == 0):
            print(f"Warning: Original dimensions {original_shape} are not perfectly divisible by the downsample factor {downsample_factor}.")
            print(f"The resulting shape will be {new_shape}, which may truncate data.")

        print(f"Resizing from {original_shape} to {new_shape} using multiprocessing...")
        
        # Determine the number of worker processes to use.
        num_processes = cpu_count()
        print(f"Using {num_processes} worker processes.")

        # Create a list of arguments for the worker function. Each tuple contains the zarr_path,
        # the index of the slice (i) to process, the new shape, and the downsample factor.
        tasks = [(zarr_path, i, new_shape, downsample_factor) for i in range(new_shape[0])]

        # Create an empty NumPy array of the target size. This array will be built piece by piece.
        output_array = np.zeros(new_shape, dtype=zarr_array.dtype)

        # Use a Pool of worker processes to execute the tasks in parallel.
        with Pool(processes=num_processes) as pool:
            results = pool.imap_unordered(worker_function, tasks)
            
            # Iterate through the results and place each completed slice in the final array.
            for i, result_slice in results:
                if result_slice is not None:
                    output_array[i, :, :] = result_slice
                    print(f"Finished processing slice {i+1}/{new_shape[0]}...")

        print("Resizing complete. Saving to NumPy file...")

        # Save the final, downsampled NumPy array to a file.
        np.save(numpy_file_path, output_array)

        print(f"Successfully saved the resized array to: {numpy_file_path}")

    except zarr.errors.GroupNotFoundError:
        print(f"Error: The Zarr array at '{zarr_path}' could not be found.")
        print("Please check that the path is correct and the Zarr directory exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
