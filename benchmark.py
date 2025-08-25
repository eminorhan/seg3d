import numpy as np
import zarr
import time
import os

def run_benchmark():
    """
    Runs the benchmark to compare reading from a zarr array vs. a numpy array in RAM.
    """
    # --- 1. SETUP: Create a Zarr array for testing ---
    print("Step 1: Setting up Zarr array...")
    
    # Define benchmark parameters
    num_samples = 1
    crop_shape = (256, 256, 256)

    # Define the directory for the Zarr array
    zarr_path = "data/jrc_mus-granule-neurons-1/jrc_mus-granule-neurons-1.zarr/recon-2/em/fibsem-int16/s0"

    # --- 2. BENCHMARK 1: Reading random crops directly from Zarr ---
    print("\nStep 2: Benchmarking direct reads from Zarr (on disk)...")
    z_array_read = zarr.open(zarr_path, mode='r')
    array_shape = z_array_read.shape
    
    zarr_read_times = []
    
    start_time_zarr_total = time.time()
    for i in range(num_samples):
        # Generate random starting coordinates for the crop
        x_start = np.random.randint(0, array_shape[0] - crop_shape[0] + 1)
        y_start = np.random.randint(0, array_shape[1] - crop_shape[1] + 1)
        z_start = np.random.randint(0, array_shape[2] - crop_shape[2] + 1)

        # Slice the Zarr array to get the crop
        start_time_zarr_sample = time.time()
        crop = z_array_read[
            x_start : x_start + crop_shape[0],
            y_start : y_start + crop_shape[1],
            z_start : z_start + crop_shape[2]
        ]
        end_time_zarr_sample = time.time()
        zarr_read_times.append(end_time_zarr_sample - start_time_zarr_sample)

    end_time_zarr_total = time.time()
    total_zarr_read_time = end_time_zarr_total - start_time_zarr_total

    # --- 3. BENCHMARK 2: Loading to RAM first, then reading from NumPy ---
    print("\nStep 3: Benchmarking reads from numpy (in RAM)...")

    # Time how long it takes to load the entire Zarr array into RAM
    start_time_load_ram = time.time()
    try:
        np_array = np.asarray(z_array_read)
        end_time_load_ram = time.time()
        load_ram_time = end_time_load_ram - start_time_load_ram
        print(f"Time to load entire array into RAM: {load_ram_time:.4f} seconds")

        numpy_read_times = []
        
        start_time_numpy_total = time.time()
        for i in range(num_samples):
            # Generate random starting coordinates for the crop
            x_start = np.random.randint(0, array_shape[0] - crop_shape[0] + 1)
            y_start = np.random.randint(0, array_shape[1] - crop_shape[1] + 1)
            z_start = np.random.randint(0, array_shape[2] - crop_shape[2] + 1)
    
            # Slice the NumPy array to get the crop
            start_time_numpy_sample = time.time()
            crop = np_array[
                x_start : x_start + crop_shape[0],
                y_start : y_start + crop_shape[1],
                z_start : z_start + crop_shape[2]
            ]
            end_time_numpy_sample = time.time()
            numpy_read_times.append(end_time_numpy_sample - start_time_numpy_sample)

        end_time_numpy_total = time.time()
        total_numpy_read_time_only = end_time_numpy_total - start_time_numpy_total
        total_numpy_read_time_with_load = total_numpy_read_time_only + load_ram_time
    
    except MemoryError:
        print("MemoryError: The array is too large to fit in RAM for this benchmark.")
        total_numpy_read_time_with_load = float('inf')
        total_numpy_read_time_only = float('inf')
    
    # --- 4. RESULTS ---
    print("\nStep 4: Displaying results...")
    
    print("\n--- Benchmark Summary ---")
    print(f"Number of samples: {num_samples}")
    print(f"Crop shape: {crop_shape}")
    print(f"Zarr array shape: {array_shape}")

    # Results from zarr (on disk)
    avg_zarr_read_time = np.mean(zarr_read_times)
    print(f"\nScenario 1: Reading crops directly from zarr (on disk)")
    print(f"  Total time for all {num_samples} samples: {total_zarr_read_time:.4f} seconds")
    print(f"  Average time per sample: {avg_zarr_read_time:.6f} seconds")

    # Results from numpy (in RAM)
    if total_numpy_read_time_only != float('inf'):
        avg_numpy_read_time = np.mean(numpy_read_times)
        print(f"\nScenario 2: Loading entire array into RAM, then reading crops")
        print(f"  Time to load entire array into RAM: {load_ram_time:.4f} seconds")
        print(f"  Total time for {num_samples} samples (from RAM): {total_numpy_read_time_only:.4f} seconds")
        print(f"  Average time per sample (from RAM): {avg_numpy_read_time:.6f} seconds")
        print(f"  Total time for Scenario 2 (load + samples): {total_numpy_read_time_with_load:.4f} seconds")
    else:
        print("\nScenario 2: Skipped due to MemoryError (array is too large to fit in RAM).")
    
if __name__ == '__main__':
    run_benchmark()
