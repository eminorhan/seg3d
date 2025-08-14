import s3fs
import os
import time
import quilt3 as q3
import concurrent.futures

# --- Configuration ---
# The number of downloads to run in parallel.
# Adjust this value based on your network bandwidth and CPU capabilities.
# A good starting point is typically between 5 and 10.
MAX_WORKERS = 144
# --------------------

def download_zarr_archive(item, bucket, root_dir):
    """
    The target function for each thread.
    It contains the logic to download a single Zarr archive.
    """
    try:
        dataset_prefix = item['Prefix']  # e.g., 'jrc_cos7-1a/'
        dataset_name = dataset_prefix.strip('/')  # e.g., 'jrc_cos7-1a'
        s3_key = f"{dataset_prefix}{dataset_name}.zarr" # e.g., 'jrc_cos7-1a/jrc_cos7-1a.zarr'
        local_dest_path = os.path.join(root_dir, dataset_name) # e.g., 'data/jrc_cos7-1a'

        # This is the actual download operation
        bucket.fetch(s3_key, local_dest_path)
        
        # Return a success message for printing later
        return f"✅ Success:   Downloaded {s3_key}"
    except Exception as e:
        # If an error occurs, return a formatted error message
        return f"❌ Failed:    Could not download {s3_key}. Reason: {e}"

# --- Main Script ---
if __name__ == "__main__":
    # 1. Initialize the S3 bucket
    b = q3.Bucket("s3://janelia-cosem-datasets")

    # 2. Define and create the local root directory
    local_root_dir = "data"
    os.makedirs(local_root_dir, exist_ok=True)

    # 3. List all top-level datasets
    print("Finding datasets in the bucket...")
    try:
        prefixes_list = b.ls()[0]
        print(f"Found {len(prefixes_list)} total datasets to process.")
    except IndexError:
        print("No prefixes (directories) found at the root of the bucket.")
        prefixes_list = []

    print("-" * 50)
    print(f"Starting parallel download with up to {MAX_WORKERS} workers...")
    print("-" * 50)

    # 4. Use ThreadPoolExecutor to manage parallel downloads
    # The 'with' statement ensures that all threads are joined before the script exits.
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each download task to the executor. This is non-blocking.
        # It creates a list of 'Future' objects, which represent the execution of each task.
        futures = [executor.submit(download_zarr_archive, item, b, local_root_dir) for item in prefixes_list]

        # Use 'as_completed' to get results as soon as they are available.
        # This will print the status of each download as it finishes, regardless of order.
        for future in concurrent.futures.as_completed(futures):
            # .result() gets the return value from the 'download_zarr_archive' function
            status_message = future.result()
            print(status_message)

    print("-" * 50)
    print("All download attempts are complete.")