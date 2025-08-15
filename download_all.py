import s3fs
import os
import time
import quilt3 as q3
import concurrent.futures

# --- Configuration ---
# The number of downloads to run in parallel.
MAX_WORKERS = 32
# --------------------

def download_zarr_archive(item, bucket, root_dir):
    """The target function for each thread with a manual retry loop."""
    max_retries = 5
    dataset_prefix = item['Prefix']
    dataset_name = dataset_prefix.strip('/')
    s3_key = f"{dataset_prefix}{dataset_name}.zarr"
    local_dest_path = os.path.join(root_dir, dataset_name)

    for attempt in range(max_retries):
        try:
            print(f"-> Attempt {attempt + 1}/{max_retries} for {s3_key}")
            bucket.fetch(s3_key, local_dest_path)
            # If fetch succeeds, return the success message and exit the loop
            return f"✅ Success:   Downloaded {s3_key}"
        except Exception as e:
            print(f"   -> Attempt {attempt + 1} failed: {e}")
            # If this wasn't the last attempt, wait before retrying
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1) # Exponential backoff: 2s, 4s, 8s...
                print(f"   -> Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                # All retries have failed
                return f"❌ Failed:    Could not download {s3_key} after {max_retries} attempts."
    return f"❌ Failed:    An unexpected error occurred with {s3_key}." # Should not be reached


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