import s3fs
import os
import time
import concurrent.futures

# --- Configuration ---
MAX_WORKERS = 32
MAX_DOWNLOAD_ATTEMPTS = 5 # Maximum number of times to try downloading a single dataset
BUCKET_NAME = 'janelia-cosem-datasets'
# --------------------

def download_zarr_archive(s3_prefix_path, s3_filesystem, root_dir):
    """
    The target function for each thread, with a manual retry loop.
    s3_prefix_path is the full S3 path to the dataset directory,
    e.g., 'janelia-cosem-datasets/jrc_cos7-1a'
    """
    # Extract the dataset name from the end of the prefix path
    dataset_name = s3_prefix_path.split('/')[-1]

    # Construct the full S3 source path for the .zarr archive
    s3_source_path = f"{s3_prefix_path}/{dataset_name}.zarr"

    # Construct the final local destination path
    local_dest_path = os.path.join(root_dir, dataset_name, f"{dataset_name}.zarr")

    # --- START OF CHANGE ---
    # First, check if the Zarr archive actually exists before trying to download it.
    if not s3_filesystem.exists(s3_source_path):
        return f"🟡 Skipped:   {s3_source_path} does not exist."
    # --- END OF CHANGE ---

    # Manual retry loop
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            print(f"-> Attempt {attempt + 1}/{MAX_DOWNLOAD_ATTEMPTS} for {s3_source_path}")

            # Create the parent directory for the download if it doesn't exist
            os.makedirs(os.path.dirname(local_dest_path), exist_ok=True)

            # Perform the recursive download using s3fs.get
            s3_filesystem.get(s3_source_path, local_dest_path, recursive=True)

            # If the download succeeds, return the success message and exit the loop
            return f"✅ Success:   Downloaded {s3_source_path}"

        except Exception as e:
            print(f"   -> Attempt {attempt + 1} failed: {e}")

            # If this wasn't the last attempt, wait before retrying
            if attempt < MAX_DOWNLOAD_ATTEMPTS - 1:
                wait_time = 2 ** (attempt + 1)  # Exponential backoff: 2s, 4s, 8s...
                print(f"   -> Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                # All retries have failed, return the final error message
                return f"❌ Failed:    Could not download {s3_source_path} after {MAX_DOWNLOAD_ATTEMPTS} attempts."

    # This line should not be reached but is included for completeness
    return f"❌ Failed:    An unexpected error occurred with {s3_source_path}."


# --- Main Script ---
if __name__ == "__main__":
    # 1. Initialize the S3 file system for anonymous access
    s3 = s3fs.S3FileSystem(anon=True)

    # 2. Define and create the local root directory
    local_root_dir = "data"
    os.makedirs(local_root_dir, exist_ok=True)

    # 3. List all top-level datasets (directories) using s3fs
    print(f"Finding datasets in the bucket '{BUCKET_NAME}'...")
    try:
        all_objects = s3.ls(BUCKET_NAME, detail=True)
        # Filter the list to include only directories
        prefix_paths = [obj['name'] for obj in all_objects if obj['type'] == 'directory']
        print(f"Found {len(prefix_paths)} total datasets to process.")
        print(f"List of datasets to be downloaded: {prefix_paths}")
    except Exception as e:
        print(f"Could not list datasets in bucket. Error: {e}")
        prefix_paths = []

    print("-" * 50)
    print(f"Starting parallel download with up to {MAX_WORKERS} workers...")
    print("-" * 50)

    # 4. Use ThreadPoolExecutor to manage parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Pass the s3 filesystem object to each worker thread
        futures = [executor.submit(download_zarr_archive, path, s3, local_root_dir) for path in prefix_paths]
        for future in concurrent.futures.as_completed(futures):
            status_message = future.result()
            print(status_message)

    print("-" * 50)
    print("All download attempts are complete.")