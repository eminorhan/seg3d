import s3fs
import os
import time
import concurrent.futures

# --- Configuration ---
MAX_WORKERS = 8  # Maximum number of simultaneous downloads
MAX_DOWNLOAD_ATTEMPTS = 5  # Maximum number of times to try downloading a single dataset
BUCKET_NAME = 'janelia-cosem-datasets'
# --------------------

def download_zarr_archive(s3_prefix_path, s3_filesystem, root_dir):
    """
    Downloads a Zarr archive from S3, gracefully resuming if partially downloaded.
    It verifies file integrity by checking local file sizes against remote ones.
    
    s3_prefix_path is the full S3 path to the dataset directory,
    e.g., 'janelia-cosem-datasets/jrc_cos7-1a'
    """
    # Extract the dataset name from the end of the prefix path
    dataset_name = s3_prefix_path.split('/')[-1]

    # Construct the full S3 source path and local destination path
    s3_source_path = f"{s3_prefix_path}/{dataset_name}.zarr"
    local_dest_path = os.path.join(root_dir, dataset_name, f"{dataset_name}.zarr")

    # First, check if the remote Zarr archive actually exists
    if not s3_filesystem.exists(s3_source_path):
        return f"🟡 Skipped:   {s3_source_path} does not exist."

    # --- START OF ROBUST RESUME LOGIC ---

    # 1. Get a list of all remote files with their details (including size).
    print(f"-> Verifying files for {dataset_name}...")
    try:
        # Use detail=True to get file size information
        remote_file_details = s3_filesystem.find(s3_source_path, detail=True)
    except Exception as e:
        return f"❌ Failed:    Could not list files in {s3_source_path}. Error: {e}"
        
    remote_files = list(remote_file_details.values())
    if not remote_files:
        return f"🟡 Skipped:   No files found in {s3_source_path}."

    # 2. Identify files that are missing or have incorrect sizes.
    missing_remote_files = []
    corresponding_local_files = []
    
    for remote_f_detail in remote_files:
        remote_f_path = remote_f_detail['name']
        remote_f_size = remote_f_detail['size']
        
        # Determine the equivalent local path for each remote file
        relative_path = remote_f_path[len(s3_source_path):].lstrip('/')
        local_f_path = os.path.join(local_dest_path, relative_path)
        
        # Check if the local file exists AND if its size matches the remote file.
        # This is the key change to handle partially downloaded files.
        if not os.path.exists(local_f_path) or os.path.getsize(local_f_path) != remote_f_size:
            missing_remote_files.append(remote_f_path)
            corresponding_local_files.append(local_f_path)
    
    # 3. If no files are missing or corrupted, we're done.
    if not missing_remote_files:
        return f"✅ Verified:  All {len(remote_files)} files for {dataset_name} are correct."
        
    print(f"-> Found {len(remote_files)} total files for {dataset_name}. "
          f"Downloading {len(missing_remote_files)} missing or incomplete files...")

    # Ensure all necessary local parent directories exist
    local_dirs = {os.path.dirname(p) for p in corresponding_local_files}
    for d in local_dirs:
        os.makedirs(d, exist_ok=True)
        
    # --- END OF ROBUST RESUME LOGIC ---

    # Manual retry loop for downloading the batch of missing files
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            print(f"   -> Attempt {attempt + 1}/{MAX_DOWNLOAD_ATTEMPTS} for {dataset_name}")

            s3_filesystem.get(missing_remote_files, corresponding_local_files)

            return f"✅ Success:   Downloaded {len(missing_remote_files)} files for {dataset_name}."

        except Exception as e:
            print(f"   -> Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_DOWNLOAD_ATTEMPTS - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"   -> Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                return (f"❌ Failed:    Could not download files for {dataset_name} "
                        f"after {MAX_DOWNLOAD_ATTEMPTS} attempts.")

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