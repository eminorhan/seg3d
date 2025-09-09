import s3fs
import os
import time
import concurrent.futures

# --- Configuration ---
MAX_WORKERS = 8  # Maximum number of simultaneous downloads
MAX_DOWNLOAD_ATTEMPTS = 5  # Maximum number of times to try downloading a single dataset
BUCKET_NAME = 'janelia-cosem-datasets'
# NEW: Define a file count threshold instead of a manual skip list
MAX_FILES_THRESHOLD = 10_000_000
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

    # Get a list of all remote files with their details (including size).
    print(f"-> Verifying files for {dataset_name}...")
    try:
        # Use detail=True to get file size information
        remote_file_details = s3_filesystem.find(s3_source_path, detail=True)
    except Exception as e:
        return f"❌ Failed:    Could not list files in {s3_source_path}. Error: {e}"
        
    remote_files = list(remote_file_details.values())
    if not remote_files:
        return f"🟡 Skipped:   No files found in {s3_source_path}."

    # Identify files that are missing or have incorrect sizes.
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
    
    # If no files are missing or corrupted, we're done.
    if not missing_remote_files:
        return f"✅ Verified:  All {len(remote_files)} files for {dataset_name} are correct."
        
    print(f"-> Found {len(remote_files)} total files for {dataset_name}. Downloading {len(missing_remote_files)} missing or incomplete files...")

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
                return (f"❌ Failed:    Could not download files for {dataset_name} after {MAX_DOWNLOAD_ATTEMPTS} attempts.")

    return f"❌ Failed:    An unexpected error occurred with {s3_source_path}."

# --- Main Script ---
if __name__ == "__main__":
    # 1. Initialize the S3 file system for anonymous access
    s3 = s3fs.S3FileSystem(anon=True)

    # 2. Define and create the local root directory
    local_root_dir = "data"
    os.makedirs(local_root_dir, exist_ok=True)

    # 3. List all top-level datasets (directories) using s3fs
    print(f"Finding all datasets in the bucket '{BUCKET_NAME}'...")
    try:
        all_objects = s3.ls(BUCKET_NAME, detail=True)
        # Filter the list to include only directories
        all_prefix_paths = [obj['name'] for obj in all_objects if obj['type'] == 'directory']
    except Exception as e:
        print(f"Could not list datasets in bucket. Error: {e}")
        all_prefix_paths = []

    # --- NEW: Filter datasets based on file count ---
    print("\n" + "-" * 50)
    print(f"Checking datasets against file limit ({MAX_FILES_THRESHOLD:,} files)...")
    prefix_paths_to_download = []
    for path in all_prefix_paths:
        dataset_name = path.split('/')[-1]
        s3_zarr_path = f"{path}/{dataset_name}.zarr"
        
        try:
            # Check if the .zarr archive exists before counting
            if not s3.exists(s3_zarr_path):
                print(f"-> Info:      '{s3_zarr_path}' not found, skipping.")
                continue

            # Efficiently count the number of files
            num_files = len(s3.find(s3_zarr_path))
            
            if num_files > MAX_FILES_THRESHOLD:
                print(f"🟡 Skipping:  {dataset_name} ({num_files:,} files > {MAX_FILES_THRESHOLD:,})")
            else:
                print(f"✅ Queued:    {dataset_name} ({num_files:,} files)")
                prefix_paths_to_download.append(path)
        
        except Exception as e:
            print(f"❌ Error checking {dataset_name}: {e}")
    # -----------------------------------------------

    print("-" * 50)
    print(f"Found {len(prefix_paths_to_download)} total datasets to process.")
    if prefix_paths_to_download:
        print(f"Starting parallel download with up to {MAX_WORKERS} workers...")
    print("-" * 50)

    # 4. Use ThreadPoolExecutor to manage parallel downloads
    # Use the newly filtered list 'prefix_paths_to_download'
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_zarr_archive, path, s3, local_root_dir) for path in prefix_paths_to_download]
        for future in concurrent.futures.as_completed(futures):
            status_message = future.result()
            print(status_message)

    print("-" * 50)
    print("All download attempts are complete.")