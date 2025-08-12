import s3fs
import os
import time

# --- configuration ---
s3_path = 'janelia-cosem-datasets/jrc_cos7-1a/jrc_cos7-1a.zarr'
local_path = 'data/jrc_cos7-1a/jrc_cos7-1a.zarr' # Download to a directory with this name

# --- hpc best practice ---
# it's better to specify an absolute path on a scratch file system, e.g.:
# username = os.environ.get('USER')
# local_path = f'/scratch/{username}/jrc_cos7-1a.zarr'

def download_zarr_from_s3(s3_path, local_path):
    """
    Downloads a Zarr store from a public S3 bucket.
    """
    if os.path.exists(local_path):
        print(f"Local path '{local_path}' already exists. Skipping download.")
        print("Delete the existing directory to re-download.")
        return

    print("Initializing anonymous S3 file system...")
    s3 = s3fs.S3FileSystem(anon=True)

    print(f"Starting download from 's3://{s3_path}' to '{local_path}'...")
    start_time = time.time()

    # The get() method with recursive=True downloads an entire "directory"
    try:
        s3.get(s3_path, local_path, recursive=True)
        end_time = time.time()
        print("\nDownload complete!")
        print(f"Total time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    download_zarr_from_s3(s3_path, local_path)